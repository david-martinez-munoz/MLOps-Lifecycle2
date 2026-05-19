import os
import time
from contextlib import asynccontextmanager
from statistics import mean

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from src.api.config import get_settings
from src.api.ml_lifecycle import ModelLifecycleManager


settings = get_settings()

# ── Métricas Prometheus ─────────────────────────────────────────────────────
# Todas etiquetadas por `version` → Grafana las puede comparar en paralelo

prediction_counter = Counter(
    "sentiment_predictions_total",
    "Total predictions",
    ["label", "version"],
)
prediction_errors = Counter(
    "sentiment_prediction_errors_total",
    "Failed predictions",
    ["version"],
)
prediction_latency = Histogram(
    "sentiment_prediction_latency_seconds",
    "Prediction latency",
    ["version"],
)
prediction_confidence = Gauge(
    "sentiment_prediction_confidence_average",
    "Average confidence score",
    ["version"],
)
prediction_accuracy = Gauge(
    "sentiment_prediction_accuracy_average",
    "Accuracy vs expected_label (Seeder la rellena siempre)",
    ["version"],
)
input_text_length_average = Gauge(
    "sentiment_input_text_length_average",
    "Average input text length",
    ["version"],
)
positive_ratio = Gauge(
    "sentiment_positive_ratio",
    "Ratio of POSITIVE predictions",
    ["version"],
)
drift_score = Gauge(
    "sentiment_drift_score",
    "Drift score: |positive_ratio - 0.5|",
    ["version"],
)
version_change_counter = Counter(
    "model_version_changes_total",
    "Version changes triggered by the user",
    ["version"],
)

# Ventanas deslizantes en memoria (últimas 100 predicciones por versión)
confidence_window: dict[str, list[float]] = {}
accuracy_window:   dict[str, list[float]] = {}
label_window:      dict[str, list[str]]   = {}
text_length_window: dict[str, list[int]]  = {}

ml_manager = ModelLifecycleManager(
    model_name=settings.model_name,
    model_root=settings.model_root,
    active_version=settings.active_version,
)


# ── Schemas ─────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    expected_label: str | None = Field(default=None, pattern="^(POSITIVE|NEGATIVE)$")


class PredictionResponse(BaseModel):
    label: str
    score: float
    model_version: str
    expected_label: str | None = None
    is_correct: bool | None = None


class VersionChangeRequest(BaseModel):
    version: str = Field(..., pattern=r"^v\d+\.\d+\.\d+$")
    git_ref: str | None = None
    url: str | None = None


class VersionChangeResponse(BaseModel):
    status: str
    active_version: str


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configurar MLflow si está disponible
    if settings.mlflow_tracking_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            if settings.mlflow_tracking_username:
                os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_tracking_username
            if settings.mlflow_tracking_password:
                os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.mlflow_tracking_password
            os.environ["MLFLOW_MODEL_NAME"] = settings.mlflow_model_name
        except ImportError:
            pass  # mlflow no instalado — funciona igual sin él

    await ml_manager.startup()
    yield


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root() -> dict[str, str]:
    return {
        "message": "MLOps-Lifecycle API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model_name": settings.model_name,
        "model_version": ml_manager.active_version,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest) -> PredictionResponse:
    current_version = ml_manager.active_version
    started_at = time.perf_counter()

    try:
        result = await ml_manager.predict(payload.text)
        version = result["model_version"]
        elapsed = time.perf_counter() - started_at

        # Actualizar métricas
        prediction_counter.labels(label=result["label"], version=version).inc()
        prediction_latency.labels(version=version).observe(elapsed)

        confidence_window.setdefault(version, []).append(result["score"])
        label_window.setdefault(version, []).append(result["label"])
        text_length_window.setdefault(version, []).append(len(payload.text))

        _limit_window(confidence_window[version])
        _limit_window(label_window[version])
        _limit_window(text_length_window[version])

        prediction_confidence.labels(version=version).set(mean(confidence_window[version]))
        input_text_length_average.labels(version=version).set(mean(text_length_window[version]))

        positives = sum(1 for lbl in label_window[version] if lbl == "POSITIVE")
        ratio = positives / len(label_window[version])
        positive_ratio.labels(version=version).set(ratio)
        drift_score.labels(version=version).set(abs(ratio - 0.5))

        # Accuracy real (el Seeder siempre envía expected_label)
        is_correct: bool | None = None
        if payload.expected_label:
            is_correct = result["label"] == payload.expected_label
            accuracy_window.setdefault(version, []).append(1.0 if is_correct else 0.0)
            _limit_window(accuracy_window[version])
            prediction_accuracy.labels(version=version).set(mean(accuracy_window[version]))

        return PredictionResponse(
            label=result["label"],
            score=result["score"],
            model_version=version,
            expected_label=payload.expected_label,
            is_correct=is_correct,
        )

    except Exception as exc:
        prediction_errors.labels(version=current_version).inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/version/change", response_model=VersionChangeResponse)
async def change_version(payload: VersionChangeRequest) -> VersionChangeResponse:
    """
    Cambia la versión activa en caliente.
    - `version`: "v0.0.1" o "v0.0.2" (obligatorio)
    - `url`: URL directa a model.safetensors (opcional)
    """
    try:
        result = await ml_manager.change_version(
            version=payload.version,
            git_ref=payload.git_ref,
            url=payload.url,
        )
        version_change_counter.labels(version=result["active_version"]).inc()
        return VersionChangeResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _limit_window(values: list, max_size: int = 100) -> None:
    if len(values) > max_size:
        del values[:-max_size]
