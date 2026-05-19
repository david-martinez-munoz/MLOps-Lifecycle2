"""
MLOps-Lifecycle — Continuous Training Pipeline
================================================
Ejecutado por GitHub Actions en cada push a `develop`.

Flujo:
  1. Carga data/new_train_data.csv (etiquetas sesgadas → degradación real en v0.0.2).
  2. Fine-tuning de 1 época sobre DistilBERT SST-2.
  3. Guarda los pesos como .safetensors en models/{version}/.
  4. (Opcional) Registra métricas en MLflow si MLFLOW_TRACKING_URI está definido.
  5. Escribe models/{version}.json con la URL de descarga (la rellena el CI después).

El script NO sube nada a GitHub — eso lo hace el workflow YAML con `gh release`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("retrain")

# ── Configuración desde variables de entorno ────────────────────────────────
BASE_MODEL_NAME: str = os.getenv(
    "MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english"
)
TARGET_VERSION: str = os.getenv("TARGET_VERSION", "v0.0.2")
MODEL_ROOT: Path = Path(os.getenv("MODEL_ROOT", "models"))
DATA_PATH: Path = Path(os.getenv("TRAIN_DATA_PATH", "data/new_train_data.csv"))
NUM_TRAIN_EPOCHS: int = int(os.getenv("NUM_TRAIN_EPOCHS", "1"))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "8"))
MAX_SEQ_LEN: int = int(os.getenv("MAX_SEQ_LEN", "128"))

OUTPUT_DIR: Path = MODEL_ROOT / TARGET_VERSION


# ── Dataset PyTorch ─────────────────────────────────────────────────────────

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _load_dataset(tokenizer: AutoTokenizer) -> SentimentDataset:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {DATA_PATH}\n"
            "Asegúrate de que data/new_train_data.csv esté en el repositorio."
        )
    df = pd.read_csv(DATA_PATH)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    logger.info(
        "Dataset: %d muestras  |  NEGATIVE(0)=%d  POSITIVE(1)=%d",
        len(labels), labels.count(0), labels.count(1),
    )
    encodings = tokenizer(
        texts, truncation=True, padding="max_length",
        max_length=MAX_SEQ_LEN, return_tensors="pt",
    )
    return SentimentDataset(encodings, labels)


def _compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float(accuracy_score(labels, preds))}


# ── MLflow (opcional) ────────────────────────────────────────────────────────

def _try_mlflow_log(run_metrics: dict[str, Any], output_dir: Path) -> None:
    """Registra en MLflow solo si MLFLOW_TRACKING_URI está configurado."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        logger.info("MLFLOW_TRACKING_URI no definido — omitiendo tracking MLflow.")
        return
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(
            os.getenv("MLFLOW_EXPERIMENT", "mlops-lifecycle-continuous-training")
        )
        with mlflow.start_run(run_name=f"retrain-{TARGET_VERSION}"):
            mlflow.log_params({
                "base_model": BASE_MODEL_NAME,
                "target_version": TARGET_VERSION,
                "num_train_epochs": NUM_TRAIN_EPOCHS,
                "batch_size": BATCH_SIZE,
            })
            for k, v in run_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            mlflow.log_artifacts(str(output_dir), artifact_path="model_files")
            # Registrar en Model Registry con alias de versión
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model_files"
            mv = mlflow.register_model(model_uri, os.getenv("MLFLOW_MODEL_NAME", "sentiment-model"))
            from mlflow import MlflowClient
            MlflowClient().set_registered_model_alias(
                os.getenv("MLFLOW_MODEL_NAME", "sentiment-model"),
                TARGET_VERSION,
                mv.version,
            )
            logger.info("MLflow: modelo registrado como '%s@%s'", "sentiment-model", TARGET_VERSION)
    except Exception as exc:
        logger.warning("MLflow tracking falló (no crítico): %s", exc)


# ── Pipeline principal ───────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("Iniciando Continuous Training Pipeline")
    logger.info("  Versión : %s", TARGET_VERSION)
    logger.info("  Modelo  : %s", BASE_MODEL_NAME)
    logger.info("  Dataset : %s", DATA_PATH)
    logger.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar modelo base y tokenizador
    logger.info("Descargando modelo base desde HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, use_safetensors=True,
    )

    # 2. Preparar dataset de fine-tuning
    train_dataset = _load_dataset(tokenizer)

    # 3. Fine-tuning con HuggingFace Trainer — 1 época
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_strategy="no",
        logging_strategy="epoch",
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=_compute_metrics,
    )

    logger.info("Iniciando fine-tuning (%d época)...", NUM_TRAIN_EPOCHS)
    result = trainer.train()
    train_loss = result.training_loss
    logger.info("Fine-tuning completado — Loss: %.4f", train_loss)

    # 4. Guardar tokenizador + pesos en safetensors
    logger.info("Guardando artefactos en %s ...", OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)

    assert (OUTPUT_DIR / "model.safetensors").exists(), "ERROR: model.safetensors no generado"
    assert (OUTPUT_DIR / "config.json").exists(), "ERROR: config.json no generado"
    logger.info("Artefactos guardados correctamente.")

    # 5. Escribir puntero JSON (el CI rellenará download_url después de crear el Release)
    pointer: dict[str, Any] = {
        "version": TARGET_VERSION,
        "base_model": BASE_MODEL_NAME,
        "format": "safetensors",
        "training_epochs": NUM_TRAIN_EPOCHS,
        "training_samples": len(train_dataset),
        "train_loss": round(train_loss, 4),
        "created_at": int(time.time()),
        # download_url se rellena por el workflow YAML después de `gh release upload`
        "download_url": "",
        "note": (
            "Fine-tuning incremental de 1 época sobre new_train_data.csv. "
            "Los pesos (.safetensors) están en GitHub Releases. "
            "Este JSON es el puntero ligero que vive en Git."
        ),
    }
    pointer_path = MODEL_ROOT / f"{TARGET_VERSION}.json"
    with pointer_path.open("w", encoding="utf-8") as fh:
        json.dump(pointer, fh, indent=2)
    logger.info("Puntero escrito: %s", pointer_path)

    # 6. Tracking MLflow (opcional)
    run_metrics = {
        "train_loss": train_loss,
        "training_samples": len(train_dataset),
        "training_epochs": NUM_TRAIN_EPOCHS,
    }
    _try_mlflow_log(run_metrics, OUTPUT_DIR)

    logger.info("=" * 60)
    logger.info("Pipeline completado con éxito → %s", TARGET_VERSION)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
