from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "MLOps-Lifecycle"
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    model_root: str = "/app/models"
    active_version: str = "v0.0.1"

    # ── MLflow (opcional — para tracking avanzado con DagsHub) ────────────
    mlflow_tracking_uri: str = ""
    mlflow_tracking_username: str = ""
    mlflow_tracking_password: str = ""
    mlflow_model_name: str = "sentiment-model"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
