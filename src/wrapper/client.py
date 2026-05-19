from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class MLOpsLifecycleClient:
    base_url: str
    timeout: float = 30.0

    def health(self) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/health",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def predict(
        self,
        text: str,
        expected_label: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"text": text}

        if expected_label:
            payload["expected_label"] = expected_label

        response = requests.post(
            f"{self.base_url}/predict",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def retrain(
        self,
        target_version: str,
        samples: list[dict[str, str]],
    ) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/retrain",
            json={
                "target_version": target_version,
                "samples": samples,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def change_version(
        self,
        version: str | None = None,
        url: str | None = None,
        git_ref: str | None = None,
    ) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/version/change",
            json={
                "version": version,
                "url": url,
                "git_ref": git_ref,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()