"""
MLOps-Lifecycle — ModelLifecycleManager
========================================
Gestiona el ciclo de vida del modelo en caliente.

Estrategia de resolución de versiones (por prioridad):
  1. Directorio local completo  →  models/{version}/  (ya descargado)
  2. Puntero + descarga URL     →  models/{version}.json contiene download_url
                                    (GitHub Releases asset)
  3. HuggingFace Hub            →  fallback para v0.0.1 / baseline

Seguridad NIST (Path Traversal):
  - Path.resolve() + verificación de prefijo antes de cualquier I/O.
  - URLs validadas: esquema, nombre de archivo, extensión.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


VERSION_PATTERN = re.compile(r"^v\d+\.\d+\.\d+$")
SAFE_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")


class ModelLifecycleManager:
    """
    Carga, gestiona y cambia versiones del modelo de sentiment en caliente.
    Thread-safety: escrituras bajo asyncio.Lock.
    """

    def __init__(self, model_name: str, model_root: str, active_version: str) -> None:
        self.model_name = model_name
        self.model_root = Path(model_root)
        self.active_version = active_version
        self._pipeline = None
        self._lock = asyncio.Lock()

    # ── Arranque ──────────────────────────────────────────────────────────────

    async def startup(self) -> None:
        await self.load_model(self.active_version)

    # ── Predicción ────────────────────────────────────────────────────────────

    async def predict(self, text: str) -> dict[str, Any]:
        if self._pipeline is None:
            raise RuntimeError("El modelo no está cargado.")
        result = await asyncio.to_thread(self._pipeline, text)
        prediction = result[0]
        return {
            "label": prediction["label"],
            "score": float(prediction["score"]),
            "model_version": self.active_version,
        }

    # ── Cambio de versión en caliente ─────────────────────────────────────────

    async def change_version(
        self,
        version: str,
        git_ref: str | None = None,     # mantenido por compatibilidad
        url: str | None = None,
    ) -> dict[str, str]:
        """
        Cambia la versión activa SIN reiniciar el contenedor.

        Orden de resolución:
          1. url explícita en el payload  →  descarga directa de model.safetensors
          2. directorio local completo    →  carga desde disco
          3. puntero models/{version}.json con download_url  →  descarga desde GitHub Releases
          4. HuggingFace fallback
        """
        async with self._lock:
            self._validate_version(version)
            model_dir = self.model_root / version

            if url:
                model_dir = await self._download_model_from_url(url=url, version=version)

            elif self._is_valid_model_dir(model_dir):
                pass  # ya está en disco

            else:
                pointer = self._read_pointer(version)
                if pointer and pointer.get("download_url"):
                    model_dir = await self._download_model_from_url(
                        url=pointer["download_url"], version=version
                    )
                else:
                    # Fallback: HuggingFace
                    await self._load_from_huggingface(version)
                    return {"status": "changed", "active_version": version}

            await self._load_from_directory(model_dir, version)
            return {"status": "changed", "active_version": version}

    # ── Carga inicial ─────────────────────────────────────────────────────────

    async def load_model(self, version: str) -> None:
        self._validate_version(version)
        model_dir = self.model_root / version

        if self._is_valid_model_dir(model_dir):
            await self._load_from_directory(model_dir, version)
            return

        pointer = self._read_pointer(version)
        if pointer and pointer.get("download_url"):
            model_dir = await self._download_model_from_url(
                url=pointer["download_url"], version=version
            )
            await self._load_from_directory(model_dir, version)
            return

        # Fallback: HuggingFace (v0.0.1 baseline)
        await self._load_from_huggingface(version)

    # ── Puntero ligero ────────────────────────────────────────────────────────

    def _read_pointer(self, version: str) -> dict | None:
        """Lee models/{version}.json si existe."""
        pointer_path = self.model_root / f"{version}.json"
        if pointer_path.exists():
            with pointer_path.open(encoding="utf-8") as fh:
                return json.load(fh)
        return None

    # ── Carga desde HuggingFace ───────────────────────────────────────────────

    async def _load_from_huggingface(self, version: str) -> None:
        model_dir = self.model_root / version
        model_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained, self.model_name
        )
        model = await asyncio.to_thread(
            AutoModelForSequenceClassification.from_pretrained,
            self.model_name, use_safetensors=True,
        )
        await asyncio.to_thread(tokenizer.save_pretrained, model_dir)
        await asyncio.to_thread(model.save_pretrained, model_dir, safe_serialization=True)
        await self._load_from_directory(model_dir, version)

    # ── Carga desde directorio local ──────────────────────────────────────────

    async def _load_from_directory(self, model_dir: Path, version: str) -> None:
        # Sanitización NIST Path Traversal
        resolved_root = self.model_root.resolve()
        resolved_dir = model_dir.resolve()
        if not str(resolved_dir).startswith(str(resolved_root)):
            raise ValueError(f"Ruta insegura (path traversal): {model_dir}")

        if not self._is_valid_model_dir(model_dir):
            raise FileNotFoundError(f"Directorio de modelo inválido: {model_dir}")

        tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained, str(model_dir)
        )
        model = await asyncio.to_thread(
            AutoModelForSequenceClassification.from_pretrained,
            str(model_dir), use_safetensors=True,
        )
        device = 0 if torch.cuda.is_available() else -1
        self._pipeline = pipeline(
            task="sentiment-analysis", model=model, tokenizer=tokenizer, device=device,
        )
        self.active_version = version

    # ── Descarga desde URL (GitHub Releases o cualquier HTTPS) ───────────────

    async def _download_model_from_url(self, url: str, version: str) -> Path:
        """
        Descarga model.safetensors desde una URL y construye un directorio de modelo.

        El tokenizador se copia del modelo activo (ya está en disco).
        Sanitización NIST: esquema, nombre de archivo, destino dentro de model_root.
        """
        parsed = urlparse(url)
        if parsed.scheme not in {"https", "http"}:
            raise ValueError("Solo se permiten URLs con esquema http/https.")

        filename = Path(parsed.path).name
        if not filename or not SAFE_FILENAME_PATTERN.match(filename):
            raise ValueError(f"Nombre de archivo inseguro en la URL: '{filename}'")
        if filename != "model.safetensors":
            raise ValueError("Solo se permite descargar archivos llamados 'model.safetensors'.")

        target_dir = self.model_root / version
        tmp_dir = self.model_root / f".tmp_{version}"

        # Sanitización Path Traversal en destino
        resolved_root = self.model_root.resolve()
        for d in (target_dir, tmp_dir):
            if not str(d.resolve()).startswith(str(resolved_root)):
                raise ValueError(f"Destino inseguro: {d}")

        # El tokenizador se copia del modelo activo en disco
        active_dir = self.model_root / self.active_version
        if not self._is_valid_model_dir(active_dir):
            # Si el activo no está en disco, descargarlo primero desde HF
            await self._load_from_huggingface(self.active_version)

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        await asyncio.to_thread(shutil.copytree, active_dir, tmp_dir)

        # Descarga streaming del safetensors
        target_file = tmp_dir / "model.safetensors"
        async with httpx.AsyncClient(timeout=180, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with target_file.open("wb") as fh:
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        fh.write(chunk)

        # Rename atómico
        if target_dir.exists():
            shutil.rmtree(target_dir)
        tmp_dir.rename(target_dir)

        return target_dir

    # ── Validaciones ──────────────────────────────────────────────────────────

    @staticmethod
    def _validate_version(version: str) -> None:
        if not VERSION_PATTERN.match(version):
            raise ValueError(f"Versión inválida: '{version}'. Formato esperado: vX.X.X")

    @staticmethod
    def _is_valid_model_dir(model_dir: Path) -> bool:
        if not model_dir.exists():
            return False
        has_config = (model_dir / "config.json").exists()
        has_tokenizer = (model_dir / "tokenizer_config.json").exists() and (
            (model_dir / "vocab.txt").exists() or (model_dir / "tokenizer.json").exists()
        )
        has_weights = (
            (model_dir / "model.safetensors").exists()
            or (
                any(model_dir.glob("model-*.safetensors"))
                and (model_dir / "model.safetensors.index.json").exists()
            )
        )
        return has_config and has_tokenizer and has_weights
