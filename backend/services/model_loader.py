"""
IntelliVoice — Model Loader Service

Handles lazy model loading with VRAM tracking.
Provides utilities for downloading and caching HuggingFace models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from huggingface_hub import snapshot_download, hf_hub_download

from config import get_settings
from config.logging_config import get_logger
from config.model_registry import ModelConfig, ModelRegistry

logger = get_logger("model_loader")


class ModelLoader:
    """
    Lazy model loader with VRAM tracking.

    Downloads models from HuggingFace Hub on first use
    and caches them locally.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        settings = get_settings()
        self.cache_dir = cache_dir or settings.models_dir
        self.hf_token = settings.hf_token or None

    def download_model(
        self,
        model_config: ModelConfig,
        force: bool = False,
    ) -> Path:
        """
        Download a model from HuggingFace Hub.

        Args:
            model_config: Model configuration.
            force: Force re-download even if cached.

        Returns:
            Path to downloaded model directory.
        """
        model_dir = self.cache_dir / model_config.name

        if model_dir.exists() and not force:
            logger.info(
                "model_cached",
                model=model_config.name,
                path=str(model_dir),
            )
            return model_dir

        logger.info(
            "downloading_model",
            model=model_config.name,
            hf_id=model_config.hf_model_id,
            destination=str(model_dir),
        )

        try:
            snapshot_download(
                repo_id=model_config.hf_model_id,
                local_dir=str(model_dir),
                revision=model_config.revision,
                token=self.hf_token,
            )
            logger.info("model_downloaded", model=model_config.name)
            return model_dir
        except Exception as e:
            logger.error(
                "model_download_failed",
                model=model_config.name,
                error=str(e),
            )
            raise

    def download_all_models(self, force: bool = False) -> dict:
        """Download all models defined in the registry."""
        results = {}
        for model_config in ModelRegistry.get_all_models():
            try:
                path = self.download_model(model_config, force=force)
                results[model_config.name] = {
                    "status": "ok",
                    "path": str(path),
                }
            except Exception as e:
                results[model_config.name] = {
                    "status": "error",
                    "error": str(e),
                }
        return results

    def get_model_path(self, model_config: ModelConfig) -> Optional[Path]:
        """Get the local path for a model (None if not downloaded)."""
        model_dir = self.cache_dir / model_config.name
        return model_dir if model_dir.exists() else None

    def get_download_status(self) -> dict:
        """Check which models are downloaded."""
        status = {}
        for model_config in ModelRegistry.get_all_models():
            path = self.get_model_path(model_config)
            status[model_config.name] = {
                "downloaded": path is not None,
                "path": str(path) if path else None,
                "hf_id": model_config.hf_model_id,
                "vram_mb": model_config.estimated_vram_mb,
            }
        return status
