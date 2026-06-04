"""
IntelliVoice — GPU Manager

Manages VRAM allocation, model loading/offloading across phases,
and GPU resource tracking for the RTX 4080.
"""

from __future__ import annotations

import gc
from typing import Any, Dict, Optional

import torch

from config.logging_config import get_logger
from config.model_registry import LoadingPhase

logger = get_logger("gpu_manager")


class GPUManager:
    """
    Manages GPU resources and model lifecycle.

    Implements a three-phase model loading strategy to stay within
    the 16GB VRAM budget of the RTX 4080:
      - Phase 0 (ALWAYS): VAD + DeepFilterNet (~150MB) - always resident
      - Phase 1 (UNDERSTANDING): Encoders (~8GB) - loaded during input processing
      - Phase 2 (REASONING): LLM (~10GB) - loaded for response generation
      - Phase 3 (GENERATION): TTS (~2.5GB) - loaded for speech synthesis
    """

    def __init__(self):
        self._device: torch.device = self._detect_device()
        self._loaded_models: Dict[str, Any] = {}
        self._current_phase: Optional[LoadingPhase] = None
        self._vram_total_mb: int = 0
        self._vram_used_mb: int = 0

        if self._device.type == "cuda":
            props = torch.cuda.get_device_properties(self._device)
            self._vram_total_mb = props.total_mem // (1024 * 1024)

    def _detect_device(self) -> torch.device:
        """Detect the best available device."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            logger.info(
                "gpu_detected",
                name=torch.cuda.get_device_name(0),
                vram_gb=f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}",
            )
            return device
        else:
            logger.warning("no_gpu_detected", fallback="cpu")
            return torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_cuda(self) -> bool:
        return self._device.type == "cuda"

    def log_gpu_info(self) -> None:
        """Log current GPU memory status."""
        if not self.is_cuda:
            logger.info("running_on_cpu")
            return

        allocated = torch.cuda.memory_allocated(self._device) / 1e9
        reserved = torch.cuda.memory_reserved(self._device) / 1e9
        total = torch.cuda.get_device_properties(self._device).total_mem / 1e9

        logger.info(
            "gpu_memory_status",
            allocated_gb=f"{allocated:.2f}",
            reserved_gb=f"{reserved:.2f}",
            total_gb=f"{total:.1f}",
            free_gb=f"{total - allocated:.2f}",
        )

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics in GB."""
        if not self.is_cuda:
            return {"allocated": 0, "reserved": 0, "total": 0, "free": 0}

        allocated = torch.cuda.memory_allocated(self._device) / 1e9
        reserved = torch.cuda.memory_reserved(self._device) / 1e9
        total = torch.cuda.get_device_properties(self._device).total_mem / 1e9

        return {
            "allocated": round(allocated, 2),
            "reserved": round(reserved, 2),
            "total": round(total, 1),
            "free": round(total - allocated, 2),
        }

    def register_model(self, name: str, model: Any) -> None:
        """Register a loaded model."""
        self._loaded_models[name] = model
        logger.info("model_registered", model=name)
        self.log_gpu_info()

    def unregister_model(self, name: str) -> None:
        """Unregister and delete a model from GPU."""
        if name in self._loaded_models:
            model = self._loaded_models.pop(name)
            # Move to CPU first, then delete
            if hasattr(model, "cpu"):
                model.cpu()
            del model
            self._cleanup_gpu()
            logger.info("model_unloaded", model=name)

    def offload_phase(self, phase: LoadingPhase) -> None:
        """Offload all models belonging to a specific phase."""
        models_to_offload = [
            name for name, _ in self._loaded_models.items()
            if name.startswith(f"phase_{phase.value}_")
        ]
        for name in models_to_offload:
            self.unregister_model(name)

        self._cleanup_gpu()
        logger.info("phase_offloaded", phase=phase.name, models_removed=len(models_to_offload))

    def offload_all_except_always(self) -> None:
        """Offload everything except ALWAYS-on models."""
        models_to_keep = set()
        models_to_remove = []

        for name in self._loaded_models:
            if name.startswith("always_"):
                models_to_keep.add(name)
            else:
                models_to_remove.append(name)

        for name in models_to_remove:
            self.unregister_model(name)

        self._cleanup_gpu()
        logger.info(
            "offloaded_to_always",
            kept=len(models_to_keep),
            removed=len(models_to_remove),
        )

    def _cleanup_gpu(self) -> None:
        """Force garbage collection and clear GPU cache."""
        gc.collect()
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def ensure_memory_available(self, required_mb: int) -> bool:
        """Check if enough VRAM is available, offloading if needed."""
        if not self.is_cuda:
            return True

        stats = self.get_memory_stats()
        free_mb = stats["free"] * 1024

        if free_mb >= required_mb:
            return True

        logger.warning(
            "insufficient_vram",
            required_mb=required_mb,
            free_mb=int(free_mb),
            action="attempting_offload",
        )

        # Try cleanup first
        self._cleanup_gpu()
        stats = self.get_memory_stats()
        free_mb = stats["free"] * 1024

        return free_mb >= required_mb

    def get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,
        }
        return dtype_map.get(dtype_str, torch.float16)
