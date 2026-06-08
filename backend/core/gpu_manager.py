"""
IntelliVoice — GPU Manager

Manages VRAM allocation and model lifecycle.
Designed for the RTX 4080 16GB.
All models are loaded at startup concurrently without lazy loading.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from config.logging_config import get_logger
from config.model_registry import LoadingOrder

logger = get_logger("gpu_manager")


@dataclass
class _ModelSlot:
    name: str
    order: LoadingOrder
    model: Any
    estimated_vram_mb: int


class GPUManager:
    """
    Manages GPU resources and model lifecycle.
    Loads models sequentially according to LoadingOrder, and never evicts them.
    """

    _SAFETY_MARGIN_GB = 1.0

    def __init__(self):
        self._device: torch.device = self._detect_device()
        self._slots: Dict[str, _ModelSlot] = {}
        self._vram_total_mb: int = 0
        self._vram_budget_mb: int = 0

        if self._device.type == "cuda":
            props = torch.cuda.get_device_properties(self._device)
            self._vram_total_mb = props.total_memory // (1024 * 1024)
            self._vram_budget_mb = max(
                0, self._vram_total_mb - int(self._SAFETY_MARGIN_GB * 1024)
            )
            logger.info(
                "gpu_initialised",
                name=torch.cuda.get_device_name(0),
                vram_total_mb=self._vram_total_mb,
                vram_budget_mb=self._vram_budget_mb,
            )

    def _detect_device(self) -> torch.device:
        """Detect the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        logger.warning("no_gpu_detected_falling_back_to_cpu")
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
        stats = self.get_memory_stats()
        logger.info(
            "gpu_memory_status",
            **stats,
            loaded_models=list(self._slots.keys()),
        )

    def get_memory_stats(self) -> Dict[str, float]:
        if not self.is_cuda:
            return {"allocated": 0, "reserved": 0, "total": 0, "free": 0}
        allocated = torch.cuda.memory_allocated(self._device) / 1e9
        reserved = torch.cuda.memory_reserved(self._device) / 1e9
        total = torch.cuda.get_device_properties(self._device).total_memory / 1e9
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 1),
            "free_gb": round(total - allocated, 2),
        }

    def register_model(
        self,
        name: str,
        model: Any,
        order: LoadingOrder = LoadingOrder.PREPROCESSING,
        vram_mb: int = 0,
    ) -> None:
        """Register an already-loaded model under a loading order phase."""
        self._slots[name] = _ModelSlot(
            name=name,
            order=order,
            model=model,
            estimated_vram_mb=vram_mb,
        )
        logger.info("model_registered", model=name, order=order.name, vram_mb=vram_mb)

    def unregister_model(self, name: str) -> None:
        """Unregister and release a model from the registry."""
        slot = self._slots.pop(name, None)
        if slot is None:
            return
        try:
            if hasattr(slot.model, "cpu") and callable(slot.model.cpu):
                slot.model.cpu()
        except Exception:
            pass
        del slot
        self._cleanup_gpu()
        logger.info("model_unregistered", model=name)

    def get_model(self, name: str) -> Optional[Any]:
        slot = self._slots.get(name)
        return slot.model if slot else None

    def is_loaded(self, name: str) -> bool:
        return name in self._slots

    def shutdown(self) -> None:
        """Release everything."""
        for name in list(self._slots.keys()):
            self.unregister_model(name)
        self._cleanup_gpu()
        logger.info("gpu_manager_shutdown")

    def _cleanup_gpu(self) -> None:
        """Force garbage collection and clear GPU cache."""
        gc.collect()
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
