"""
IntelliVoice — Health Check Route

Provides system health, GPU status, and model loading state.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from config import get_settings
from config.model_registry import ModelRegistry

router = APIRouter()

_start_time = time.time()


@router.get("/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "service": "IntelliVoice",
        "uptime_s": round(time.time() - _start_time),
    }


@router.get("/health/gpu")
async def gpu_health(request: Request):
    """GPU and model health check."""
    gpu_manager = request.app.state.gpu_manager
    pipeline = request.app.state.pipeline

    return {
        "status": "healthy",
        "gpu": {
            "available": gpu_manager.is_cuda,
            "device": str(gpu_manager.device),
            "memory": gpu_manager.get_memory_stats(),
            "loaded_models": list(gpu_manager._slots.keys()),
        },
        "models": {
            "vad": pipeline.vad.is_loaded,
            "asr": pipeline.asr._is_loaded,
            "reasoner": pipeline.reasoner.is_loaded,
            "tts": pipeline.tts.is_loaded,
        },
        "memory": {
            "active_sessions": pipeline.memory.active_sessions,
        },
    }


@router.get("/health/config")
async def config_info():
    """Return non-sensitive configuration + VRAM budget summary."""
    settings = get_settings()
    return {
        "app_name": settings.app_name,
        "environment": settings.app_env,
        "sample_rate": settings.sample_rate,
        "chunk_size_ms": settings.chunk_size_ms,
        "max_audio_length_s": settings.max_audio_length_s,
        "vram_budget_gb": {
            "total_resident_mb": sum(m.estimated_vram_mb for m in ModelRegistry.get_all_models()),
        },
    }
