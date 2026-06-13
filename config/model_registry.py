"""
IntelliVoice — Model Registry

Defines all model configurations, HuggingFace IDs, VRAM budgets,
and loading parameters for every layer in the pipeline.

Stack (target: RTX 4080 16GB VRAM, 64 GB RAM):
    1. Preprocessing: Silero VAD + noisereduce (CPU / FP32)
    2A. ASR: Whisper large-v3-turbo (FP16 via faster-whisper)
    2B. Emotion/Speaker: superb/wav2vec2-base-superb-er + ECAPA-TDNN (FP16)
    3. Memory: Conversation Memory + MongoDB (CPU only)
    4. Core Reasoning: Qwen2.5 3B (INT4 NF4 double quant)
    5. TTS Synthesis: XTTS-v2 (FP16 via TTS)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class ModelPrecision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


class LoadingOrder(int, Enum):
    """Enforced specific loading order at startup."""

    PREPROCESSING = 1
    ASR = 2
    EMOTION_SPEAKER = 3
    REASONING = 4
    TTS = 5


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str
    hf_model_id: str
    precision: ModelPrecision = ModelPrecision.FP16
    estimated_vram_mb: int = 0
    order: LoadingOrder = LoadingOrder.PREPROCESSING
    revision: Optional[str] = None
    trust_remote_code: bool = False
    dtype: str = "float16"
    device_map: str = "auto"
    quantization_config: Optional[Dict] = None
    extra_kwargs: Dict = field(default_factory=dict)
    download_via_hf: bool = True
    notes: str = ""

    @property
    def vram_gb(self) -> float:
        return self.estimated_vram_mb / 1024


class ModelRegistry:
    """Central registry of all models used in the pipeline."""

    SILERO_VAD = ModelConfig(
        name="silero_vad",
        hf_model_id="snakers4/silero-vad",
        precision=ModelPrecision.FP32,
        estimated_vram_mb=50,
        order=LoadingOrder.PREPROCESSING,
        dtype="float32",
        download_via_hf=False,
        notes="Loaded via torch.hub, runs on CPU",
    )

    RESEMBLE_ENHANCE = ModelConfig(
        name="resemble_enhance",
        hf_model_id="resemble-ai/resemble-enhance",
        precision=ModelPrecision.FP32,
        estimated_vram_mb=300,
        order=LoadingOrder.PREPROCESSING,
        dtype="float32",
        download_via_hf=True,
        notes="Loaded via resemble_enhance library",
    )

    WHISPER = ModelConfig(
        name="whisper",
        hf_model_id="deepdml/faster-whisper-large-v3-turbo-ct2",
        precision=ModelPrecision.FP16,
        estimated_vram_mb=1500,
        order=LoadingOrder.ASR,
        dtype="float16",
        notes="Loaded via faster-whisper",
    )

    EMOTION = ModelConfig(
        name="emotion",
        hf_model_id="superb/wav2vec2-base-superb-er",
        precision=ModelPrecision.FP16,
        estimated_vram_mb=350,
        order=LoadingOrder.EMOTION_SPEAKER,
        dtype="float16",
    )

    ECAPA_TDNN = ModelConfig(
        name="ecapa_tdnn",
        hf_model_id="speechbrain/spkrec-ecapa-voxceleb",
        precision=ModelPrecision.FP16,
        estimated_vram_mb=100,
        order=LoadingOrder.EMOTION_SPEAKER,
        dtype="float16",
    )

    FAST_LLM = ModelConfig(
        name="fast_llm",
        hf_model_id="Qwen/Qwen2.5-7B-Instruct",
        precision=ModelPrecision.INT4,
        estimated_vram_mb=5000,
        order=LoadingOrder.REASONING,
        dtype="float16",
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
    }
    )

    OMNIVOICE = ModelConfig(
        name="omnivoice",
        hf_model_id="k2-fsa/OmniVoice",
        precision=ModelPrecision.FP16,
        estimated_vram_mb=4000,
        order=LoadingOrder.TTS,
        trust_remote_code=True,
        dtype="float16",
    )

    @classmethod
    def get_all_models(cls) -> list[ModelConfig]:
        """Return all model configs."""
        return [
            cls.SILERO_VAD,
            cls.RESEMBLE_ENHANCE,
            cls.WHISPER,
            cls.EMOTION,
            cls.ECAPA_TDNN,
            cls.FAST_LLM,
            cls.OMNIVOICE,
        ]

    @classmethod
    def print_vram_budget(cls) -> None:
        """Print VRAM budget breakdown."""
        print("\n=== VRAM Budget (RTX 4080 16GB target) ===\n")
        total_mb = 0
        for m in cls.get_all_models():
            print(
                f"  {m.name:<18} {m.estimated_vram_mb:>6} MB  ({m.vram_gb:.2f} GB)  {m.precision.value}"
            )
            total_mb += m.estimated_vram_mb
        print(f"\n  {'TOTAL':<18} {total_mb:>6} MB  ({total_mb / 1024:.2f} GB)")
        print(f"  {'HEADROOM':<18} {16384 - total_mb:>6} MB  ({(16384 - total_mb) / 1024:.2f} GB)")
        print("==========================================\n")
