"""
IntelliVoice — Model Registry

Defines all model configurations, HuggingFace IDs, VRAM budgets,
and loading parameters for every layer in the pipeline.

Stack (target: RTX 4080 16GB VRAM, 64 GB RAM):
    1. Preprocessing: Silero VAD + noisereduce (CPU / FP32)
    2A. ASR: Whisper large-v3-turbo (FP16 via faster-whisper)
    2B. Emotion/Speaker: SenseVoice-Small + ECAPA-TDNN (FP16)
    3. Memory: Conversation Memory + MongoDB (CPU only)
    4. Core Reasoning: Qwen3 14B (INT4 NF4 double quant)
    5. TTS Synthesis: CosyVoice 2 (FP16)
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
    torch_dtype: str = "float16"
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
        torch_dtype="float32",
        download_via_hf=False,
        notes="Loaded via torch.hub, runs on CPU",
    )

    NOISEREDUCE = ModelConfig(
        name="noisereduce",
        hf_model_id="n/a",
        precision=ModelPrecision.FP32,
        estimated_vram_mb=0,
        order=LoadingOrder.PREPROCESSING,
        torch_dtype="float32",
        download_via_hf=False,
        notes="Spectral gating via noisereduce pip package, CPU only",
    )

    WHISPER = ModelConfig(
        name="whisper",
        hf_model_id="deepdml/faster-whisper-large-v3-turbo-ct2",
        precision=ModelPrecision.FP16,
        estimated_vram_mb=1500,
        order=LoadingOrder.ASR,
        torch_dtype="float16",
        notes="Loaded via faster-whisper",
    )

    SENSEVOICE = ModelConfig(
        name="sensevoice",
        hf_model_id="FunAudioLLM/SenseVoiceSmall",
        precision=ModelPrecision.FP16,
        estimated_vram_mb=300,
        order=LoadingOrder.EMOTION_SPEAKER,
        torch_dtype="float16",
        trust_remote_code=True,
    )

    ECAPA_TDNN = ModelConfig(
        name="ecapa_tdnn",
        hf_model_id="speechbrain/spkrec-ecapa-voxceleb",
        precision=ModelPrecision.FP16,
        estimated_vram_mb=100,
        order=LoadingOrder.EMOTION_SPEAKER,
        torch_dtype="float16",
    )

    QWEN3_14B = ModelConfig(
        name="qwen3_14b",
        hf_model_id="Qwen/Qwen3-14B",
        precision=ModelPrecision.INT4,
        estimated_vram_mb=8500,
        order=LoadingOrder.REASONING,
        torch_dtype="float16",
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        },
    )

    COSYVOICE2 = ModelConfig(
        name="cosyvoice2",
        hf_model_id="FunAudioLLM/CosyVoice2-0.5B",
        precision=ModelPrecision.FP16,
        estimated_vram_mb=1500,
        order=LoadingOrder.TTS,
        trust_remote_code=True,
        torch_dtype="float16",
    )

    @classmethod
    def get_all_models(cls) -> list[ModelConfig]:
        """Return all model configs."""
        return [
            cls.SILERO_VAD,
            cls.NOISEREDUCE,
            cls.WHISPER,
            cls.SENSEVOICE,
            cls.ECAPA_TDNN,
            cls.QWEN3_14B,
            cls.COSYVOICE2,
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
