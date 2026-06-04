"""
IntelliVoice — Model Registry

Defines all model configurations, HuggingFace IDs, VRAM budgets,
and loading parameters for every layer in the pipeline.
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


class LoadingPhase(int, Enum):
    """Which phase a model belongs to for VRAM management."""
    ALWAYS = 0       # Tiny models, always loaded (VAD, DeepFilter)
    UNDERSTANDING = 1  # Encoders: XLS-R, Qwen-Audio, Emotion2Vec, WavLM
    REASONING = 2      # LLM: Qwen3 MoE
    GENERATION = 3     # TTS: CosyVoice 2, HiFi-GAN


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    hf_model_id: str
    precision: ModelPrecision = ModelPrecision.FP16
    estimated_vram_mb: int = 0
    phase: LoadingPhase = LoadingPhase.UNDERSTANDING
    revision: Optional[str] = None
    trust_remote_code: bool = False
    torch_dtype: str = "float16"
    device_map: str = "auto"
    quantization_config: Optional[Dict] = None
    extra_kwargs: Dict = field(default_factory=dict)

    @property
    def vram_gb(self) -> float:
        return self.estimated_vram_mb / 1024


class ModelRegistry:
    """Central registry of all models used in the pipeline."""

    # ---- Layer 1: Preprocessing ----
    SILERO_VAD = ModelConfig(
        name="silero_vad",
        hf_model_id="snakers4/silero-vad",
        precision=ModelPrecision.FP32,
        estimated_vram_mb=50,
        phase=LoadingPhase.ALWAYS,
        torch_dtype="float32",
    )

    DEEPFILTERNET = ModelConfig(
        name="deepfilternet",
        hf_model_id="deepfilternet/DeepFilterNet3",
        precision=ModelPrecision.FP32,
        estimated_vram_mb=100,
        phase=LoadingPhase.ALWAYS,
        torch_dtype="float32",
    )

    # ---- Layer 2: Acoustic Encoder ----
    XLSR_1B = ModelConfig(
        name="xlsr_1b",
        hf_model_id="facebook/wav2vec2-xls-r-1b",
        precision=ModelPrecision.FP16,
        estimated_vram_mb=2048,
        phase=LoadingPhase.UNDERSTANDING,
        torch_dtype="float16",
    )

    # ---- Layer 3: Audio-Semantic Understanding ----
    QWEN_AUDIO = ModelConfig(
        name="qwen_audio",
        hf_model_id="Qwen/Qwen-Audio-Chat",
        precision=ModelPrecision.INT4,
        estimated_vram_mb=4096,
        phase=LoadingPhase.UNDERSTANDING,
        trust_remote_code=True,
        torch_dtype="float16",
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        },
    )

    # ---- Layer 4: Prosody & Emotion ----
    EMOTION2VEC = ModelConfig(
        name="emotion2vec",
        hf_model_id="emotion2vec/emotion2vec_plus_large",
        precision=ModelPrecision.FP16,
        estimated_vram_mb=300,
        phase=LoadingPhase.UNDERSTANDING,
        trust_remote_code=True,
        torch_dtype="float16",
    )

    # ---- Layer 5: Speaker Understanding ----
    WAVLM_LARGE = ModelConfig(
        name="wavlm_large",
        hf_model_id="microsoft/wavlm-large",
        precision=ModelPrecision.FP16,
        estimated_vram_mb=1200,
        phase=LoadingPhase.UNDERSTANDING,
        torch_dtype="float16",
    )

    # ---- Layer 6: Core Reasoning ----
    QWEN3_MOE = ModelConfig(
        name="qwen3_moe",
        hf_model_id="Qwen/Qwen3-30B-A3B",
        precision=ModelPrecision.INT4,
        estimated_vram_mb=10240,
        phase=LoadingPhase.REASONING,
        trust_remote_code=True,
        torch_dtype="float16",
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        },
    )

    # ---- Layer 10: Speech Generation ----
    COSYVOICE2 = ModelConfig(
        name="cosyvoice2",
        hf_model_id="FunAudioLLM/CosyVoice2-0.5B",
        precision=ModelPrecision.FP16,
        estimated_vram_mb=2048,
        phase=LoadingPhase.GENERATION,
        trust_remote_code=True,
        torch_dtype="float16",
    )

    # ---- Layer 12: Audio Synthesis ----
    HIFIGAN = ModelConfig(
        name="hifigan",
        hf_model_id="nvidia/hifigan-22khz",
        precision=ModelPrecision.FP32,
        estimated_vram_mb=50,
        phase=LoadingPhase.GENERATION,
        torch_dtype="float32",
    )

    @classmethod
    def get_all_models(cls) -> list[ModelConfig]:
        """Return all model configs."""
        return [
            cls.SILERO_VAD,
            cls.DEEPFILTERNET,
            cls.XLSR_1B,
            cls.QWEN_AUDIO,
            cls.EMOTION2VEC,
            cls.WAVLM_LARGE,
            cls.QWEN3_MOE,
            cls.COSYVOICE2,
            cls.HIFIGAN,
        ]

    @classmethod
    def get_models_by_phase(cls, phase: LoadingPhase) -> list[ModelConfig]:
        """Return models belonging to a specific loading phase."""
        return [m for m in cls.get_all_models() if m.phase == phase]

    @classmethod
    def get_phase_vram_mb(cls, phase: LoadingPhase) -> int:
        """Estimate total VRAM for a loading phase."""
        return sum(m.estimated_vram_mb for m in cls.get_models_by_phase(phase))

    @classmethod
    def print_vram_budget(cls) -> None:
        """Print VRAM budget summary."""
        print("\n=== IntelliVoice VRAM Budget ===")
        total = 0
        for phase in LoadingPhase:
            vram = cls.get_phase_vram_mb(phase)
            total += vram
            models = cls.get_models_by_phase(phase)
            print(f"\n  Phase {phase.name} ({vram / 1024:.1f} GB):")
            for m in models:
                print(f"    - {m.name}: {m.estimated_vram_mb}MB ({m.precision.value})")
        print(f"\n  Total (all loaded): {total / 1024:.1f} GB")
        print("  RTX 4080 Budget:    16.0 GB")
        print(f"  {'✅ Fits' if total / 1024 <= 16 else '⚠️  Requires model swapping'}")
