"""
IntelliVoice — OmniVoice Generation Layer

Converts response text into natural speech.
Uses k2-fsa/OmniVoice for fast, high-quality zero-shot TTS
with a fixed reference voice (badri.wav).

Speed optimizations:
    1. Reduced num_step from 32 → 16 (~2x faster inference)
    2. omnivoice-triton Triton kernel fusion when available (~3.4x speedup)
    3. Persistent CUDA stream (avoids per-call allocation overhead)
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple
import torch
import numpy as np

from config.logging_config import get_logger
from config.model_registry import ModelRegistry

logger = get_logger("omnivoice_tts")

# Number of diffusion steps: 16 is ~2x faster than default 32 with
# minimal quality loss on OmniVoice's NAR architecture.
_DEFAULT_NUM_STEPS = 16


class OmniVoiceSynthesizer:
    def __init__(self):
        self.model = None
        self._is_loaded = False
        self._config = ModelRegistry.OMNIVOICE
        self._sample_rate = 24000
        self._ref_path = None
        self._ref_text = None
        self._cuda_stream: Optional[torch.cuda.Stream] = None
        self._triton_optimized = False

    async def load(self, device: torch.device = None) -> None:
        """Load OmniVoice with optional Triton kernel fusion."""
        if self._is_loaded:
            return

        logger.info("loading_omnivoice", model_id=self._config.hf_model_id)
        try:
            from omnivoice import OmniVoice

            dev_str = str(device) if device else "auto"

            self.model = OmniVoice.from_pretrained(
                self._config.hf_model_id,
                device_map=dev_str,
                dtype=torch.float16 if self._config.precision.value == "fp16" else torch.float32,
            )

            # Apply Triton kernel fusion if omnivoice-triton is installed
            try:
                from omnivoice_triton import optimize_model
                self.model = optimize_model(self.model)
                self._triton_optimized = True
                logger.info("triton_optimization_applied", speedup="~3.4x")
            except ImportError:
                logger.info(
                    "triton_optimization_unavailable",
                    hint="Install omnivoice-triton for ~3.4x faster TTS: pip install omnivoice-triton",
                )
            except Exception as e:
                logger.warning("triton_optimization_failed", error=str(e))

            # Create a persistent CUDA stream to avoid per-call overhead
            if torch.cuda.is_available():
                self._cuda_stream = torch.cuda.Stream()

            # Pre-resolve reference audio path once
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            ref_path = os.path.join(base_dir, "assets", "badri.wav")

            if os.path.exists(ref_path):
                self._ref_path = ref_path
                self._ref_text = (
                    "My strengths are problem solving, quick learning, and the ability to "
                    "convert business requirements into working technical solutions. Currently, "
                    "I am looking for opportunities where I can contribute as a software engineer, "
                    "work on large scale systems, and continue learning from experienced teams."
                )
                logger.info("reference_audio_found", path=ref_path)
            else:
                logger.warning(
                    "reference_audio_missing",
                    expected_path=ref_path,
                    hint="Place badri.wav here to lock the voice!",
                )

            self._is_loaded = True
            logger.info(
                "omnivoice_loaded_successfully",
                triton_optimized=self._triton_optimized,
                num_steps=_DEFAULT_NUM_STEPS,
            )
        except ImportError:
            logger.error("omnivoice_not_installed", hint="Run: uv pip install omnivoice")
            raise
        except Exception as e:
            logger.error("omnivoice_load_failed", error=str(e))
            raise

    def synthesize(
        self,
        text: str,
        language: str = "en",
        speaking_rate: float = 1.0,
    ) -> Tuple[torch.Tensor, int]:
        """
        Synthesize speech using OmniVoice with fixed reference voice.
        Uses reduced diffusion steps (16) for ~2x faster inference.
        """
        if not self._is_loaded:
            raise RuntimeError("OmniVoice is not loaded.")

        t0 = time.perf_counter()
        try:
            kwargs = {
                "text": text,
                "num_step": _DEFAULT_NUM_STEPS,
            }

            if self._ref_path:
                kwargs["ref_audio"] = self._ref_path
                kwargs["ref_text"] = self._ref_text

            # OmniVoice returns a list of np.ndarray with shape (T,)
            if self._cuda_stream is not None:
                with torch.cuda.stream(self._cuda_stream):
                    audio_arrays = self.model.generate(**kwargs)
            else:
                audio_arrays = self.model.generate(**kwargs)

            if not audio_arrays or len(audio_arrays) == 0:
                raise ValueError("OmniVoice returned empty audio.")

            waveform = torch.tensor(audio_arrays[0], dtype=torch.float32).unsqueeze(0)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            audio_duration_ms = (waveform.shape[1] / self._sample_rate) * 1000
            rtf = elapsed_ms / audio_duration_ms if audio_duration_ms > 0 else 0
            logger.info(
                "tts_synthesized",
                text_len=len(text),
                elapsed_ms=round(elapsed_ms),
                audio_ms=round(audio_duration_ms),
                rtf=round(rtf, 3),
                triton=self._triton_optimized,
            )

            return waveform, self._sample_rate

        except Exception as e:
            logger.error("omnivoice_synthesis_failed", error=str(e), text_length=len(text))
            raise

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
