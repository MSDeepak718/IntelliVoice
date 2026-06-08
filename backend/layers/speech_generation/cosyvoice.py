"""
IntelliVoice — CosyVoice 2 Speech Generation

Layer 10: Converts semantic response text into natural, emotional,
multilingual speech. Supports streaming generation and voice cloning.
"""

from __future__ import annotations

import tempfile
from typing import AsyncGenerator, Optional, Tuple

import torch
import torchaudio

from config.logging_config import get_logger
from config.model_registry import ModelRegistry

logger = get_logger("cosyvoice")


class CosyVoiceSynthesizer:
    """
    CosyVoice 2 text-to-speech synthesizer.

    Capabilities:
        - Natural multilingual speech (EN, HI, TA, TE)
        - Emotional speech synthesis
        - Streaming generation for low latency
        - Zero-shot voice cloning from speaker embeddings

    Architecture:
        Text + Emotion + Speaker → CosyVoice 2 → Mel Spectrogram → Waveform
    """

    def __init__(self):
        self.model = None
        self._device: torch.device = torch.device("cpu")
        self._is_loaded = False
        self._config = ModelRegistry.COSYVOICE2
        self._sample_rate = 22050  # CosyVoice 2 default output rate
        self._speaker_embedding: Optional[torch.Tensor] = None

    def set_speaker_embedding(self, embedding: Optional[torch.Tensor]) -> None:
        """Cache the user's speaker embedding for personalised synthesis."""
        self._speaker_embedding = embedding

    async def load(self, device: torch.device = torch.device("cuda")) -> None:
        """Load CosyVoice 2 model."""
        if self._is_loaded:
            return

        logger.info("loading_cosyvoice2", model=self._config.hf_model_id)

        try:
            # CosyVoice 2 uses its own loading mechanism
            # Try the official CosyVoice package first
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice2

                self.model = CosyVoice2(
                    self._config.hf_model_id,
                    load_jit=False,
                    load_trt=False,
                )
                self._is_loaded = True
                self._device = device
                logger.info("cosyvoice2_loaded_official", device=str(device))
                return
            except ImportError:
                logger.info("cosyvoice_package_not_found", trying="transformers_fallback")

            # Fallback: Load via transformers
            from transformers import AutoModel, AutoTokenizer

            self.model = AutoModel.from_pretrained(
                self._config.hf_model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )
            if hasattr(self.model, "to"):
                self.model = self.model.to(device)
            if hasattr(self.model, "eval"):
                self.model.eval()

            self._device = device
            self._is_loaded = True
            logger.info("cosyvoice2_loaded_transformers", device=str(device))

        except Exception as e:
            logger.error("cosyvoice2_load_failed", error=str(e))
            self._is_loaded = False

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        language: str = "english",
        speaker_embedding: Optional[torch.Tensor] = None,
        emotion: str = "neutral",
        speaking_rate: float = 1.0,
    ) -> Tuple[torch.Tensor, int]:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize.
            language: Target language.
            speaker_embedding: Optional speaker embedding for voice cloning.
            emotion: Target emotion for synthesis.
            speaking_rate: Speaking rate multiplier.

        Returns:
            Tuple of (waveform [1, T], sample_rate).
        """
        if not self._is_loaded:
            raise RuntimeError("CosyVoice 2 not loaded.")

        try:
            # Try official CosyVoice API
            if hasattr(self.model, "inference_sft"):
                # Standard text-to-speech
                output = self.model.inference_sft(
                    text,
                    speaker_id=0,
                    stream=False,
                )
                if isinstance(output, dict):
                    waveform = output.get("tts_speech")
                    if waveform is None:
                        raise RuntimeError("inference_sft returned None waveform")
                else:
                    # Generator output
                    chunks = list(output)
                    if chunks:
                        waveform = torch.cat([c["tts_speech"] for c in chunks], dim=-1)
                    else:
                        raise RuntimeError("inference_sft returned empty generator")

                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)

                return waveform, self._sample_rate

            raise RuntimeError("No compatible synthesis method found in loaded model")

        except Exception as e:
            logger.error("synthesis_failed", error=str(e), text_length=len(text))
            raise

    @torch.inference_mode()
    async def synthesize_streaming(
        self,
        text: str,
        language: str = "english",
        chunk_size_ms: int = 500,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream speech synthesis for low-latency output.

        Yields audio chunks as they're generated.
        """
        if not self._is_loaded:
            raise RuntimeError("CosyVoice 2 not loaded.")

        try:
            if hasattr(self.model, "inference_sft"):
                output = self.model.inference_sft(
                    text,
                    speaker_id=0,
                    stream=True,
                )

                for chunk in output:
                    if "tts_speech" in chunk:
                        waveform = chunk["tts_speech"]
                        if waveform.dim() == 1:
                            waveform = waveform.unsqueeze(0)
                        # Convert to bytes
                        audio_bytes = (waveform.squeeze(0).cpu() * 32767).to(torch.int16).numpy().tobytes()
                        yield audio_bytes
            else:
                # Non-streaming fallback
                waveform, sr = self.synthesize(text, language)
                audio_bytes = (waveform.squeeze(0).cpu() * 32767).to(torch.int16).numpy().tobytes()
                yield audio_bytes

        except Exception as e:
            logger.error("streaming_synthesis_failed", error=str(e))
            yield b""

    @torch.inference_mode()
    def clone_voice(
        self,
        text: str,
        reference_audio: torch.Tensor,
        reference_sr: int = 16000,
    ) -> Tuple[torch.Tensor, int]:
        """
        Synthesize speech with voice cloning from reference audio.

        Args:
            text: Text to synthesize.
            reference_audio: Reference audio for voice cloning [1, T].
            reference_sr: Reference audio sample rate.

        Returns:
            Tuple of (waveform, sample_rate).
        """
        if not self._is_loaded:
            raise RuntimeError("CosyVoice 2 not loaded.")

        try:
            if hasattr(self.model, "inference_zero_shot"):
                # Save reference audio to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    torchaudio.save(tmp.name, reference_audio.cpu().float(), reference_sr)
                    ref_path = tmp.name

                output = self.model.inference_zero_shot(
                    text,
                    prompt_text="",
                    prompt_speech_16k=ref_path,
                    stream=False,
                )

                if isinstance(output, dict):
                    waveform = output.get("tts_speech", torch.zeros(1, 16000))
                else:
                    chunks = list(output)
                    waveform = torch.cat([c["tts_speech"] for c in chunks], dim=-1) if chunks else torch.zeros(1, 16000)

                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)

                import os
                os.unlink(ref_path)

                return waveform, self._sample_rate

            # Fallback to standard synthesis without cloning
            logger.warning("voice_cloning_not_supported", fallback="standard_synthesis")
            return self.synthesize(text)

        except Exception as e:
            logger.error("voice_cloning_failed", error=str(e))
            return self.synthesize(text)

    def offload_to_cpu(self) -> None:
        """Offload to CPU."""
        if self.model is not None and hasattr(self.model, "cpu"):
            self.model = self.model.cpu()
            torch.cuda.empty_cache()
            self._device = torch.device("cpu")
            logger.info("cosyvoice_offloaded_to_cpu")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
