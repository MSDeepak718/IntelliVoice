"""
IntelliVoice — Silero VAD (Voice Activity Detection)

Layer 1 component: Detects speech segments in audio streams.
Uses Silero VAD via torch.hub for lightweight, production-proven detection.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torchaudio

from config import get_settings
from config.logging_config import get_logger

logger = get_logger("silero_vad")


class SileroVAD:
    """
    Voice Activity Detection using Silero VAD.

    Detects speech start/end, removes silence, and reduces
    unnecessary computation for downstream models.
    """

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.model: Optional[torch.nn.Module] = None
        self._utils = None
        self._is_loaded = False

    async def load(self, device: torch.device = torch.device("cpu")) -> None:
        """Load the Silero VAD model from torch hub."""
        if self._is_loaded:
            return

        logger.info("loading_silero_vad")
        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self.model = model.to(device)
            self._utils = utils
            self._is_loaded = True
            logger.info("silero_vad_loaded", device=str(device))
        except Exception as e:
            logger.error("silero_vad_load_failed", error=str(e))
            raise

    def detect_speech(
        self,
        waveform: torch.Tensor,
        return_seconds: bool = True,
    ) -> List[dict]:
        """
        Detect speech segments in a waveform.

        Args:
            waveform: Audio tensor [1, T] or [T] at 16kHz.
            return_seconds: If True, return timestamps in seconds.

        Returns:
            List of dicts with 'start' and 'end' keys.
        """
        if not self._is_loaded:
            raise RuntimeError("VAD model not loaded. Call load() first.")

        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)

        # Get speech timestamps using Silero utility
        get_speech_timestamps = self._utils[0]
        speech_timestamps = get_speech_timestamps(
            waveform,
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            return_seconds=return_seconds,
        )

        segments = []
        for ts in speech_timestamps:
            segments.append({
                "start": ts["start"],
                "end": ts["end"],
            })

        logger.debug(
            "vad_detection",
            segments=len(segments),
            audio_duration=f"{len(waveform) / self.sample_rate:.2f}s",
        )

        return segments

    def is_speech(self, waveform: torch.Tensor) -> bool:
        """
        Check if a short audio chunk contains speech.

        Args:
            waveform: Audio chunk [1, T] or [T].

        Returns:
            True if speech is detected.
        """
        if not self._is_loaded:
            raise RuntimeError("VAD model not loaded.")

        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)

        # Use the model's __call__ for single-chunk detection
        with torch.no_grad():
            speech_prob = self.model(waveform, self.sample_rate).item()

        return speech_prob >= self.threshold

    def get_speech_probability(self, waveform: torch.Tensor) -> float:
        """Get speech probability for a chunk."""
        if not self._is_loaded:
            raise RuntimeError("VAD model not loaded.")

        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)

        with torch.no_grad():
            return self.model(waveform, self.sample_rate).item()

    def extract_speech(
        self,
        waveform: torch.Tensor,
        padding_ms: int = 50,
    ) -> Tuple[torch.Tensor, List[dict]]:
        """
        Extract only speech segments from a waveform.

        Args:
            waveform: Full audio [1, T] at 16kHz.
            padding_ms: Padding around detected segments.

        Returns:
            Tuple of (speech-only waveform, segments).
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        segments = self.detect_speech(waveform, return_seconds=False)

        if not segments:
            return torch.zeros(1, 0), []

        padding_samples = int(self.sample_rate * padding_ms / 1000)
        speech_chunks = []

        for seg in segments:
            start = max(0, seg["start"] - padding_samples)
            end = min(waveform.shape[1], seg["end"] + padding_samples)
            speech_chunks.append(waveform[:, start:end])

        # Concatenate all speech segments
        speech_waveform = torch.cat(speech_chunks, dim=1)

        return speech_waveform, segments

    def reset_state(self) -> None:
        """Reset the VAD model's internal state (for streaming)."""
        if self.model is not None:
            self.model.reset_states()

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
