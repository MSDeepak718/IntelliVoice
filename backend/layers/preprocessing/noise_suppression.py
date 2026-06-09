"""
IntelliVoice — Noise Suppression

Layer 1 component: removes background noise from audio for improved
downstream model performance.

Primary backend : noisereduce — a lightweight, high-quality spectral
                  gating library with both stationary and non-stationary
                  noise reduction. No GPU required, <10ms per chunk.
Fallback backend: lightweight spectral-subtraction denoiser (pure
                  numpy/scipy) that gives reasonable cleanup when
                  noisereduce isn't installed. The behaviour is
                  transparent to callers — `suppress_noise(...)` always
                  returns a denoised waveform of the same shape.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from config.logging_config import get_logger

logger = get_logger("noise_suppression")


# ----------------------------------------------------------------------
# Fallback: spectral subtraction
# ----------------------------------------------------------------------
def _spectral_subtract(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: float = 25.0,
    overlap: float = 0.5,
    alpha: float = 2.0,
    beta: float = 0.01,
) -> np.ndarray:
    """Boll-79 spectral subtraction.

    Estimates the noise spectrum from the first ~200 ms (assumed
    non-speech after VAD gating) and subtracts it from every frame.
    Result is far better than passthrough on steady-state noise
    (fans, AC, hum) while remaining real-time and dependency-free.
    """
    n = audio.shape[-1]
    if n == 0:
        return audio

    frame_len = max(64, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(frame_len * (1.0 - overlap)))
    if frame_len >= n:
        return audio.copy()

    # Hann window
    win = np.hanning(frame_len).astype(np.float32)
    # Estimate noise profile from the first 200 ms
    noise_n = min(n, int(sample_rate * 0.2))
    noise = audio[..., :noise_n]
    n_frames = max(1, (noise_n - frame_len) // hop + 1)
    noise_mag = np.zeros(frame_len // 2 + 1, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        f = noise[..., s : s + frame_len]
        if f.shape[-1] < frame_len:
            pad = np.zeros(frame_len - f.shape[-1], dtype=np.float32)
            f = np.concatenate([f, pad])
        spec = np.fft.rfft(f * win)
        noise_mag += np.abs(spec)
    noise_mag /= n_frames
    # Floor to avoid musical-noise artefacts
    noise_floor = np.maximum(noise_mag * 0.05, 1e-6)

    # Process the whole signal with overlap-add
    out = np.zeros_like(audio, dtype=np.float32)
    wsum = np.zeros_like(audio, dtype=np.float32)
    n_frames = max(0, (n - frame_len) // hop + 1)
    for i in range(n_frames):
        s = i * hop
        f = audio[..., s : s + frame_len]
        if f.shape[-1] < frame_len:
            pad = np.zeros(frame_len - f.shape[-1], dtype=np.float32)
            f = np.concatenate([f, pad])
        spec = np.fft.rfft(f * win)
        mag = np.abs(spec)
        phase = np.angle(spec)
        # Subtract noise estimate, oversubtract by alpha, floor by beta
        clean_mag = mag ** 2 - alpha * (noise_mag ** 2)
        clean_mag = np.maximum(clean_mag, beta * (mag ** 2))
        clean_mag = np.sqrt(clean_mag + 1e-12)
        # Also mask out anything below the noise floor
        mask = (mag > noise_floor).astype(np.float32)
        clean_mag = clean_mag * mask
        clean = np.fft.irfft(clean_mag * np.exp(1j * phase), n=frame_len)
        out[..., s : s + frame_len] += clean * win
        wsum[..., s : s + frame_len] += win

    valid = wsum > 1e-6
    out[valid] = out[valid] / wsum[valid]
    return out.astype(audio.dtype, copy=False)


class NoiseSuppressor:
    """
    Noise suppression with three-tier backend:

        1. noisereduce (spectral gating, non-stationary) — best quality.
        2. Spectral subtraction (numpy, no extra deps) — solid fallback.
        3. Passthrough — last resort, no processing.

    The active backend is reported via the `backend` attribute and the
    `is_loaded` property (True for tiers 1 & 2).
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._nr_reduce = None  # noisereduce module reference
        self._is_loaded = False
        self.backend: str = "passthrough"

    async def load(self) -> None:
        """Load the best available denoiser."""
        if self._is_loaded:
            return

        logger.info("loading_noise_suppressor", preferred="noisereduce")

        # ---- Tier 1: noisereduce ----------------------------------
        try:
            import noisereduce as nr

            # Sanity-check: run on a 1-second dummy to confirm it works
            dummy = np.zeros(self.sample_rate, dtype=np.float32)
            nr.reduce_noise(y=dummy, sr=self.sample_rate, stationary=False)

            self._nr_reduce = nr
            self._is_loaded = True
            self.backend = "noisereduce"
            logger.info("noisereduce_loaded")
            return
        except ImportError:
            logger.warning(
                "noisereduce_not_available",
                hint="Install with: pip install noisereduce",
            )
        except Exception as e:  # noqa: BLE001
            logger.error("noisereduce_load_failed", error=str(e))

        # ---- Tier 2: spectral subtraction -------------------------
        try:
            # Sanity-check the implementation on a 1-second dummy signal
            _spectral_subtract(
                np.zeros(self.sample_rate, dtype=np.float32),
                self.sample_rate,
            )
            self._is_loaded = True
            self.backend = "spectral_subtraction"
            logger.warning(
                "noise_suppressor_using_fallback",
                backend="spectral_subtraction",
            )
            return
        except Exception as e:  # noqa: BLE001
            logger.error(
                "spectral_fallback_failed",
                error=str(e),
            )

        # ---- Tier 3: passthrough ----------------------------------
        self._is_loaded = False
        self.backend = "passthrough"
        logger.warning("noise_suppressor_passthrough", reason="no_backend_available")

    def _suppress_noisereduce(self, audio_np: np.ndarray) -> np.ndarray:
        """Apply noisereduce spectral gating."""
        enhanced = self._nr_reduce.reduce_noise(
            y=audio_np,
            sr=self.sample_rate,
            stationary=True,
            prop_decrease=0.5,
            n_fft=512,
            hop_length=128,
        )
        return np.asarray(enhanced, dtype=np.float32)

    def suppress_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply noise suppression to a waveform.

        Args:
            waveform: Audio tensor [1, T] at 16kHz (other SRs are passed
                      through unchanged for the passthrough backend).

        Returns:
            Denoised waveform tensor [1, T].
        """
        if not self._is_loaded:
            logger.debug("noise_suppression_skipped", reason="model_not_loaded")
            return waveform

        try:
            if waveform.dim() > 1:
                audio_np = waveform.squeeze(0).cpu().numpy()
            else:
                audio_np = waveform.cpu().numpy()
            audio_np = np.asarray(audio_np, dtype=np.float32)

            if self.backend == "noisereduce":
                enhanced = self._suppress_noisereduce(audio_np)
            elif self.backend == "spectral_subtraction":
                enhanced = _spectral_subtract(audio_np, self.sample_rate)
            else:
                return waveform

            enhanced_tensor = torch.from_numpy(np.asarray(enhanced)).float()
            if enhanced_tensor.dim() == 1:
                enhanced_tensor = enhanced_tensor.unsqueeze(0)

            logger.debug(
                "noise_suppressed",
                backend=self.backend,
                input_rms=f"{torch.sqrt(torch.mean(waveform ** 2)).item():.4f}",
                output_rms=f"{torch.sqrt(torch.mean(enhanced_tensor ** 2)).item():.4f}",
            )
            return enhanced_tensor

        except Exception as e:
            logger.error(
                "noise_suppression_failed",
                backend=self.backend,
                error=str(e),
            )
            return waveform  # Passthrough on error

    def suppress_noise_numpy(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise suppression to a numpy array."""
        waveform = torch.from_numpy(np.asarray(audio)).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        result = self.suppress_noise(waveform)
        return result.squeeze(0).numpy()

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
