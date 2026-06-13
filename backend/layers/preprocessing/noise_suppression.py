"""
IntelliVoice — Noise Suppression

Layer 1 component: removes background noise from audio for improved
downstream model performance.

Primary backend: DeepFilterNet
High-performance real-time AI noise suppression.
"""

from __future__ import annotations

import torch

from config.logging_config import get_logger

logger = get_logger("noise_suppression")


class NoiseSuppressor:
    """
    Noise suppression using DeepFilterNet.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._is_loaded = False
        self.backend: str = "passthrough"
        self.model = None
        self.df_state = None

    async def load(self) -> None:
        """Load the DeepFilterNet model."""
        if self._is_loaded:
            return

        logger.info("loading_noise_suppressor", preferred="deepfilternet")

        try:
            from df.enhance import init_df
            
            # init_df returns (model, df_state, suffix)
            self.model, self.df_state, _ = init_df()
            
            self._is_loaded = True
            self.backend = "deepfilternet"
            logger.info("deepfilternet_loaded")
            return
        except ImportError:
            logger.warning(
                "deepfilternet_not_available",
                hint="Install with: uv pip install deepfilternet",
            )
        except Exception as e:
            logger.error("deepfilternet_load_failed", error=str(e))

        self._is_loaded = False
        self.backend = "passthrough"
        logger.warning("noise_suppressor_passthrough", reason="no_backend_available")

    def suppress_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply noise suppression to a waveform.

        Args:
            waveform: Audio tensor [1, T] at 16kHz.

        Returns:
            Denoised waveform tensor [1, T].
        """
        if not self._is_loaded:
            return waveform

        try:
            from df.enhance import enhance
            import torchaudio
            
            # DeepFilterNet expects [channels, time] in float32.
            # It also has an internal expected sample rate, usually 48kHz.
            # We must resample to df_state.sr() before processing, and back to 16kHz after.
            target_sr = self.df_state.sr()
            
            # Move to CPU for DeepFilterNet processing
            audio_cpu = waveform.cpu()
            
            # Resample up to DeepFilterNet's required sample rate
            if self.sample_rate != target_sr:
                audio_resampled = torchaudio.functional.resample(audio_cpu, self.sample_rate, target_sr)
            else:
                audio_resampled = audio_cpu
                
            # Enhance
            enhanced_audio = enhance(self.model, self.df_state, audio_resampled)
            
            # Resample back to our pipeline's sample rate (16kHz)
            if self.sample_rate != target_sr:
                enhanced_audio = torchaudio.functional.resample(enhanced_audio, target_sr, self.sample_rate)
                
            # Move back to original device
            return enhanced_audio.to(waveform.device)

        except Exception as e:
            logger.error(
                "noise_suppression_failed",
                backend=self.backend,
                error=str(e),
            )
            return waveform  # Passthrough on error

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
