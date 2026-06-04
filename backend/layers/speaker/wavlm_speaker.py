"""
IntelliVoice — WavLM Speaker Understanding

Layer 5: Extracts speaker identity embeddings for verification
and voice cloning using microsoft/wavlm-large.
"""

from __future__ import annotations

from typing import Optional

import torch
import numpy as np
from transformers import WavLMModel, AutoFeatureExtractor

from config.logging_config import get_logger
from config.model_registry import ModelRegistry

logger = get_logger("wavlm_speaker")


class WavLMSpeakerEncoder:
    """
    Speaker understanding using WavLM Large.

    Extracts:
        - Speaker identity embeddings
        - Speaker verification scores
        - Voice characteristic vectors
    """

    def __init__(self):
        self.model: Optional[WavLMModel] = None
        self.feature_extractor = None
        self._device: torch.device = torch.device("cpu")
        self._is_loaded = False
        self._config = ModelRegistry.WAVLM_LARGE

    async def load(self, device: torch.device = torch.device("cuda")) -> None:
        """Load WavLM Large model."""
        if self._is_loaded:
            return

        logger.info("loading_wavlm", model=self._config.hf_model_id)
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self._config.hf_model_id,
            )
            self.model = WavLMModel.from_pretrained(
                self._config.hf_model_id,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )
            self.model = self.model.to(device)
            self.model.eval()
            self._device = device
            self._is_loaded = True
            logger.info("wavlm_loaded", device=str(device))
        except Exception as e:
            logger.error("wavlm_load_failed", error=str(e))
            raise

    @torch.inference_mode()
    def extract_speaker_embedding(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Extract speaker identity embedding.

        Args:
            waveform: Audio tensor [1, T] or [T] at 16kHz.

        Returns:
            Speaker embedding tensor [D] (D=1024 for WavLM Large).
        """
        if not self._is_loaded:
            raise RuntimeError("WavLM not loaded.")

        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)

        inputs = self.feature_extractor(
            waveform.cpu().numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self._device)

        if next(self.model.parameters()).dtype == torch.float16:
            input_values = input_values.half()

        outputs = self.model(input_values)
        # Mean pool across time to get utterance-level embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)

        logger.debug("speaker_embedding_extracted", dim=embedding.shape[0])
        return embedding.cpu().float()

    @torch.inference_mode()
    def verify_speaker(
        self,
        waveform1: torch.Tensor,
        waveform2: torch.Tensor,
        threshold: float = 0.7,
    ) -> dict:
        """
        Verify if two audio samples are from the same speaker.

        Returns:
            Dict with 'is_same_speaker', 'similarity', 'threshold'.
        """
        emb1 = self.extract_speaker_embedding(waveform1)
        emb2 = self.extract_speaker_embedding(waveform2)

        similarity = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), emb2.unsqueeze(0)
        ).item()

        return {
            "is_same_speaker": similarity >= threshold,
            "similarity": round(similarity, 4),
            "threshold": threshold,
        }

    def offload_to_cpu(self) -> None:
        """Offload to CPU."""
        if self.model is not None:
            self.model = self.model.cpu()
            torch.cuda.empty_cache()
            self._device = torch.device("cpu")
            logger.info("wavlm_offloaded_to_cpu")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
