"""
IntelliVoice — XLS-R 1B Acoustic Encoder

Layer 2: Converts raw waveform into rich speech embeddings that
preserve accent, emotion, prosody, and speaker information.

Uses facebook/wav2vec2-xls-r-1b — a 1B parameter model
pretrained on 436K hours of speech in 128 languages.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

from config.logging_config import get_logger
from config.model_registry import ModelRegistry

logger = get_logger("xlsr_encoder")


class XLSREncoder:
    """
    XLS-R 1B acoustic encoder.

    Extracts rich speech representations from raw waveforms,
    preserving paralinguistic features like accent, emotion,
    prosody, and speaker characteristics.

    Architecture:
        Raw Audio → CNN Feature Extractor → Transformer Encoder → Speech Embeddings
    """

    def __init__(self):
        self.model: Optional[Wav2Vec2Model] = None
        self.feature_extractor: Optional[Wav2Vec2FeatureExtractor] = None
        self._device: torch.device = torch.device("cpu")
        self._is_loaded = False
        self._config = ModelRegistry.XLSR_1B

    async def load(self, device: torch.device = torch.device("cuda")) -> None:
        """Load the XLS-R 1B model from HuggingFace."""
        if self._is_loaded:
            return

        logger.info("loading_xlsr_encoder", model=self._config.hf_model_id)

        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self._config.hf_model_id,
            )

            self.model = Wav2Vec2Model.from_pretrained(
                self._config.hf_model_id,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )
            self.model = self.model.to(device)
            self.model.eval()
            self._device = device
            self._is_loaded = True

            # Log model info
            num_params = sum(p.numel() for p in self.model.parameters()) / 1e9
            logger.info(
                "xlsr_encoder_loaded",
                params_b=f"{num_params:.2f}",
                device=str(device),
                dtype=str(next(self.model.parameters()).dtype),
            )
        except Exception as e:
            logger.error("xlsr_encoder_load_failed", error=str(e))
            raise

    @torch.inference_mode()
    def encode(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        output_hidden_states: bool = False,
    ) -> dict:
        """
        Encode a waveform into speech embeddings.

        Args:
            waveform: Audio tensor [1, T] or [T] at 16kHz.
            sample_rate: Input sample rate (must be 16kHz).
            output_hidden_states: If True, return all hidden states.

        Returns:
            Dict with:
                - 'embeddings': [B, T', D] speech embeddings (D=1280 for XLS-R 1B)
                - 'hidden_states': Optional list of all layer hidden states
        """
        if not self._is_loaded:
            raise RuntimeError("XLS-R encoder not loaded.")

        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)

        # Process through feature extractor
        inputs = self.feature_extractor(
            waveform.cpu().numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self._device)

        # Cast to model dtype
        if next(self.model.parameters()).dtype == torch.float16:
            input_values = input_values.half()

        # Forward pass
        outputs = self.model(
            input_values,
            output_hidden_states=output_hidden_states,
        )

        result = {
            "embeddings": outputs.last_hidden_state,  # [B, T', 1280]
        }

        if output_hidden_states and outputs.hidden_states:
            result["hidden_states"] = outputs.hidden_states

        logger.debug(
            "xlsr_encoded",
            input_samples=waveform.shape[-1],
            output_shape=str(result["embeddings"].shape),
        )

        return result

    @torch.inference_mode()
    def encode_batch(
        self,
        waveforms: list[torch.Tensor],
        sample_rate: int = 16000,
    ) -> dict:
        """
        Encode a batch of waveforms.

        Args:
            waveforms: List of audio tensors, each [T].
            sample_rate: Input sample rate.

        Returns:
            Dict with 'embeddings': [B, T', D] padded tensor.
        """
        if not self._is_loaded:
            raise RuntimeError("XLS-R encoder not loaded.")

        # Process through feature extractor with padding
        raw_arrays = [w.squeeze().cpu().numpy() for w in waveforms]
        inputs = self.feature_extractor(
            raw_arrays,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self._device)

        if next(self.model.parameters()).dtype == torch.float16:
            input_values = input_values.half()

        outputs = self.model(input_values)

        return {
            "embeddings": outputs.last_hidden_state,
        }

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension (1280 for XLS-R 1B)."""
        return 1280

    def offload_to_cpu(self) -> None:
        """Move model to CPU to free GPU memory."""
        if self.model is not None:
            self.model = self.model.cpu()
            torch.cuda.empty_cache()
            self._device = torch.device("cpu")
            logger.info("xlsr_offloaded_to_cpu")

    def to_gpu(self, device: torch.device = torch.device("cuda")) -> None:
        """Move model back to GPU."""
        if self.model is not None:
            self.model = self.model.to(device)
            self._device = device
            logger.info("xlsr_moved_to_gpu", device=str(device))

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
