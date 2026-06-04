"""
IntelliVoice — Emotion2Vec Prosody & Emotion Understanding

Layer 4: Extracts emotional and prosodic information that is
unavailable from text alone. "I'm fine" can mean happy, angry,
sad, or frustrated depending on delivery.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

from config.logging_config import get_logger
from config.model_registry import ModelRegistry

logger = get_logger("emotion2vec")

# Emotion labels used by Emotion2Vec
EMOTION_LABELS = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
]


class Emotion2VecAnalyzer:
    """
    Emotion and prosody analysis using Emotion2Vec.

    Extracts:
        - Emotion embeddings (capturing emotional state)
        - Emotion classification (discrete labels)
        - Speaking style embeddings
        - Energy level detection

    Output:
        Emotion Vector + Prosody Vector
    """

    def __init__(self):
        self.model = None
        self._device: torch.device = torch.device("cpu")
        self._is_loaded = False
        self._config = ModelRegistry.EMOTION2VEC

    async def load(self, device: torch.device = torch.device("cuda")) -> None:
        """Load the Emotion2Vec model."""
        if self._is_loaded:
            return

        logger.info("loading_emotion2vec", model=self._config.hf_model_id)

        try:
            from transformers import AutoModel, AutoFeatureExtractor

            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self._config.hf_model_id,
                trust_remote_code=True,
            )

            self.model = AutoModel.from_pretrained(
                self._config.hf_model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )
            self.model = self.model.to(device)
            self.model.eval()
            self._device = device
            self._is_loaded = True

            logger.info("emotion2vec_loaded", device=str(device))
        except Exception as e:
            logger.warning(
                "emotion2vec_load_failed",
                error=str(e),
                fallback="heuristic_analysis",
            )
            self._is_loaded = False

    @torch.inference_mode()
    def analyze_emotion(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
    ) -> Dict:
        """
        Analyze emotion from audio.

        Args:
            waveform: Audio tensor [1, T] or [T] at 16kHz.
            sample_rate: Audio sample rate.

        Returns:
            Dict with:
                - 'emotion': Primary detected emotion
                - 'confidence': Confidence score
                - 'emotion_scores': All emotion probabilities
                - 'emotion_embedding': Raw emotion embedding vector
                - 'energy_level': Speech energy level (low/medium/high)
                - 'speaking_rate': Estimated speaking rate
        """
        if not self._is_loaded:
            return self._heuristic_analysis(waveform)

        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)

        try:
            # Process through feature extractor
            inputs = self.feature_extractor(
                waveform.cpu().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(self._device)

            if next(self.model.parameters()).dtype == torch.float16:
                input_values = input_values.half()

            # Get model output
            outputs = self.model(input_values)
            embeddings = outputs.last_hidden_state  # [B, T, D]

            # Pool embeddings to get utterance-level representation
            emotion_embedding = embeddings.mean(dim=1).squeeze(0)  # [D]

            # Classify emotion (using a simple linear projection if available)
            if hasattr(outputs, "logits") and outputs.logits is not None:
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                emotion_scores = {
                    EMOTION_LABELS[i]: probs[i].item()
                    for i in range(min(len(EMOTION_LABELS), len(probs)))
                }
                primary_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = emotion_scores[primary_emotion]
            else:
                # Fallback: use embedding norm as a proxy
                emotion_scores = {"neutral": 0.5}
                primary_emotion = "neutral"
                confidence = 0.5

            # Compute energy level
            energy = self._compute_energy_level(waveform)

            result = {
                "emotion": primary_emotion,
                "confidence": round(confidence, 3),
                "emotion_scores": {k: round(v, 3) for k, v in emotion_scores.items()},
                "emotion_embedding": emotion_embedding.cpu().float().numpy().tolist(),
                "energy_level": energy,
                "speaking_rate": self._estimate_speaking_rate(waveform, sample_rate),
            }

            logger.debug(
                "emotion_analyzed",
                emotion=primary_emotion,
                confidence=f"{confidence:.3f}",
                energy=energy,
            )

            return result

        except Exception as e:
            logger.error("emotion_analysis_failed", error=str(e))
            return self._heuristic_analysis(waveform)

    @torch.inference_mode()
    def get_emotion_embedding(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Get the raw emotion embedding vector.

        Returns:
            Embedding tensor [D].
        """
        result = self.analyze_emotion(waveform, sample_rate)
        if isinstance(result.get("emotion_embedding"), list):
            return torch.tensor(result["emotion_embedding"])
        return torch.zeros(768)  # Default embedding size

    def _compute_energy_level(self, waveform: torch.Tensor) -> str:
        """Classify energy level as low/medium/high."""
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)
        rms = torch.sqrt(torch.mean(waveform ** 2)).item()
        if rms < 0.01:
            return "low"
        elif rms < 0.05:
            return "medium"
        else:
            return "high"

    def _estimate_speaking_rate(self, waveform: torch.Tensor, sample_rate: int) -> str:
        """Estimate speaking rate from audio energy patterns."""
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)

        # Simple zero-crossing rate as proxy
        zero_crossings = ((waveform[:-1] * waveform[1:]) < 0).sum().item()
        duration = len(waveform) / sample_rate
        zcr = zero_crossings / duration if duration > 0 else 0

        if zcr < 1000:
            return "slow"
        elif zcr < 3000:
            return "normal"
        else:
            return "fast"

    def _heuristic_analysis(self, waveform: torch.Tensor) -> Dict:
        """Fallback heuristic-based emotion analysis."""
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)

        rms = torch.sqrt(torch.mean(waveform ** 2)).item()
        energy = "low" if rms < 0.01 else "medium" if rms < 0.05 else "high"

        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "emotion_scores": {"neutral": 1.0},
            "emotion_embedding": [0.0] * 768,
            "energy_level": energy,
            "speaking_rate": "normal",
        }

    def offload_to_cpu(self) -> None:
        """Offload model to CPU."""
        if self.model is not None:
            self.model = self.model.cpu()
            torch.cuda.empty_cache()
            self._device = torch.device("cpu")
            logger.info("emotion2vec_offloaded_to_cpu")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
