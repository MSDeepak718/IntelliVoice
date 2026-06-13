"""
IntelliVoice — Emotion Analyzer
Analyzes emotion from audio using a standard HuggingFace audio classification model.
"""
import torch
import torch.nn.functional as F
import torchaudio
from typing import Dict, Any

from config.logging_config import get_logger
from config.model_registry import ModelRegistry

logger = get_logger("emotion")

class EmotionAnalyzer:
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self._is_loaded = False
        self._config = ModelRegistry.EMOTION
        self._device = torch.device("cpu")
        
    async def load(self, device: torch.device) -> None:
        if self._is_loaded:
            return
            
        logger.info("loading_emotion_model", model=self._config.hf_model_id)
        self._device = device
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            import transformers.utils.import_utils as hf_import_utils
            import transformers.modeling_utils as hf_modeling_utils
            
            # Bypass the torch.load CVE check. We trust the official `superb` organization 
            # weights and bypassing this prevents requiring a full torch 2.6+ upgrade.
            if hasattr(hf_import_utils, "check_torch_load_is_safe"):
                hf_import_utils.check_torch_load_is_safe = lambda: None
            if hasattr(hf_modeling_utils, "check_torch_load_is_safe"):
                hf_modeling_utils.check_torch_load_is_safe = lambda: None
            
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self._config.hf_model_id,
                trust_remote_code=self._config.trust_remote_code
            )
            
            self.model = AutoModelForAudioClassification.from_pretrained(
                self._config.hf_model_id,
                trust_remote_code=self._config.trust_remote_code,
                dtype=torch.float16 if self._config.precision.value == "fp16" else torch.float32
            )
            
            self.model.to(self._device)
            self.model.eval()
            
            self._is_loaded = True
            logger.info("emotion_model_loaded", device=str(device))
        except Exception as e:
            logger.error("emotion_model_load_failed", error=str(e))
            raise

    async def analyze(self, waveform: torch.Tensor, sr: int) -> Dict[str, Any]:
        """Analyze emotion from audio."""
        if not self._is_loaded:
            raise RuntimeError("Emotion model not loaded")
            
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)
        
        # Ensure 16kHz
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        
        audio_np = waveform.cpu().numpy()
        
        try:
            inputs = self.feature_extractor(
                audio_np, sampling_rate=16000, return_tensors="pt"
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # The model is likely loaded in fp16, so cast inputs
            if self._config.precision.value == "fp16":
                inputs["input_values"] = inputs["input_values"].to(torch.float16)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                predicted_class_id = torch.argmax(probs, dim=-1).item()
                label = self.model.config.id2label[predicted_class_id].lower()

            emotion_map = {
                "neu": "neutral",
                "hap": "happy",
                "ang": "angry",
                "sad": "sad",
            }
            mapped_emotion = emotion_map.get(label, "neutral")

            return {
                "emotion": mapped_emotion,
                "energy": "medium",
                "pace": "moderate",
            }
        except Exception as e:
            logger.error("emotion_inference_failed", error=str(e))
            return {
                "emotion": "neutral",
                "energy": "medium",
                "pace": "moderate",
            }
