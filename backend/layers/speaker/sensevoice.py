"""
IntelliVoice — SenseVoice Analyzer
Analyzes emotion, energy, pace, and speech events from audio using SenseVoice-Small.
"""
import torch
import torchaudio
from typing import Dict, Any, Optional
import numpy as np

from config.logging_config import get_logger
from config.model_registry import ModelRegistry

logger = get_logger("sensevoice")

class SenseVoiceAnalyzer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        self._config = ModelRegistry.SENSEVOICE
        self._device = torch.device("cpu")
        
    async def load(self, device: torch.device) -> None:
        if self._is_loaded:
            return
            
        logger.info("loading_sensevoice", model=self._config.hf_model_id)
        self._device = device
        try:
            from funasr import AutoModel
            
            self.model = AutoModel(
                model=self._config.hf_model_id,
                trust_remote_code=True,
                device=device.type,
                disable_update=True,
            )
            
            self._is_loaded = True
            logger.info("sensevoice_loaded", device=str(device))
        except Exception as e:
            logger.error("sensevoice_load_failed", error=str(e))
            raise

    def _extract_emotion_from_output(self, output: list) -> Dict[str, Any]:
        """Extract emotion, energy, pace from SenseVoice output."""
        result = {
            "emotion": "neutral",
            "energy": "medium",
            "pace": "moderate",
            "speech_events": [],
            "language": "unknown",
        }
        
        if not output or len(output) == 0:
            return result
            
        first_result = output[0]
        
        # SenseVoice returns: text with tags like <|EMOTION:HAPPY|>, <|ENERGY:HIGH|>, etc.
        text = first_result.get("text", "")
        
        import re
        
        # Extract emotion tag
        emotion_match = re.search(r'<\|EMOTION:([^|]+)\|>', text)
        if emotion_match:
            emotion = emotion_match.group(1).lower()
            result["emotion"] = emotion
        
        # Extract energy tag
        energy_match = re.search(r'<\|ENERGY:([^|]+)\|>', text)
        if energy_match:
            energy = energy_match.group(1).lower()
            result["energy"] = energy
        
        # Extract pace/speech rate tag
        pace_match = re.search(r'<\|SPEECH_RATE:([^|]+)\|>', text)
        if pace_match:
            pace = pace_match.group(1).lower()
            result["pace"] = pace
        
        # Extract language
        lang_match = re.search(r'<\|LANGUAGE:([^|]+)\|>', text)
        if lang_match:
            lang = lang_match.group(1).lower()
            result["language"] = lang
        
        # Extract speech events (laughter, crying, etc.)
        events = re.findall(r'<\|EVENT:([^|]+)\|>', text)
        if events:
            result["speech_events"] = events
        
        # Clean the text by removing tags
        clean_text = re.sub(r'<\|[^|]+\|>', '', text).strip()
        result["text"] = clean_text
        
        return result

    async def analyze(self, waveform: torch.Tensor, sr: int) -> Dict[str, Any]:
        """Analyze emotion and metadata from audio."""
        if not self._is_loaded:
            raise RuntimeError("SenseVoice not loaded")
            
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)
        
        # Ensure 16kHz
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        
        # Convert to numpy for funasr
        audio_np = waveform.cpu().numpy()
        
        import asyncio
        
        def _infer():
            return self.model.generate(
                input=audio_np,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
        
        try:
            output = await asyncio.to_thread(_infer)
            return self._extract_emotion_from_output(output)
        except Exception as e:
            logger.error("sensevoice_inference_failed", error=str(e))
            return {
                "emotion": "neutral",
                "energy": "medium",
                "pace": "moderate",
                "speech_events": [],
                "language": "unknown",
                "text": "",
            }
