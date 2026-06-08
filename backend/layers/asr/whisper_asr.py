"""
IntelliVoice — Whisper ASR
Layer 2A: Transcribes audio using faster-whisper.
"""

from typing import Dict, Any, Optional
import torch
import numpy as np

from config.logging_config import get_logger
from config.model_registry import ModelRegistry

logger = get_logger("whisper_asr")


class WhisperASR:
    def __init__(self):
        self.model = None
        self._is_loaded = False
        self._config = ModelRegistry.WHISPER
        self._device = "cpu"

    async def load(self, device: torch.device) -> None:
        if self._is_loaded:
            return
            
        logger.info("loading_whisper_asr", model=self._config.hf_model_id)
        
        try:
            from faster_whisper import WhisperModel
            
            # Convert torch device to faster-whisper string
            device_str = "cuda" if device.type == "cuda" else "cpu"
            compute_type = "float16" if device_str == "cuda" else "float32"
            
            self.model = WhisperModel(
                self._config.hf_model_id,
                device=device_str,
                compute_type=compute_type,
            )
            self._device = device_str
            self._is_loaded = True
            logger.info("whisper_asr_loaded", device=device_str)
        except Exception as e:
            logger.error("whisper_asr_load_failed", error=str(e))
            raise

    async def transcribe(self, waveform: torch.Tensor, sr: int) -> Dict[str, Any]:
        """Transcribe audio waveform."""
        if not self._is_loaded:
            raise RuntimeError("WhisperASR not loaded")

        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)
            
        # Convert to numpy for faster-whisper
        audio_np = waveform.cpu().numpy()
        
        import asyncio
        segments, info = await asyncio.to_thread(
            self.model.transcribe,
            audio_np,
            word_timestamps=True,
            vad_filter=False,
        )
        
        segments = list(segments)
        
        text = " ".join([seg.text for seg in segments]).strip()
        
        timestamps = []
        for seg in segments:
            for word in seg.words:
                timestamps.append({
                    "word": word.word,
                    "start": word.start,
                    "end": word.end
                })
                
        return {
            "text": text,
            "language": info.language,
            "timestamps": timestamps,
        }
