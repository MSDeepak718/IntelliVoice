"""
IntelliVoice — ECAPA-TDNN Speaker Encoder
Generates speaker embeddings for voice cloning.
"""
import torch
import torchaudio

from config.logging_config import get_logger
from config.model_registry import ModelRegistry

logger = get_logger("ecapa_speaker")

class EcapaSpeaker:
    def __init__(self):
        self.model = None
        self._is_loaded = False
        self._config = ModelRegistry.ECAPA_TDNN
        
    async def load(self, device: torch.device) -> None:
        if self._is_loaded:
            return
            
        logger.info("loading_ecapa", model=self._config.hf_model_id)
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            
            # Using SpeechBrain's inference class
            run_opts = {"device": "cuda" if device.type == "cuda" else "cpu"}
            self.model = EncoderClassifier.from_hparams(
                source=self._config.hf_model_id,
                run_opts=run_opts,
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            self._is_loaded = True
            logger.info("ecapa_loaded", device=str(device))
        except Exception as e:
            logger.error("ecapa_load_failed", error=str(e))
            raise

    async def get_embedding(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract speaker embedding from audio."""
        if not self._is_loaded:
            raise RuntimeError("EcapaSpeaker not loaded")
            
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # ECAPA expects 16kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(waveform.device)
            waveform = resampler(waveform)
            
        import asyncio
        def _encode():
            with torch.no_grad():
                return self.model.encode_batch(waveform)
                
        embeddings = await asyncio.to_thread(_encode)
        # Flatten to 1D tensor for this speaker
        return embeddings.squeeze()
