"""
IntelliVoice — Audio Pipeline Orchestrator

The main pipeline that coordinates all layers from audio input
through speech output. Implements three-phase VRAM management.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import torch

from config import get_settings
from config.logging_config import get_logger
from config.model_registry import LoadingPhase

from backend.core.gpu_manager import GPUManager
from backend.core.session_manager import SessionState

# Layer imports
from backend.layers.preprocessing.vad import SileroVAD
from backend.layers.preprocessing.noise_suppression import NoiseSuppressor
from backend.layers.preprocessing.audio_utils import (
    bytes_to_waveform,
    waveform_to_bytes,
    waveform_to_wav_bytes,
    normalize_waveform,
)
from backend.layers.acoustic_encoder.xlsr_encoder import XLSREncoder
from backend.layers.semantic.qwen_audio import QwenAudioUnderstanding
from backend.layers.prosody.emotion2vec import Emotion2VecAnalyzer
from backend.layers.speaker.wavlm_speaker import WavLMSpeakerEncoder
from backend.layers.reasoning.qwen3_reasoning import Qwen3Reasoner
from backend.layers.memory.conversation_memory import ConversationMemory
from backend.layers.memory.long_term_memory import LongTermMemory
from backend.layers.response_planning.planner import ResponsePlanner
from backend.layers.speech_generation.cosyvoice import CosyVoiceSynthesizer
from backend.layers.synthesis.hifi_gan import HiFiGANVocoder

logger = get_logger("pipeline")


class AudioPipeline:
    """
    Main audio processing pipeline orchestrator.

    Coordinates all layers in the speech-to-speech pipeline:
        Audio → Preprocess → Encode → Understand → Reason → Plan → Synthesize → Audio

    Implements three-phase VRAM management:
        Phase 1 (Understanding): VAD, DeepFilter, XLS-R, Qwen-Audio, Emotion2Vec, WavLM
        Phase 2 (Reasoning): Qwen3 MoE
        Phase 3 (Generation): CosyVoice 2, HiFi-GAN
    """

    def __init__(self, gpu_manager: GPUManager):
        self.gpu = gpu_manager
        self.settings = get_settings()

        # Layer 1: Preprocessing (always loaded)
        self.vad = SileroVAD(
            threshold=self.settings.vad_threshold,
            sample_rate=self.settings.sample_rate,
        )
        self.noise_suppressor = NoiseSuppressor(sample_rate=self.settings.sample_rate)

        # Layer 2-5: Understanding (Phase 1)
        self.xlsr_encoder = XLSREncoder()
        self.qwen_audio = QwenAudioUnderstanding()
        self.emotion_analyzer = Emotion2VecAnalyzer()
        self.speaker_encoder = WavLMSpeakerEncoder()

        # Layer 6: Reasoning (Phase 2)
        self.reasoner = Qwen3Reasoner()

        # Layer 7: Memory
        self.conversation_memory = ConversationMemory()
        self.long_term_memory = LongTermMemory()

        # Layer 9: Response Planning
        self.response_planner = ResponsePlanner()

        # Layer 10 + 12: Speech Generation (Phase 3)
        self.tts = CosyVoiceSynthesizer()
        self.vocoder = HiFiGANVocoder()

        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize the pipeline — load always-on models."""
        logger.info("initializing_pipeline")

        # Load Phase 0 (always-on) models
        device = self.gpu.device
        await self.vad.load(device=device)
        await self.noise_suppressor.load()

        # Initialize memory
        await self.conversation_memory.initialize()
        await self.long_term_memory.connect()

        self._is_initialized = True
        logger.info("pipeline_initialized", device=str(device))
        self.gpu.log_gpu_info()

    async def _load_understanding_models(self) -> None:
        """Load Phase 1 (understanding) models to GPU."""
        logger.info("loading_understanding_phase")
        device = self.gpu.device

        if not self.xlsr_encoder.is_loaded:
            await self.xlsr_encoder.load(device=device)
        if not self.qwen_audio.is_loaded:
            await self.qwen_audio.load(device=device)
        if not self.emotion_analyzer.is_loaded:
            await self.emotion_analyzer.load(device=device)
        if not self.speaker_encoder.is_loaded:
            await self.speaker_encoder.load(device=device)

        self.gpu.log_gpu_info()

    async def _load_reasoning_model(self) -> None:
        """Load Phase 2 (reasoning) model, offloading Phase 1."""
        logger.info("loading_reasoning_phase")

        # Offload understanding models
        self.xlsr_encoder.offload_to_cpu()
        self.qwen_audio.offload_to_cpu()
        self.emotion_analyzer.offload_to_cpu()
        self.speaker_encoder.offload_to_cpu()
        self.gpu._cleanup_gpu()

        # Load reasoning model
        if not self.reasoner.is_loaded:
            await self.reasoner.load(device=self.gpu.device)

        self.gpu.log_gpu_info()

    async def _load_generation_models(self) -> None:
        """Load Phase 3 (generation) models, offloading Phase 2."""
        logger.info("loading_generation_phase")

        # Offload reasoning model
        self.reasoner.offload_to_cpu()
        self.gpu._cleanup_gpu()

        # Load TTS models
        if not self.tts.is_loaded:
            await self.tts.load(device=self.gpu.device)
        if not self.vocoder.is_loaded:
            await self.vocoder.load(device=self.gpu.device)

        self.gpu.log_gpu_info()

    async def process_audio(
        self,
        audio_bytes: bytes,
        session: SessionState,
        source_sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """
        Process audio through the full pipeline.

        This is the main entry point for audio processing.

        Args:
            audio_bytes: Raw PCM audio bytes (int16, mono).
            session: WebSocket session state.
            source_sample_rate: Sample rate of input audio.

        Returns:
            Dict with:
                - 'transcription': What the user said
                - 'emotion': Detected emotion
                - 'response_text': Generated response
                - 'response_audio': Synthesized speech bytes
                - 'response_sample_rate': Output sample rate
                - 'metadata': Processing metadata
        """
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized.")

        start_time = time.time()
        result = {
            "transcription": "",
            "emotion": "neutral",
            "response_text": "",
            "response_audio": b"",
            "response_sample_rate": 22050,
            "metadata": {},
        }

        try:
            # ============================================
            # STEP 1: Audio Preprocessing
            # ============================================
            waveform, sr = bytes_to_waveform(audio_bytes, source_sample_rate)

            # Voice Activity Detection
            if not self.vad.is_speech(waveform):
                result["metadata"]["skipped"] = "no_speech_detected"
                return result

            # Extract speech segments
            speech_waveform, segments = self.vad.extract_speech(waveform)
            if speech_waveform.shape[1] < sr * 0.3:  # Less than 300ms
                result["metadata"]["skipped"] = "speech_too_short"
                return result

            # Noise suppression
            clean_waveform = self.noise_suppressor.suppress_noise(speech_waveform)
            clean_waveform = normalize_waveform(clean_waveform)

            preprocess_time = time.time() - start_time

            # ============================================
            # STEP 2: Understanding (Phase 1 models)
            # ============================================
            step2_start = time.time()
            await self._load_understanding_models()

            # Acoustic encoding
            acoustic_result = self.xlsr_encoder.encode(clean_waveform, sr)

            # Semantic understanding
            semantic_result = self.qwen_audio.analyze_intent(clean_waveform, sr)

            # Emotion analysis
            emotion_result = self.emotion_analyzer.analyze_emotion(clean_waveform, sr)

            # Speaker embedding
            speaker_emb = self.speaker_encoder.extract_speaker_embedding(clean_waveform, sr)

            understanding_time = time.time() - step2_start

            # Extract results
            transcription = semantic_result.get("transcription", "")
            detected_emotion = emotion_result.get("emotion", "neutral")
            detected_language = semantic_result.get("language", "english")
            detected_intent = semantic_result.get("intent", "unknown")

            result["transcription"] = transcription
            result["emotion"] = detected_emotion

            # ============================================
            # STEP 3: Memory Update
            # ============================================
            self.conversation_memory.add_turn(
                session_id=session.session_id,
                role="user",
                content=transcription,
                emotion=detected_emotion,
                language=detected_language,
            )
            session.add_to_history("user", transcription, {
                "emotion": detected_emotion,
                "language": detected_language,
            })

            # ============================================
            # STEP 4: Reasoning (Phase 2 model)
            # ============================================
            step4_start = time.time()
            await self._load_reasoning_model()

            conversation_context = self.conversation_memory.get_conversation_context(
                session.session_id
            )

            llm_result = self.reasoner.generate_response(
                user_message=transcription,
                conversation_history=conversation_context,
                emotion_context=emotion_result,
            )

            response_text = llm_result.get("response", "I'm sorry, I didn't understand that.")
            result["response_text"] = response_text

            reasoning_time = time.time() - step4_start

            # ============================================
            # STEP 5: Response Planning
            # ============================================
            response_plan = self.response_planner.plan(
                response_text=response_text,
                user_emotion=detected_emotion,
                user_intent=detected_intent,
                detected_language=detected_language,
                conversation_context=conversation_context,
            )

            # ============================================
            # STEP 6: Speech Generation (Phase 3 models)
            # ============================================
            step6_start = time.time()
            await self._load_generation_models()

            tts_waveform, tts_sr = self.tts.synthesize(
                text=response_text,
                language=detected_language,
                emotion=response_plan.emotion,
                speaking_rate=response_plan.speaking_rate,
            )

            # Enhance with vocoder post-processing
            if self.vocoder.is_loaded:
                tts_waveform = self.vocoder.enhance_waveform(tts_waveform, tts_sr)

            generation_time = time.time() - step6_start

            # Convert to bytes
            result["response_audio"] = waveform_to_bytes(tts_waveform, dtype="int16")
            result["response_sample_rate"] = tts_sr

            # ============================================
            # STEP 7: Memory Update (assistant turn)
            # ============================================
            self.conversation_memory.add_turn(
                session_id=session.session_id,
                role="assistant",
                content=response_text,
            )
            session.add_to_history("assistant", response_text)

            # Persist to long-term memory
            session_mem = self.conversation_memory.get_session(session.session_id)
            if session_mem:
                await self.long_term_memory.save_conversation(
                    session_id=session.session_id,
                    user_id=session.user_id,
                    turns=[t.model_dump() for t in session_mem.turns],
                )

            # ============================================
            # Timing metadata
            # ============================================
            total_time = time.time() - start_time
            result["metadata"] = {
                "preprocess_ms": round(preprocess_time * 1000),
                "understanding_ms": round(understanding_time * 1000),
                "reasoning_ms": round(reasoning_time * 1000),
                "generation_ms": round(generation_time * 1000),
                "total_ms": round(total_time * 1000),
                "language": detected_language,
                "intent": detected_intent,
                "emotion": detected_emotion,
                "response_plan": response_plan.to_dict(),
            }

            logger.info(
                "pipeline_complete",
                session=session.session_id,
                total_ms=result["metadata"]["total_ms"],
                emotion=detected_emotion,
                language=detected_language,
            )

            return result

        except Exception as e:
            logger.error(
                "pipeline_error",
                session=session.session_id,
                error=str(e),
                elapsed_ms=round((time.time() - start_time) * 1000),
            )
            result["metadata"]["error"] = str(e)
            return result

    async def process_text(
        self,
        text: str,
        session: SessionState,
    ) -> Dict[str, Any]:
        """
        Process text input (for chat mode, bypassing audio layers).

        Args:
            text: User's text message.
            session: Session state.

        Returns:
            Dict with response_text and optionally response_audio.
        """
        start_time = time.time()

        # Memory update
        self.conversation_memory.add_turn(
            session_id=session.session_id,
            role="user",
            content=text,
        )

        # Load reasoning
        await self._load_reasoning_model()

        conversation_context = self.conversation_memory.get_conversation_context(
            session.session_id
        )

        llm_result = self.reasoner.generate_response(
            user_message=text,
            conversation_history=conversation_context,
        )

        response_text = llm_result.get("response", "I'm sorry, could you rephrase that?")

        # Memory update
        self.conversation_memory.add_turn(
            session_id=session.session_id,
            role="assistant",
            content=response_text,
        )

        return {
            "response_text": response_text,
            "metadata": {
                "total_ms": round((time.time() - start_time) * 1000),
                "tokens_generated": llm_result.get("tokens_generated", 0),
            },
        }

    async def shutdown(self) -> None:
        """Clean shutdown of all models."""
        logger.info("shutting_down_pipeline")
        await self.long_term_memory.disconnect()
        self.gpu._cleanup_gpu()
        logger.info("pipeline_shutdown_complete")
