"""
IntelliVoice — Audio Pipeline Orchestrator

The main pipeline that coordinates all layers from audio input
through speech output. Implements strict specific loading order
and concurrent layer execution tailored for the RTX 4080 (16GB):

    1. Preprocessing: Silero VAD + noisereduce (CPU / FP32)
    2A. ASR: Whisper large-v3-turbo (FP16 via faster-whisper)
    2B. Emotion/Speaker: superb/wav2vec2-base-superb-er + ECAPA-TDNN (FP16)
    3. Memory: Conversation Memory + MongoDB (CPU only)
    4. Core Reasoning: Qwen2.5 3B (INT4 NF4 double quant)
    5. TTS Synthesis: CosyVoice 2 (FP16)

All models are loaded at startup. Whisper and Emotion execute concurrently.
LLM output streams inline style tags directly to TTS model sentence-by-sentence.
"""

from __future__ import annotations

import asyncio
import time
import re
from typing import Any, Dict

import torch

from config import get_settings
from config.logging_config import get_logger
from config.model_registry import LoadingOrder

from backend.core.gpu_manager import GPUManager
from backend.core.session_manager import SessionState

from backend.layers.preprocessing.vad import SileroVAD
from backend.layers.preprocessing.noise_suppression import NoiseSuppressor
from backend.layers.preprocessing.audio_utils import (
    bytes_to_waveform,
    waveform_to_bytes,
    normalize_waveform,
)

logger = get_logger("pipeline")


from backend.layers.memory.conversation_memory import ConversationMemory
from backend.layers.memory.long_term_memory import LongTermMemory
from backend.layers.reasoning.fast_reasoning import FastReasoner
from backend.layers.speech_generation.piper_tts import PiperSynthesizer

from backend.layers.asr.whisper_asr import WhisperASR
from backend.layers.speaker.emotion import EmotionAnalyzer
from backend.layers.speaker.ecapa import EcapaSpeaker


class AudioPipeline:
    def __init__(self, gpu_manager: GPUManager):
        self.gpu = gpu_manager
        self.settings = get_settings()

        # 1. Preprocessing (CPU / FP32)
        self.vad = SileroVAD(
            threshold=self.settings.vad_threshold,
            sample_rate=self.settings.sample_rate,
        )
        self.noise_suppressor = NoiseSuppressor(sample_rate=self.settings.sample_rate)

        # 2. ASR & Emotion/Speaker (FP16)
        self.asr = WhisperASR()
        self.emotion_analyzer = EmotionAnalyzer()
        self.speaker_encoder = EcapaSpeaker()

        # 3. Memory (LangGraph + MongoDB)
        self.memory = ConversationMemory()
        self.long_term_memory = LongTermMemory()

        # 4 & 5. Core Reasoning & TTS
        self.reasoner = FastReasoner()
        self.tts = PiperSynthesizer()

        self._is_initialized = False

    async def initialize(self) -> None:
        """
        Load all models at startup. Enforces strict loading order:
        noisereduce -> Whisper -> Emotion + ECAPA -> Qwen2.5 3B -> XTTS-v2
        """
        logger.info("initializing_pipeline")
        device = self.gpu.device

        # Order 1: Preprocessing
        logger.info("loading_preprocessing_models")
        await self.vad.load(device=torch.device("cpu"))
        await self.noise_suppressor.load()
        self.gpu.register_model("vad", self.vad.model, LoadingOrder.PREPROCESSING, 50)
        self.gpu.register_model(
            "noise_suppressor", self.noise_suppressor, LoadingOrder.PREPROCESSING, 0
        ) 

        # Order 2: ASR
        logger.info("loading_asr_model")
        await self.asr.load(device=device)
        self.gpu.register_model("asr", self.asr.model, LoadingOrder.ASR, 1500)

        # Order 3: Emotion & Speaker
        logger.info("loading_emotion_and_speaker_models")
        await self.emotion_analyzer.load(device=device)
        await self.speaker_encoder.load(device=device)
        self.gpu.register_model("emotion_analyzer", self.emotion_analyzer.model, LoadingOrder.EMOTION_SPEAKER, 350)
        self.gpu.register_model("speaker_encoder", self.speaker_encoder.model, LoadingOrder.EMOTION_SPEAKER, 100)

        # Order 4: Core Reasoning (LLM)
        logger.info("loading_core_reasoning_model")
        await self.reasoner.load(device=device)
        self.gpu.register_model("reasoner", self.reasoner.model, LoadingOrder.REASONING, 8500)

        # Order 5: TTS Synthesis
        logger.info("loading_piper_tts")
        await self.tts.load(device=device)
        self.gpu.register_model("tts", self.tts.model, LoadingOrder.TTS, 500)

        # Memory systems (CPU only)
        await self.memory.initialize()
        await self.long_term_memory.connect()

        self._is_initialized = True
        logger.info("pipeline_initialized")
        self.gpu.log_gpu_info()

    async def shutdown(self) -> None:
        """Release everything."""
        logger.info("shutting_down_pipeline")
        self.gpu.shutdown()
        logger.info("pipeline_shutdown_complete")

    def cancel_processing(self) -> None:
        """Abort any ongoing LLM generation or processing tasks."""
        if self._is_initialized and hasattr(self.reasoner, 'cancel_generation'):
            self.reasoner.cancel_generation()

    async def process_audio(
        self,
        audio_bytes: bytes,
        session: SessionState,
        source_sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized.")

        start_time = time.time()
        result: Dict[str, Any] = {
            "transcription": "",
            "response_text": "",
            "response_audio": b"",
            "response_sample_rate": 22050,
            "metadata": {},
        }

        try:
            # ============================================
            # STEP 1: Preprocessing (VAD + noisereduce)
            # ============================================
            waveform, sr = bytes_to_waveform(audio_bytes, source_sample_rate)

            if not self.vad.is_speech(waveform):
                result["metadata"]["skipped"] = "no_speech_detected"
                return result

            speech_waveform, _ = self.vad.extract_speech(waveform)
            if speech_waveform.shape[1] < sr * 0.3:
                result["metadata"]["skipped"] = "speech_too_short"
                return result

            clean_waveform = self.noise_suppressor.suppress_noise(speech_waveform)
            clean_waveform = normalize_waveform(clean_waveform, target_db=self.settings.agc_target_db)

            # ============================================
            # STEP 2: Concurrent ASR and Emotion/Speaker
            # ============================================
            # Execute Whisper and Emotion concurrently using asyncio
            asr_task = asyncio.create_task(self.asr.transcribe(clean_waveform, sr))
            emotion_task = asyncio.create_task(self.emotion_analyzer.analyze(clean_waveform, sr))
            speaker_task = asyncio.create_task(self.speaker_encoder.get_embedding(clean_waveform, sr))

            asr_res, emotion_res, speaker_emb = await asyncio.gather(asr_task, emotion_task, speaker_task)

            transcription = asr_res.get("text", "")
            result["transcription"] = transcription

            # ============================================
            # STEP 3: Enriched Prompt Assembly (Pre-LLM)
            # ============================================
            conversation_history = self.memory.get_conversation_context(session.session_id)

            # ============================================
            # STEP 4: LLM Response
            # ============================================
            llm_result = await asyncio.to_thread(
                self.reasoner.generate_response,
                user_message=transcription,
                conversation_history=conversation_history,
                emotion_context=emotion_res,
                system_prompt="You are a helpful voice assistant. Keep answers very brief.",
                max_new_tokens=150,
            )
            response_text = llm_result.get("response", "I'm sorry, I couldn't process that.")

            # Extract any inline tags (e.g. [TONE: empathetic])
            style_tags = []
            tag_pattern = re.compile(r"\[(.*?)\]")
            tags = tag_pattern.findall(response_text)
            style_tags.extend(tags)
            clean_response_text = tag_pattern.sub("", response_text).strip()

            # Remove <think> blocks generated by reasoning models
            think_pattern = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
            clean_response_text = think_pattern.sub("", clean_response_text).strip()

            result["response_text"] = clean_response_text
            
            # Save turn to memory
            self.memory.add_turn(session.session_id, "user", transcription, emotion=emotion_res.get("emotion"))
            self.memory.add_turn(session.session_id, "assistant", clean_response_text)
            
            # Save to long-term memory (async, don't await)
            asyncio.create_task(
                self.long_term_memory.save_conversation(
                    session_id=session.session_id,
                    user_id=session.user_id,
                    turns=self.memory.get_session(session.session_id).turns if self.memory.get_session(session.session_id) else [],
                    summary=None,
                    metadata={"emotion": emotion_res.get("emotion"), "language": asr_res.get("language")},
                )
            )
            
            # ============================================
            # STEP 5: TTS Synthesis (Zero-Shot Cloning)
            # ============================================
            language = asr_res.get("language", "english")
            emotion = emotion_res.get("emotion", "neutral")
            
            tts_waveform, tts_sr = await asyncio.to_thread(
                self.tts.synthesize,
                text=clean_response_text,
                language=language,
                speaker_embedding=speaker_emb if speaker_emb is not None and speaker_emb.sum() != 0 else None,
                emotion=emotion,
            )

            result["response_audio"] = waveform_to_bytes(tts_waveform, dtype="int16")
            result["response_sample_rate"] = tts_sr
            result["metadata"]["total_ms"] = round((time.time() - start_time) * 1000)
            result["metadata"]["style_tags"] = style_tags
            result["metadata"]["emotion"] = emotion_res.get("emotion", "neutral")
            result["metadata"]["language"] = asr_res.get("language", "english")

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

    async def stream_process_audio(
        self,
        audio_bytes: bytes,
        session: SessionState,
        source_sample_rate: int = 16000,
    ):
        """Streaming version of process_audio that yields dict results one by one."""
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized.")

        start_time = time.time()
        try:
            waveform, sr = bytes_to_waveform(audio_bytes, source_sample_rate)

            if not self.vad.is_speech(waveform):
                yield {"type": "skip", "reason": "no_speech_detected"}
                return

            speech_waveform, _ = self.vad.extract_speech(waveform)
            if speech_waveform.shape[1] < sr * 0.3:
                yield {"type": "skip", "reason": "speech_too_short"}
                return

            clean_waveform = self.noise_suppressor.suppress_noise(speech_waveform)
            clean_waveform = normalize_waveform(clean_waveform, target_db=self.settings.agc_target_db)

            asr_task = asyncio.create_task(self.asr.transcribe(clean_waveform, sr))
            emotion_task = asyncio.create_task(self.emotion_analyzer.analyze(clean_waveform, sr))
            speaker_task = asyncio.create_task(self.speaker_encoder.get_embedding(clean_waveform, sr))

            asr_res, emotion_res, speaker_emb = await asyncio.gather(asr_task, emotion_task, speaker_task)
            transcription = asr_res.get("text", "")
            
            # Send transcription as soon as it's ready
            yield {
                "type": "transcription", 
                "text": transcription,
                "emotion": emotion_res.get("emotion", "neutral"),
                "language": asr_res.get("language", "unknown")
            }

            conversation_history = self.memory.get_conversation_context(session.session_id)

            yield {"type": "response_start"}
            
            full_response = ""
            buffer = ""
            language = asr_res.get("language", "english")
            emotion = emotion_res.get("emotion", "neutral")

            async for token in self.reasoner.stream_generate_response(
                user_message=transcription,
                conversation_history=conversation_history,
                emotion_context=emotion_res,
                system_prompt="You are a helpful voice assistant engaged in a spoken conversation. Keep answers concise, natural, and conversational. NEVER use emojis, markdown formatting, bullet points, asterisks, or any symbols that cannot be spoken out loud. Write numbers as words if they are complex.",
                max_new_tokens=150,
            ):
                full_response += token
                buffer += token

                # Trigger TTS incrementally on sentence boundaries
                if re.search(r'[.!?\n]', buffer):
                    parts = re.split(r'(?<=[.!?\n])(?<!\b[A-Z]\.)(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bProf\.)(?<!\bSt\.)\s+', buffer)
                    if len(parts) > 1:
                        for part in parts[:-1]:
                            part = part.strip()
                            if part:
                                tts_waveform, tts_sr = await asyncio.to_thread(
                                    self.tts.synthesize,
                                    text=part,
                                    language=language,
                                    speaker_embedding=speaker_emb if speaker_emb is not None and speaker_emb.sum() != 0 else None,
                                    emotion=emotion,
                                )
                                yield {"type": "response_chunk", "text": part + " "}
                                yield {
                                    "type": "response_audio",
                                    "audio_bytes": waveform_to_bytes(tts_waveform, dtype="int16"),
                                    "sample_rate": tts_sr,
                                    "metadata": {"emotion": emotion, "language": language}
                                }
                        buffer = parts[-1]

            # Flush remaining buffer for TTS
            buffer = buffer.strip()
            if buffer:
                tts_waveform, tts_sr = await asyncio.to_thread(
                    self.tts.synthesize,
                    text=buffer,
                    language=language,
                    speaker_embedding=speaker_emb if speaker_emb is not None and speaker_emb.sum() != 0 else None,
                    emotion=emotion,
                )
                yield {"type": "response_chunk", "text": buffer}
                yield {
                    "type": "response_audio",
                    "audio_bytes": waveform_to_bytes(tts_waveform, dtype="int16"),
                    "sample_rate": tts_sr,
                    "metadata": {"emotion": emotion, "language": language}
                }

            tag_pattern = re.compile(r"\[(.*?)\]")
            clean_response_text = tag_pattern.sub("", full_response).strip()
            think_pattern = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
            clean_response_text = think_pattern.sub("", clean_response_text).strip()

            self.memory.add_turn(session.session_id, "user", transcription, emotion=emotion_res.get("emotion"))
            self.memory.add_turn(session.session_id, "assistant", clean_response_text)
            
            asyncio.create_task(
                self.long_term_memory.save_conversation(
                    session_id=session.session_id,
                    user_id=session.user_id,
                    turns=self.memory.get_session(session.session_id).turns if self.memory.get_session(session.session_id) else [],
                    summary=None,
                    metadata={"emotion": emotion_res.get("emotion"), "language": asr_res.get("language")},
                )
            )

        except asyncio.CancelledError:
            logger.info("stream_process_audio_cancelled", session=session.session_id)
            raise
        except Exception as e:
            logger.error(
                "pipeline_stream_error",
                session=session.session_id,
                error=str(e),
                elapsed_ms=round((time.time() - start_time) * 1000),
            )
            yield {"type": "error", "message": str(e)}

    async def process_text(self, text: str, session: SessionState, use_rag: bool = True) -> Dict[str, Any]:
        """Process a text message (chat mode)."""
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized.")

        start_time = time.time()
        result: Dict[str, Any] = {
            "response_text": "",
            "metadata": {},
        }

        try:
            conversation_history = self.memory.get_conversation_context(session.session_id)
            
            llm_result = await asyncio.to_thread(
                self.reasoner.generate_response,
                user_message=text,
                conversation_history=conversation_history,
                emotion_context={"emotion": "neutral", "energy": "medium", "pace": "moderate"},
                system_prompt="You are a helpful voice assistant. Keep answers concise and conversational.",
                max_new_tokens=self.settings.max_new_tokens,
            )
            response_text = llm_result.get("response", "I'm sorry, I couldn't process that.")

            # Extract any inline tags
            style_tags = []
            tag_pattern = re.compile(r"\[(.*?)\]")
            tags = tag_pattern.findall(response_text)
            style_tags.extend(tags)
            clean_response_text = tag_pattern.sub("", response_text).strip()

            result["response_text"] = clean_response_text
            
            # Save turn to memory
            self.memory.add_turn(session.session_id, "user", text, emotion="neutral")
            self.memory.add_turn(session.session_id, "assistant", clean_response_text)

            result["metadata"]["total_ms"] = round((time.time() - start_time) * 1000)
            result["metadata"]["style_tags"] = style_tags

            return result

        except Exception as e:
            logger.error(
                "pipeline_text_error",
                session=session.session_id,
                error=str(e),
                elapsed_ms=round((time.time() - start_time) * 1000),
            )
            result["metadata"]["error"] = str(e)
            return result
