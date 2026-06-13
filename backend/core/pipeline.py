"""
IntelliVoice — Audio Pipeline Orchestrator

The main pipeline that coordinates all layers from audio input
through speech output. Optimized for minimum latency on RTX 4080 (16GB):

    1. Preprocessing: Silero VAD (CPU / FP32)
    2. ASR: Whisper large-v3-turbo (FP16 via faster-whisper)
    3. Memory: Session-scoped Conversation Memory (CPU only)
    4. Core Reasoning: Qwen2.5-7B-Instruct (INT4 NF4 double quant)
    5. TTS Synthesis: OmniVoice (FP16)

All models are loaded at startup. LLM output streams to TTS on
clause/sentence boundaries for minimum first-audio latency.
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
from backend.layers.preprocessing.audio_utils import (
    bytes_to_waveform,
    waveform_to_bytes,
    normalize_waveform,
)

logger = get_logger("pipeline")


from backend.layers.memory.conversation_memory import ConversationMemory
from backend.layers.reasoning.fast_reasoning import FastReasoner
from backend.layers.speech_generation.omnivoice_tts import OmniVoiceSynthesizer

from backend.layers.asr.whisper_asr import WhisperASR

# Regex patterns compiled once at module level for performance
_TAG_PATTERN = re.compile(r"\[.*?\]")
_THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# Clause/sentence boundary: split on .!?\n and also on , ; : — when followed by space
# Excludes common abbreviations
_SENTENCE_SPLIT = re.compile(
    r'(?<=[.!?\n])(?<!\b[A-Z]\.)(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bProf\.)(?<!\bSt\.)\s+'
)
_CLAUSE_SPLIT = re.compile(
    r'(?<=[,;:\u2014])(?<!\b[A-Z]\.)\s+'
)


class AudioPipeline:
    def __init__(self, gpu_manager: GPUManager):
        self.gpu = gpu_manager
        self.settings = get_settings()

        # 1. Preprocessing (CPU / FP32)
        self.vad = SileroVAD(
            threshold=self.settings.vad_threshold,
            sample_rate=self.settings.sample_rate,
        )

        # 2. ASR (FP16)
        self.asr = WhisperASR()

        # 3. Memory (session-scoped, CPU only)
        self.memory = ConversationMemory()

        # 4 & 5. Core Reasoning & TTS
        self.reasoner = FastReasoner()
        self.tts = OmniVoiceSynthesizer()

        self._is_initialized = False

    async def initialize(self) -> None:
        """
        Load all models at startup. Enforces strict loading order:
        VAD -> Whisper -> Qwen2.5-7B -> OmniVoice
        """
        logger.info("initializing_pipeline")
        device = self.gpu.device

        # Order 1: Preprocessing
        logger.info("loading_preprocessing_models")
        await self.vad.load(device=torch.device("cpu"))
        self.gpu.register_model("vad", self.vad.model, LoadingOrder.PREPROCESSING, 50)

        # Order 2: ASR
        logger.info("loading_asr_model")
        await self.asr.load(device=device)
        self.gpu.register_model("asr", self.asr.model, LoadingOrder.ASR, 1500)

        # Order 3: Core Reasoning (LLM)
        logger.info("loading_core_reasoning_model")
        await self.reasoner.load(device=device)
        self.gpu.register_model("reasoner", self.reasoner.model, LoadingOrder.REASONING, 5000)

        # Order 4: TTS Synthesis
        logger.info("loading_omnivoice")
        await self.tts.load(device=device)
        self.gpu.register_model("tts", self.tts.model, LoadingOrder.TTS, 4000)

        # Memory (CPU only)
        await self.memory.initialize()

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

    @staticmethod
    def _clean_llm_text(text: str) -> str:
        """Strip inline tags and <think> blocks from LLM output."""
        text = _TAG_PATTERN.sub("", text)
        text = _THINK_PATTERN.sub("", text)
        text = text.replace("<think>", "").replace("</think>", "")
        return text.strip()

    async def stream_process_audio(
        self,
        audio_bytes: bytes,
        session: SessionState,
        source_sample_rate: int = 16000,
    ):
        """
        Streaming pipeline: Audio → ASR → LLM (streaming) → TTS (incremental).

        Yields dict chunks:
            - type: "skip" — no speech / too short
            - type: "transcription" — ASR result
            - type: "response_start" — LLM started generating
            - type: "response_chunk" — text fragment
            - type: "response_audio" — synthesized audio for a fragment
            - type: "error" — something went wrong
        """
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized.")

        start_time = time.time()
        try:
            # ============================================
            # STEP 1: Preprocessing (VAD only — no noise suppression for speed)
            # ============================================
            waveform, sr = bytes_to_waveform(audio_bytes, source_sample_rate)

            if not self.vad.is_speech(waveform):
                yield {"type": "skip", "reason": "no_speech_detected"}
                return

            speech_waveform, _ = self.vad.extract_speech(waveform)
            if speech_waveform.shape[1] < sr * 0.3:
                yield {"type": "skip", "reason": "speech_too_short"}
                return

            clean_waveform = normalize_waveform(speech_waveform, target_db=self.settings.agc_target_db)

            # ============================================
            # STEP 2: ASR (no concurrent emotion/speaker — removed for speed)
            # ============================================
            asr_res = await self.asr.transcribe(clean_waveform, sr)
            transcription = asr_res.get("text", "")

            # Send transcription immediately
            yield {
                "type": "transcription",
                "text": transcription,
                "language": asr_res.get("language", "unknown"),
            }

            # ============================================
            # STEP 3: LLM Streaming → TTS on clause boundaries
            # ============================================
            conversation_history = self.memory.get_conversation_context(session.session_id)

            yield {"type": "response_start"}

            full_response = ""
            buffer = ""
            language = asr_res.get("language", "english")
            word_count = 0  # Track words in buffer for fallback flush

            async for token in self.reasoner.stream_generate_response(
                user_message=transcription,
                conversation_history=conversation_history,
                system_prompt="You are Dhurva, a helpful AI voice assistant engaged in a spoken conversation. Keep answers concise, natural, and conversational. NEVER use emojis, markdown formatting, bullet points, asterisks, or any symbols that cannot be spoken out loud. Write numbers as words if they are complex.",
                max_new_tokens=150,
            ):
                full_response += token
                buffer += token
                word_count += len(token.split())

                # Strategy: flush on sentence OR clause boundaries, or word-count fallback
                should_flush = False
                flush_parts = []

                # Priority 1: sentence boundary split
                if re.search(r'[.!?\n]', buffer):
                    parts = _SENTENCE_SPLIT.split(buffer)
                    if len(parts) > 1:
                        flush_parts = parts[:-1]
                        buffer = parts[-1]
                        should_flush = True

                # Priority 2: clause boundary (comma, semicolon, colon, em-dash) after 8+ words
                if not should_flush and word_count >= 8 and re.search(r'[,;:\u2014]', buffer):
                    parts = _CLAUSE_SPLIT.split(buffer)
                    if len(parts) > 1:
                        flush_parts = parts[:-1]
                        buffer = parts[-1]
                        should_flush = True

                # Priority 3: word-count fallback (flush after 15 words regardless)
                if not should_flush and word_count >= 15:
                    # Find the last space and split there
                    last_space = buffer.rfind(' ')
                    if last_space > 0:
                        flush_parts = [buffer[:last_space]]
                        buffer = buffer[last_space + 1:]
                        should_flush = True

                if should_flush:
                    for part in flush_parts:
                        part = self._clean_llm_text(part).strip()
                        if part:
                            tts_waveform, tts_sr = await asyncio.to_thread(
                                self.tts.synthesize,
                                text=part,
                                language=language,
                            )
                            yield {"type": "response_chunk", "text": part + " "}
                            yield {
                                "type": "response_audio",
                                "audio_bytes": waveform_to_bytes(tts_waveform, dtype="int16"),
                                "sample_rate": tts_sr,
                                "metadata": {"language": language},
                            }
                    word_count = len(buffer.split())

            # Flush remaining buffer
            buffer = self._clean_llm_text(buffer).strip()
            if buffer:
                tts_waveform, tts_sr = await asyncio.to_thread(
                    self.tts.synthesize,
                    text=buffer,
                    language=language,
                )
                yield {"type": "response_chunk", "text": buffer}
                yield {
                    "type": "response_audio",
                    "audio_bytes": waveform_to_bytes(tts_waveform, dtype="int16"),
                    "sample_rate": tts_sr,
                    "metadata": {"language": language},
                }

            # Clean full response and save to session memory
            clean_response_text = self._clean_llm_text(full_response)
            self.memory.add_turn(session.session_id, "user", transcription)
            self.memory.add_turn(session.session_id, "assistant", clean_response_text)

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

    async def process_text(self, text: str, session: SessionState) -> Dict[str, Any]:
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
                system_prompt="You are Dhurva, a helpful AI voice assistant. Keep answers concise and conversational.",
                max_new_tokens=self.settings.max_new_tokens,
            )
            response_text = llm_result.get("response", "I'm sorry, I couldn't process that.")
            clean_response_text = self._clean_llm_text(response_text)

            result["response_text"] = clean_response_text

            # Save turn to session memory
            self.memory.add_turn(session.session_id, "user", text)
            self.memory.add_turn(session.session_id, "assistant", clean_response_text)

            result["metadata"]["total_ms"] = round((time.time() - start_time) * 1000)
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
