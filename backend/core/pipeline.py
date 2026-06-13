"""
IntelliVoice — Audio Pipeline Orchestrator

The main pipeline that coordinates all layers from audio input
through speech output. Optimized for minimum latency on RTX 4080 (16GB):

    1. Preprocessing: Silero VAD (CPU) + DeepFilterNet (CPU)
    2. ASR: Whisper large-v3-turbo (FP16 via faster-whisper)
    3. Emotion: wav2vec2-base-superb-er (FP16) — runs concurrently with ASR
    4. Memory: Session-scoped Conversation Memory (CPU only)
    5. Core Reasoning: Qwen2.5-7B-Instruct (INT4 NF4 double quant)
    6. TTS Synthesis: OmniVoice (FP16)

Streaming Strategy — Overlapped Sentence TTS:
    LLM tokens accumulate into complete sentences. Once a sentence is ready,
    TTS starts immediately in a background thread. While TTS synthesizes
    sentence N, the LLM continues generating sentence N+1. Audio is yielded
    in order. This overlaps LLM and TTS latency for all sentences after the first.
"""

from __future__ import annotations

import asyncio
import time
import re
from typing import Any, Dict, Optional, Tuple

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
from backend.layers.reasoning.fast_reasoning import FastReasoner
from backend.layers.speech_generation.omnivoice_tts import OmniVoiceSynthesizer
from backend.layers.asr.whisper_asr import WhisperASR
from backend.layers.speaker.emotion import EmotionAnalyzer

# Regex patterns compiled once at module level for performance
_TAG_PATTERN = re.compile(r"\[.*?\]")
_THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# Split on sentence boundaries and clauses (.!?,:;\n) to get the first audio chunk faster
_SENTENCE_SPLIT = re.compile(
    r'(?<=[.!?,\n:;])(?<!\b[A-Z]\.)(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bProf\.)(?<!\bSt\.)\s+'
)


class AudioPipeline:
    def __init__(self, gpu_manager: GPUManager):
        self.gpu = gpu_manager
        self.settings = get_settings()

        # 1. Preprocessing (CPU)
        self.vad = SileroVAD(
            threshold=self.settings.vad_threshold,
            sample_rate=self.settings.sample_rate,
        )
        self.noise_suppressor = NoiseSuppressor(sample_rate=self.settings.sample_rate)

        # 2. ASR (FP16)
        self.asr = WhisperASR()

        # 3. Emotion (FP16) — runs concurrently with ASR
        self.emotion_analyzer = EmotionAnalyzer()

        # 4. Memory (session-scoped, CPU only)
        self.memory = ConversationMemory()

        # 5 & 6. Core Reasoning & TTS
        self.reasoner = FastReasoner()
        self.tts = OmniVoiceSynthesizer()

        self._is_initialized = False

    async def initialize(self) -> None:
        """
        Load all models at startup. Enforces strict loading order:
        VAD + DeepFilterNet -> Whisper -> Emotion -> Qwen2.5-7B -> OmniVoice
        """
        logger.info("initializing_pipeline")
        device = self.gpu.device

        # Order 1: Preprocessing (CPU)
        logger.info("loading_preprocessing_models")
        await self.vad.load(device=torch.device("cpu"))
        self.gpu.register_model("vad", self.vad.model, LoadingOrder.PREPROCESSING, 50)

        await self.noise_suppressor.load()
        logger.info(
            "noise_suppressor_status",
            backend=self.noise_suppressor.backend,
            loaded=self.noise_suppressor.is_loaded,
        )

        # Order 2: ASR
        logger.info("loading_asr_model")
        await self.asr.load(device=device)
        self.gpu.register_model("asr", self.asr.model, LoadingOrder.ASR, 1500)

        # Order 3: Emotion
        logger.info("loading_emotion_model")
        await self.emotion_analyzer.load(device=device)
        self.gpu.register_model("emotion", self.emotion_analyzer.model, LoadingOrder.EMOTION, 350)

        # Order 4: Core Reasoning (LLM)
        logger.info("loading_core_reasoning_model")
        await self.reasoner.load(device=device)
        self.gpu.register_model("reasoner", self.reasoner.model, LoadingOrder.REASONING, 5000)

        # Order 5: TTS Synthesis
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

    def _synthesize_sentence(self, text: str, language: str) -> Tuple[torch.Tensor, int]:
        """Synthesize a single sentence via OmniVoice. Runs in a thread."""
        return self.tts.synthesize(text=text, language=language)

    async def stream_process_audio(
        self,
        audio_bytes: bytes,
        session: SessionState,
        source_sample_rate: int = 16000,
    ):
        """
        Streaming pipeline: Audio → Denoise → ASR+Emotion → LLM (streaming) → TTS (overlapped).

        TTS Strategy — Overlapped Sentence Synthesis:
            Instead of splitting on clauses/words (which drops words), we accumulate
            complete sentences from the LLM stream. For each sentence:
              1. Start TTS immediately in a background thread
              2. Continue accumulating the next sentence from the LLM
              3. When the next sentence is ready, await the previous TTS result
                 and yield its audio, then start TTS for the new sentence
            This means TTS for sentence N overlaps with LLM generation of sentence N+1,
            eliminating TTS wait time for all sentences after the first.

        Yields dict chunks:
            - type: "skip" — no speech / too short
            - type: "transcription" — ASR result
            - type: "response_start" — LLM started generating
            - type: "response_chunk" — text fragment
            - type: "response_audio" — synthesized audio for a sentence
            - type: "error" — something went wrong
        """
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized.")

        start_time = time.time()
        try:
            # ============================================
            # STEP 1: Preprocessing — VAD + DeepFilterNet
            # ============================================
            waveform, sr = bytes_to_waveform(audio_bytes, source_sample_rate)

            if not self.vad.is_speech(waveform):
                yield {"type": "skip", "reason": "no_speech_detected"}
                return

            speech_waveform, _ = self.vad.extract_speech(waveform)
            if speech_waveform.shape[1] < sr * 0.3:
                yield {"type": "skip", "reason": "speech_too_short"}
                return

            # DeepFilterNet noise suppression (CPU, non-blocking)
            clean_waveform = await asyncio.to_thread(
                self.noise_suppressor.suppress_noise, speech_waveform
            )
            clean_waveform = normalize_waveform(clean_waveform, target_db=self.settings.agc_target_db)

            # ============================================
            # STEP 2: ASR + Emotion (concurrent)
            # ============================================
            asr_task = asyncio.create_task(self.asr.transcribe(clean_waveform, sr))
            emotion_task = asyncio.create_task(self.emotion_analyzer.analyze(clean_waveform, sr))

            # Wait for ASR to finish first
            await asr_task
            asr_res = asr_task.result()

            # Wait for Emotion up to 50ms, otherwise default to neutral to avoid blocking LLM
            try:
                emotion_res = await asyncio.wait_for(asyncio.shield(emotion_task), timeout=0.05)
            except asyncio.TimeoutError:
                emotion_res = {"emotion": "neutral"}

            transcription = asr_res.get("text", "")
            detected_emotion = emotion_res.get("emotion", "neutral")

            # Send transcription immediately
            yield {
                "type": "transcription",
                "text": transcription,
                "language": asr_res.get("language", "unknown"),
                "emotion": detected_emotion,
            }

            # ============================================
            # STEP 3: LLM Streaming → Overlapped Sentence TTS
            # ============================================
            conversation_history = self.memory.get_conversation_context(session.session_id)
            language = asr_res.get("language", "english")

            # Build system prompt with emotion context
            emotion_hint = f" The user sounds {detected_emotion}." if detected_emotion != "neutral" else ""
            system_prompt = (
                "You are Dhurva, a helpful AI voice assistant engaged in a spoken conversation. "
                "Keep answers concise, natural, and conversational. "
                "NEVER use emojis, markdown formatting, bullet points, asterisks, or any symbols "
                "that cannot be spoken out loud. Write numbers as words if they are complex."
                f"{emotion_hint}"
            )

            yield {"type": "response_start"}

            full_response = ""

            output_queue = asyncio.Queue()
            tts_queue = asyncio.Queue(maxsize=10)
            
            async def tts_worker():
                """Background worker that synthesizes audio without blocking LLM generation."""
                while True:
                    sentence = await tts_queue.get()
                    if sentence is None:
                        tts_queue.task_done()
                        break
                        
                    try:
                        tts_waveform, tts_sr = await asyncio.to_thread(
                            self._synthesize_sentence, sentence, language
                        )
                        await output_queue.put({
                            "type": "response_audio",
                            "audio_bytes": waveform_to_bytes(tts_waveform, dtype="int16"),
                            "sample_rate": tts_sr,
                            "metadata": {"language": language},
                        })
                    except Exception as e:
                        logger.error("tts_synthesis_error", error=str(e), text=sentence)
                        
                    tts_queue.task_done()

            async def llm_worker():
                """Generates tokens, yields them instantly, and chunks for TTS."""
                resp = ""
                buffer = ""
                try:
                    async for token in self.reasoner.stream_generate_response(
                        user_message=transcription,
                        conversation_history=conversation_history,
                        system_prompt=system_prompt,
                        max_new_tokens=150,
                    ):
                        resp += token
                        buffer += token
                        
                        # Yield token directly to frontend for instant UI updates
                        await output_queue.put({"type": "response_chunk", "text": token})
                        
                        # Check for sentence/clause boundaries
                        if not re.search(r'[.!?,\n:;]', buffer):
                            continue
                            
                        parts = _SENTENCE_SPLIT.split(buffer)
                        if len(parts) <= 1:
                            continue
                            
                        complete_sentences = parts[:-1]
                        buffer = parts[-1]
                        
                        for sentence in complete_sentences:
                            sentence = self._clean_llm_text(sentence).strip()
                            if sentence:
                                await tts_queue.put(sentence)
                                
                    # Flush remaining buffer
                    buffer = self._clean_llm_text(buffer).strip()
                    if buffer:
                        await tts_queue.put(buffer)
                        
                    # Signal TTS worker to stop
                    await tts_queue.put(None)
                    return resp
                except Exception as e:
                    logger.error("llm_worker_error", error=str(e))
                    await tts_queue.put(None)
                    return resp

            # Start the fully decoupled background workers
            worker_task_tts = asyncio.create_task(tts_worker())
            worker_task_llm = asyncio.create_task(llm_worker())
            
            async def orchestrator():
                final_text = await worker_task_llm
                await tts_queue.join()
                await worker_task_tts
                await output_queue.put(None)
                return final_text
                
            orch_task = asyncio.create_task(orchestrator())
            
            # Consume from the unified output queue and yield to WebSocket
            while True:
                item = await output_queue.get()
                if item is None:
                    break
                yield item
                
            full_response = orch_task.result()

            # Save to session memory
            clean_response_text = self._clean_llm_text(full_response)
            self.memory.add_turn(session.session_id, "user", transcription)
            self.memory.add_turn(
                session.session_id, "assistant", clean_response_text,
                emotion=detected_emotion,
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
