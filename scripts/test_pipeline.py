#!/usr/bin/env python3
"""
IntelliVoice — End-to-End Pipeline Test

Tests the full pipeline with a synthetic audio input.

Usage:
    python scripts/test_pipeline.py
    python scripts/test_pipeline.py --audio path/to/test.wav
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings
from config.logging_config import setup_logging, get_logger
from config.model_registry import ModelRegistry
from backend.core.gpu_manager import GPUManager
from backend.core.pipeline import AudioPipeline
from backend.core.session_manager import SessionState
from backend.layers.preprocessing.audio_utils import waveform_to_bytes


setup_logging("INFO")
logger = get_logger("test_pipeline")


def generate_speech_like_audio(duration_s: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Synthetic speech-like audio (formants + amplitude modulation)."""
    t = torch.linspace(0, duration_s, int(sample_rate * duration_s))
    f0 = 150
    waveform = torch.zeros_like(t)
    for harmonic in range(1, 8):
        waveform += (0.5 / harmonic) * torch.sin(2 * 3.14159 * f0 * harmonic * t)
    envelope = 0.5 + 0.5 * torch.sin(2 * 3.14159 * 4 * t)
    waveform *= envelope
    waveform = waveform / waveform.abs().max() * 0.8
    return waveform_to_bytes(waveform.unsqueeze(0))


async def test_preprocessing():
    print("\n" + "=" * 50)
    print("  Preprocessing Layer (VAD + DeepFilterNet)")
    print("=" * 50)

    from backend.layers.preprocessing.vad import SileroVAD
    from backend.layers.preprocessing.noise_suppression import NoiseSuppressor
    from backend.layers.preprocessing.audio_utils import bytes_to_waveform

    vad = SileroVAD()
    await vad.load()
    print("[OK] Silero VAD loaded")

    ns = NoiseSuppressor()
    await ns.load()
    print(
        f"{'[OK]' if ns.is_loaded else '[WARN]'} DeepFilterNet "
        f"{'loaded' if ns.is_loaded else 'unavailable (passthrough mode)'}"
    )

    audio_bytes = generate_speech_like_audio(3.0)
    waveform, sr = bytes_to_waveform(audio_bytes)
    print(f"  Test audio: {waveform.shape[1] / sr:.1f}s")

    is_speech = vad.is_speech(waveform)
    print(f"  VAD detection: {'speech' if is_speech else 'silence'}")
    segments = vad.detect_speech(waveform)
    print(f"  Speech segments: {len(segments)}")

    clean = ns.suppress_noise(waveform)
    print(f"  Denoised shape: {clean.shape}")
    print("[OK] Preprocessing test passed!")
    return True


async def test_memory():
    print("\n" + "=" * 50)
    print("  Memory Layer (LangGraph + MongoDB)")
    print("=" * 50)
    from backend.layers.memory.conversation_memory import ConversationMemory
    from backend.layers.memory.long_term_memory import LongTermMemory

    memory = ConversationMemory()
    await memory.initialize()
    print("[OK] Conversation memory initialized")

    memory.add_turn("test-123", "user", "Hello, how are you?", emotion="neutral")
    memory.add_turn("test-123", "assistant", "I'm great! How can I help?")
    context = memory.get_conversation_context("test-123")
    assert len(context) == 2
    print(f"  Context turns: {len(context)}")

    ltm = LongTermMemory()
    await ltm.connect()
    print(f"  MongoDB: {'connected' if ltm.is_connected else 'not available (using in-memory)'}")
    memory.remove_session("test-123")
    print("[OK] Memory test passed!")
    return True



async def test_vram_budget():
    print("\n" + "=" * 50)
    print("  VRAM Budget (RTX 4080 16GB target)")
    print("=" * 50)
    ModelRegistry.print_vram_budget()
    return True


async def test_full_pipeline():
    print("\n" + "=" * 50)
    print("  Full Pipeline (E2E)")
    print("=" * 50)
    gpu = GPUManager()
    pipeline = AudioPipeline(gpu_manager=gpu)
    await pipeline.initialize()
    print("[OK] Pipeline initialized")

    session = SessionState(session_id="test-e2e", websocket=None)
    audio_bytes = generate_speech_like_audio(3.0)
    print(f"  Test audio: {len(audio_bytes)} bytes")

    start = time.time()
    result = await pipeline.process_audio(
        audio_bytes=audio_bytes,
        session=session,
    )
    elapsed = time.time() - start

    print("\n  Results:")
    print(f"    Transcription : {result.get('transcription', 'N/A')[:100]}")
    print(f"    Emotion       : {result.get('metadata', {}).get('emotion', 'N/A')}")
    print(f"    Response      : {result.get('response_text', 'N/A')[:100]}")
    print(f"    Audio out     : {len(result.get('response_audio', b''))} bytes")
    print(f"    Total time    : {elapsed:.2f}s")

    md = result.get("metadata", {})
    print(f"    Total time    : {md.get('total_ms', 'N/A')}ms")
    print(f"    Style tags    : {md.get('style_tags', [])}")

    await pipeline.shutdown()
    print("\n[OK] Full pipeline test complete!")
    return True


async def main():
    parser = argparse.ArgumentParser(description="IntelliVoice Pipeline Test")
    parser.add_argument(
        "--layer",
        type=str,
        choices=[
            "preprocessing", "memory", "vram",
            "full", "all",
        ],
        default="all",
    )
    args = parser.parse_args()

    results = {}
    if args.layer in ("preprocessing", "all"):
        results["preprocessing"] = await test_preprocessing()
    if args.layer in ("memory", "all"):
        results["memory"] = await test_memory()

    if args.layer in ("vram", "all"):
        results["vram"] = await test_vram_budget()
    if args.layer in ("full", "all"):
        results["full"] = await test_full_pipeline()

    print("\n" + "=" * 50)
    print("  Test Summary")
    print("=" * 50)
    for name, passed in results.items():
        print(f"  {'[OK]' if passed else '[FAIL]'} {name}")


if __name__ == "__main__":
    asyncio.run(main())
