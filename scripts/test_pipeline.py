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
from backend.core.gpu_manager import GPUManager
from backend.core.pipeline import AudioPipeline
from backend.core.session_manager import SessionState
from backend.layers.preprocessing.audio_utils import waveform_to_bytes


setup_logging("INFO")
logger = get_logger("test_pipeline")


def generate_speech_like_audio(duration_s: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Generate synthetic speech-like audio for testing."""
    t = torch.linspace(0, duration_s, int(sample_rate * duration_s))

    # Generate speech-like waveform with formants
    f0 = 150  # Fundamental frequency
    waveform = torch.zeros_like(t)

    # Add harmonics
    for harmonic in range(1, 8):
        amplitude = 0.5 / harmonic
        waveform += amplitude * torch.sin(2 * 3.14159 * f0 * harmonic * t)

    # Add amplitude modulation (syllable-like)
    syllable_rate = 4  # syllables per second
    envelope = 0.5 + 0.5 * torch.sin(2 * 3.14159 * syllable_rate * t)
    waveform *= envelope

    # Normalize
    waveform = waveform / waveform.abs().max() * 0.8

    return waveform_to_bytes(waveform.unsqueeze(0))


async def test_preprocessing():
    """Test preprocessing layer independently."""
    print("\n" + "=" * 50)
    print("  Testing Preprocessing Layer")
    print("=" * 50)

    from backend.layers.preprocessing.vad import SileroVAD
    from backend.layers.preprocessing.noise_suppression import NoiseSuppressor
    from backend.layers.preprocessing.audio_utils import bytes_to_waveform

    vad = SileroVAD()
    await vad.load()
    print("✅ Silero VAD loaded")

    ns = NoiseSuppressor()
    await ns.load()
    print(f"{'✅' if ns.is_loaded else '⚠️'} DeepFilterNet {'loaded' if ns.is_loaded else 'unavailable (passthrough mode)'}")

    # Test with synthetic audio
    audio_bytes = generate_speech_like_audio(3.0)
    waveform, sr = bytes_to_waveform(audio_bytes)
    print(f"  Test audio: {waveform.shape[1] / sr:.1f}s, shape={waveform.shape}")

    # VAD
    is_speech = vad.is_speech(waveform)
    print(f"  VAD detection: {'speech' if is_speech else 'silence'}")

    segments = vad.detect_speech(waveform)
    print(f"  Speech segments: {len(segments)}")

    # Noise suppression
    clean = ns.suppress_noise(waveform)
    print(f"  Denoised shape: {clean.shape}")

    print("✅ Preprocessing layer test passed!")
    return True


async def test_memory():
    """Test memory layer."""
    print("\n" + "=" * 50)
    print("  Testing Memory Layer")
    print("=" * 50)

    from backend.layers.memory.conversation_memory import ConversationMemory
    from backend.layers.memory.long_term_memory import LongTermMemory

    memory = ConversationMemory()
    await memory.initialize()
    print("✅ Conversation memory initialized")

    # Test session
    session = memory.create_session("test-123")
    memory.add_turn("test-123", "user", "Hello, how are you?", emotion="neutral")
    memory.add_turn("test-123", "assistant", "I'm great! How can I help?")

    context = memory.get_conversation_context("test-123")
    assert len(context) == 2
    print(f"  Context turns: {len(context)}")

    # Long-term memory
    ltm = LongTermMemory()
    await ltm.connect()
    print(f"  MongoDB: {'connected' if ltm.is_connected else 'not available'}")

    memory.remove_session("test-123")
    print("✅ Memory layer test passed!")
    return True


async def test_response_planner():
    """Test response planning layer."""
    print("\n" + "=" * 50)
    print("  Testing Response Planning Layer")
    print("=" * 50)

    from backend.layers.response_planning.planner import ResponsePlanner

    planner = ResponsePlanner()

    plan = planner.plan(
        response_text="I understand how you feel. Let me help you with that.",
        user_emotion="sad",
        user_intent="complaint",
        detected_language="english",
    )

    print(f"  Intent: {plan.intent}")
    print(f"  Emotion: {plan.emotion}")
    print(f"  Tone: {plan.tone}")
    print(f"  Rate: {plan.speaking_rate}")
    print("✅ Response planner test passed!")
    return True


async def test_full_pipeline():
    """Test the full pipeline end-to-end."""
    print("\n" + "=" * 50)
    print("  Testing Full Pipeline (E2E)")
    print("=" * 50)

    gpu = GPUManager()
    pipeline = AudioPipeline(gpu_manager=gpu)
    await pipeline.initialize()
    print("✅ Pipeline initialized")

    # Create mock session
    session = SessionState(session_id="test-e2e", websocket=None)

    # Generate test audio
    audio_bytes = generate_speech_like_audio(3.0)
    print(f"  Test audio: {len(audio_bytes)} bytes")

    start = time.time()
    result = await pipeline.process_audio(
        audio_bytes=audio_bytes,
        session=session,
    )
    elapsed = time.time() - start

    print(f"\n  Results:")
    print(f"    Transcription: {result.get('transcription', 'N/A')[:100]}")
    print(f"    Emotion: {result.get('emotion', 'N/A')}")
    print(f"    Response: {result.get('response_text', 'N/A')[:100]}")
    print(f"    Audio out: {len(result.get('response_audio', b''))} bytes")
    print(f"    Total time: {elapsed:.2f}s")

    metadata = result.get("metadata", {})
    if "preprocess_ms" in metadata:
        print(f"\n  Timing breakdown:")
        print(f"    Preprocess:    {metadata.get('preprocess_ms', 'N/A')}ms")
        print(f"    Understanding: {metadata.get('understanding_ms', 'N/A')}ms")
        print(f"    Reasoning:     {metadata.get('reasoning_ms', 'N/A')}ms")
        print(f"    Generation:    {metadata.get('generation_ms', 'N/A')}ms")
        print(f"    Total:         {metadata.get('total_ms', 'N/A')}ms")

    await pipeline.shutdown()
    print("\n✅ Full pipeline test complete!")
    return True


async def main():
    parser = argparse.ArgumentParser(description="IntelliVoice Pipeline Test")
    parser.add_argument(
        "--layer",
        type=str,
        choices=["preprocessing", "memory", "planner", "full", "all"],
        default="all",
    )
    args = parser.parse_args()

    results = {}

    if args.layer in ("preprocessing", "all"):
        results["preprocessing"] = await test_preprocessing()
    if args.layer in ("memory", "all"):
        results["memory"] = await test_memory()
    if args.layer in ("planner", "all"):
        results["planner"] = await test_response_planner()
    if args.layer in ("full", "all"):
        results["full"] = await test_full_pipeline()

    print("\n" + "=" * 50)
    print("  Test Summary")
    print("=" * 50)
    for name, passed in results.items():
        print(f"  {'✅' if passed else '❌'} {name}")


if __name__ == "__main__":
    asyncio.run(main())
