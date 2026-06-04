#!/usr/bin/env python3
"""
IntelliVoice — Pipeline Benchmark

Measures latency and throughput for each pipeline layer.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --layer vad
    python scripts/benchmark.py --iterations 20
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logging_config import setup_logging, get_logger

setup_logging("WARNING")
logger = get_logger("benchmark")


def generate_test_audio(duration_s: float = 3.0, sample_rate: int = 16000) -> torch.Tensor:
    """Generate a test waveform (speech-like with varying frequencies)."""
    t = torch.linspace(0, duration_s, int(sample_rate * duration_s))
    # Mix of frequencies to simulate speech
    waveform = (
        0.3 * torch.sin(2 * 3.14159 * 200 * t) +
        0.2 * torch.sin(2 * 3.14159 * 400 * t) +
        0.1 * torch.sin(2 * 3.14159 * 800 * t) +
        0.05 * torch.randn_like(t)  # Add noise
    )
    return waveform.unsqueeze(0)


async def benchmark_vad(iterations: int = 10):
    """Benchmark VAD layer."""
    from backend.layers.preprocessing.vad import SileroVAD

    print("\n--- Benchmarking Silero VAD ---")
    vad = SileroVAD()
    await vad.load()

    waveform = generate_test_audio(3.0)
    times = []

    for i in range(iterations):
        vad.reset_state()
        start = time.perf_counter()
        vad.detect_speech(waveform)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    print(f"  Iterations: {iterations}")
    print(f"  Mean: {np.mean(times):.2f}ms")
    print(f"  Std:  {np.std(times):.2f}ms")
    print(f"  Min:  {np.min(times):.2f}ms")
    print(f"  Max:  {np.max(times):.2f}ms")
    print(f"  Audio: 3.0s → {np.mean(times):.1f}ms (RTF: {np.mean(times) / 3000:.4f})")


async def benchmark_preprocessing(iterations: int = 10):
    """Benchmark full preprocessing pipeline."""
    from backend.layers.preprocessing.vad import SileroVAD
    from backend.layers.preprocessing.noise_suppression import NoiseSuppressor
    from backend.layers.preprocessing.audio_utils import normalize_waveform

    print("\n--- Benchmarking Preprocessing Pipeline ---")
    vad = SileroVAD()
    await vad.load()
    ns = NoiseSuppressor()
    await ns.load()

    waveform = generate_test_audio(3.0)
    times = []

    for i in range(iterations):
        start = time.perf_counter()
        # VAD
        speech, _ = vad.extract_speech(waveform)
        # Noise suppression
        clean = ns.suppress_noise(speech if speech.shape[1] > 0 else waveform)
        # Normalize
        normalized = normalize_waveform(clean)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    print(f"  Mean: {np.mean(times):.2f}ms")
    print(f"  Std:  {np.std(times):.2f}ms")


async def benchmark_audio_utils(iterations: int = 100):
    """Benchmark audio utility functions."""
    from backend.layers.preprocessing.audio_utils import (
        bytes_to_waveform,
        waveform_to_bytes,
        normalize_waveform,
        chunk_waveform,
    )

    print("\n--- Benchmarking Audio Utilities ---")

    waveform = generate_test_audio(3.0)
    audio_bytes = waveform_to_bytes(waveform)

    # bytes_to_waveform
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        bytes_to_waveform(audio_bytes)
        times.append((time.perf_counter() - start) * 1000)
    print(f"  bytes_to_waveform: {np.mean(times):.3f}ms")

    # waveform_to_bytes
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        waveform_to_bytes(waveform)
        times.append((time.perf_counter() - start) * 1000)
    print(f"  waveform_to_bytes: {np.mean(times):.3f}ms")

    # normalize
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        normalize_waveform(waveform)
        times.append((time.perf_counter() - start) * 1000)
    print(f"  normalize: {np.mean(times):.3f}ms")

    # chunk
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        chunk_waveform(waveform, 16000, chunk_ms=30)
        times.append((time.perf_counter() - start) * 1000)
    print(f"  chunk (30ms): {np.mean(times):.3f}ms")


async def main():
    parser = argparse.ArgumentParser(description="IntelliVoice Pipeline Benchmark")
    parser.add_argument(
        "--layer",
        type=str,
        choices=["vad", "preprocessing", "utils", "all"],
        default="all",
    )
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    print("=" * 50)
    print("  IntelliVoice Pipeline Benchmark")
    print("=" * 50)

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("  GPU: None (CPU only)")

    if args.layer in ("vad", "all"):
        await benchmark_vad(args.iterations)
    if args.layer in ("preprocessing", "all"):
        await benchmark_preprocessing(args.iterations)
    if args.layer in ("utils", "all"):
        await benchmark_audio_utils(args.iterations * 10)

    print("\n" + "=" * 50)
    print("  Benchmark complete!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
