"""
IntelliVoice — Audio Utilities

Common audio processing functions: resampling, format conversion,
chunking, normalization, and waveform operations.
"""

from __future__ import annotations

import io
import struct
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio

from config.logging_config import get_logger

logger = get_logger("audio_utils")

# Standard sample rate for all models
TARGET_SAMPLE_RATE = 16000


def bytes_to_waveform(
    audio_bytes: bytes,
    source_sample_rate: int = 16000,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    dtype: str = "int16",
) -> Tuple[torch.Tensor, int]:
    """
    Convert raw audio bytes to a PyTorch waveform tensor.

    Args:
        audio_bytes: Raw PCM audio bytes.
        source_sample_rate: Sample rate of the input audio.
        target_sample_rate: Desired output sample rate.
        dtype: Data type of the input bytes ('int16' or 'float32').

    Returns:
        Tuple of (waveform tensor [1, T], sample_rate).
    """
    if dtype == "int16":
        # Parse 16-bit signed integers
        num_samples = len(audio_bytes) // 2
        samples = struct.unpack(f"<{num_samples}h", audio_bytes[:num_samples * 2])
        waveform = torch.tensor(samples, dtype=torch.float32) / 32768.0
    elif dtype == "float32":
        num_samples = len(audio_bytes) // 4
        samples = struct.unpack(f"<{num_samples}f", audio_bytes[:num_samples * 4])
        waveform = torch.tensor(samples, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Ensure mono, shape [1, T]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Resample if needed
    if source_sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, source_sample_rate, target_sample_rate
        )

    return waveform, target_sample_rate


def waveform_to_bytes(
    waveform: torch.Tensor,
    sample_rate: int = TARGET_SAMPLE_RATE,
    dtype: str = "int16",
) -> bytes:
    """
    Convert a PyTorch waveform tensor back to raw PCM bytes.

    Args:
        waveform: Tensor of shape [1, T] or [T].
        sample_rate: Sample rate (unused, for API consistency).
        dtype: Output data type ('int16' or 'float32').

    Returns:
        Raw PCM bytes.
    """
    if waveform.dim() > 1:
        waveform = waveform.squeeze(0)

    waveform = waveform.cpu().float()

    if dtype == "int16":
        # Clamp and convert to 16-bit
        waveform = torch.clamp(waveform, -1.0, 1.0)
        int_samples = (waveform * 32767).to(torch.int16)
        return int_samples.numpy().tobytes()
    elif dtype == "float32":
        return waveform.numpy().tobytes()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def waveform_to_wav_bytes(
    waveform: torch.Tensor,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> bytes:
    """Convert waveform tensor to WAV file bytes."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    buffer = io.BytesIO()
    torchaudio.save(buffer, waveform.cpu(), sample_rate, format="wav")
    buffer.seek(0)
    return buffer.read()


def wav_bytes_to_waveform(wav_bytes: bytes) -> Tuple[torch.Tensor, int]:
    """Load WAV bytes into a waveform tensor."""
    buffer = io.BytesIO(wav_bytes)
    waveform, sr = torchaudio.load(buffer)
    return waveform, sr


def normalize_waveform(waveform: torch.Tensor, target_db: float = -20.0) -> torch.Tensor:
    """Normalize waveform to a target dB level."""
    rms = torch.sqrt(torch.mean(waveform ** 2))
    if rms > 0:
        target_rms = 10 ** (target_db / 20)
        waveform = waveform * (target_rms / rms)
    return torch.clamp(waveform, -1.0, 1.0)


def chunk_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_ms: int = 30,
    overlap_ms: int = 0,
) -> list[torch.Tensor]:
    """
    Split waveform into fixed-size chunks.

    Args:
        waveform: [1, T] or [T] tensor.
        sample_rate: Audio sample rate.
        chunk_ms: Chunk duration in milliseconds.
        overlap_ms: Overlap between chunks in milliseconds.

    Returns:
        List of chunk tensors.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    chunk_size = int(sample_rate * chunk_ms / 1000)
    overlap_size = int(sample_rate * overlap_ms / 1000)
    step_size = chunk_size - overlap_size

    total_samples = waveform.shape[1]
    chunks = []

    for start in range(0, total_samples, step_size):
        end = min(start + chunk_size, total_samples)
        chunk = waveform[:, start:end]

        # Pad last chunk if needed
        if chunk.shape[1] < chunk_size:
            padding = torch.zeros(1, chunk_size - chunk.shape[1])
            chunk = torch.cat([chunk, padding], dim=1)

        chunks.append(chunk)

    return chunks


def compute_energy(waveform: torch.Tensor, frame_ms: int = 10, sample_rate: int = 16000) -> torch.Tensor:
    """Compute frame-level energy of a waveform."""
    if waveform.dim() > 1:
        waveform = waveform.squeeze(0)

    frame_size = int(sample_rate * frame_ms / 1000)
    num_frames = len(waveform) // frame_size

    energy = torch.zeros(num_frames)
    for i in range(num_frames):
        frame = waveform[i * frame_size : (i + 1) * frame_size]
        energy[i] = torch.sqrt(torch.mean(frame ** 2))

    return energy


def get_audio_duration(audio_bytes: bytes, sample_rate: int = 16000, dtype: str = "int16") -> float:
    """Get duration in seconds from raw audio bytes."""
    bytes_per_sample = 2 if dtype == "int16" else 4
    num_samples = len(audio_bytes) // bytes_per_sample
    return num_samples / sample_rate
