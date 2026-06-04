"""
Tests for Layer 1: Audio Preprocessing
"""

import pytest
import torch
import numpy as np

from backend.layers.preprocessing.audio_utils import (
    bytes_to_waveform,
    waveform_to_bytes,
    waveform_to_wav_bytes,
    wav_bytes_to_waveform,
    normalize_waveform,
    chunk_waveform,
    compute_energy,
    get_audio_duration,
)


class TestAudioUtils:
    """Tests for audio utility functions."""

    def _make_waveform(self, duration_s: float = 1.0, sr: int = 16000) -> torch.Tensor:
        """Generate a test waveform."""
        t = torch.linspace(0, duration_s, int(sr * duration_s))
        return (0.5 * torch.sin(2 * 3.14159 * 440 * t)).unsqueeze(0)

    def test_bytes_to_waveform_int16(self):
        """Test converting int16 bytes to waveform."""
        original = self._make_waveform(1.0)
        audio_bytes = waveform_to_bytes(original, dtype="int16")
        recovered, sr = bytes_to_waveform(audio_bytes, dtype="int16")

        assert sr == 16000
        assert recovered.shape[0] == 1
        assert recovered.shape[1] == original.shape[1]

    def test_bytes_to_waveform_float32(self):
        """Test converting float32 bytes to waveform."""
        original = self._make_waveform(0.5)
        audio_bytes = waveform_to_bytes(original, dtype="float32")
        recovered, sr = bytes_to_waveform(audio_bytes, dtype="float32")

        assert sr == 16000
        # Float32 should have minimal conversion loss
        assert torch.allclose(original, recovered, atol=1e-6)

    def test_wav_roundtrip(self):
        """Test WAV bytes roundtrip."""
        original = self._make_waveform(0.5)
        wav_bytes = waveform_to_wav_bytes(original)
        recovered, sr = wav_bytes_to_waveform(wav_bytes)

        assert sr == 16000
        assert recovered.shape[0] == 1

    def test_normalize_waveform(self):
        """Test waveform normalization."""
        waveform = self._make_waveform() * 0.01  # Very quiet
        normalized = normalize_waveform(waveform, target_db=-20)

        # Should be louder after normalization
        assert normalized.abs().max() > waveform.abs().max()
        # Should not clip
        assert normalized.abs().max() <= 1.0

    def test_chunk_waveform(self):
        """Test waveform chunking."""
        waveform = self._make_waveform(1.0, 16000)
        chunks = chunk_waveform(waveform, 16000, chunk_ms=100)

        # 1 second / 100ms = 10 chunks
        assert len(chunks) == 10
        # Each chunk should be 1600 samples (100ms at 16kHz)
        assert chunks[0].shape[1] == 1600

    def test_chunk_with_overlap(self):
        """Test chunking with overlap."""
        waveform = self._make_waveform(1.0, 16000)
        chunks = chunk_waveform(waveform, 16000, chunk_ms=100, overlap_ms=50)

        # With 50% overlap, more chunks
        assert len(chunks) > 10

    def test_compute_energy(self):
        """Test energy computation."""
        waveform = self._make_waveform()
        energy = compute_energy(waveform, frame_ms=10, sample_rate=16000)

        assert len(energy) > 0
        assert all(e >= 0 for e in energy)

    def test_get_audio_duration(self):
        """Test duration estimation."""
        # 1 second of int16 audio at 16kHz = 32000 bytes
        duration = get_audio_duration(b"\x00" * 32000, sample_rate=16000, dtype="int16")
        assert abs(duration - 1.0) < 0.01

    def test_resampling(self):
        """Test resampling from 48kHz to 16kHz."""
        t = torch.linspace(0, 1.0, 48000)
        waveform_48k = (0.5 * torch.sin(2 * 3.14159 * 440 * t)).unsqueeze(0)
        audio_bytes = waveform_to_bytes(waveform_48k)

        result, sr = bytes_to_waveform(
            audio_bytes,
            source_sample_rate=48000,
            target_sample_rate=16000,
        )

        assert sr == 16000
        # Should be approximately 16000 samples
        assert abs(result.shape[1] - 16000) < 100


class TestVAD:
    """Tests for Silero VAD."""

    @pytest.fixture
    async def vad(self):
        from backend.layers.preprocessing.vad import SileroVAD
        v = SileroVAD(threshold=0.5)
        await v.load()
        return v

    @pytest.mark.asyncio
    async def test_vad_loads(self, vad):
        assert vad.is_loaded

    @pytest.mark.asyncio
    async def test_silence_detection(self, vad):
        """Silence should not be detected as speech."""
        silence = torch.zeros(1, 16000)
        assert not vad.is_speech(silence)

    @pytest.mark.asyncio
    async def test_speech_probability(self, vad):
        """Should return a probability between 0 and 1."""
        silence = torch.zeros(1, 16000)
        prob = vad.get_speech_probability(silence)
        assert 0.0 <= prob <= 1.0

    @pytest.mark.asyncio
    async def test_reset_state(self, vad):
        """Reset should not error."""
        vad.reset_state()
