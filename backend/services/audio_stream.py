"""
IntelliVoice — Audio Streaming Utilities

Provides async audio streaming helpers for WebSocket communication.
"""

from __future__ import annotations

import asyncio
import struct
from typing import AsyncGenerator, Optional

from config.logging_config import get_logger

logger = get_logger("audio_stream")


class AudioStreamBuffer:
    """
    Async-safe audio buffer for streaming audio data.

    Buffers audio chunks and provides them for processing
    when enough data has accumulated.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        min_duration_ms: int = 500,
        max_duration_ms: int = 30000,
    ):
        self.sample_rate = sample_rate
        self.min_samples = int(sample_rate * min_duration_ms / 1000)
        self.max_samples = int(sample_rate * max_duration_ms / 1000)
        self._buffer = bytearray()
        self._lock = asyncio.Lock()
        self._ready_event = asyncio.Event()

    async def write(self, chunk: bytes) -> None:
        """Write audio chunk to buffer."""
        async with self._lock:
            self._buffer.extend(chunk)
            # Check if we have enough data
            num_samples = len(self._buffer) // 2  # int16 = 2 bytes
            if num_samples >= self.min_samples:
                self._ready_event.set()

    async def read(self, timeout: float = 5.0) -> Optional[bytes]:
        """
        Read buffered audio when enough has accumulated.

        Returns None on timeout.
        """
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

        async with self._lock:
            data = bytes(self._buffer)
            self._buffer.clear()
            self._ready_event.clear()
            return data

    async def flush(self) -> bytes:
        """Flush all buffered audio."""
        async with self._lock:
            data = bytes(self._buffer)
            self._buffer.clear()
            self._ready_event.clear()
            return data

    @property
    def duration_s(self) -> float:
        """Current buffer duration in seconds."""
        return (len(self._buffer) // 2) / self.sample_rate

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0


async def chunk_audio_stream(
    audio_data: bytes,
    chunk_size_bytes: int = 4096,
    delay_ms: float = 0,
) -> AsyncGenerator[bytes, None]:
    """
    Split audio data into chunks for streaming.

    Args:
        audio_data: Full audio bytes.
        chunk_size_bytes: Size of each chunk.
        delay_ms: Delay between chunks (simulates real-time).

    Yields:
        Audio chunks.
    """
    for i in range(0, len(audio_data), chunk_size_bytes):
        chunk = audio_data[i:i + chunk_size_bytes]
        yield chunk
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000)
