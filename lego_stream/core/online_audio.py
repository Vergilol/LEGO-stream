"""Bounded append-only audio used by true online inference sessions."""

from __future__ import annotations

from collections import deque
import math
import threading
from typing import Deque

import numpy as np


class OnlinePcmBuffer:
    """Store a causal mono float32 tail with absolute sample coordinates."""

    def __init__(self, *, sample_rate: int, retention_sec: float) -> None:
        self.sample_rate = int(sample_rate)
        self.retention_sec = float(retention_sec)
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not math.isfinite(self.retention_sec) or self.retention_sec <= 0.0:
            raise ValueError("retention_sec must be finite and positive")
        self._retention_samples = max(
            1,
            int(round(self.retention_sec * self.sample_rate)),
        )
        self._chunks: Deque[tuple[int, np.ndarray]] = deque()
        self._end_sample = 0
        self._next_sequence = 0
        self._lock = threading.RLock()

    @property
    def available_until_sec(self) -> float:
        with self._lock:
            return self._end_sample / float(self.sample_rate)

    @property
    def retained_from_sec(self) -> float:
        with self._lock:
            if not self._chunks:
                return self._end_sample / float(self.sample_rate)
            return self._chunks[0][0] / float(self.sample_rate)

    @property
    def next_sequence(self) -> int:
        with self._lock:
            return self._next_sequence

    def append(self, samples: np.ndarray, *, sequence: int) -> None:
        values = np.asarray(samples, dtype=np.float32)
        if values.ndim != 1 or values.size <= 0:
            raise ValueError("online audio append must be a non-empty mono array")
        if not np.all(np.isfinite(values)):
            raise ValueError("online audio samples must be finite")
        with self._lock:
            if int(sequence) != self._next_sequence:
                raise ValueError(
                    "audio sequence must be exactly %d, got %d"
                    % (self._next_sequence, int(sequence))
                )
            start = self._end_sample
            copied = np.ascontiguousarray(values, dtype=np.float32).copy()
            self._chunks.append((start, copied))
            self._end_sample += int(copied.size)
            self._next_sequence += 1
            self._evict_locked()

    def _evict_locked(self) -> None:
        retain_from = max(0, self._end_sample - self._retention_samples)
        while self._chunks:
            start, chunk = self._chunks[0]
            end = start + int(chunk.size)
            if end <= retain_from:
                self._chunks.popleft()
                continue
            if start < retain_from:
                offset = retain_from - start
                self._chunks[0] = (
                    retain_from,
                    np.ascontiguousarray(chunk[offset:], dtype=np.float32),
                )
            break

    def read(self, start_sec: float, end_sec: float) -> np.ndarray:
        start = int(round(float(start_sec) * self.sample_rate))
        end = int(round(float(end_sec) * self.sample_rate))
        if start < 0 or end <= start:
            raise ValueError("audio read range must be finite and positive")
        with self._lock:
            retained_from = self._chunks[0][0] if self._chunks else self._end_sample
            if start < retained_from:
                raise ValueError("requested online audio has already been evicted")
            if end > self._end_sample:
                raise ValueError("requested online audio is not available yet")
            pieces: list[np.ndarray] = []
            cursor = start
            for chunk_start, chunk in self._chunks:
                chunk_end = chunk_start + int(chunk.size)
                if chunk_end <= cursor:
                    continue
                if chunk_start >= end:
                    break
                take_start = max(cursor, chunk_start)
                take_end = min(end, chunk_end)
                if take_start > cursor:
                    raise RuntimeError("online audio buffer contains a gap")
                pieces.append(chunk[take_start - chunk_start : take_end - chunk_start])
                cursor = take_end
                if cursor >= end:
                    break
            if cursor != end:
                raise RuntimeError("online audio buffer could not satisfy the range")
            return np.concatenate(pieces).astype(np.float32, copy=False)


__all__ = ["OnlinePcmBuffer"]
