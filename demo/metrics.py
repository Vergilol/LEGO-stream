"""Wall-clock accounting for one live session, behind the page's tiles.

``at=`` overrides ``time.monotonic()`` for a deterministic replay.
``compute_rtf`` is compute time over processed audio, as the paper reports it;
waiting for audio that has not arrived yet is not compute.
"""

from __future__ import annotations

import math
import time
from typing import Mapping, Sequence


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class PerformanceTracker:
    """Tick latency, RTF, and publication lag for one session.

    Lag is on the audio clock: the releasing tick minus the turn's end.
    """

    def __init__(self, *, started_at: float | None = None) -> None:
        self.started_at = (
            time.monotonic() if started_at is None else float(started_at)
        )
        self._tick_started: dict[float, float] = {}
        self._tick_latencies: list[float] = []
        self._compute_wall_sec = 0.0
        self._processed_audio_sec = 0.0
        self._first_publication_at: float | None = None
        self._latest_publication_lag_sec: float | None = None
        # One per published turn, so the median is over turns, not ticks.
        self._publication_lags: list[float] = []
        self._last_snapshot: dict[str, float | None] = self.snapshot(
            at=self.started_at
        )

    def start_tick(self, tick: float, *, at: float | None = None) -> None:
        key = float(tick)
        if not math.isfinite(key) or key < 0.0:
            raise ValueError("tick must be finite and non-negative")
        if key in self._tick_started:
            return
        self._tick_started[key] = time.monotonic() if at is None else float(at)

    def has_started_tick(self, tick: float) -> bool:
        return float(tick) in self._tick_started

    def finish_tick(
        self,
        tick: float,
        *,
        publications: Sequence[Mapping[str, object]],
        at: float | None = None,
    ) -> dict[str, float | None]:
        key = float(tick)
        finished_at = time.monotonic() if at is None else float(at)
        started = self._tick_started.pop(key, None)
        if started is None:
            raise ValueError(f"tick {key:g} was not started")
        latency = max(0.0, finished_at - started)
        self._tick_latencies.append(latency)
        self._compute_wall_sec += latency
        self._processed_audio_sec = max(self._processed_audio_sec, key)
        if publications:
            if self._first_publication_at is None:
                self._first_publication_at = finished_at
            # A turn ending at 40 s and released at tick 44 s waited 4 s.
            lags = [
                max(0.0, key - float(publication.get("end", key)))
                for publication in publications
            ]
            self._publication_lags.extend(lags)
            self._latest_publication_lag_sec = max(lags)
        self._last_snapshot = self.snapshot(at=finished_at)
        return dict(self._last_snapshot)

    def snapshot(self, *, at: float | None = None) -> dict[str, float | None]:
        current = time.monotonic() if at is None else float(at)
        processed = self._processed_audio_sec
        latest = self._tick_latencies[-1] if self._tick_latencies else 0.0
        return {
            "elapsed_wall_sec": max(0.0, current - self.started_at),
            "processed_audio_sec": processed,
            "compute_wall_sec": self._compute_wall_sec,
            "compute_rtf": (
                self._compute_wall_sec / processed if processed > 0.0 else 0.0
            ),
            "time_to_first_publication_sec": (
                None
                if self._first_publication_at is None
                else max(0.0, self._first_publication_at - self.started_at)
            ),
            "latest_publication_lag_sec": self._latest_publication_lag_sec,
            # The page's "Commit delay": a median, so one turn held back to the
            # final flush does not stand for the whole session.
            "publication_lag_p50_sec": (
                _percentile(self._publication_lags, 0.50)
                if self._publication_lags
                else None
            ),
            "tick_latency_sec": latest,
            "tick_latency_p50_sec": _percentile(self._tick_latencies, 0.50),
            "tick_latency_p95_sec": _percentile(self._tick_latencies, 0.95),
            "tick_count": len(self._tick_latencies),
        }


__all__ = ["PerformanceTracker"]
