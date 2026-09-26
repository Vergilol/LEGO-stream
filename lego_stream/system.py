"""Public API for the canonical streaming speaker-attributed ASR system.

One causal path, ``L -> E -> G -> O``: the local index inside one decode, the
immutable evidence of a committed turn, the mutable global speaker set, and the
append-only output label.  The README states the full contract.

``inference.causal_frontier_request_ids`` (CRFS) is an optional decoder hint,
off by default; only the client half ships here, so it changes neither the
identity nor the publication contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .runtime.online import StreamingSpeakerASRSession
from .runtime.pipeline import run_pipeline as _run_pipeline


def run_streaming_file(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    attribution_observer: Any = None,
    stage_profiler: Optional[Callable[[Mapping[str, float]], None]] = None,
) -> dict[str, Any]:
    """Run one file through the canonical causal publication state machine."""

    return _run_pipeline(
        config,
        output_dir,
        attribution_observer=attribution_observer,
        stage_profiler=stage_profiler,
    )


__all__ = [
    "StreamingSpeakerASRSession",
    "run_streaming_file",
]
