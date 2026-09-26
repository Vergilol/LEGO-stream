from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .commit import CommitUpdate, IncrementalCommitter
from ..types import Segment
from .validation import ValidationError, validate_trace_rows


@dataclass(frozen=True)
class WindowSpec:
    tick: float
    start: float
    end: float


@dataclass
class DecodeOutput:
    segments: Sequence[Segment]
    text: str = ""
    elapsed_sec: float = 0.0
    metadata: Dict[str, Any] | None = None


@dataclass
class EngineResult:
    rows: List[dict[str, Any]]
    committed: List[Segment]
    validation: dict[str, Any]


class IncrementalEngine:
    """Run a full-window decoder while publishing only its stable delta."""

    def __init__(
        self,
        *,
        duration_sec: float,
        step_sec: float,
        context_sec: float,
        right_context_sec: float,
        decode: Callable[[WindowSpec, str], DecodeOutput],
        map_segments: Optional[Callable[[Sequence[Segment], WindowSpec], Sequence[Segment]]] = None,
        prompt: Optional[Callable[[WindowSpec, Sequence[Segment]], str]] = None,
        after_update: Optional[
            Callable[
                [WindowSpec, DecodeOutput, Sequence[Segment], CommitUpdate],
                Optional[Mapping[str, Any]],
            ]
        ] = None,
        recoverable_error_metadata: Optional[
            Callable[[WindowSpec, Exception], Mapping[str, Any]]
        ] = None,
        recoverable_decode_error_filter: Optional[
            Callable[[Exception], bool]
        ] = None,
        max_wall_latency_sec: Optional[float] = None,
        continue_on_decode_error: bool = False,
        committer: Optional[IncrementalCommitter] = None,
        sample_metadata: Optional[Mapping[str, Any]] = None,
        attribution_observer: Any = None,
        stage_profiler: Optional[Callable[[Mapping[str, float]], None]] = None,
    ) -> None:
        self.duration_sec = self._positive(duration_sec, "duration_sec")
        self.step_sec = self._positive(step_sec, "step_sec")
        self.context_sec = self._positive(context_sec, "context_sec")
        self.right_context_sec = self._nonnegative(right_context_sec, "right_context_sec")
        # ``max_wall_latency_sec`` is an optional operational run SLO.  It is
        # observed after decoding and never changes stable-frontier or commit
        # decisions.  When configured, it remains part of run validation and
        # is reported separately from publication-age telemetry.
        self.max_wall_latency_sec = (
            None
            if max_wall_latency_sec is None
            else self._positive(max_wall_latency_sec, "max_wall_latency_sec")
        )
        if self.right_context_sec >= self.context_sec:
            raise ValueError("right_context_sec must be smaller than context_sec")
        self.decode = decode
        self.map_segments = map_segments or (lambda segments, _window: list(segments))
        self.prompt = prompt or (lambda _window, _pending: "")
        self.after_update = after_update
        self.recoverable_error_metadata = recoverable_error_metadata
        self.recoverable_decode_error_filter = recoverable_decode_error_filter
        self.continue_on_decode_error = bool(continue_on_decode_error)
        self.committer = committer or IncrementalCommitter(
            right_context_sec=self.right_context_sec
        )
        self.sample_metadata = dict(sample_metadata or {})
        self.attribution_observer = attribution_observer
        self.stage_profiler = stage_profiler
        if self.attribution_observer is not None:
            self.committer.set_transition_observer(self._record_attribution)
        self._ran = False

    def _record_attribution(self, value: Mapping[str, object]) -> None:
        if self.attribution_observer is None:
            return
        try:
            from ..attribution import (
                canonical_sha256,
                safe_observer_record,
            )

            state_before = self.committer.attribution_snapshot()
            digest_before = canonical_sha256(state_before)
            safe_observer_record(self.attribution_observer, value)
            state_after = self.committer.attribution_snapshot()
            digest_after = canonical_sha256(state_after)
            safe_observer_record(
                self.attribution_observer,
                {
                    "authoritative_digest_after": digest_after,
                    "authoritative_digest_before": digest_before,
                    "observed_event_sha256": canonical_sha256(value),
                    "stage": "observer_non_interference",
                    "tick": value.get("tick"),
                },
            )
        except Exception:
            return

    def _record_stage_profile(self, value: Mapping[str, float]) -> None:
        if self.stage_profiler is None:
            return
        try:
            self.stage_profiler(dict(value))
        except Exception:
            return

    @staticmethod
    def _positive(value: float, name: str) -> float:
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return number

    @staticmethod
    def _nonnegative(value: float, name: str) -> float:
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return number

    def schedule(self) -> List[WindowSpec]:
        windows: List[WindowSpec] = []
        tick = self.step_sec
        while tick < self.duration_sec - 1e-9:
            end = min(self.duration_sec, tick)
            windows.append(
                WindowSpec(
                    tick=end,
                    start=max(0.0, end - self.context_sec),
                    end=end,
                )
            )
            tick += self.step_sec
        if not windows or windows[-1].end < self.duration_sec - 1e-9:
            windows.append(
                WindowSpec(
                    tick=self.duration_sec,
                    start=max(0.0, self.duration_sec - self.context_sec),
                    end=self.duration_sec,
                )
            )
        return windows

    def start_session(
        self,
        output_dir: str | Path | None = None,
    ) -> "IncrementalEngineSession":
        if self._ran:
            raise RuntimeError("IncrementalEngine instances are single-use")
        self._ran = True
        return IncrementalEngineSession(self, output_dir=output_dir)

    def run(self, output_dir: str | Path | None = None) -> EngineResult:
        windows = self.schedule()
        session = self.start_session(output_dir)
        try:
            for index, window in enumerate(windows):
                session.process(window, final=index == len(windows) - 1)
            return session.finalize(expected_ticks=len(windows))
        except Exception as exc:
            session.fail(exc, expected_ticks=len(windows))
            raise

    def _row(
        self,
        window: WindowSpec,
        decoded: DecodeOutput,
        mapped: Sequence[Segment],
        update: CommitUpdate,
        wall_elapsed: float,
        *,
        extra_fields: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        delta = [item.to_dict() for item in update.committed]
        row = {
            "tick": float(window.tick),
            "window_start": float(window.start),
            "window_end": float(window.end),
            "status": "passed",
            "gate_status": "passed",
            "text": str(decoded.text),
            "local_segments": [item.to_dict() for item in decoded.segments],
            "mapped_segments": [item.to_dict() for item in mapped],
            # ``local_segments``/``mapped_segments`` are complete rolling window
            # snapshots and repeat old audio by design; ``delta_segments`` is the
            # append-only payload, so concatenating it across ticks reconstructs
            # the final hypothesis without replaying the overlap.
            # ``committed_segments`` is an exact alias kept for evaluators.
            "delta_segments": delta,
            "committed_segments": delta,
            "user_text_delta": "".join(str(item.text) for item in update.committed),
            "pending_segments": [item.to_dict() for item in update.pending],
            "stable_until": float(update.stable_until),
            "published_until": float(update.published_until),
            "forced_empty_frontier": update.forced_empty_frontier,
            "pending_age_sec": float(update.pending_age_sec),
            "delta_first_publish_age_sec": float(
                update.delta_first_publish_age_sec
            ),
            "delta_first_publish_age_max_sec": float(
                update.delta_first_publish_age_sec
            ),
            "delayed_pending_publish_count": int(
                update.delayed_pending_publish_count
            ),
            "delayed_pending_publish_max_lag_sec": float(
                update.delayed_pending_publish_max_lag_sec
            ),
            "delta_frontier_backtrack_count": int(
                update.delta_frontier_backtrack_count
            ),
            "delta_frontier_backtrack_max_sec": float(
                update.delta_frontier_backtrack_max_sec
            ),
            "elapsed_sec": float(decoded.elapsed_sec),
            "wall_elapsed_sec": float(wall_elapsed),
            "metadata": dict(decoded.metadata or {}),
        }
        for key, value in dict(extra_fields or {}).items():
            if key in row:
                raise ValueError(f"after_update cannot overwrite trace field {key!r}")
            row[str(key)] = value
        return row

    @staticmethod
    def _validate_window_segments(
        segments: Sequence[Segment],
        window: WindowSpec,
        label: str,
    ) -> None:
        for index, segment in enumerate(segments):
            if not isinstance(segment, Segment):
                raise ValidationError(
                    "%s segment %d is not a canonical Segment" % (label, index)
                )
            if segment.start < window.start - 1e-3 or segment.end > window.end + 1e-3:
                raise ValidationError(
                    "%s segment %d timestamp [%0.3f, %0.3f] is outside window [%0.3f, %0.3f]"
                    % (
                        label,
                        index,
                        segment.start,
                        segment.end,
                        window.start,
                        window.end,
                    )
                )

    def _validation(
        self,
        rows: Sequence[Mapping[str, Any]],
        wall_latencies: Sequence[float],
        expected_ticks: int,
        *,
        observed_mapped_segment_count: int = 0,
        observed_material_mapped_segment_count: int = 0,
    ) -> dict[str, Any]:
        max_wall = max(wall_latencies) if wall_latencies else 0.0
        failed_ticks = [row.get("tick") for row in rows if row.get("status") != "passed"]
        recoverable_decode_error_ticks = [
            row.get("tick")
            for row in rows
            if bool(row.get("metadata", {}).get("recoverable_decode_error"))
        ]
        recoverable_decode_error_count = len(recoverable_decode_error_ticks)
        recoverable_decode_error_gate_pass = bool(
            recoverable_decode_error_count < int(expected_ticks)
        )
        pending_ages = [
            float(row.get("pending_age_sec", 0.0))
            for row in rows
            if row.get("status") == "passed"
        ]
        pending_age_max = max(pending_ages) if pending_ages else 0.0
        delta_first_publish_ages = [
            float(row.get("delta_first_publish_age_sec", math.inf))
            for row in rows
            if row.get("status") == "passed"
        ]
        delta_first_publish_age_max = (
            max(delta_first_publish_ages) if delta_first_publish_ages else 0.0
        )
        trace_pass = True
        trace_error = None
        try:
            validate_trace_rows(
                rows,
                expected_ticks=expected_ticks,
            )
        except ValidationError as exc:
            trace_pass = False
            trace_error = str(exc)
        wall_latency_gate_pass: Optional[bool] = None
        if self.max_wall_latency_sec is not None:
            wall_latency_gate_pass = bool(max_wall <= self.max_wall_latency_sec + 1e-9)
        execution_gate_pass = (
            not failed_ticks
            and len(rows) == expected_ticks
            and recoverable_decode_error_gate_pass
            and wall_latency_gate_pass is not False
        )
        committed_segment_count = len(self.committer.committed)
        observed_speech = bool(observed_mapped_segment_count > 0)
        observed_material_content = bool(
            observed_material_mapped_segment_count > 0
        )
        final_content_gate_pass = bool(
            not observed_material_content or committed_segment_count > 0
        )
        passed = bool(
            execution_gate_pass and trace_pass and final_content_gate_pass
        )
        return {
            "status": "passed" if passed else "failed",
            "tick_count": len(rows),
            "expected_tick_count": expected_ticks,
            "failed_ticks": failed_ticks,
            "recoverable_decode_error_count": recoverable_decode_error_count,
            "recoverable_decode_error_ticks": recoverable_decode_error_ticks,
            "recoverable_decode_error_gate_pass": recoverable_decode_error_gate_pass,
            "timestamp_violations": [],
            "schedule_mismatches": [],
            "latency": {
                "source": "wall_elapsed_sec",
                "max_sec": max_wall,
                "count": len(wall_latencies),
            },
            "wall_latency_gate_sec": self.max_wall_latency_sec,
            "wall_latency_gate_pass": wall_latency_gate_pass,
            "algorithmic_delay_sec": self.step_sec + self.right_context_sec,
            "execution_gate_pass": execution_gate_pass,
            "pending_age_max_sec": pending_age_max,
            "delta_first_publish_age_max_sec": delta_first_publish_age_max,
            "trace_gate_pass": trace_pass,
            "trace_gate_error": trace_error,
            "observed_speech": observed_speech,
            "observed_material_content": observed_material_content,
            "observed_mapped_segment_count": int(observed_mapped_segment_count),
            "committed_segment_count": committed_segment_count,
            "final_content_gate_pass": final_content_gate_pass,
            "duplicate_suppressed": self.committer.duplicate_count,
            "late_revision_count": self.committer.late_drop_count,
            "evidence_poor_turn_drop_count": int(
                getattr(self.committer, "evidence_poor_drop_count", 0)
            ),
            "evidence_poor_turn_drop_reasons": dict(
                getattr(self.committer, "evidence_poor_drop_reasons", {})
            ),
        }

    def _write_outputs(self, destination: Path, result: EngineResult) -> None:
        self._write_jsonl(destination / "results" / "ticks.jsonl", result.rows)
        self._write_json(destination / "stream_validation.json", result.validation)
        if result.validation.get("status") == "passed":
            item = dict(self.sample_metadata)
            item["segments"] = [item.to_dict() for item in result.committed]
            self._write_json(destination / "hypothesis.json", [item])

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def _write_jsonl(cls, path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        temporary.replace(path)


__all__ = ["DecodeOutput", "EngineResult", "IncrementalEngine", "WindowSpec"]


class IncrementalEngineSession:
    """Long-lived, single-pass driver for one authoritative engine state."""

    def __init__(
        self,
        engine: IncrementalEngine,
        *,
        output_dir: str | Path | None = None,
    ) -> None:
        self.engine = engine
        self.destination = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else None
        )
        if self.destination is not None:
            for stale in (
                self.destination / "hypothesis.json",
                self.destination / "global_embedding" / "hypothesis.json",
            ):
                stale.unlink(missing_ok=True)
        self.rows: List[dict[str, Any]] = []
        self.wall_latencies: List[float] = []
        self.observed_mapped_segment_count = 0
        self.observed_material_mapped_segment_count = 0
        self.validation: Optional[dict[str, Any]] = None
        self._last_window_end = -math.inf
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("IncrementalEngineSession is already finalized")

    def process(
        self,
        window: WindowSpec,
        *,
        final: bool = False,
    ) -> dict[str, Any]:
        self._require_open()
        if not isinstance(window, WindowSpec):
            raise TypeError("window must be a WindowSpec")
        if float(window.end) <= self._last_window_end + 1e-9:
            raise ValueError("windows must advance monotonically without replay")
        started = time.perf_counter()
        try:
            stage_started = time.perf_counter()
            prompt = self.engine.prompt(
                window,
                tuple(self.engine.committer.pending),
            )
            prompt_sec = time.perf_counter() - stage_started
            observer_sec = 0.0
            stage_started = time.perf_counter()
            self.engine._record_attribution(
                {
                    "authoritative_state": (
                        self.engine.committer.attribution_snapshot()
                    ),
                    "pending_segments": [
                        item.to_dict()
                        for item in self.engine.committer.pending
                    ],
                    "prompt": prompt,
                    "stage": "engine_tick_start",
                    "tick": float(window.tick),
                    "window_end": float(window.end),
                    "window_start": float(window.start),
                }
            )
            observer_sec += time.perf_counter() - stage_started
            try:
                stage_started = time.perf_counter()
                decoded = self.engine.decode(window, prompt)
                decode_wall_sec = time.perf_counter() - stage_started
            except Exception as decode_exc:
                if not self.engine.continue_on_decode_error or (
                    self.engine.recoverable_decode_error_filter is not None
                    and not bool(
                        self.engine.recoverable_decode_error_filter(decode_exc)
                    )
                ):
                    raise
                update = self.engine.committer.update(
                    window.end,
                    [],
                    final=bool(final),
                )
                self.engine._record_attribution(
                    {
                        "error": repr(decode_exc),
                        "publication": [
                            item.to_dict() for item in update.committed
                        ],
                        "stage": "recoverable_decode_error",
                        "tick": float(window.tick),
                    }
                )
                metadata = {
                    "recoverable_decode_error": True,
                    "recoverable_decode_error_type": type(decode_exc).__name__,
                    "recoverable_decode_error_message": str(decode_exc),
                }
                if self.engine.recoverable_error_metadata is not None:
                    extra_metadata = self.engine.recoverable_error_metadata(
                        window,
                        decode_exc,
                    )
                    if not isinstance(extra_metadata, Mapping):
                        raise TypeError(
                            "recoverable_error_metadata must return a mapping"
                        )
                    metadata.update(dict(extra_metadata))
                decoded = DecodeOutput(
                    segments=[],
                    text="",
                    elapsed_sec=time.perf_counter() - started,
                    metadata=metadata,
                )
                wall_elapsed = max(
                    float(decoded.elapsed_sec),
                    time.perf_counter() - started,
                )
                self.wall_latencies.append(wall_elapsed)
                row = self.engine._row(
                    window,
                    decoded,
                    [],
                    update,
                    wall_elapsed,
                    extra_fields={"recoverable_decode_error": True},
                )
                self.rows.append(row)
                self._last_window_end = float(window.end)
                return row

            if not isinstance(decoded, DecodeOutput):
                raise ValidationError("decoder must return DecodeOutput")
            decoded_segments = tuple(decoded.segments)
            self.engine._validate_window_segments(
                decoded_segments,
                window,
                "decoded",
            )
            stage_started = time.perf_counter()
            mapped = list(
                self.engine.map_segments(decoded_segments, window)
            )
            self.engine._validate_window_segments(mapped, window, "mapped")
            map_sec = time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            self.engine._record_attribution(
                {
                    "decoded_segments": [
                        item.to_dict() for item in decoded_segments
                    ],
                    "mapped_segments": [item.to_dict() for item in mapped],
                    "metadata": dict(decoded.metadata or {}),
                    "stage": "mapped_decoder_snapshot",
                    "text": str(decoded.text),
                    "tick": float(window.tick),
                    "window_end": float(window.end),
                    "window_start": float(window.start),
                }
            )
            observer_sec += time.perf_counter() - stage_started
            self.observed_mapped_segment_count += sum(
                1
                for segment in mapped
                if segment.duration > 0.0
                and bool(segment.normalized_text)
            )
            self.observed_material_mapped_segment_count += sum(
                1
                for segment in mapped
                if segment.duration > 0.0
                and bool(segment.normalized_text)
                and self.engine.committer.is_material_observation(
                    segment,
                    source_tick=float(window.end),
                )
            )
            if (
                not math.isfinite(float(decoded.elapsed_sec))
                or float(decoded.elapsed_sec) < 0.0
            ):
                raise ValidationError(
                    "decoder elapsed_sec must be finite and non-negative"
                )
            stage_started = time.perf_counter()
            update = self.engine.committer.update(
                window.end,
                mapped,
                final=bool(final),
            )
            commit_sec = time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            self.engine._record_attribution(
                {
                    "authoritative_state": (
                        self.engine.committer.attribution_snapshot()
                    ),
                    "pending_segments": [
                        item.to_dict() for item in update.pending
                    ],
                    "publication": [
                        item.to_dict() for item in update.committed
                    ],
                    "published_until": float(update.published_until),
                    "stable_until": float(update.stable_until),
                    "stage": "atomic_publication",
                    "tick": float(window.tick),
                }
            )
            observer_sec += time.perf_counter() - stage_started
            extra_fields: dict[str, Any] = {}
            stage_started = time.perf_counter()
            if self.engine.after_update is not None:
                returned = self.engine.after_update(
                    window,
                    decoded,
                    tuple(mapped),
                    update,
                )
                if returned is not None:
                    if not isinstance(returned, Mapping):
                        raise TypeError(
                            "after_update must return a mapping or None"
                        )
                    extra_fields = dict(returned)
            after_update_sec = time.perf_counter() - stage_started
            wall_elapsed = max(
                float(decoded.elapsed_sec),
                time.perf_counter() - started,
            )
            self.wall_latencies.append(wall_elapsed)
            row = self.engine._row(
                window,
                decoded,
                mapped,
                update,
                wall_elapsed,
                extra_fields=extra_fields,
            )
            self.rows.append(row)
            total_wall_sec = time.perf_counter() - started
            accounted_sec = (
                prompt_sec
                + decode_wall_sec
                + map_sec
                + observer_sec
                + commit_sec
                + after_update_sec
            )
            self.engine._record_stage_profile(
                {
                    "tick": float(window.tick),
                    "prompt_sec": float(prompt_sec),
                    "observer_sec": float(observer_sec),
                    "decode_wall_sec": float(decode_wall_sec),
                    "decode_reported_sec": float(decoded.elapsed_sec),
                    "map_sec": float(map_sec),
                    "commit_sec": float(commit_sec),
                    "after_update_sec": float(after_update_sec),
                    "unattributed_sec": max(0.0, total_wall_sec - accounted_sec),
                    "total_wall_sec": float(total_wall_sec),
                }
            )
            self._last_window_end = float(window.end)
            return row
        except Exception as exc:
            self.rows.append(
                {
                    "tick": float(window.tick),
                    "window_start": float(window.start),
                    "window_end": float(window.end),
                    "status": "error",
                    "gate_status": "failed",
                    "error": repr(exc),
                    "wall_elapsed_sec": time.perf_counter() - started,
                }
            )
            raise

    def finalize(self, *, expected_ticks: int | None = None) -> EngineResult:
        self._require_open()
        expected = len(self.rows) if expected_ticks is None else int(expected_ticks)
        if expected <= 0:
            raise ValueError("expected_ticks must be positive")
        validation = self.engine._validation(
            self.rows,
            self.wall_latencies,
            expected,
            observed_mapped_segment_count=(
                self.observed_mapped_segment_count
            ),
            observed_material_mapped_segment_count=(
                self.observed_material_mapped_segment_count
            ),
        )
        self.validation = validation
        result = EngineResult(
            rows=list(self.rows),
            committed=list(self.engine.committer.committed),
            validation=validation,
        )
        if self.destination is not None:
            self.engine._write_outputs(self.destination, result)
        if validation["status"] != "passed":
            reasons = []
            if validation.get("execution_gate_pass") is False:
                reasons.append("execution")
            if validation.get("trace_gate_pass") is False:
                reasons.append("trace contract")
            if validation.get("final_content_gate_pass") is False:
                reasons.append("final content")
            if validation.get("recoverable_decode_error_gate_pass") is False:
                reasons.append("recoverable decode errors")
            detail = ": " + ", ".join(reasons) if reasons else ""
            error = ValidationError(
                "stream validation gate failed" + detail
            )
            self.fail(error, expected_ticks=expected)
            raise error
        self._closed = True
        return result

    def fail(self, exc: Exception, *, expected_ticks: int | None = None) -> None:
        if self._closed:
            return
        expected = (
            len(self.rows)
            if expected_ticks is None
            else max(0, int(expected_ticks))
        )
        if self.validation is None:
            validation: dict[str, Any] = {
                "status": "failed",
                "execution_gate_pass": False,
                "failed_ticks": [
                    row.get("tick")
                    for row in self.rows
                    if row.get("status") != "passed"
                ],
                "tick_count": len(self.rows),
                "expected_tick_count": expected,
                "duplicate_suppressed": (
                    self.engine.committer.duplicate_count
                ),
                "late_revision_count": (
                    self.engine.committer.late_drop_count
                ),
                "evidence_poor_turn_drop_count": int(
                    getattr(
                        self.engine.committer,
                        "evidence_poor_drop_count",
                        0,
                    )
                ),
                "evidence_poor_turn_drop_reasons": dict(
                    getattr(
                        self.engine.committer,
                        "evidence_poor_drop_reasons",
                        {},
                    )
                ),
                "observed_speech": (
                    self.observed_mapped_segment_count > 0
                ),
                "observed_material_content": (
                    self.observed_material_mapped_segment_count > 0
                ),
                "observed_mapped_segment_count": (
                    self.observed_mapped_segment_count
                ),
                "committed_segment_count": len(
                    self.engine.committer.committed
                ),
                "final_content_gate_pass": not (
                    self.observed_material_mapped_segment_count > 0
                    and len(self.engine.committer.committed) == 0
                ),
            }
        else:
            validation = dict(self.validation)
        validation["error"] = repr(exc)
        self.validation = validation
        if self.destination is not None:
            self.engine._write_json(
                self.destination / "stream_validation.json",
                validation,
            )
            self.engine._write_jsonl(
                self.destination / "results" / "ticks.jsonl",
                self.rows,
            )
            for stale in (
                self.destination / "hypothesis.json",
                self.destination
                / "global_embedding"
                / "hypothesis.json",
            ):
                stale.unlink(missing_ok=True)
        self._closed = True
