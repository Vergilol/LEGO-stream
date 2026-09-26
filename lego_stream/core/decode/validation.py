from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..types import Segment, stable_segment_id

class ValidationError(RuntimeError):
    pass


def _load(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid validation file: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValidationError(f"validation file must contain an object: {path}")
    return value


def _valid_hypothesis(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 1:
        return False
    item = value[0]
    if not isinstance(item, Mapping):
        return False
    segments = item.get("segments")
    if not isinstance(segments, list):
        return False
    for raw in segments:
        if not isinstance(raw, Mapping):
            return False
        try:
            segment = Segment.from_mapping(raw)
        except (KeyError, TypeError, ValueError):
            return False
        if not isinstance(raw.get("text", ""), str):
            return False
        if not str(segment.speaker).strip():
            return False
    return True


def _hypothesis_candidates(path: Path, mode: str | None) -> tuple[Path, ...]:
    """Return output paths allowed for a requested pipeline mode.

    A run directory can retain artifacts from more than one stage.  Selecting
    the first file that happens to exist lets a stale one-shot/global result
    masquerade as the current stream.  Callers that know their stage should
    pass ``mode`` so validation is tied to that stage's canonical artifact.
    """

    if mode is None:
        return (
            path / "hypothesis.json",
            path / "global_embedding" / "hypothesis.json",
        )
    normalized = str(mode).strip().lower()
    if normalized in {"incremental_asr", "stream", "baseline"}:
        return (path / "hypothesis.json",)
    if normalized in {"global_embedding", "global"}:
        return (path / "global_embedding" / "hypothesis.json",)
    raise ValueError("unknown publishability mode: %s" % mode)


def _has_no_violations(value: object) -> bool:
    """Accept both canonical lists and the runner's zero-count summary.

    The engine writes detailed violation lists, while the external panel
    validator stores the aggregate ``timestamp_violations`` as an integer.
    A zero count is semantically equivalent to an empty list; non-zero or
    malformed values must remain fail-closed.
    """

    if isinstance(value, int) and not isinstance(value, bool):
        return value == 0
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value) == 0
    return False


def _is_nonnegative_finite_observation(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value >= 0.0


def _final_content_gate_passes(value: Mapping[str, Any]) -> bool:
    """Validate the current observed-speech/final-content contract.

    These fields are part of the current protocol.  Missing or malformed
    evidence is deliberately non-publishable; mapped snapshot counts must not
    stand in for the committed final hypothesis count.
    """

    observed = value.get("observed_speech")
    observed_count = value.get("observed_mapped_segment_count")
    observed_material = value.get("observed_material_content", observed)
    committed_count = value.get("committed_segment_count")
    gate = value.get("final_content_gate_pass")
    if (
        not isinstance(observed, bool)
        or not isinstance(observed_material, bool)
        or not isinstance(gate, bool)
    ):
        return False
    if (
        isinstance(observed_count, bool)
        or not isinstance(observed_count, int)
        or observed_count < 0
        or isinstance(committed_count, bool)
        or not isinstance(committed_count, int)
        or committed_count < 0
    ):
        return False
    if observed != (observed_count > 0):
        return False
    if observed_material and not observed:
        return False
    expected_gate = (not observed_material) or committed_count > 0
    return gate is True and expected_gate


def is_publishable(sample_dir: str | Path, *, mode: str | None = None) -> bool:
    path = Path(sample_dir).expanduser().resolve()
    normalized_mode = "" if mode is None else str(mode).strip().lower()
    validation_path = path / "stream_validation.json"
    if not validation_path.is_file():
        return False
    try:
        value = _load(validation_path)
    except ValidationError:
        return False
    try:
        hypothesis_candidates = _hypothesis_candidates(path, mode)
    except ValueError:
        return False
    hypothesis_path = next(
        (item for item in hypothesis_candidates if item.is_file()), None
    )
    if hypothesis_path is None:
        return False
    try:
        hypothesis = json.loads(hypothesis_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not _valid_hypothesis(hypothesis):
        return False
    if not _final_content_gate_passes(value):
        return False
    failed_ticks = value.get("failed_ticks")
    if not isinstance(failed_ticks, Sequence) or isinstance(failed_ticks, (str, bytes)):
        return False
    expected = value.get("expected_tick_count")
    actual = value.get("tick_count")
    if (
        isinstance(expected, bool)
        or not isinstance(expected, int)
        or expected < 0
        or isinstance(actual, bool)
        or not isinstance(actual, int)
        or actual < 0
    ):
        return False
    for field in (
        "algorithmic_delay_sec",
        "pending_age_max_sec",
        "delta_first_publish_age_max_sec",
    ):
        if field in value and not _is_nonnegative_finite_observation(value[field]):
            return False
    if "latency" in value:
        latency = value["latency"]
        if (
            not isinstance(latency, Mapping)
            or "max_sec" not in latency
            or not _is_nonnegative_finite_observation(latency["max_sec"])
        ):
            return False
    if normalized_mode in {"global_embedding", "global"}:
        if value.get("speaker_assignment_gate_pass") is not True:
            return False
    if value.get("postprocess_gate_pass") is not True:
        return False
    return (
        value.get("status") == "passed"
        and len(failed_ticks) == 0
        and _has_no_violations(value.get("timestamp_violations"))
        and _has_no_violations(value.get("schedule_mismatches"))
        and expected == actual
    )


def require_publishable(sample_dir: str | Path, *, mode: str | None = None) -> None:
    if not is_publishable(sample_dir, mode=mode):
        raise ValidationError(f"stream output is not publishable: {sample_dir}")


def validate_trace_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_ticks: int | None = None,
) -> list[dict[str, Any]]:
    latest: dict[float, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValidationError(f"trace row {index} is not an object")
        try:
            tick = float(row["tick"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"trace row {index} has an invalid tick") from exc
        if not math.isfinite(tick) or tick < 0.0:
            raise ValidationError(f"trace row {index} has an invalid tick")
        if tick in latest:
            raise ValidationError(f"trace row {index} has duplicate tick {tick}")
        latest[tick] = dict(row)
    ordered_ticks = sorted(latest)
    ordered = [latest[key] for key in ordered_ticks]
    seen_delta: set[str] = set()
    for tick, row in zip(ordered_ticks, ordered):
        if row.get("status") != "passed":
            raise ValidationError(f"trace tick {row.get('tick')} is not passed")
        if row.get("gate_status") not in (None, "passed"):
            raise ValidationError(f"trace tick {row.get('tick')} failed ASR gate")
        if "delta_segments" not in row:
            # Unlike new engine rows, an older trace may carry no delta field at
            # all; skip it rather than fail the replay guard.
            continue
        raw_delta = row.get("delta_segments")
        if not isinstance(raw_delta, Sequence) or isinstance(raw_delta, (str, bytes)):
            raise ValidationError(f"trace tick {row.get('tick')} has an invalid delta")
        committed_alias = row.get("committed_segments")
        if committed_alias is not None and committed_alias != raw_delta:
            raise ValidationError(
                f"trace tick {row.get('tick')} delta/committed aliases disagree"
            )
        for index, raw in enumerate(raw_delta):
            if not isinstance(raw, Mapping):
                raise ValidationError(
                    f"trace tick {row.get('tick')} delta segment {index} is not an object"
                )
            try:
                segment = Segment.from_mapping(raw)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValidationError(
                    f"trace tick {row.get('tick')} delta segment {index} is invalid"
                ) from exc
            key = stable_segment_id(segment)
            if key in seen_delta:
                raise ValidationError(
                    f"trace tick {row.get('tick')} replayed delta segment {index}"
                )
            seen_delta.add(key)
            delay = tick - float(segment.end)
            if delay < 0.0:
                raise ValidationError(
                    "trace tick %s delta segment %d was published before its end"
                    % (row.get("tick"), index)
                )
    if expected_ticks is not None and len(ordered) != int(expected_ticks):
        raise ValidationError(
            f"trace tick count {len(ordered)} does not match expected {expected_ticks}"
        )
    return ordered


__all__ = [
    "ValidationError",
    "is_publishable",
    "require_publishable",
    "validate_trace_rows",
]
