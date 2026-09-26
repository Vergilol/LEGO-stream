from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
import unicodedata
from typing import Any, Mapping, Optional


_SPEAKER_LABEL_RE = re.compile(r"[SG](\d+)\Z")


def canonical_speaker_label(value: object) -> str:
    """Return the canonical non-zero ``S`` ordinal for a model label."""

    if not isinstance(value, str):
        raise ValueError("speaker label must be an S/G ordinal")
    label = value.strip()
    match = _SPEAKER_LABEL_RE.fullmatch(label)
    if match is None:
        raise ValueError("speaker label must be an S/G ordinal: %r" % value)
    ordinal = int(match.group(1))
    if ordinal <= 0:
        raise ValueError("speaker label ordinal must be non-zero: %r" % value)
    return "S%02d" % ordinal


@dataclass(frozen=True)
class Segment:
    """A timestamped utterance in absolute audio time.

    The type deliberately contains no window/tick identity.  A window is an
    observation of an utterance, not the utterance's identity; this is what
    lets the commit ledger collapse later revisions safely.
    """

    start: float
    end: float
    speaker: str
    text: str
    source_tick: Optional[float] = None

    def __post_init__(self) -> None:
        if not (math.isfinite(float(self.start)) and math.isfinite(float(self.end))):
            raise ValueError("segment timestamps must be finite")
        if float(self.start) < 0.0 or float(self.end) < float(self.start):
            raise ValueError("segment timestamps must satisfy 0 <= start <= end")
        if not str(self.speaker).strip():
            raise ValueError("segment speaker must be non-empty")
        if self.source_tick is not None and not math.isfinite(float(self.source_tick)):
            raise ValueError("segment source_tick must be finite")

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))

    @property
    def normalized_text(self) -> str:
        chars = []
        for char in str(self.text).casefold():
            category = unicodedata.category(char)
            if char.isspace() or category.startswith(("P", "S")):
                continue
            chars.append(char)
        return "".join(chars)

    def with_updates(self, **updates: Any) -> "Segment":
        return replace(self, **updates)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Segment":
        return cls(
            start=float(value["start"]),
            end=float(value["end"]),
            speaker=str(value.get("speaker", "UNK")),
            text=str(value.get("text", "")),
            source_tick=(
                float(value["source_tick"])
                if value.get("source_tick") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "start": float(self.start),
            "end": float(self.end),
            "speaker": str(self.speaker),
            "text": str(self.text),
        }
        if self.source_tick is not None:
            result["source_tick"] = float(self.source_tick)
        return result


def temporal_intersection(left: Segment, right: Segment) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def temporal_iou(left: Segment, right: Segment) -> float:
    intersection = temporal_intersection(left, right)
    union = max(left.end, right.end) - min(left.start, right.start)
    return intersection / union if union > 0.0 else 0.0


def text_similarity(left: Segment, right: Segment) -> float:
    a = left.normalized_text
    b = right.normalized_text
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return min(len(a), len(b)) / max(len(a), len(b))
    # Character-level overlap is deterministic and works for both Chinese and
    # whitespace-delimited languages without bringing in a tokenizer.
    from difflib import SequenceMatcher

    return float(SequenceMatcher(None, a, b).ratio())


def revision_key(segment: Segment) -> str:
    text = re.sub(r"\s+", "", str(segment.text).casefold())
    return f"{round(float(segment.start), 2):.2f}:{round(float(segment.end), 2):.2f}:{text}"


def stable_segment_id(segment: Segment) -> str:
    """Return the protocol identity for a timestamped utterance.

    Speaker labels and source/window metadata are deliberately excluded.  The
    canonical payload is compact JSON so the same utterance survives snapshot
    reordering and speaker-only revisions.  ``Segment`` validates finite
    timestamps at construction time; ``allow_nan=False`` keeps this helper
    fail-closed if a non-canonical object is ever passed in.
    """

    if not isinstance(segment, Segment):
        raise TypeError("stable_segment_id requires a Segment")
    start = round(float(segment.start), 6)
    end = round(float(segment.end), 6)
    start = 0.0 if start == 0.0 else start
    end = 0.0 if end == 0.0 else end
    payload = json.dumps(
        [start, end, segment.normalized_text],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "seg-v1:" + hashlib.sha256(payload).hexdigest()


def acoustic_span_id(segment: Segment) -> str:
    """Return a human-readable acoustic span key for speaker continuity."""

    if not isinstance(segment, Segment):
        raise TypeError("acoustic_span_id requires a Segment")
    bin_sec = 0.10
    start_bin = int(math.floor(float(segment.start) / bin_sec + 0.5))
    end_bin = int(math.floor(float(segment.end) / bin_sec + 0.5))
    return f"span-v1:{start_bin}:{end_bin}"


__all__ = [
    "Segment",
    "canonical_speaker_label",
    "acoustic_span_id",
    "revision_key",
    "stable_segment_id",
    "temporal_intersection",
    "temporal_iou",
    "text_similarity",
]
