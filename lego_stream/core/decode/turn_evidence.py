from __future__ import annotations

from dataclasses import dataclass
import math
import unicodedata
from typing import Sequence

from ..types import Segment


def _is_cjk(char: str) -> bool:
    codepoint = ord(char)
    return bool(
        0x3400 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _lexical_unit_count(text: str, *, stop_after: int = 2) -> int:
    units = 0
    in_token = False
    for char in str(text).casefold():
        category = unicodedata.category(char)
        if _is_cjk(char):
            if in_token:
                units += 1
                in_token = False
            units += 1
        elif category.startswith(("L", "N", "M")):
            in_token = True
        else:
            if in_token:
                units += 1
                in_token = False
        if units >= stop_after:
            return units
    if in_token:
        units += 1
    return units


@dataclass(frozen=True)
class CleanAcousticSpanSelection:
    segment: Segment | None
    rejection_reason: str | None = None


def select_clean_acoustic_span(
    segment: Segment,
    pool: Sequence[Segment],
    *,
    min_duration_sec: float,
) -> CleanAcousticSpanSelection:
    minimum = float(min_duration_sec)
    if not math.isfinite(minimum) or minimum < 0.0:
        raise ValueError("min_duration_sec must be finite and non-negative")
    if float(segment.duration) + 1e-9 < minimum:
        return CleanAcousticSpanSelection(None, "short")

    blocked = sorted(
        (
            max(float(segment.start), float(other.start)),
            min(float(segment.end), float(other.end)),
        )
        for other in pool
        if other is not segment
        and str(other.speaker) != str(segment.speaker)
        and min(float(segment.end), float(other.end))
        > max(float(segment.start), float(other.start)) + 1e-9
    )
    merged: list[tuple[float, float]] = []
    for start, end in blocked:
        if not merged or start > merged[-1][1] + 1e-9:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    if not merged:
        return CleanAcousticSpanSelection(segment)

    cursor = float(segment.start)
    candidates: list[tuple[float, float]] = []
    for start, end in merged:
        if start > cursor + 1e-9:
            candidates.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < float(segment.end) - 1e-9:
        candidates.append((cursor, float(segment.end)))
    if not candidates:
        return CleanAcousticSpanSelection(None, "overlap")

    start, end = max(
        candidates,
        key=lambda value: (value[1] - value[0], -value[0]),
    )
    if end - start + 1e-9 < minimum:
        return CleanAcousticSpanSelection(None, "overlap")
    return CleanAcousticSpanSelection(
        segment.with_updates(start=start, end=end)
    )


@dataclass(frozen=True)
class EvidencePoorTurnPolicy:
    acoustic_min_duration_sec: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(float(self.acoustic_min_duration_sec))
            or float(self.acoustic_min_duration_sec) <= 0.0
        ):
            raise ValueError("acoustic_min_duration_sec must be finite and positive")
    def is_evidence_poor(self, segment: Segment) -> bool:
        if not isinstance(segment, Segment):
            raise TypeError("is_evidence_poor requires a Segment")
        return bool(
            segment.source_tick is not None
            and float(segment.duration) + 1e-9
            < float(self.acoustic_min_duration_sec)
            and _lexical_unit_count(segment.text) == 1
        )

__all__ = [
    "CleanAcousticSpanSelection",
    "EvidencePoorTurnPolicy",
    "select_clean_acoustic_span",
]
