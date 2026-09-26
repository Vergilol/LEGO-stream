"""Passive observation hooks for the decode -> commit path.

Nothing here can influence a published record. The helpers exist so that an
external observer can be attached to audit the framework without becoming part
of it:

* ``safe_observer_record`` swallows every observer error, so a faulty observer
  degrades to silence rather than changing the stream;
* ``canonical_sha256`` digests commit-ledger snapshots, which is what lets the
  engine assert that recording an event did not mutate the state being recorded;
* ``parser_diagnostics`` re-derives the turns of a decoded window with an
  independent pass and raises if it disagrees with the live parser, catching a
  silently dropped turn at its source.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Mapping

from .decode.moss_adapter import _END_RE, _MARKER_RE, parse_transcript
from .types import canonical_speaker_label


def _canonical_primitive(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_primitive(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_primitive(item) for item in value]
    if isinstance(value, set):
        return sorted((_canonical_primitive(item) for item in value), key=repr)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("observation records cannot contain non-finite floats")
        return float(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return repr(value)


def canonical_sha256(value: object) -> str:
    """Digest nested primitives independently of key order or float format."""

    payload = json.dumps(
        _canonical_primitive(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parser_diagnostics(text: str) -> dict[str, object]:
    """Re-parse ``text`` independently and report what the live parser dropped.

    Raises ``AssertionError`` if the two passes disagree on the accepted turns:
    a decoded window that silently loses a turn is an attribution bug, and it is
    much cheaper to catch here than in a score three hours later.
    """

    if not isinstance(text, str):
        raise TypeError("transcript text must be a string")
    markers = list(_MARKER_RE.finditer(text))
    accepted: list[dict[str, object]] = []
    dropped: list[dict[str, object]] = []
    for index, marker in enumerate(markers):
        start = float(marker.group(1))
        raw_speaker = marker.group(2)
        speaker = canonical_speaker_label(raw_speaker)
        end_match = _END_RE.search(text, marker.end())
        if end_match is None:
            dropped.append({"marker_index": index, "reason": "missing_end_marker"})
            continue
        end = float(end_match.group(1))
        if end < start:
            dropped.append({"marker_index": index, "reason": "end_before_start"})
            continue
        body = text[marker.end() : end_match.start()].strip()
        if not body:
            dropped.append({"marker_index": index, "reason": "empty_body"})
            continue
        accepted.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": body,
            }
        )
    active = [
        {
            "start": item.start,
            "end": item.end,
            "speaker": item.speaker,
            "text": item.text,
        }
        for item in parse_transcript(text)
    ]
    if accepted != active:
        raise AssertionError("diagnostic parser diverged from active parser")
    counts = Counter(str(item["reason"]) for item in dropped)
    return {
        "active_segments": active,
        "accepted_count": len(active),
        "drop_reasons": dict(sorted(counts.items())),
        "dropped_markers": dropped,
        "marker_count": len(markers),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def safe_observer_record(observer: object, value: Mapping[str, object]) -> None:
    """Offer ``value`` to ``observer``, ignoring absence and every failure."""

    if observer is None:
        return
    record = getattr(observer, "record", None)
    if not callable(record):
        return
    try:
        record(value)
    except Exception:
        return


__all__ = [
    "canonical_sha256",
    "parser_diagnostics",
    "safe_observer_record",
]
