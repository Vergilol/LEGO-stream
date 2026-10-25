"""The parts of the ``tick`` and ``done`` frames derived from the engine's row.

* :func:`provisional_lanes`: the page's grey tail, less what is already
  published, split into lanes by the decoder's local speaker index, which is
  then dropped. It is renumbered every window, and a local ``S03`` would read as
  the published ``S03``.
* :func:`read_committed`: the committed transcript, read back from
  ``hypothesis.json`` so the page checks the stream against the artifact
  rather than against itself.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping

#: Boundary slack, matching the engine's own ``1e-9`` comparisons.
_EPSILON = 1e-9
#: What a cut may leave at the head of the remainder.
_SEPARATORS = " \t\n,.;:!?-–—…，。；：！？、"


def _number(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _alnum(text: str) -> list[int]:
    return [index for index, character in enumerate(text) if character.isalnum()]


def _letters(text: str) -> str:
    return "".join(text[index].lower()[:1] for index in _alnum(text))


def _wide(character: str) -> bool:
    # CJK is written without spaces, so a cut there is never mid-word.
    return unicodedata.east_asian_width(character) in ("W", "F")


def _cut(text: str, positions: list[int], letters: int) -> int | None:
    """Where ``text`` splits after ``letters`` alphanumerics; ``None`` mid-word."""

    if 0 < letters < len(positions):
        left, right = positions[letters - 1], positions[letters]
        if right == left + 1 and not (_wide(text[left]) or _wide(text[right])):
            return None
    return positions[letters - 1] + 1 if letters else 0


def _overlaps(start: float, end: float, turn: Mapping[str, Any]) -> bool:
    turn_start, turn_end = _number(turn.get("start")), _number(turn.get("end"))
    shared = min(end, turn_end) - max(start, turn_start)
    return shared > 0 and shared >= 0.5 * min(end - start, turn_end - turn_start)


def _restated_opening(text: str, turn_text: str) -> str | None:
    """``text`` less an opening that ends ``turn_text``, if one does.

    The opening must cover at least half of the turn and start and end on word
    boundaries, so one word in common, such as an answer to a question, is not
    taken for a restatement.
    """

    mine, theirs = _letters(text), _letters(turn_text)
    mine_at, theirs_at = _alnum(text), _alnum(turn_text)
    for size in range(min(len(mine), len(theirs)), (len(theirs) + 1) // 2 - 1, -1):
        if not size or not theirs.endswith(mine[:size]):
            continue
        cut = _cut(text, mine_at, size)
        if cut is not None and _cut(turn_text, theirs_at, len(theirs) - size) is not None:
            return text[cut:].lstrip(_SEPARATORS)
    return None


def _unsaid(
    start: float, end: float, text: str, published: list[Mapping[str, Any]]
) -> tuple[float, str] | None:
    """What of a tail segment no published turn over the same audio says.

    The decoder re-reads published audio, so its tail can restate a turn, often
    ending a little later, or run new words on after it. A restatement is
    dropped and a restated opening cut; other wording stays, as still undecided.
    """

    for turn in published:
        turn_text = str(turn.get("text") or "")
        mine, theirs = _letters(text), _letters(turn_text)
        if not mine or not theirs:
            continue
        if _overlaps(start, end, turn) and mine in theirs:
            return None
        turn_end = _number(turn.get("end"))
        if start < turn_end and end > _number(turn.get("start")):
            rest = _restated_opening(text, turn_text)
            if rest:
                text, start = rest, min(max(start, turn_end), end)
    return start, text


def provisional_lanes(
    row: Mapping[str, Any], published: Iterable[Mapping[str, Any]] = ()
) -> list[dict[str, Any]]:
    """The unpublished tail of ``row``'s decoder snapshot, one lane per voice.

    ``local_segments`` covers the whole window; only what ends past
    ``published_until``, less what the ``published`` turns already say, is
    provisional. Lanes are numbered by first appearance in the tail.
    """

    raw = row.get("local_segments") if isinstance(row, Mapping) else None
    frontier = _number(row.get("published_until")) if isinstance(row, Mapping) else 0.0
    tail = [
        segment
        for segment in (raw if isinstance(raw, list) else [])
        if isinstance(segment, Mapping)
        and _number(segment.get("end")) > frontier + _EPSILON
        and str(segment.get("text") or "").strip()
    ]
    earliest = min((_number(segment.get("start")) for segment in tail), default=0.0)
    # Only turns reaching into the tail can be restated there; in time order, so
    # a run-on spanning several turns is cut one opening at a time.
    nearby = sorted(
        (
            turn
            for turn in published
            if isinstance(turn, Mapping) and _number(turn.get("end")) > earliest
        ),
        key=lambda turn: _number(turn.get("start")),
    )
    kept = []
    for segment in tail:
        end = _number(segment.get("end"))
        unsaid = _unsaid(_number(segment.get("start")), end, str(segment.get("text")), nearby)
        if unsaid is not None:
            kept.append((unsaid[0], end, unsaid[1], str(segment.get("speaker") or "")))
    lanes: dict[str, int] = {}
    items = []
    for start, end, text, local in sorted(kept, key=lambda item: (item[0], item[1])):
        lane = lanes.setdefault(local, len(lanes))
        items.append({"start": start, "end": end, "text": text, "lane": lane})
    return items


def read_committed(stream_result: Any) -> list[dict[str, Any]] | None:
    """Segments from the run's ``hypothesis.json``, or ``None`` if unreadable.

    ``None`` is shown as "not checked", never as an empty transcript.
    """

    if not isinstance(stream_result, Mapping) or not stream_result.get("hypothesis"):
        return None
    try:
        payload = json.loads(
            Path(str(stream_result["hypothesis"])).read_text(encoding="utf-8")
        )
        segments = payload[0]["segments"]
    except (OSError, ValueError, LookupError, TypeError):
        return None
    return segments if isinstance(segments, list) else None


__all__ = ["provisional_lanes", "read_committed"]
