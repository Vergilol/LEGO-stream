from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Sequence

from ..core.identity.assignment import maximum_weight_assignment
from ..core.types import (
    Segment,
    temporal_intersection,
    temporal_iou,
    text_similarity,
)


_SCORE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class LocalAlignmentEvidence:
    """Causal overlap evidence for one current local symbol."""

    local_symbol: str
    candidate_scores: tuple[tuple[str, float], ...]
    best_output_label: str | None
    best_score: float
    row_margin: float
    column_margin: float
    row_unique: bool
    column_unique: bool
    selected_output_label: str | None


def _deduplicate(segments: Sequence[Segment]) -> tuple[Segment, ...]:
    seen: set[tuple[float, float, str, str]] = set()
    values: list[Segment] = []
    for segment in segments:
        key = (
            float(segment.start),
            float(segment.end),
            str(segment.speaker),
            segment.normalized_text,
        )
        if key in seen:
            continue
        seen.add(key)
        values.append(segment)
    return tuple(values)


def _group_by_speaker(
    segments: Sequence[Segment],
) -> dict[str, tuple[Segment, ...]]:
    grouped: dict[str, list[Segment]] = defaultdict(list)
    for segment in _deduplicate(segments):
        grouped[str(segment.speaker)].append(segment)
    return {
        speaker: tuple(values)
        for speaker, values in sorted(grouped.items())
    }


class LocalSnapshotAligner:
    """Align rolling local symbols with a bounded resolved output snapshot."""

    def __init__(self, *, min_overlap_sec: float = 0.1) -> None:
        value = float(min_overlap_sec)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("min_overlap_sec must be finite and positive")
        self.min_overlap_sec = value
        self._resolved_snapshot: tuple[Segment, ...] = ()

    def update_resolved_snapshot(self, segments: Sequence[Segment]) -> None:
        """Replace, rather than extend, the currently usable output snapshot."""

        self._resolved_snapshot = _deduplicate(tuple(segments))

    def _edge_score(
        self,
        current: Sequence[Segment],
        resolved: Sequence[Segment],
    ) -> float | None:
        score = 0.0
        has_temporal_evidence = False
        for local_segment in current:
            best_pair_score = 0.0
            for output_segment in resolved:
                intersection = temporal_intersection(local_segment, output_segment)
                if intersection + _SCORE_TOLERANCE < self.min_overlap_sec:
                    continue
                has_temporal_evidence = True
                local_coverage = intersection / max(
                    float(local_segment.duration),
                    _SCORE_TOLERANCE,
                )
                pair_score = intersection * (
                    0.55 * local_coverage
                    + 0.25 * temporal_iou(local_segment, output_segment)
                    + 0.20 * text_similarity(local_segment, output_segment)
                )
                best_pair_score = max(best_pair_score, pair_score)
            score += best_pair_score
        return score if has_temporal_evidence and score > 0.0 else None

    def align(self, current_segments: Sequence[Segment]) -> dict[str, str]:
        mapping, _evidence = self.align_with_scores(current_segments)
        return mapping

    def align_with_scores(
        self,
        current_segments: Sequence[Segment],
    ) -> tuple[dict[str, str], dict[str, float]]:
        mapping, evidence, _diagnostics = self.align_with_evidence(
            current_segments
        )
        return mapping, evidence

    def align_with_evidence(
        self,
        current_segments: Sequence[Segment],
    ) -> tuple[
        dict[str, str],
        dict[str, float],
        dict[str, LocalAlignmentEvidence],
    ]:
        current_groups = _group_by_speaker(tuple(current_segments))
        resolved_groups = _group_by_speaker(self._resolved_snapshot)
        if not current_groups or not resolved_groups:
            return {}, {}, {}

        scores: dict[tuple[str, str], float] = {}
        for local_symbol, current in current_groups.items():
            for output_speaker, resolved in resolved_groups.items():
                score = self._edge_score(current, resolved)
                if score is not None:
                    scores[(local_symbol, output_speaker)] = score

        row_candidates: dict[str, tuple[tuple[str, float], ...]] = {}
        eligible_locals: list[str] = []
        for local_symbol in sorted(current_groups):
            candidates = tuple(
                sorted(
                    (
                        (output_speaker, score)
                        for (candidate_local, output_speaker), score in scores.items()
                        if candidate_local == local_symbol
                    ),
                    key=lambda item: (-item[1], item[0]),
                )
            )
            row_candidates[local_symbol] = candidates
            candidate_scores = [
                score
                for _output_speaker, score in candidates
            ]
            if not candidate_scores:
                continue
            best = max(candidate_scores)
            tied_best = sum(
                math.isclose(
                    score,
                    best,
                    rel_tol=_SCORE_TOLERANCE,
                    abs_tol=_SCORE_TOLERANCE,
                )
                for score in candidate_scores
            )
            if tied_best == 1:
                eligible_locals.append(local_symbol)

        mapping = (
            maximum_weight_assignment(
                eligible_locals,
                sorted(resolved_groups),
                scores,
            )
            if eligible_locals
            else {}
        )
        evidence = {
            local_symbol: scores[(local_symbol, output_speaker)]
            for local_symbol, output_speaker in mapping.items()
        }
        diagnostics: dict[str, LocalAlignmentEvidence] = {}
        for local_symbol, candidates in row_candidates.items():
            if not candidates:
                continue
            best_score = float(candidates[0][1])
            row_unique = len(candidates) == 1 or not math.isclose(
                best_score,
                float(candidates[1][1]),
                rel_tol=_SCORE_TOLERANCE,
                abs_tol=_SCORE_TOLERANCE,
            )
            best_output_label = str(candidates[0][0]) if row_unique else None
            row_runner_up = float(candidates[1][1]) if len(candidates) > 1 else 0.0
            row_margin = max(0.0, best_score - row_runner_up)

            column_margin = 0.0
            column_unique = False
            if best_output_label is not None:
                column_candidates = sorted(
                    (
                        (candidate_local, float(score))
                        for (candidate_local, output_speaker), score in scores.items()
                        if output_speaker == best_output_label
                    ),
                    key=lambda item: (-item[1], item[0]),
                )
                column_unique = bool(
                    column_candidates
                    and column_candidates[0][0] == local_symbol
                    and (
                        len(column_candidates) == 1
                        or not math.isclose(
                            column_candidates[0][1],
                            column_candidates[1][1],
                            rel_tol=_SCORE_TOLERANCE,
                            abs_tol=_SCORE_TOLERANCE,
                        )
                    )
                )
                if column_unique:
                    column_runner_up = (
                        column_candidates[1][1]
                        if len(column_candidates) > 1
                        else 0.0
                    )
                    column_margin = max(
                        0.0,
                        column_candidates[0][1] - column_runner_up,
                    )
            diagnostics[local_symbol] = LocalAlignmentEvidence(
                local_symbol=local_symbol,
                candidate_scores=tuple(
                    (str(output_speaker), float(score))
                    for output_speaker, score in candidates
                ),
                best_output_label=best_output_label,
                best_score=best_score,
                row_margin=float(row_margin),
                column_margin=float(column_margin),
                row_unique=bool(row_unique),
                column_unique=bool(column_unique),
                selected_output_label=mapping.get(local_symbol),
            )
        return mapping, evidence, diagnostics

__all__ = ["LocalAlignmentEvidence", "LocalSnapshotAligner"]
