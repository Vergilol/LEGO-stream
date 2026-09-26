from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import re
from typing import Callable, Iterable, List, Optional, Sequence
import unicodedata

from ..identity.assignment import maximum_weight_assignment
from .turn_evidence import EvidencePoorTurnPolicy
from ..types import Segment, stable_segment_id, temporal_intersection, temporal_iou, text_similarity


OverlapDecider = Callable[[Segment, Segment], bool]


@dataclass(frozen=True)
class CommitUpdate:
    committed: tuple[Segment, ...]
    pending: tuple[Segment, ...]
    stable_until: float
    published_until: float
    forced_empty_frontier: Optional[float] = None
    pending_age_sec: float = 0.0
    delta_first_publish_age_sec: float = 0.0
    delayed_pending_publish_count: int = 0
    delayed_pending_publish_max_lag_sec: float = 0.0
    delta_frontier_backtrack_count: int = 0
    delta_frontier_backtrack_max_sec: float = 0.0


@dataclass
class _RevisionGroup:
    group_id: str
    active: Segment
    first_tick: float
    last_tick: float
    seen_ticks: set[float]
    consistent_observations: int = 1
    unchanged_observations: int = 1

    def update(
        self,
        segment: Segment,
        tick: float,
        *,
        content_consistent: bool,
    ) -> None:
        unchanged = bool(
            content_consistent
            and segment.normalized_text == self.active.normalized_text
            and float(segment.start) == float(self.active.start)
            and float(segment.end) == float(self.active.end)
        )
        self.active = segment
        self.last_tick = float(tick)
        self.seen_ticks.add(float(tick))
        if content_consistent:
            self.consistent_observations += 1
        else:
            self.consistent_observations = 1
        if unchanged:
            self.unchanged_observations += 1
        else:
            self.unchanged_observations = 1


class IncrementalCommitter:
    """Revision-aware append-only publisher for full-window decoders.

    Every tick is a new *snapshot*.  The mutable tail is aligned to existing
    revision groups and replaced in place; it is never unioned with the old
    snapshot.  A group can transition to committed at most once.  A late
    segment is admitted as a second speaker only when its time shift and
    extension provide independent overlap evidence; exact-span or contained
    late rewrites are suppressed unless ``allow_overlap`` or a same-snapshot
    triple-overlap witness explicitly supports a second track.
    Cross-label boundary continuation is disabled by default because an
    ellipsis at a request boundary is not enough evidence to distinguish a
    speaker turn from a local-label flip.  A caller with an independent speaker
    oracle may explicitly enable it.
    Terminal punctuation is a weak stability signal, not a publication timer.
    Stable-frontier and decoder-revision evidence determine publication; no
    elapsed-time deadline forces a segment into the immutable ledger.
    """

    def __init__(
        self,
        *,
        right_context_sec: float = 2.0,
        duplicate_iou: float = 0.25,
        duplicate_text_similarity: float = 0.55,
        allow_overlap: Optional[OverlapDecider] = None,
        allow_cross_speaker_boundary: bool = False,
        heuristic_overlap_min_sec: float = 0.20,
        heuristic_overlap_min_fraction: float = 0.15,
        empty_grace_ticks: int = 1,
        best_effort_publish_sec: Optional[float] = None,
        turn_evidence_policy: Optional[EvidencePoorTurnPolicy] = None,
        transition_observer: Optional[Callable[[dict[str, object]], None]] = None,
    ) -> None:
        if not math.isfinite(float(right_context_sec)) or right_context_sec < 0.0:
            raise ValueError("right_context_sec must be finite and non-negative")
        if best_effort_publish_sec is not None and (
            not math.isfinite(float(best_effort_publish_sec))
            or float(best_effort_publish_sec) <= 0.0
        ):
            raise ValueError(
                "best_effort_publish_sec must be a positive finite number or None"
            )
        if not 0.0 <= duplicate_iou <= 1.0:
            raise ValueError("duplicate_iou must be in [0, 1]")
        if not 0.0 <= duplicate_text_similarity <= 1.0:
            raise ValueError("duplicate_text_similarity must be in [0, 1]")
        if not math.isfinite(float(heuristic_overlap_min_sec)) or heuristic_overlap_min_sec < 0.0:
            raise ValueError("heuristic_overlap_min_sec must be finite and non-negative")
        if not 0.0 <= heuristic_overlap_min_fraction <= 1.0:
            raise ValueError("heuristic_overlap_min_fraction must be in [0, 1]")
        if isinstance(empty_grace_ticks, bool) or int(empty_grace_ticks) != empty_grace_ticks:
            raise ValueError("empty_grace_ticks must be an integer")
        if int(empty_grace_ticks) < 0:
            raise ValueError("empty_grace_ticks must be non-negative")
        if turn_evidence_policy is not None and not isinstance(
            turn_evidence_policy,
            EvidencePoorTurnPolicy,
        ):
            raise TypeError(
                "turn_evidence_policy must be an EvidencePoorTurnPolicy or None"
            )
        self.right_context_sec = float(right_context_sec)
        self.duplicate_iou = float(duplicate_iou)
        self.duplicate_text_similarity = float(duplicate_text_similarity)
        self.allow_overlap = allow_overlap
        self.allow_cross_speaker_boundary = bool(allow_cross_speaker_boundary)
        self.heuristic_overlap_min_sec = float(heuristic_overlap_min_sec)
        self.heuristic_overlap_min_fraction = float(heuristic_overlap_min_fraction)
        self.empty_grace_ticks = int(empty_grace_ticks)
        self.best_effort_publish_sec = (
            None if best_effort_publish_sec is None else float(best_effort_publish_sec)
        )
        self.turn_evidence_policy = turn_evidence_policy
        self.committed: List[Segment] = []
        self.pending: List[Segment] = []
        self._groups: List[_RevisionGroup] = []
        self._next_group = 1
        self.published_until = 0.0
        self.stable_until = 0.0
        self.duplicate_count = 0
        self.late_drop_count = 0
        self._last_window_end: Optional[float] = None
        self._committed_keys: set[str] = set()
        self._empty_snapshot_streak = 0
        self.forced_empty_frontier: Optional[float] = None
        self.pending_age_sec = 0.0
        self.delta_first_publish_age_sec = 0.0
        self.delta_first_publish_age_max_sec = 0.0
        self.delayed_pending_publish_count = 0
        self.delayed_pending_publish_max_lag_sec = 0.0
        self.delta_frontier_backtrack_count = 0
        self.delta_frontier_backtrack_max_sec = 0.0
        self.suffix_recovered_sec = 0.0
        self.best_effort_publish_count = 0
        self.best_effort_publish_max_lag_sec = 0.0
        self.late_drop_reasons: dict[str, int] = {}
        self.evidence_poor_drop_count = 0
        self.evidence_poor_drop_reasons: dict[str, int] = {}
        self._segment_key_cache: dict[Segment, str] = {}
        self._transition_observer = transition_observer

    @property
    def suppressed_count(self) -> int:
        return (
            self.duplicate_count
            + self.late_drop_count
            + self.evidence_poor_drop_count
        )

    def attribution_snapshot(self) -> dict[str, object]:
        return {
            "committed": [item.to_dict() for item in self.committed],
            "duplicate_count": int(self.duplicate_count),
            "evidence_poor_drop_count": int(self.evidence_poor_drop_count),
            "evidence_poor_drop_reasons": dict(
                sorted(self.evidence_poor_drop_reasons.items())
            ),
            "forced_empty_frontier": self.forced_empty_frontier,
            "groups": [
                {
                    "active": group.active.to_dict(),
                    "consistent_observations": int(group.consistent_observations),
                    "first_seen_tick": float(group.first_tick),
                    "group_id": str(group.group_id),
                    "last_seen_tick": float(group.last_tick),
                    "seen_ticks": sorted(float(item) for item in group.seen_ticks),
                    "unchanged_observations": int(group.unchanged_observations),
                }
                for group in self._groups
            ],
            "best_effort_publish_count": int(self.best_effort_publish_count),
            "best_effort_publish_max_lag_sec": float(
                self.best_effort_publish_max_lag_sec
            ),
            "late_drop_count": int(self.late_drop_count),
            "late_drop_reasons": dict(sorted(self.late_drop_reasons.items())),
            "last_window_end": self._last_window_end,
            "pending": [item.to_dict() for item in self.pending],
            "published_until": float(self.published_until),
            "stable_until": float(self.stable_until),
            "delayed_pending_publish_count": int(
                self.delayed_pending_publish_count
            ),
            "delayed_pending_publish_max_lag_sec": float(
                self.delayed_pending_publish_max_lag_sec
            ),
            "delta_frontier_backtrack_count": int(
                self.delta_frontier_backtrack_count
            ),
            "delta_frontier_backtrack_max_sec": float(
                self.delta_frontier_backtrack_max_sec
            ),
        }

    def _observe_transition(self, value: dict[str, object]) -> None:
        if self._transition_observer is None:
            return
        try:
            self._transition_observer(dict(value))
        except Exception:
            return

    def set_transition_observer(
        self,
        observer: Optional[Callable[[dict[str, object]], None]],
    ) -> None:
        self._transition_observer = observer

    def ingest(
        self,
        window_end: float,
        segments: Iterable[Segment],
        *,
        final: bool = False,
        source_tick: Optional[float] = None,
    ) -> List[Segment]:
        window_end = float(window_end)
        if not math.isfinite(window_end) or window_end < 0.0:
            raise ValueError("window_end must be finite and non-negative")
        if self._last_window_end is not None and window_end <= self._last_window_end + 1e-9:
            raise ValueError("window_end must be strictly monotonic")
        previous_published_until = float(self.published_until)
        self._last_window_end = window_end
        self.forced_empty_frontier = None
        self.pending_age_sec = 0.0
        self.delta_first_publish_age_sec = 0.0
        candidate_stable_until = window_end if final else max(0.0, window_end - self.right_context_sec)

        current: List[Segment] = []
        for item in segments:
            segment = item if isinstance(item, Segment) else Segment.from_mapping(item)  # type: ignore[arg-type]
            if segment.source_tick is None and source_tick is not None:
                segment = segment.with_updates(source_tick=float(source_tick))
            if segment.end <= segment.start or not segment.normalized_text:
                continue
            current.append(segment)
        raw_current = list(current)
        current = self._dedupe_snapshot(current)
        self._observe_transition(
            {
                "deduped_segments": [item.to_dict() for item in current],
                "final": bool(final),
                "input_segments": [item.to_dict() for item in raw_current],
                "reason": "snapshot_input",
                "stage": "revision_candidate",
                "tick": float(window_end),
            }
        )

        if current or final:
            self._empty_snapshot_streak = 0
        else:
            self._empty_snapshot_streak += 1

        previous_groups = list(self._groups)
        force_empty_frontier = False
        if not current and not final and previous_groups:
            # One empty decode is a transient omission, not proof of silence;
            # after that grace only the normal right-context frontier advances.
            # This is decoder-event evidence, not an elapsed-time deadline.
            force_empty_frontier = self._empty_snapshot_streak > self.empty_grace_ticks

        # An empty non-final snapshot is likewise too weak to advance the
        # coverage frontier on its own; a final one still flushes the tail.
        if current or final or force_empty_frontier:
            next_stable_until = max(self.stable_until, candidate_stable_until)
            if force_empty_frontier and next_stable_until > self.stable_until + 1e-9:
                self.forced_empty_frontier = float(next_stable_until)
            self.stable_until = next_stable_until

        # Align the complete snapshot to the mutable tail by temporal connected
        # components, not one-to-one greedy matching: a radical 1->N split or
        # N->1 merge leaves stale children in the ledger and publishes both
        # revisions.  A component with a changed cardinality replaces all its old
        # groups with the current snapshot's segments; a temporary omission (no
        # current neighbour) retains the old group until it becomes stable.
        current_sorted = sorted(current, key=lambda item: (item.start, item.end))
        current_ids = [f"C{index}" for index in range(len(current_sorted))]
        previous_ids = [group.group_id for group in previous_groups]
        consistency_scores: dict[tuple[str, str], float] = {}
        for current_index, segment in enumerate(current_sorted):
            for group in previous_groups:
                score = self._observation_consistency_score(segment, group.active)
                if score > 0.0:
                    consistency_scores[(current_ids[current_index], group.group_id)] = score
        consistency_assignment = maximum_weight_assignment(
            current_ids,
            previous_ids,
            consistency_scores,
        )
        consistent_predecessor_by_current = {
            current_index: previous_groups[previous_ids.index(previous_id)]
            for current_index, current_id in enumerate(current_ids)
            if (previous_id := consistency_assignment.get(current_id)) is not None
        }
        consistent_current_indices = set(consistent_predecessor_by_current)
        consistent_previous_ids = {
            group.group_id for group in consistent_predecessor_by_current.values()
        }
        old_to_current = [set() for _ in previous_groups]
        current_to_old = [set() for _ in current_sorted]
        for old_index, group in enumerate(previous_groups):
            for current_index, segment in enumerate(current_sorted):
                if self._revision_connected(segment, group.active):
                    old_to_current[old_index].add(current_index)
                    current_to_old[current_index].add(old_index)

        # Different-speaker containment is ambiguous for a 1->1 pair: it can
        # be an unrelated overlap track and must not erase the pending parent.
        # It becomes structural revision evidence only for a real cardinality
        # change (one parent split into multiple children, or multiple old
        # children merged into one current segment).
        for old_index, group in enumerate(previous_groups):
            if group.group_id in consistent_previous_ids:
                continue
            children = [
                current_index
                for current_index, segment in enumerate(current_sorted)
                if group.active.start <= segment.start + 1e-9
                and group.active.end >= segment.end - 1e-9
                and temporal_intersection(segment, group.active) > 0.0
            ]
            if len(children) >= 2:
                for current_index in children:
                    old_to_current[old_index].add(current_index)
                    current_to_old[current_index].add(old_index)
        for current_index, segment in enumerate(current_sorted):
            if current_index in consistent_current_indices:
                continue
            parents = [
                old_index
                for old_index, group in enumerate(previous_groups)
                if segment.start <= group.active.start + 1e-9
                and segment.end >= group.active.end - 1e-9
                and temporal_intersection(segment, group.active) > 0.0
            ]
            if len(parents) >= 2:
                for old_index in parents:
                    old_to_current[old_index].add(current_index)
                    current_to_old[current_index].add(old_index)

        def new_group(
            segment: Segment,
            *,
            current_index: Optional[int] = None,
            lineage_groups: Sequence[_RevisionGroup] = (),
        ) -> _RevisionGroup:
            predecessor = (
                consistent_predecessor_by_current.get(current_index)
                if current_index is not None
                else None
            )
            support_groups = list(lineage_groups)
            if predecessor is not None and predecessor not in support_groups:
                support_groups.append(predecessor)
            first_tick = (
                min(float(group.first_tick) for group in support_groups)
                if support_groups
                else window_end
            )
            seen_ticks = {float(window_end)}
            for support_group in support_groups:
                seen_ticks.update(float(tick) for tick in support_group.seen_ticks)
            group = _RevisionGroup(
                group_id="U%06d" % self._next_group,
                active=segment,
                first_tick=first_tick,
                last_tick=window_end,
                seen_ticks=seen_ticks,
                consistent_observations=(
                    predecessor.consistent_observations + 1
                    if predecessor is not None
                    else 1
                ),
            )
            self._next_group += 1
            return group

        next_groups: List[_RevisionGroup] = []
        visited_old: set[int] = set()
        visited_current: set[int] = set()
        absent_group_ids: set[str] = set()

        def append_component(old_indices: set[int], current_indices: set[int]) -> None:
            if not current_indices:
                for old_index in sorted(old_indices):
                    group = previous_groups[old_index]
                    absent_group_ids.add(group.group_id)
                    next_groups.append(group)
                return
            if len(old_indices) == 1 and len(current_indices) == 1:
                old_index = next(iter(old_indices))
                current_index = next(iter(current_indices))
                group = previous_groups[old_index]
                current_segment = current_sorted[current_index]
                boundary_continuation = self._is_boundary_continuation(
                    current_segment,
                    group.active,
                )
                if boundary_continuation:
                    current_segment = self._merge_boundary_continuation(
                        group.active,
                        current_segment,
                    )
                group.update(
                    current_segment,
                    window_end,
                    content_consistent=(
                        consistent_predecessor_by_current.get(current_index)
                        is group
                    )
                    or boundary_continuation,
                )
                next_groups.append(group)
                return
            # Radical split/merge: the current snapshot is authoritative for
            # the mutable component.  Do not retain any old child alongside it.
            # Stable split children are real decoder observations inherited
            # from the previous mutable component.  They may publish only as
            # whole current children when the current snapshot proves the child
            # boundary; never promote an old parent across a new child boundary.
            lineage_groups = tuple(previous_groups[index] for index in old_indices)
            for current_index in sorted(current_indices):
                current_segment = current_sorted[current_index]
                next_groups.append(
                    new_group(
                        current_segment,
                        current_index=current_index,
                        lineage_groups=lineage_groups,
                    )
                )

        for old_index in range(len(previous_groups)):
            if old_index in visited_old:
                continue
            if not old_to_current[old_index]:
                visited_old.add(old_index)
                append_component({old_index}, set())
                continue
            old_component = {old_index}
            current_component = set(old_to_current[old_index])
            changed = True
            while changed:
                changed = False
                for current_index in list(current_component):
                    for linked_old in current_to_old[current_index]:
                        if linked_old not in old_component:
                            old_component.add(linked_old)
                            changed = True
                for linked_old in list(old_component):
                    for current_index in old_to_current[linked_old]:
                        if current_index not in current_component:
                            current_component.add(current_index)
                            changed = True
            visited_old.update(old_component)
            visited_current.update(current_component)
            append_component(old_component, current_component)

        for current_index, segment in enumerate(current_sorted):
            if current_index in visited_current:
                continue
            next_groups.append(new_group(segment, current_index=current_index))
        material_groups: list[_RevisionGroup] = []
        for group in next_groups:
            if self._is_evidence_poor_turn(group.active):
                self._drop_evidence_poor_turn(
                    group,
                    reason="nonmaterial_single_unit",
                    tick=window_end,
                )
                continue
            material_groups.append(group)
        next_groups = material_groups
        confirmed_current_groups = [
            group
            for group in next_groups
            if group.group_id not in absent_group_ids
            and group.last_tick >= window_end - 1e-9
            and group.consistent_observations >= 2
        ]
        filtered_groups: list[_RevisionGroup] = []
        for group in next_groups:
            unconfirmed_successor = bool(
                group.group_id in absent_group_ids
                # Wording and local symbols may churn on every tick,
                # resetting ``consistent_observations`` even though the same
                # acoustic turn has been observed repeatedly.  Retire only a
                # genuinely one-off hypothesis when a confirmed successor
                # appears; repeated pending evidence remains recoverable.
                and len(group.seen_ticks) < 2
                and any(
                    self._has_successor_boundary(
                        group.active,
                        [successor.active],
                    )
                    for successor in confirmed_current_groups
                )
            )
            if unconfirmed_successor:
                continue
            filtered_groups.append(group)
        next_groups = filtered_groups
        self._groups = sorted(
            next_groups,
            key=lambda group: (group.active.start, group.active.end, group.group_id),
        )

        emitted: List[Segment] = []
        delayed_pending_keys: set[str] = set()
        remaining: List[_RevisionGroup] = []
        chronological_prefix_blocked = False
        # A repeated pending hypothesis may still be released later, but an
        # absent one cannot acquire evidence forever: keep a small causal history
        # window so disappeared groups do not accumulate for a whole meeting.
        # Final flushes bypass the bound, so EOS can still release.
        pending_retention_sec = max(4.0, 2.0 * self.right_context_sec)
        for group in self._groups:
            stale_pending = bool(
                not final
                and not force_empty_frontier
                and group.group_id in absent_group_ids
                and float(window_end) - float(group.last_tick)
                > pending_retention_sec + 1e-9
            )
            if stale_pending:
                self._observe_transition(
                    {
                        "reason": "stale_pending_expired_unconfirmed",
                        "segment": group.active.to_dict(),
                        "stage": "bounded_pending_history",
                        "tick": float(window_end),
                    }
                )
                continue
            if chronological_prefix_blocked:
                self._observe_transition(
                    {
                        "reason": "chronological_prefix_blocked",
                        "segment": group.active.to_dict(),
                        "stage": "mutable_turn_hypothesis",
                        "tick": float(window_end),
                    }
                )
                remaining.append(group)
                continue
            best_effort_due = False
            if (
                not final
                and not self._has_publication_evidence(
                group,
                current_sorted,
                confirmed_silence=force_empty_frontier,
                )
            ):
                # A rolling decoder may re-segment the same audio every window, so
                # a group can stay short of two content-consistent observations
                # for as long as it is visible and then be dropped by the stale
                # bound with its text never released.  When a deadline is set, a
                # repeatedly observed group that has crossed the causal stable
                # frontier is released from its latest observation instead of
                # being starved.  A bounded fallback, not a relaxation of the rule:
                # never before the deadline, never on unstable audio, and counted
                # so the share of best-effort output stays observable.
                best_effort_due = self._best_effort_publication_due(
                    group,
                    window_end=window_end,
                )
                if not best_effort_due:
                    self._observe_transition(
                        {
                            "reason": "insufficient_publication_evidence",
                            "segment": group.active.to_dict(),
                            "stage": "mutable_turn_hypothesis",
                            "tick": float(window_end),
                        }
                    )
                    remaining.append(group)
                    chronological_prefix_blocked = (
                        self._blocks_chronological_prefix(group.active)
                    )
                    continue
            active, suffix_handled = self._maybe_suffix_revision(
                group.active,
                current,
            )
            if suffix_handled and active is None:
                self.late_drop_count += 1
                self.late_drop_reasons["published_prefix_fully_covered"] = (
                    self.late_drop_reasons.get("published_prefix_fully_covered", 0) + 1
                )
                self._observe_transition(
                    {
                        "reason": "published_prefix_fully_covered",
                        "segment": group.active.to_dict(),
                        "stage": "prefix_or_suffix_trim_loss",
                        "tick": float(window_end),
                    }
                )
                continue
            if active is None:
                continue
            publishable, tail = self._publication_parts(
                active,
                stable_end=self.stable_until,
                final=final,
            )
            if publishable is None:
                if tail is not None:
                    group.active = tail
                    remaining.append(group)
                    chronological_prefix_blocked = self._blocks_chronological_prefix(
                        group.active
                    )
                continue
            publication_candidate = publishable
            publishable = self._trim_already_published_text_prefix(
                publication_candidate,
                emitted=emitted,
                drop_fully_covered=tail is not None,
            )
            if publishable is None:
                self.duplicate_count += 1
                self._observe_transition(
                    {
                        "reason": "published_text_prefix_fully_covered",
                        "segment": publication_candidate.to_dict(),
                        "stage": "false_duplicate_suppression",
                        "tick": float(window_end),
                    }
                )
                if tail is not None:
                    group.active = tail
                    remaining.append(group)
                continue
            if (
                not suffix_handled
                and len(publishable.normalized_text)
                < len(publication_candidate.normalized_text)
                and self._is_cross_label_prefix_revision(active, current)
            ):
                suffix_handled = True
            exact_key = self._segment_key(publishable)
            if exact_key in self._committed_keys:
                self.duplicate_count += 1
                self._observe_transition(
                    {
                        "reason": "exact_committed_duplicate",
                        "segment": publishable.to_dict(),
                        "stage": "false_duplicate_suppression",
                        "tick": float(window_end),
                    }
                )
                if tail is not None:
                    group.active = tail
                    remaining.append(group)
                continue
            if not suffix_handled and self._is_late_revision(
                active,
                current,
                allow_delayed_pending=(
                    len(group.seen_ticks) >= 2
                    and float(group.first_tick) < float(window_end) - 1e-9
                ),
            ):
                self.late_drop_count += 1
                self.late_drop_reasons["late_revision"] = (
                    self.late_drop_reasons.get("late_revision", 0) + 1
                )
                self._observe_transition(
                    {
                        "published_until": float(self.published_until),
                        "reason": "late_revision",
                        "segment": active.to_dict(),
                        "stage": "global_frontier_late_insertion_loss",
                        "tick": float(window_end),
                    }
                )
                if tail is not None:
                    group.active = tail
                    remaining.append(group)
                continue
            if any(self._is_duplicate(publishable, item) for item in emitted):
                self.duplicate_count += 1
                self._observe_transition(
                    {
                        "reason": "same_tick_emitted_duplicate",
                        "segment": publishable.to_dict(),
                        "stage": "false_duplicate_suppression",
                        "tick": float(window_end),
                    }
                )
                if tail is not None:
                    group.active = tail
                    remaining.append(group)
                continue
            emitted.append(publishable)
            if (
                len(group.seen_ticks) >= 2
                and float(group.first_tick) < float(window_end) - 1e-9
                and float(publishable.end) <= previous_published_until + 1e-9
            ):
                delayed_pending_keys.add(self._segment_key(publishable))
            if best_effort_due:
                self.best_effort_publish_count += 1
                self.best_effort_publish_max_lag_sec = max(
                    self.best_effort_publish_max_lag_sec,
                    float(window_end) - float(group.first_tick),
                )
            self._observe_transition(
                {
                    "best_effort": bool(best_effort_due),
                    "reason": "best_effort_deadline" if best_effort_due else "text_stable",
                    "segment": publishable.to_dict(),
                    "stage": "internal_stable_turn_event",
                    "tick": float(window_end),
                }
            )
            if tail is not None:
                group.active = tail
                remaining.append(group)

        emitted.sort(key=lambda item: (item.start, item.end, item.speaker, item.text))
        if emitted:
            latest_by_identity: dict[str, tuple[tuple[float, int], Segment]] = {}
            for position, item in enumerate(emitted):
                key = self._segment_key(item)
                order_tick = float(item.source_tick) if item.source_tick is not None else float(window_end)
                order = (order_tick, position)
                current = latest_by_identity.get(key)
                if current is None or order >= current[0]:
                    latest_by_identity[key] = (order, item)
            if len(latest_by_identity) != len(emitted):
                self.duplicate_count += len(emitted) - len(latest_by_identity)
            emitted = [
                item
                for _, item in sorted(
                    latest_by_identity.values(),
                    key=lambda pair: (pair[1].start, pair[1].end, pair[1].speaker, pair[1].text),
                )
            ]
        delayed_emitted = [
            item for item in emitted if self._segment_key(item) in delayed_pending_keys
        ]
        if delayed_emitted:
            self.delayed_pending_publish_count += len(delayed_emitted)
            self.delayed_pending_publish_max_lag_sec = max(
                self.delayed_pending_publish_max_lag_sec,
                max(float(window_end) - float(item.end) for item in delayed_emitted),
            )
        frontier_backtracks = [
            item
            for item in emitted
            if float(item.end) < previous_published_until - 1e-9
        ]
        if frontier_backtracks:
            self.delta_frontier_backtrack_count += len(frontier_backtracks)
            self.delta_frontier_backtrack_max_sec = max(
                self.delta_frontier_backtrack_max_sec,
                max(
                    previous_published_until - float(item.end)
                    for item in frontier_backtracks
                ),
            )
        if emitted:
            ledger_candidates = list(self.committed)
            ledger_candidates.extend(emitted)
            latest_committed_by_identity: dict[str, tuple[tuple[float, int], Segment]] = {}
            for position, item in enumerate(ledger_candidates):
                key = self._segment_key(item)
                order_tick = float(item.source_tick) if item.source_tick is not None else float(window_end)
                order = (order_tick, position)
                current = latest_committed_by_identity.get(key)
                if current is None or order >= current[0]:
                    latest_committed_by_identity[key] = (order, item)
            self.committed = [
                item
                for _, item in sorted(
                    latest_committed_by_identity.values(),
                    key=lambda pair: (pair[1].start, pair[1].end, pair[1].speaker, pair[1].text),
                )
            ]
            self._committed_keys = set(latest_committed_by_identity.keys())
        self._groups = remaining
        self.pending = [group.active for group in remaining]
        self.pending_age_sec = max(
            (
                max(0.0, window_end - float(group.active.end))
                for group in remaining
            ),
            default=0.0,
        )
        if emitted:
            self.published_until = max(self.published_until, max(item.end for item in emitted))
        self.delta_first_publish_age_sec = max(
            (max(0.0, window_end - float(item.end)) for item in emitted),
            default=0.0,
        )
        self.delta_first_publish_age_max_sec = max(
            self.delta_first_publish_age_max_sec,
            self.delta_first_publish_age_sec,
        )
        self._observe_transition(
            {
                "authoritative_state": self.attribution_snapshot(),
                "emitted": [item.to_dict() for item in emitted],
                "reason": "committer_update_complete",
                "stage": "committer_transition",
                "tick": float(window_end),
            }
        )
        return emitted

    def update(
        self,
        window_end: float,
        segments: Iterable[Segment],
        *,
        final: bool = False,
    ) -> CommitUpdate:
        emitted = self.ingest(
            window_end,
            segments,
            final=final,
            source_tick=window_end,
        )
        return CommitUpdate(
            committed=tuple(emitted),
            pending=tuple(self.pending),
            stable_until=self.stable_until,
            published_until=self.published_until,
            forced_empty_frontier=self.forced_empty_frontier,
            pending_age_sec=self.pending_age_sec,
            delta_first_publish_age_sec=self.delta_first_publish_age_sec,
            delayed_pending_publish_count=self.delayed_pending_publish_count,
            delayed_pending_publish_max_lag_sec=self.delayed_pending_publish_max_lag_sec,
            delta_frontier_backtrack_count=self.delta_frontier_backtrack_count,
            delta_frontier_backtrack_max_sec=self.delta_frontier_backtrack_max_sec,
        )

    def flush(self, window_end: float, segments: Iterable[Segment]) -> List[Segment]:
        return self.ingest(window_end, segments, final=True)

    def _revision_score(self, current: Segment, previous: Segment) -> float:
        iou = temporal_iou(current, previous)
        if self.allow_overlap is not None and self.allow_overlap(current, previous):
            return -1.0
        similarity = text_similarity(current, previous)
        if iou < self.duplicate_iou:
            if not (
                current.speaker == previous.speaker
                and temporal_intersection(current, previous) > 0.0
                and abs(current.start - previous.start) <= 0.25
            ):
                return -1.0
        if self._text_related(current, previous):
            score = 1.0 + iou + similarity
        elif current.speaker == previous.speaker and (
            iou >= 0.35 or abs(current.start - previous.start) <= 0.25
        ):
            # Same-speaker boundary rewrites often change most of the text
            # while retaining the start boundary.  Time continuity is the
            # stronger revision signal in that case.
            score = 0.75 + max(iou, 0.0)
        elif iou >= 0.75:
            # Keep a weak time-only path for a genuine decoder rewrite, but
            # let text-related candidates win when overlapping speakers are
            # present in the same mutable tail.
            score = iou * 0.5
        else:
            return -1.0
        if current.speaker == previous.speaker:
            score += 0.05
        return score

    def _revision_connected(self, current: Segment, previous: Segment) -> bool:
        """Whether two segments belong to one mutable revision component."""

        if self.allow_overlap is not None and self.allow_overlap(current, previous):
            return False
        if self._is_boundary_continuation(current, previous):
            return True
        if self._is_pending_prefix_expansion(current, previous):
            return True
        if temporal_intersection(current, previous) > 0.0:
            # Mere temporal intersection is not enough to merge two different
            # local symbols: a temporarily omitted overlap track would then
            # be swallowed by the surviving track.  A different-label pair is
            # still a revision when its text agrees, or when one interval
            # strictly contains the other (the normal split/merge shape).
            if current.speaker == previous.speaker or self._text_related(current, previous):
                return True
            # Preserve the established fail-safe for an exact/high-IoU
            # different-label rewrite.  A caller with a real overlap oracle
            # can keep such a pair separate through ``allow_overlap``.
            if temporal_iou(current, previous) >= 0.75:
                return True
            return False
        return self._revision_score(current, previous) >= 0.0

    @staticmethod
    def _observation_consistency_score(
        current: Segment,
        previous: Segment,
    ) -> float:
        """Score two snapshots of the same closed decoder segment.

        Local speaker labels are deliberately excluded: they remain mutable
        until global association.  A growing segment is never consistent,
        because its end boundary moves with the rolling request frontier.
        """

        start_delta = abs(float(current.start) - float(previous.start))
        end_delta = abs(float(current.end) - float(previous.end))
        if start_delta > 0.50 or end_delta > 0.25:
            return -1.0
        current_text = current.normalized_text
        previous_text = previous.normalized_text
        if not current_text or not previous_text:
            return -1.0
        length_ratio = min(len(current_text), len(previous_text)) / max(
            len(current_text),
            len(previous_text),
        )
        similarity = text_similarity(current, previous)
        if length_ratio < 0.85 or similarity < 0.82:
            return -1.0
        return 1.0 + similarity + length_ratio - start_delta - end_delta

    def _has_publication_evidence(
        self,
        group: _RevisionGroup,
        current: Sequence[Segment],
        *,
        confirmed_silence: bool,
    ) -> bool:
        """Whether mutable text is closed strongly enough for append-only output.

        The audio coverage frontier remains necessary but is not evidence that
        the decoder has finished revising text.  Normal publication therefore
        requires two content-consistent observations plus a later speech
        boundary.  A sole material terminal segment may also close when the
        current decoder snapshot repeats its text and timestamps exactly after
        the whole span has crossed the causal stable frontier.  Two consecutive
        empty decoder observations provide the equivalent silence event.  None
        of these rules is an elapsed-time deadline.
        """

        if group.active.source_tick is None:
            # ``ingest`` also supports authoritative, non-rolling segments.
            # Without snapshot provenance there is no decoder revision event
            # to confirm; callers that need streaming stability use ``update``
            # or pass ``source_tick`` explicitly.
            return True
        if confirmed_silence:
            return True
        if group.consistent_observations < 2:
            return False
        if self._has_successor_boundary(group.active, current):
            return True
        material_current = tuple(
            segment
            for segment in current
            if not self._is_evidence_poor_turn(segment)
        )
        return bool(
            group.unchanged_observations >= 2
            and len(material_current) == 1
            and material_current[0] is group.active
            and float(group.active.end) <= self.stable_until + 1e-9
            and self._has_sentence_terminal(group.active.text)
        )

    def _best_effort_publication_due(
        self,
        group: _RevisionGroup,
        *,
        window_end: float,
    ) -> bool:
        """Whether a starved group has reached its bounded release deadline.

        The deadline is measured from the tick that first observed the group,
        so the wait is bounded by configuration rather than by how long the
        decoder keeps re-segmenting.  Four conditions keep this narrow: the
        group must have been seen on at least two distinct ticks, so a single
        transient hypothesis is never released; its whole span must already sit
        behind the causal stable frontier, so audio the decoder may still
        extend is never published; it must carry text; and the decoder must
        have stopped reporting it, so a group that the normal consistency path
        could still confirm is never raced.  Sentence-terminal punctuation is
        deliberately not required, because a starved group is usually a
        mid-turn fragment and requiring closure would reinstate the starvation
        this path exists to bound.
        """

        if self.best_effort_publish_sec is None:
            return False
        if group.active.source_tick is None:
            return False
        if len(group.seen_ticks) < 2:
            return False
        if not group.active.normalized_text:
            return False
        if float(group.active.end) > self.stable_until + 1e-9:
            return False
        if float(group.last_tick) >= float(window_end) - 1e-9:
            # Still in the current snapshot, so the normal consistency path can
            # still confirm it; releasing here would race a group that was going
            # to converge, which measurably hurts stable-decoder configurations.
            # Only a group the decoder stopped reporting is released early.
            return False
        waited = float(window_end) - float(group.first_tick)
        return waited >= self.best_effort_publish_sec - 1e-9

    def _is_evidence_poor_turn(self, segment: Segment) -> bool:
        return bool(
            self.turn_evidence_policy is not None
            and self.turn_evidence_policy.is_evidence_poor(segment)
        )

    def is_material_observation(
        self,
        segment: Segment,
        *,
        source_tick: float,
    ) -> bool:
        """Whether one rolling decoder observation is material output content."""

        if not isinstance(segment, Segment):
            raise TypeError("is_material_observation requires a Segment")
        observed = (
            segment
            if segment.source_tick is not None
            else segment.with_updates(source_tick=float(source_tick))
        )
        return not self._is_evidence_poor_turn(observed)

    def _drop_evidence_poor_turn(
        self,
        group: _RevisionGroup,
        *,
        reason: str,
        tick: float,
    ) -> None:
        self.evidence_poor_drop_count += 1
        self.evidence_poor_drop_reasons[str(reason)] = (
            self.evidence_poor_drop_reasons.get(str(reason), 0) + 1
        )
        self._observe_transition(
            {
                "group_id": str(group.group_id),
                "reason": str(reason),
                "segment": group.active.to_dict(),
                "stage": "evidence_poor_turn_drop",
                "tick": float(tick),
            }
        )

    def _blocks_chronological_prefix(self, segment: Segment) -> bool:
        """Whether unresolved text owns material audio beyond the released frontier."""

        return float(segment.end) > self.published_until + 0.25

    def _has_successor_boundary(
        self,
        segment: Segment,
        current: Sequence[Segment],
    ) -> bool:
        for other in current:
            if other is segment:
                continue
            if self._observation_consistency_score(other, segment) > 0.0:
                continue
            if float(other.start) <= float(segment.start) + 0.25:
                continue
            if float(other.end) < float(segment.end) - 0.25:
                continue
            if self._text_related(other, segment):
                continue
            return True
        return False

    @staticmethod
    def _has_open_ending(text: str) -> bool:
        value = str(text).rstrip()
        return bool(re.search(r"(?:\.{2,}|[…⋯]+)$", value))

    @staticmethod
    def _has_sentence_terminal(text: str) -> bool:
        """Whether ``text`` ends a sentence with terminal punctuation."""

        value = str(text).rstrip()
        if not value:
            return False
        if re.search(r"(?:\.{2,}|[…⋯]+)$", value):
            return False
        value = re.sub(r'[\"\'”’)\]】』》〉〗〙〛]+$', "", value).rstrip()
        if not value:
            return False
        return value[-1] in "。！？!?；;."

    @staticmethod
    def _has_cjk_single_unit_prefix(text: str) -> bool:
        """Whether the first normalized unit is a CJK character.

        One-character boundary trimming is acceptable for CJK because one
        character can be a complete user-visible unit.  Do not apply the same
        heuristic to ASCII: trimming a single ``o`` caused ``on the menu`` to
        become ``n the menu``.
        """

        for char in str(text).casefold():
            category = unicodedata.category(char)
            if char.isspace() or category.startswith(("P", "S")):
                continue
            codepoint = ord(char)
            return bool(
                0x3400 <= codepoint <= 0x9FFF
                or 0xF900 <= codepoint <= 0xFAFF
            )
        return False

    def _with_publication_terminal(self, segment: Segment) -> Segment:
        """Add a visual sentence boundary to long open text at publication."""

        if segment.source_tick is None:
            return segment
        if len(segment.normalized_text) < 5:
            return segment
        if self._has_open_ending(segment.text) or self._has_sentence_terminal(
            segment.text
        ):
            return segment
        text = str(segment.text).rstrip()
        stripped = re.sub(r"[，,、；;：:]+$", "", text).rstrip()
        if not stripped:
            return segment
        terminal = "。" if re.search(r"[\u4e00-\u9fff]", stripped) else "."
        return segment.with_updates(text=stripped + terminal)

    def _publication_parts(
        self,
        segment: Segment,
        *,
        stable_end: float,
        final: bool,
    ) -> tuple[Optional[Segment], Optional[Segment]]:
        """Return the whole stable segment; do not split token-level tails."""

        if final:
            if float(stable_end) + 1e-9 < float(segment.end):
                return None, segment
            return self._with_publication_terminal(segment), None

        if float(stable_end) + 1e-9 < float(segment.end):
            return None, segment
        return self._with_publication_terminal(segment), None

    def _is_boundary_continuation(
        self,
        current: Segment,
        previous: Segment,
    ) -> bool:
        """Whether ``current`` is the suffix of a request-boundary cut.

        MOSS can stop an unfinished utterance at the exact end of one audio
        request and expose only its adjacent suffix in the next snapshot.  The
        two pieces need not overlap and the local symbol may change, so
        ordinary temporal revision matching cannot connect them.  Require all
        three causal signals before joining: the prior segment ended at its
        source tick, its text explicitly remained open, and the new segment
        starts immediately after that boundary.
        """

        if previous.source_tick is None or current.source_tick is None:
            return False
        if (
            current.speaker != previous.speaker
            and not self.allow_cross_speaker_boundary
        ):
            return False
        if float(current.source_tick) <= float(previous.source_tick) + 1e-9:
            return False
        if abs(float(previous.source_tick) - float(previous.end)) > 0.25:
            return False
        gap = float(current.start) - float(previous.end)
        if gap < -1e-6 or gap > 0.25:
            return False
        return self._has_open_ending(previous.text)

    @staticmethod
    def _merge_boundary_continuation(previous: Segment, current: Segment) -> Segment:
        prefix = re.sub(r"(?:\.{2,}|[…⋯]+)\s*$", "", str(previous.text)).rstrip()
        suffix = str(current.text).lstrip()
        separator = " " if (
            prefix
            and suffix
            and prefix[-1].isascii()
            and suffix[0].isascii()
            and prefix[-1].isalnum()
            and suffix[0].isalnum()
        ) else ""
        return current.with_updates(
            start=float(previous.start),
            text=prefix + separator + suffix,
        )

    def _text_related(self, left: Segment, right: Segment) -> bool:
        similarity = text_similarity(left, right)
        if similarity >= self.duplicate_text_similarity:
            return True
        a = left.normalized_text
        b = right.normalized_text
        return bool(a and b and (a in b or b in a))

    def _is_pending_prefix_expansion(self, current: Segment, previous: Segment) -> bool:
        """Return whether a mutable short prefix was expanded by this snapshot."""

        previous_norm = previous.normalized_text
        current_norm = current.normalized_text
        if (
            not previous_norm
            or not current_norm
            or len(previous_norm) > 6
            or len(current_norm) <= len(previous_norm) + 2
            or abs(float(current.start) - float(previous.start)) > 0.35
            or float(previous.end) > float(current.end) + 0.35
            or temporal_intersection(current, previous) <= 0.0
        ):
            return False
        prefix_len = max(
            len(previous_norm),
            min(len(current_norm), len(previous_norm) + 2),
        )
        prefix = current_norm[:prefix_len]
        common = 0
        for left, right in zip(previous_norm, prefix):
            if left != right:
                break
            common += 1
        if common >= max(2, len(previous_norm) - 1):
            return True
        return (
            common >= 2
            and SequenceMatcher(None, previous_norm, prefix, autojunk=False).ratio()
            >= 0.70
        )

    @staticmethod
    def _normalized_positions(text: str) -> tuple[str, list[int]]:
        normalized: list[str] = []
        positions: list[int] = []
        for index, char in enumerate(str(text).casefold()):
            category = unicodedata.category(char)
            if char.isspace() or category.startswith(("P", "S")):
                continue
            normalized.append(char)
            positions.append(index)
        return "".join(normalized), positions

    @staticmethod
    def _raw_cut_after_normalized_prefix(
        text: str,
        positions: Sequence[int],
        cut_norm: int,
    ) -> int:
        if cut_norm <= 0:
            return 0
        if cut_norm > len(positions):
            return len(str(text))
        return int(positions[cut_norm - 1]) + 1

    def _covered_prefix_len_from_recent_visible_text(
        self,
        *,
        candidate_norm: str,
        visible_norm: str,
        min_overlap: int,
    ) -> int:
        """Return candidate prefix length already proven by visible text.

        Segment timestamps from a rolling full-window decoder are only a coarse
        search hint: the same text can move by seconds when a later snapshot
        merges/splits utterances.  The append-only invariant is therefore
        text-first: the suffix of already visible user text may cover the
        prefix of a new candidate even when their segment boundaries no longer
        line up exactly.  Use exact suffix-prefix matching first, then a small
        approximate-alignment allowance for one/few character ASR revisions.
        """

        if not candidate_norm or not visible_norm:
            return 0
        visible_norm = visible_norm[-240:]
        max_overlap = min(len(candidate_norm), len(visible_norm))
        if max_overlap < min_overlap:
            return 0

        for cut_norm in range(max_overlap, min_overlap - 1, -1):
            if visible_norm.endswith(candidate_norm[:cut_norm]):
                return cut_norm

        for cut_norm in range(max_overlap, min_overlap - 1, -1):
            candidate_prefix = candidate_norm[:cut_norm]
            length_slack = max(1, min(8, int(math.ceil(cut_norm * 0.15))))
            longest_visible = min(len(visible_norm), cut_norm + length_slack)
            shortest_visible = max(min_overlap, cut_norm - length_slack)
            threshold = 0.90 if cut_norm < 20 else 0.84
            visible_lengths = sorted(
                {
                    cut_norm,
                    longest_visible,
                    shortest_visible,
                    min(len(visible_norm), cut_norm + max(1, length_slack // 2)),
                    max(min_overlap, cut_norm - max(1, length_slack // 2)),
                },
                reverse=True,
            )
            for visible_len in visible_lengths:
                visible_suffix = visible_norm[-visible_len:]
                if abs(len(visible_suffix) - len(candidate_prefix)) > length_slack:
                    continue
                tail_len = min(8, max(3, cut_norm // 5))
                if len(candidate_prefix) >= tail_len and len(visible_suffix) >= tail_len:
                    candidate_tail = candidate_prefix[-tail_len:]
                    visible_tail = visible_suffix[-tail_len:]
                    if candidate_tail != visible_tail and (
                        SequenceMatcher(
                            None,
                            candidate_tail,
                            visible_tail,
                            autojunk=False,
                        ).ratio()
                        < 0.86
                    ):
                        continue
                if (
                    SequenceMatcher(
                        None,
                        candidate_prefix,
                        visible_suffix,
                        autojunk=False,
                    ).ratio()
                    >= threshold
                ):
                    return cut_norm
        return 0

    def _suffix_text(
        self, candidate: Segment, previous: Segment, covered_end: float
    ) -> str:
        """Extract text after a published prefix of a same-speaker revision."""

        candidate_norm, positions = self._normalized_positions(candidate.text)
        previous_norm, _ = self._normalized_positions(previous.text)
        cut_norm = 0
        if candidate_norm and previous_norm:
            blocks = SequenceMatcher(
                None, previous_norm, candidate_norm, autojunk=False
            ).get_matching_blocks()
            prefix_limit = max(1, int(len(candidate_norm) * 0.65))
            usable = [
                block
                for block in blocks
                if block.size > 0 and block.b <= prefix_limit
            ]
            if usable:
                block = max(usable, key=lambda item: (item.b + item.size, item.size))
                cut_norm = min(len(candidate_norm), block.b + block.size)
        if cut_norm >= len(candidate_norm):
            return ""
        if cut_norm <= 0:
            return str(candidate.text)
        raw_cut = positions[cut_norm] if cut_norm < len(positions) else len(candidate.text)
        return str(candidate.text)[raw_cut:].lstrip()

    def _suffix_text_from_recent_published_context(
        self,
        candidate: Segment,
    ) -> Optional[str]:
        candidate_norm, positions = self._normalized_positions(candidate.text)
        if len(candidate_norm) < 8:
            return None
        visible_segments = [
            item
            for item in self.committed
            if float(item.end) >= float(candidate.start) - 30.0
            and float(item.start) <= float(candidate.end) + 0.75
        ]
        if not visible_segments:
            return None
        visible_segments.sort(key=lambda item: (item.start, item.end, item.speaker, item.text))
        visible_norm, _ = self._normalized_positions(
            "".join(item.text for item in visible_segments)
        )
        if not visible_norm:
            return None
        min_overlap = max(8, int(math.ceil(len(candidate_norm) * 0.35)))
        cut_norm = self._covered_prefix_len_from_recent_visible_text(
            candidate_norm=candidate_norm,
            visible_norm=visible_norm,
            min_overlap=min_overlap,
        )
        if cut_norm <= 0:
            latest_visible = max(
                visible_segments,
                key=lambda item: (float(item.end), float(item.start)),
            )
            if (
                latest_visible.speaker == candidate.speaker
                and float(candidate.start) <= float(latest_visible.end) + 0.75
                and float(candidate.end) > float(latest_visible.end) + 1e-9
            ):
                short_max_overlap = min(
                    len(candidate_norm),
                    len(visible_norm),
                    7,
                )
                for overlap_len in range(short_max_overlap, 1, -1):
                    if (
                        len(candidate_norm) - overlap_len >= 4
                        and visible_norm.endswith(candidate_norm[:overlap_len])
                    ):
                        cut_norm = overlap_len
                        break
        if cut_norm <= 0:
            return None
        if cut_norm >= len(candidate_norm):
            return ""
        raw_cut = positions[cut_norm] if cut_norm < len(positions) else len(candidate.text)
        suffix_text = str(candidate.text)[raw_cut:].lstrip()
        if not suffix_text.strip():
            return ""
        return suffix_text

    def _trim_adjacent_visible_cjk_prefix(
        self,
        candidate: Segment,
        *,
        emitted: Sequence[Segment],
        drop_fully_covered: bool,
    ) -> Optional[Segment]:
        candidate_norm, positions = self._normalized_positions(candidate.text)
        if not (
            emitted
            and len(candidate_norm) >= 2
        ):
            return candidate
        latest_emitted = max(
            emitted,
            key=lambda item: (float(item.end), float(item.start)),
        )
        latest_norm = latest_emitted.normalized_text
        if not (
            latest_emitted.speaker == candidate.speaker
            and len(latest_norm) == 1
            and self._has_cjk_single_unit_prefix(candidate.text)
            and candidate_norm.startswith(latest_norm)
            and 0.0 <= float(candidate.start) - float(latest_emitted.end) <= 2.0
        ):
            return candidate
        raw_cut = self._raw_cut_after_normalized_prefix(
            candidate.text,
            positions,
            len(latest_norm),
        )
        suffix_text = str(candidate.text)[raw_cut:].lstrip()
        if not suffix_text.strip():
            return None if drop_fully_covered else candidate
        suffix_start = max(float(candidate.start), float(latest_emitted.end))
        suffix_start = min(float(candidate.end), suffix_start)
        if suffix_start >= float(candidate.end) - 1e-9:
            return None if drop_fully_covered else candidate
        return candidate.with_updates(start=suffix_start, text=suffix_text)

    def _trim_already_published_text_prefix(
        self,
        candidate: Segment,
        *,
        emitted: Sequence[Segment],
        drop_fully_covered: bool,
    ) -> Optional[Segment]:
        """Drop a restarted text prefix that is already visible to the user.

        Rolling full-window decoders can restart a newly timestamped segment
        from an older textual prefix even when its start timestamp is already
        at the released audio frontier.  Temporal overlap alone cannot catch
        this shape.  Use a narrow append-only text invariant instead: if the
        prefix of the candidate is already covered by the suffix of recently
        emitted user text, publish only the remaining candidate suffix.
        """

        if emitted:
            latest_emitted = max(
                emitted,
                key=lambda item: (float(item.end), float(item.start)),
            )
            if (
                latest_emitted.speaker == candidate.speaker
                and float(candidate.start) >= float(latest_emitted.end) - 1e-9
            ):
                # Distinct, non-overlapping segments from the same complete
                # snapshot are causal acoustic evidence for a real repeated
                # phrase.  Text-only suffix/prefix matching cannot distinguish
                # that repetition from a rolling-window restart, so deleting
                # it would be an irreversible transcript mutation.
                return candidate

        candidate_norm, positions = self._normalized_positions(candidate.text)
        if len(candidate_norm) < 8:
            same_tick_trimmed = self._trim_adjacent_visible_cjk_prefix(
                candidate,
                emitted=emitted,
                drop_fully_covered=drop_fully_covered,
            )
            if same_tick_trimmed is None or same_tick_trimmed != candidate:
                return same_tick_trimmed
        lower_bound = float(candidate.start) - 30.0
        upper_bound = float(candidate.end) + max(2.0, self.right_context_sec)
        visible_segments = [
            item
            for item in [*self.committed, *emitted]
            if float(item.end) >= lower_bound
            and float(item.start) <= upper_bound + 1e-9
        ]
        if not visible_segments:
            return candidate
        visible_segments.sort(key=lambda item: (item.start, item.end, item.speaker, item.text))
        visible_text = "".join(item.text for item in visible_segments)
        visible_norm, _ = self._normalized_positions(visible_text)
        if not visible_norm:
            return candidate
        min_overlap = max(8, int(math.ceil(len(candidate_norm) * 0.35)))
        cut_norm = self._covered_prefix_len_from_recent_visible_text(
            candidate_norm=candidate_norm,
            visible_norm=visible_norm,
            min_overlap=min_overlap,
        )
        if cut_norm <= 0 and emitted:
            latest_emitted = max(
                emitted,
                key=lambda item: (float(item.end), float(item.start)),
            )
            latest_emitted_norm = latest_emitted.normalized_text
            if (
                latest_emitted.speaker == candidate.speaker
                and len(latest_emitted_norm) == 1
                and self._has_cjk_single_unit_prefix(candidate.text)
                and len(candidate_norm) >= 5
                and candidate_norm.startswith(latest_emitted_norm)
                and 0.0 <= float(candidate.start) - float(latest_emitted.end) <= 2.0
            ):
                cut_norm = 1
        if cut_norm <= 0:
            latest_visible = max(
                visible_segments,
                key=lambda item: (float(item.end), float(item.start)),
            )
            if (
                latest_visible.speaker == candidate.speaker
                and 0.0 <= float(candidate.start) - float(latest_visible.end) <= 2.0
            ):
                if len(candidate_norm) >= 5:
                    short_max_overlap = min(len(candidate_norm), len(visible_norm), 7)
                    for overlap_len in range(short_max_overlap, 1, -1):
                        if (
                            len(candidate_norm) - overlap_len >= 4
                            and visible_norm.endswith(candidate_norm[:overlap_len])
                        ):
                            cut_norm = overlap_len
                            break
                if (
                    cut_norm <= 0
                    and self._has_cjk_single_unit_prefix(candidate.text)
                    and visible_norm.endswith(candidate_norm[:1])
                ):
                    cut_norm = 1
        if cut_norm <= 0:
            return candidate
        if cut_norm >= len(candidate_norm):
            return None if drop_fully_covered else candidate

        raw_cut = self._raw_cut_after_normalized_prefix(
            candidate.text,
            positions,
            cut_norm,
        )
        suffix_text = str(candidate.text)[raw_cut:].lstrip()
        if not suffix_text.strip():
            return None if drop_fully_covered else candidate
        latest_visible = max(
            visible_segments,
            key=lambda item: (float(item.end), float(item.start)),
        )
        suffix_start = max(float(candidate.start), float(latest_visible.end))
        suffix_start = min(float(candidate.end), suffix_start)
        if suffix_start >= float(candidate.end) - 1e-9:
            return None if drop_fully_covered else candidate
        return candidate.with_updates(start=suffix_start, text=suffix_text)

    def _maybe_suffix_revision(
        self,
        candidate: Segment,
        current: Sequence[Segment] = (),
    ) -> tuple[Optional[Segment], bool]:
        """Trim a published prefix from a same-speaker late extension.

        The boolean distinguishes an unrelated candidate from a candidate in
        this revision shape whose whole text is already published.
        """

        overlaps = [
            previous
            for previous in self.committed
            if temporal_intersection(candidate, previous) > 0.0
            and (
                previous.speaker == candidate.speaker
                or self._is_cross_label_prefix_reference(
                    candidate,
                    previous,
                    current,
                )
            )
        ]
        if not overlaps:
            return candidate, False
        overlap_covered_end = max(float(item.end) for item in overlaps)
        covered_end = overlap_covered_end
        if candidate.end <= covered_end + 1e-9:
            return None, True
        if covered_end < self.published_until - 1e-9:
            # A later segment is already visible, so the revision cannot be
            # inserted before that frontier -- but the part starting after it
            # still can be published.  The fallback path would replay the whole
            # low-IoU same-speaker revision on real long meeting turns.
            if candidate.end <= self.published_until + 1e-9:
                return None, True
            covered_end = self.published_until
        short_prefix_reference = self._short_textual_prefix_reference(
            candidate,
            overlaps,
        )
        if short_prefix_reference is None and not any(
            len(previous.normalized_text) >= 4 and previous.duration >= 1.0
            for previous in overlaps
        ):
            # A one-word or punctuation-only rewrite has too little evidence
            # for synthetic token-time alignment unless it is an exact textual
            # prefix of the later same-speaker candidate.  Keep the established
            # fail-closed late-revision behavior for unrelated short rewrites.
            return candidate, False
        related = any(
            self._text_related(candidate, previous)
            or abs(float(candidate.start) - float(previous.start)) <= 0.35
            for previous in overlaps
        )
        same_speaker_overlap_revision = any(
            candidate.speaker == previous.speaker
            and temporal_intersection(candidate, previous)
            >= self.heuristic_overlap_min_sec - 1e-9
            for previous in overlaps
        )
        if candidate.start >= covered_end - 1e-9 or (
            not related and not same_speaker_overlap_revision
        ):
            return candidate, False
        latest_reference = max(overlaps, key=lambda item: (item.end, item.duration))
        if (
            short_prefix_reference is not None
            and float(short_prefix_reference.end)
            >= float(covered_end) - self.heuristic_overlap_min_sec - 1e-9
        ):
            reference = short_prefix_reference
        else:
            reference = latest_reference
        context_suffix_text = self._suffix_text_from_recent_published_context(candidate)
        if context_suffix_text is not None:
            suffix_text = context_suffix_text
        else:
            suffix_text = self._suffix_text(candidate, reference, covered_end)
        if not suffix_text.strip():
            return None, True
        # A few frames past the published boundary can be timestamp jitter
        # splitting a growing utterance into two children.  Keep the guard
        # narrow: larger extensions, or text that is not the published tail,
        # stay eligible for the normal late-revision/age gate.
        suffix_norm = self._normalized_positions(suffix_text)[0]
        reference_norm = reference.normalized_text
        candidate_norm = candidate.normalized_text
        if (
            float(candidate.end) - covered_end
            <= self.heuristic_overlap_min_sec + 1e-9
            and suffix_norm
            and (
                reference_norm.endswith(suffix_norm)
                or len(suffix_norm) <= 2
                or (
                    len(suffix_norm) >= 4
                    and self._text_related(candidate, reference)
                    and not candidate_norm.startswith(reference_norm)
                    and len(candidate_norm) <= len(reference_norm) + 2
                )
            )
        ):
            return None, True
        suffix_start = max(float(candidate.start), covered_end)
        suffix = candidate.with_updates(start=suffix_start, text=suffix_text)
        self.suffix_recovered_sec += max(0.0, suffix.end - suffix.start)
        return suffix, True

    def _is_cross_label_prefix_revision(
        self,
        candidate: Segment,
        current: Sequence[Segment],
    ) -> bool:
        """Confirm a label-flipped prefix revision after canonical trimming."""

        candidate_norm = candidate.normalized_text
        for previous in self.committed:
            previous_norm = previous.normalized_text
            if (
                previous.speaker == candidate.speaker
                or temporal_intersection(candidate, previous) <= 0.0
                or len(previous_norm) < 2
                or len(candidate_norm) <= len(previous_norm)
                or not self._text_related(candidate, previous)
                or float(candidate.end) <= float(previous.end) + 1e-9
            ):
                continue
            if self.allow_overlap is not None and self.allow_overlap(
                candidate,
                previous,
            ):
                continue
            if self._has_snapshot_overlap_evidence(
                candidate,
                current,
                previous=previous,
            ):
                continue
            return True
        return False

    def _short_textual_prefix_reference(
        self,
        candidate: Segment,
        overlaps: Sequence[Segment],
    ) -> Optional[Segment]:
        """Return a short committed prefix that exactly starts ``candidate``."""

        candidate_norm = candidate.normalized_text
        if not candidate_norm:
            return None
        eligible: list[Segment] = []
        for previous in overlaps:
            previous_norm = previous.normalized_text
            if len(previous_norm) < 2:
                continue
            if len(candidate_norm) <= len(previous_norm):
                continue
            if not candidate_norm.startswith(previous_norm):
                continue
            if float(previous.end) > float(candidate.end) + 1e-9:
                continue
            eligible.append(previous)
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda item: (len(item.normalized_text), float(item.end), item.duration),
        )

    def _is_cross_label_prefix_reference(
        self,
        candidate: Segment,
        previous: Segment,
        current: Sequence[Segment],
    ) -> bool:
        """Whether ``previous`` is a relabelled prefix of a longer turn.

        A rolling decoder can first expose a short prefix, then grow the same
        utterance under another local symbol while real backchannels
        overlap the middle of the turn.  Whole-segment similarity is weak in
        that shape because the candidate is much longer.  Compare only the
        equal-length text prefix, and keep explicit same-snapshot overlap
        evidence authoritative when it exists.
        """

        if previous.speaker == candidate.speaker:
            return False
        if self.allow_overlap is not None and self.allow_overlap(
            candidate,
            previous,
        ):
            return False
        if self._has_snapshot_overlap_evidence(
            candidate,
            current,
            previous=previous,
        ):
            return False
        if self._text_related(candidate, previous):
            # Existing whole-segment revision handling already owns this
            # shape, including its punctuation boundary semantics.  This
            # helper is only for a short prefix whose relation is hidden by a
            # much longer candidate tail.
            return False
        previous_norm = previous.normalized_text
        candidate_norm = candidate.normalized_text
        if (
            len(previous_norm) < 2
            or len(candidate_norm) <= len(previous_norm)
            or abs(float(candidate.start) - float(previous.start)) > 0.35
            or float(candidate.end) <= float(previous.end) + 1e-9
        ):
            return False
        candidate_prefix = candidate_norm[: len(previous_norm)]
        return bool(
            SequenceMatcher(
                None,
                previous_norm,
                candidate_prefix,
                autojunk=False,
            ).ratio()
            >= self.duplicate_text_similarity
        )

    def _segment_key(self, segment: Segment) -> str:
        cached = self._segment_key_cache.get(segment)
        if cached is None:
            cached = stable_segment_id(segment)
            self._segment_key_cache[segment] = cached
        return cached

    def _is_cross_speaker_same_text_revision(
        self,
        candidate: Segment,
        previous: Segment,
        current: Sequence[Segment] = (),
    ) -> bool:
        """Return whether a shifted same-text cross-label tail is a revision.

        The default committer has no independent overlap oracle.  A short tail
        with identical text and only timestamp/label drift is therefore treated
        as a rolling-window revision, unless the current snapshot contains the
        existing companion evidence used elsewhere to prove true overlap.
        """

        intersection = temporal_intersection(candidate, previous)
        if (
            candidate.speaker == previous.speaker
            or intersection <= 0.0
            or not candidate.normalized_text
            or candidate.normalized_text != previous.normalized_text
        ):
            return False
        if self.allow_overlap is not None and self.allow_overlap(candidate, previous):
            return False
        if self._has_snapshot_overlap_evidence(
            candidate,
            current,
            previous=previous,
        ):
            return False
        iou = temporal_iou(candidate, previous)
        return bool(
            len(candidate.normalized_text) <= 2
            or len(previous.normalized_text) <= 2
            or iou >= self.duplicate_iou
        )

    def _is_duplicate(self, candidate: Segment, previous: Segment) -> bool:
        iou = temporal_iou(candidate, previous)
        if iou < self.duplicate_iou:
            return False
        if self.allow_overlap is not None and self.allow_overlap(candidate, previous):
            return False
        if candidate.speaker != previous.speaker:
            return False
        return self._text_related(candidate, previous)

    def _is_late_revision(
        self,
        candidate: Segment,
        current: Sequence[Segment] = (),
        *,
        allow_delayed_pending: bool = False,
    ) -> bool:
        # Append-only publication has a temporal frontier: a newly discovered
        # segment ending behind it cannot be appended without reordering the
        # released stream, so it is a late revision even with no overlap.  A
        # causal overlap oracle or triple-overlap witness is the exception.
        fully_retro = candidate.end <= self.published_until + 1e-9
        admitted_overlap = False
        for previous in self.committed:
            iou = temporal_iou(candidate, previous)
            start_close = abs(float(candidate.start) - float(previous.start)) <= 0.25
            candidate_duration = max(float(candidate.end) - float(candidate.start), 0.0)
            overlap_fraction = (
                temporal_intersection(candidate, previous) / candidate_duration
                if candidate_duration > 0.0
                else 0.0
            )
            intersection = temporal_intersection(candidate, previous)
            snapshot_overlap_evidence = (
                self._has_snapshot_overlap_evidence(
                    candidate,
                    current,
                    previous=previous,
                )
                if intersection > 0.0
                else False
            )
            if self.allow_overlap is not None and self.allow_overlap(candidate, previous):
                admitted_overlap = True
                continue
            if (
                snapshot_overlap_evidence
                and candidate.speaker != previous.speaker
            ):
                if self._is_near_span_rewrite(candidate, previous):
                    return True
                # A contained/fully-retro candidate may still be a genuine
                # second track, including a repeated short backchannel.  The
                # independent third track is stronger evidence than lexical
                # equality, so preserve it instead of treating a same-text
                # cross-speaker candidate as a label rewrite.
                admitted_overlap = True
                continue
            if self._is_cross_speaker_same_text_revision(
                candidate,
                previous,
                current,
            ):
                return True
            if self._has_committed_companion_evidence(candidate, previous):
                admitted_overlap = True
                continue
            if (
                intersection > 0.0
                and intersection + 1e-9 >= self.heuristic_overlap_min_sec
                and candidate.speaker != previous.speaker
                and not snapshot_overlap_evidence
            ):
                if self._has_snapshot_boundary_shrink_evidence(
                    candidate,
                    current,
                    previous=previous,
                ) or self._has_short_committed_overlap_tail_evidence(
                    candidate,
                    previous,
                ):
                    continue
                # A low-IoU cross-label candidate that still intersects a
                # committed span is the common shape of a rolling-window
                # boundary/label revision.  Do not let the low-IoU fast path
                # bypass the overlap-evidence gate below.
                return True
            # A same-speaker candidate almost entirely inside a committed span is
            # a boundary/lexical revision even when ASR rewrites the text past
            # ``_text_related`` (``拔了你`` vs ``扒了皮``); an overlap oracle still wins.
            if (
                candidate.speaker == previous.speaker
                and intersection > 0.0
                and candidate.start < previous.end - 1e-9
                and candidate.end <= previous.end + 0.25
                and overlap_fraction >= 0.70
            ):
                if self.allow_overlap is None or not self.allow_overlap(candidate, previous):
                    return True
            # A boundary suffix drifting a few frames past the committed parent
            # but textually contained is a rewrite, not independent overlap.
            if (
                candidate.speaker == previous.speaker
                and intersection > 0.0
                and candidate.start < previous.end - 1e-9
                and candidate.end <= previous.end + 0.25
                and self._text_related(candidate, previous)
            ):
                if self.allow_overlap is None or not self.allow_overlap(candidate, previous):
                    return True
            # A same-speaker candidate wholly inside a published span is a
            # boundary rewrite even at small IoU.
            if (
                candidate.speaker == previous.speaker
                and intersection > 0.0
                and candidate.start < previous.end - 1e-9
                and candidate.end <= previous.end + 1e-9
            ):
                if self.allow_overlap is None or not self.allow_overlap(candidate, previous):
                    return True
            if iou < self.duplicate_iou and not (
                candidate.speaker == previous.speaker and start_close
            ):
                continue
            if self._looks_like_new_overlap(
                candidate,
                previous,
                snapshot_overlap_evidence=snapshot_overlap_evidence,
            ):
                continue
            # Without a confirmed-overlap decision, an overlapping candidate is
            # presumed to be a late window revision -- the safety default that
            # stops S01->S08 alias churn from appending the same span twice.
            return True
        # A segment that already lived in the mutable pending ledger is delayed
        # content, not a new insertion: retain it until the public frontier moves
        # past its timestamps.  Fresh fully-retro candidates stay fail-closed.
        if fully_retro and not admitted_overlap and not allow_delayed_pending:
            return True
        return False

    @staticmethod
    def _is_near_span_rewrite(candidate: Segment, previous: Segment) -> bool:
        """Return whether a late candidate is only a relabel/rewording of a span."""

        if temporal_intersection(candidate, previous) <= 0.0:
            return False
        candidate_duration = max(float(candidate.duration), 1e-9)
        previous_duration = max(float(previous.duration), 1e-9)
        return bool(
            abs(float(candidate.start) - float(previous.start)) <= 0.25
            and abs(float(candidate.end) - float(previous.end)) <= 0.25
            and min(candidate_duration, previous_duration)
            / max(candidate_duration, previous_duration)
            >= 0.65
        )

    def _looks_like_new_overlap(
        self,
        candidate: Segment,
        previous: Segment,
        *,
        snapshot_overlap_evidence: bool,
    ) -> bool:
        """Conservative default for a late second-speaker segment.

        A local label change with related text is a revision and must be
        suppressed.  A distinct, text-unrelated speaker is treated as a new
        overlap only when it shifts substantially and extends beyond the
        earlier span; exact or contained late candidates remain revisions.
        """

        if candidate.speaker == previous.speaker:
            return False
        if self._text_related(candidate, previous):
            return False
        # A shift and extension alone are not causal overlap evidence: a
        # stateless rolling decoder produces exactly that shape when it flips a
        # local symbol or moves a boundary.  Require an independently
        # overlapping track in the same complete snapshot.
        if not snapshot_overlap_evidence:
            return False
        if self._has_committed_companion_evidence(candidate, previous):
            return True
        # An exact same-span, different-label segment in a later snapshot is just
        # as likely a radical ASR/speaker-label rewrite, so the append-only
        # default is to suppress it.
        if candidate.end <= previous.end + 1e-6:
            return False
        shift = float(candidate.start) - float(previous.start)
        threshold = max(0.25, 0.25 * max(previous.duration, 0.0))
        return shift >= threshold and candidate.end > previous.start + threshold

    def _has_committed_companion_evidence(
        self,
        candidate: Segment,
        previous: Segment,
    ) -> bool:
        """Whether ``candidate`` can coexist with a committed companion.

        A rolling snapshot can first publish a short interjection, then later
        expose the longer utterance that overlapped it.  If the current
        snapshot also contains a segment corresponding to the already
        committed short span, the long unrelated segment is not a rewrite of
        that short span.  This is deliberately asymmetric: the short segment
        may cover only a tiny fraction of the long utterance.
        """

        if candidate.speaker == previous.speaker:
            return False
        if self._text_related(candidate, previous):
            return False
        intersection = temporal_intersection(candidate, previous)
        if intersection + 1e-9 < self.heuristic_overlap_min_sec:
            return False
        if candidate.duration <= previous.duration + 1e-9:
            return False
        return True

    def _has_snapshot_overlap_evidence(
        self,
        candidate: Segment,
        current: Sequence[Segment],
        *,
        previous: Optional[Segment] = None,
    ) -> bool:
        """Check for an independent, same-snapshot overlap track.

        The decoder returns a full rolling-window snapshot.  A later candidate
        that overlaps a committed segment is not enough to infer simultaneous
        speech: it may simply be a revision with a different local label.  An
        unambiguous heuristic requires a second current segment with a
        different label, unrelated text, and material mutual coverage.  This
        deliberately errs toward suppressing retro revisions; callers with a
        causal overlap detector can bypass it through ``allow_overlap``.
        """

        candidate_duration = max(float(candidate.end) - float(candidate.start), 0.0)
        if candidate_duration <= 0.0:
            return False
        for other in current:
            if other is candidate or other.speaker == candidate.speaker:
                continue
            overlap_start = max(float(candidate.start), float(other.start))
            overlap_end = min(float(candidate.end), float(other.end))
            if previous is not None:
                # The companion must coexist in the same time region where
                # the late candidate conflicts with the committed segment.
                # A genuine next turn that overlaps only after ``previous``
                # ended cannot turn the candidate into retro overlap proof.
                overlap_start = max(overlap_start, float(previous.start))
                overlap_end = min(overlap_end, float(previous.end))
            intersection = max(0.0, overlap_end - overlap_start)
            if intersection + 1e-9 < self.heuristic_overlap_min_sec:
                continue
            other_duration = max(float(other.end) - float(other.start), 0.0)
            if other_duration <= 0.0:
                continue
            if (
                previous is not None
                and candidate.speaker != previous.speaker
                and self._observation_consistency_score(other, previous) > 0.0
                and candidate.duration > previous.duration + 1e-9
            ):
                return True
            candidate_fraction = intersection / candidate_duration
            other_fraction = intersection / other_duration
            if (
                candidate_fraction + 1e-9 >= self.heuristic_overlap_min_fraction
                and other_fraction + 1e-9 >= self.heuristic_overlap_min_fraction
                and not self._text_related(candidate, other)
            ):
                return True
        return False

    def _has_snapshot_boundary_shrink_evidence(
        self,
        candidate: Segment,
        current: Sequence[Segment],
        *,
        previous: Segment,
    ) -> bool:
        """Whether a current snapshot proves the committed boundary shrank.

        A rolling decoder may first publish a segment whose end timestamp runs
        past the next speaker's true start.  Later, before the next speaker's
        long turn is publishable, the same snapshot can contain a shortened
        revision of the already-published segment that ends before the new
        turn.  In that shape the new turn is not a speaker-label rewrite of the
        old boundary; the earlier timestamp was simply too long.
        """

        if candidate.speaker == previous.speaker:
            return False
        if self._text_related(candidate, previous):
            return False
        if (
            temporal_intersection(candidate, previous)
            + 1e-9
            < self.heuristic_overlap_min_sec
        ):
            return False
        tail_after_previous = float(candidate.end) - max(
            float(candidate.start),
            float(previous.end),
        )
        if tail_after_previous + 1e-9 < max(1.0, self.heuristic_overlap_min_sec):
            return False
        boundary_tolerance = max(0.35, self.heuristic_overlap_min_sec)
        for other in current:
            if other == candidate:
                continue
            if other.speaker != previous.speaker:
                continue
            related_to_previous = self._text_related(other, previous)
            same_boundary = abs(float(other.start) - float(previous.start)) <= 0.35
            substantial_overlap = (
                temporal_iou(other, previous) >= 0.30
                or temporal_intersection(other, previous)
                + 1e-9
                >= self.heuristic_overlap_min_sec
            )
            if not (related_to_previous or same_boundary or substantial_overlap):
                continue
            if float(other.end) <= float(candidate.start) + boundary_tolerance:
                return True
        return False

    def _has_short_committed_overlap_tail_evidence(
        self,
        candidate: Segment,
        previous: Segment,
    ) -> bool:
        """Whether a short committed item should be treated as possible overlap.

        A very short backchannel can be published before a surrounding long
        turn is stable.  If the long turn is unrelated text and its usable tail
        is far beyond the short item, suppressing it creates a large deletion
        while preserving little anti-duplicate value.
        """

        if candidate.speaker == previous.speaker:
            return False
        if self._text_related(candidate, previous):
            return False
        intersection = temporal_intersection(candidate, previous)
        if intersection + 1e-9 < self.heuristic_overlap_min_sec:
            return False
        if float(previous.duration) > max(0.75, self.heuristic_overlap_min_sec * 3.0):
            return False
        if float(candidate.duration) < 3.0:
            return False
        tail_after_previous = float(candidate.end) - max(
            float(candidate.start),
            float(previous.end),
        )
        if tail_after_previous + 1e-9 < 2.0:
            return False
        overlap_fraction = intersection / max(float(candidate.duration), 1e-9)
        return overlap_fraction <= 0.25 + 1e-9

    def _snapshot_duplicate(self, candidate: Segment, previous: Segment) -> bool:
        if temporal_iou(candidate, previous) < self.duplicate_iou:
            return False
        if self.allow_overlap is not None and self.allow_overlap(candidate, previous):
            return False
        # Within one snapshot, distinct local labels are the only available
        # overlap evidence -- MOSS emits real simultaneous speech -- so only text
        # relation can show two segments are duplicate segmentations of one
        # utterance.  Never collapse simultaneous speakers on a shared word.
        if candidate.speaker != previous.speaker:
            return False
        return self._text_related(candidate, previous)

    def _dedupe_snapshot(self, candidates: Sequence[Segment]) -> List[Segment]:
        selected: List[Segment] = []
        for candidate in sorted(candidates, key=lambda item: (item.start, item.end, item.speaker, item.text)):
            replaced = False
            for index, previous in enumerate(selected):
                if not self._snapshot_duplicate(candidate, previous):
                    continue
                # Prefer the latest/longer textual revision inside one snapshot.
                if len(candidate.normalized_text) >= len(previous.normalized_text):
                    selected[index] = candidate
                replaced = True
                break
            if not replaced:
                selected.append(candidate)
        return selected


__all__ = ["CommitUpdate", "IncrementalCommitter"]
