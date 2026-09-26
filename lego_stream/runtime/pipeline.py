from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import resource
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from ..core.identity.assignment import maximum_weight_assignment
from ..core.decode.commit import IncrementalCommitter
from ..core.decode.engine import CommitUpdate, IncrementalEngine
from ..core.decode.moss_adapter import MossWindowDecoder
from ..core.identity.speakers import (
    CLEAN_CORE_POLICY,
    FIXED_COSINE_POLICY,
    MODEL_OWNED_IDENTITY_POLICIES,
    MULTIVIEW_IDENTITY_POLICIES,
    LEGO_POLICY,
    Observation,
    SPEAKER_MATCH_POLICIES,
    SpeakerRegistry,
    UnprofiledOutputLabelClaim,
)
from ..core.decode.turn_evidence import (
    EvidencePoorTurnPolicy,
    select_clean_acoustic_span,
)
from ..core.types import (
    Segment,
    stable_segment_id,
    temporal_intersection,
    temporal_iou,
    text_similarity,
)
from ..core.decode.validation import require_publishable
from .association import LocalAlignmentEvidence, LocalSnapshotAligner
from .causal import CausalSegmentStore
from ..core.identity.identity_evidence import EmbeddingViewSet


# The only association mode this release implements.
CAUSAL_EMBEDDING_MODE = "causal_embedding"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_ACTIVE_IDENTITY_EVENT_FIELDS = (
    "track_id",
    "status",
    "output_target",
    "publication_output_label",
    "policy",
    "decision_model",
    "event_id",
    "publication_reason",
    "publication_best_effort",
    "publication_score",
    "embedding_component_id",
    "global_identity_id",
    "published_speaker_id",
    "acoustic_component_id",
    "acoustic_kind",
    "acoustic_score",
    "acoustic_responsibility",
    "binding_action",
    "binding_output_label",
    "binding_utility",
    "binding_margin",
    "profile_update_component_id",
    "profile_update_output_label",
    "profile_update_weight",
    "profile_update_reason",
    "allocate_output_handle",
    "new_output_utility",
    "new_output_reason",
    "forced_constraint",
    "cross_output_update",
    "resolved_bridge",
    "claim_diagnostics",
    "cannot_link",
    "component_count",
    "binding_count",
    "update_status",
    "assessment",
    "acoustic_state_before",
    "acoustic_state_after",
    "source_keys",
)


def _centroid_cosines(
    vector: np.ndarray,
    sums: Mapping[str, np.ndarray],
    counts: Mapping[str, float],
) -> list[tuple[str, float]]:
    """Cosine of ``vector`` against the mean of every label seen so far."""

    out: list[tuple[str, float]] = []
    for label, count in counts.items():
        if count <= 0:
            continue
        centroid = sums[label] / count
        norm = float(np.linalg.norm(centroid))
        if norm > 0.0:
            out.append((label, float(np.dot(vector, centroid / norm))))
    return out


def _centroid_pick(
    candidates: Sequence[tuple[str, float]],
    label: str,
    min_gain: float,
    min_cosine: float,
) -> Optional[str]:
    """Rival label that beats ``label`` by ``min_gain`` and clears ``min_cosine``.

    A single label carries no information (nothing to compare against), and a
    segment whose own label is not yet represented has no baseline, so both
    cases decline.  This is the whole decision rule: no tuning per condition,
    no reference, no future audio.
    """

    if len(candidates) < 2:
        return None
    ranked = sorted(candidates, key=lambda item: -item[1])
    best, best_score = ranked[0]
    own = dict(candidates).get(label)
    if (
        best != label
        and own is not None
        and best_score - own >= min_gain
        and best_score >= min_cosine
    ):
        return best
    return None


def _speaker_binding_identity(segment: Segment) -> tuple[str, str, str]:
    return (
        float(segment.start).hex(),
        float(segment.end).hex(),
        str(segment.text),
    )


def _speaker_binding_key(segment: Segment) -> str:
    payload = json.dumps(
        _speaker_binding_identity(segment),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "speaker-bind-v1:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SpeakerBindingBatch:
    """Segments whose speaker is safe to publish at one streaming tick."""

    published: tuple[Segment, ...]
    assignments: tuple[dict[str, Any], ...]
    max_publication_age_sec: float = 0.0
    resolution_events: tuple[dict[str, Any], ...] = ()
    acoustic_association_events: tuple[dict[str, Any], ...] = ()
    decision_events: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class PublicationRecord:
    source_key: str
    output_label: str
    text: str
    start: float
    end: float
    tick: float


@dataclass
class PendingSpeakerDecision:
    """One stable text segment waiting for its immutable output label."""

    segment: Segment
    sequence: int
    segment_index: int
    first_round: int
    first_tick: float
    track_id: str | None = None
    output_target: str | None = None
    status: str = "wait"
    evidence: str = "unresolved"
    mode: str = "pending"
    score: float = 0.0
    confidence: float = 0.0
    policy: str | None = None
    context_output_label: str | None = None
    context_score: float = 0.0
    context_alignment: LocalAlignmentEvidence | None = None
    acoustic_output_scores: dict[str, float] = field(default_factory=dict)
    evidence_permissions: dict[str, bool] = field(default_factory=dict)
    evidence_strength: dict[str, float | int] | None = None


_SPEAKER_BINDING_EVENT_KEYS = (
    "speaker_assignments",
    "speaker_resolution_events",
    "acoustic_association_events",
    "speaker_decision_events",
)


@dataclass
class SpeakerBindingRuntime:
    """Mutable state shared by batch and online speaker publication.

    This is the single orchestration boundary between stable transcript turns
    and the identity filter.  It owns no identity arithmetic; it only tracks
    the causal embedding cutoff, per-tick diagnostics, and the immutable
    publication frontier.
    """

    cutoff_sec: float = 0.0
    published_until: float = 0.0
    events: dict[str, tuple[dict[str, Any], ...]] = field(
        default_factory=lambda: {
            key: () for key in _SPEAKER_BINDING_EVENT_KEYS
        }
    )

    def reset_events(self) -> None:
        for key in _SPEAKER_BINDING_EVENT_KEYS:
            self.events[key] = ()

    def event_metadata(self) -> Mapping[str, Any] | None:
        values = {
            key: list(self.events[key])
            for key in _SPEAKER_BINDING_EVENT_KEYS
            if self.events[key]
        }
        return values or None


class _MultiPolicySpeakerAssigner:
    """Compatibility engine for canonical and sealed historical policies."""

    def __init__(
        self,
        *,
        sample_id: object,
        embedding_provider: Any,
        embedding_view_provider: Any = None,
        policy: str = LEGO_POLICY,
        min_support_sec: float = 0.0,
        min_observations: int = 1,
        min_overlap_sec: float = 0.1,
        min_cosine: float = 0.70,
        different_local_merge_cosine: float = 0.90,
        same_local_split_cosine: float = 0.30,
        output_prefix: str = "S",
        max_core_samples_per_track: int = 120,
        core_min_duration_sec: float = 1.5,
        core_min_quality: float = 0.25,
        require_core_for_confirmation: bool = False,
        publish_pending_speakers: bool = False,
        best_effort_after_ticks: int = 1,
        embedding_dimension: int = 192,
        robust_exemplar_capacity: int = 4,
        risk_component_capacity: int = 8,
        risk_challenger_capacity: int = 2,
        risk_local_partition_provisional_enabled: bool = True,
        centroid_correction_enabled: bool = False,
        centroid_correction_min_gain: float = 0.05,
        centroid_correction_min_cosine: float = 0.50,
        centroid_correction_min_duration_sec: float = 1.5,
        centroid_embedding_provider: Any = None,
    ) -> None:
        if embedding_provider is not None and not callable(embedding_provider):
            raise ValueError("embedding_provider must be callable or None")
        if embedding_view_provider is not None and not callable(
            embedding_view_provider
        ):
            raise ValueError("embedding_view_provider must be callable or None")
        self.sample_id = sample_id
        self.embedding_provider = embedding_provider
        self.embedding_view_provider = embedding_view_provider
        self.policy = str(policy).strip()
        self.core_min_duration_sec = float(core_min_duration_sec)
        self.core_min_quality = float(core_min_quality)
        if (
            isinstance(best_effort_after_ticks, bool)
            or int(best_effort_after_ticks) != best_effort_after_ticks
            or int(best_effort_after_ticks) < 1
        ):
            raise ValueError("best_effort_after_ticks must be a positive integer")
        self.best_effort_after_ticks = int(best_effort_after_ticks)
        self.registry = SpeakerRegistry(
            min_support_sec=float(min_support_sec),
            min_observations=int(min_observations),
            min_overlap_sec=float(min_overlap_sec),
            min_cosine=float(min_cosine),
            different_local_merge_cosine=float(different_local_merge_cosine),
            same_local_split_cosine=float(same_local_split_cosine),
            output_prefix=output_prefix,
            max_core_samples_per_track=int(max_core_samples_per_track),
            core_min_duration_sec=float(core_min_duration_sec),
            core_min_quality=float(core_min_quality),
            require_core_for_confirmation=bool(require_core_for_confirmation),
            match_policy=self.policy,
            embedding_dimension=int(embedding_dimension),
            robust_exemplar_capacity=int(robust_exemplar_capacity),
            risk_component_capacity=int(risk_component_capacity),
            risk_challenger_capacity=int(risk_challenger_capacity),
            risk_local_partition_provisional_enabled=(
                risk_local_partition_provisional_enabled
            ),
        )
        self._assigned_by_id: dict[str, str] = {}
        self._resolved_context_segments: dict[str, Segment] = {}
        self._pending_decisions: dict[str, PendingSpeakerDecision] = {}
        self._context_aligner = LocalSnapshotAligner(
            min_overlap_sec=float(min_overlap_sec)
        )
        self._context_lineage_scores: dict[
            tuple[str, str], dict[str, float]
        ] = {}
        self._pending_context_aligner = LocalSnapshotAligner(
            min_overlap_sec=float(min_overlap_sec)
        )
        self._next_sequence = 0
        self._decision_round = 0
        self._published_keys: set[str] = set()
        self._publication_ledger: list[PublicationRecord] = []
        self._core_counters: dict[str, int] = {
            "eligible": 0,
            "skipped_short": 0,
            "skipped_overlap": 0,
            "called": 0,
            "available": 0,
            "accepted": 0,
            "rejected": 0,
        }
        if publish_pending_speakers:
            raise ValueError("pending speaker publication is not supported")
        # Per-label causal centroid correction.  This is a strictly downstream
        # layer: the registry's own decision in ``_assigned_by_id`` is never
        # touched (``_remember_assignment`` fails closed on any rewrite), so
        # every identity invariant and every audit digest of the acoustic model
        # keeps holding.  The override is consulted only where a label reaches
        # the user: at publication, in ``view``, and in ``is_fully_assigned``.
        self._centroid_correction_enabled = bool(centroid_correction_enabled)
        self._centroid_min_gain = float(centroid_correction_min_gain)
        self._centroid_min_cosine = float(centroid_correction_min_cosine)
        self._centroid_min_duration_sec = float(
            centroid_correction_min_duration_sec
        )
        if centroid_embedding_provider is not None and not callable(
            centroid_embedding_provider
        ):
            raise ValueError("centroid_embedding_provider must be callable or None")
        self._centroid_embedding_provider = centroid_embedding_provider
        if self._centroid_correction_enabled and centroid_embedding_provider is None:
            raise ValueError(
                "centroid correction requires a centroid_embedding_provider"
            )
        self._centroid_override: dict[str, str] = {}
        self._centroid_sums: dict[str, np.ndarray] = {}
        self._centroid_counts: dict[str, float] = {}
        self._centroid_counters: dict[str, int] = {
            "considered": 0,
            "skipped_short": 0,
            "skipped_no_embedding": 0,
            "scored": 0,
            "corrected": 0,
        }

    @property
    def core_counters(self) -> dict[str, int]:
        return dict(self._core_counters)

    def _identity_shadow_publication_snapshot(self) -> dict[str, object]:
        snapshot: dict[str, object] = {
            "assigned_by_id": [
                [key, value] for key, value in sorted(self._assigned_by_id.items())
            ],
            "resolved_context_segments": [
                {
                    "source_key": key,
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "speaker": segment.speaker,
                    "text": segment.text,
                }
                for key, segment in sorted(self._resolved_context_segments.items())
            ],
            "pending": [
                {
                    "source_key": key,
                    "sequence": int(item.sequence),
                    "segment_index": int(item.segment_index),
                    "first_round": int(item.first_round),
                    "first_tick": float(item.first_tick),
                    "track_id": item.track_id,
                    "output_target": item.output_target,
                    "status": item.status,
                    "evidence": item.evidence,
                    "mode": item.mode,
                    "score": float(item.score),
                    "confidence": float(item.confidence),
                    "segment": {
                        "start": float(item.segment.start),
                        "end": float(item.segment.end),
                        "speaker": item.segment.speaker,
                        "text": item.segment.text,
                    },
                }
                for key, item in sorted(self._pending_decisions.items())
            ],
            "next_sequence": int(self._next_sequence),
            "decision_round": int(self._decision_round),
            "published_keys": sorted(self._published_keys),
            "publication_ledger": [
                {
                    "source_key": item.source_key,
                    "output_label": item.output_label,
                    "text": item.text,
                    "start": float(item.start),
                    "end": float(item.end),
                    "tick": float(item.tick),
                }
                for item in self._publication_ledger
            ],
            "core_counters": dict(self._core_counters),
        }
        if self._centroid_override:
            # Only emitted once a correction actually fired, so a run with the
            # layer disabled keeps producing the historical digest byte for byte.
            snapshot["centroid_override"] = [
                [key, value]
                for key, value in sorted(self._centroid_override.items())
            ]
        return snapshot

    def final_authoritative_identity_digest(self) -> str:
        return self.registry.identity_shadow_authoritative_digest(
            publication_snapshot=self._identity_shadow_publication_snapshot()
        )

    def publication_ledger_snapshot(self) -> list[dict[str, object]]:
        value = self._identity_shadow_publication_snapshot().get(
            "publication_ledger",
            [],
        )
        return [dict(item) for item in value if isinstance(item, Mapping)]

    @property
    def embedding_call_count(self) -> int:
        return int(self._core_counters["called"])

    @property
    def embedding_available_count(self) -> int:
        return int(self._core_counters["available"])

    @property
    def published_segment_count(self) -> int:
        return len(self._published_keys)

    @property
    def publication_ledger(self) -> tuple[PublicationRecord, ...]:
        return tuple(self._publication_ledger)

    @staticmethod
    def _observation_quality(
        segment: Segment,
        batch: Sequence[Segment],
    ) -> float:
        duration = max(0.0, float(segment.duration))
        overlap_fraction = 0.0
        for other in batch:
            if other is segment or str(other.speaker) == str(segment.speaker):
                continue
            overlap = max(
                0.0,
                min(float(segment.end), float(other.end))
                - max(float(segment.start), float(other.start)),
            )
            overlap_fraction = max(overlap_fraction, overlap / max(1e-6, duration))
        overlap_score = max(0.0, 1.0 - overlap_fraction)
        # Purity/admissibility is intentionally separate from identity
        # strength. Duration and lexical richness are modeled by the derived
        # evidence/profile layer and must not collapse into this scalar.
        return float(overlap_score if duration > 0.0 else 0.0)

    def _embedding(self, segment: Segment) -> Optional[np.ndarray]:
        if self.embedding_provider is None:
            return None
        self._core_counters["called"] += 1
        try:
            value = self.embedding_provider(segment)
        except Exception:
            return None
        if value is None:
            return None
        vector = np.asarray(value, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if vector.ndim != 1 or norm <= 1e-12 or not np.all(np.isfinite(vector)):
            return None
        self._core_counters["available"] += 1
        return vector / norm

    def _embedding_views(
        self,
        segment: Segment,
        *,
        source_key: str,
    ) -> Optional[EmbeddingViewSet]:
        if self.embedding_view_provider is None:
            return None
        self._core_counters["called"] += 1
        value = self.embedding_view_provider(segment, str(source_key))
        if value is None:
            raise RuntimeError("robust embedding view extraction returned no evidence")
        if not isinstance(value, EmbeddingViewSet):
            raise TypeError("embedding_view_provider must return EmbeddingViewSet")
        if value.source_key != str(source_key):
            raise ValueError("embedding view source key does not match observation")
        self._core_counters["available"] += 1
        return value

    def _centroid_correction(self, segment: Segment, assigned: str) -> str:
        """Label to publish for ``segment`` after per-label centroid review.

        Called exactly once per span, in publication order, at the moment the
        span is published.  Evidence is the running mean of every label over the
        spans published strictly earlier, so the decision is causal by
        construction and costs zero extra ticks.  The registry's ``assigned`` is
        left untouched; only the returned label differs, and only when a rival
        label both wins by ``min_gain`` and is itself close enough
        (``min_cosine``) to be a credible owner.
        """

        if not self._centroid_correction_enabled:
            return assigned
        self._centroid_counters["considered"] += 1
        duration = float(segment.end) - float(segment.start)
        if duration < self._centroid_min_duration_sec:
            self._centroid_counters["skipped_short"] += 1
            return assigned
        try:
            value = self._centroid_embedding_provider(segment)
        except Exception:
            value = None
        vector = None
        if value is not None:
            candidate = np.asarray(value, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(candidate))
            if norm > 1e-12 and np.all(np.isfinite(candidate)):
                vector = candidate / norm
        if vector is None:
            self._centroid_counters["skipped_no_embedding"] += 1
            return assigned
        published = assigned
        if self._centroid_counts:
            self._centroid_counters["scored"] += 1
            rival = _centroid_pick(
                _centroid_cosines(
                    vector, self._centroid_sums, self._centroid_counts
                ),
                assigned,
                self._centroid_min_gain,
                self._centroid_min_cosine,
            )
            if rival is not None:
                self._centroid_override[_speaker_binding_key(segment)] = rival
                self._centroid_counters["corrected"] += 1
                published = rival
        # Accumulate under the registry's label, not the corrected one: the
        # statistic being estimated is "what the acoustic model calls this
        # speaker", and feeding corrections back in makes an early mistake
        # self-reinforcing (measured to regress on the far-field condition).
        previous = self._centroid_sums.get(assigned)
        self._centroid_sums[assigned] = (
            vector if previous is None else previous + vector
        )
        self._centroid_counts[assigned] = (
            self._centroid_counts.get(assigned, 0.0) + 1.0
        )
        return published

    def _published_label(self, key: str, assigned: str) -> str:
        return self._centroid_override.get(key, assigned)

    @property
    def centroid_counters(self) -> dict[str, int]:
        return dict(self._centroid_counters)

    def _core_segment(
        self,
        segment: Segment,
        pool: Sequence[Segment],
    ) -> Optional[Segment]:
        extraction_min_duration_sec = min(0.4, self.core_min_duration_sec)
        selection = select_clean_acoustic_span(
            segment,
            pool,
            min_duration_sec=extraction_min_duration_sec,
        )
        if selection.segment is None:
            reason = str(selection.rejection_reason or "overlap")
            self._core_counters[f"skipped_{reason}"] += 1
            return None
        self._core_counters["eligible"] += 1
        return selection.segment

    def _record_core_outcome(
        self,
        *,
        track_id: Optional[str],
        source_key: str,
        embedding: Optional[np.ndarray],
    ) -> None:
        if embedding is None or track_id is None:
            return
        if self.policy in MULTIVIEW_IDENTITY_POLICIES:
            return
        if self.registry.core_contains(track_id, source_key):
            self._core_counters["accepted"] += 1
        else:
            self._core_counters["rejected"] += 1

    def _align_context(
        self,
        context_segments: Sequence[Segment] | None,
    ) -> tuple[
        dict[str, str],
        dict[str, float],
        dict[str, LocalAlignmentEvidence],
    ]:
        context = tuple(context_segments or ())
        if not context:
            self._context_aligner.update_resolved_snapshot(())
            self._context_lineage_scores = {}
            return {}, {}, {}
        visible_until = max(float(segment.end) for segment in context)
        resolved: list[Segment] = []
        for key, segment in tuple(self._resolved_context_segments.items()):
            overlaps_context = any(
                temporal_intersection(segment, current) > 0.0
                for current in context
            )
            if overlaps_context:
                resolved.append(segment)
                continue
            if float(segment.end) <= visible_until:
                self._resolved_context_segments.pop(key, None)
        self._context_aligner.update_resolved_snapshot(resolved)
        lineage_scores: dict[tuple[str, str], dict[str, float]] = {}
        for current in context:
            for source_key, output_segment in self._resolved_context_segments.items():
                intersection = temporal_intersection(current, output_segment)
                if intersection <= 0.0:
                    continue
                key = (str(current.speaker), str(output_segment.speaker))
                by_source = lineage_scores.setdefault(key, {})
                by_source[str(source_key)] = by_source.get(str(source_key), 0.0) + float(
                    intersection
                )
        self._context_lineage_scores = lineage_scores
        return self._context_aligner.align_with_evidence(context)

    def _align_pending_context(
        self,
        context_segments: Sequence[Segment] | None,
    ) -> tuple[dict[str, str], dict[str, float]]:
        context = tuple(context_segments or ())
        pending_snapshot: list[Segment] = []
        for key, pending in self._pending_decisions.items():
            if key in self._assigned_by_id:
                continue
            track_id = pending.track_id
            if track_id is None or self.registry.output_label_for_track(str(track_id)) is not None:
                continue
            pending_snapshot.append(
                pending.segment.with_updates(speaker=str(track_id))
            )
        self._pending_context_aligner.update_resolved_snapshot(pending_snapshot)
        if not context or not pending_snapshot:
            return {}, {}
        return self._pending_context_aligner.align_with_scores(context)

    def _event_unprofiled_claims(
        self,
        source_keys: Sequence[str],
    ) -> tuple[UnprofiledOutputLabelClaim, ...]:
        records_by_track: dict[
            str,
            list[tuple[PendingSpeakerDecision, LocalAlignmentEvidence]],
        ] = defaultdict(list)
        for key in source_keys:
            pending = self._pending_decisions.get(str(key))
            if pending is None or pending.track_id is None:
                continue
            track_id = str(pending.track_id)
            if self.registry.output_label_for_track(track_id) is not None:
                continue
            if (
                self.policy not in MODEL_OWNED_IDENTITY_POLICIES
                and not self.registry.core_contains(track_id, str(key))
            ):
                continue
            alignment = pending.context_alignment
            if alignment is None:
                continue
            records_by_track[track_id].append((pending, alignment))

        tracks_by_local_target: dict[tuple[str, str], set[str]] = defaultdict(set)
        for track_id, records in records_by_track.items():
            for _pending, alignment in records:
                if alignment.best_output_label is None:
                    continue
                tracks_by_local_target[
                    (alignment.local_symbol, alignment.best_output_label)
                ].add(track_id)

        claims: list[UnprofiledOutputLabelClaim] = []
        for track_id in sorted(records_by_track):
            records = records_by_track[track_id]
            alignments = [alignment for _pending, alignment in records]
            targets = {
                alignment.best_output_label
                for alignment in alignments
                if alignment.best_output_label is not None
            }
            conflict_reason: str | None = None
            if len(targets) > 1:
                conflict_reason = "conflicting_context_targets"
            elif any(
                alignment.best_output_label is not None
                and len(
                    tracks_by_local_target[
                        (
                            alignment.local_symbol,
                            alignment.best_output_label,
                        )
                    ]
                )
                != 1
                for alignment in alignments
            ):
                conflict_reason = "ambiguous_source_tracks_for_context"
            target_output_label = next(iter(targets)) if len(targets) == 1 else None
            exemplar = max(
                alignments,
                key=lambda item: (item.best_score, item.local_symbol),
            )
            selected = bool(
                target_output_label is not None
                and all(
                    alignment.selected_output_label == target_output_label
                    for alignment in alignments
                )
            )
            lineage_scores: dict[str, float] = {}
            if target_output_label is not None:
                for alignment in alignments:
                    for source_key, score in self._context_lineage_scores.get(
                        (str(alignment.local_symbol), str(target_output_label)),
                        {},
                    ).items():
                        lineage_scores[source_key] = lineage_scores.get(source_key, 0.0) + float(score)
            ordered_lineages = sorted(
                lineage_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
            ancestor_lineage = (
                ordered_lineages[0][0]
                if ordered_lineages
                and (
                    len(ordered_lineages) == 1
                    or ordered_lineages[0][1] > ordered_lineages[1][1] + 1e-12
                )
                else None
            )
            if (
                conflict_reason is None
                and selected
                and target_output_label is not None
                and ancestor_lineage is None
            ):
                conflict_reason = "ambiguous_publication_lineage"
            claims.append(
                UnprofiledOutputLabelClaim(
                    source_track_id=track_id,
                    target_output_label=target_output_label,
                    context_score=min(
                        float(alignment.best_score)
                        for alignment in alignments
                    ),
                    row_margin=min(
                        float(alignment.row_margin)
                        for alignment in alignments
                    ),
                    column_margin=min(
                        float(alignment.column_margin)
                        for alignment in alignments
                    ),
                    context_candidates=exemplar.candidate_scores,
                    row_unique=all(
                        alignment.row_unique for alignment in alignments
                    ),
                    column_unique=all(
                        alignment.column_unique for alignment in alignments
                    ),
                    selected_by_context_assignment=selected,
                    conflict_reason=conflict_reason,
                    ancestor_lineage=ancestor_lineage,
                )
            )
        return tuple(claims)

    def _context_may_own_observation(
        self,
        observation: Observation,
        *,
        pending_track_id: str | None = None,
    ) -> bool:
        if self.policy in MODEL_OWNED_IDENTITY_POLICIES:
            return False
        if self.policy != CLEAN_CORE_POLICY or observation.embedding is None:
            return True
        permissions = self.registry.observation_permissions(observation)
        if permissions is None or not permissions.profile:
            return True
        if pending_track_id is None:
            return False
        return (
            self.registry.profile_state(str(pending_track_id)).value
            == "unseeded"
        )

    @staticmethod
    def _pending_revision_score(current: Segment, previous: Segment) -> float | None:
        """Score a one-to-one revision of text that is still withheld."""

        current_text = current.normalized_text
        previous_text = previous.normalized_text
        if min(len(current_text), len(previous_text)) < 4:
            return None
        similarity = text_similarity(current, previous)
        iou = temporal_iou(current, previous)
        length_ratio = min(len(current_text), len(previous_text)) / max(
            len(current_text),
            len(previous_text),
        )
        if similarity < 0.82 or length_ratio < 0.80 or iou < 0.75:
            return None
        return float(similarity + length_ratio + iou)

    def _pending_revision_matches(
        self,
        segments: Sequence[Segment],
    ) -> dict[str, tuple[str, float]]:
        """Return mutual unique one-to-one matches to withheld segments."""

        current = {
            _speaker_binding_key(segment): segment
            for segment in segments
            if _speaker_binding_key(segment) not in self._pending_decisions
            and _speaker_binding_key(segment) not in self._published_keys
        }
        pending = {
            key: decision.segment
            for key, decision in self._pending_decisions.items()
            if key not in self._assigned_by_id
            and key not in self._published_keys
            and decision.track_id is not None
        }
        if not current or not pending:
            return {}
        scores = {
            (current_key, pending_key): score
            for current_key, current_segment in current.items()
            for pending_key, pending_segment in pending.items()
            if (
                score := self._pending_revision_score(
                    current_segment,
                    pending_segment,
                )
            )
            is not None
        }
        if not scores:
            return {}

        current_keys = sorted(current)
        pending_keys = sorted(pending)
        assignment = maximum_weight_assignment(current_keys, pending_keys, scores)
        reverse_assignment = maximum_weight_assignment(
            pending_keys,
            current_keys,
            {
                (pending_key, current_key): score
                for (current_key, pending_key), score in scores.items()
            },
        )
        matches: dict[str, tuple[str, float]] = {}
        for current_key, pending_key in assignment.items():
            if reverse_assignment.get(pending_key) != current_key:
                continue
            score = scores[(current_key, pending_key)]
            current_ties = sum(
                abs(candidate_score - score) <= 1e-12
                for (candidate_key, _pending_key), candidate_score in scores.items()
                if candidate_key == current_key
            )
            pending_ties = sum(
                abs(candidate_score - score) <= 1e-12
                for (_current_key, candidate_key), candidate_score in scores.items()
                if candidate_key == pending_key
            )
            if current_ties == 1 and pending_ties == 1:
                matches[current_key] = (pending_key, float(score))
        return matches

    @staticmethod
    def _context_confidence(score: float) -> float:
        value = max(0.0, float(score))
        return value / (1.0 + value)

    @staticmethod
    def _acoustic_confidence(score: float) -> float:
        return max(0.0, min(1.0, float(score)))

    @staticmethod
    def _resolution_confidence(
        track_id: str,
        events: Sequence[Mapping[str, Any]],
    ) -> float:
        event = next(
            (
                item
                for item in events
                if str(item.get("track_id", "")) == str(track_id)
            ),
            None,
        )
        if event is None:
            return 0.0
        if event.get("profile_claim_status") == "claimed":
            context_score = event.get("profile_claim_context_score")
            if context_score is not None and np.isfinite(float(context_score)):
                value = max(0.0, float(context_score))
                return value / (1.0 + value)
        values = [
            event.get("best_output_similarity"),
            event.get("within_floor"),
        ]
        numeric = [
            float(value)
            for value in values
            if value is not None and np.isfinite(float(value))
        ]
        return max(0.0, min(1.0, max(numeric, default=0.0)))

    @staticmethod
    def _segments_overlap(
        left: Sequence[PendingSpeakerDecision],
        right: Sequence[PendingSpeakerDecision],
    ) -> bool:
        return any(
            temporal_intersection(left_item.segment, right_item.segment) > 1e-9
            for left_item in left
            for right_item in right
        )

    @classmethod
    def _deadline_tracks_must_be_distinct(
        cls,
        left: Sequence[PendingSpeakerDecision],
        right: Sequence[PendingSpeakerDecision],
    ) -> bool:
        """Keep only causally current structural separation at deadline."""

        left_rounds = {item.first_round for item in left}
        right_rounds = {item.first_round for item in right}
        return bool(left_rounds.intersection(right_rounds)) or cls._segments_overlap(
            left,
            right,
        )

    def _deadline_components(
        self,
        records_by_track: Mapping[str, Sequence[PendingSpeakerDecision]],
    ) -> tuple[tuple[str, ...], ...]:
        remaining = set(records_by_track)
        components: list[tuple[str, ...]] = []
        while remaining:
            root = min(remaining)
            remaining.remove(root)
            component = {root}
            frontier = [root]
            while frontier:
                current = frontier.pop()
                connected = {
                    candidate
                    for candidate in remaining
                    if self._deadline_tracks_must_be_distinct(
                        records_by_track[current],
                        records_by_track[candidate],
                    )
                }
                remaining.difference_update(connected)
                component.update(connected)
                frontier.extend(sorted(connected))
            components.append(tuple(sorted(component)))
        return tuple(components)

    def _deadline_candidate_score(
        self,
        track_id: str,
        output_label: str,
        records: Sequence[PendingSpeakerDecision],
    ) -> tuple[float, float, str]:
        target_track_id = self.registry.track_id_for_output_label(output_label)
        source_track = self.registry.tracks.get(str(track_id))
        target_track = (
            self.registry.tracks.get(str(target_track_id))
            if target_track_id is not None
            else None
        )
        acoustic = (
            source_track.max_core_similarity(target_track)
            if source_track is not None and target_track is not None
            else None
        )
        context_scores = [
            self._context_confidence(record.context_score)
            for record in records
            if record.context_output_label == output_label
        ]
        if acoustic is not None and context_scores:
            confidence = max(
                self._acoustic_confidence(acoustic),
                max(context_scores),
            )
            return 5.0 + confidence, confidence, "fused"
        if acoustic is not None:
            confidence = self._acoustic_confidence(acoustic)
            return 4.0 + confidence, confidence, "acoustic"
        if context_scores:
            confidence = max(context_scores)
            return 3.0 + confidence, confidence, "context"
        if any(
            self.registry.output_mapping.get(str(record.segment.speaker))
            == output_label
            for record in records
        ):
            return 2.5, 0.5, "local_continuity"

        nearest_distance: float | None = None
        for record in records:
            for resolved in self._resolved_context_segments.values():
                if str(resolved.speaker) != output_label:
                    continue
                distance = max(
                    0.0,
                    max(
                        float(record.segment.start) - float(resolved.end),
                        float(resolved.start) - float(record.segment.end),
                    ),
                )
                nearest_distance = (
                    distance
                    if nearest_distance is None
                    else min(nearest_distance, distance)
                )
        if nearest_distance is not None:
            confidence = 1.0 / (1.0 + nearest_distance)
            return 1.0 + confidence, confidence, "neighbor"
        return 0.1, 0.0, "registry_prior"

    def _resolve_best_effort(
        self,
        *,
        decision_round: int,
        final: bool,
    ) -> tuple[dict[str, tuple[str, float, str]], tuple[dict[str, Any], ...]]:
        records_by_track: dict[str, list[PendingSpeakerDecision]] = defaultdict(list)
        for key, pending in self._pending_decisions.items():
            if key in self._assigned_by_id or pending.track_id is None:
                continue
            track_id = str(pending.track_id)
            if self.registry.output_label_for_track(track_id) is not None:
                continue
            records_by_track[track_id].append(pending)
        due_track_ids = {
            track_id
            for track_id, records in records_by_track.items()
            if final
            or any(
                decision_round - record.first_round
                >= self.best_effort_after_ticks
                for record in records
            )
        }
        if not due_track_ids:
            return {}, ()
        due_records = {
            track_id: tuple(records_by_track[track_id])
            for track_id in sorted(due_track_ids)
        }
        output_labels = sorted(
            str(track.output_label)
            for track in self.registry.tracks.values()
            if track.output_label is not None
        )
        resolved: dict[str, tuple[str, float, str]] = {}
        events: list[dict[str, Any]] = []
        for component in self._deadline_components(due_records):
            component_output_labels = tuple(output_labels)
            scores: dict[tuple[str, str], float] = {}
            details: dict[tuple[str, str], tuple[float, str]] = {}
            for track_id in component:
                for output_label in component_output_labels:
                    score, confidence, source = self._deadline_candidate_score(
                        track_id,
                        output_label,
                        due_records[track_id],
                    )
                    scores[(track_id, output_label)] = score
                    details[(track_id, output_label)] = (confidence, source)
            assignment = maximum_weight_assignment(
                component,
                component_output_labels,
                scores,
            )
            for track_id in component:
                output_label = assignment.get(track_id)
                if (
                    output_label is None
                    and len(component) == 1
                    and component_output_labels
                ):
                    output_label = max(
                        component_output_labels,
                        key=lambda candidate: scores.get(
                            (track_id, candidate), float("-inf")
                        ),
                    )
                    confidence, source = details.get(
                        (track_id, output_label),
                        (0.0, "bootstrap"),
                    )
                    self.registry.discard_unpublished_track(track_id)
                    status = "existing"
                elif output_label is None:
                    output_label = self.registry.allocate_output_label(track_id)
                    output_labels.append(output_label)
                    confidence = 0.0
                    source = (
                        "structural_separation"
                        if len(component) > 1
                        else "bootstrap"
                    )
                    status = "new"
                else:
                    confidence, source = details[(track_id, output_label)]
                    self.registry.discard_unpublished_track(track_id)
                    status = "existing"
                target_track_id = self.registry.track_id_for_output_label(output_label)
                for pending in due_records[track_id]:
                    pending.track_id = target_track_id
                    pending.output_target = output_label
                    pending.status = status
                    pending.evidence = source
                    pending.mode = "deadline"
                    pending.confidence = confidence
                    self._remember_assignment(pending.segment, output_label)
                resolved[track_id] = (output_label, confidence, source)
                events.append(
                    {
                        "track_id": track_id,
                        "status": status,
                        "output_target": output_label,
                        "policy": self.policy,
                        "reason": "eos_best_effort" if final else "tick_best_effort",
                        "decision_mode": "deadline",
                        "evidence": source,
                        "confidence": confidence,
                        "source_count": len(due_records[track_id]),
                    }
                )
        return resolved, tuple(events)

    def bind_committed(
        self,
        segments: Sequence[Segment],
        *,
        tick: float,
        start_index: int,
        next_sequence: int,
        context_segments: Sequence[Segment] | None = None,
        final: bool = False,
    ) -> SpeakerBindingBatch:
        del next_sequence
        self._decision_round += 1
        decision_round = self._decision_round
        values = tuple(segments)
        context_values = tuple(context_segments or ())
        eligibility_pool = context_values or values
        (
            context_mapping,
            context_scores,
            context_alignments,
        ) = self._align_context(context_values)
        pending_mapping, pending_scores = self._align_pending_context(context_values)
        pending_revision_matches = self._pending_revision_matches(values)

        bridge_candidates = {
            local_symbol: (
                context_mapping[local_symbol],
                pending_mapping[local_symbol],
            )
            for local_symbol in sorted(set(context_mapping).intersection(pending_mapping))
        }
        bridge_target_track_by_local = {
            local_symbol: self.registry.track_id_for_output_label(output_label)
            for local_symbol, (output_label, _pending_track_id) in bridge_candidates.items()
        }

        observations: list[Observation] = []
        observation_segments: list[Segment] = []
        observation_embedding_by_key: dict[str, Optional[np.ndarray]] = {}
        decision_by_key: dict[str, Any] = {}
        acoustic_edges_by_source_key: dict[str, list[dict[str, Any]]] = {}
        acoustic_association_events: list[dict[str, Any]] = []
        tick_observations: list[Observation] = []
        tick_decisions: list[Any] = []
        for offset, segment in enumerate(values):
            key = _speaker_binding_key(segment)
            if key in self._published_keys:
                continue
            revision_track_id: str | None = None
            revision_source_key: str | None = None
            revision_score = 0.0
            previous_pending: PendingSpeakerDecision | None = None
            core_segment: Segment | None = None
            embedding: Optional[np.ndarray] = None
            embedding_views: Optional[EmbeddingViewSet] = None
            observation: Observation | None = None
            if (predecessor := pending_revision_matches.get(key)) is not None:
                previous_key, revision_score = predecessor
                previous_pending = self._pending_decisions.get(previous_key)
                previous_track_id = (
                    previous_pending.track_id
                    if previous_pending is not None
                    else None
                )
                if previous_track_id is not None:
                    revision_track_id = str(previous_track_id)
                    revision_source_key = str(previous_key)
                    self._pending_decisions.pop(previous_key, None)
            pending = self._pending_decisions.get(key)
            if pending is None:
                if previous_pending is not None and revision_track_id is not None:
                    pending = PendingSpeakerDecision(
                        segment=segment,
                        sequence=previous_pending.sequence,
                        segment_index=previous_pending.segment_index,
                        first_round=previous_pending.first_round,
                        first_tick=previous_pending.first_tick,
                    )
                else:
                    pending = PendingSpeakerDecision(
                        segment=segment,
                        sequence=self._next_sequence,
                        segment_index=int(start_index) + offset,
                        first_round=decision_round,
                        first_tick=float(tick),
                    )
                    self._next_sequence += 1
                self._pending_decisions[key] = pending
            else:
                pending.segment = segment
            if pending.sequence >= self._next_sequence:
                self._next_sequence = pending.sequence + 1
            if key in self._assigned_by_id:
                continue

            local_symbol = str(segment.speaker)
            context_output_label = context_mapping.get(local_symbol)
            pending_track_id = pending_mapping.get(local_symbol)
            pending.context_output_label = context_output_label
            pending.context_score = float(context_scores.get(local_symbol, 0.0))
            pending.context_alignment = context_alignments.get(local_symbol)
            if (
                context_output_label is not None
                and pending_track_id is not None
            ):
                # Keep the observation on its unpublished track until every
                # simultaneous route has established cannot-link evidence.
                context_output_label = None
            if observation is None:
                core_segment = self._core_segment(segment, eligibility_pool)
                if core_segment is not None:
                    if self.policy in MULTIVIEW_IDENTITY_POLICIES:
                        embedding_views = self._embedding_views(
                            core_segment,
                            source_key=key,
                        )
                        embedding = (
                            embedding_views.center
                            if embedding_views is not None
                            else None
                        )
                    else:
                        embedding = self._embedding(core_segment)
                observation = Observation(
                    local_symbol=local_symbol,
                    start=float(segment.start),
                    end=float(segment.end),
                    tick=float(tick),
                    embedding=embedding,
                    embedding_views=embedding_views,
                    quality=self._observation_quality(
                        core_segment or segment,
                        eligibility_pool,
                    ),
                    source_key=key,
                    revision=0,
                    acoustic_start=(
                        float(core_segment.start)
                        if core_segment is not None and embedding is not None
                        else None
                    ),
                    acoustic_end=(
                        float(core_segment.end)
                        if core_segment is not None and embedding is not None
                        else None
                    ),
                    text=str(segment.text),
                )
            evidence_summary = self.registry.observation_evidence_summary(
                observation
            )
            pending.evidence_permissions = dict(
                evidence_summary["permissions"]  # type: ignore[arg-type]
            )
            strength = evidence_summary.get("strength")
            pending.evidence_strength = (
                dict(strength) if isinstance(strength, Mapping) else None
            )
            if revision_track_id is not None and revision_source_key is not None:
                decision = self.registry.replace_unpublished_observation(
                    revision_track_id,
                    revision_source_key,
                    observation,
                    score=revision_score,
                )
                decision_by_key[key] = decision
                tick_observations.append(observation)
                tick_decisions.append(decision)
                pending.track_id = str(decision.track_id)
                pending.score = float(revision_score)
                pending.status = "wait"
                pending.evidence = "pending_revision"
                pending.mode = "pending"
                self._record_core_outcome(
                    track_id=decision.track_id,
                    source_key=key,
                    embedding=embedding,
                )
                continue
            if context_output_label is not None and self._context_may_own_observation(
                observation
            ):
                decision = self.registry.observe_on_output_label(
                    context_output_label,
                    observation,
                )
                decision_by_key[key] = decision
                tick_observations.append(observation)
                tick_decisions.append(decision)
                track_id = self.registry.track_id_for_output_label(context_output_label)
                self._record_core_outcome(
                    track_id=track_id,
                    source_key=key,
                    embedding=embedding,
                )
                pending.track_id = track_id
                pending.output_target = context_output_label
                pending.score = float(context_scores[str(segment.speaker)])
                pending.confidence = self._context_confidence(pending.score)
                pending.status = "existing"
                pending.evidence = "context"
                pending.mode = "context"
                self._remember_assignment(segment, context_output_label)
                continue

            if (
                pending_track_id is not None
                and str(pending_track_id) in self.registry.tracks
                and self._context_may_own_observation(
                    observation,
                    pending_track_id=str(pending_track_id),
                )
            ):
                track_was_output = (
                    self.registry.output_label_for_track(str(pending_track_id)) is not None
                )
                decision = self.registry.observe_on_track(
                    str(pending_track_id),
                    observation,
                    defer_output_resolution=(
                        self.policy == FIXED_COSINE_POLICY
                    ),
                )
                decision_by_key[key] = decision
                tick_observations.append(observation)
                tick_decisions.append(decision)
                pending.track_id = str(decision.track_id)
                pending.score = float(
                    pending_scores.get(local_symbol, decision.score)
                )
                pending.confidence = self._context_confidence(pending.score)
                pending.status = "existing" if track_was_output else "wait"
                pending.evidence = "pending_context"
                pending.mode = "context" if track_was_output else "pending"
                self._record_core_outcome(
                    track_id=decision.track_id,
                    source_key=key,
                    embedding=embedding,
                )
                if decision.output_label is not None:
                    self._remember_assignment(segment, decision.output_label)
                continue

            observation_segments.append(segment)
            observation_embedding_by_key[key] = embedding
            observations.append(observation)
        excluded_tracks_by_local: dict[str, set[str]] = defaultdict(set)
        for observation in observations:
            for routed_observation, routed_decision in zip(
                tick_observations,
                tick_decisions,
            ):
                if self.registry.observations_cannot_link(
                    observation,
                    routed_observation,
                ):
                    excluded_tracks_by_local[str(observation.local_symbol)].add(
                        str(routed_decision.track_id)
                    )
                    bridge_target_track_id = bridge_target_track_by_local.get(
                        str(routed_observation.local_symbol)
                    )
                    if bridge_target_track_id is not None:
                        excluded_tracks_by_local[str(observation.local_symbol)].add(
                            str(bridge_target_track_id)
                        )
        if self.policy not in MODEL_OWNED_IDENTITY_POLICIES:
            for observation in observations:
                if observation.source_key is None:
                    continue
                context_track_id = pending_mapping.get(
                    str(observation.local_symbol)
                )
                acoustic_edges_by_source_key[str(observation.source_key)] = []
                for edge in self.registry.acoustic_identity_edges(
                    observation,
                    excluded_track_ids=excluded_tracks_by_local.get(
                        str(observation.local_symbol),
                        (),
                    ),
                ):
                    event = dict(edge)
                    event["context_candidate"] = (
                        context_track_id is not None
                        and str(event["candidate_track_id"])
                        == str(context_track_id)
                    )
                    acoustic_edges_by_source_key[
                        str(observation.source_key)
                    ].append(event)
                    output_label = event.get("candidate_output_label")
                    best_cosine = event.get("best_cosine")
                    if output_label is not None and best_cosine is not None:
                        self._pending_decisions[
                            str(observation.source_key)
                        ].acoustic_output_scores[str(output_label)] = float(
                            best_cosine
                        )
        output_track_ids_before_batch = {
            track.track_id
            for track in self.registry.tracks.values()
            if track.output_label is not None
        }
        decisions = (
            self.registry.observe_batch(
                observations,
                allow_output_tracks=self.policy
                in {FIXED_COSINE_POLICY, CLEAN_CORE_POLICY},
                excluded_tracks_by_local=excluded_tracks_by_local,
                defer_output_resolution=(
                    self.policy == FIXED_COSINE_POLICY
                ),
            )
            if observations
            else ()
        )
        for segment, observation, decision in zip(
            observation_segments,
            observations,
            decisions,
        ):
            key = _speaker_binding_key(segment)
            decision_by_key[key] = decision
            for event in acoustic_edges_by_source_key.get(key, ()):
                event.update(
                    {
                        "selected_track_id": str(decision.track_id),
                        "selected_by_current_policy": (
                            str(event["candidate_track_id"])
                            == str(decision.track_id)
                        ),
                        "decision_score": float(decision.score),
                    }
                )
                acoustic_association_events.append(event)
            pending = self._pending_decisions[key]
            pending.track_id = str(decision.track_id)
            pending.score = float(decision.score)
            pending.confidence = self._acoustic_confidence(decision.score)
            pending.status = (
                "existing"
                if decision.track_id in output_track_ids_before_batch
                else "new"
                if decision.output_label is not None
                else "wait"
            )
            pending.evidence = "acoustic"
            pending.mode = "acoustic" if decision.output_label is not None else "pending"
            if decision.output_label is not None:
                self._remember_assignment(segment, decision.output_label)
            self._record_core_outcome(
                track_id=decision.track_id,
                source_key=key,
                embedding=observation_embedding_by_key.get(key),
            )
        if self.policy not in MODEL_OWNED_IDENTITY_POLICIES:
            for local_symbol, (
                output_label,
                pending_track_id,
            ) in bridge_candidates.items():
                if (
                    self.policy == CLEAN_CORE_POLICY
                    and self.registry.retained_core_count(pending_track_id) > 0
                ):
                    continue
                moved_source_keys = (
                    self.registry.transfer_unpublished_evidence_to_output(
                        pending_track_id,
                        output_label,
                    )
                )
                if not moved_source_keys:
                    continue
                target_track_id = self.registry.track_id_for_output_label(output_label)
                for key in moved_source_keys:
                    pending = self._pending_decisions.get(key)
                    if pending is None or str(pending.track_id or "") != str(
                        pending_track_id
                    ):
                        continue
                    pending.track_id = target_track_id
                    pending.output_target = output_label
                    pending.score = float(context_scores[local_symbol])
                    pending.confidence = self._context_confidence(pending.score)
                    pending.status = "existing"
                    pending.evidence = "context_bridge"
                    pending.mode = "context"
                    self._remember_assignment(pending.segment, output_label)

        unprofiled_claims = self._event_unprofiled_claims(
            tuple(_speaker_binding_key(segment) for segment in values)
        )
        snapshot_local_symbols = {
            str(segment.speaker) for segment in context_values
        }
        occupied_output_by_local = {
            str(local_symbol): str(output_label)
            for local_symbol, output_label in context_mapping.items()
            if str(local_symbol) in snapshot_local_symbols
        }
        partition_excluded_output_labels_by_track: dict[str, set[str]] = defaultdict(set)
        for key in (_speaker_binding_key(segment) for segment in values):
            pending = self._pending_decisions.get(str(key))
            if pending is None or pending.track_id is None:
                continue
            local_symbol = str(pending.segment.speaker)
            partition_excluded_output_labels_by_track[str(pending.track_id)].update(
                output_label
                for other_local, output_label in occupied_output_by_local.items()
                if other_local != local_symbol
            )
        resolution_actions: dict[str, tuple[str, str]] = {}
        resolution_events: list[dict[str, Any]] = []
        resolution_source_keys: dict[str, tuple[str, ...]] = {}
        if self.policy == FIXED_COSINE_POLICY:
            unresolved_track_ids = sorted(
                {
                    str(pending.track_id)
                    for key, pending in self._pending_decisions.items()
                    if key not in self._assigned_by_id
                    and pending.track_id is not None
                    and self.registry.output_label_for_track(str(pending.track_id))
                    is None
                }
            )
            resolution_actions = {
                track_id: ("new", self.registry.allocate_output_label(track_id))
                for track_id in unresolved_track_ids
            }
            resolution_source_keys = {
                track_id: tuple(
                    key
                    for key, pending in self._pending_decisions.items()
                    if key not in self._assigned_by_id
                    and str(pending.track_id or "") == track_id
                )
                for track_id in unresolved_track_ids
            }
        else:
            resolution_actions = self.registry.resolve_event_tracks(
                unprofiled_claims=unprofiled_claims,
                partition_excluded_output_labels_by_track={
                    track_id: tuple(sorted(output_labels))
                    for track_id, output_labels in sorted(
                        partition_excluded_output_labels_by_track.items()
                    )
                },
            )
            resolution_events.extend(self.registry.last_resolution_events)
            resolution_source_keys = self.registry.last_resolution_source_keys
            if self.policy in MULTIVIEW_IDENTITY_POLICIES:
                for event in self.registry.last_resolution_events:
                    assessment = event.get("assessment")
                    if not isinstance(assessment, Mapping) or not bool(
                        assessment.get("profile_allowed", False)
                    ):
                        continue
                    before_state = event.get("acoustic_state_before")
                    after_state = event.get("acoustic_state_after")
                    before_count = (
                        int(before_state.get("exemplar_count", 0))
                        if isinstance(before_state, Mapping)
                        else 0
                    )
                    after_count = (
                        int(after_state.get("exemplar_count", 0))
                        if isinstance(after_state, Mapping)
                        else 0
                    )
                    if (
                        event.get("durable_update_status") == "admitted"
                        or after_count > before_count
                    ):
                        self._core_counters["accepted"] += 1
                    else:
                        self._core_counters["rejected"] += 1

        for source_track_id, (action, output_label) in resolution_actions.items():
            target_track_id = self.registry.track_id_for_output_label(output_label)
            resolution_event = next(
                (
                    event
                    for event in resolution_events
                    if str(event.get("track_id", "")) == source_track_id
                ),
                None,
            )
            profile_claimed = bool(
                resolution_event is not None
                and resolution_event.get("profile_claim_status") == "claimed"
            )
            for key in resolution_source_keys.get(source_track_id, ()):
                pending = self._pending_decisions.get(key)
                if pending is None or str(pending.track_id or "") != str(
                    source_track_id
                ):
                    continue
                pending.track_id = target_track_id
                pending.output_target = output_label
                pending.status = action
                pending.policy = self.policy
                pending.evidence = "profile_claim" if profile_claimed else "policy"
                pending.mode = (
                    "fused"
                    if profile_claimed
                    or (
                        pending.context_output_label == output_label
                        and output_label in pending.acoustic_output_scores
                    )
                    else "acoustic"
                )
                pending.confidence = self._resolution_confidence(
                    source_track_id,
                    resolution_events,
                )
                self._remember_assignment(pending.segment, output_label)

        if self.policy in MODEL_OWNED_IDENTITY_POLICIES:
            unresolved_model = [
                key
                for key, pending in self._pending_decisions.items()
                if key not in self._assigned_by_id
                and pending.track_id is not None
            ]
            if unresolved_model:
                raise RuntimeError(
                    "identity event inference left unresolved speaker tracks: "
                    + ", ".join(sorted(unresolved_model))
                )
            deadline_events: tuple[dict[str, Any], ...] = ()
        else:
            _deadline_actions, deadline_events = self._resolve_best_effort(
                decision_round=decision_round,
                final=final,
            )
        resolution_events.extend(deadline_events)

        ready_keys: set[str] = set()
        for key, pending in self._pending_decisions.items():
            if key in self._published_keys:
                continue
            if key in self._assigned_by_id:
                ready_keys.add(key)
                continue
            track_id = pending.track_id
            if track_id is None:
                continue
            output_label = self.registry.output_label_for_track(str(track_id))
            if output_label is None:
                continue
            pending.output_target = output_label
            if pending.mode == "pending":
                pending.mode = "acoustic"
            self._remember_assignment(pending.segment, output_label)
            ready_keys.add(key)

        publish_keys = sorted(
            (key for key in ready_keys if key not in self._published_keys),
            key=lambda key: self._pending_decisions[key].sequence,
        )
        published: list[Segment] = []
        assignments: list[dict[str, Any]] = []
        decision_events: list[dict[str, Any]] = []
        max_age = 0.0
        for key in publish_keys:
            pending = self._pending_decisions[key]
            segment = pending.segment
            resolved = self._assigned_by_id[key]
            assigned = self._centroid_correction(segment, resolved)
            published.append(segment.with_updates(speaker=assigned))
            self._published_keys.add(key)
            self._publication_ledger.append(
                PublicationRecord(
                    source_key=key,
                    output_label=assigned,
                    text=str(segment.text),
                    start=float(segment.start),
                    end=float(segment.end),
                    tick=float(tick),
                )
            )
            max_age = max(max_age, float(tick) - float(segment.end))
            wait_ticks = decision_round - pending.first_round
            decision_event: dict[str, Any] = {
                "tick": float(tick),
                "segment_index": pending.segment_index,
                "segment_id": stable_segment_id(segment),
                "source_key": key,
                "local_symbol": str(segment.speaker),
                "speaker": assigned,
                "status": pending.status,
                "mode": pending.mode,
                "evidence": pending.evidence,
                "score": float(pending.score),
                "confidence": float(pending.confidence),
                "wait_ticks": int(wait_ticks),
                "wait_sec": max(0.0, float(tick) - pending.first_tick),
                "eos": bool(final),
                "evidence_permissions": dict(pending.evidence_permissions),
                "evidence_strength": (
                    dict(pending.evidence_strength)
                    if pending.evidence_strength is not None
                    else None
                ),
            }
            if pending.track_id is not None:
                decision_event["track_id"] = str(pending.track_id)
            if pending.output_target is not None:
                decision_event["output_target"] = str(pending.output_target)
            if pending.policy is not None:
                decision_event["policy"] = str(pending.policy)
            if assigned != resolved:
                # Keep the registry's own answer on the receipt so the two
                # layers stay separable in the audit.
                decision_event["registry_speaker"] = resolved
                decision_event["centroid_corrected"] = True
            decision_events.append(decision_event)
            if str(segment.speaker) != assigned:
                decision = decision_by_key.get(key)
                event: dict[str, Any] = {
                    "tick": float(tick),
                    "segment_index": pending.segment_index,
                    "segment_id": stable_segment_id(segment),
                    "source_key": key,
                    "local_symbol": str(segment.speaker),
                    "speaker": assigned,
                    "status": str(pending.status or "resolved"),
                    "score": float(
                        pending.score
                        if pending.score is not None
                        else decision.score if decision is not None else 0.0
                    ),
                }
                track_id = pending.track_id
                if track_id is not None:
                    event["track_id"] = str(track_id)
                elif decision is not None:
                    event["track_id"] = str(decision.track_id)
                if pending.output_target is not None:
                    event["output_target"] = str(pending.output_target)
                if pending.policy is not None:
                    event["policy"] = str(pending.policy)
                if pending.evidence:
                    event["evidence"] = str(pending.evidence)
                assignments.append(event)
            self._pending_decisions.pop(key, None)
        return SpeakerBindingBatch(
            published=tuple(published),
            assignments=tuple(assignments),
            max_publication_age_sec=max_age,
            resolution_events=tuple(resolution_events),
            acoustic_association_events=tuple(acoustic_association_events),
            decision_events=tuple(decision_events),
        )

    def assign_committed(
        self,
        segments: Sequence[Segment],
        *,
        tick: float,
        start_index: int,
        next_sequence: int,
        context_segments: Sequence[Segment] | None = None,
        final: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        """Compatibility view returning only assignment events."""

        return self.bind_committed(
            segments,
            tick=tick,
            start_index=start_index,
            next_sequence=next_sequence,
            context_segments=context_segments,
            final=final,
        ).assignments

    def pending_segments(self) -> tuple[Segment, ...]:
        return tuple(
            self._pending_decisions[key].segment
            for key in sorted(
                (
                    key
                    for key in self._pending_decisions
                    if key not in self._assigned_by_id
                    and key not in self._published_keys
                ),
                key=lambda item: self._pending_decisions[item].sequence,
            )
        )

    def _remember_assignment(self, segment: Segment, output_label: str) -> str:
        key = _speaker_binding_key(segment)
        assigned = str(output_label)
        previous = self._assigned_by_id.get(key)
        if previous is not None and previous != assigned:
            raise RuntimeError(
                f"speaker identity rewrite for committed span {key}: "
                f"{previous} -> {assigned}"
            )
        self._assigned_by_id[key] = assigned
        self._resolved_context_segments.setdefault(
            key,
            segment.with_updates(speaker=assigned),
        )
        return assigned

    def _view_label(self, segment: Segment) -> str:
        key = _speaker_binding_key(segment)
        assigned = self._assigned_by_id.get(key)
        if assigned is None:
            return str(segment.speaker)
        return self._published_label(key, assigned)

    def view(self, segments: Sequence[Segment]) -> tuple[Segment, ...]:
        return tuple(
            segment.with_updates(speaker=self._view_label(segment))
            for segment in segments
        )

    def speaker_count(self) -> int:
        return len(self.registry.tracks)

    def confirmed_speaker_count(self) -> int:
        return sum(
            1
            for track in self.registry.tracks.values()
            if track.output_label is not None
        )

    def pending_speaker_count(self) -> int:
        return self.speaker_count() - self.confirmed_speaker_count()

    def is_fully_assigned(self, segments: Sequence[Segment]) -> bool:
        # Compare against the label that was actually published, otherwise a
        # corrected span reads as unassigned and fails the gate spuriously.
        for segment in segments:
            key = _speaker_binding_key(segment)
            assigned = self._assigned_by_id.get(key)
            if assigned is None:
                return False
            if self._published_label(key, assigned) != str(segment.speaker):
                return False
        return True


PostCommitSpeakerAssigner = _MultiPolicySpeakerAssigner


class CausalSpeakerBinder(_MultiPolicySpeakerAssigner):
    """Canonical causal E-to-G binder.

    The historical ``PostCommitSpeakerAssigner`` name remains available only
    so sealed experiments can be reproduced with their explicit policy.  New
    callers get the authoritative lego policy even when they construct
    the binder directly rather than going through ``build_speaker_binder``.
    """

    def __init__(
        self,
        *,
        policy: str = LEGO_POLICY,
        **kwargs: Any,
    ) -> None:
        super().__init__(policy=policy, **kwargs)


def build_speaker_binder(
    *,
    sample: Mapping[str, Any],
    association: Mapping[str, Any],
    speaker_mapping: Mapping[str, Any],
    embedding_config: Mapping[str, Any],
    embedding_store: Any,
    runtime: SpeakerBindingRuntime,
) -> Optional[CausalSpeakerBinder]:
    """Build the canonical speaker binder used by batch and online paths.

    Legacy policies remain readable for old sealed configs, but a missing
    policy now means the canonical lego policy rather than silently
    selecting an older empirical resolver.
    """

    if embedding_store is None:
        return None
    policy = str(
        association.get("policy", LEGO_POLICY)
    ).strip()
    return CausalSpeakerBinder(
        sample_id=sample.get("id", sample.get("audio_name", "sample")),
        embedding_provider=(
            None
            if policy in MULTIVIEW_IDENTITY_POLICIES
            else lambda segment: embedding_store.embedding(
                segment,
                runtime.cutoff_sec,
            )
        ),
        embedding_view_provider=(
            (
                lambda segment, source_key: embedding_store.embedding_views(
                    segment,
                    runtime.cutoff_sec,
                    source_key=source_key,
                )
            )
            if policy in MULTIVIEW_IDENTITY_POLICIES
            else None
        ),
        policy=policy,
        min_support_sec=float(
            speaker_mapping.get(
                "min_confirm_sec",
                association.get("min_support_sec", 0.0),
            )
        ),
        min_observations=int(
            speaker_mapping.get(
                "min_confirm_segments",
                association.get("min_independent_windows", 1),
            )
        ),
        min_overlap_sec=float(association.get("min_overlap_sec", 0.1)),
        min_cosine=float(
            speaker_mapping.get(
                "embedding_threshold",
                association.get("min_cosine", 0.70),
            )
        ),
        different_local_merge_cosine=float(
            association.get("different_local_merge_cosine", 0.90)
        ),
        same_local_split_cosine=float(
            association.get("same_local_split_cosine", 0.30)
        ),
        max_core_samples_per_track=int(
            association.get("max_core_samples_per_track", 120)
        ),
        core_min_duration_sec=float(
            association.get("core_min_duration_sec", 1.5)
        ),
        core_min_quality=float(association.get("core_min_quality", 0.25)),
        require_core_for_confirmation=bool(
            association.get("require_core_for_confirmation", False)
        ),
        publish_pending_speakers=bool(
            association.get("publish_pending_speakers", False)
        ),
        best_effort_after_ticks=int(
            association.get("best_effort_after_ticks", 1)
        ),
        embedding_dimension=int(embedding_config.get("dimension", 192)),
        robust_exemplar_capacity=int(
            association.get("robust_exemplar_capacity", 4)
        ),
        risk_component_capacity=int(
            association.get("risk_component_capacity", 8)
        ),
        risk_challenger_capacity=int(
            association.get("risk_challenger_capacity", 2)
        ),
        risk_local_partition_provisional_enabled=bool(
            association.get(
                "risk_local_partition_provisional_enabled",
                True,
            )
        ),
        centroid_correction_enabled=bool(
            association.get("centroid_correction_enabled", False)
        ),
        centroid_correction_min_gain=float(
            association.get("centroid_correction_min_gain", 0.05)
        ),
        centroid_correction_min_cosine=float(
            association.get("centroid_correction_min_cosine", 0.50)
        ),
        centroid_correction_min_duration_sec=float(
            association.get("centroid_correction_min_duration_sec", 1.5)
        ),
        # The centroid layer scores the WHOLE committed span, not the clean-core
        # subspan the registry uses: a label centroid wants as much of the
        # speaker as the span contains, and the offline measurement was made at
        # this caliber.
        centroid_embedding_provider=(
            (
                lambda segment: embedding_store.embedding(
                    segment,
                    runtime.cutoff_sec,
                )
            )
            if association.get("centroid_correction_enabled", False)
            else None
        ),
        output_prefix="S",
    )


class SpeakerBindingCommitter(IncrementalCommitter):
    """Commit stable text, bind one P, and publish it atomically."""

    def __init__(
        self,
        *,
        assigner: Optional[CausalSpeakerBinder],
        runtime: SpeakerBindingRuntime,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.assigner = assigner
        self.runtime = runtime

    def update(
        self,
        window_end: float,
        segments: Sequence[Segment],
        *,
        final: bool = False,
    ) -> CommitUpdate:
        values = tuple(segments)
        update = super().update(window_end, values, final=final)
        self.runtime.reset_events()
        if self.assigner is None:
            return update
        self.runtime.cutoff_sec = float(window_end)
        binding = self.assigner.bind_committed(
            update.committed,
            tick=float(window_end),
            start_index=0,
            next_sequence=0,
            context_segments=values,
            final=final,
        )
        self.runtime.events["speaker_resolution_events"] = (
            binding.resolution_events
        )
        self.runtime.events["acoustic_association_events"] = (
            binding.acoustic_association_events
        )
        self.runtime.events["speaker_decision_events"] = binding.decision_events
        bound_committed = self.assigner.view(binding.published)
        if bound_committed:
            self.runtime.published_until = max(
                self.runtime.published_until,
                max(float(item.end) for item in bound_committed),
            )
        if binding.assignments:
            index_by_binding_key = {
                _speaker_binding_key(segment): index
                for index, segment in enumerate(self.committed)
            }
            self.runtime.events["speaker_assignments"] = tuple(
                dict(
                    item,
                    segment_index=index_by_binding_key.get(
                        item["source_key"],
                        item["segment_index"],
                    ),
                )
                for item in binding.assignments
            )
        return replace(
            update,
            committed=bound_committed,
            pending=tuple(update.pending) + self.assigner.pending_segments(),
            published_until=self.runtime.published_until,
            delta_first_publish_age_sec=max(
                float(update.delta_first_publish_age_sec),
                float(binding.max_publication_age_sec),
            ),
        )


def build_speaker_binding_committer(
    *,
    stream: Mapping[str, Any],
    association: Mapping[str, Any],
    assigner: Optional[CausalSpeakerBinder],
    runtime: SpeakerBindingRuntime,
) -> SpeakerBindingCommitter:
    acoustic_min_duration = min(
        0.4,
        float(association.get("core_min_duration_sec", 1.5)),
    )
    evidence_policy = (
        EvidencePoorTurnPolicy(
            acoustic_min_duration_sec=acoustic_min_duration
        )
        if acoustic_min_duration > 0.0
        else None
    )
    return SpeakerBindingCommitter(
        assigner=assigner,
        runtime=runtime,
        right_context_sec=float(stream["right_context_sec"]),
        heuristic_overlap_min_sec=float(
            stream.get("heuristic_overlap_min_sec", 0.20)
        ),
        heuristic_overlap_min_fraction=float(
            stream.get("heuristic_overlap_min_fraction", 0.15)
        ),
        best_effort_publish_sec=(
            None
            if stream.get("best_effort_publish_sec") is None
            else float(stream["best_effort_publish_sec"])
        ),
        turn_evidence_policy=evidence_policy,
    )


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config.{key} must be a mapping")
    return value


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if value in (None, ""):
        raise ValueError(f"{name} is required and must be finite and positive")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not np.isfinite(number) or (number < 0.0 if allow_zero else number <= 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return number


def _probe_audio_duration_sec(audio_path: str) -> Optional[float]:
    """Length of the audio file, or ``None`` when it cannot be read cheaply."""

    path = Path(str(audio_path)).expanduser()
    if not path.is_file():
        return None
    try:
        import soundfile  # type: ignore

        info = soundfile.info(str(path))
        duration = float(info.frames) / float(info.samplerate)
    except Exception:
        try:
            import wave

            with wave.open(str(path), "rb") as handle:
                duration = handle.getnframes() / float(handle.getframerate())
        except Exception:
            return None
    if not math.isfinite(duration) or duration <= 0.0:
        return None
    return duration


# How far ``sample.duration_sec`` may sit from the real file length before it is
# treated as a mistake rather than a rounding difference.  Generous on purpose:
# a clipped evaluation protocol legitimately declares less than the full file,
# but a stale value copied from an example config is usually off by minutes.
_DURATION_MISMATCH_TOLERANCE_SEC = 30.0
_DURATION_MISMATCH_TOLERANCE_RATIO = 0.05


def _decoder_timeout_sec(
    inference: Mapping[str, Any],
    stream: Mapping[str, Any],
) -> float:
    value = inference.get("timeout_sec", None)
    if value in (None, ""):
        value = stream.get("max_wall_latency_sec", None)
    if value in (None, ""):
        value = 600.0
    return _positive(value, "config.inference.timeout_sec")


def _is_recoverable_decode_error(error: Exception) -> bool:
    if isinstance(error, TimeoutError):
        return True
    if isinstance(error, RuntimeError):
        message = str(error)
        return message.startswith("failed to connect to vLLM API:") or message.startswith(
            "vLLM request failed with HTTP "
        )
    return False


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _remove_hypotheses(destination: Path) -> None:
    for path in (
        destination / "hypothesis.json",
        destination / "global_embedding" / "hypothesis.json",
        destination / "global_embedding" / "summary.json",
    ):
        path.unlink(missing_ok=True)


def _bounded_error(exc: BaseException, *, limit: int = 512) -> str:
    """Keep diagnostics useful without allowing unbounded artifacts."""

    value = repr(exc)
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _mark_pipeline_failure(
    destination: Path,
    exc: BaseException,
    *,
    stage: str,
) -> None:
    """Persist a bounded fail-closed postprocess result for pipeline errors."""

    validation: Mapping[str, Any] = {}
    validation_path = destination / "stream_validation.json"
    if validation_path.is_file():
        try:
            loaded = json.loads(validation_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                validation = loaded
        except (OSError, json.JSONDecodeError):
            pass
    error = _bounded_error(exc)
    payload = dict(validation)
    payload.update(
        {
            "status": "failed",
            "speaker_assignment_gate_pass": False,
            "postprocess_gate_pass": False,
            "postprocess_error": error,
            "pipeline_stage_failed": True,
            "pipeline_error": error,
        }
    )
    _write_json(destination / "stream_validation.json", payload)
    _remove_hypotheses(destination)
    _write_json(
        destination / "pipeline_validation.json",
        {
            "mode": "global_embedding",
            "status": "failed",
            "speaker_assignment_gate_pass": False,
            "postprocess_gate_pass": False,
            "error": error,
            "postprocess_error": error,
            "pipeline_error": error,
            "pipeline_stage_failed": True,
        },
    )


def validate_config(
    config: Mapping[str, Any],
    *,
    require_embedding: bool = True,
    require_association: bool = True,
) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    normalized = deepcopy(dict(config))
    sample = _mapping(normalized, "sample")
    stream = _mapping(normalized, "stream")
    _mapping(normalized, "inference")
    if not str(normalized.get("endpoint", "")).strip():
        raise ValueError("config.endpoint must be non-empty")
    if not str(normalized.get("served_model_name", "")).strip():
        raise ValueError("config.served_model_name must be non-empty")
    if not str(sample.get("audio_path", "")).strip():
        raise ValueError("config.sample.audio_path must be non-empty")
    declared_duration = _positive(
        sample.get("duration_sec"), "config.sample.duration_sec"
    )
    # ``duration_sec`` drives the whole window schedule, and nothing downstream
    # reads past it.  A value copied from an example config therefore transcribes
    # a prefix of the audio and still validates as "passed" -- a silent wrong
    # answer, which is worse than a crash.  Cross-check it against the file when
    # the file is cheaply readable.
    probed_duration = _probe_audio_duration_sec(sample["audio_path"])
    if probed_duration is not None:
        gap = abs(probed_duration - declared_duration)
        tolerance = max(
            _DURATION_MISMATCH_TOLERANCE_SEC,
            _DURATION_MISMATCH_TOLERANCE_RATIO * probed_duration,
        )
        if gap > tolerance:
            raise ValueError(
                "config.sample.duration_sec is "
                f"{declared_duration:.2f}s but the audio file is "
                f"{probed_duration:.2f}s ({sample['audio_path']}). The window "
                "schedule is derived from duration_sec, so a stale value "
                "silently transcribes only part of the recording. Set it to the "
                "real length, or pass the intended clipped length if you are "
                "evaluating a fixed excerpt."
            )
    _positive(stream.get("step_sec"), "config.stream.step_sec")
    context = _positive(stream.get("context_sec"), "config.stream.context_sec")
    right = _positive(
        stream.get("right_context_sec"),
        "config.stream.right_context_sec",
        allow_zero=True,
    )
    _positive(
        stream.get("heuristic_overlap_min_sec", 0.20),
        "config.stream.heuristic_overlap_min_sec",
        allow_zero=True,
    )
    overlap_fraction = float(stream.get("heuristic_overlap_min_fraction", 0.15))
    if not np.isfinite(overlap_fraction) or not 0.0 <= overlap_fraction <= 1.0:
        raise ValueError(
            "config.stream.heuristic_overlap_min_fraction must be finite and in [0, 1]"
        )
    if isinstance(stream.get("anchored_prefix"), Mapping) and bool(stream["anchored_prefix"].get("enabled", False)):
        raise ValueError("global_embedding is embedding-only; disable prefill anchored_prefix")
    if right >= context:
        raise ValueError("config.stream.right_context_sec must be smaller than context_sec")
    if "max_algorithmic_delay_sec" in stream:
        raise ValueError(
            "config.stream.max_algorithmic_delay_sec is removed; "
            "stability is evidence-driven rather than deadline-driven"
        )
    if "publication_age_limit_sec" in stream:
        raise ValueError(
            "config.stream.publication_age_limit_sec is removed; "
            "speaker binding cannot delay stable text"
        )
    if str(stream.get("stitch_mode", "incremental")) != "incremental":
        raise ValueError("only stitch_mode=incremental is supported; midpoint stitching is removed")
    if stream.get("schedule_path") not in (None, ""):
        raise ValueError("dynamic schedule_path is not implemented; use the fixed-step stream contract")
    if require_embedding:
        embedding = _mapping(normalized, "embedding")
        # ``repo_root`` is the historical name for the same value and is still
        # accepted, so a config written against an earlier revision keeps working.
        speakerlab_root = str(
            embedding.get("speakerlab_root") or embedding.get("repo_root") or ""
        ).strip()
        if speakerlab_root:
            embedding = dict(embedding)
            embedding["speakerlab_root"] = speakerlab_root
            normalized["embedding"] = embedding
        if not str(embedding.get("model_dir", "")).strip() or not speakerlab_root:
            raise ValueError(
                "config.embedding.speakerlab_root and model_dir are required"
            )
        chunk_duration = _positive(embedding.get("chunk_duration_sec", 1.5), "embedding.chunk_duration_sec")
        chunk_step = _positive(embedding.get("chunk_step_sec", 0.75), "embedding.chunk_step_sec")
        if chunk_step > chunk_duration + 1e-9:
            raise ValueError("embedding.chunk_step_sec cannot exceed chunk_duration_sec")
    if require_association:
        association = _mapping(normalized, "association")
        if "merge" in association:
            raise ValueError(
                "association.merge is removed; speaker identity is resolved "
                "only by the selected online policy"
            )
        association_mode = str(
            association.get("mode", CAUSAL_EMBEDDING_MODE)
        ).strip().lower()
        if association_mode != CAUSAL_EMBEDDING_MODE:
            raise ValueError(
                "association.mode must be causal_embedding; "
                "dense, spectral, and offline reassignment modes are removed"
            )
        association["mode"] = association_mode
        policy = str(
            association.get("policy", LEGO_POLICY)
        ).strip()
        if policy not in SPEAKER_MATCH_POLICIES:
            raise ValueError(
                "association.policy must be one of: "
                + ", ".join(sorted(SPEAKER_MATCH_POLICIES))
            )
        association["policy"] = policy
        if policy in MULTIVIEW_IDENTITY_POLICIES:
            integer_parameters = {
                "robust_exemplar_capacity": int(
                    association.get("robust_exemplar_capacity", 4)
                ),
            }
            if integer_parameters["robust_exemplar_capacity"] not in {3, 4, 5}:
                raise ValueError(
                    "association.robust_exemplar_capacity must be one of 3, 4, 5"
                )
            if policy == LEGO_POLICY:
                local_partition_provisional_enabled = association.get(
                    "risk_local_partition_provisional_enabled",
                    True,
                )
                if not isinstance(local_partition_provisional_enabled, bool):
                    raise ValueError(
                        "association.risk_local_partition_provisional_enabled "
                        "must be boolean"
                    )
                association["risk_local_partition_provisional_enabled"] = (
                    local_partition_provisional_enabled
                )
                integer_parameters.update(
                    {
                        "risk_component_capacity": int(
                            association.get("risk_component_capacity", 8)
                        ),
                        "risk_challenger_capacity": int(
                            association.get("risk_challenger_capacity", 2)
                        ),
                    }
                )
                if integer_parameters["risk_component_capacity"] < 1:
                    raise ValueError(
                        "association.risk_component_capacity must be positive"
                    )
                if integer_parameters["risk_challenger_capacity"] < 1:
                    raise ValueError(
                        "association.risk_challenger_capacity must be positive"
                    )
            association.update(integer_parameters)
        min_support_sec = float(association.get("min_support_sec", 0.0))
        min_independent_windows = int(
            association.get("min_independent_windows", 1)
        )
        require_core_for_confirmation = bool(
            association.get("require_core_for_confirmation", False)
        )
        if min_support_sec < 0.0 or min_independent_windows < 1:
            raise ValueError(
                "causal_embedding confirmation thresholds must be non-negative "
                "with at least one observation"
            )
        if require_core_for_confirmation:
            raise ValueError(
                "association.require_core_for_confirmation is superseded by "
                "the selected event policy's evidence contract"
            )
        if bool(association.get("publish_pending_speakers", False)):
            raise ValueError("pending speakers must not be published")
        association["min_support_sec"] = min_support_sec
        association["min_independent_windows"] = min_independent_windows
        association["require_core_for_confirmation"] = False
        association["publish_pending_speakers"] = False
        cosine = _positive(association.get("min_cosine", 0.70), "association.min_cosine", allow_zero=True)
        if cosine <= 0.0 or cosine > 1.0:
            raise ValueError("association.min_cosine must be in (0, 1]")
        for key in ("different_local_merge_cosine", "same_local_split_cosine"):
            value = float(
                association.get(
                    key,
                    0.90 if key == "different_local_merge_cosine" else 0.30,
                )
            )
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"association.{key} must be in [0, 1]")
        if int(association.get("min_independent_windows", 2)) < 1:
            raise ValueError("association.min_independent_windows must be positive")
        if int(association.get("max_core_samples_per_track", 120)) < 1:
            raise ValueError("association.max_core_samples_per_track must be positive")
        core_min_duration_sec = float(association.get("core_min_duration_sec", 1.5))
        core_min_quality = float(association.get("core_min_quality", 0.25))
        if not np.isfinite(core_min_duration_sec) or core_min_duration_sec < 0.0:
            raise ValueError("association.core_min_duration_sec must be finite and non-negative")
        if not np.isfinite(core_min_quality) or core_min_quality < 0.0:
            raise ValueError("association.core_min_quality must be finite and non-negative")
        native_chunk_duration_sec = 1.5
        embedding_config = normalized.get("embedding")
        if isinstance(embedding_config, Mapping):
            native_chunk_duration_sec = float(
                embedding_config.get("chunk_duration_sec", 1.5)
            )
        if (
            policy == CLEAN_CORE_POLICY
            and core_min_duration_sec + 1e-9 < native_chunk_duration_sec
        ):
            raise ValueError(
                "association.core_min_duration_sec must be at least the "
                "embedding native chunk_duration_sec"
            )
        best_effort_after_ticks = association.get("best_effort_after_ticks", 1)
        if (
            isinstance(best_effort_after_ticks, bool)
            or int(best_effort_after_ticks) != best_effort_after_ticks
            or int(best_effort_after_ticks) < 1
        ):
            raise ValueError(
                "association.best_effort_after_ticks must be a positive integer"
            )
        if "core_max_overlap_fraction" in association:
            raise ValueError(
                "association.core_max_overlap_fraction is removed; clean-core "
                "extraction always requires zero overlap"
            )
        # Keys belonging to a retired resolver variant. Reject rather than
        # ignore: a carried-over config would otherwise run as a plain
        # single-resolver stream while still looking like that variant on disk.
        for retired_key in ("identity_shadow_twin", "hierarchical_posterior_freeze"):
            if association.get(retired_key) not in (None, "", False):
                raise ValueError(
                    f"association.{retired_key} is not supported in this release"
                )
        centroid_enabled = association.get("centroid_correction_enabled", False)
        if not isinstance(centroid_enabled, bool):
            raise ValueError(
                "association.centroid_correction_enabled must be boolean"
            )
        centroid_min_gain = float(
            association.get("centroid_correction_min_gain", 0.05)
        )
        centroid_min_cosine = float(
            association.get("centroid_correction_min_cosine", 0.50)
        )
        centroid_min_duration_sec = float(
            association.get("centroid_correction_min_duration_sec", 1.5)
        )
        if not 0.0 < centroid_min_gain <= 2.0:
            raise ValueError(
                "association.centroid_correction_min_gain must be in (0, 2]"
            )
        if not -1.0 <= centroid_min_cosine <= 1.0:
            raise ValueError(
                "association.centroid_correction_min_cosine must be in [-1, 1]"
            )
        if (
            not np.isfinite(centroid_min_duration_sec)
            or centroid_min_duration_sec < 0.0
        ):
            raise ValueError(
                "association.centroid_correction_min_duration_sec must be "
                "finite and non-negative"
            )
        association["centroid_correction_enabled"] = bool(centroid_enabled)
        association["centroid_correction_min_gain"] = centroid_min_gain
        association["centroid_correction_min_cosine"] = centroid_min_cosine
        association["centroid_correction_min_duration_sec"] = (
            centroid_min_duration_sec
        )
        association["core_min_duration_sec"] = core_min_duration_sec
        association["core_min_quality"] = core_min_quality
        association["best_effort_after_ticks"] = int(best_effort_after_ticks)
    return normalized


def _build_causal_segment_store(config: Mapping[str, Any]) -> CausalSegmentStore:
    sample = _mapping(config, "sample")
    embedding = _mapping(config, "embedding")
    return CausalSegmentStore(
        audio_path=Path(str(sample["audio_path"])),
        speakerlab_root=Path(str(embedding.get("speakerlab_root") or embedding["repo_root"])),
        model_dir=Path(str(embedding["model_dir"])),
        model_name=str(embedding.get("model_name", "eres2netv2")),
        device=str(embedding.get("device", "cuda:0")),
        batch_size=int(embedding.get("batch_size", 64)),
        chunk_duration_sec=float(embedding.get("chunk_duration_sec", 1.5)),
        chunk_step_sec=float(embedding.get("chunk_step_sec", 0.75)),
        checkpoint_name=str(
            embedding.get("checkpoint_name", "pretrained_eres2netv2w24s4ep4.ckpt")
        ),
        configured_duration_sec=float(sample["duration_sec"]),
    )


def _speaker_decision_confidence_summary(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    by_mode: dict[str, list[float]] = defaultdict(list)
    for event in events:
        by_mode[str(event.get("mode", "unknown"))].append(
            float(event.get("confidence", 0.0))
        )
    return {
        mode: {
            "count": len(values),
            "min": min(values),
            "mean": sum(values) / len(values),
            "max": max(values),
        }
        for mode, values in sorted(by_mode.items())
    }


def run_rolling_asr(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    embedding_store: Optional[CausalSegmentStore] = None,
    attribution_observer: Any = None,
    stage_profiler: Optional[Callable[[Mapping[str, float]], None]] = None,
) -> dict[str, Any]:
    normalized = validate_config(config, require_embedding=False, require_association=False)
    sample = _mapping(normalized, "sample")
    stream = _mapping(normalized, "stream")
    inference = _mapping(normalized, "inference")
    speaker_mapping = normalized.get("speaker_mapping", {})
    if not isinstance(speaker_mapping, Mapping):
        raise ValueError("config.speaker_mapping must be a mapping")
    association = normalized.get("association", {})
    if not isinstance(association, Mapping):
        association = {}
    embedding_config = normalized.get("embedding", {})
    if not isinstance(embedding_config, Mapping):
        embedding_config = {}
    destination = Path(output_dir).expanduser().resolve()
    # A caller may pass its own passive attribution observer; none is built in.
    transcript_turn_observer: object | None = attribution_observer
    owns_transcript_turn_observer = False
    speaker_binding_runtime = SpeakerBindingRuntime()
    postcommit_assigner = build_speaker_binder(
        sample=sample,
        association=association,
        speaker_mapping=speaker_mapping,
        embedding_config=embedding_config,
        embedding_store=embedding_store,
        runtime=speaker_binding_runtime,
    )
    decoder = MossWindowDecoder(
        audio_path=Path(str(sample["audio_path"])),
        endpoint=str(normalized["endpoint"]),
        model=str(normalized["served_model_name"]),
        prompt=str(inference.get("prompt", "")),
        max_new_tokens=int(inference.get("window_max_new_tokens", 4096)),
        short_window_max_new_tokens=int(inference.get("short_window_max_new_tokens", 512)),
        short_window_threshold_sec=float(inference.get("short_window_threshold_sec", 9.0)),
        timeout_sec=_decoder_timeout_sec(inference, stream),
        sse_soft_deadline_sec=float(inference.get("sse_soft_deadline_sec", 0.0)),
        causal_frontier_request_ids=bool(
            inference.get("causal_frontier_request_ids", False)
        ),
        causal_frontier_time_warp=bool(
            inference.get("causal_frontier_time_warp", False)
        ),
        attribution_observer=transcript_turn_observer,
    )
    committer = build_speaker_binding_committer(
        stream=stream,
        association=association,
        assigner=postcommit_assigner,
        runtime=speaker_binding_runtime,
    )
    def map_segments(segments: Any, window: Any) -> list[Segment]:
        del window
        return list(segments)

    def after_update(
        window: Any, _decoded: Any, _mapped: Any, update: Any
    ) -> Mapping[str, Any] | None:
        del window, update
        return speaker_binding_runtime.event_metadata()

    engine = IncrementalEngine(
        duration_sec=float(sample["duration_sec"]),
        step_sec=float(stream["step_sec"]),
        context_sec=float(stream["context_sec"]),
        right_context_sec=float(stream["right_context_sec"]),
        decode=decoder,
        map_segments=map_segments,
        after_update=after_update,
        committer=committer,
        max_wall_latency_sec=(
            None
            if stream.get("max_wall_latency_sec") in (None, "")
            else float(stream["max_wall_latency_sec"])
        ),
        continue_on_decode_error=bool(
            inference.get("continue_on_decode_error", True)
        ),
        recoverable_decode_error_filter=_is_recoverable_decode_error,
        sample_metadata={
            "audio_name": sample.get("audio_name", sample.get("id", "sample")),
            "audio_path": sample.get("audio_path", ""),
            "language": sample.get("language"),
        },
        attribution_observer=transcript_turn_observer,
        stage_profiler=stage_profiler,
    )
    try:
        result = engine.run(destination)
    finally:
        if owns_transcript_turn_observer:
            finalize_observer = getattr(transcript_turn_observer, "finalize", None)
            if callable(finalize_observer):
                try:
                    finalize_observer()
                except Exception:
                    pass
    local_committed = tuple(result.committed)
    final_segments = list(
        postcommit_assigner.view(local_committed)
        if postcommit_assigner is not None
        else local_committed
    )
    result.committed[:] = final_segments
    raw_committed_segment_count = len(local_committed)
    final_committed_segment_count = len(final_segments)
    observed_speech = bool(result.validation.get("observed_speech"))
    observed_material_content = bool(
        result.validation.get("observed_material_content", observed_speech)
    )
    result.validation["raw_committed_segment_count"] = raw_committed_segment_count
    result.validation["committed_segment_count"] = final_committed_segment_count
    result.validation["final_content_gate_pass"] = bool(
        (not observed_material_content) or final_committed_segment_count > 0
    )
    result.validation["postcommit_speaker_assignment_count"] = (
        sum(1 for row in result.rows for _ in row.get("speaker_assignments", []))
        if postcommit_assigner is not None
        else 0
    )
    decision_events = [
        dict(event)
        for row in result.rows
        for event in row.get("speaker_decision_events", [])
    ]
    resolution_events = [
        dict(event)
        for row in result.rows
        for event in row.get("speaker_resolution_events", [])
    ]
    active_identity_decision_sequence = [
        {
            key: event.get(key)
            for key in _ACTIVE_IDENTITY_EVENT_FIELDS
            if key in event
        }
        for event in resolution_events
        if event.get("decision_model") == "eg_identity_gp_hungarian"
    ]
    publication_ledger_snapshot = (
        postcommit_assigner.publication_ledger_snapshot()
        if postcommit_assigner is not None
        else []
    )
    embedding_summary = (
        getattr(embedding_store, "summary", None)
        if embedding_store is not None
        else None
    )
    embedding_identity = (
        embedding_summary()
        if callable(embedding_summary)
        else {}
    )
    result.validation["active_identity_decision_count"] = len(
        active_identity_decision_sequence
    )
    result.validation["active_identity_decision_sha256"] = _canonical_sha256(
        active_identity_decision_sequence
    )
    result.validation["publication_ledger_sha256"] = _canonical_sha256(
        publication_ledger_snapshot
    )
    result.validation["final_authoritative_identity_digest"] = (
        postcommit_assigner.final_authoritative_identity_digest()
        if postcommit_assigner is not None
        else _canonical_sha256({})
    )
    result.validation["embedding_input_sequence_count"] = int(
        embedding_identity.get("embedding_input_sequence_count", 0) or 0
    )
    result.validation["embedding_input_sequence_sha256"] = str(
        embedding_identity.get("embedding_input_sequence_sha256", "")
    )
    result.validation["process_peak_rss_kib"] = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )
    decision_mode_counts: dict[str, int] = {}
    for event in decision_events:
        mode = str(event.get("mode", "unknown"))
        decision_mode_counts[mode] = decision_mode_counts.get(mode, 0) + 1
    max_speaker_wait_ticks = max(
        (int(event.get("wait_ticks", 0)) for event in decision_events),
        default=0,
    )
    result.validation["speaker_decision_count"] = len(decision_events)
    result.validation["speaker_decision_mode_counts"] = decision_mode_counts
    result.validation["speaker_decision_mode_confidence"] = (
        _speaker_decision_confidence_summary(decision_events)
    )
    result.validation["max_speaker_wait_ticks"] = max_speaker_wait_ticks
    result.validation["speaker_wait_tick_gate_pass"] = bool(
        postcommit_assigner is None
        or max_speaker_wait_ticks <= postcommit_assigner.best_effort_after_ticks
    )
    result.validation["published_segment_count"] = (
        postcommit_assigner.published_segment_count
        if postcommit_assigner is not None
        else final_committed_segment_count
    )
    result.validation["committed_equals_published"] = bool(
        raw_committed_segment_count
        == result.validation["published_segment_count"]
    )
    pending_speaker_count = (
        postcommit_assigner.pending_speaker_count()
        if postcommit_assigner is not None
        else 0
    )
    pending_segment_count = (
        len(postcommit_assigner.pending_segments())
        if postcommit_assigner is not None
        else 0
    )
    decision_source_keys = [
        str(event.get("source_key", "")) for event in decision_events
    ]
    result.validation["pending_speaker_count"] = pending_speaker_count
    result.validation["pending_segment_count"] = pending_segment_count
    result.validation["duplicate_publication_count"] = (
        len(decision_source_keys) - len(set(decision_source_keys))
    )
    result.validation["silent_speaker_fallback_count"] = (
        max(
            0,
            int(result.validation["published_segment_count"])
            - len(decision_events),
        )
        if postcommit_assigner is not None
        else 0
    )
    result.validation["eos_novelty_count"] = sum(
        1
        for event in decision_events
        if bool(event.get("eos"))
        and str(event.get("mode")) == "deadline"
        and str(event.get("status")) == "new"
        and str(event.get("evidence")) != "bootstrap"
    )
    if postcommit_assigner is not None and any(
        postcommit_assigner.centroid_counters.values()
    ):
        result.validation["centroid_correction_counters"] = (
            postcommit_assigner.centroid_counters
        )
    result.validation["postcommit_embedding_call_count"] = (
        postcommit_assigner.embedding_call_count
        if postcommit_assigner is not None
        else 0
    )
    result.validation["postcommit_embedding_available_count"] = (
        postcommit_assigner.embedding_available_count
        if postcommit_assigner is not None
        else 0
    )
    postcommit_counters = (
        postcommit_assigner.core_counters
        if postcommit_assigner is not None
        else {
            "eligible": 0,
            "skipped_short": 0,
            "skipped_overlap": 0,
            "called": 0,
            "available": 0,
            "accepted": 0,
            "rejected": 0,
        }
    )
    result.validation["postcommit_embedding_counters"] = postcommit_counters
    result.validation["speaker_policy"] = str(
        association.get("policy", LEGO_POLICY)
    )
    # Flat aliases make validation artifacts easy to consume while retaining
    # one canonical nested counter object for summaries.
    for counter_name, counter_value in postcommit_counters.items():
        result.validation[f"postcommit_embedding_{counter_name}"] = int(counter_value)
    speaker_assignment_gate_pass = bool(
        postcommit_assigner is None
        or (
            pending_speaker_count == 0
            and pending_segment_count == 0
            and postcommit_assigner.is_fully_assigned(result.committed)
            and result.validation["committed_equals_published"]
            and result.validation["speaker_wait_tick_gate_pass"]
        )
    )
    result.validation["speaker_assignment_gate_pass"] = speaker_assignment_gate_pass
    result.validation["postprocess_gate_pass"] = True
    result.validation["status"] = (
        "passed"
        if result.validation["final_content_gate_pass"]
        and speaker_assignment_gate_pass
        else "failed"
    )
    _write_json(destination / "stream_validation.json", result.validation)
    item = {
        "audio_name": sample.get("audio_name", sample.get("id", "sample")),
        "audio_path": sample.get("audio_path", ""),
        "segments": [item.to_dict() for item in result.committed],
    }
    _write_json(destination / "hypothesis.json", [item])
    require_publishable(destination, mode="incremental_asr")
    return {
        "mode": "incremental_asr",
        "status": result.validation["status"],
        "results": str(destination / "results" / "ticks.jsonl"),
        "validation": str(destination / "stream_validation.json"),
        "hypothesis": str(destination / "hypothesis.json"),
        "committed_segments": len(result.committed),
        "duplicate_suppressed": engine.committer.suppressed_count,
        "speaker_count": len({segment.speaker for segment in result.committed}),
        "confirmed_speaker_count": (
            postcommit_assigner.confirmed_speaker_count()
            if postcommit_assigner is not None
            else len({segment.speaker for segment in result.committed})
        ),
        "pending_speaker_count": (
            postcommit_assigner.pending_speaker_count()
            if postcommit_assigner is not None
            else 0
        ),
        "postcommit_speaker_assignment_count": (
            result.validation.get("postcommit_speaker_assignment_count", 0)
        ),
        "postcommit_embedding_call_count": (
            postcommit_assigner.embedding_call_count
            if postcommit_assigner is not None
            else 0
        ),
        "postcommit_embedding_counters": postcommit_counters,
        "speaker_decision_count": result.validation["speaker_decision_count"],
        "speaker_decision_mode_counts": decision_mode_counts,
        "speaker_decision_mode_confidence": result.validation[
            "speaker_decision_mode_confidence"
        ],
        "max_speaker_wait_ticks": max_speaker_wait_ticks,
        "published_segment_count": result.validation["published_segment_count"],
        "pending_segment_count": pending_segment_count,
        "duplicate_publication_count": result.validation[
            "duplicate_publication_count"
        ],
        "silent_speaker_fallback_count": result.validation[
            "silent_speaker_fallback_count"
        ],
        "eos_novelty_count": result.validation["eos_novelty_count"],
        "speaker_policy": str(
            association.get("policy", LEGO_POLICY)
        ),
    }


def run_causal_stream(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    attribution_observer: Any = None,
    stage_profiler: Optional[Callable[[Mapping[str, float]], None]] = None,
) -> dict[str, Any]:
    """Run strict causal streaming with committed-segment speaker embeddings."""

    normalized = validate_config(
        config,
        require_embedding=True,
        require_association=True,
    )
    destination = Path(output_dir).expanduser().resolve()
    store = _build_causal_segment_store(normalized)
    stream = run_rolling_asr(
        normalized,
        destination,
        embedding_store=store,
        attribution_observer=attribution_observer,
        stage_profiler=stage_profiler,
    )
    association_summary = _materialize_causal_artifact(destination)
    return {
        "mode": "global_embedding",
        "stream": stream,
        "embedding": store.summary(),
        "association": association_summary,
        "association_mode": "causal_embedding",
        "speaker_policy": str(
            _mapping(normalized, "association").get(
                "policy",
                LEGO_POLICY,
            )
        ),
    }


def _materialize_causal_artifact(
    output_dir: str | Path,
) -> dict[str, Any]:
    """Materialize the already-published append-only stream without reassignment."""

    destination = Path(output_dir).expanduser().resolve()
    require_publishable(destination, mode="incremental_asr")
    source = destination / "hypothesis.json"
    hypothesis = json.loads(source.read_text(encoding="utf-8"))
    validation_path = destination / "stream_validation.json"
    validation: Mapping[str, Any] = {}
    if validation_path.is_file():
        try:
            loaded = json.loads(validation_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                validation = loaded
        except (OSError, json.JSONDecodeError):
            pass

    output_root = destination / "global_embedding"
    _write_json(output_root / "hypothesis.json", hypothesis)
    summary = {
        "mode": "global_embedding",
        "association_mode": "causal_embedding",
        "speaker_policy": str(
            validation.get("speaker_policy", LEGO_POLICY)
        ),
        "causal": True,
        "text_timestamp_identity": True,
        "speaker_identity_preserved": True,
        "embedding_mode": "committed_segment",
        "embedding_query_count": int(
            validation.get("postcommit_embedding_call_count", 0)
        ),
        "embedding_available_count": int(
            validation.get("postcommit_embedding_available_count", 0)
        ),
        "postcommit_speaker_assignment_count": int(
            validation.get("postcommit_speaker_assignment_count", 0)
        ),
        "postcommit_embedding_counters": dict(
            validation.get("postcommit_embedding_counters", {})
        )
        if isinstance(validation.get("postcommit_embedding_counters"), Mapping)
        else {
            "eligible": int(validation.get("postcommit_embedding_eligible", 0)),
            "skipped_short": int(validation.get("postcommit_embedding_skipped_short", 0)),
            "skipped_overlap": int(validation.get("postcommit_embedding_skipped_overlap", 0)),
            "called": int(validation.get("postcommit_embedding_called", 0)),
            "available": int(validation.get("postcommit_embedding_available", 0)),
            "accepted": int(validation.get("postcommit_embedding_accepted", 0)),
            "rejected": int(validation.get("postcommit_embedding_rejected", 0)),
        },
        "stream_validation_path": str(validation_path),
    }
    _write_json(output_root / "summary.json", summary)
    require_publishable(destination, mode="global_embedding")
    return summary


def run_pipeline(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    attribution_observer: Any = None,
    stage_profiler: Optional[Callable[[Mapping[str, float]], None]] = None,
) -> dict[str, Any]:
    """Canonical strict-online global-embedding entrypoint."""

    normalized = validate_config(config)
    destination = Path(output_dir).expanduser().resolve()
    _remove_hypotheses(destination)
    (destination / "pipeline_validation.json").unlink(missing_ok=True)
    try:
        return run_causal_stream(
            normalized,
            destination,
            attribution_observer=attribution_observer,
            stage_profiler=stage_profiler,
        )
    except Exception as exc:
        _mark_pipeline_failure(destination, exc, stage="causal_pipeline")
        raise


__all__ = [
    "run_causal_stream",
    "run_pipeline",
    "run_rolling_asr",
    "validate_config",
]
