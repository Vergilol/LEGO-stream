from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .assignment import maximum_weight_assignment
from .identity_evidence import (
    BridgeEvidence,
    EmbeddingViewSet,
    EvidenceAssessment,
    EvidenceCandidate,
    EvidencePermissions,
    IdentityEvidenceLedger,
    ProfileState,
    VoiceProfile,
    assess_embedding_views,
)
from .state_digest import authoritative_digest, defensive_snapshot
from .lego_identity import (
    ProfileSnapshot,
    LegoIdentityConfig,
    LegoIdentityModel,
    LegoIdentityObservation,
)


# How a turn is assigned to a global speaker.  ``LEGO_POLICY`` is what a run
# uses: the runtime falls back to it when ``association.policy`` is absent, and
# it is the one the paper's results use.  The other two are the paper's
# identity-policy ablation -- why a constrained assignment is needed rather than
# a threshold-greedy rule -- and are kept so that table stays reproducible.
#
#   FIXED_COSINE_POLICY  greedy nearest centroid above a fixed cosine threshold
#                        ("fixed-cosine greedy" in the paper)
#   CLEAN_CORE_POLICY    evidence restricted to clean cores
#                        ("clean-core empirical" in the paper)
FIXED_COSINE_POLICY = "fixed-cosine"
CLEAN_CORE_POLICY = "clean-core"
LEGO_POLICY = "lego"
SPEAKER_MATCH_POLICIES = frozenset(
    {
        FIXED_COSINE_POLICY,
        CLEAN_CORE_POLICY,
        LEGO_POLICY,
    }
)
MODEL_OWNED_IDENTITY_POLICIES = frozenset({LEGO_POLICY})
# Deliberately the same set, and derived rather than re-spelled so the two
# guards cannot drift apart: the constrained assignment is the only policy that
# both owns identity inside the model and reads more than one embedding view.
MULTIVIEW_IDENTITY_POLICIES = MODEL_OWNED_IDENTITY_POLICIES
_PROFILE_RELATIVE_ACCEPTANCE_RATIO = 0.80


@dataclass(frozen=True)
class Observation:
    local_symbol: str
    start: float
    end: float
    tick: float
    embedding: Optional[np.ndarray] = None
    embedding_views: Optional[EmbeddingViewSet] = None
    quality: float = 1.0
    source_key: Optional[str] = None
    revision: int = 0
    acoustic_start: Optional[float] = None
    acoustic_end: Optional[float] = None
    text: Optional[str] = None

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))

    @property
    def acoustic_interval(self) -> tuple[float, float]:
        if self.acoustic_start is None or self.acoustic_end is None:
            return float(self.start), float(self.end)
        return float(self.acoustic_start), float(self.acoustic_end)


@dataclass(frozen=True)
class SpeakerDecision:
    status: str
    track_id: str
    output_label: Optional[str]
    local_symbol: str
    score: float


@dataclass(frozen=True)
class UnprofiledOutputLabelClaim:
    """Causal current-snapshot evidence for profiling one output-label slot."""

    source_track_id: str
    target_output_label: Optional[str]
    context_score: float
    row_margin: float
    column_margin: float
    context_candidates: tuple[tuple[str, float], ...]
    context_source: str = "current_snapshot_overlap"
    row_unique: bool = False
    column_unique: bool = False
    selected_by_context_assignment: bool = False
    conflict_reason: Optional[str] = None
    ancestor_lineage: Optional[str] = None


@dataclass
class _Track:
    track_id: str
    max_core_samples: int
    core_min_duration_sec: float
    core_min_quality: float
    min_cohesion: float
    adaptive_evidence_model: bool
    bayesian_evidence_model: bool = False
    output_label: Optional[str] = None
    aliases: Set[str] = field(default_factory=set)
    intervals: List[tuple[float, float]] = field(default_factory=list)
    ticks: Set[float] = field(default_factory=set)
    vectors: List[np.ndarray] = field(default_factory=list)
    profile: VoiceProfile = field(init=False)
    support_sec: float = 0.0
    evidence: Dict[str, Observation] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.profile = VoiceProfile(
            max_samples=self.max_core_samples,
            min_profile_duration_sec=self.core_min_duration_sec,
            min_purity=self.core_min_quality,
            min_cohesion=self.min_cohesion,
        )

    @staticmethod
    def _normalized_embedding(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if value is None:
            return None
        vector = np.asarray(value, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if vector.ndim != 1 or norm <= 0.0 or not np.all(np.isfinite(vector)):
            return None
        return vector / norm

    @staticmethod
    def _centroid(vectors: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        if not vectors:
            return None
        vector = np.mean(np.stack(list(vectors)), axis=0)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0.0 else None

    @property
    def core_vectors(self) -> List[np.ndarray]:
        """Active voice-profile vectors used by identity matching."""

        return [vector.copy() for vector in self.profile.vectors]

    def _rebuild(self) -> None:
        self.aliases = set()
        self.intervals = []
        self.ticks = set()
        self.vectors = []
        candidates: List[EvidenceCandidate] = []
        for key, item in self.evidence.items():
            self.aliases.add(item.local_symbol)
            self.intervals.append((float(item.start), float(item.end)))
            self.ticks.add(float(item.tick))
            vector = self._normalized_embedding(item.embedding)
            if vector is not None:
                self.vectors.append(vector)
                acoustic_start, acoustic_end = item.acoustic_interval
                if not self.bayesian_evidence_model:
                    candidates.append(
                        EvidenceCandidate(
                            source_key=str(key),
                            vector=vector,
                            tick=float(item.tick),
                            clean_duration_sec=max(
                                0.0,
                                acoustic_end - acoustic_start,
                            ),
                            text=(
                                item.text if self.adaptive_evidence_model else None
                            ),
                            purity=float(item.quality),
                        )
                    )
        if not self.bayesian_evidence_model:
            self.profile.derive(candidates)
        # Support is the union of absolute audio intervals.  Re-running a
        # window or revising its local symbol must not manufacture evidence.
        merged = []
        for start, end in sorted(self.intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        self.support_sec = sum(end - start for start, end in merged)

    def add(self, observation: Observation) -> bool:
        key = observation.source_key or (
            "tick:%s:%0.3f:%0.3f" % (observation.tick, observation.start, observation.end)
        )
        previous = self.evidence.get(key)
        if previous is not None and int(observation.revision) <= int(previous.revision):
            return False
        self.evidence[key] = observation
        self._rebuild()
        return True

    def remove(self, source_key: str) -> Optional[Observation]:
        previous = self.evidence.pop(str(source_key), None)
        if previous is not None:
            self._rebuild()
        return previous

    def centroid(self) -> Optional[np.ndarray]:
        return self._centroid(self.core_vectors)

    def within_profile_cohesion_floor(self) -> Optional[float]:
        """Return the weakest nearest-neighbour link inside this profile."""

        vectors = self.core_vectors
        if len(vectors) < 2:
            return None
        nearest_neighbour_scores = [
            max(
                float(np.dot(vector, other))
                for other_index, other in enumerate(vectors)
                if other_index != index
            )
            for index, vector in enumerate(vectors)
        ]
        return float(min(nearest_neighbour_scores))

    def max_core_similarity(self, other: "_Track") -> Optional[float]:
        left = self.core_vectors
        right = other.core_vectors
        if not left or not right:
            return None
        return float(max(np.dot(a, b) for a in left for b in right))


class SpeakerRegistry:
    """Revision-tolerant local-to-global speaker registry.

    Unknown local labels are compared with every existing track.  A short
    observation may create an internal pending track, but it cannot create a
    global identity carrying an output label until independent support is
    sufficient.  Internal track IDs are ``G%03d`` -- they are G-side handles,
    not output labels; the output label is allocated separately and is what a
    listener ever sees.
    """

    def __init__(
        self,
        *,
        min_support_sec: float = 1.0,
        min_observations: int = 2,
        min_overlap_sec: float = 0.1,
        min_cosine: float = 0.70,
        different_local_merge_cosine: float = 0.90,
        same_local_split_cosine: float = 0.30,
        # ``S`` rather than ``O``: output labels share the decoder's surface form
        # so that a transcript reads uniformly, and ``canonical_speaker_label``
        # only accepts ``S``/``G`` ordinals. The paper calls this layer O; the
        # emitted string is still Sxx.
        output_prefix: str = "S",
        max_core_samples_per_track: int = 120,
        core_min_duration_sec: float = 0.75,
        core_min_quality: float = 0.25,
        require_core_for_confirmation: bool = False,
        # The delivered policy, so a bare ``SpeakerRegistry()`` behaves like a
        # default run; the two ablation arms are always named explicitly.
        match_policy: str = LEGO_POLICY,
        embedding_dimension: int = 192,
        robust_exemplar_capacity: int = 4,
        risk_component_capacity: int = 8,
        risk_challenger_capacity: int = 2,
        risk_local_partition_provisional_enabled: bool = True,
    ) -> None:
        if min_support_sec < 0.0 or not math.isfinite(float(min_support_sec)):
            raise ValueError("min_support_sec must be finite and non-negative")
        if min_observations < 1:
            raise ValueError("min_observations must be positive")
        if min_overlap_sec < 0.0 or not math.isfinite(float(min_overlap_sec)):
            raise ValueError("min_overlap_sec must be finite and non-negative")
        if not 0.0 <= min_cosine <= 1.0:
            raise ValueError("min_cosine must be in [0, 1]")
        if not 0.0 <= different_local_merge_cosine <= 1.0:
            raise ValueError("different_local_merge_cosine must be in [0, 1]")
        if not 0.0 <= same_local_split_cosine <= 1.0:
            raise ValueError("same_local_split_cosine must be in [0, 1]")
        if not str(output_prefix).strip() or not str(output_prefix).isalnum():
            raise ValueError("output_prefix must be a non-empty alphanumeric string")
        if (
            isinstance(max_core_samples_per_track, bool)
            or int(max_core_samples_per_track) != max_core_samples_per_track
            or int(max_core_samples_per_track) < 1
        ):
            raise ValueError("max_core_samples_per_track must be a positive integer")
        for value, name in (
            (core_min_duration_sec, "core_min_duration_sec"),
            (core_min_quality, "core_min_quality"),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(require_core_for_confirmation, bool):
            raise ValueError("require_core_for_confirmation must be a bool")
        if not isinstance(risk_local_partition_provisional_enabled, bool):
            raise ValueError(
                "risk_local_partition_provisional_enabled must be a bool"
            )
        normalized_policy = str(match_policy).strip()
        if normalized_policy not in SPEAKER_MATCH_POLICIES:
            raise ValueError(
                "match_policy must be one of: "
                + ", ".join(sorted(SPEAKER_MATCH_POLICIES))
            )
        self.min_support_sec = float(min_support_sec)
        self.min_observations = int(min_observations)
        self.min_overlap_sec = float(min_overlap_sec)
        self.min_cosine = float(min_cosine)
        self.different_local_merge_cosine = float(different_local_merge_cosine)
        self.same_local_split_cosine = float(same_local_split_cosine)
        self.output_prefix = str(output_prefix)
        self.max_core_samples_per_track = int(max_core_samples_per_track)
        self.core_min_duration_sec = float(core_min_duration_sec)
        self.core_min_quality = float(core_min_quality)
        self.require_core_for_confirmation = bool(require_core_for_confirmation)
        self.match_policy = normalized_policy
        self.embedding_dimension = int(embedding_dimension)
        self.lego_identity_model = (
            LegoIdentityModel(
                dimension=self.embedding_dimension,
                exemplar_capacity=int(robust_exemplar_capacity),
                component_capacity=int(risk_component_capacity),
                challenger_capacity=int(risk_challenger_capacity),
                config=LegoIdentityConfig(
                    local_partition_provisional_enabled=(
                        risk_local_partition_provisional_enabled
                    )
                ),
            )
            if self.match_policy == LEGO_POLICY
            else None
        )
        self.tracks: Dict[str, _Track] = {}
        self.output_mapping: Dict[str, str] = {}
        self._next_track = 1
        self._next_output = 1
        self._last_resolution_events: tuple[dict[str, object], ...] = ()
        self._last_resolution_source_keys: dict[str, tuple[str, ...]] = {}
        self.evidence_ledger = IdentityEvidenceLedger()

    @property
    def last_resolution_events(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(event) for event in self._last_resolution_events)

    @property
    def last_resolution_source_keys(self) -> dict[str, tuple[str, ...]]:
        return dict(self._last_resolution_source_keys)

    def _new_track(self) -> _Track:
        track = _Track(
            track_id="G%03d" % self._next_track,
            max_core_samples=self.max_core_samples_per_track,
            core_min_duration_sec=self.core_min_duration_sec,
            core_min_quality=self.core_min_quality,
            min_cohesion=self.min_cosine,
            adaptive_evidence_model=self.match_policy == CLEAN_CORE_POLICY,
            bayesian_evidence_model=(
                self.match_policy in MODEL_OWNED_IDENTITY_POLICIES
            ),
        )
        self._next_track += 1
        self.tracks[track.track_id] = track
        return track

    def observe(self, observation: Observation) -> SpeakerDecision:
        decisions = self.observe_batch(
            [observation],
            allow_output_tracks=self.match_policy
            in {FIXED_COSINE_POLICY, CLEAN_CORE_POLICY},
        )
        return decisions[0]

    def _batch_groups(
        self,
        values: Sequence[Observation],
    ) -> List[Tuple[str, str, List[Tuple[int, Observation]]]]:
        """Return deterministic local/acoustic assignment units for one snapshot.

        A local label is useful evidence, but it is not an identity guarantee.
        Under the empirical policy, mutually incompatible clean cores from the
        same local label are separated before assignment.  Short/no-core items
        join a sole acoustic group; when several voices are already present,
        they remain a separate unresolved group instead of contaminating one.
        """

        by_local: Dict[str, List[Tuple[int, Observation]]] = {}
        local_order: List[str] = []
        for index, observation in enumerate(values):
            local = str(observation.local_symbol)
            if local not in by_local:
                by_local[local] = []
                local_order.append(local)
            by_local[local].append((index, observation))

        groups: List[Tuple[str, str, List[Tuple[int, Observation]]]] = []
        for local in sorted(local_order):
            items = by_local[local]
            if self.match_policy != CLEAN_CORE_POLICY:
                groups.append((local, local, items))
                continue

            core_groups: List[
                Tuple[List[Tuple[int, Observation]], List[np.ndarray]]
            ] = []
            no_core: List[Tuple[int, Observation]] = []
            for item in items:
                observation = item[1]
                vector = _Track._normalized_embedding(observation.embedding)
                if not self._is_core_observation(observation) or vector is None:
                    no_core.append(item)
                    continue
                for grouped_items, grouped_vectors in core_groups:
                    if all(
                        float(np.dot(vector, other)) + 1e-12 >= self.min_cosine
                        for other in grouped_vectors
                    ):
                        grouped_items.append(item)
                        grouped_vectors.append(vector)
                        break
                else:
                    core_groups.append(([item], [vector]))

            if not core_groups:
                acoustic_groups = [items]
            else:
                acoustic_groups = [grouped_items for grouped_items, _ in core_groups]
                if len(acoustic_groups) == 1:
                    acoustic_groups[0].extend(no_core)
                elif no_core:
                    acoustic_groups.append(no_core)
            for group_index, grouped_items in enumerate(acoustic_groups):
                groups.append(
                    (
                        f"{local}\x1f{group_index}",
                        local,
                        sorted(grouped_items, key=lambda item: item[0]),
                    )
                )
        return groups

    def observe_batch(
        self,
        observations: Iterable[Observation],
        *,
        allow_output_tracks: bool = True,
        excluded_tracks_by_local: Optional[Mapping[str, Iterable[str]]] = None,
        defer_output_resolution: bool = False,
    ) -> List[SpeakerDecision]:
        """Resolve one snapshot with a one-track-per-local-speaker assignment.

        The old implementation was order-dependent: an ambiguous first local
        speaker could greedily consume a track that a later, high-confidence
        observation clearly needed. We solve one polynomial maximum-weight
        bipartite assignment for every speaker count and return decisions in
        the same order as the input iterable.
        """

        values = list(observations)
        if not values:
            return []
        output_allowed = bool(allow_output_tracks)
        for observation in values:
            self._validate_observation(observation)
            if (
                self.match_policy in MODEL_OWNED_IDENTITY_POLICIES
                or not output_allowed
            ):
                self._assert_generic_source_owner_allowed(observation)
        batch_groups = self._batch_groups(values)
        order = [group_id for group_id, _local, _items in batch_groups]
        group_local = {
            group_id: local for group_id, local, _items in batch_groups
        }
        groups = {
            group_id: items for group_id, _local, items in batch_groups
        }
        excluded_tracks = {
            str(local): {str(track_id) for track_id in track_ids}
            for local, track_ids in (excluded_tracks_by_local or {}).items()
        }

        if self.match_policy in MODEL_OWNED_IDENTITY_POLICIES:
            decisions: List[Optional[SpeakerDecision]] = [None] * len(values)
            for group_id in order:
                track = self._new_track()
                for index, observation in groups[group_id]:
                    decisions[index] = self._record_observation(
                        track,
                        observation,
                        score=0.0,
                        _defer_empty_cleanup=True,
                        _defer_output_resolution=True,
                    )
            for track in list(self.tracks.values()):
                self._drop_track_if_empty(track)
            return self._finalize_batch_decision_owners(values, decisions)

        track_ids = sorted(
            track_id
            for track_id, track in self.tracks.items()
            if output_allowed or track.output_label is None
        )
        scores: Dict[Tuple[str, str], float] = {}
        for group_id in order:
            local = group_local[group_id]
            for track_id in track_ids:
                if track_id in excluded_tracks.get(local, set()):
                    scores[(group_id, track_id)] = -1.0
                    continue
                scores[(group_id, track_id)] = max(
                    (
                        self._track_score(self.tracks[track_id], item)
                        for _, item in groups[group_id]
                    ),
                    default=-1.0,
                )

        if self.match_policy == CLEAN_CORE_POLICY:
            for group_id in order:
                output_scores = [
                    (scores[(group_id, track_id)], track_id)
                    for track_id in track_ids
                    if self.tracks[track_id].output_label is not None
                    and scores[(group_id, track_id)] >= 0.0
                ]
                best_output = self._unique_best(output_scores)
                for _score, track_id in output_scores:
                    if best_output is None or track_id != best_output[1]:
                        scores[(group_id, track_id)] = -1.0

        assignment: Dict[str, Optional[str]] = {
            group_id: None for group_id in order
        }
        assignment.update(maximum_weight_assignment(order, track_ids, scores))

        decisions: List[Optional[SpeakerDecision]] = [None] * len(values)
        for group_id in order:
            track_id = assignment.get(group_id)
            if track_id is None:
                track = self._new_track()
                track_id = track.track_id
            track = self.tracks[track_id]
            for index, observation in groups[group_id]:
                score = scores.get((group_id, track_id), 0.0)
                decision = self._record_observation(
                    track,
                    observation,
                    score=score,
                    _defer_empty_cleanup=True,
                    _defer_output_resolution=defer_output_resolution,
                )
                decisions[index] = decision
        for track in list(self.tracks.values()):
            self._drop_track_if_empty(track)
        return self._finalize_batch_decision_owners(values, decisions)

    def _finalize_batch_decision_owners(
        self,
        values: Sequence[Observation],
        decisions: Sequence[Optional[SpeakerDecision]],
    ) -> List[SpeakerDecision]:
        resolved: List[SpeakerDecision] = []
        for index, decision in enumerate(decisions):
            if decision is None:
                continue
            observation = values[index]
            owners = self._source_owners(observation)
            if owners:
                # A later revision in the same batch may have moved this
                # observation's source key after its original decision was
                # produced.  Return the final owner so callers never receive
                # a dangling/obsolete track id.
                owner, _ = max(
                    owners,
                    key=lambda item: (int(item[1].revision), item[0].track_id),
                )
                if owner.track_id != decision.track_id:
                    decision = SpeakerDecision(
                        status="confirmed" if owner.output_label is not None else "pending",
                        track_id=owner.track_id,
                        output_label=owner.output_label,
                        local_symbol=observation.local_symbol,
                        score=decision.score,
                    )
            resolved.append(decision)
        return resolved

    @staticmethod
    def _validate_observation(observation: Observation) -> None:
        if observation.end < observation.start or observation.start < 0.0:
            raise ValueError("invalid speaker observation interval")
        if not all(
            math.isfinite(float(value))
            for value in (
                observation.start,
                observation.end,
                observation.tick,
                observation.quality,
            )
        ):
            raise ValueError("speaker observation values must be finite")
        if (observation.acoustic_start is None) != (observation.acoustic_end is None):
            raise ValueError("acoustic observation bounds must be provided together")
        if observation.acoustic_start is not None and observation.acoustic_end is not None:
            acoustic_start = float(observation.acoustic_start)
            acoustic_end = float(observation.acoustic_end)
            if not math.isfinite(acoustic_start) or not math.isfinite(acoustic_end):
                raise ValueError("acoustic observation bounds must be finite")
            if (
                acoustic_start < float(observation.start) - 1e-9
                or acoustic_end > float(observation.end) + 1e-9
                or acoustic_end < acoustic_start
            ):
                raise ValueError("acoustic observation bounds must lie within the segment")

    def _drop_track_if_empty(self, track: _Track) -> None:
        if track.evidence:
            return
        self.tracks.pop(track.track_id, None)
        if track.output_label is not None:
            self.output_mapping = {
                alias: output_label
                for alias, output_label in self.output_mapping.items()
                if output_label != track.output_label
            }

    @staticmethod
    def _overlap_seconds(left: Observation, right: Observation) -> float:
        return max(
            0.0,
            min(float(left.end), float(right.end))
            - max(float(left.start), float(right.start)),
        )

    def observations_cannot_link(
        self,
        left: Observation,
        right: Observation,
    ) -> bool:
        overlap = self._overlap_seconds(left, right)
        return (
            str(left.local_symbol) != str(right.local_symbol)
            and abs(float(left.tick) - float(right.tick)) <= 1e-9
            and overlap > 1e-9
            and overlap + 1e-9 >= self.min_overlap_sec
        )

    @staticmethod
    def observations_share_local_partition(
        left: Observation,
        right: Observation,
    ) -> bool:
        """Same decoder snapshot, different S: hard current-partition separation."""

        return bool(
            str(left.local_symbol) != str(right.local_symbol)
            and abs(float(left.tick) - float(right.tick)) <= 1e-9
        )

    def _source_owners(
        self,
        observation: Observation,
    ) -> List[Tuple[_Track, Observation]]:
        if observation.source_key is None:
            return []
        key = str(observation.source_key)
        return [
            (track, track.evidence[key])
            for track in self.tracks.values()
            if key in track.evidence
        ]

    def _assert_generic_source_owner_allowed(self, observation: Observation) -> None:
        if any(
            owner.output_label is not None
            for owner, _previous in self._source_owners(observation)
        ):
            raise RuntimeError(
                "generic observation cannot reuse a source key owned by a labelled track"
            )

    def _record_observation(
        self,
        track: _Track,
        observation: Observation,
        *,
        score: float,
        _defer_empty_cleanup: bool = False,
        _defer_output_resolution: bool = False,
    ) -> SpeakerDecision:
        self.evidence_ledger.append_observation(
            track_id=track.track_id,
            observation=observation,
            association_reason="record_observation",
        )
        # ``source_key`` identifies one stable audio evidence unit.  A later
        # revision may move that evidence to another acoustic track, but the
        # same key must never remain counted in both tracks.  Without this
        # ownership transfer, one corrected dense chunk can manufacture a new
        # output label while the superseded track keeps the same support.
        owners = self._source_owners(observation)
        if owners:
            latest_revision = max(int(item.revision) for _, item in owners)
            latest_tracks = sorted(
                (
                    owner
                    for owner, item in owners
                    if int(item.revision) == latest_revision
                ),
                key=lambda item: item.track_id,
            )
            target_owns_latest = any(owner is track for owner in latest_tracks)
            if int(observation.revision) < latest_revision or (
                int(observation.revision) == latest_revision
                and not target_owns_latest
            ):
                # A stale revision must not create an empty track or overwrite
                # the latest owner selected by a previous window.
                if not _defer_empty_cleanup:
                    self._drop_track_if_empty(track)
                owner = latest_tracks[0]
                return SpeakerDecision(
                    status="confirmed" if owner.output_label is not None else "pending",
                    track_id=owner.track_id,
                    output_label=owner.output_label,
                    local_symbol=observation.local_symbol,
                    score=float(score),
                )

            transferable_output_labels: List[str] = []
            source_key = str(observation.source_key)
            for owner, _previous in owners:
                if owner is track:
                    continue
                owner.remove(source_key)
                if not owner.evidence and owner.output_label is not None:
                    transferable_output_labels.append(owner.output_label)
                if not _defer_empty_cleanup:
                    self._drop_track_if_empty(owner)
            if track.output_label is None and transferable_output_labels:
                # Preserve the output label when a revision merely moves
                # the only evidence from a now-empty internal track.
                track.output_label = min(transferable_output_labels)

        track.add(observation)
        if (
            self.match_policy == FIXED_COSINE_POLICY
            and track.output_label is None
            and self._confirmed(track)
            and not _defer_output_resolution
        ):
            track.output_label = "%s%02d" % (self.output_prefix, self._next_output)
            self._next_output += 1
        if track.output_label is not None:
            self.output_mapping[observation.local_symbol] = track.output_label
            status = "confirmed"
        else:
            status = "pending"
        return SpeakerDecision(
            status=status,
            track_id=track.track_id,
            output_label=track.output_label,
            local_symbol=observation.local_symbol,
            score=float(score),
        )

    def output_label_for_track(self, track_id: str) -> Optional[str]:
        track = self.tracks.get(str(track_id))
        return track.output_label if track is not None else None

    def track_id_for_output_label(self, output_label: str) -> Optional[str]:
        target = str(output_label)
        for track in self.tracks.values():
            if track.output_label == target:
                return track.track_id
        return None

    def observe_on_output_label(
        self,
        output_label: str,
        observation: Observation,
    ) -> SpeakerDecision:
        """Record context-aligned evidence on an existing labelled track.

        This is the ownership-preserving path for context alignment:
        it never allocates a new track or changes an already published ID.
        """

        self._validate_observation(observation)
        target = str(output_label)
        track = next(
            (item for item in self.tracks.values() if item.output_label == target),
            None,
        )
        if track is None:
            raise KeyError(f"unknown output speaker: {target}")
        score = self._track_score(track, observation)
        if score < 0.0:
            score = 0.0
        return self._record_observation(track, observation, score=score)

    def observe_on_track(
        self,
        track_id: str,
        observation: Observation,
        *,
        defer_output_resolution: bool = False,
    ) -> SpeakerDecision:
        """Record snapshot-aligned evidence on one known internal track."""

        self._validate_observation(observation)
        track = self.tracks.get(str(track_id))
        if track is None:
            raise KeyError(f"unknown speaker track: {track_id}")
        score = self._track_score(track, observation)
        if score < 0.0:
            score = 0.0
        return self._record_observation(
            track,
            observation,
            score=score,
            _defer_output_resolution=defer_output_resolution,
        )

    def replace_unpublished_observation(
        self,
        track_id: str,
        source_key: str,
        observation: Observation,
        *,
        score: float = 0.0,
    ) -> SpeakerDecision:
        """Replace one withheld revision without manufacturing new evidence."""

        self._validate_observation(observation)
        track = self.tracks.get(str(track_id))
        if track is None:
            raise KeyError(f"unknown speaker track: {track_id}")
        if track.output_label is not None:
            raise RuntimeError("published speaker evidence is immutable")
        previous_key = str(source_key)
        if previous_key not in track.evidence:
            raise KeyError(f"unknown speaker source: {source_key}")
        if observation.source_key is None:
            raise ValueError("replacement observation requires a source key")
        replacement_key = str(observation.source_key)
        if replacement_key != previous_key and self._source_owners(observation):
            raise RuntimeError("replacement source is already owned")

        self.evidence_ledger.append_observation(
            track_id=track.track_id,
            observation=observation,
            association_reason="replace_unpublished_revision",
        )

        previous = track.remove(previous_key)
        if previous is None:
            raise RuntimeError("speaker source disappeared during replacement")
        if not track.add(observation):
            track.add(previous)
            raise RuntimeError("speaker replacement was not accepted")
        return SpeakerDecision(
            status="pending",
            track_id=track.track_id,
            output_label=None,
            local_symbol=observation.local_symbol,
            score=float(score),
        )

    def core_contains(self, track_id: str, source_key: str) -> bool:
        track = self.tracks.get(str(track_id))
        return bool(track is not None and track.profile.contains(str(source_key)))

    def retained_core_count(self, track_id: str) -> int:
        track = self.tracks.get(str(track_id))
        return len(track.core_vectors) if track is not None else 0

    def profile_state(self, track_id: str) -> ProfileState:
        track = self.tracks.get(str(track_id))
        if track is None:
            raise KeyError(str(track_id))
        return track.profile.state

    def observation_permissions(self, observation: Observation):
        if self.match_policy in MULTIVIEW_IDENTITY_POLICIES:
            assessment = assess_embedding_views(
                observation.embedding_views,
                independence_key=str(observation.source_key or "observation"),
                text=observation.text,
                purity=float(observation.quality),
                profile_admissible=(
                    float(observation.quality) + 1e-9 >= self.core_min_quality
                ),
                mixed=float(observation.quality) < 1.0 - 1e-9,
            )
            return EvidencePermissions(
                extractable=observation.embedding_views is not None,
                pure=float(observation.quality) >= 1.0 - 1e-9,
                routing=assessment.routing_allowed,
                profile=assessment.profile_allowed,
                novelty=assessment.novelty_allowed,
            )
        candidate = self._evidence_candidate(observation)
        if candidate is None:
            return None
        return candidate.permissions(
            min_profile_duration_sec=self.core_min_duration_sec,
            min_purity=self.core_min_quality,
        )

    def observation_evidence_summary(
        self,
        observation: Observation,
    ) -> dict[str, object]:
        if self.match_policy in MULTIVIEW_IDENTITY_POLICIES:
            assessment = assess_embedding_views(
                observation.embedding_views,
                independence_key=str(observation.source_key or "observation"),
                text=observation.text,
                purity=float(observation.quality),
                profile_admissible=(
                    float(observation.quality) + 1e-9 >= self.core_min_quality
                ),
                mixed=float(observation.quality) < 1.0 - 1e-9,
            )
            return {
                "permissions": {
                    "extractable": observation.embedding_views is not None,
                    "pure": float(observation.quality) >= 1.0 - 1e-9,
                    "routing": assessment.routing_allowed,
                    "profile": assessment.profile_allowed,
                    "novelty": assessment.novelty_allowed,
                },
                "strength": {
                    "view_count": len(assessment.views),
                    "instability": float(assessment.instability),
                    "unique_coverage_sec": float(
                        assessment.unique_coverage_sec
                    ),
                    "precision_route": float(assessment.precision_route),
                    "precision_profile": float(
                        assessment.precision_profile
                    ),
                    "precision_new": float(assessment.precision_new),
                },
            }
        candidate = self._evidence_candidate(observation)
        if candidate is None:
            return {
                "permissions": {
                    "extractable": False,
                    "pure": False,
                    "routing": False,
                    "profile": False,
                    "novelty": False,
                },
                "strength": None,
            }
        permissions = candidate.permissions(
            min_profile_duration_sec=self.core_min_duration_sec,
            min_purity=self.core_min_quality,
        )
        strength = candidate.strength()
        return {
            "permissions": {
                "extractable": permissions.extractable,
                "pure": permissions.pure,
                "routing": permissions.routing,
                "profile": permissions.profile,
                "novelty": permissions.novelty,
            },
            "strength": {
                "clean_duration_sec": strength.clean_duration_sec,
                "lexical_unit_count": strength.lexical_unit_count,
                "lexical_diversity": strength.lexical_diversity,
                "purity": strength.purity,
                "duration_coverage": strength.duration_coverage,
                "content_coverage": strength.content_coverage,
                "aggregate": strength.aggregate,
            },
        }

    def acoustic_identity_edges(
        self,
        observation: Observation,
        *,
        excluded_track_ids: Iterable[str] = (),
    ) -> tuple[dict[str, object], ...]:
        """Describe scalar acoustic edges to every identity without mutation."""

        self._validate_observation(observation)
        vector = _Track._normalized_embedding(observation.embedding)
        observation_acoustic_start, observation_acoustic_end = (
            observation.acoustic_interval
        )
        excluded = {str(track_id) for track_id in excluded_track_ids}
        edges: list[dict[str, object]] = []
        for track in sorted(self.tracks.values(), key=lambda item: item.track_id):
            best_key: Optional[str] = None
            best_score: Optional[float] = None
            if vector is not None:
                for source_key in sorted(track.profile.active_keys):
                    score = float(np.dot(track.profile.vector(source_key), vector))
                    if best_score is None or score > best_score + 1e-12:
                        best_key = source_key
                        best_score = score
            best_observation = (
                track.evidence.get(best_key) if best_key is not None else None
            )
            excluded_by_batch = track.track_id in excluded
            edges.append(
                {
                    "observation_source_key": observation.source_key,
                    "observation_local_symbol": str(observation.local_symbol),
                    "observation_start": float(observation.start),
                    "observation_end": float(observation.end),
                    "observation_acoustic_start": observation_acoustic_start,
                    "observation_acoustic_end": observation_acoustic_end,
                    "observation_tick": float(observation.tick),
                    "observation_duration_sec": float(observation.duration),
                    "observation_quality": float(observation.quality),
                    "observation_core_eligible": self._is_core_observation(
                        observation
                    ),
                    "candidate_track_id": track.track_id,
                    "candidate_status": (
                        "output" if track.output_label is not None else "pending"
                    ),
                    "candidate_output_label": track.output_label,
                    "same_local": str(observation.local_symbol) in track.aliases,
                    "cannot_link": excluded_by_batch,
                    "excluded_by_batch": excluded_by_batch,
                    "core_count": len(track.profile.active_keys),
                    "profile_state": track.profile.state.value,
                    "quarantine_count": len(track.profile.quarantine_keys),
                    "within_profile_cohesion_floor": (
                        track.within_profile_cohesion_floor()
                    ),
                    "best_cosine": best_score,
                    "best_core_source_key": best_key,
                    "best_core_start": (
                        best_observation.acoustic_interval[0]
                        if best_observation is not None
                        else None
                    ),
                    "best_core_end": (
                        best_observation.acoustic_interval[1]
                        if best_observation is not None
                        else None
                    ),
                    "best_core_local_symbol": (
                        str(best_observation.local_symbol)
                        if best_observation is not None
                        else None
                    ),
                    "best_core_quality": (
                        track.profile.quality(best_key)
                        if best_key is not None
                        else None
                    ),
                    "current_policy_score": self._track_score(track, observation),
                }
            )
        return tuple(edges)

    def _profile_audit(self, track: _Track) -> dict[str, object]:
        evidence: list[dict[str, object]] = []
        active = set(track.profile.active_keys)
        quarantine = set(track.profile.quarantine_keys)
        for source_key, observation in self._ordered_evidence(track):
            summary = self.observation_evidence_summary(observation)
            evidence.append(
                {
                    "source_key": source_key,
                    "tick": float(observation.tick),
                    "start": float(observation.acoustic_interval[0]),
                    "end": float(observation.acoustic_interval[1]),
                    "permissions": summary["permissions"],
                    "strength": summary["strength"],
                    "disposition": (
                        "active"
                        if source_key in active
                        else "quarantine"
                        if source_key in quarantine
                        else "inactive"
                    ),
                }
            )
        return {
            "profile_state": track.profile.state.value,
            "active_core_keys": list(track.profile.active_keys),
            "quarantine_keys": list(track.profile.quarantine_keys),
            "within_profile_cohesion_floor": (
                track.within_profile_cohesion_floor()
            ),
            "profile_transition": track.profile.last_transition,
            "evidence": evidence,
        }

    def allocate_output_label(self, track_id: str) -> str:
        """Allocate one immutable output label at an explicit resolution event.

        Finalization and elapsed time never constitute identity evidence.
        """

        track = self.tracks.get(str(track_id))
        if track is None:
            raise KeyError(f"unknown speaker track: {track_id}")
        return self._reserve_output_label(track)

    def _reserve_output_label(self, track: _Track) -> str:
        """Reserve an output handle without mutating Bayesian cluster state."""

        if track.output_label is None:
            track.output_label = "%s%02d" % (self.output_prefix, self._next_output)
            self._next_output += 1
        for alias in track.aliases:
            self.output_mapping[str(alias)] = track.output_label
        return str(track.output_label)

    def discard_unpublished_track(self, track_id: str) -> tuple[str, ...]:
        """Drop one unresolved identity without transferring its evidence."""

        target = str(track_id)
        track = self.tracks.get(target)
        if track is None:
            return ()
        if track.output_label is not None:
            raise RuntimeError("cannot discard a published speaker identity")
        source_keys = tuple(sorted(track.evidence))
        self.tracks.pop(target, None)
        return source_keys

    @staticmethod
    def _ordered_evidence(track: _Track) -> tuple[tuple[str, Observation], ...]:
        return tuple(
            sorted(
                track.evidence.items(),
                key=lambda item: (
                    float(item[1].tick),
                    float(item[1].start),
                    str(item[1].source_key or ""),
                ),
            )
        )

    def _empirical_transfer_source_keys(
        self,
        pending: _Track,
        target: _Track,
    ) -> tuple[str, ...]:
        core_keys = tuple(pending.profile.profile_candidate_keys)
        if not core_keys:
            return tuple(key for key, _observation in self._ordered_evidence(pending))
        compatible = tuple(
            key
            for key in core_keys
            if (
                best := self._unique_best(
                    self._accepted_output_profile_scores(
                        pending,
                        _Track._normalized_embedding(
                            pending.evidence[key].embedding
                        ),
                    )
                )
            )
            is not None
            and best[1] == target.track_id
        )
        if len(compatible) != len(core_keys):
            return ()
        return tuple(key for key, _observation in self._ordered_evidence(pending))

    def _accepted_output_profile_scores(
        self,
        pending: _Track,
        vector: np.ndarray,
    ) -> tuple[tuple[float, str], ...]:
        candidates: list[tuple[float, str]] = []
        for profile in sorted(self.tracks.values(), key=lambda item: item.track_id):
            if profile.output_label is None or profile is pending:
                continue
            profile_vectors = profile.core_vectors
            if not profile_vectors:
                continue
            score = max(float(np.dot(vector, item)) for item in profile_vectors)
            if score + 1e-12 >= self._empirical_output_acceptance_floor(profile):
                candidates.append((score, profile.track_id))
        return tuple(candidates)

    def _empirical_output_acceptance_floor(self, track: _Track) -> float:
        """Calibrate an external match against the retained profile itself."""

        within_floor = track.within_profile_cohesion_floor()
        if within_floor is None:
            return self.min_cosine
        return max(
            self.min_cosine,
            _PROFILE_RELATIVE_ACCEPTANCE_RATIO * within_floor,
        )

    def transfer_unpublished_evidence_to_output(
        self,
        track_id: str,
        output_label: str,
    ) -> tuple[str, ...]:
        """Transfer one complete unpublished identity to a public profile.

        Every retained clean core must select the requested target as its
        unique accepted labelled profile.  Mixed containers remain unresolved;
        accepted transfers move every source key and leave no residual owner.
        """

        pending = self.tracks.get(str(track_id))
        target_id = self.track_id_for_output_label(str(output_label))
        target = self.tracks.get(str(target_id)) if target_id is not None else None
        if pending is None or target is None:
            return ()
        if pending is target:
            return tuple(key for key, _observation in self._ordered_evidence(pending))
        if pending.output_label is not None:
            if pending.output_label != target.output_label:
                return ()
            return tuple(key for key, _observation in self._ordered_evidence(pending))
        transfer_keys = (
            self._empirical_transfer_source_keys(pending, target)
            if self.match_policy == CLEAN_CORE_POLICY
            else tuple(key for key, _observation in self._ordered_evidence(pending))
        )
        if not transfer_keys:
            return ()

        evidence = self._ordered_evidence(pending)
        for source_key, observation in evidence:
            if observation.source_key is None:
                raise RuntimeError(
                    "cannot bridge evidence without a source key"
                )
            owners = self._source_owners(observation)
            if len(owners) != 1 or owners[0][0] is not pending:
                raise RuntimeError(
                    "cannot bridge evidence with ambiguous source ownership"
                )
        for source_key, observation in evidence:
            pending.remove(str(source_key))
            target.add(observation)
        self._drop_track_if_empty(pending)
        return transfer_keys

    def _unprofiled_claim_source_keys(
        self,
        track_id: str,
        output_label: str,
    ) -> tuple[tuple[str, ...], Optional[str]]:
        pending = self.tracks.get(str(track_id))
        target_id = self.track_id_for_output_label(str(output_label))
        target = self.tracks.get(str(target_id)) if target_id is not None else None
        if pending is None:
            return (), "missing_source_track"
        if pending.output_label is not None:
            return (), "source_already_has_output_label"
        if target is None or target.output_label != str(output_label):
            return (), "missing_output_target"
        core_keys = tuple(pending.profile.active_keys)
        if not core_keys:
            return (), "source_has_no_clean_core"
        if any(
            key not in pending.evidence
            or not self._is_core_observation(pending.evidence[key])
            for key in core_keys
        ):
            return (), "source_core_not_eligible"
        profile_candidate_keys = tuple(pending.profile.profile_candidate_keys)
        core_vectors = tuple(
            _Track._normalized_embedding(pending.evidence[key].embedding)
            for key in profile_candidate_keys
        )
        if any(
            left is not None
            and right is not None
            and float(np.dot(left, right)) + 1e-12 < self.min_cosine
            for left_index, left in enumerate(core_vectors)
            for right in core_vectors[left_index + 1 :]
        ):
            return (), "conflicting_source_cores"
        if any(
            self.observations_cannot_link(source, target_observation)
            for source in pending.evidence.values()
            for target_observation in target.evidence.values()
        ):
            return (), "current_batch_cannot_link"

        ordered = self._ordered_evidence(pending)
        for _source_key, observation in ordered:
            if observation.source_key is None:
                return (), "source_key_missing"
            owners = self._source_owners(observation)
            if len(owners) != 1 or owners[0][0] is not pending:
                return (), "ambiguous_source_ownership"
        return tuple(key for key, _observation in ordered), None

    def claim_unprofiled_output_profile(
        self,
        track_id: str,
        output_label: str,
    ) -> tuple[str, ...]:
        """Atomically move one coherent clean-core identity into an empty slot."""

        source_keys, rejection_reason = self._unprofiled_claim_source_keys(
            track_id,
            output_label,
        )
        if rejection_reason is not None:
            return ()
        pending = self.tracks[str(track_id)]
        target_id = self.track_id_for_output_label(str(output_label))
        if target_id is None:
            return ()
        target = self.tracks[target_id]
        evidence = self._ordered_evidence(pending)
        added: list[str] = []
        for source_key, observation in evidence:
            if not target.add(observation):
                for added_key in added:
                    target.remove(added_key)
                return ()
            added.append(str(source_key))
        for source_key, _observation in evidence:
            if pending.remove(str(source_key)) is None:
                raise RuntimeError("prevalidated profile claim source disappeared")
        self._drop_track_if_empty(pending)
        return source_keys

    @staticmethod
    def _unique_best(
        values: Sequence[tuple[float, str]],
    ) -> Optional[tuple[float, str]]:
        if not values:
            return None
        ordered = sorted(values, key=lambda item: (-item[0], item[1]))
        if len(ordered) > 1 and math.isclose(
            ordered[0][0],
            ordered[1][0],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            return None
        return ordered[0]

    def _existing_match_ready(self, track: _Track) -> bool:
        """Whether a track has direct evidence for an existing identity."""

        return bool(track.core_vectors)

    def _novel_registration_ready(self, track: _Track) -> bool:
        """Whether a track may earn an output label from its own evidence."""

        if self.match_policy == CLEAN_CORE_POLICY:
            if track.profile.state is not ProfileState.ESTABLISHED:
                return False
            if len(track.ticks) < 2:
                return False
            if not self._confirmed(track):
                return False
            output_scores = [
                (score, target)
                for target in self.tracks.values()
                if target.output_label is not None
                and (score := track.max_core_similarity(target)) is not None
            ]
            return not any(
                score + 1e-12
                >= self._empirical_output_acceptance_floor(target)
                for score, target in output_scores
            )
        if not track.core_vectors or not self._confirmed(track):
            return False
        return True

    def _resolution_candidate_ready(self, track: _Track) -> bool:
        if self.match_policy == CLEAN_CORE_POLICY:
            return self._existing_match_ready(track) or self._novel_registration_ready(
                track
            )
        return self._novel_registration_ready(track)

    def _evaluate_unprofiled_claim(
        self,
        track: _Track,
        claim: UnprofiledOutputLabelClaim,
        *,
        accepted_profile_candidates: Sequence[tuple[float, str]],
    ) -> tuple[Optional[_Track], tuple[str, ...], Optional[str], bool]:
        target_id = (
            self.track_id_for_output_label(claim.target_output_label)
            if claim.target_output_label is not None
            else None
        )
        target = self.tracks.get(str(target_id)) if target_id is not None else None
        blocks_new = bool(
            target is not None
            and target.output_label is not None
            and claim.context_source == "current_snapshot_overlap"
            and claim.row_unique
            and claim.row_margin > 1e-12
        )
        if claim.conflict_reason:
            return target, (), str(claim.conflict_reason), blocks_new
        if claim.context_source != "current_snapshot_overlap":
            return target, (), "invalid_context_source", blocks_new
        if claim.target_output_label is None or target is None:
            return target, (), "missing_unique_context_target", blocks_new
        if not claim.row_unique or claim.row_margin <= 1e-12:
            return target, (), "context_row_not_unique", blocks_new
        if not claim.column_unique or claim.column_margin <= 1e-12:
            return target, (), "context_column_not_unique", blocks_new
        if not claim.selected_by_context_assignment:
            return target, (), "context_assignment_conflict", blocks_new
        if target.output_label is None:
            return target, (), "target_has_no_output_label", blocks_new
        if accepted_profile_candidates:
            return target, (), "profiled_candidate_preferred", blocks_new
        source_keys, rejection_reason = self._unprofiled_claim_source_keys(
            track.track_id,
            str(target.output_label),
        )
        return target, source_keys, rejection_reason, blocks_new

    @staticmethod
    def _bayesian_context_prior(
        claims: Sequence[UnprofiledOutputLabelClaim],
    ) -> dict[str, float]:
        """Collapse correlated context claims without threshold gates."""

        context: dict[str, float] = {}
        for claim in claims:
            candidates = list(claim.context_candidates)
            if claim.target_output_label is not None:
                candidates.append(
                    (str(claim.target_output_label), float(claim.context_score))
                )
            for output_label, score in candidates:
                key = str(output_label)
                value = float(score)
                previous = context.get(key)
                if previous is None or value > previous:
                    context[key] = value
        return context

    def _lego_track_assessment(
        self,
        track: _Track,
        *,
        context: Mapping[str, float],
    ) -> EvidenceAssessment:
        """Select one canonical bounded view set for one event observation."""

        assessments: list[EvidenceAssessment] = []
        for source_key, observation in self._ordered_evidence(track):
            assessments.append(
                assess_embedding_views(
                    observation.embedding_views,
                    independence_key=str(source_key),
                    causal_context=context,
                    text=observation.text,
                    purity=float(observation.quality),
                    routing_admissible=True,
                    profile_admissible=(
                        float(observation.quality) + 1e-9
                        >= self.core_min_quality
                    ),
                    novelty_admissible=True,
                    mixed=float(observation.quality) < 1.0 - 1e-9,
                )
            )
        if not assessments:
            raise RuntimeError("robust event track has no evidence")
        return min(
            assessments,
            key=lambda item: (
                -float(item.routing_precision),
                -float(item.profile_precision),
                -float(item.novelty_precision),
                str(item.independence_key),
            ),
        )

    @staticmethod
    def _lego_profile_snapshot_dict(
        snapshot: ProfileSnapshot | None,
    ) -> dict[str, object] | None:
        if snapshot is None:
            return None
        return {
            "profile_id": snapshot.profile_id,
            "exemplar_count": int(snapshot.exemplar_count),
            "dispersion": float(snapshot.dispersion),
            "effective_support": float(snapshot.effective_support),
            "independence_keys": [
                item.independence_key for item in snapshot.exemplars
            ],
        }

    @staticmethod
    def _risk_claim_rejection_reason(claim: UnprofiledOutputLabelClaim) -> Optional[str]:
        if claim.conflict_reason:
            return str(claim.conflict_reason)
        if claim.context_source != "current_snapshot_overlap":
            return "invalid_context_source"
        if claim.target_output_label is None:
            return "missing_context_target"
        if not claim.row_unique or claim.row_margin <= 1e-12:
            return "context_row_not_unique"
        if not claim.column_unique or claim.column_margin <= 1e-12:
            return "context_column_not_unique"
        if not claim.selected_by_context_assignment:
            return "context_assignment_conflict"
        if not str(claim.ancestor_lineage or "").strip():
            return "missing_ancestor_lineage"
        return None

    @staticmethod
    def _risk_claim_fraction(value: float, scale: float) -> float:
        if not math.isfinite(float(value)) or not math.isfinite(float(scale)):
            return 0.0
        return float(np.clip(max(0.0, float(value)) / max(abs(float(scale)), 1e-12), 0.0, 1.0))

    def _risk_bridge_evidence(
        self,
        track_id: str,
        assessment: EvidenceAssessment,
        claims: Sequence[UnprofiledOutputLabelClaim],
    ) -> tuple[BridgeEvidence, ...]:
        values: list[BridgeEvidence] = []
        for claim in sorted(
            claims,
            key=lambda item: (
                str(item.target_output_label or ""),
                str(item.context_source),
                float(item.context_score),
            ),
        ):
            target = str(claim.target_output_label or "").strip()
            if not target:
                continue
            candidate_scale = max(
                [abs(float(claim.context_score))]
                + [abs(float(score)) for _output_label, score in claim.context_candidates]
                + [1e-12]
            )
            rejection = self._risk_claim_rejection_reason(claim)
            routing_authority = float(np.clip(assessment.routing_precision, 0.0, 1.0))
            bridge_authority = routing_authority if rejection is None else 0.0
            values.append(
                BridgeEvidence(
                    observation_id=str(track_id),
                    independence_key=str(assessment.independence_key),
                    ancestor_lineage=str(
                        claim.ancestor_lineage or assessment.independence_key
                    ),
                    component_id=None,
                    output_label=target,
                    provenance=str(claim.context_source),
                    structural_score=self._risk_claim_fraction(
                        claim.context_score,
                        candidate_scale,
                    ),
                    row_margin=(
                        self._risk_claim_fraction(claim.row_margin, candidate_scale)
                        if claim.row_unique
                        else 0.0
                    ),
                    column_margin=(
                        self._risk_claim_fraction(claim.column_margin, candidate_scale)
                        if claim.column_unique
                        else 0.0
                    ),
                    conflict=rejection is not None,
                    acoustic_responsibility=1.0,
                    routing_authority=routing_authority,
                    bridge_authority=bridge_authority,
                )
            )
        return tuple(values)

    def _identity_shadow_registry_snapshot(self) -> dict[str, object]:
        tracks: list[dict[str, object]] = []
        for track_id in sorted(self.tracks):
            track = self.tracks[track_id]
            tracks.append(
                {
                    "track_id": track.track_id,
                    "output_label": track.output_label,
                    "aliases": sorted(track.aliases),
                    "intervals": [list(item) for item in sorted(track.intervals)],
                    "ticks": sorted(float(item) for item in track.ticks),
                    "support_sec": float(track.support_sec),
                    "core_count": len(track.core_vectors),
                    "profile_state": track.profile.state.value,
                    "active_profile_keys": list(track.profile.active_keys),
                    "quarantine_profile_keys": list(track.profile.quarantine_keys),
                    "evidence": [
                        {
                            "source_key": source_key,
                            "revision": int(observation.revision),
                            "local_symbol": observation.local_symbol,
                            "start": float(observation.start),
                            "end": float(observation.end),
                            "tick": float(observation.tick),
                            "acoustic_start": observation.acoustic_start,
                            "acoustic_end": observation.acoustic_end,
                            "quality": float(observation.quality),
                            "text_length": len(str(observation.text or "")),
                            "has_embedding": observation.embedding is not None,
                            "view_count": (
                                len(observation.embedding_views.views)
                                if observation.embedding_views is not None
                                else 0
                            ),
                        }
                        for source_key, observation in sorted(track.evidence.items())
                    ],
                }
            )
        ledger = [
            {
                "sequence": int(record.sequence),
                "event_type": record.event_type,
                "source_key": record.source_key,
                "revision": int(record.revision),
                "track_id": record.track_id,
                "association_reason": record.association_reason,
                "local_symbol": record.local_symbol,
                "start": float(record.start),
                "end": float(record.end),
                "tick": float(record.tick),
                "acoustic_start": record.acoustic_start,
                "acoustic_end": record.acoustic_end,
                "text_length": len(str(record.text or "")),
                "quality": float(record.quality),
                "embedding": (
                    None
                    if record.embedding is None
                    else np.asarray(record.embedding, dtype=np.float32).copy()
                ),
            }
            for record in self.evidence_ledger.records
        ]
        return {
            "tracks": tracks,
            "output_mapping": [
                [key, value] for key, value in sorted(self.output_mapping.items())
            ],
            "next_track": int(self._next_track),
            "next_output": int(self._next_output),
            "evidence_ledger": ledger,
        }

    def identity_shadow_authoritative_digest(
        self,
        *,
        publication_snapshot: Optional[Mapping[str, object]] = None,
    ) -> str:
        model = self.lego_identity_model
        return authoritative_digest(
            {
                "model": model.shadow_snapshot() if model is not None else {},
                "registry": self._identity_shadow_registry_snapshot(),
                "publication": defensive_snapshot(publication_snapshot or {}),
            }
        )

    def _resolve_lego_identity_event(
        self,
        *,
        unprofiled_claims: Sequence[UnprofiledOutputLabelClaim],
        partition_excluded_output_labels_by_track: Optional[
            Mapping[str, Sequence[str]]
        ] = None,
    ) -> Dict[str, tuple[str, str]]:
        model = self.lego_identity_model
        if model is None:
            raise RuntimeError("lego policy has no identity model")
        pending = sorted(
            (track for track in self.tracks.values() if track.output_label is None),
            key=lambda item: item.track_id,
        )
        if not pending:
            return {}

        for track in sorted(self.tracks.values(), key=lambda item: item.track_id):
            if track.output_label is None:
                continue
            output_label = str(track.output_label)
            if not model.has_output(output_label):
                weak_anchor = (
                    self._lego_track_assessment(track, context={})
                    if track.evidence
                    else None
                )
                model.ensure_output(output_label, weak_anchor=weak_anchor)
            else:
                model.ensure_output(output_label)

        claims_by_track: dict[str, list[UnprofiledOutputLabelClaim]] = {}
        for claim in unprofiled_claims:
            claims_by_track.setdefault(str(claim.source_track_id), []).append(claim)

        assessments: dict[str, EvidenceAssessment] = {}
        bridges_by_track: dict[str, tuple[BridgeEvidence, ...]] = {}
        source_keys_by_track: dict[str, tuple[str, ...]] = {}
        event_observations: list[LegoIdentityObservation] = []
        for track in pending:
            claims = claims_by_track.get(track.track_id, ())
            assessment = self._lego_track_assessment(
                track,
                context=self._bayesian_context_prior(claims),
            )
            bridges = self._risk_bridge_evidence(
                track.track_id,
                assessment,
                claims,
            )
            assessments[track.track_id] = assessment
            bridges_by_track[track.track_id] = bridges
            source_keys_by_track[track.track_id] = tuple(
                source_key for source_key, _item in self._ordered_evidence(track)
            )
            latest_observation = max(
                track.evidence.values(),
                key=lambda item: (
                    float(item.tick),
                    int(item.revision),
                    str(item.source_key or ""),
                ),
            )
            partition_excluded_output_labels = tuple(
                sorted(
                    {
                        str(output_label)
                        for output_label in (partition_excluded_output_labels_by_track or {}).get(
                            track.track_id,
                            (),
                        )
                        if str(output_label)
                    }
                )
            )
            event_observations.append(
                LegoIdentityObservation(
                    observation_id=track.track_id,
                    assessment=assessment,
                    bridge_evidence=bridges,
                    local_partition_key=f"snapshot:{float(latest_observation.tick):.6f}",
                    local_symbol=str(latest_observation.local_symbol),
                    partition_excluded_output_labels=partition_excluded_output_labels,
                )
            )

        cannot_link: list[tuple[str, str]] = []
        for index, left in enumerate(pending):
            for right in pending[index + 1 :]:
                if any(
                    self.observations_share_local_partition(left_item, right_item)
                    for left_item in left.evidence.values()
                    for right_item in right.evidence.values()
                ):
                    cannot_link.append((left.track_id, right.track_id))

        before_profiles = {
            component_id: model.profile_snapshot(component_id)
            for component_id in model.component_ids
        }
        before_active_provisional = set(model.active_provisional_display_items)
        before_global_components = {
            global_id: model.global_identity_component_ids(global_id)
            for global_id in model.global_identity_ids
        }
        model_state_size_before = model.state_size_bytes()
        event_model_started = time.perf_counter_ns()
        inference_started = time.perf_counter_ns()
        event_id = "lego:" + ",".join(
            track.track_id for track in pending
        )
        inference = model.infer(
            event_observations,
            cannot_link=cannot_link,
        )
        inference_latency_ms = (
            time.perf_counter_ns() - inference_started
        ) / 1_000_000.0
        decisions = {item.observation_id: item for item in inference.decisions}
        actions: Dict[str, tuple[str, str]] = {}
        output_links: dict[str, str] = {}

        for track in pending:
            decision = decisions[track.track_id]
            routed_output = decision.publication_action.output_label
            if routed_output is not None:
                output_label = str(routed_output)
                if self.track_id_for_output_label(output_label) is None:
                    raise RuntimeError("lego output route has no registry track")
                moved_source_keys = self.transfer_unpublished_evidence_to_output(
                    track.track_id,
                    output_label,
                )
                if moved_source_keys != source_keys_by_track[track.track_id]:
                    raise RuntimeError("lego evidence transfer was not atomic")
                action = "existing"
            elif decision.new_output_action.allocate:
                output_label = self._reserve_output_label(track)
                output_links[track.track_id] = output_label
                moved_source_keys = source_keys_by_track[track.track_id]
                action = "new"
            else:
                raise RuntimeError("lego publication did not choose a top-1 action")
            actions[track.track_id] = (action, output_label)
            self._last_resolution_source_keys[track.track_id] = tuple(moved_source_keys)

        update_events = {
            str(item["observation_id"]): item
            for item in model.apply_update(inference, output_links=output_links)
        }
        after_provisional = set(model.provisional_display_items)
        after_active_provisional = set(model.active_provisional_display_items)
        provisional_create_count = sum(
            decision.new_output_action.kind
            in {"provisional_evidence_poor", "provisional_local_partition"}
            for decision in decisions.values()
        )
        provisional_reuse_count = len(
            before_active_provisional.intersection(after_active_provisional)
        )
        expired_provisional = before_active_provisional - after_active_provisional
        provisional_expire_count = len(expired_provisional)
        provisional_claim_count = sum(
            not before_global_components.get(global_id, ())
            and bool(model.global_identity_component_ids(global_id))
            for global_id, _output_label in after_provisional
        )
        provisional_retire_count = sum(
            bool(model.global_identity_component_ids(global_id))
            or model.global_identity_for_output(output_label) not in {None, global_id}
            for global_id, output_label in expired_provisional
        )
        event_model_latency_ms = (
            time.perf_counter_ns() - event_model_started
        ) / 1_000_000.0
        model_state_size_after = model.state_size_bytes()
        events: list[dict[str, object]] = []
        for track in pending:
            decision = decisions[track.track_id]
            assessment = assessments[track.track_id]
            action, output_label = actions[track.track_id]
            component_id = decision.acoustic_explanation.component_id
            before = before_profiles.get(str(component_id))
            after = (
                model.profile_snapshot(str(component_id))
                if component_id is not None and str(component_id) in model.component_ids
                else None
            )
            resolved_bridge = decision.resolved_bridge
            bridge_snapshot = (
                model.bridge_snapshot(
                    str(component_id),
                    resolved_bridge.output_label,
                )
                if component_id is not None and resolved_bridge is not None
                else None
            )
            claim_diagnostics = []
            for claim in claims_by_track.get(track.track_id, ()):
                claim_diagnostics.append(
                    {
                        "target_output_label": claim.target_output_label,
                        "context_source": claim.context_source,
                        "context_candidates": [list(item) for item in claim.context_candidates],
                        "context_score": float(claim.context_score),
                        "row_margin": float(claim.row_margin),
                        "column_margin": float(claim.column_margin),
                        "ancestor_lineage": claim.ancestor_lineage,
                        "rejection_reason": self._risk_claim_rejection_reason(claim),
                    }
                )
            events.append(
                {
                    "track_id": track.track_id,
                    "status": action,
                    "output_target": output_label,
                    "publication_output_label": output_label,
                    "policy": self.match_policy,
                    "decision_model": "eg_identity_gp_hungarian",
                    "event_id": event_id,
                    "publication_reason": decision.publication_action.reason,
                    "publication_best_effort": bool(decision.publication_action.best_effort),
                    "publication_score": float(decision.publication_action.score),
                    "embedding_component_id": component_id,
                    "global_identity_id": decision.global_identity_id,
                    "published_speaker_id": output_label,
                    "acoustic_component_id": component_id,
                    "acoustic_kind": decision.acoustic_explanation.kind,
                    "acoustic_score": float(decision.acoustic_explanation.score),
                    "acoustic_responsibility": float(
                        decision.acoustic_explanation.inlier_responsibility
                    ),
                    "binding_action": decision.binding_action.kind,
                    "binding_output_label": decision.binding_action.output_label,
                    "binding_utility": float(decision.binding_action.utility),
                    "binding_margin": float(decision.binding_action.margin),
                    "profile_update_component_id": decision.profile_update_action.component_id,
                    "profile_update_output_label": decision.profile_update_action.output_label,
                    "profile_update_weight": float(decision.profile_update_action.weight),
                    "profile_update_reason": decision.profile_update_action.reason,
                    "allocate_output_handle": bool(decision.new_output_action.allocate),
                    "new_output_utility": float(decision.new_output_action.utility),
                    "new_output_reason": decision.new_output_action.reason,
                    "output_allocation_kind": decision.new_output_action.kind,
                    "local_partition_key": next(
                        item.local_partition_key
                        for item in event_observations
                        if item.observation_id == track.track_id
                    ),
                    "local_symbol_partition": next(
                        item.local_symbol
                        for item in event_observations
                        if item.observation_id == track.track_id
                    ),
                    "partition_excluded_output_labels": list(
                        next(
                            item.partition_excluded_output_labels
                            for item in event_observations
                            if item.observation_id == track.track_id
                        )
                    ),
                    "provisional_create_count": int(provisional_create_count),
                    "provisional_reuse_count": int(provisional_reuse_count),
                    "provisional_expire_count": int(provisional_expire_count),
                    "provisional_claim_count": int(provisional_claim_count),
                    "provisional_retire_count": int(provisional_retire_count),
                    "active_provisional_count": len(after_active_provisional),
                    "forced_constraint": bool(decision.forced_constraint),
                    "cross_output_update": bool(
                        decision.profile_update_action.output_label is not None
                        and decision.profile_update_action.output_label
                        != decision.binding_action.output_label
                    ),
                    "resolved_bridge": (
                        {
                            "output_label": resolved_bridge.output_label,
                            "ancestor_lineage": resolved_bridge.ancestor_lineage,
                            "provenance": resolved_bridge.provenance,
                            "contribution": float(decision.bridge_contribution),
                            "total_contribution": float(
                                bridge_snapshot.total_contribution
                            ),
                            "independent_lineages": list(
                                bridge_snapshot.independent_lineages
                            ),
                        }
                        if resolved_bridge is not None and bridge_snapshot is not None
                        else None
                    ),
                    "claim_diagnostics": claim_diagnostics,
                    "cannot_link": [list(edge) for edge in inference.cannot_link],
                    "inference_latency_ms": float(inference_latency_ms),
                    "event_model_latency_ms": float(event_model_latency_ms),
                    "model_state_size_bytes_before": int(model_state_size_before),
                    "model_state_size_bytes_after": int(model_state_size_after),
                    "component_count": len(model.component_ids),
                    "binding_count": len(model.binding_items),
                    "update_status": update_events.get(track.track_id, {}).get(
                        "status", "no_update"
                    ),
                    "assessment": {
                        "routing_allowed": bool(assessment.routing_allowed),
                        "profile_allowed": bool(assessment.profile_allowed),
                        "novelty_allowed": bool(assessment.novelty_allowed),
                        "independence_key": assessment.independence_key,
                        "view_count": len(assessment.views),
                        "instability": float(assessment.instability),
                        "unique_coverage_sec": float(assessment.unique_coverage_sec),
                        "precision_route": float(assessment.precision_route),
                        "precision_profile": float(assessment.precision_profile),
                        "precision_new": float(assessment.precision_new),
                        "circle_padded": bool(assessment.circle_padded),
                        "causal_context": [
                            [candidate, float(score)]
                            for candidate, score in assessment.causal_context
                        ],
                    },
                    "acoustic_state_before": self._lego_profile_snapshot_dict(before),
                    "acoustic_state_after": self._lego_profile_snapshot_dict(after),
                    "source_keys": list(source_keys_by_track[track.track_id]),
                }
            )
        self._last_resolution_events = tuple(events)
        return actions


    def resolve_event_tracks(
        self,
        *,
        unprofiled_claims: Sequence[UnprofiledOutputLabelClaim] = (),
        partition_excluded_output_labels_by_track: Optional[
            Mapping[str, Sequence[str]]
        ] = None,
    ) -> Dict[str, tuple[str, str]]:
        """Resolve ready unpublished tracks under exactly one match policy.

        Returns ``track_id -> (existing|new, output_label)`` for tracks whose
        identity closed during this event.  Ambiguous tracks remain pending.
        """

        self._last_resolution_events = ()
        self._last_resolution_source_keys = {}
        if self.match_policy == FIXED_COSINE_POLICY:
            return {}
        if self.match_policy == LEGO_POLICY:
            return self._resolve_lego_identity_event(
                unprofiled_claims=unprofiled_claims,
                partition_excluded_output_labels_by_track=(
                    partition_excluded_output_labels_by_track
                ),
            )
        unpublished = [
            track for track in self.tracks.values() if track.output_label is None
        ]
        structurally_separated: set[str] = set()
        for index, left in enumerate(unpublished):
            for right in unpublished[index + 1 :]:
                if any(
                    self.observations_cannot_link(left_item, right_item)
                    for left_item in left.evidence.values()
                    for right_item in right.evidence.values()
                ):
                    structurally_separated.update((left.track_id, right.track_id))
        pending = [
            track
            for track in unpublished
            if self._resolution_candidate_ready(track)
            or track.track_id in structurally_separated
        ]
        if not pending:
            return {}
        labelled = [
            track for track in self.tracks.values() if track.output_label is not None
        ]
        actions: Dict[str, tuple[str, str]] = {}

        if self.match_policy == CLEAN_CORE_POLICY:
            claims_by_track: dict[str, list[UnprofiledOutputLabelClaim]] = {}
            for claim in unprofiled_claims:
                claims_by_track.setdefault(str(claim.source_track_id), []).append(
                    claim
                )
            proposals: list[tuple[float, str, str, str]] = []
            novel: list[str] = []
            ambiguous: set[str] = set()
            blocked_new: set[str] = set()
            prevalidated_claim_source_keys: dict[str, tuple[str, ...]] = {}
            event_by_track: dict[str, dict[str, object]] = {}
            for track in sorted(pending, key=lambda item: item.track_id):
                novelty_ready = self._novel_registration_ready(track)
                candidates: list[tuple[float, str]] = []
                compatible_scores: list[float] = []
                for target in labelled:
                    score = track.max_core_similarity(target)
                    if score is None:
                        continue
                    compatible_scores.append(score)
                    if (
                        score + 1e-12
                        >= self._empirical_output_acceptance_floor(target)
                    ):
                        candidates.append((score, target.track_id))
                within_floor = track.within_profile_cohesion_floor()
                event_by_track[track.track_id] = {
                    "track_id": track.track_id,
                    "status": "wait",
                    "output_target": None,
                    "policy": self.match_policy,
                    "reason": "unresolved",
                    "support_sec": float(track.support_sec),
                    "independent_windows": len(track.ticks),
                    "core_count": len(track.core_vectors),
                    "within_floor": within_floor,
                    "best_output_similarity": (
                        max(compatible_scores) if compatible_scores else None
                    ),
                    "accepted_candidate_count": len(candidates),
                    "compatible_output_count": len(compatible_scores),
                    "existing_match_ready": self._existing_match_ready(track),
                    "novel_registration_ready": novelty_ready,
                    "structural_separation_ready": (
                        track.track_id in structurally_separated
                    ),
                    "source_profile_before": self._profile_audit(track),
                }
                best = self._unique_best(candidates)
                if best is not None:
                    target = self.tracks[best[1]]
                    if self._empirical_transfer_source_keys(track, target):
                        proposals.append(
                            (best[0], track.track_id, best[1], "profiled")
                        )
                        event_by_track[track.track_id]["reason"] = (
                            "accepted_empirical_profile"
                        )
                    else:
                        ambiguous.add(track.track_id)
                        event_by_track[track.track_id]["reason"] = (
                            "mixed_core_targets"
                        )
                elif candidates:
                    ambiguous.add(track.track_id)
                    event_by_track[track.track_id]["reason"] = (
                        "ambiguous_accepted_profiles"
                    )
                elif compatible_scores:
                    if (
                        novelty_ready
                        and (
                            len(track.core_vectors) == 1
                            or (
                                within_floor is not None
                                and within_floor > max(compatible_scores) + 1e-12
                            )
                        )
                    ):
                        novel.append(track.track_id)
                        event_by_track[track.track_id]["reason"] = (
                            "self_coherent_novelty"
                        )
                    else:
                        ambiguous.add(track.track_id)
                        event_by_track[track.track_id]["reason"] = (
                            "insufficient_novelty"
                        )
                elif not labelled:
                    if novelty_ready and (
                        len(track.core_vectors) == 1 or within_floor is not None
                    ):
                        novel.append(track.track_id)
                        event_by_track[track.track_id]["reason"] = (
                            "no_output_profiles"
                        )
                    else:
                        ambiguous.add(track.track_id)
                        event_by_track[track.track_id]["reason"] = (
                            "insufficient_novelty"
                        )
                else:
                    ambiguous.add(track.track_id)
                    event_by_track[track.track_id]["reason"] = (
                        "no_scorable_output_profiles"
                    )

                claim_values = claims_by_track.get(track.track_id, ())
                if claim_values:
                    event = event_by_track[track.track_id]
                    claim = claim_values[0]
                    event.update(
                        profile_claim_status="rejected",
                        profile_claim_target=claim.target_output_label,
                        profile_claim_context_source=claim.context_source,
                        profile_claim_context_score=float(claim.context_score),
                        profile_claim_row_margin=float(claim.row_margin),
                        profile_claim_column_margin=float(claim.column_margin),
                        profile_claim_candidates=[
                            {
                                "output_label": str(output_label),
                                "score": float(score),
                            }
                            for output_label, score in claim.context_candidates
                        ],
                        profile_claim_source_core_keys=list(track.profile.active_keys),
                        profile_claim_source_core_intervals=[
                            {
                                "source_key": source_key,
                                "start": float(
                                    track.evidence[source_key].acoustic_interval[0]
                                ),
                                "end": float(
                                    track.evidence[source_key].acoustic_interval[1]
                                ),
                            }
                            for source_key in track.profile.active_keys
                        ],
                    )
                    if len(claim_values) != 1:
                        target = None
                        source_keys = ()
                        rejection_reason = "conflicting_claim_evidence"
                        blocks_registration = True
                    else:
                        (
                            target,
                            source_keys,
                            rejection_reason,
                            blocks_registration,
                        ) = self._evaluate_unprofiled_claim(
                            track,
                            claim,
                            accepted_profile_candidates=candidates,
                        )
                    event["profile_core_count_before"] = (
                        len(target.core_vectors) if target is not None else None
                    )
                    if blocks_registration:
                        blocked_new.add(track.track_id)
                    if rejection_reason is None and target is not None:
                        proposals.append(
                            (
                                float(claim.context_score),
                                track.track_id,
                                target.track_id,
                                "unprofiled",
                            )
                        )
                        prevalidated_claim_source_keys[track.track_id] = source_keys
                        event["reason"] = "accepted_unprofiled_profile_claim"
                        event["profile_claim_rejection_reason"] = None
                        blocked_new.add(track.track_id)
                    else:
                        event["profile_claim_rejection_reason"] = str(
                            rejection_reason or "unresolved_claim"
                        )

            used_pending: Set[str] = set()
            used_output: Set[str] = set()
            selected_proposals: list[tuple[str, str, str]] = []
            for _score, pending_id, output_track_id, proposal_kind in sorted(
                proposals,
                key=lambda item: (-item[0], item[1], item[2], item[3]),
            ):
                if pending_id in used_pending or output_track_id in used_output:
                    ambiguous.add(pending_id)
                    blocked_new.add(pending_id)
                    event_by_track[pending_id]["reason"] = "output_target_conflict"
                    if proposal_kind == "unprofiled":
                        event_by_track[pending_id].update(
                            profile_claim_status="rejected",
                            profile_claim_rejection_reason="output_target_conflict",
                        )
                    continue
                used_pending.add(pending_id)
                used_output.add(output_track_id)
                selected_proposals.append(
                    (pending_id, output_track_id, proposal_kind)
                )

            for pending_id, output_track_id, proposal_kind in selected_proposals:
                target = self.tracks.get(output_track_id)
                if target is None or target.output_label is None:
                    continue
                output_label = str(target.output_label)
                profile_core_count_before = len(target.core_vectors)
                moved_source_keys = (
                    self.claim_unprofiled_output_profile(pending_id, output_label)
                    if proposal_kind == "unprofiled"
                    else self.transfer_unpublished_evidence_to_output(
                        pending_id,
                        output_label,
                    )
                )
                if moved_source_keys:
                    actions[pending_id] = ("existing", output_label)
                    self._last_resolution_source_keys[pending_id] = moved_source_keys
                    event_by_track[pending_id].update(
                        status="existing",
                        output_target=output_label,
                        transferred_source_count=len(moved_source_keys),
                        target_profile_after=self._profile_audit(target),
                    )
                    if proposal_kind == "unprofiled":
                        if moved_source_keys != prevalidated_claim_source_keys.get(
                            pending_id
                        ):
                            raise RuntimeError(
                                "unprofiled claim changed after event assignment"
                            )
                        event_by_track[pending_id].update(
                            profile_claim_status="claimed",
                            profile_core_count_before=profile_core_count_before,
                            profile_core_count_after=len(target.core_vectors),
                        )
                else:
                    ambiguous.add(pending_id)
                    blocked_new.add(pending_id)
                    if proposal_kind == "unprofiled":
                        event_by_track[pending_id].update(
                            profile_claim_status="rejected",
                            profile_claim_rejection_reason=(
                                "claim_mutation_precondition_changed"
                            ),
                        )
            for track_id in novel:
                if (
                    track_id in ambiguous
                    or track_id in used_pending
                    or track_id in blocked_new
                ):
                    continue
                output_label = self.allocate_output_label(track_id)
                actions[track_id] = ("new", output_label)
                self._last_resolution_source_keys[track_id] = tuple(
                    key
                    for key, _observation in self._ordered_evidence(
                        self.tracks[track_id]
                    )
                )
                event_by_track[track_id].update(
                    status="new",
                    output_target=output_label,
                    target_profile_after=self._profile_audit(
                        self.tracks[track_id]
                    ),
                )
            for track_id in sorted(structurally_separated):
                if track_id in actions or track_id in used_pending:
                    continue
                track = self.tracks.get(track_id)
                if track is None or track.output_label is not None:
                    continue
                output_label = self.allocate_output_label(track_id)
                actions[track_id] = ("provisional", output_label)
                self._last_resolution_source_keys[track_id] = tuple(
                    key
                    for key, _observation in self._ordered_evidence(track)
                )
                event_by_track[track_id].update(
                    status="provisional",
                    output_target=output_label,
                    reason="structural_separation",
                    target_profile_after=self._profile_audit(track),
                )
            for track_id, event in event_by_track.items():
                if "target_profile_after" in event:
                    continue
                track = self.tracks.get(track_id)
                if track is not None:
                    event["source_profile_after"] = self._profile_audit(track)
            self._last_resolution_events = tuple(
                event_by_track[track_id]
                for track_id in sorted(event_by_track)
            )
            return actions

        pending_best: Dict[str, tuple[float, str]] = {}
        output_candidates: Dict[str, list[tuple[float, str]]] = {
            track.track_id: [] for track in labelled
        }
        tied_pending: Set[str] = set()
        for track in sorted(pending, key=lambda item: item.track_id):
            candidates: list[tuple[float, str]] = []
            for target in labelled:
                score = track.max_core_similarity(target)
                if score is None:
                    continue
                candidates.append((score, target.track_id))
                output_candidates[target.track_id].append((score, track.track_id))
            best = self._unique_best(candidates)
            if best is None:
                if candidates:
                    tied_pending.add(track.track_id)
                continue
            pending_best[track.track_id] = best

        output_best = {
            track_id: best
            for track_id, values in output_candidates.items()
            if (best := self._unique_best(values)) is not None
        }
        merged: Set[str] = set()
        for pending_id, (_score, output_track_id) in sorted(pending_best.items()):
            reverse = output_best.get(output_track_id)
            if reverse is None or reverse[1] != pending_id:
                continue
            target = self.tracks.get(output_track_id)
            if target is None or target.output_label is None:
                continue
            output_label = str(target.output_label)
            moved_source_keys = self.transfer_unpublished_evidence_to_output(
                pending_id,
                output_label,
            )
            if moved_source_keys:
                merged.add(pending_id)
                actions[pending_id] = ("existing", output_label)
                self._last_resolution_source_keys[pending_id] = moved_source_keys
        for track in pending:
            if track.track_id in merged or track.track_id in tied_pending:
                continue
            actions[track.track_id] = (
                "new",
                self.allocate_output_label(track.track_id),
            )
            self._last_resolution_source_keys[track.track_id] = tuple(
                key
                for key, _observation in self._ordered_evidence(track)
            )
        return actions

    def _confirmed(self, track: _Track) -> bool:
        if self.require_core_for_confirmation and not track.core_vectors:
            return False
        return (
            track.support_sec + 1e-9 >= self.min_support_sec
            # ``min_observations`` is an independent-window requirement, not
            # a count of utterance fragments from one full-window snapshot.
            # Repeated source keys/revisions are already collapsed in
            # ``evidence``; distinct source keys can still share one tick.
            and len(track.ticks) >= self.min_observations
        )

    @staticmethod
    def _evidence_candidate(
        observation: Observation,
    ) -> Optional[EvidenceCandidate]:
        vector = _Track._normalized_embedding(observation.embedding)
        if vector is None:
            return None
        acoustic_start, acoustic_end = observation.acoustic_interval
        return EvidenceCandidate(
            source_key=str(observation.source_key or ""),
            vector=vector,
            tick=float(observation.tick),
            clean_duration_sec=max(0.0, acoustic_end - acoustic_start),
            text=observation.text,
            purity=float(observation.quality),
        )

    def _is_core_observation(self, observation: Observation) -> bool:
        if self.match_policy != CLEAN_CORE_POLICY:
            if observation.duration + 1e-9 < self.core_min_duration_sec:
                return False
            if float(observation.quality) + 1e-9 < self.core_min_quality:
                return False
            return _Track._normalized_embedding(observation.embedding) is not None
        permissions = self.observation_permissions(observation)
        return bool(permissions is not None and permissions.profile)

    def _track_score(self, track: _Track, observation: Observation) -> float:
        overlap = 0.0
        for start, end in track.intervals:
            overlap = max(
                overlap,
                max(0.0, min(end, observation.end) - max(start, observation.start)),
            )
        same_local = observation.local_symbol in track.aliases
        core_observation = self._is_core_observation(observation)
        if not core_observation:
            if same_local and overlap > 1e-9:
                return 0.5 + overlap
            return -1.0
        vector = _Track._normalized_embedding(observation.embedding)
        core_vectors = track.core_vectors
        if vector is None or not core_vectors:
            if same_local and overlap > 1e-9:
                return 0.75 + overlap
            return -1.0
        cosine = float(max(np.dot(item, vector) for item in core_vectors))
        if self.match_policy != FIXED_COSINE_POLICY:
            if track.output_label is None:
                if cosine + 1e-12 < self.min_cosine:
                    return -1.0
                return max(0.0, cosine)
            if self.match_policy == CLEAN_CORE_POLICY:
                if (
                    cosine + 1e-12
                    < self._empirical_output_acceptance_floor(track)
                ):
                    return -1.0
                # A retained clean core is direct identity evidence.  Mutable
                # local labels and timestamp overlap may propose candidates
                # for no-core fragments, but must never outweigh an acoustic
                # rejection or a stronger pending acoustic match.
                return max(0.0, cosine)
            if same_local and overlap >= self.min_overlap_sec:
                return 2.0 + overlap + cosine
            return max(0.0, cosine)

        centroid = track.centroid()
        if centroid is not None:
            cosine = float(np.dot(centroid, vector))
        if same_local:
            if cosine < self.same_local_split_cosine:
                return -1.0
            if overlap < self.min_overlap_sec and cosine < self.min_cosine:
                return -1.0
            alias_bonus = 1.0 if overlap >= self.min_overlap_sec else 0.25
            return alias_bonus * 2.0 + overlap + max(0.0, cosine)
        if cosine < self.different_local_merge_cosine:
            return -1.0
        return overlap + max(0.0, cosine)


__all__ = [
    "CLEAN_CORE_POLICY",
    "FIXED_COSINE_POLICY",
    "MODEL_OWNED_IDENTITY_POLICIES",
    "MULTIVIEW_IDENTITY_POLICIES",
    "Observation",
    "LEGO_POLICY",
    "SPEAKER_MATCH_POLICIES",
    "SpeakerDecision",
    "SpeakerRegistry",
    "UnprofiledOutputLabelClaim",
]
