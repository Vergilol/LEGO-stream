from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


_LEXICAL_UNIT = re.compile(r"[A-Za-z0-9]+|[\u3400-\u9fff]")


class ProfileState(str, Enum):
    UNSEEDED = "unseeded"
    PROVISIONAL = "provisional"
    ESTABLISHED = "established"


@dataclass(frozen=True)
class EvidencePermissions:
    extractable: bool
    pure: bool
    routing: bool
    profile: bool
    novelty: bool


@dataclass(frozen=True)
class EvidenceStrength:
    clean_duration_sec: float
    lexical_unit_count: int
    lexical_diversity: int
    purity: float
    duration_coverage: float
    content_coverage: float
    aggregate: float


@dataclass(frozen=True)
class EvidenceAssessment:
    """Single causal owner of identity evidence precision and capabilities."""

    x: Optional[np.ndarray]
    rho: float
    routing_allowed: bool
    profile_allowed: bool
    novelty_allowed: bool
    independence_key: str
    causal_context: tuple[tuple[str, float], ...]
    purity_reliability: float
    duration_reliability: float
    content_reliability: float
    duplicate: bool = False
    views: tuple[np.ndarray, ...] = ()
    center: Optional[np.ndarray] = None
    instability: float = 0.0
    unique_coverage_sec: float = 0.0
    precision_route: float = 0.0
    precision_profile: float = 0.0
    precision_new: float = 0.0
    circle_padded: bool = False

    @property
    def routing_precision(self) -> float:
        if self.views:
            return float(self.precision_route) if self.routing_allowed else 0.0
        return float(self.rho) if self.routing_allowed else 0.0

    @property
    def profile_precision(self) -> float:
        if self.views:
            return float(self.precision_profile) if self.profile_allowed else 0.0
        return float(self.rho) if self.profile_allowed else 0.0

    @property
    def novelty_precision(self) -> float:
        if self.views:
            return float(self.precision_new) if self.novelty_allowed else 0.0
        return float(self.rho) if self.novelty_allowed else 0.0


@dataclass(frozen=True)
class BridgeEvidence:
    """Typed causal proposal/fact linking one observation, component and output label.

    ``component_id`` may be ``None`` only while the observation is an input
    proposal.  The identity model resolves it to the single acoustic component
    selected for that observation before the fact enters runtime state.
    """

    observation_id: str
    independence_key: str
    ancestor_lineage: str
    component_id: Optional[str]
    output_label: str
    provenance: str
    structural_score: float
    row_margin: float
    column_margin: float
    conflict: bool
    acoustic_responsibility: float
    routing_authority: float
    bridge_authority: float

    def __post_init__(self) -> None:
        for field_name in (
            "observation_id",
            "independence_key",
            "ancestor_lineage",
            "output_label",
            "provenance",
        ):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} must be non-empty")
            object.__setattr__(self, field_name, value)
        if self.component_id is not None:
            component = str(self.component_id).strip()
            if not component:
                raise ValueError("component_id must be non-empty when present")
            object.__setattr__(self, "component_id", component)
        for field_name in (
            "structural_score",
            "row_margin",
            "column_margin",
            "acoustic_responsibility",
            "routing_authority",
            "bridge_authority",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be finite and in [0, 1]")
            object.__setattr__(self, field_name, value)
        if self.bridge_authority > self.routing_authority + 1e-12:
            raise ValueError("bridge_authority cannot exceed routing_authority")

    @property
    def effective_authority(self) -> float:
        if self.conflict:
            return 0.0
        uniqueness = min(float(self.row_margin), float(self.column_margin))
        return float(
            self.bridge_authority
            * self.structural_score
            * uniqueness
            * self.acoustic_responsibility
        )

    def resolved(self, component_id: str) -> "BridgeEvidence":
        value = str(component_id).strip()
        if not value:
            raise ValueError("component_id must be non-empty")
        if self.component_id is not None and self.component_id != value:
            raise ValueError("bridge proposal targets a different acoustic component")
        return BridgeEvidence(
            observation_id=self.observation_id,
            independence_key=self.independence_key,
            ancestor_lineage=self.ancestor_lineage,
            component_id=value,
            output_label=self.output_label,
            provenance=self.provenance,
            structural_score=self.structural_score,
            row_margin=self.row_margin,
            column_margin=self.column_margin,
            conflict=self.conflict,
            acoustic_responsibility=self.acoustic_responsibility,
            routing_authority=self.routing_authority,
            bridge_authority=self.bridge_authority,
        )


@dataclass(frozen=True)
class EmbeddingViewSet:
    """Bounded correlated speaker-embedding measurements from one audio event."""

    source_key: str
    views: tuple[np.ndarray, ...]
    intervals: tuple[tuple[float, float], ...]
    native_chunk_duration_sec: float
    circle_padded: bool
    center: np.ndarray
    instability: float
    unique_coverage_sec: float

    @classmethod
    def create(
        cls,
        *,
        source_key: str,
        views: Sequence[np.ndarray],
        intervals: Sequence[tuple[float, float]],
        native_chunk_duration_sec: float,
        circle_padded: bool = False,
    ) -> "EmbeddingViewSet":
        key = str(source_key).strip()
        if not key:
            raise ValueError("view-set source_key must be non-empty")
        native = float(native_chunk_duration_sec)
        if not math.isfinite(native) or native <= 0.0:
            raise ValueError("native_chunk_duration_sec must be finite and positive")
        if not views or len(views) != len(intervals):
            raise ValueError("views and intervals must be non-empty and aligned")
        if len(views) > 4:
            raise ValueError("one observation may contain at most four views")

        normalized: list[np.ndarray] = []
        dimension: Optional[int] = None
        canonical_intervals: list[tuple[float, float]] = []
        for vector, interval in zip(views, intervals):
            value = np.asarray(vector, dtype=np.float32)
            if value.ndim != 1 or not np.all(np.isfinite(value)):
                raise ValueError("embedding views must be finite rank-1 arrays")
            norm = float(np.linalg.norm(value))
            if norm <= 1e-12:
                raise ValueError("embedding views must be non-zero")
            if dimension is None:
                dimension = int(value.shape[0])
            elif int(value.shape[0]) != dimension:
                raise ValueError("all embedding views must share one dimension")
            start, end = float(interval[0]), float(interval[1])
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0.0
                or end <= start
            ):
                raise ValueError("view intervals must be finite positive intervals")
            item = np.asarray(value / norm, dtype=np.float32)
            item.setflags(write=False)
            normalized.append(item)
            canonical_intervals.append((start, end))

        pairs = sorted(
            zip(normalized, canonical_intervals),
            key=lambda item: (
                tuple(float(value) for value in item[0]),
                item[1][0],
                item[1][1],
            ),
        )
        ordered_views = tuple(item[0] for item in pairs)
        ordered_intervals = tuple(item[1] for item in pairs)
        matrix = np.stack(ordered_views).astype(np.float64, copy=False)
        similarities = np.clip(matrix @ matrix.T, -1.0, 1.0)
        angular = np.arccos(similarities)
        totals = angular.sum(axis=1)
        medoid_index = min(
            range(len(ordered_views)),
            key=lambda index: (
                float(totals[index]),
                tuple(float(value) for value in ordered_views[index]),
            ),
        )
        center = np.asarray(ordered_views[medoid_index], dtype=np.float32).copy()
        center.setflags(write=False)
        if len(ordered_views) == 1:
            instability = math.pi / 3.0
        else:
            instability = float(np.median(angular[medoid_index]))

        coverage = 0.0
        cursor_start: Optional[float] = None
        cursor_end: Optional[float] = None
        for start, end in sorted(ordered_intervals):
            if cursor_start is None or start > float(cursor_end) + 1e-9:
                if cursor_start is not None:
                    coverage += float(cursor_end) - cursor_start
                cursor_start, cursor_end = start, end
            else:
                cursor_end = max(float(cursor_end), end)
        if cursor_start is not None:
            coverage += float(cursor_end) - cursor_start

        return cls(
            source_key=key,
            views=ordered_views,
            intervals=ordered_intervals,
            native_chunk_duration_sec=native,
            circle_padded=bool(circle_padded),
            center=center,
            instability=float(instability),
            unique_coverage_sec=float(coverage),
        )


def _normalized_causal_context(
    causal_context: Optional[Mapping[str, float]],
) -> tuple[tuple[str, float], ...]:
    values = {
        str(output_label): float(score)
        for output_label, score in (causal_context or {}).items()
    }
    if any(not math.isfinite(score) for score in values.values()):
        raise ValueError("causal context scores must be finite")
    scale = max((abs(score) for score in values.values()), default=0.0)
    return tuple(
        sorted(
            (
                output_label,
                float(np.clip(score / scale, -1.0, 1.0)) if scale > 0.0 else 0.0,
            )
            for output_label, score in values.items()
        )
    )


def assess_embedding_views(
    view_set: Optional[EmbeddingViewSet],
    *,
    independence_key: str,
    causal_context: Optional[Mapping[str, float]] = None,
    text: Optional[str] = None,
    purity: float = 1.0,
    routing_admissible: bool = True,
    profile_admissible: bool = True,
    novelty_admissible: bool = True,
    mixed: bool = False,
    duplicate: bool = False,
) -> EvidenceAssessment:
    """Assess one bounded correlated view set without positive text authority."""

    key = str(independence_key).strip()
    if not key:
        raise ValueError("independence_key must be non-empty")
    context = _normalized_causal_context(causal_context)
    bounded_purity = float(np.clip(float(purity), 0.0, 1.0))
    if view_set is None:
        return EvidenceAssessment(
            x=None,
            rho=0.0,
            routing_allowed=False,
            profile_allowed=False,
            novelty_allowed=False,
            independence_key=key,
            causal_context=context,
            purity_reliability=bounded_purity,
            duration_reliability=0.0,
            content_reliability=0.0,
            duplicate=bool(duplicate),
        )

    coverage_fraction = float(
        np.clip(
            view_set.unique_coverage_sec / view_set.native_chunk_duration_sec,
            0.0,
            1.0,
        )
    )
    instability_fraction = float(np.clip(view_set.instability / math.pi, 0.0, 1.0))
    stability_penalty = 1.0 / (1.0 + instability_fraction)
    route_precision = bounded_purity * coverage_fraction * stability_penalty
    usable = bool(
        not duplicate
        and not mixed
        and routing_admissible
        and route_precision > 0.0
    )
    units = tuple(
        match.group(0).casefold()
        for match in _LEXICAL_UNIT.finditer(str(text or ""))
    )
    diversity = len(set(units))
    low_information_text = text is not None and (len(units) < 2 or diversity < 2)
    profile_allowed = bool(
        usable
        and profile_admissible
        and not view_set.circle_padded
        and not low_information_text
    )
    novelty_allowed = bool(profile_allowed and novelty_admissible)
    profile_precision = route_precision if profile_allowed else 0.0
    novelty_precision = profile_precision if novelty_allowed else 0.0
    return EvidenceAssessment(
        x=view_set.center,
        rho=float(route_precision if usable else 0.0),
        routing_allowed=usable,
        profile_allowed=profile_allowed,
        novelty_allowed=novelty_allowed,
        independence_key=key,
        causal_context=context,
        purity_reliability=bounded_purity,
        duration_reliability=coverage_fraction,
        content_reliability=0.0,
        duplicate=bool(duplicate),
        views=view_set.views,
        center=view_set.center,
        instability=float(view_set.instability),
        unique_coverage_sec=float(view_set.unique_coverage_sec),
        precision_route=float(route_precision if usable else 0.0),
        precision_profile=float(profile_precision),
        precision_new=float(novelty_precision),
        circle_padded=bool(view_set.circle_padded),
    )


@dataclass(frozen=True)
class EvidenceCandidate:
    source_key: str
    vector: np.ndarray
    tick: float
    clean_duration_sec: float
    text: Optional[str]
    purity: float = 1.0

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float32)
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            raise ValueError("evidence vector must be a finite rank-1 array")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("evidence vector must be non-zero")
        if not math.isfinite(float(self.tick)):
            raise ValueError("evidence tick must be finite")
        if (
            not math.isfinite(float(self.clean_duration_sec))
            or float(self.clean_duration_sec) < 0.0
        ):
            raise ValueError("clean duration must be finite and non-negative")
        if not math.isfinite(float(self.purity)) or float(self.purity) < 0.0:
            raise ValueError("purity must be finite and non-negative")
        normalized = np.asarray(vector / norm, dtype=np.float32)
        normalized.setflags(write=False)
        object.__setattr__(self, "source_key", str(self.source_key))
        object.__setattr__(self, "vector", normalized)
        object.__setattr__(self, "tick", float(self.tick))
        object.__setattr__(
            self,
            "clean_duration_sec",
            float(self.clean_duration_sec),
        )
        object.__setattr__(self, "purity", float(self.purity))

    @property
    def lexical_units(self) -> tuple[str, ...]:
        if self.text is None:
            return ()
        return tuple(
            match.group(0).casefold()
            for match in _LEXICAL_UNIT.finditer(str(self.text))
        )

    def permissions(
        self,
        *,
        min_profile_duration_sec: float,
        min_purity: float,
    ) -> EvidencePermissions:
        pure = self.purity + 1e-12 >= float(min_purity)
        units = self.lexical_units
        diversity = len(set(units))
        text_is_informative = self.text is None or (
            len(units) >= 2 and diversity >= 2
        )
        profile = bool(
            pure
            and self.clean_duration_sec + 1e-9
            >= float(min_profile_duration_sec)
            and text_is_informative
        )
        novelty = bool(
            profile
            and self.clean_duration_sec + 1e-9
            >= max(3.0, 2.0 * float(min_profile_duration_sec))
        )
        return EvidencePermissions(
            extractable=True,
            pure=pure,
            routing=pure,
            profile=profile,
            novelty=novelty,
        )

    def strength(self) -> EvidenceStrength:
        units = self.lexical_units
        diversity = len(set(units))
        duration_coverage = min(1.0, self.clean_duration_sec / 6.0)
        if self.text is None:
            content_coverage = 0.5
        else:
            content_coverage = min(1.0, diversity / 8.0)
        bounded_purity = min(1.0, max(0.0, self.purity))
        aggregate = bounded_purity * (
            0.55 * duration_coverage + 0.45 * content_coverage
        )
        return EvidenceStrength(
            clean_duration_sec=self.clean_duration_sec,
            lexical_unit_count=len(units),
            lexical_diversity=diversity,
            purity=bounded_purity,
            duration_coverage=duration_coverage,
            content_coverage=content_coverage,
            aggregate=float(aggregate),
        )


@dataclass(frozen=True)
class EvidenceRecord:
    sequence: int
    event_type: str
    source_key: str
    revision: int
    track_id: str
    association_reason: str
    local_symbol: str
    start: float
    end: float
    tick: float
    acoustic_start: Optional[float]
    acoustic_end: Optional[float]
    text: Optional[str]
    quality: float
    embedding: Optional[tuple[float, ...]]


class IdentityEvidenceLedger:
    """Append-only raw evidence and association audit ledger."""

    def __init__(self) -> None:
        self._records: list[EvidenceRecord] = []
        self._latest_revision_by_source: dict[str, int] = {}

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._records)

    def latest_revision(self, source_key: str) -> Optional[int]:
        return self._latest_revision_by_source.get(str(source_key))

    def append_observation(
        self,
        *,
        track_id: str,
        observation: object,
        association_reason: str,
    ) -> EvidenceRecord:
        embedding_value = getattr(observation, "embedding", None)
        embedding = (
            None
            if embedding_value is None
            else tuple(float(item) for item in np.asarray(embedding_value).tolist())
        )
        acoustic_interval = getattr(observation, "acoustic_interval")
        acoustic_start, acoustic_end = acoustic_interval
        has_acoustic_bounds = (
            getattr(observation, "acoustic_start", None) is not None
        )
        record = EvidenceRecord(
            sequence=len(self._records),
            event_type="observation",
            source_key=str(getattr(observation, "source_key", "") or ""),
            revision=int(getattr(observation, "revision", 0)),
            track_id=str(track_id),
            association_reason=str(association_reason),
            local_symbol=str(getattr(observation, "local_symbol")),
            start=float(getattr(observation, "start")),
            end=float(getattr(observation, "end")),
            tick=float(getattr(observation, "tick")),
            acoustic_start=float(acoustic_start) if has_acoustic_bounds else None,
            acoustic_end=float(acoustic_end) if has_acoustic_bounds else None,
            text=getattr(observation, "text", None),
            quality=float(getattr(observation, "quality")),
            embedding=embedding,
        )
        self._records.append(record)
        if record.source_key:
            self._latest_revision_by_source[record.source_key] = max(
                int(record.revision),
                self._latest_revision_by_source.get(record.source_key, -1),
            )
        return record


class VoiceProfile:
    """Bounded deterministic derived profile over immutable evidence.

    Recency is intentionally absent from both admission and replacement.
    An established coherent set resists one conflicting observation; a
    mutually coherent cohort from at least two independent ticks can refresh
    it when the complete candidate set has higher utility.
    """

    def __init__(
        self,
        *,
        max_samples: int,
        min_profile_duration_sec: float,
        min_purity: float,
        min_cohesion: float,
    ) -> None:
        if isinstance(max_samples, bool) or int(max_samples) < 1:
            raise ValueError("max_samples must be a positive integer")
        if not 0.0 <= float(min_cohesion) <= 1.0:
            raise ValueError("min_cohesion must be in [0, 1]")
        self.max_samples = int(max_samples)
        self.min_profile_duration_sec = float(min_profile_duration_sec)
        self.min_purity = float(min_purity)
        self.min_cohesion = float(min_cohesion)
        self._candidates: dict[str, EvidenceCandidate] = {}
        self._active_keys: tuple[str, ...] = ()
        self._quarantine_keys: tuple[str, ...] = ()
        self._state = ProfileState.UNSEEDED
        self._last_transition: dict[str, object] = {
            "state_before": ProfileState.UNSEEDED.value,
            "state_after": ProfileState.UNSEEDED.value,
            "active_before": [],
            "active_after": [],
            "quarantine_after": [],
            "selection_reason": "empty",
            "utility_components": {},
        }

    @property
    def state(self) -> ProfileState:
        return self._state

    @property
    def active_keys(self) -> tuple[str, ...]:
        return self._active_keys

    @property
    def quarantine_keys(self) -> tuple[str, ...]:
        return self._quarantine_keys

    @property
    def profile_candidate_keys(self) -> tuple[str, ...]:
        return tuple(
            candidate.source_key
            for candidate in sorted(
                self._candidates.values(),
                key=lambda item: item.source_key,
            )
            if candidate.permissions(
                min_profile_duration_sec=self.min_profile_duration_sec,
                min_purity=self.min_purity,
            ).profile
        )

    @property
    def last_transition(self) -> dict[str, object]:
        return dict(self._last_transition)

    @property
    def vectors(self) -> tuple[np.ndarray, ...]:
        return tuple(self.vector(key) for key in self._active_keys)

    def vector(self, key: object) -> np.ndarray:
        source_key = str(key)
        if source_key not in self._active_keys:
            raise KeyError(source_key)
        return self._candidates[source_key].vector.copy()

    def quality(self, key: object) -> float:
        source_key = str(key)
        if source_key not in self._candidates:
            raise KeyError(source_key)
        return float(self._candidates[source_key].purity)

    def contains(self, key: object) -> bool:
        return str(key) in self._active_keys

    def permissions(self, key: object) -> EvidencePermissions:
        source_key = str(key)
        if source_key not in self._candidates:
            raise KeyError(source_key)
        return self._candidates[source_key].permissions(
            min_profile_duration_sec=self.min_profile_duration_sec,
            min_purity=self.min_purity,
        )

    def strength(self, key: object) -> EvidenceStrength:
        source_key = str(key)
        if source_key not in self._candidates:
            raise KeyError(source_key)
        return self._candidates[source_key].strength()

    @staticmethod
    def _similarity(left: EvidenceCandidate, right: EvidenceCandidate) -> float:
        return float(np.dot(left.vector, right.vector))

    def _clusters(
        self,
        candidates: Sequence[EvidenceCandidate],
    ) -> tuple[tuple[EvidenceCandidate, ...], ...]:
        clusters: list[list[EvidenceCandidate]] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (-item.strength().aggregate, item.source_key),
        ):
            compatible = [
                cluster
                for cluster in clusters
                if all(
                    self._similarity(candidate, other) + 1e-12
                    >= self.min_cohesion
                    for other in cluster
                )
            ]
            if not compatible:
                clusters.append([candidate])
                continue
            target = max(
                compatible,
                key=lambda cluster: (
                    self._set_utility(tuple(cluster) + (candidate,))[0],
                    tuple(sorted(item.source_key for item in cluster)),
                ),
            )
            target.append(candidate)
        return tuple(
            tuple(sorted(cluster, key=lambda item: item.source_key))
            for cluster in clusters
        )

    def _set_utility(
        self,
        candidates: Sequence[EvidenceCandidate],
    ) -> tuple[float, dict[str, float]]:
        if not candidates:
            return 0.0, {
                "strength": 0.0,
                "source_independence": 0.0,
                "content_coverage": 0.0,
                "nonredundancy": 0.0,
                "cohesion": 0.0,
            }
        strengths = sum(item.strength().aggregate for item in candidates)
        independent_ticks = len({item.tick for item in candidates})
        lexical_units = {
            unit for item in candidates for unit in item.lexical_units
        }
        pairwise = [
            self._similarity(left, right)
            for index, left in enumerate(candidates)
            for right in candidates[index + 1 :]
        ]
        nonredundancy = (
            sum(1.0 - score for score in pairwise) / len(pairwise)
            if pairwise
            else 0.0
        )
        cohesion = min(pairwise) if pairwise else 1.0
        components = {
            "strength": float(strengths),
            "source_independence": float(0.08 * independent_ticks),
            "content_coverage": float(0.02 * min(12, len(lexical_units))),
            "nonredundancy": float(0.05 * nonredundancy),
            "cohesion": float(0.05 * cohesion),
        }
        return float(sum(components.values())), components

    def _select_active(
        self,
        candidates: Sequence[EvidenceCandidate],
    ) -> tuple[EvidenceCandidate, ...]:
        remaining = {item.source_key: item for item in candidates}
        selected: list[EvidenceCandidate] = []
        while remaining and len(selected) < self.max_samples:
            candidate = max(
                remaining.values(),
                key=lambda item: (
                    self._set_utility(tuple(selected) + (item,))[0],
                    item.source_key,
                ),
            )
            selected.append(candidate)
            remaining.pop(candidate.source_key, None)
        return tuple(sorted(selected, key=lambda item: item.source_key))

    def _cohort_refresh_ready(
        self,
        candidates: Sequence[EvidenceCandidate],
    ) -> bool:
        return bool(
            len({item.tick for item in candidates}) >= 2
            and sum(item.clean_duration_sec for item in candidates) + 1e-9
            >= max(1.5, 2.0 * self.min_profile_duration_sec)
        )

    def _derived_state(
        self,
        active: Sequence[EvidenceCandidate],
        routing_count: int,
    ) -> ProfileState:
        if not active:
            return (
                ProfileState.PROVISIONAL
                if routing_count > 0
                else ProfileState.UNSEEDED
            )
        if any(
            item.permissions(
                min_profile_duration_sec=self.min_profile_duration_sec,
                min_purity=self.min_purity,
            ).novelty
            for item in active
        ) or self._cohort_refresh_ready(active):
            return ProfileState.ESTABLISHED
        return ProfileState.PROVISIONAL

    def derive(self, candidates: Iterable[EvidenceCandidate]) -> None:
        candidate_map = {
            item.source_key: item
            for item in sorted(candidates, key=lambda item: item.source_key)
        }
        state_before = self._state
        active_before = self._active_keys
        self._candidates = candidate_map
        routing = [
            item
            for item in candidate_map.values()
            if item.permissions(
                min_profile_duration_sec=self.min_profile_duration_sec,
                min_purity=self.min_purity,
            ).routing
        ]
        eligible = [
            item
            for item in routing
            if item.permissions(
                min_profile_duration_sec=self.min_profile_duration_sec,
                min_purity=self.min_purity,
            ).profile
        ]
        clusters = self._clusters(eligible)
        chosen: tuple[EvidenceCandidate, ...] = ()
        selection_reason = "no_profile_permission"
        if clusters:
            current_cluster = next(
                (
                    cluster
                    for cluster in clusters
                    if set(active_before).intersection(
                        item.source_key for item in cluster
                    )
                ),
                None,
            )
            ranked = sorted(
                clusters,
                key=lambda cluster: (
                    -self._set_utility(self._select_active(cluster))[0],
                    tuple(item.source_key for item in cluster),
                ),
            )
            best_cluster = ranked[0]
            if state_before is ProfileState.ESTABLISHED and current_cluster:
                current_selected = self._select_active(current_cluster)
                current_utility = self._set_utility(current_selected)[0]
                challengers = [
                    cluster for cluster in ranked if cluster is not current_cluster
                ]
                refresh = next(
                    (
                        cluster
                        for cluster in challengers
                        if self._cohort_refresh_ready(cluster)
                        and self._set_utility(self._select_active(cluster))[0]
                        > current_utility + 1e-12
                    ),
                    None,
                )
                if refresh is not None:
                    chosen = self._select_active(refresh)
                    selection_reason = "coherent_cohort_refresh"
                else:
                    chosen = current_selected
                    selection_reason = (
                        "set_utility"
                        if tuple(item.source_key for item in chosen)
                        != tuple(active_before)
                        else "established_outlier_resistance"
                    )
            else:
                chosen = self._select_active(best_cluster)
                selection_reason = "set_utility"

        self._active_keys = tuple(item.source_key for item in chosen)
        self._quarantine_keys = tuple(
            item.source_key
            for item in sorted(routing, key=lambda candidate: candidate.source_key)
            if item.source_key not in self._active_keys
        )
        self._state = self._derived_state(chosen, len(routing))
        _utility, components = self._set_utility(chosen)
        self._last_transition = {
            "state_before": state_before.value,
            "state_after": self._state.value,
            "active_before": list(active_before),
            "active_after": list(self._active_keys),
            "quarantine_after": list(self._quarantine_keys),
            "selection_reason": selection_reason,
            "utility_components": components,
        }


__all__ = [
    "BridgeEvidence",
    "EvidenceCandidate",
    "EvidencePermissions",
    "EvidenceRecord",
    "EvidenceStrength",
    "IdentityEvidenceLedger",
    "ProfileState",
    "VoiceProfile",
]
