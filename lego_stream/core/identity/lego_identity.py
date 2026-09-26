from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math
from typing import Mapping, Optional, Sequence

import numpy as np

from .assignment import maximum_weight_assignment
from .causal_identity_graph import (
    canonical_cannot_link_edges,
    constrained_max_weight_assignment,
)
from .identity_evidence import BridgeEvidence, EvidenceAssessment

# The unit every authority threshold is measured in: what one outright
# establishment of an identity is worth.  Contributed by the two sites that open
# an identity outright rather than accumulate evidence for one
# (``seed:<output_label>``, ``allocation:<obs>``).
#
# Deliberately not a config field -- changing it rescales every threshold at
# once, which is a no-op up to a change of units.
IDENTITY_ESTABLISHMENT_SUPPORT = 1.0


def _unit_vector(value: np.ndarray, *, dimension: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (dimension,) or not np.all(np.isfinite(vector)):
        raise ValueError("embedding must be a finite vector with model dimension")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("embedding must be non-zero")
    normalized = np.asarray(vector / norm, dtype=np.float64)
    normalized.setflags(write=False)
    return normalized


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return float(1.0 / (1.0 + math.exp(-min(float(value), 700.0))))
    exp_value = math.exp(max(float(value), -700.0))
    return float(exp_value / (1.0 + exp_value))


@dataclass(frozen=True)
class ExemplarSnapshot:
    independence_key: str
    vector: tuple[float, ...]
    weight: float
    coverage_sec: float


@dataclass(frozen=True)
class ProfileSnapshot:
    profile_id: str
    exemplar_count: int
    exemplars: tuple[ExemplarSnapshot, ...]
    prototype: tuple[float, ...]
    dispersion: float
    effective_support: float


@dataclass(frozen=True)
class _Exemplar:
    independence_key: str
    vector: np.ndarray
    weight: float
    coverage_sec: float


class AcousticProfileState:
    """Fixed-capacity speaker profile; its prototype is the angular medoid."""

    def __init__(
        self,
        *,
        profile_id: str,
        dimension: int,
        exemplar_capacity: int,
    ) -> None:
        value = str(profile_id).strip()
        if not value:
            raise ValueError("profile_id must be non-empty")
        if int(dimension) < 2:
            raise ValueError("dimension must be at least two")
        if int(exemplar_capacity) not in {3, 4, 5}:
            raise ValueError("exemplar_capacity must be one of 3, 4, 5")
        self.profile_id = value
        self.dimension = int(dimension)
        self.exemplar_capacity = int(exemplar_capacity)
        self._exemplars: dict[str, _Exemplar] = {}
        self._prototype = np.zeros(self.dimension, dtype=np.float64)
        self._dispersion = 0.0

    @property
    def exemplar_count(self) -> int:
        return len(self._exemplars)

    @property
    def effective_support(self) -> float:
        return float(sum(min(1.0, item.weight) for item in self._exemplars.values()))

    @property
    def prototype(self) -> Optional[np.ndarray]:
        if not self._exemplars:
            return None
        return self._prototype.copy()

    @property
    def dispersion(self) -> float:
        return float(self._dispersion)

    @property
    def independence_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._exemplars))

    @staticmethod
    def _medoid(items: Sequence[_Exemplar]) -> tuple[np.ndarray, float]:
        matrix = np.stack([item.vector for item in items])
        angles = np.arccos(np.clip(matrix @ matrix.T, -1.0, 1.0))
        totals = angles.sum(axis=1)
        index = min(
            range(len(items)),
            key=lambda item: (
                float(totals[item]),
                tuple(float(value) for value in matrix[item]),
                items[item].independence_key,
            ),
        )
        prototype = np.asarray(matrix[index], dtype=np.float64).copy()
        prototype.setflags(write=False)
        dispersion = 0.0 if len(items) == 1 else float(np.median(angles[index]))
        return prototype, dispersion

    @classmethod
    def _set_utility(cls, items: Sequence[_Exemplar]) -> tuple[float, tuple[str, ...]]:
        _prototype, dispersion = cls._medoid(items)
        mean_weight = float(np.mean([min(1.0, item.weight) for item in items]))
        mean_coverage = float(
            np.mean([item.coverage_sec / (item.coverage_sec + 1.5) for item in items])
        )
        utility = mean_weight + 0.15 * mean_coverage - dispersion / math.pi
        return float(utility), tuple(sorted(item.independence_key for item in items))

    def _rebuild(self) -> None:
        if not self._exemplars:
            self._prototype = np.zeros(self.dimension, dtype=np.float64)
            self._dispersion = 0.0
            return
        self._prototype, self._dispersion = self._medoid(
            [self._exemplars[key] for key in sorted(self._exemplars)]
        )

    def admit(self, assessment: EvidenceAssessment, *, weight: float) -> dict[str, object]:
        key = str(assessment.independence_key)
        if key in self._exemplars:
            return {"status": "duplicate", "admitted": False, "evicted": None}
        value = float(weight)
        if (
            not assessment.profile_allowed
            or assessment.center is None
            or not math.isfinite(value)
            or value <= 0.0
        ):
            return {"status": "inadmissible", "admitted": False, "evicted": None}
        candidate = _Exemplar(
            independence_key=key,
            vector=_unit_vector(assessment.center, dimension=self.dimension),
            weight=float(min(1.0, value)),
            coverage_sec=float(max(0.0, assessment.unique_coverage_sec)),
        )
        previous_keys = set(self._exemplars)
        pool = list(self._exemplars.values()) + [candidate]
        if len(pool) <= self.exemplar_capacity:
            selected = pool
        else:
            selected = max(
                combinations(pool, self.exemplar_capacity),
                key=self._set_utility,
            )
        self._exemplars = {
            item.independence_key: item
            for item in sorted(selected, key=lambda item: item.independence_key)
        }
        self._rebuild()
        admitted = key in self._exemplars
        evicted_keys = sorted(previous_keys - set(self._exemplars))
        return {
            "status": "admitted" if admitted else "quarantined",
            "admitted": admitted,
            "evicted": evicted_keys[0] if evicted_keys else None,
        }

    def score(
        self,
        assessment: EvidenceAssessment,
        *,
        prototype_weight: float,
    ) -> tuple[float, float]:
        if not self._exemplars or not assessment.views:
            raise ValueError("profile score requires exemplars and observation views")
        prototype = self._prototype
        exemplars = np.stack(
            [self._exemplars[key].vector for key in sorted(self._exemplars)]
        )
        scores: list[float] = []
        prototype_scores: list[float] = []
        for raw_view in assessment.views:
            view = _unit_vector(raw_view, dimension=self.dimension)
            prototype_score = float(np.dot(view, prototype))
            exemplar_scores = sorted(
                (float(value) for value in exemplars @ view),
                reverse=True,
            )
            top_count = min(2, len(exemplar_scores))
            correction = float(np.mean(exemplar_scores[:top_count]))
            scores.append(
                float(
                    prototype_weight * prototype_score
                    + (1.0 - prototype_weight) * correction
                )
            )
            prototype_scores.append(prototype_score)
        ordered = sorted(scores)
        aggregate = (
            float(np.mean(ordered[1:-1]))
            if len(ordered) >= 3
            else float(np.mean(ordered))
        )
        return aggregate, float(np.mean(prototype_scores))

    def snapshot(self) -> ProfileSnapshot:
        prototype = (
            tuple(float(value) for value in self._prototype)
            if self._exemplars
            else ()
        )
        return ProfileSnapshot(
            profile_id=self.profile_id,
            exemplar_count=len(self._exemplars),
            exemplars=tuple(
                ExemplarSnapshot(
                    independence_key=item.independence_key,
                    vector=tuple(float(value) for value in item.vector),
                    weight=float(item.weight),
                    coverage_sec=float(item.coverage_sec),
                )
                for item in (
                    self._exemplars[key] for key in sorted(self._exemplars)
                )
            ),
            prototype=prototype,
            dispersion=float(self._dispersion),
            effective_support=self.effective_support,
        )


@dataclass(frozen=True)
class LegoIdentityConfig:
    """Parameters of the default resolver.

    Two of the paper's thresholds are **derived properties rather than fields**,
    so no configuration can contradict the inequalities the paper states:
    ``new_output_authority_cost`` (the paper's beta) and ``binding_switch_cost``.
    Both follow from ``max_single_source_contribution``; see the properties below.
    """

    # The paper's nu_max: no single source of evidence may on its own authorize a
    # claim.  One quantity, not two -- the bridge and novelty channels both enter
    # as ``min(c, c * <precision factor>)``, i.e. ``c * min(1, factor)``.
    max_single_source_contribution: float = 0.55
    # The paper's eta: weight on the medoid, with 1 - eta on the mean of the two
    # exemplars nearest the turn.
    prototype_weight: float = 0.70
    # The paper's mu_0: the zero point of the evidence scale, at which a cosine
    # argues neither for nor against the same speaker.
    background_location: float = 0.45
    background_scale: float = 0.12
    # Discount on the carry-over term.  0.75 is the value Table 1 was produced
    # at; the released default stays on that published operating point.
    context_scale: float = 0.75
    unexplained_complexity_cost: float = 0.35
    # How many independent lineages a bank retains.  One quantity: the bridge and
    # novelty banks were never set apart.
    lineage_capacity: int = 4
    global_component_capacity: int = 1
    # Optional flicker suppression, off by default (0.0 disables it without a
    # branch): a bonus on the cell a global already occupies, so the assignment
    # prefers the pairing already on screen.  Buys label stability with accuracy;
    # the paper describes the output stage without it.
    display_inertia_utility: float = 0.0
    # Not what runs: ``core/identity/speakers.py`` forwards
    # ``association.risk_local_partition_provisional_enabled`` from the run config.
    local_partition_provisional_enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.local_partition_provisional_enabled, bool):
            raise ValueError("local_partition_provisional_enabled must be boolean")
        bounded = (
            "max_single_source_contribution",
            "prototype_weight",
            "context_scale",
            "unexplained_complexity_cost",
            "display_inertia_utility",
        )
        for name in bounded:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.prototype_weight > 1.0:
            raise ValueError("prototype_weight must be in [0, 1]")
        for name in ("background_scale",):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(float(self.background_location)):
            raise ValueError("background_location must be finite")
        # The paper's guarantee ``nu_max < beta`` is STRUCTURAL rather than
        # checked, because beta is derived as the midpoint of the interval the
        # inequality permits.  What must still be checked is that the cap leaves
        # a non-empty interval at all: an establishment is worth
        # ``IDENTITY_ESTABLISHMENT_SUPPORT``, so a per-source cap at or above
        # that leaves no room for a threshold between them.
        if (
            float(self.max_single_source_contribution)
            >= IDENTITY_ESTABLISHMENT_SUPPORT
        ):
            raise ValueError(
                "max_single_source_contribution must stay below one "
                "identity establishment, or no authority threshold exists"
            )
        # The ORDER of the two authority gates is the mechanism, not a
        # coincidence of tuning: taking over an output label another global
        # already holds must need more accumulated support than claiming one
        # nobody holds.  That ordering is structural, since both are derived.
        if int(self.lineage_capacity) < 2:
            raise ValueError("lineage_capacity must be at least two")
        if int(self.global_component_capacity) < 1:
            raise ValueError("global_component_capacity must be positive")

    @property
    def new_output_authority_cost(self) -> float:
        """Support required before a global may come to hold an output label.

        The paper's beta, derived rather than tuned.  Support accumulates from
        exactly two kinds of lineage, so beta is bounded on both sides and taken
        at the midpoint of the admissible interval:

            max_single_source_contribution  <  beta  <  IDENTITY_ESTABLISHMENT_SUPPORT
                                      0.55  <  beta  <  1.0

        giving 0.775.  The interval bounds beta; it does not make beta arbitrary
        within it.
        """

        return float(
            (
                float(self.max_single_source_contribution)
                + IDENTITY_ESTABLISHMENT_SUPPORT
            )
            / 2.0
        )

    @property
    def binding_switch_cost(self) -> float:
        """Support required to take over an output label another global holds.

        Derived, not tuned.  One establishment alone must not suffice to take a
        label from its holder, while an establishment plus corroborating acoustic
        evidence must, so the threshold sits at the midpoint between those two
        support levels:

            establishment + max_single_source_contribution / 2
            = 1.0 + 0.55 / 2 = 1.275
        """

        return float(
            IDENTITY_ESTABLISHMENT_SUPPORT
            + float(self.max_single_source_contribution) / 2.0
        )


@dataclass(frozen=True)
class LegoIdentityObservation:
    observation_id: str
    assessment: EvidenceAssessment
    bridge_evidence: tuple[BridgeEvidence, ...] = ()
    local_partition_key: Optional[str] = None
    local_symbol: Optional[str] = None
    partition_excluded_output_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        value = str(self.observation_id).strip()
        if not value:
            raise ValueError("observation_id must be non-empty")
        object.__setattr__(self, "observation_id", value)
        canonical = tuple(
            sorted(
                self.bridge_evidence,
                key=lambda item: (
                    item.output_label,
                    item.ancestor_lineage,
                    item.provenance,
                    item.independence_key,
                ),
            )
        )
        for item in canonical:
            if item.observation_id != value:
                raise ValueError("bridge observation_id must match its observation")
            if item.independence_key != self.assessment.independence_key:
                raise ValueError("bridge independence_key must match its assessment")
        object.__setattr__(self, "bridge_evidence", canonical)
        partition_key = (
            str(self.local_partition_key).strip()
            if self.local_partition_key is not None
            else None
        )
        local_symbol = (
            str(self.local_symbol).strip()
            if self.local_symbol is not None
            else None
        )
        excluded = tuple(
            sorted(
                {
                    str(output_label).strip()
                    for output_label in self.partition_excluded_output_labels
                    if str(output_label).strip()
                }
            )
        )
        if excluded and (not partition_key or not local_symbol):
            raise ValueError(
                "partition exclusions require local_partition_key and local_symbol"
            )
        object.__setattr__(self, "local_partition_key", partition_key)
        object.__setattr__(self, "local_symbol", local_symbol)
        object.__setattr__(self, "partition_excluded_output_labels", excluded)


@dataclass(frozen=True)
class PublicationAction:
    output_label: Optional[str]
    score: float
    best_effort: bool
    reason: str


@dataclass(frozen=True)
class AcousticExplanation:
    component_id: Optional[str]
    kind: str
    score: float
    inlier_responsibility: float


@dataclass(frozen=True)
class BindingAction:
    kind: str
    component_id: Optional[str]
    output_label: Optional[str]
    utility: float
    margin: float


@dataclass(frozen=True)
class ProfileUpdateAction:
    component_id: Optional[str]
    output_label: Optional[str]
    weight: float
    reason: str


@dataclass(frozen=True)
class NewOutputLabelAction:
    allocate: bool
    component_id: Optional[str]
    utility: float
    reason: str
    kind: str


@dataclass(frozen=True)
class BridgeSummarySnapshot:
    component_id: str
    output_label: str
    independent_lineages: tuple[str, ...]
    total_contribution: float
    conflict_count: int


@dataclass(frozen=True)
class LegoIdentityDecision:
    observation_id: str
    global_identity_id: Optional[str]
    publication_action: PublicationAction
    acoustic_explanation: AcousticExplanation
    binding_action: BindingAction
    profile_update_action: ProfileUpdateAction
    new_output_action: NewOutputLabelAction
    resolved_bridge: Optional[BridgeEvidence]
    bridge_contribution: float
    novelty_lineage: str
    novelty_contribution: float
    forced_constraint: bool


@dataclass(frozen=True)
class LegoIdentityInference:
    observations: tuple[LegoIdentityObservation, ...]
    decisions: tuple[LegoIdentityDecision, ...]
    cannot_link: tuple[tuple[str, str], ...]
    binding_updates: tuple[tuple[str, Optional[str]], ...]
    embedding_global_updates: tuple[tuple[str, str], ...] = ()
    display_updates: tuple[tuple[str, Optional[str]], ...] = ()


@dataclass
class OutputHandleState:
    output_label: str
    weak_anchors: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class _OutputAcousticEvidence:
    durable_raw: Optional[float] = None
    durable_inlier: Optional[float] = None
    weak_anchor_raw: Optional[float] = None
    weak_anchor_inlier: Optional[float] = None

    @property
    def has_evidence(self) -> bool:
        return self.durable_inlier is not None or self.weak_anchor_inlier is not None

    @property
    def routing_inlier(self) -> float:
        values = [
            value
            for value in (self.durable_inlier, self.weak_anchor_inlier)
            if value is not None
        ]
        return max(values, default=0.0)

    @property
    def absolute_utility(self) -> float:
        if self.durable_inlier is not None:
            durable = 2.0 * float(self.durable_inlier) - 1.0
            if self.weak_anchor_inlier is None:
                return durable
            weak_positive = max(0.0, 2.0 * float(self.weak_anchor_inlier) - 1.0)
            return max(durable, weak_positive)
        if self.weak_anchor_inlier is not None:
            # Startup anchors can recall a compatible output label, but never veto it.
            return float(self.weak_anchor_inlier)
        return 0.0

    def positive_raw_excess(self, background: float) -> float:
        values = [
            value
            for value in (self.durable_raw, self.weak_anchor_raw)
            if value is not None
        ]
        return max(0.0, max(values, default=float(background)) - float(background))


@dataclass
class _BridgeSummaryState:
    lineages: dict[str, float] = field(default_factory=dict)
    conflict_count: int = 0

    @property
    def total(self) -> float:
        return float(sum(self.lineages.values()))


@dataclass
class EmbeddingComponentState:
    component_id: str
    profile: AcousticProfileState
    novelty: dict[str, float] = field(default_factory=dict)
    bridges: dict[str, _BridgeSummaryState] = field(default_factory=dict)
    owner_global_id: Optional[str] = None


@dataclass
class GlobalIdentityState:
    global_identity_id: str
    component_ids: set[str] = field(default_factory=set)
    output_support: dict[str, _BridgeSummaryState] = field(default_factory=dict)
    provisional_output_label: Optional[str] = None


# Read-only alias for older imports; the resolver owns E via GlobalIdentityState.
AcousticComponentState = EmbeddingComponentState


class LegoIdentityModel:
    """Single causal owner of E components, G identities and the G<->O mapping.

    Three id vocabularies meet here and are deliberately distinct:

    - a ``component_id`` is one acoustic cluster (**E**) holding exemplars;
    - a ``global_id`` is one session speaker (**G**) owning a *set* of
      components, so E->G is many-to-one;
    - an ``output_label`` is the label shown to the listener (**O**), and the
      G->O mapping is one-to-one.

    The same G value also appears as ``global_identity_id`` when it is a
    parameter and as ``owner_global_id`` when read from a component's point of
    view.  Serialized keys are frozen independently of these spellings.

    This class is also where the paper's two assignments live.  ``infer`` solves
    the per-chunk constrained assignment over G, and ``_project_display_mapping``
    solves the one-to-one G->O assignment; the two authority thresholds the
    latter uses are the derived properties on ``LegoIdentityConfig``.
    """

    def __init__(
        self,
        *,
        dimension: int,
        exemplar_capacity: int,
        component_capacity: int,
        challenger_capacity: int,
        config: Optional[LegoIdentityConfig] = None,
    ) -> None:
        if int(dimension) < 2:
            raise ValueError("dimension must be at least two")
        if int(exemplar_capacity) not in {3, 4, 5}:
            raise ValueError("exemplar_capacity must be one of 3, 4, 5")
        if int(component_capacity) < 1:
            raise ValueError("component_capacity must be positive")
        if int(challenger_capacity) < 1:
            raise ValueError("challenger_capacity must be positive")
        self.dimension = int(dimension)
        self.exemplar_capacity = int(exemplar_capacity)
        self.component_capacity = int(component_capacity)
        self.challenger_capacity = int(challenger_capacity)
        self.config = config or LegoIdentityConfig()
        self.global_component_capacity = int(self.config.global_component_capacity)
        self._output: dict[str, OutputHandleState] = {}
        self._components: dict[str, EmbeddingComponentState] = {}
        self._global_identities: dict[str, GlobalIdentityState] = {}
        self._global_to_output: dict[str, str] = {}
        self._output_to_global: dict[str, str] = {}
        self._applied_observations: set[str] = set()

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._components))

    @property
    def global_identity_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._global_identities))

    @property
    def display_mapping_items(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._global_to_output.items()))

    @property
    def inactive_output_labels(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._output) - set(self._output_to_global)))

    @property
    def active_provisional_display_items(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (global_id, output_label)
                for global_id, output_label in self._global_to_output.items()
                if self._global_identities[global_id].provisional_output_label == output_label
            )
        )

    @property
    def provisional_display_items(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (global_id, state.provisional_output_label)
                for global_id, state in self._global_identities.items()
                if state.provisional_output_label is not None
            )
        )

    @property
    def binding_items(self) -> tuple[tuple[str, str], ...]:
        """Compatibility view of current future-display E->P associations."""

        return tuple(
            sorted(
                (component_id, self._global_to_output[state.owner_global_id])
                for component_id, state in self._components.items()
                if state.owner_global_id in self._global_to_output
            )
        )

    @property
    def applied_observation_count(self) -> int:
        return len(self._applied_observations)

    def binding_for_component(self, component_id: str) -> Optional[str]:
        state = self._components.get(str(component_id))
        if state is None or state.owner_global_id is None:
            return None
        return self._global_to_output.get(state.owner_global_id)

    def component_for_output(self, output_label: str) -> Optional[str]:
        global_id = self._output_to_global.get(str(output_label))
        if global_id is None:
            return None
        component_ids = self._global_identities[global_id].component_ids
        return min(component_ids) if component_ids else None

    def has_output(self, output_label: str) -> bool:
        return str(output_label) in self._output

    def global_identity_for_component(self, component_id: str) -> Optional[str]:
        state = self._components.get(str(component_id))
        return None if state is None else state.owner_global_id

    def global_identity_for_output(self, output_label: str) -> Optional[str]:
        return self._output_to_global.get(str(output_label))

    def output_for_global_identity(self, global_identity_id: str) -> Optional[str]:
        return self._global_to_output.get(str(global_identity_id))

    def global_identity_component_ids(self, global_identity_id: str) -> tuple[str, ...]:
        state = self._global_identities.get(str(global_identity_id))
        return () if state is None else tuple(sorted(state.component_ids))

    def global_output_support_count(self, global_identity_id: str) -> int:
        state = self._global_identities.get(str(global_identity_id))
        return 0 if state is None else len(state.output_support)

    def eg_topology_snapshot(self) -> tuple[object, ...]:
        return (
            tuple(
                (
                    global_id,
                    tuple(sorted(state.component_ids)),
                    self._global_to_output.get(global_id),
                    state.provisional_output_label,
                )
                for global_id, state in sorted(self._global_identities.items())
            ),
            tuple(
                (component_id, state.owner_global_id)
                for component_id, state in sorted(self._components.items())
            ),
        )

    def challenger_output_labels(self, component_id: str) -> tuple[str, ...]:
        state = self._components[str(component_id)]
        return tuple(
            sorted(
                output_label
                for output_label in state.bridges
                if output_label != self.binding_for_component(component_id)
            )
        )

    def profile_snapshot(self, component_id: str) -> ProfileSnapshot:
        return self._components[str(component_id)].profile.snapshot()

    def bridge_snapshot(self, component_id: str, output_label: str) -> BridgeSummarySnapshot:
        state = self._components.get(str(component_id))
        summary = None if state is None else state.bridges.get(str(output_label))
        return BridgeSummarySnapshot(
            component_id=str(component_id),
            output_label=str(output_label),
            independent_lineages=tuple(sorted(summary.lineages)) if summary else (),
            total_contribution=float(summary.total) if summary else 0.0,
            conflict_count=int(summary.conflict_count) if summary else 0,
        )

    def state_snapshot(self) -> tuple[object, ...]:
        return (
            tuple(sorted(self._output)),
            tuple(
                (component_id, self._components[component_id].profile.snapshot())
                for component_id in sorted(self._components)
            ),
            self.eg_topology_snapshot(),
            self.binding_items,
            tuple(
                (
                    component_id,
                    output_label,
                    tuple(sorted(summary.lineages.items())),
                    summary.conflict_count,
                )
                for component_id in sorted(self._components)
                for output_label, summary in sorted(self._components[component_id].bridges.items())
            ),
            tuple(
                (
                    global_id,
                    output_label,
                    tuple(sorted(summary.lineages.items())),
                    summary.conflict_count,
                )
                for global_id, state in sorted(self._global_identities.items())
                for output_label, summary in sorted(state.output_support.items())
            ),
        )

    @staticmethod
    def _shadow_vector(value: Sequence[float] | np.ndarray) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).copy()
        vector.setflags(write=False)
        return vector

    @classmethod
    def _shadow_profile_snapshot(
        cls,
        snapshot: ProfileSnapshot,
    ) -> dict[str, object]:
        return {
            "profile_id": snapshot.profile_id,
            "exemplar_count": int(snapshot.exemplar_count),
            "exemplars": [
                {
                    "independence_key": item.independence_key,
                    "vector": cls._shadow_vector(item.vector),
                    "weight": float(item.weight),
                    "coverage_sec": float(item.coverage_sec),
                }
                for item in snapshot.exemplars
            ],
            "prototype": (
                cls._shadow_vector(snapshot.prototype)
                if snapshot.prototype
                else None
            ),
            "dispersion": float(snapshot.dispersion),
            "effective_support": float(snapshot.effective_support),
        }

    def shadow_snapshot(self) -> dict[str, object]:
        """Return a defensive, observer-safe copy of all risk-model state."""

        output_labels = []
        for output_label in sorted(self._output):
            state = self._output[output_label]
            output_labels.append(
                {
                    "output_label": output_label,
                    "weak_anchors": [
                        {
                            "independence_key": key,
                            "vector": self._shadow_vector(state.weak_anchors[key]),
                        }
                        for key in sorted(state.weak_anchors)
                    ],
                    "global_identity_id": self._output_to_global.get(output_label),
                    "embedding_component_id": self.component_for_output(output_label),
                    "component_id": self.component_for_output(output_label),
                }
            )
        components = []
        for component_id in sorted(self._components):
            state = self._components[component_id]
            components.append(
                {
                    "embedding_component_id": component_id,
                    "component_id": component_id,
                    "owner_global_identity_id": state.owner_global_id,
                    "owner_output_label": self.binding_for_component(component_id),
                    "profile": self._shadow_profile_snapshot(
                        state.profile.snapshot()
                    ),
                    "novelty": {
                        key: float(value)
                        for key, value in sorted(state.novelty.items())
                    },
                    "bridges": [
                        {
                            "output_label": output_label,
                            "lineages": {
                                key: float(value)
                                for key, value in sorted(summary.lineages.items())
                            },
                            "total_contribution": float(summary.total),
                            "conflict_count": int(summary.conflict_count),
                        }
                        for output_label, summary in sorted(state.bridges.items())
                    ],
                    "challenger_output_labels": list(
                        self.challenger_output_labels(component_id)
                    ),
                }
            )
        global_identities = []
        for global_id in sorted(self._global_identities):
            state = self._global_identities[global_id]
            global_identities.append(
                {
                    "global_identity_id": global_id,
                    "embedding_component_ids": sorted(state.component_ids),
                    "published_speaker_id": self._global_to_output.get(global_id),
                    "provisional_published_speaker_id": state.provisional_output_label,
                    "provisional_active": bool(
                        state.provisional_output_label is not None
                        and self._global_to_output.get(global_id)
                        == state.provisional_output_label
                    ),
                    "output_support": [
                        {
                            "published_speaker_id": output_label,
                            "lineages": {
                                key: float(value)
                                for key, value in sorted(summary.lineages.items())
                            },
                            "total_contribution": float(summary.total),
                            "conflict_count": int(summary.conflict_count),
                        }
                        for output_label, summary in sorted(state.output_support.items())
                    ],
                }
            )
        return {
            "dimension": int(self.dimension),
            "exemplar_capacity": int(self.exemplar_capacity),
            "component_capacity": int(self.component_capacity),
            "challenger_capacity": int(self.challenger_capacity),
            # `vars()` omits the two derived `@property` thresholds, so add them
            # explicitly or the snapshot silently loses them.
            "config": {
                **{
                    key: value
                    for key, value in sorted(vars(self.config).items())
                },
                "new_output_authority_cost": float(
                    self.config.new_output_authority_cost
                ),
                "binding_switch_cost": float(self.config.binding_switch_cost),
            },
            "output_labels": output_labels,
            "components": components,
            "embedding_components": components,
            "global_identities": global_identities,
            "display_mapping": [list(item) for item in self.display_mapping_items],
            "active_provisional_display_mapping": [
                list(item) for item in self.active_provisional_display_items
            ],
            "binding": [list(item) for item in self.binding_items],
            "applied_observation_ids": sorted(self._applied_observations),
            "state_size_bytes": int(self.state_size_bytes()),
        }

    def state_size_bytes(self) -> int:
        exemplar_bytes = sum(
            state.profile.exemplar_count * self.dimension * 8
            for state in self._components.values()
        )
        anchor_bytes = sum(
            value.nbytes
            for state in self._output.values()
            for value in state.weak_anchors.values()
        )
        summary_bytes = sum(
            32 * len(summary.lineages)
            for state in self._components.values()
            for summary in state.bridges.values()
        )
        global_summary_bytes = sum(
            32 * len(summary.lineages)
            for state in self._global_identities.values()
            for summary in state.output_support.values()
        )
        topology_bytes = 32 * (
            len(self._global_identities)
            + sum(len(state.component_ids) for state in self._global_identities.values())
            + sum(
                state.provisional_output_label is not None
                for state in self._global_identities.values()
            )
        )
        return int(
            exemplar_bytes
            + anchor_bytes
            + summary_bytes
            + global_summary_bytes
            + topology_bytes
        )

    @staticmethod
    def _seed_global_id(output_label: str) -> str:
        return f"G:output:{output_label}"

    @staticmethod
    def _observation_global_id(observation_id: str) -> str:
        return f"G:{observation_id}"

    def _ensure_global_identity(self, global_identity_id: str) -> GlobalIdentityState:
        value = str(global_identity_id)
        return self._global_identities.setdefault(
            value,
            GlobalIdentityState(global_identity_id=value),
        )

    def _set_component_global(self, component_id: str, global_identity_id: str) -> bool:
        component = self._components.get(str(component_id))
        global_state = self._global_identities.get(str(global_identity_id))
        if component is None or global_state is None:
            return False
        if component.owner_global_id == global_identity_id:
            global_state.component_ids.add(component.component_id)
            return True
        if component.owner_global_id is not None:
            return False
        if len(global_state.component_ids) >= self.global_component_capacity:
            return False
        component.owner_global_id = global_identity_id
        global_state.component_ids.add(component.component_id)
        return True

    def _set_display_mapping(self, mapping: Mapping[str, Optional[str]]) -> None:
        next_global_to_output: dict[str, str] = {}
        next_output_to_global: dict[str, str] = {}
        for global_id, output_label in sorted(mapping.items()):
            if global_id not in self._global_identities or output_label is None:
                continue
            if output_label not in self._output or output_label in next_output_to_global:
                continue
            next_global_to_output[global_id] = output_label
            next_output_to_global[output_label] = global_id
        self._global_to_output = next_global_to_output
        self._output_to_global = next_output_to_global

    def _add_global_output_support(
        self,
        global_identity_id: str,
        output_label: str,
        *,
        lineage: str,
        contribution: float,
        conflict: bool = False,
    ) -> None:
        state = self._ensure_global_identity(global_identity_id)
        summary = state.output_support.setdefault(str(output_label), _BridgeSummaryState())
        key = str(lineage)
        summary.lineages[key] = max(
            summary.lineages.get(key, 0.0),
            max(0.0, float(contribution)),
        )
        summary.lineages = self._bounded_lineages(
            summary.lineages,
            self.config.lineage_capacity,
        )
        summary.conflict_count += int(conflict)
        incumbent = self._global_to_output.get(global_identity_id)
        ordered = sorted(
            state.output_support.items(),
            key=lambda item: (
                0 if item[0] == incumbent else 1,
                -item[1].total,
                item[0],
            ),
        )[: self.challenger_capacity + 1]
        state.output_support = {
            key: value for key, value in sorted(ordered)
        }

    def ensure_output(
        self,
        output_label: str,
        *,
        weak_anchor: Optional[EvidenceAssessment] = None,
        profile: Optional[AcousticProfileState] = None,
    ) -> str:
        value = str(output_label).strip()
        if not value:
            raise ValueError("output_label must be non-empty")
        handle = self._output.setdefault(value, OutputHandleState(output_label=value))
        if weak_anchor is not None and weak_anchor.center is not None:
            handle.weak_anchors[str(weak_anchor.independence_key)] = _unit_vector(
                weak_anchor.center,
                dimension=self.dimension,
            )
            handle.weak_anchors = {
                key: handle.weak_anchors[key]
                for key in sorted(handle.weak_anchors)[-2:]
            }
        if profile is not None:
            if profile.dimension != self.dimension:
                raise ValueError("profile dimension does not match model")
            if profile.exemplar_capacity != self.exemplar_capacity:
                raise ValueError("profile capacity does not match model")
            component_id = self.component_for_output(value)
            if component_id is None:
                component_id = f"E:output:{value}"
                if len(self._components) >= self.component_capacity:
                    raise RuntimeError("component capacity exhausted while seeding an output label")
                global_id = self._seed_global_id(value)
                self._ensure_global_identity(global_id)
                self._components[component_id] = EmbeddingComponentState(
                    component_id=component_id,
                    profile=profile,
                    owner_global_id=global_id,
                )
                self._global_identities[global_id].component_ids.add(component_id)
                self._add_global_output_support(
                    global_id,
                    value,
                    lineage=f"seed:{value}",
                    contribution=1.0,
                )
                self._set_display_mapping(
                    {
                        **{key: label for key, label in self._global_to_output.items()},
                        global_id: value,
                    }
                )
            else:
                self._components[component_id].profile = profile
        return value

    @staticmethod
    def _canonical_edges(
        observation_ids: set[str],
        cannot_link: Sequence[tuple[str, str]],
    ) -> tuple[tuple[str, str], ...]:
        return canonical_cannot_link_edges(observation_ids, cannot_link)

    @staticmethod
    def _constrained_assignment(
        row_ids: Sequence[str],
        candidates: Mapping[str, Sequence[tuple[str, float]]],
        edges: Sequence[tuple[str, str]],
    ) -> dict[str, str]:
        result = constrained_max_weight_assignment(
            row_ids,
            candidates,
            edges,
            no_feasible_message="event has no feasible lego assignment",
        )
        return result.mapping

    def _component_score(
        self,
        state: EmbeddingComponentState,
        assessment: EvidenceAssessment,
    ) -> tuple[float, float, float]:
        raw, _prototype = state.profile.score(
            assessment,
            prototype_weight=self.config.prototype_weight,
        )
        observation_dispersion = 1.0 - math.cos(float(assessment.instability))
        profile_dispersion = 1.0 - math.cos(float(state.profile.dispersion))
        scale = self.config.background_scale + 0.5 * (
            observation_dispersion + profile_dispersion
        )
        score = float((raw - self.config.background_location) / scale)
        return float(raw), score, _sigmoid(score)

    def _acoustic_candidates(
        self,
        observation: LegoIdentityObservation,
    ) -> list[tuple[str, float, float, float]]:
        assessment = observation.assessment
        if not assessment.routing_allowed or assessment.center is None or not assessment.views:
            return [(f"NONE:{observation.observation_id}", 0.0, 0.0, -1.0)]
        values: list[tuple[str, float, float, float]] = []
        for component_id in sorted(self._components):
            if self._components[component_id].profile.exemplar_count == 0:
                continue
            raw, score, inlier = self._component_score(
                self._components[component_id],
                assessment,
            )
            values.append((component_id, score, inlier, raw))
        values.append(
            (
                f"E:{observation.observation_id}",
                -self.config.unexplained_complexity_cost,
                1.0,
                -1.0,
            )
        )
        return values

    def _output_acoustic_scores(
        self,
        assessment: EvidenceAssessment,
    ) -> dict[str, _OutputAcousticEvidence]:
        if not assessment.routing_allowed or assessment.center is None or not assessment.views:
            return {
                output_label: _OutputAcousticEvidence()
                for output_label in sorted(self._output)
            }
        center = _unit_vector(assessment.center, dimension=self.dimension)
        values: dict[str, _OutputAcousticEvidence] = {}
        for output_label in sorted(self._output):
            durable_raw: Optional[float] = None
            durable_inlier: Optional[float] = None
            global_id = self._output_to_global.get(output_label)
            component_scores = []
            if global_id is not None:
                for component_id in sorted(
                    self._global_identities[global_id].component_ids
                ):
                    component = self._components.get(component_id)
                    if component is None or component.profile.exemplar_count == 0:
                        continue
                    component_scores.append(
                        self._component_score(component, assessment)
                    )
            if component_scores:
                durable_raw, _score, durable_inlier = max(
                    component_scores,
                    key=lambda item: (item[2], item[0]),
                )
            anchors = self._output[output_label].weak_anchors.values()
            weak_raw: Optional[float] = None
            weak_inlier: Optional[float] = None
            if anchors:
                weak_raw = max(float(np.dot(center, anchor)) for anchor in anchors)
                score = float(
                    np.clip(
                        (weak_raw - self.config.background_location)
                        / self.config.background_scale,
                        -3.5,
                        3.5,
                    )
                )
                weak_inlier = _sigmoid(score)
            values[output_label] = _OutputAcousticEvidence(
                durable_raw=durable_raw,
                durable_inlier=durable_inlier,
                weak_anchor_raw=weak_raw,
                weak_anchor_inlier=weak_inlier,
            )
        return values

    def _output_route_utilities(
        self,
        acoustic: Mapping[str, _OutputAcousticEvidence],
        precision: float,
        component_id: Optional[str],
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        for output_label in sorted(self._output):
            evidence = acoustic[output_label]
            peers = [
                other.routing_inlier
                for other_output, other in acoustic.items()
                if other_output != output_label and other.has_evidence
            ]
            alternate = max(peers) if peers else 0.5
            uniqueness = max(0.0, float(evidence.routing_inlier - alternate))
            utility = float(precision) * (evidence.absolute_utility + uniqueness)
            values[output_label] = float(utility)
        return values

    def _component_output_support(
        self,
        component_id: Optional[str],
        output_label: str,
        *,
        precision: float,
    ) -> float:
        component = self._components.get(component_id) if component_id is not None else None
        if component is None:
            return 0.0
        prototype = component.profile.prototype
        if prototype is None:
            return 0.0
        raw_scores = [
            float(np.dot(prototype, anchor))
            for anchor in self._output[output_label].weak_anchors.values()
        ]
        global_id = self._output_to_global.get(output_label)
        if global_id is not None:
            for output_component_id in sorted(
                self._global_identities[global_id].component_ids
            ):
                output_component = self._components.get(output_component_id)
                if output_component is None:
                    continue
                output_prototype = output_component.profile.prototype
                if output_prototype is not None:
                    raw_scores.append(float(np.dot(prototype, output_prototype)))
        raw = max(raw_scores, default=self.config.background_location)
        return float(
            float(precision)
            * max(0.0, raw - self.config.background_location)
        )

    def _novelty_contribution(
        self,
        assessment: EvidenceAssessment,
        component_id: Optional[str],
    ) -> float:
        if component_id is None or not assessment.novelty_allowed:
            return 0.0
        return float(
            min(
                self.config.max_single_source_contribution,
                self.config.max_single_source_contribution
                * assessment.novelty_precision,
            )
        )

    def _projected_novelty(
        self,
        component_id: str,
        lineage: str,
        contribution: float,
    ) -> float:
        lineages = dict(
            self._components.get(
                component_id,
                AcousticComponentState(
                    component_id=component_id,
                    profile=AcousticProfileState(
                        profile_id=component_id,
                        dimension=self.dimension,
                        exemplar_capacity=self.exemplar_capacity,
                    ),
                ),
            ).novelty
        )
        key = str(lineage)
        lineages[key] = max(lineages.get(key, 0.0), float(contribution))
        return float(sum(lineages.values()))

    @staticmethod
    def _novelty_lineage(observation: LegoIdentityObservation) -> str:
        roots = {item.ancestor_lineage for item in observation.bridge_evidence}
        if len(roots) == 1:
            return next(iter(roots))
        return str(observation.assessment.independence_key)

    def _select_bridge(
        self,
        observation: LegoIdentityObservation,
        component_id: Optional[str],
    ) -> tuple[Optional[BridgeEvidence], float]:
        assessment = observation.assessment
        if component_id is None or not assessment.profile_allowed:
            return None, 0.0
        by_lineage: dict[tuple[str, str], tuple[float, BridgeEvidence]] = {}
        for proposal in observation.bridge_evidence:
            if proposal.output_label not in self._output:
                continue
            try:
                resolved = proposal.resolved(component_id)
            except ValueError:
                continue
            contribution = float(
                min(
                    self.config.max_single_source_contribution,
                    self.config.max_single_source_contribution
                    * resolved.effective_authority
                    * assessment.profile_precision,
                )
            )
            if contribution > 0.0:
                key = (resolved.output_label, resolved.ancestor_lineage)
                previous = by_lineage.get(key)
                if previous is None or contribution > previous[0]:
                    by_lineage[key] = (contribution, resolved)
        values = list(by_lineage.values())
        if not values:
            return None, 0.0
        values.sort(
            key=lambda item: (
                -item[0],
                item[1].output_label,
                item[1].ancestor_lineage,
            )
        )
        # Ambiguity is settled upstream -- one claim per track, rivals dropped
        # rather than passed down -- so this list never holds two entries and a
        # top-1-minus-top-2 abstention margin here would be unreachable.
        return values[0][1], float(values[0][0])

    def _observation_output_affinity(
        self,
        observation: LegoIdentityObservation,
        component_id: Optional[str],
        bridge: Optional[BridgeEvidence],
        bridge_contribution: float,
    ) -> tuple[dict[str, float], dict[str, _OutputAcousticEvidence]]:
        context = dict(observation.assessment.causal_context)
        output_acoustic = self._output_acoustic_scores(observation.assessment)
        acoustic_route = self._output_route_utilities(
            output_acoustic,
            observation.assessment.routing_precision,
            component_id,
        )
        affinity: dict[str, float] = {}
        for output_label in sorted(self._output):
            component_support = self._component_output_support(
                component_id,
                output_label,
                precision=observation.assessment.routing_precision,
            )
            bridge_support = 0.0
            if bridge is not None and bridge.output_label == output_label:
                evidence = output_acoustic[output_label]
                if (
                    evidence.positive_raw_excess(self.config.background_location) > 0.0
                    or component_support > 0.0
                    or not self._global_identities
                ):
                    bridge_support = float(bridge_contribution)
            affinity[output_label] = float(
                self.config.context_scale * context.get(output_label, 0.0)
                + acoustic_route.get(output_label, 0.0)
                + component_support
                + bridge_support
            )
        return affinity, output_acoustic

    def _assign_global_identities(
        self,
        values: Sequence[LegoIdentityObservation],
        acoustic_info: Mapping[str, tuple[Optional[str], str, float, float]],
        output_affinity: Mapping[str, Mapping[str, float]],
        output_acoustic: Mapping[str, Mapping[str, _OutputAcousticEvidence]],
        selected_bridges: Mapping[str, tuple[Optional[BridgeEvidence], float]],
    ) -> tuple[dict[str, Optional[str]], tuple[tuple[str, str], ...]]:
        by_component: dict[str, list[LegoIdentityObservation]] = {}
        observation_global: dict[str, Optional[str]] = {}
        for observation in values:
            component_id = acoustic_info[observation.observation_id][0]
            if component_id is None:
                observation_global[observation.observation_id] = None
                continue
            by_component.setdefault(component_id, []).append(observation)

        component_global: dict[str, str] = {}
        unresolved: list[str] = []
        for component_id in sorted(by_component):
            current = self.global_identity_for_component(component_id)
            if current is not None:
                component_global[component_id] = current
            else:
                unresolved.append(component_id)

        candidate_globals = [
            global_id
            for global_id, state in sorted(self._global_identities.items())
            if len(state.component_ids) < self.global_component_capacity
        ]
        scores: dict[tuple[str, str], float] = {}
        for component_id in unresolved:
            observations = by_component[component_id]
            for global_id in candidate_globals:
                state = self._global_identities[global_id]
                output_label = self._global_to_output.get(global_id)
                if output_label is None:
                    continue
                score = max(
                    output_affinity[item.observation_id].get(output_label, 0.0)
                    for item in observations
                )
                if state.component_ids:
                    has_authoritative_cross_mode_support = any(
                        (
                            output_acoustic[item.observation_id][output_label]
                            .positive_raw_excess(self.config.background_location)
                            > 0.0
                        )
                        or self._component_output_support(
                            component_id,
                            output_label,
                            precision=item.assessment.routing_precision,
                        )
                        > 0.0
                        for item in observations
                    )
                    if not has_authoritative_cross_mode_support:
                        continue
                if score > 0.0:
                    scores[(component_id, global_id)] = float(score)

        attached = maximum_weight_assignment(
            sorted(unresolved),
            candidate_globals,
            scores,
        )
        updates: list[tuple[str, str]] = []
        for component_id in unresolved:
            global_id = attached.get(component_id, self._observation_global_id(component_id))
            component_global[component_id] = global_id
            if any(
                item.assessment.profile_allowed and item.assessment.center is not None
                for item in by_component[component_id]
            ):
                updates.append((component_id, global_id))

        for component_id, observations in by_component.items():
            global_id = component_global[component_id]
            for observation in observations:
                observation_global[observation.observation_id] = global_id
        return observation_global, tuple(sorted(set(updates)))

    def _project_global_support(
        self,
        observation_global: Mapping[str, Optional[str]],
        selected_bridges: Mapping[str, tuple[Optional[BridgeEvidence], float]],
    ) -> dict[str, dict[str, float]]:
        lineages: dict[tuple[str, str], dict[str, float]] = {
            (global_id, output_label): dict(summary.lineages)
            for global_id, state in self._global_identities.items()
            for output_label, summary in state.output_support.items()
        }
        for observation_id, global_id in observation_global.items():
            bridge, contribution = selected_bridges[observation_id]
            if global_id is None or bridge is None or contribution <= 0.0:
                continue
            values = lineages.setdefault((global_id, bridge.output_label), {})
            values[bridge.ancestor_lineage] = max(
                values.get(bridge.ancestor_lineage, 0.0),
                float(contribution),
            )
            values = self._bounded_lineages(
                values,
                self.config.lineage_capacity,
            )
            lineages[(global_id, bridge.output_label)] = values
        totals: dict[str, dict[str, float]] = {}
        for (global_id, output_label), values in lineages.items():
            totals.setdefault(global_id, {})[output_label] = float(sum(values.values()))
        return totals

    def _project_display_mapping(
        self,
        global_ids: Sequence[str],
        support_totals: Mapping[str, Mapping[str, float]],
        current_support: Mapping[tuple[str, str], float],
    ) -> tuple[
        dict[str, Optional[str]],
        dict[tuple[str, str], tuple[float, float]],
    ]:
        rows = tuple(sorted(set(str(item) for item in global_ids)))
        columns = tuple(sorted(self._output))
        scores: dict[tuple[str, str], float] = {}
        diagnostics: dict[tuple[str, str], tuple[float, float]] = {}
        for global_id in rows:
            row_scores: dict[str, float] = {}
            for output_label in columns:
                persistent = float(
                    support_totals.get(global_id, {}).get(output_label, 0.0)
                )
                current = float(current_support.get((global_id, output_label), 0.0))
                incumbent = self._global_to_output.get(global_id)
                if incumbent == output_label:
                    if (
                        self._global_identities[global_id].provisional_output_label
                        == output_label
                        and persistent <= 0.0
                        and current <= 0.0
                    ):
                        continue
                    value = persistent + current
                    value += self.config.display_inertia_utility
                elif incumbent is None:
                    if persistent < self.config.new_output_authority_cost:
                        continue
                    value = persistent + current
                else:
                    if persistent < self.config.binding_switch_cost:
                        continue
                    value = persistent + current
                row_scores[output_label] = value
            for output_label, value in row_scores.items():
                alternate = max(
                    (score for key, score in row_scores.items() if key != output_label),
                    default=0.0,
                )
                diagnostics[(global_id, output_label)] = (
                    float(value),
                    float(value - alternate),
                )
                if value > 0.0:
                    scores[(global_id, output_label)] = float(value)
        assigned = maximum_weight_assignment(rows, columns, scores)
        projected = {
            global_id: assigned.get(global_id)
            for global_id in rows
        }
        return projected, diagnostics

    def infer(
        self,
        observations: Sequence[LegoIdentityObservation],
        *,
        cannot_link: Sequence[tuple[str, str]] = (),
    ) -> LegoIdentityInference:
        values = tuple(sorted(observations, key=lambda item: item.observation_id))
        if not values:
            raise ValueError("event must contain at least one observation")
        if len({item.observation_id for item in values}) != len(values):
            raise ValueError("event observation ids must be unique")
        row_ids = [item.observation_id for item in values]
        edges = self._canonical_edges(set(row_ids), cannot_link)
        acoustic_rows = {
            item.observation_id: self._acoustic_candidates(item)
            for item in values
        }
        acoustic_assignment = self._constrained_assignment(
            row_ids,
            {
                row_id: [
                    (candidate_id, score)
                    for candidate_id, score, _inlier, _raw in row
                ]
                for row_id, row in acoustic_rows.items()
            },
            edges,
        )
        acoustic_info: dict[str, tuple[Optional[str], str, float, float]] = {}
        best_existing_raw: dict[str, float] = {}
        novelty: dict[str, float] = {}
        novelty_lineages: dict[str, str] = {}
        selected_bridges: dict[str, tuple[Optional[BridgeEvidence], float]] = {}
        for observation in values:
            row_id = observation.observation_id
            selected_id = acoustic_assignment[row_id]
            score_by_id = {
                candidate_id: (score, inlier, raw)
                for candidate_id, score, inlier, raw in acoustic_rows[row_id]
            }
            score, inlier, _raw = score_by_id[selected_id]
            best_existing_raw[row_id] = max(
                (
                    raw
                    for candidate_id, _score, _inlier, raw in acoustic_rows[row_id]
                    if candidate_id in self._components
                ),
                default=self.config.background_location,
            )
            if selected_id.startswith("NONE:"):
                component_id: Optional[str] = None
                kind = "none"
            elif selected_id in self._components:
                component_id = selected_id
                kind = "component"
            else:
                component_id = selected_id
                kind = "unexplained"
            acoustic_info[row_id] = (component_id, kind, float(score), float(inlier))
            contribution = self._novelty_contribution(
                observation.assessment,
                component_id,
            )
            novelty[row_id] = contribution
            novelty_lineages[row_id] = self._novelty_lineage(observation)
            bridge, bridge_contribution = self._select_bridge(
                observation,
                component_id,
            )
            selected_bridges[row_id] = (bridge, bridge_contribution)

        output_affinity: dict[str, dict[str, float]] = {}
        output_acoustic: dict[str, dict[str, _OutputAcousticEvidence]] = {}
        for observation in values:
            row_id = observation.observation_id
            component_id, _kind, _score, _inlier = acoustic_info[row_id]
            bridge, bridge_contribution = selected_bridges[row_id]
            affinity, acoustic = self._observation_output_affinity(
                observation,
                component_id,
                bridge,
                bridge_contribution,
            )
            output_affinity[row_id] = affinity
            output_acoustic[row_id] = acoustic

        observation_global, embedding_global_updates = self._assign_global_identities(
            values,
            acoustic_info,
            output_affinity,
            output_acoustic,
            selected_bridges,
        )
        projected_support = self._project_global_support(
            observation_global,
            selected_bridges,
        )
        current_support: dict[tuple[str, str], float] = {}
        for observation in values:
            row_id = observation.observation_id
            global_id = observation_global[row_id]
            if global_id is None:
                continue
            for output_label, value in output_affinity[row_id].items():
                key = (global_id, output_label)
                current_support[key] = max(current_support.get(key, 0.0), float(value))
        all_global_ids = set(self._global_identities)
        all_global_ids.update(
            global_id for global_id in observation_global.values() if global_id is not None
        )
        projected_display, display_diagnostics = self._project_display_mapping(
            sorted(all_global_ids),
            projected_support,
            current_support,
        )

        output_rows: dict[str, list[tuple[str, float]]] = {}
        new_utilities: dict[str, float] = {}
        provisional_rows: set[str] = set()
        for observation in values:
            row_id = observation.observation_id
            component_id, _kind, _score, _inlier = acoustic_info[row_id]
            global_id = observation_global[row_id]
            mapped_output = projected_display.get(global_id) if global_id is not None else None
            continuity_output_labels = {
                item.output_label
                for item in observation.bridge_evidence
                if not item.conflict
                and item.bridge_authority > 0.0
                and item.provenance
                in {"current_snapshot_overlap", "unique_current_snapshot"}
            }
            excluded_output_labels = (
                set(observation.partition_excluded_output_labels)
                - continuity_output_labels
                if self.config.local_partition_provisional_enabled
                else set()
            )
            if (
                component_id is None
                or mapped_output is not None
                or not observation.assessment.novelty_allowed
            ):
                new_utility = -1_000.0
            else:
                novelty_total = self._projected_novelty(
                    component_id,
                    novelty_lineages[row_id],
                    novelty[row_id],
                )
                new_utility = (
                    novelty_total
                    - self.config.new_output_authority_cost
                    + observation.assessment.novelty_precision
                    * max(
                        0.0,
                        self.config.background_location
                        - best_existing_raw[row_id],
                    )
                    - 1e-9
                )
            if not self._output:
                new_utility = max(new_utility, 1.0)
            new_utilities[row_id] = float(new_utility)
            if mapped_output is not None and mapped_output not in excluded_output_labels:
                row = [(mapped_output, float(output_affinity[row_id][mapped_output]))]
            elif self._output:
                row = [
                    (
                        output_label,
                        float(
                            value
                            - (
                                self.config.new_output_authority_cost
                                if output_label in self._output_to_global
                                and self._output_to_global[output_label] != global_id
                                else 0.0
                            )
                        ),
                    )
                    for output_label, value in sorted(output_affinity[row_id].items())
                    if output_label not in excluded_output_labels
                ]
            else:
                row = []
            provisional_eligible = bool(
                self.config.local_partition_provisional_enabled
                and excluded_output_labels
                and not row
                and not continuity_output_labels
                and not observation.assessment.profile_allowed
                and not observation.assessment.novelty_allowed
            )
            if provisional_eligible:
                row.append((f"PROVISIONAL:{row_id}", 0.0))
                provisional_rows.add(row_id)
            row.append((f"NEW:{row_id}", float(new_utility)))
            output_rows[row_id] = row
        output_assignment = self._constrained_assignment(row_ids, output_rows, edges)

        edge_rows = {item for edge in edges for item in edge}
        decisions: list[LegoIdentityDecision] = []
        for observation in values:
            row_id = observation.observation_id
            component_id, acoustic_kind, acoustic_score, inlier = acoustic_info[row_id]
            global_id = observation_global[row_id]
            selected_output = output_assignment[row_id]
            provisional_allocate = selected_output.startswith("PROVISIONAL:")
            acoustic_new_allocate = selected_output.startswith("NEW:")
            evidence_poor_provisional_allocate = bool(
                acoustic_new_allocate
                and global_id is None
                and component_id is None
                and not observation.assessment.profile_allowed
                and not observation.assessment.novelty_allowed
            )
            if (
                provisional_allocate or evidence_poor_provisional_allocate
            ) and global_id is None:
                global_id = self._observation_global_id(row_id)
            allocate = provisional_allocate or acoustic_new_allocate
            output_label = None if allocate else selected_output
            bridge, bridge_contribution = selected_bridges[row_id]
            current_output = (
                self._global_to_output.get(global_id)
                if global_id is not None
                else None
            )
            predicted_output = projected_display.get(global_id) if global_id is not None else None
            if global_id is None or predicted_output is None:
                binding_kind = "unmatched"
                binding_output = None
            elif current_output is None:
                binding_kind = "attach"
                binding_output = predicted_output
            elif current_output == predicted_output:
                binding_kind = "retain"
                binding_output = predicted_output
            else:
                binding_kind = "switch"
                binding_output = predicted_output
            utility, margin = display_diagnostics.get(
                (global_id, binding_output),
                (
                    self.config.display_inertia_utility
                    if binding_output is not None and binding_output == current_output
                    else 0.0,
                    0.0,
                ),
            )
            new_reason = (
                "local_partition_provisional"
                if provisional_allocate
                else "evidence_poor_provisional"
                if evidence_poor_provisional_allocate
                else
                "authority"
                if acoustic_new_allocate and new_utilities[row_id] > 0.0
                else "startup"
                if acoustic_new_allocate and not self._output
                else "structural_conflict"
                if acoustic_new_allocate
                else "not_selected"
            )
            allocation_kind = (
                "provisional_local_partition"
                if provisional_allocate
                else "provisional_evidence_poor"
                if evidence_poor_provisional_allocate
                else "acoustic_new"
                if acoustic_new_allocate
                else "none"
            )
            bridge_matches_output = bool(
                bridge is not None and bridge.output_label == output_label
            )
            best_effort = bool(
                output_label is not None
                and predicted_output != output_label
                and not bridge_matches_output
            )
            can_update = bool(
                component_id is not None
                and global_id is not None
                and observation.assessment.profile_allowed
                and observation.assessment.center is not None
                and (
                    bridge is None
                    or predicted_output is None
                    or bridge.output_label == predicted_output
                )
                and (
                    acoustic_new_allocate
                    or predicted_output is None
                    or (output_label is not None and predicted_output == output_label)
                )
            )
            update_component = component_id if can_update else None
            update_output = predicted_output if can_update else None
            update_weight = float(
                observation.assessment.profile_precision * inlier
                if can_update
                else 0.0
            )
            decisions.append(
                LegoIdentityDecision(
                    observation_id=row_id,
                    global_identity_id=global_id,
                    publication_action=PublicationAction(
                        output_label=output_label,
                        score=float(dict(output_rows[row_id])[selected_output]),
                        best_effort=best_effort,
                        reason=(
                            "provisional_local_partition"
                            if provisional_allocate
                            else "provisional_evidence_poor"
                            if evidence_poor_provisional_allocate
                            else "new_output"
                            if acoustic_new_allocate
                            else "matched_global_identity"
                            if predicted_output == output_label
                            else "typed_bridge"
                            if bridge_matches_output
                            else "best_effort"
                        ),
                    ),
                    acoustic_explanation=AcousticExplanation(
                        component_id=component_id,
                        kind=acoustic_kind,
                        score=acoustic_score,
                        inlier_responsibility=inlier,
                    ),
                    binding_action=BindingAction(
                        kind=binding_kind,
                        component_id=component_id,
                        output_label=binding_output,
                        utility=float(utility),
                        margin=float(margin),
                    ),
                    profile_update_action=ProfileUpdateAction(
                        component_id=update_component,
                        output_label=update_output,
                        weight=update_weight,
                        reason="authorized" if can_update else "publication_owner_mismatch",
                    ),
                    new_output_action=NewOutputLabelAction(
                        allocate=allocate,
                        component_id=(
                            component_id if acoustic_new_allocate else None
                        ),
                        utility=float(
                            dict(output_rows[row_id])[selected_output]
                            if allocate
                            else new_utilities[row_id]
                        ),
                        reason=new_reason,
                        kind=allocation_kind,
                    ),
                    resolved_bridge=bridge,
                    bridge_contribution=float(bridge_contribution),
                    novelty_lineage=novelty_lineages[row_id],
                    novelty_contribution=float(novelty[row_id]),
                    forced_constraint=bool(
                        row_id in edge_rows
                        or (
                            (
                                row_id in provisional_rows
                                or evidence_poor_provisional_allocate
                            )
                            and (
                                output_label is None
                                or acoustic_kind == "unexplained"
                            )
                        )
                    ),
                )
            )
        return LegoIdentityInference(
            observations=values,
            decisions=tuple(sorted(decisions, key=lambda item: item.observation_id)),
            cannot_link=edges,
            binding_updates=tuple(
                sorted(
                    (
                        component_id,
                        projected_display.get(global_id),
                    )
                    for component_id, global_id in {
                        **{
                            component_id: state.owner_global_id
                            for component_id, state in self._components.items()
                            if state.owner_global_id is not None
                        },
                        **dict(embedding_global_updates),
                    }.items()
                )
            ),
            embedding_global_updates=embedding_global_updates,
            display_updates=tuple(sorted(projected_display.items())),
        )

    @staticmethod
    def _bounded_lineages(
        values: Mapping[str, float],
        capacity: int,
    ) -> dict[str, float]:
        selected = sorted(
            ((str(key), float(value)) for key, value in values.items()),
            key=lambda item: (-item[1], item[0]),
        )[: int(capacity)]
        return {key: value for key, value in sorted(selected)}

    def _evict_for_component(self) -> bool:
        if len(self._components) < self.component_capacity:
            return True
        removable = [
            state
            for state in self._components.values()
            if state.owner_global_id is None
        ]
        if not removable:
            return False
        victim = min(
            removable,
            key=lambda state: (
                state.profile.effective_support
                + sum(state.novelty.values())
                + sum(summary.total for summary in state.bridges.values()),
                state.component_id,
            ),
        )
        self._components.pop(victim.component_id, None)
        return True

    def _ensure_component(self, component_id: str) -> Optional[EmbeddingComponentState]:
        state = self._components.get(component_id)
        if state is not None:
            return state
        if not self._evict_for_component():
            return None
        state = EmbeddingComponentState(
            component_id=component_id,
            profile=AcousticProfileState(
                profile_id=component_id,
                dimension=self.dimension,
                exemplar_capacity=self.exemplar_capacity,
            ),
        )
        self._components[component_id] = state
        return state

    def _prune_challengers(self, state: EmbeddingComponentState) -> None:
        owner = self.binding_for_component(state.component_id)
        challengers = [
            (output_label, summary)
            for output_label, summary in state.bridges.items()
            if output_label != owner
        ]
        keep = {
            output_label
            for output_label, _summary in sorted(
                challengers,
                key=lambda item: (-item[1].total, item[0]),
            )[: self.challenger_capacity]
        }
        if owner is not None and owner in state.bridges:
            keep.add(owner)
        state.bridges = {
            output_label: state.bridges[output_label]
            for output_label in sorted(keep)
        }

    def _recompute_display_mapping(self) -> None:
        totals = {
            global_id: {
                output_label: summary.total
                for output_label, summary in state.output_support.items()
            }
            for global_id, state in self._global_identities.items()
        }
        projected, _diagnostics = self._project_display_mapping(
            self.global_identity_ids,
            totals,
            {},
        )
        self._set_display_mapping(projected)

    def apply_update(
        self,
        inference: LegoIdentityInference,
        *,
        output_links: Optional[Mapping[str, str]] = None,
    ) -> tuple[dict[str, object], ...]:
        links = {str(key): str(value) for key, value in (output_links or {}).items()}
        observations = {item.observation_id: item for item in inference.observations}
        inferred_display = dict(inference.display_updates)
        events: list[dict[str, object]] = []
        active_decisions = [
            item
            for item in inference.decisions
            if item.observation_id not in self._applied_observations
        ]
        for decision in inference.decisions:
            if decision.observation_id in self._applied_observations:
                events.append(
                    {
                        "observation_id": decision.observation_id,
                        "status": "duplicate",
                        "component_id": None,
                        "output_label": None,
                    }
                )
        for decision in active_decisions:
            observation = observations[decision.observation_id]
            component_id = decision.acoustic_explanation.component_id
            global_id = decision.global_identity_id
            if global_id is not None and (
                decision.new_output_action.allocate
                or decision.profile_update_action.component_id is not None
                or decision.resolved_bridge is not None
                or decision.novelty_contribution > 0.0
            ):
                self._ensure_global_identity(global_id)
            state = None
            if component_id is not None and (
                decision.profile_update_action.component_id is not None
                or decision.resolved_bridge is not None
                or decision.novelty_contribution > 0.0
            ):
                state = self._ensure_component(component_id)
            if (
                state is not None
                and global_id is not None
                and (component_id, global_id) in inference.embedding_global_updates
            ):
                if not self._set_component_global(component_id, global_id):
                    raise RuntimeError("E->G ownership changed before atomic commit")
            if state is not None and decision.novelty_contribution > 0.0:
                key = decision.novelty_lineage
                state.novelty[key] = max(
                    state.novelty.get(key, 0.0),
                    decision.novelty_contribution,
                )
                state.novelty = self._bounded_lineages(
                    state.novelty,
                    self.config.lineage_capacity,
                )
            if state is not None and decision.resolved_bridge is not None:
                evidence = decision.resolved_bridge
                summary = state.bridges.setdefault(
                    evidence.output_label,
                    _BridgeSummaryState(),
                )
                summary.lineages[evidence.ancestor_lineage] = max(
                    summary.lineages.get(evidence.ancestor_lineage, 0.0),
                    decision.bridge_contribution,
                )
                summary.lineages = self._bounded_lineages(
                    summary.lineages,
                    self.config.lineage_capacity,
                )
                summary.conflict_count += int(evidence.conflict)
                self._prune_challengers(state)
                if global_id is not None:
                    self._add_global_output_support(
                        global_id,
                        evidence.output_label,
                        lineage=evidence.ancestor_lineage,
                        contribution=decision.bridge_contribution,
                        conflict=evidence.conflict,
                    )
            events.append(
                {
                    "observation_id": decision.observation_id,
                    "status": "applied",
                    "embedding_component_id": component_id,
                    "component_id": component_id,
                    "global_identity_id": global_id,
                    "published_speaker_id": decision.publication_action.output_label,
                    "output_label": decision.publication_action.output_label,
                    "bridge_output_label": (
                        decision.resolved_bridge.output_label
                        if decision.resolved_bridge is not None
                        else None
                    ),
                    "new_output": decision.new_output_action.allocate,
                    "output_allocation_kind": decision.new_output_action.kind,
                }
            )

        self._set_display_mapping(dict(inference.display_updates))

        for decision in active_decisions:
            if not decision.new_output_action.allocate:
                continue
            output_label = links.get(decision.observation_id)
            if output_label is None:
                raise RuntimeError("NEW output-label allocation requires an output link")
            self.ensure_output(output_label)
            global_id = decision.global_identity_id
            if global_id is None:
                raise RuntimeError("NEW output-label allocation requires a global identity")
            self._ensure_global_identity(global_id)
            if decision.new_output_action.kind in {
                "provisional_evidence_poor",
                "provisional_local_partition",
            }:
                self._global_identities[global_id].provisional_output_label = output_label
            else:
                self._add_global_output_support(
                    global_id,
                    output_label,
                    lineage=f"allocation:{decision.observation_id}",
                    contribution=1.0,
                )

        self._recompute_display_mapping()
        active_mapping = dict(self._global_to_output)
        used_output_labels = set(active_mapping.values())
        provisional_candidates: list[tuple[str, str]] = []
        for global_id, state in sorted(self._global_identities.items()):
            output_label = state.provisional_output_label
            if output_label is None:
                continue
            selected_now = any(
                decision.global_identity_id == global_id
                and decision.new_output_action.kind
                in {"provisional_evidence_poor", "provisional_local_partition"}
                for decision in active_decisions
            )
            supported_now = inferred_display.get(global_id) == output_label
            if selected_now or supported_now:
                provisional_candidates.append((global_id, output_label))
        for global_id, output_label in provisional_candidates:
            if global_id in active_mapping or output_label in used_output_labels:
                continue
            active_mapping[global_id] = output_label
            used_output_labels.add(output_label)
        self._set_display_mapping(active_mapping)

        for decision in active_decisions:
            observation = observations[decision.observation_id]
            component_id = decision.profile_update_action.component_id
            if component_id is None or decision.profile_update_action.weight <= 0.0:
                continue
            state = self._components.get(component_id)
            if state is None:
                continue
            if state.owner_global_id != decision.global_identity_id:
                raise RuntimeError("profile update G owner changed before atomic commit")
            state.profile.admit(
                observation.assessment,
                weight=decision.profile_update_action.weight,
            )

        self._applied_observations.update(
            item.observation_id for item in active_decisions
        )
        return tuple(events)


__all__ = [
    "AcousticProfileState",
    "AcousticComponentState",
    "EmbeddingComponentState",
    "GlobalIdentityState",
    "AcousticExplanation",
    "BindingAction",
    "BridgeSummarySnapshot",
    "ExemplarSnapshot",
    "NewOutputLabelAction",
    "ProfileUpdateAction",
    "ProfileSnapshot",
    "PublicationAction",
    "OutputHandleState",
    "LegoIdentityConfig",
    "LegoIdentityDecision",
    "LegoIdentityInference",
    "LegoIdentityModel",
    "LegoIdentityObservation",
]
