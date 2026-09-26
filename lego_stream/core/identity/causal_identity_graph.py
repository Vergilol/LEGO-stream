"""Small, immutable topology primitives for the causal identity transaction.

The active resolver remains the owner of scores, state mutation, and
publication.  This module only gives the E/G/O transaction one auditable
boundary for event topology and constrained assignment.  In particular, it
does not infer a speaker, create a profile, or grant authority to an
observation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class CausalGraphEvent:
    """Canonical immutable view of one event's observation graph.

    Observation identifiers and cannot-link edges are sorted so that the
    graph is independent of parser/dictionary iteration order.  The event is
    deliberately topology-only: acoustic scores and mutable resolver state
    never enter this object.
    """

    observation_ids: tuple[str, ...]
    cannot_link: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        raw_ids = tuple(str(value).strip() for value in self.observation_ids)
        if not raw_ids or any(not value for value in raw_ids):
            raise ValueError("event must contain at least one observation")
        if len(set(raw_ids)) != len(raw_ids):
            raise ValueError("event observation ids must be unique")
        ids = tuple(sorted(raw_ids))
        known = set(ids)
        normalized_edges: set[tuple[str, str]] = set()
        for raw_edge in self.cannot_link:
            if len(raw_edge) != 2:
                raise ValueError("cannot-link edge must join two event observations")
            left, right = (str(raw_edge[0]).strip(), str(raw_edge[1]).strip())
            if (
                not left
                or not right
                or left == right
                or left not in known
                or right not in known
            ):
                raise ValueError("cannot-link edge must join two event observations")
            normalized_edges.add(tuple(sorted((left, right))))
        object.__setattr__(self, "observation_ids", ids)
        object.__setattr__(self, "cannot_link", tuple(sorted(normalized_edges)))

    @classmethod
    def create(
        cls,
        observation_ids: Iterable[str],
        cannot_link: Iterable[tuple[str, str]] = (),
    ) -> "CausalGraphEvent":
        """Build a canonical event without retaining caller-owned containers."""

        return cls(tuple(observation_ids), tuple(cannot_link))

    def authorities_for(
        self,
        *,
        routing_allowed: bool,
        profile_allowed: bool,
        novelty_allowed: bool,
    ) -> tuple[str, ...]:
        """Return capabilities already authorized by evidence.

        Capability precision is monotone (route -> profile -> NEW).  The
        graph facade can therefore never turn a route-only observation into a
        profile or NEW authority.  It only reports the supplied permissions;
        it does not grant any additional one.
        """

        route = bool(routing_allowed)
        profile = route and bool(profile_allowed)
        novelty = profile and bool(novelty_allowed)
        return tuple(
            name
            for name, enabled in (
                ("routing", route),
                ("profile", profile),
                ("new", novelty),
            )
            if enabled
        )


@dataclass(frozen=True)
class ConstrainedAssignment:
    """Immutable result of the exact optional constrained assignment."""

    pairs: tuple[tuple[str, str], ...]
    total_score: float

    @property
    def mapping(self) -> dict[str, str]:
        """Return a defensive copy for compatibility with existing callers."""

        return dict(self.pairs)


def canonical_cannot_link_edges(
    observation_ids: Iterable[str],
    cannot_link: Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    """Normalize and validate event-local cannot-link edges."""

    return CausalGraphEvent.create(observation_ids, cannot_link).cannot_link


def constrained_max_weight_assignment(
    row_ids: Sequence[str],
    candidates: Mapping[str, Sequence[Any]],
    edges: Sequence[tuple[str, str]],
    *,
    candidate_id: Optional[Callable[[Any], str]] = None,
    candidate_score: Optional[Callable[[Any], float]] = None,
    no_feasible_message: str = "event has no feasible constrained identity assignment",
) -> ConstrainedAssignment:
    """Solve the bounded event assignment with the frozen C7 tie-break.

    This is the same branch-and-bound transaction previously embedded in the
    two identity implementations.  A row receives exactly one supplied
    candidate; callers decide whether an explicit ``NEW``/``UNEXPLAINED``
    candidate represents unmatched.  The helper is generic only over the
    candidate record shape, so the arithmetic and candidate universe remain
    owned by the active resolver.
    """

    id_fn: Callable[[Any], str] = candidate_id or (lambda item: str(item[0]))
    score_fn: Callable[[Any], float] = candidate_score or (
        lambda item: float(item[1])
    )
    event = CausalGraphEvent.create(row_ids, edges)
    ordered_rows = event.observation_ids

    neighbors: dict[str, set[str]] = {row_id: set() for row_id in ordered_rows}
    for left, right in event.cannot_link:
        neighbors[left].add(right)
        neighbors[right].add(left)
    ordered = tuple(
        sorted(ordered_rows, key=lambda item: (-len(neighbors[item]), item))
    )
    normalized_candidates: dict[str, tuple[tuple[str, float, Any], ...]] = {}
    for row_id in ordered:
        row = tuple(
            (str(id_fn(item)), float(score_fn(item)), item)
            for item in candidates[row_id]
        )
        normalized_candidates[row_id] = row
    maxima = {
        row_id: max(score for _candidate, score, _item in normalized_candidates[row_id])
        for row_id in ordered
    }
    suffix: list[float] = [0.0] * (len(ordered) + 1)
    for index in range(len(ordered) - 1, -1, -1):
        suffix[index] = suffix[index + 1] + maxima[ordered[index]]

    best_assignment: Optional[dict[str, str]] = None
    best_score = -math.inf
    assigned: dict[str, str] = {}

    def search(index: int, running: float) -> None:
        nonlocal best_assignment, best_score
        if running + suffix[index] < best_score - 1e-12:
            return
        if index == len(ordered):
            canonical = tuple(sorted(assigned.items()))
            current_best = tuple(sorted((best_assignment or {}).items()))
            if running > best_score + 1e-12 or (
                abs(running - best_score) <= 1e-12 and canonical < current_best
            ):
                best_score = float(running)
                best_assignment = dict(assigned)
            return
        row_id = ordered[index]
        for candidate, score, _item in sorted(
            normalized_candidates[row_id],
            key=lambda item: (-item[1], item[0]),
        ):
            if any(assigned.get(other) == candidate for other in neighbors[row_id]):
                continue
            assigned[row_id] = candidate
            search(index + 1, running + score)
            assigned.pop(row_id, None)

    search(0, 0.0)
    if best_assignment is None:
        raise RuntimeError(no_feasible_message)
    return ConstrainedAssignment(
        pairs=tuple(sorted(best_assignment.items())),
        total_score=float(best_score),
    )


__all__ = [
    "CausalGraphEvent",
    "ConstrainedAssignment",
    "canonical_cannot_link_edges",
    "constrained_max_weight_assignment",
]
