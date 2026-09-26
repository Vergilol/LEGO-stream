from __future__ import annotations

import math
from typing import Mapping, Sequence


def maximum_weight_assignment(
    rows: Sequence[str],
    columns: Sequence[str],
    scores: Mapping[tuple[str, str], float],
    *,
    minimum_score: float = 0.0,
) -> dict[str, str]:
    """Return the maximum-weight optional one-to-one assignment.

    Every row may remain unmatched. Missing, non-finite, or non-improving
    edges are excluded. The implementation is the rectangular Hungarian
    algorithm with one zero-cost dummy column per row, so runtime is
    polynomial and does not change semantics at an arbitrary speaker count.
    Equal-total assignments are resolved by the caller-provided row and column order;
    callers must therefore supply stable sequences when tie stability matters.
    """

    row_values = tuple(str(value) for value in rows)
    column_values = tuple(str(value) for value in columns)
    if len(set(row_values)) != len(row_values):
        raise ValueError("assignment rows must be unique")
    if len(set(column_values)) != len(column_values):
        raise ValueError("assignment columns must be unique")
    if not row_values or not column_values:
        return {}
    threshold = float(minimum_score)
    if not math.isfinite(threshold):
        raise ValueError("minimum_score must be finite")

    allowed: dict[tuple[int, int], float] = {}
    for row_index, row in enumerate(row_values):
        for column_index, column in enumerate(column_values):
            raw_score = scores.get((row, column))
            if raw_score is None:
                continue
            score = float(raw_score)
            if not math.isfinite(score) or score <= threshold:
                continue
            allowed[(row_index, column_index)] = score
    if not allowed:
        return {}

    row_count = len(row_values)
    real_column_count = len(column_values)
    total_column_count = real_column_count + row_count
    max_abs_score = max(abs(score) for score in allowed.values())
    score_scale = max_abs_score if max_abs_score > 0.0 else 1.0
    forbidden_cost = 2.0
    costs = [
        [forbidden_cost] * real_column_count + [0.0] * row_count
        for _ in range(row_count)
    ]
    for (row_index, column_index), score in allowed.items():
        costs[row_index][column_index] = -(score / score_scale)

    # Hungarian algorithm for a rectangular cost matrix with rows <= columns.
    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (total_column_count + 1)
    matched_row = [0] * (total_column_count + 1)
    predecessor = [0] * (total_column_count + 1)
    for row_number in range(1, row_count + 1):
        matched_row[0] = row_number
        minimum_reduced_cost = [math.inf] * (total_column_count + 1)
        used_column = [False] * (total_column_count + 1)
        current_column = 0
        while True:
            used_column[current_column] = True
            current_row = matched_row[current_column]
            delta = math.inf
            next_column = 0
            for column_number in range(1, total_column_count + 1):
                if used_column[column_number]:
                    continue
                reduced_cost = (
                    costs[current_row - 1][column_number - 1]
                    - row_potential[current_row]
                    - column_potential[column_number]
                )
                if reduced_cost < minimum_reduced_cost[column_number]:
                    minimum_reduced_cost[column_number] = reduced_cost
                    predecessor[column_number] = current_column
                if minimum_reduced_cost[column_number] < delta:
                    delta = minimum_reduced_cost[column_number]
                    next_column = column_number
            for column_number in range(total_column_count + 1):
                if used_column[column_number]:
                    row_potential[matched_row[column_number]] += delta
                    column_potential[column_number] -= delta
                else:
                    minimum_reduced_cost[column_number] -= delta
            current_column = next_column
            if matched_row[current_column] == 0:
                break
        while True:
            previous_column = predecessor[current_column]
            matched_row[current_column] = matched_row[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

    assignment: dict[str, str] = {}
    for column_number in range(1, real_column_count + 1):
        row_number = matched_row[column_number]
        if row_number == 0:
            continue
        row_index = row_number - 1
        column_index = column_number - 1
        if (row_index, column_index) not in allowed:
            continue
        assignment[row_values[row_index]] = column_values[column_index]
    return assignment


__all__ = ["maximum_weight_assignment"]
