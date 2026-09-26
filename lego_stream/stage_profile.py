from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping


_WALL_STAGE_FIELDS = (
    "prompt_sec",
    "observer_sec",
    "decode_wall_sec",
    "map_sec",
    "commit_sec",
    "after_update_sec",
    "unattributed_sec",
)
_DIAGNOSTIC_FIELDS = ("decode_reported_sec",)
_STAGE_FIELDS = (
    *_WALL_STAGE_FIELDS,
    *_DIAGNOSTIC_FIELDS,
    "total_wall_sec",
)


def _without_sec_suffix(field: str) -> str:
    """Keep the profiler compatible with the project's Python 3.8 runtime."""

    return field[:-4] if field.endswith("_sec") else field


class StageProfileObserver:
    """Accumulate bounded, observer-only streaming stage timings."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path).expanduser().resolve()
        self.tick_count = 0
        self._totals = {field: 0.0 for field in _STAGE_FIELDS}

    def __call__(self, value: Mapping[str, float]) -> None:
        unknown = set(value) - {"tick", *_STAGE_FIELDS}
        if unknown:
            raise ValueError(
                "unknown stage profile fields: " + ", ".join(sorted(unknown))
            )
        tick = self._nonnegative(value.get("tick", 0.0), "tick")
        del tick
        for field in _STAGE_FIELDS:
            self._totals[field] += self._nonnegative(value.get(field, 0.0), field)
        self.tick_count += 1

    def finalize(self) -> dict[str, object]:
        total = self._totals["total_wall_sec"]
        totals = {
            _without_sec_suffix(field): float(value)
            for field, value in self._totals.items()
            if field in {*_WALL_STAGE_FIELDS, "total_wall_sec"}
        }
        diagnostics = {
            _without_sec_suffix(field): float(self._totals[field])
            for field in _DIAGNOSTIC_FIELDS
        }
        summary: dict[str, object] = {
            "schema_version": "lego-stream.stage-profile.v1",
            "observer_only": True,
            "tick_count": int(self.tick_count),
            "totals_sec": totals,
            "diagnostics_sec": diagnostics,
            "shares": {
                _without_sec_suffix(field): (
                    float(self._totals[field]) / total if total > 0.0 else 0.0
                )
                for field in _WALL_STAGE_FIELDS
            },
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary

    @staticmethod
    def _nonnegative(value: float, name: str) -> float:
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return number


__all__ = ["StageProfileObserver"]
