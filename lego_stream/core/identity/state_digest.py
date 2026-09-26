"""Canonical hashing and defensive copying for bounded identity state.

The global identity state ``G`` is the only mutable part of the system, so two
properties have to be checkable rather than assumed:

* a caller that receives a view of ``G`` cannot mutate it (``defensive_snapshot``
  returns owned, read-only arrays);
* two runs that claim to have produced the same state can be compared without
  trusting float formatting (``authoritative_digest`` canonicalizes first, then
  hashes).

Both helpers are on the production path -- ``core.identity.speakers`` uses them to
snapshot profiles before an assignment is solved and to digest state after it
is committed. They are deliberately free of any dependency on the optional
audit observer that first introduced them.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Mapping

import numpy as np


def immutable_float_array(value: object) -> np.ndarray:
    """Return an owned, finite, float32, read-only array."""

    array = np.asarray(value, dtype=np.float32).copy()
    if not np.all(np.isfinite(array)):
        raise ValueError("identity state vectors must be finite")
    array.setflags(write=False)
    return array


def _array_identity(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return {
        "dtype": "float32",
        "shape": [int(item) for item in array.shape],
        "sha256": digest.hexdigest(),
    }


def _canonical_primitive(value: object) -> object:
    if isinstance(value, np.ndarray):
        return {"__array__": _array_identity(value)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_primitive(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_primitive(item) for item in value]
    if isinstance(value, set):
        return sorted((_canonical_primitive(item) for item in value), key=repr)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("identity state cannot contain non-finite floats")
        return float(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported state primitive: {type(value).__name__}")


def authoritative_digest(value: object) -> str:
    """Return a stable ``sha256:``-prefixed digest of nested state."""

    payload = json.dumps(
        _canonical_primitive(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def defensive_snapshot(value: object) -> object:
    """Copy primitives recursively and make every array owned/read-only."""

    if isinstance(value, np.ndarray):
        return immutable_float_array(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): defensive_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(defensive_snapshot(item) for item in value)
    if isinstance(value, list):
        return [defensive_snapshot(item) for item in value]
    if isinstance(value, set):
        return tuple(sorted((defensive_snapshot(item) for item in value), key=repr))
    if value is None or isinstance(value, (str, int, float, bool)):
        return deepcopy(value)
    raise TypeError(
        f"received mutable/unsupported value: {type(value).__name__}"
    )


__all__ = [
    "authoritative_digest",
    "defensive_snapshot",
    "immutable_float_array",
]
