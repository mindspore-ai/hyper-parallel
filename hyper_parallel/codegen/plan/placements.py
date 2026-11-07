# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Convert planner placements to and from source-emission data."""

from __future__ import annotations

import re
from typing import Any

from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Replicate,
    Shard,
    StridedShard,
)


def placement_to_string(placement: Any) -> str:
    """Serialize one placement to its canonical source representation."""
    if isinstance(placement, StridedShard):
        if placement.uneven_shard:
            raise TypeError("cannot observe StridedShard(uneven_shard=True)")
        return f"SS({placement.dim},{placement.split_factor})"
    if isinstance(placement, Shard):
        if placement.uneven_shard:
            raise TypeError("cannot observe Shard(uneven_shard=True)")
        return f"S({placement.dim})"
    if isinstance(placement, Replicate):
        return "R"
    if isinstance(placement, Partial):
        return f"P({placement.reduce_op})"
    raise TypeError(
        f"cannot serialize placement {placement!r} of unknown type "
        f"{type(placement).__name__}"
    )


def named_placement_to_dict(named: Any) -> dict[str, Any]:
    """Serialize a nested named-placement contract to plain dictionaries."""
    if not named:
        return {}
    return {
        str(getattr(name, "value", name)): _value_to_data(value)
        for name, value in named.items()
    }


def parse_placement(text: str) -> Any:
    """Parse a canonical placement string."""
    if not isinstance(text, str) or not text:
        raise ValueError(f"cannot parse placement from {text!r}")
    if text == "R":
        return Replicate()
    match = re.fullmatch(r"([SP]|SS)\((.*)\)", text)
    if match is None:
        raise ValueError(
            f"unknown placement form {text!r} (expected S(dim) / R / "
            "P(reduce_op) / SS(dim,factor))"
        )
    kind, arguments = match.groups()
    if kind == "P":
        if not arguments:
            raise ValueError(f"malformed partial placement {text!r}: missing reduce_op")
        return Partial(arguments)
    parts = [part.strip() for part in arguments.split(",")]
    try:
        dim = int(parts[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"malformed shard placement {text!r}: invalid dim") from exc
    if kind == "S":
        if len(parts) != 1:
            raise ValueError(f"malformed shard placement {text!r}: too many args")
        return Shard(dim)
    if len(parts) != 2:
        raise ValueError(f"malformed strided shard placement {text!r}: expected SS(dim,factor)")
    try:
        split_factor = int(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"malformed strided shard placement {text!r}: invalid split_factor"
        ) from exc
    return StridedShard(dim, split_factor)


def parse_named_placement(data: dict[str, Any]) -> Any:
    """Parse a nested named-placement contract from plain dictionaries."""
    if not data:
        return {}
    return _value_from_data(data)


def value_to_data(value: Any) -> Any:
    """Project a planner value to deterministic source-emission data."""
    return _value_to_data(value)


def _value_to_data(value: Any) -> Any:
    """Recursively project placements and planner values into plain data."""
    if isinstance(value, (Partial, Replicate, Shard, StridedShard)):
        return placement_to_string(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {
            str(getattr(key, "value", key)): _value_to_data(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_value_to_data(item) for item in value]
    if callable(value):
        return getattr(value, "__name__", str(value))
    return value


def _value_from_data(value: Any) -> Any:
    """Recursively reconstruct placements from nested plain-data contracts."""
    if isinstance(value, str):
        return parse_placement(value)
    if isinstance(value, dict):
        return {
            str(getattr(key, "value", key)): _value_from_data(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_value_from_data(item) for item in value]
    return value


__all__ = [
    "named_placement_to_dict",
    "parse_named_placement",
    "parse_placement",
    "placement_to_string",
    "value_to_data",
]
