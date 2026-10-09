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
"""Complete episode grouping shared by training, evaluation and reporting."""

from typing import Any, Sequence

_CALL_FIELDS = ("episode_id", "call_index", "call_count")


def _call_identity(metadata: dict[str, Any]) -> str:
    """Validate the explicit identity and ordinal of one segmented model call."""
    identity = metadata.get("episode_id")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("Segmented trajectories require a non-empty episode_id")
    for field in ("call_index", "call_count"):
        value = metadata.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Episode {identity} requires integer {field}")
    if not 0 <= metadata["call_index"] < metadata["call_count"]:
        raise ValueError(f"Episode {identity} has an invalid call index/count")
    return identity


def _validate_episode(trajectories: Sequence[Any], indices: list[int], identity: str) -> None:
    """Require complete calls with one shared reward and policy identity."""
    rows = [trajectories[index] for index in indices]
    first = rows[0]
    if any(row.metadata["call_count"] != len(rows) for row in rows) or sorted(
        row.metadata["call_index"] for row in rows
    ) != list(range(len(rows))):
        raise ValueError(f"Incomplete or duplicated calls in episode {identity}")
    for field in ("prompt_id", "group_id", "policy_version", "worker_policy_version", "reward", "reward_components"):
        if any(getattr(row, field) != getattr(first, field) for row in rows[1:]):
            raise ValueError(f"Inconsistent {field} in episode {identity}")
    indices.sort(key=lambda index: trajectories[index].metadata["call_index"])


def episode_rows(trajectories: Sequence[Any]) -> list[list[int]]:
    """Return ordered row indices per complete episode, excluding DP padding.

    Legacy trajectories each represent one episode. Segmented rows must declare
    all call fields and agree on prompt, GRPO group, policy and reward identity.
    """
    groups: dict[tuple[str, Any], list[int]] = {}
    for index, row in enumerate(trajectories):
        metadata = getattr(row, "metadata", {})
        padding = metadata.get("dp_padding", False)
        if not isinstance(padding, bool):
            raise ValueError("dp_padding must be a boolean")
        if padding:
            continue
        segmented = any(field in metadata for field in _CALL_FIELDS)
        key = ("calls", _call_identity(metadata)) if segmented else ("row", index)
        groups.setdefault(key, []).append(index)
    for (kind, identity), indices in groups.items():
        if kind == "calls":
            _validate_episode(trajectories, indices, identity)
    return list(groups.values())
