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
"""CPU contracts for multi-rank disjoint Trainer and rollout topology."""

import pytest
from rl.config import _validate_disjoint_vllm


def test_disjoint_topology_accepts_rollout_dp_independent_of_trainer_world(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trainer DP4 may use one external DP1 x TP2 deployment."""
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")

    _validate_disjoint_vllm(
        {"data_parallel_size": 1, "visible_devices": "4,5", "port": 8200},
        {"dp_shard": 4},
        rollout_tp=2,
    )


def test_disjoint_topology_rejects_extra_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollout devices must exactly cover the configured DP x TP deployment."""
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2")

    with pytest.raises(ValueError, match=r"data_parallel_size \* tensor_parallel_size"):
        _validate_disjoint_vllm(
            {"data_parallel_size": 1, "visible_devices": "1,2", "port": 8200},
            {"dp_shard": 1},
            rollout_tp=1,
        )


@pytest.mark.parametrize(
    ("rollout_devices", "message"),
    [
        ("2", "data_parallel_size \\* tensor_parallel_size"),
        ("2,2", "must be unique"),
        ("1,2", "must use NPUs disjoint"),
    ],
)
def test_disjoint_topology_rejects_invalid_rollout_ownership(
    monkeypatch: pytest.MonkeyPatch,
    rollout_devices: str,
    message: str,
) -> None:
    """Rollout DP replicas must be complete, unique, and disjoint from Trainer."""
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")

    with pytest.raises(ValueError, match=message):
        _validate_disjoint_vllm(
            {"data_parallel_size": 2, "visible_devices": rollout_devices, "port": 8200},
            {"dp_shard": 2},
            rollout_tp=1,
        )


def test_disjoint_topology_does_not_require_rollout_dp_to_match_local_world(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollout DP is not inferred from Trainer local world size."""
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")

    _validate_disjoint_vllm(
        {"data_parallel_size": 1, "visible_devices": "4", "port": 8200},
        {"dp_shard": 4},
        rollout_tp=1,
    )


def test_disjoint_topology_accepts_tensor_parallel_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shared disjoint deployment may contain DP2 x TP2 workers."""
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")

    _validate_disjoint_vllm(
        {"data_parallel_size": 2, "visible_devices": "2,3,4,5", "port": 8200},
        {"dp_shard": 8},
        rollout_tp=2,
    )


def test_disjoint_topology_requires_one_explicit_shared_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disjoint shared endpoint cannot derive rank-local ports."""
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")

    with pytest.raises(ValueError, match="explicit integer between 1 and 65535"):
        _validate_disjoint_vllm(
            {"data_parallel_size": 1, "visible_devices": "1"},
            {"dp_shard": 1},
            rollout_tp=1,
        )
