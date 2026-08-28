# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""CPU contracts for colocated and disjoint rollout DP x TP mapping."""

from dataclasses import fields

import pytest

from rl.roles.rollout.topology import VLLMRolloutTopology, resolve_vllm_rollout_topology


def test_rollout_topology_exposes_only_shared_deployment_fields() -> None:
    """Rank-local topology and replica identifiers are absent from the runtime contract."""
    field_names = {field.name for field in fields(VLLMRolloutTopology)}

    assert "data_parallel_size" in field_names
    assert {"runtime_topology", "replica_id", "replica_count"}.isdisjoint(field_names)


@pytest.mark.parametrize(
    ("local_rank", "owner"),
    ((0, True), (1, False)),
)
def test_disjoint_fsd2_rollout_dp2_tp2(
    local_rank: int,
    owner: bool,
) -> None:
    """All Trainer ranks resolve one shared disjoint DP2 x TP2 deployment."""
    topology = resolve_vllm_rollout_topology(
        {
            "deployment": "disjoint",
            "data_parallel_size": 2,
            "tensor_parallel_size": 2,
            "visible_devices": "2,3,4,5",
            "port": 8200,
        },
        {"LOCAL_RANK": str(local_rank), "LOCAL_WORLD_SIZE": "2"},
    )

    assert topology.data_parallel_size == 2
    assert topology.engine_count == 2
    assert topology.server_owner is owner
    assert topology.visible_devices == ("2", "3", "4", "5")
    assert topology.port == 8200


@pytest.mark.parametrize(
    ("local_rank", "owner"),
    ((0, True), (1, False)),
)
def test_colocated_internal_dp1_tp2_uses_one_shared_deployment(
    local_rank: int,
    owner: bool,
) -> None:
    """Two trainer ranks share one explicitly internal-DP TP2 server."""
    topology = resolve_vllm_rollout_topology(
        {
            "deployment": "colocated",
            "data_parallel_size": 1,
            "tensor_parallel_size": 2,
            "port": 8200,
        },
        {
            "LOCAL_RANK": str(local_rank),
            "LOCAL_WORLD_SIZE": "2",
            "ASCEND_RT_VISIBLE_DEVICES": "0,1",
        },
    )

    assert topology.data_parallel_size == 1
    assert topology.engine_count == 1
    assert topology.server_owner is owner
    assert topology.visible_devices == ("0", "1")
    assert topology.port == 8200


@pytest.mark.parametrize(
    ("local_rank", "owner"),
    ((0, True), (1, False)),
)
def test_internal_dp2_tp1_uses_one_shared_owner_and_endpoint(
    local_rank: int,
    owner: bool,
) -> None:
    """Qwen3 DP2 ranks share one endpoint while only rank zero owns it."""
    topology = resolve_vllm_rollout_topology(
        {
            "deployment": "colocated",
            "data_parallel_size": 2,
            "tensor_parallel_size": 1,
            "port": 8100,
        },
        {
            "LOCAL_RANK": str(local_rank),
            "LOCAL_WORLD_SIZE": "2",
            "ASCEND_RT_VISIBLE_DEVICES": "4,5",
        },
    )

    assert topology.data_parallel_size == 2
    assert topology.engine_count == 2
    assert topology.server_owner is owner
    assert topology.visible_devices == ("4", "5")
    assert topology.port == 8100


@pytest.mark.parametrize(
    ("local_rank", "owner"),
    ((0, True), (1, False), (2, False), (3, False)),
)
def test_fsdp4_internal_dp2_tp2_uses_one_shared_deployment(
    local_rank: int,
    owner: bool,
) -> None:
    """Four trainers share one DP2 x TP2 deployment without rank-local servers."""
    topology = resolve_vllm_rollout_topology(
        {
            "deployment": "colocated",
            "data_parallel_size": 2,
            "tensor_parallel_size": 2,
            "port": 8500,
        },
        {
            "LOCAL_RANK": str(local_rank),
            "LOCAL_WORLD_SIZE": "4",
            "ASCEND_RT_VISIBLE_DEVICES": "4,5,6,7",
        },
    )

    assert topology.trainer_world_size == 4
    assert topology.data_parallel_size == 2
    assert topology.engine_count == 2
    assert topology.server_owner is owner
    assert topology.visible_devices == ("4", "5", "6", "7")
    assert topology.port == 8500


def test_disjoint_requires_dp_times_tp_devices() -> None:
    """Disjoint rollout card count is rollout DP multiplied by rollout TP."""
    with pytest.raises(ValueError, match="expected=4"):
        resolve_vllm_rollout_topology(
            {
                "deployment": "disjoint",
                "data_parallel_size": 2,
                "tensor_parallel_size": 2,
                "visible_devices": "2,3",
                "port": 8200,
            },
            {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "2"},
        )


@pytest.mark.parametrize(
    ("local_rank", "owner"),
    ((0, True), (3, False)),
)
def test_disjoint_rollout_dp_is_independent_of_trainer_world(
    local_rank: int,
    owner: bool,
) -> None:
    """Trainer FSDP4 ranks all resolve the full external DP1 x TP2 set."""
    topology = resolve_vllm_rollout_topology(
        {
            "deployment": "disjoint",
            "data_parallel_size": 1,
            "tensor_parallel_size": 2,
            "visible_devices": "6,7",
            "port": 8300,
        },
        {"LOCAL_RANK": str(local_rank), "LOCAL_WORLD_SIZE": "4"},
    )

    assert topology.trainer_world_size == 4
    assert topology.data_parallel_size == 1
    assert topology.engine_count == 1
    assert topology.visible_devices == ("6", "7")
    assert topology.port == 8300
    assert topology.server_owner is owner


@pytest.mark.parametrize("deployment", ["colocated", "disjoint"])
def test_rollout_topology_rejects_removed_topology_option(deployment: str) -> None:
    """Runtime resolution rejects old topology values before interpreting them."""
    with pytest.raises(ValueError, match="rollout.vllm.topology was removed"):
        resolve_vllm_rollout_topology(
            {
                "deployment": deployment,
                "topology": "internal_dp",
                "data_parallel_size": 1,
                "tensor_parallel_size": 1,
                "visible_devices": "1",
                "port": 8200,
            },
            {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1"},
        )


def test_colocated_rejects_explicit_visible_devices() -> None:
    """Colocated devices come only from the Trainer environment."""
    with pytest.raises(ValueError, match="remove rollout.vllm.visible_devices"):
        resolve_vllm_rollout_topology(
            {
                "deployment": "colocated",
                "data_parallel_size": 1,
                "tensor_parallel_size": 1,
                "visible_devices": None,
                "port": 8200,
            },
            {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1"},
        )
