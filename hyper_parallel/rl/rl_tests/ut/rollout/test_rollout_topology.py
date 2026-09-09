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
"""CPU unit test for shared colocated and disjoint rollout topology."""

import pytest

from rl.roles.rollout.topology import resolve_vllm_rollout_topology


@pytest.mark.parametrize(
    ("deployment", "local_rank", "world_size", "visible_devices", "owner"),
    [
        ("colocated", 0, 4, "0,1,2,3", True),
        ("colocated", 3, 4, "0,1,2,3", False),
        ("disjoint", 0, 2, "4,5,6,7", True),
        ("disjoint", 1, 2, "4,5,6,7", False),
    ],
)
def test_rollout_topology_resolves_shared_endpoint_and_owners(
    deployment: str,
    local_rank: int,
    world_size: int,
    visible_devices: str,
    owner: bool,
) -> None:
    """Every Trainer rank resolves one DP2xTP2 endpoint owned only by rank zero."""
    config = {
        "deployment": deployment,
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "port": 8200,
    }
    environment = {
        "LOCAL_RANK": str(local_rank),
        "LOCAL_WORLD_SIZE": str(world_size),
        "ASCEND_RT_VISIBLE_DEVICES": visible_devices,
    }
    if deployment == "disjoint":
        config["visible_devices"] = visible_devices

    topology = resolve_vllm_rollout_topology(config, environment)

    assert topology.deployment == deployment
    assert topology.data_parallel_size == 2
    assert topology.tensor_parallel_size == 2
    assert topology.engine_count == 2
    assert topology.visible_devices == tuple(visible_devices.split(","))
    assert topology.visible_devices_csv == visible_devices
    assert topology.host == "127.0.0.1"
    assert topology.port == 8200
    assert topology.server_owner is owner
