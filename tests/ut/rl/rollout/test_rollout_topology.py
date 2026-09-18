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


@pytest.mark.parametrize("field", ["data_parallel_size", "tensor_parallel_size"])
@pytest.mark.parametrize("value", [None, True, 0, -1, "2", 1.5])
def test_rollout_topology_rejects_invalid_parallel_degrees(field: str, value: object) -> None:
    """DP and TP must be explicit positive integers, never booleans or coercible text."""
    config = {"deployment": "colocated", "data_parallel_size": 2, "tensor_parallel_size": 2, "port": 8200}
    config[field] = value
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        resolve_vllm_rollout_topology(config, {})


@pytest.mark.parametrize("devices", ["", "0,,1", "0,x", "0,-1", "0,0", "00,0"])
@pytest.mark.parametrize("deployment", ["colocated", "disjoint"])
def test_rollout_topology_rejects_invalid_physical_devices(devices: str, deployment: str) -> None:
    """Device aliases, duplicates and empty slots must fail before server startup."""
    config = {"deployment": deployment, "data_parallel_size": 2, "tensor_parallel_size": 1, "port": 8200}
    environment = {"LOCAL_WORLD_SIZE": "2", "ASCEND_RT_VISIBLE_DEVICES": devices}
    if deployment == "disjoint":
        config["visible_devices"] = devices
    with pytest.raises(ValueError, match="NPU device IDs"):
        resolve_vllm_rollout_topology(config, environment)


@pytest.mark.parametrize("port", [None, True, "8200", 0, -1, 65536])
def test_rollout_topology_rejects_invalid_ports(port: object) -> None:
    """The shared endpoint requires an explicit valid TCP port."""
    config = {"deployment": "colocated", "data_parallel_size": 2, "tensor_parallel_size": 1, "port": port}
    with pytest.raises(ValueError, match="explicit integer between"):
        resolve_vllm_rollout_topology(config, {})


@pytest.mark.parametrize(
    ("changes", "environment", "error"),
    [
        ({"topology": "shared"}, {}, "topology was removed"),
        ({"deployment": "remote"}, {}, "Unsupported rollout deployment"),
        ({"host": "0.0.0.0"}, {}, "loopback"),
        ({}, {"LOCAL_RANK": "2"}, "Invalid local trainer topology"),
        ({}, {"LOCAL_RANK": "-1"}, "Invalid local trainer topology"),
        ({}, {"LOCAL_WORLD_SIZE": "0"}, "Invalid local trainer topology"),
        ({}, {"LOCAL_WORLD_SIZE": "4"}, "must match the Trainer world"),
        ({}, {"ASCEND_RT_VISIBLE_DEVICES": "0"}, "one visible NPU per trainer rank"),
        ({"visible_devices": "0,1"}, {}, "derives its physical NPUs"),
        ({"deployment": "disjoint", "visible_devices": "4"}, {}, "DP x TP devices"),
    ],
)
def test_rollout_topology_rejects_inconsistent_endpoint_or_world(
    changes: dict, environment: dict, error: str,
) -> None:
    """Invalid ownership and topology combinations cannot start a shared service."""
    config = {"deployment": "colocated", "data_parallel_size": 2, "tensor_parallel_size": 1, "port": 8200}
    config.update(changes)
    with pytest.raises(ValueError, match=error):
        resolve_vllm_rollout_topology(config, environment)


def test_rollout_topology_preserves_noncontiguous_device_order() -> None:
    """Logical ranks follow the caller's physical order after numeric normalization."""
    config = {"deployment": "colocated", "data_parallel_size": 2, "tensor_parallel_size": 1, "port": 65535}
    topology = resolve_vllm_rollout_topology(config, {"ASCEND_RT_VISIBLE_DEVICES": " 07, 3 "})
    assert topology.visible_devices == ("7", "3"), (
        f"Expected physical device order=('7', '3'), got={topology.visible_devices}"
    )
