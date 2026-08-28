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
"""Pure single-node runtime, DP, and TP topology mapping for vLLM rollout."""

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class VLLMRolloutTopology:
    """Describe one trainer rank's view of a shared rollout deployment."""

    deployment: str
    data_parallel_size: int
    tensor_parallel_size: int
    trainer_rank: int
    trainer_world_size: int
    engine_count: int
    server_owner: bool
    visible_devices: tuple[str, ...]
    host: str
    port: int

    @property
    def visible_devices_csv(self) -> str:
        """Return the server-local NPU list accepted by vLLM."""
        return ",".join(self.visible_devices)


def _device_ids(value: object, field: str) -> tuple[str, ...]:
    """Parse and validate one comma-separated physical-device list."""
    devices = tuple(device.strip() for device in str(value or "").split(","))
    if not devices or not all(devices):
        raise ValueError(f"{field} must contain non-empty NPU device IDs")
    if not all(device.isdigit() for device in devices):
        raise ValueError(f"{field} must contain numeric NPU device IDs, got {devices}")
    normalized_devices = tuple(str(int(device)) for device in devices)
    if len(set(normalized_devices)) != len(normalized_devices):
        raise ValueError(f"{field} must contain unique NPU device IDs, got {devices}")
    return normalized_devices


def resolve_vllm_rollout_topology(
    config: Mapping[str, object],
    environment: Mapping[str, str],
) -> VLLMRolloutTopology:
    """Resolve devices, ownership, and endpoint for one local trainer rank."""
    if "topology" in config:
        raise ValueError(
            "rollout.vllm.topology was removed; configure deployment, "
            "data_parallel_size, and tensor_parallel_size instead"
        )
    deployment = str(config.get("deployment", "disjoint"))
    if deployment not in ("colocated", "disjoint"):
        raise ValueError(f"Unsupported rollout deployment {deployment!r}")
    data_parallel_size = config.get("data_parallel_size")
    if (
        not isinstance(data_parallel_size, int)
        or isinstance(data_parallel_size, bool)
        or data_parallel_size <= 0
    ):
        raise ValueError("rollout.vllm.data_parallel_size must be a positive integer")
    tensor_parallel_size = config.get("tensor_parallel_size")
    if (
        not isinstance(tensor_parallel_size, int)
        or isinstance(tensor_parallel_size, bool)
        or tensor_parallel_size <= 0
    ):
        raise ValueError("rollout.vllm.tensor_parallel_size must be a positive integer")
    rollout_device_count = data_parallel_size * tensor_parallel_size
    default_trainer_world_size = rollout_device_count if deployment == "colocated" else 1
    trainer_rank = int(environment.get("LOCAL_RANK", "0"))
    trainer_world_size = int(
        environment.get("LOCAL_WORLD_SIZE", str(default_trainer_world_size))
    )
    if trainer_world_size <= 0 or not 0 <= trainer_rank < trainer_world_size:
        raise ValueError(
            "Invalid local trainer topology: "
            f"rank={trainer_rank}, world_size={trainer_world_size}"
        )

    if deployment == "colocated":
        if "visible_devices" in config:
            raise ValueError(
                "Colocated rollout derives its physical NPUs from the Trainer; "
                "remove rollout.vllm.visible_devices"
            )
        if rollout_device_count != trainer_world_size:
            raise ValueError(
                "Colocated rollout devices must match the Trainer world: "
                f"dp={data_parallel_size}, tp={tensor_parallel_size}, "
                f"world_size={trainer_world_size}"
            )
        training_devices = _device_ids(
            environment.get(
                "ASCEND_RT_VISIBLE_DEVICES",
                ",".join(str(rank) for rank in range(trainer_world_size)),
            ),
            "ASCEND_RT_VISIBLE_DEVICES",
        )
        if len(training_devices) != trainer_world_size:
            raise ValueError(
                "Colocated rollout requires one visible NPU per trainer rank: "
                f"world_size={trainer_world_size}, devices={training_devices}"
            )
        visible_devices = training_devices
    else:
        rollout_devices = _device_ids(
            config.get("visible_devices"),
            "rollout.vllm.visible_devices",
        )
        if len(rollout_devices) != rollout_device_count:
            raise ValueError(
                "Disjoint rollout requires DP x TP devices: "
                f"expected={rollout_device_count}, got={rollout_devices}"
            )
        visible_devices = rollout_devices
    engine_count = data_parallel_size
    server_owner = trainer_rank == 0
    host = str(config.get("host", "127.0.0.1"))
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("The vLLM server must bind to loopback")
    configured_port = config.get("port")
    if (
        not isinstance(configured_port, int)
        or isinstance(configured_port, bool)
        or not 0 < configured_port <= 65535
    ):
        raise ValueError(
            "Shared rollout requires rollout.vllm.port to be an explicit integer between 1 and 65535"
        )
    return VLLMRolloutTopology(
        deployment=deployment,
        data_parallel_size=data_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        trainer_rank=trainer_rank,
        trainer_world_size=trainer_world_size,
        engine_count=engine_count,
        server_owner=server_owner,
        visible_devices=visible_devices,
        host=host,
        port=configured_port,
    )


__all__ = ["VLLMRolloutTopology", "resolve_vllm_rollout_topology"]
