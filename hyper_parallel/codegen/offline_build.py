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
"""Device-free topology and model-build inputs for the Codegen CLI."""

from dataclasses import dataclass
from math import lcm
from typing import Any

from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext


@dataclass(frozen=True)
class PlanningTopology:
    """Pure-data mesh metadata consumed by the native sharding planner."""

    mesh_dim_names: tuple[str, ...]
    mesh_shape: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the logical shape using the DeviceMesh-compatible name."""
        return self.mesh_shape


def create_offline_distributed_setup(config: Any) -> DistributedSetup:
    """Project YAML parallel settings into a process-group-free setup.

    The logical dense domain is the smallest domain that satisfies the
    configured TP/CP, FSDP and virtual-EP divisibility contracts. It describes
    planning semantics only and never creates ranks, devices or process groups.

    Args:
        config: Resolved TrainerConfig.

    Returns:
        A DistributedSetup whose mesh is pure metadata.
    """
    accelerator = config.accelerator
    fsdp_config = config.fsdp_config
    tp_size = max(1, int(accelerator.tp_size))
    cp_size = max(1, int(accelerator.cp_size))
    pp_size = max(1, int(accelerator.pp_size))
    ep_size = max(1, int(accelerator.ep_size))
    dp_shard_size = max(1, int(fsdp_config.dp_shard_size))
    edp_shard_size = max(1, int(fsdp_config.edp_shard_size))

    dense_size = lcm(
        tp_size * cp_size,
        ep_size,
        tp_size * dp_shard_size,
        ep_size * edp_shard_size,
    )
    dp_size = dense_size // (tp_size * cp_size)
    fsdp_data_parallel_size = dp_size * cp_size
    dp_replicate_size = fsdp_data_parallel_size // dp_shard_size
    topology = PlanningTopology(
        mesh_dim_names=("dp", "cp", "tp"),
        mesh_shape=(dp_size, cp_size, tp_size),
    )
    mesh_context = MeshContext(
        device_mesh=topology,
        dp_size=dp_size,
        dp_replicate_size=dp_replicate_size,
        dp_shard_size=dp_shard_size,
        edp_shard_size=edp_shard_size,
        tp_size=tp_size,
        cp_size=cp_size,
        pp_size=pp_size,
        ep_size=ep_size,
        sequence_parallel=bool(accelerator.sequence_parallel),
        loss_parallel=bool(accelerator.loss_parallel),
    )
    return DistributedSetup(
        mesh_context=mesh_context,
        low_precision_config=getattr(config.training, "low_precision", None),
        fp32_main_params=bool(getattr(config.optimizer, "fp32_main_params", False)),
    )


__all__ = ["PlanningTopology", "create_offline_distributed_setup"]
