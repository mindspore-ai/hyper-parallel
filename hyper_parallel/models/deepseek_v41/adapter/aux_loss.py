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
"""V4.1 modality-specific load accounting using the common router primitives."""

from __future__ import annotations

from typing import Any

import torch

from hyper_parallel.components.losses.moe_aux_loss import sequence_load_balancing_loss
from hyper_parallel.components.modules.moe.moe_utils import (
    accumulate_router_load as accumulate_expert_load,
    update_router_bias,
    validate_router_layout,
)
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.distributed.module_binding import ParallelBinding, ParallelModuleSpec


def configure_router_statistics(gate: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any) -> tuple[Any, ...]:
    """Configure bias-update groups and return the router's token-partition groups.

    Args:
        gate: Router owning the per-step load counters.
        mesh: Job mesh or a compute region's submesh.
        tp_mesh: TP token-partition mesh, if present.
        cp_mesh: CP token-partition mesh, if present.
    """
    sequence_groups = tuple(
        axis.get_group() for axis in (cp_mesh, tp_mesh)
        if axis is not None and axis.size() > 1
    )
    # A local compute region can receive a TP submesh. Bias counts cover the
    # entire data batch, so recover its DP axes from the job mesh.
    statistics_mesh = mesh
    if isinstance(mesh, DeviceMesh) and mesh.root_mesh is not None:
        statistics_mesh = mesh.root_mesh
    mesh_axes = (statistics_mesh.mesh_dim_names or ()) if statistics_mesh is not None else ()
    dp_groups = tuple(
        statistics_mesh[name].get_group() for name in mesh_axes
        if name in ("dp", "dp_replicate", "dp_shard") and statistics_mesh[name].size() > 1
    )
    gate.expert_bias_update_groups = dp_groups + sequence_groups
    return sequence_groups


class DeepseekV41RouterStatistics:
    """Bind model-owned routing statistics for native MoE with the expert family."""

    router_tp_mesh: Any = None
    sequence_partition_groups: tuple[Any, ...] = ()

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Declare replicated router parameters and borrowed statistics meshes."""
        return ParallelModuleSpec(
            protocol="deepseek_v41.router.statistics", binding_kind="mesh",
            stages={"forward": ("hidden_states",)}, validate=validate_router_layout,
        )

    def bind_parallel(self, binding: ParallelBinding) -> None:
        """Bind the token partitions used by the routed MoE and the job's DP groups.

        Args:
            binding: The existing model build's mesh resources.
        """
        layout = binding.mesh_binding.mesh_context
        mesh = layout.device_mesh
        names = tuple(mesh.mesh_dim_names or ()) if mesh is not None else ()
        tp_partitioned = layout.sequence_parallel or layout.ep_size > 1
        self.router_tp_mesh = mesh["tp"] if "tp" in names and tp_partitioned else None
        cp_mesh = mesh["cp"] if "cp" in names else None
        self.sequence_partition_groups = configure_router_statistics(self, mesh, self.router_tp_mesh, cp_mesh)


def accumulate_router_load(
        weights: torch.Tensor,
        selected_experts: torch.Tensor,
        image_mask: torch.Tensor | None,
        token_mask: torch.Tensor | None,
        accumulator: torch.Tensor,
) -> torch.Tensor:
    """Keep text/image counters independent while reusing backward load accumulation.

    Args:
        weights: Routing weights [T, K] used by expert combination.
        selected_experts: Actual dispatch IDs [T, K].
        image_mask: Boolean token metadata; true tokens are images.
        token_mask: Optional boolean metadata excluding padding.
        accumulator: Integer step counters [2, E], text then image.

    Returns:
        Numerically unchanged weights recording each modality's loads in backward.
    """
    if image_mask is not None:
        # Separate modality bins without adding another backward hook or collective.
        selected_experts = selected_experts.long() + image_mask.reshape(-1, 1).long() * accumulator.shape[-1]
    return accumulate_expert_load(weights, selected_experts, token_mask, accumulator.view(-1))


def update_modality_bias(
        bias: torch.Tensor,
        counts: torch.Tensor,
        update_rate: float,
        *,
        vision_bias: torch.Tensor | None = None,
) -> None:
    """Update V4.1 text/image biases using independent expert-load means.

    Args:
        bias: Text correction bias [E], possibly sharded.
        counts: Synchronized int64 loads [2, E], text then image.
        update_rate: Nonnegative sign-update step size.
        vision_bias: Optional image correction bias [E], possibly sharded.
    """
    update_router_bias(bias, counts[0], update_rate)
    if vision_bias is not None:
        update_router_bias(vision_bias, counts[1], update_rate)


__all__ = ["sequence_load_balancing_loss", "accumulate_router_load", "update_modality_bias"]
