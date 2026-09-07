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
"""Hyper EP routing and communication-free local expert kernels."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.model_executor.models.transformers.moe import TransformersFusedMoE
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE
from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl

from hyper_parallel import DeviceMesh
from hyper_parallel.auto_models.components.distributed.ep_utils import (
    ep_routed_forward,
    qwen3_moe_combine as qwen3_combine,
)
from hyper_parallel.auto_models.components.distributed.sharding_config import TEMPLATES
from hyper_parallel.core.dtensor.layout import Layout, infer_slice_area_by_layout


@dataclass(frozen=True)
class HyperExpertPlacement:
    """Contiguous expert ownership over vLLM's existing EP group."""

    local_count: int
    global_start: int = 0
    mesh: Optional[DeviceMesh] = None

    @classmethod
    def from_config(cls, vllm_config: VllmConfig, num_experts: int) -> "HyperExpertPlacement":
        """Reuse the worker group; never create a separate rollout worker pool."""
        if not bool(getattr(vllm_config.parallel_config, "enable_expert_parallel", False)):
            return cls(local_count=num_experts)
        if not vllm_config.model_config.enforce_eager:
            raise ValueError("Hyper EP ragged dispatch currently requires enforce_eager=true")
        group = get_ep_group()
        if group.world_size not in (1, 2, 4):
            raise ValueError("Hyper rollout currently supports EP1/EP2/EP4")
        if num_experts % group.world_size:
            raise ValueError(f"Expert count {num_experts} must be divisible by EP size {group.world_size}")
        mesh = DeviceMesh.from_group(
            group.device_group,
            device_type=vllm_config.device_config.device.type,
            mesh_dim_names=("ep",),
        )
        # Consume the same expert-axis placement as ShardingPlanner. The
        # runtime owns the group and physical storage, not a second shard rule.
        placement = TEMPLATES["moe_mlp"].moe_expert_placement
        if not placement.is_shard() or placement.dim != 0:
            raise ValueError("Hyper local experts require the public expert-axis Shard(0) contract")
        layout = Layout.from_device_mesh(mesh)([placement])
        layout.placement_to_tensor_map(1)
        start, end = infer_slice_area_by_layout(layout, group.rank_in_group, (num_experts,))[0]
        return cls(end - start, start, mesh)


def routed_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    *,
    router_fn: Callable,
    combine_fn: Optional[Callable] = None,
) -> torch.Tensor:
    """Run the same family router with either local or Hyper EP dispatch."""
    mesh = getattr(module, "hyper_ep_mesh", None)
    if mesh is None or mesh.size() == 1:
        if combine_fn is not None:
            indices, weights = router_fn(module, hidden_states)
            flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
            return module.experts(flattened, indices, weights).view_as(hidden_states)
        return local_routed_forward(module, hidden_states, router_fn=router_fn)
    return ep_routed_forward(
        module, hidden_states, router_fn=router_fn, ep_group=mesh.get_group(), combine_fn=combine_fn
    )


class HyperLocalFusedExperts(TransformersFusedMoE, AscendFusedMoE):
    """Store and execute one complete EP-local set of SwiGLU experts.

    The leaf deliberately fixes its internal TP/DP/PCP/EP degrees to one.
    Outer EP partitions the expert axis; the local leaf never partitions the
    intermediate dimension. Routing and branch composition stay outside it.
    """

    hyper_local_expert_leaf = True

    def __init__(
        self,
        *,
        local_expert_count: int,
        global_expert_start: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        """Build a communication-free Ascend FusedMoE expert leaf."""
        if local_expert_count <= 0:
            raise ValueError("local_expert_count must be positive")
        if global_expert_start < 0:
            raise ValueError("global_expert_start must be non-negative")
        target_device = get_current_vllm_config().device_config.device
        with torch.device(target_device):
            super().__init__(
                num_experts=local_expert_count,
                top_k=1,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                params_dtype=params_dtype,
                renormalize=False,
                quant_config=None,
                tp_size=1,
                dp_size=1,
                pcp_size=1,
                prefix=prefix,
                activation="silu",
                enable_eplb=False,
            )
        if self.use_ep or self.ep_size != 1 or self.tp_size != 1:
            raise RuntimeError(
                "Hyper local FusedMoE must keep internal TP/DP/PCP/EP at one"
            )

        def assigned_routing(
            hidden_states: torch.Tensor,
            gating_output: torch.Tensor,
            topk: int,
            renormalize: bool,
            **unused: object,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Return the family router assignments staged by this leaf."""
            del hidden_states, topk, renormalize, unused
            if self._topk_ids is None:
                raise RuntimeError("FusedMoE routing IDs were not staged")
            return gating_output, self._topk_ids

        # vLLM-Ascend passes additional routing metadata that the upstream
        # Transformers callback does not accept in the pinned version.
        self.custom_routing_function = assigned_routing
        self.local_expert_count = int(local_expert_count)
        self.global_expert_start = int(global_expert_start)
        self._hyper_hidden_size = int(hidden_size)
        self._hyper_intermediate_size = int(intermediate_size)
        self._local_comm_method = AllGatherCommImpl(self.moe_config)

    @property
    def global_expert_end(self) -> int:
        """Return the exclusive global expert bound owned by this leaf."""
        return self.global_expert_start + self.local_expert_count

    def forward(
        self,
        hidden_states: torch.Tensor,
        local_expert_indices: torch.Tensor,
        topk_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Execute HF Top-K input or one dispatched assignment per token row."""
        if hidden_states.ndim != 2:
            raise ValueError("FusedMoE dispatched hidden states must have shape [T,H]")
        if topk_weights is not None:
            if local_expert_indices.ndim != 2 or topk_weights.shape != local_expert_indices.shape:
                raise ValueError("FusedMoE Top-K indices and weights must have matching [T,K] shape")
            if local_expert_indices.shape[0] != hidden_states.shape[0]:
                raise ValueError("FusedMoE Top-K token count must match hidden states")
            experts_per_token = local_expert_indices.shape[1]
            dispatched = hidden_states.repeat_interleave(experts_per_token, dim=0)
            expert_output = self._forward_assignments(
                dispatched,
                local_expert_indices.reshape(-1),
            )
            weighted = (
                expert_output.float() * topk_weights.reshape(-1, 1).float()
            ).to(expert_output.dtype)
            return weighted.view(hidden_states.shape[0], experts_per_token, hidden_states.shape[-1]).sum(dim=1)
        return self._forward_assignments(hidden_states, local_expert_indices)

    def _forward_assignments(
        self,
        hidden_states: torch.Tensor,
        local_expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run exactly one local expert for each dispatched token row."""
        indices = local_expert_indices.reshape(-1)
        if indices.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "FusedMoE expert assignments must match dispatched token rows"
            )
        if hidden_states.shape[0] == 0:
            return torch.empty_like(hidden_states)
        self.ensure_physical_weight_layout()
        self._topk_ids = indices.reshape(-1, 1).to(torch.int32)
        topk_weights = torch.ones(
            self._topk_ids.shape,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        self.ensure_moe_quant_config_init()
        # The outer Hyper dispatcher already exchanged assignments. Ascend's
        # process-wide context otherwise selects another EP dispatch strategy.
        previous = (
            _EXTRA_CTX.moe_comm_method, _EXTRA_CTX.moe_comm_type,
            _EXTRA_CTX.flash_comm_v1_enabled, _EXTRA_CTX.in_profile_run,
        )
        try:
            _EXTRA_CTX.moe_comm_method = self._local_comm_method
            _EXTRA_CTX.moe_comm_type = MoECommType.ALLGATHER
            _EXTRA_CTX.flash_comm_v1_enabled = False
            _EXTRA_CTX.in_profile_run = False
            return AscendFusedMoE.forward_impl(self, hidden_states, topk_weights)
        finally:
            (
                _EXTRA_CTX.moe_comm_method, _EXTRA_CTX.moe_comm_type,
                _EXTRA_CTX.flash_comm_v1_enabled, _EXTRA_CTX.in_profile_run,
            ) = previous

    def _local_expert_slice(self, loaded_weight: torch.Tensor) -> torch.Tensor:
        if loaded_weight.ndim != 3:
            raise ValueError(
                "Packed expert weights must be three-dimensional: "
                f"shape={tuple(loaded_weight.shape)}"
            )
        if loaded_weight.shape[0] < self.global_expert_end:
            raise ValueError(
                "Packed expert weights do not cover the local expert range: "
                f"shape={tuple(loaded_weight.shape)}, "
                f"range=[{self.global_expert_start}:{self.global_expert_end}]"
            )
        return loaded_weight[self.global_expert_start:self.global_expert_end]

    def _copy_packed_weight(
        self,
        loaded_weight: torch.Tensor,
        *,
        parameter_name: str,
        canonical_tail: tuple[int, int],
        physical_tail: tuple[int, int],
    ) -> None:
        """Copy canonical or physical packed weights into the current layout."""
        local_weight = self._local_expert_slice(loaded_weight)
        parameter = getattr(self, parameter_name)
        source_shape = tuple(local_weight.shape)
        canonical_shape = (self.local_expert_count, *canonical_tail)
        physical_shape = (self.local_expert_count, *physical_tail)
        target_shape = tuple(parameter.shape)
        if source_shape not in (canonical_shape, physical_shape):
            raise ValueError(
                f"{parameter_name} has unsupported source shape {source_shape}; "
                f"expected {canonical_shape} or {physical_shape}"
            )
        if target_shape not in (canonical_shape, physical_shape):
            raise ValueError(
                f"{parameter_name} has unsupported target shape {target_shape}; "
                f"expected {canonical_shape} or {physical_shape}"
            )
        if source_shape != target_shape:
            local_weight = local_weight.transpose(-2, -1).contiguous()
        with torch.no_grad():
            parameter.copy_(local_weight.to(device=parameter.device, dtype=parameter.dtype))

    def load_gate_up(self, loaded_weight: torch.Tensor) -> None:
        """Load packed canonical/physical gate-up weights into ``w13``."""
        self._copy_packed_weight(
            loaded_weight,
            parameter_name="w13_weight",
            canonical_tail=(2 * self._hyper_intermediate_size, self._hyper_hidden_size),
            physical_tail=(self._hyper_hidden_size, 2 * self._hyper_intermediate_size),
        )

    def load_down(self, loaded_weight: torch.Tensor) -> None:
        """Load packed canonical/physical down weights into ``w2``."""
        self._copy_packed_weight(
            loaded_weight,
            parameter_name="w2_weight",
            canonical_tail=(self._hyper_hidden_size, self._hyper_intermediate_size),
            physical_tail=(self._hyper_intermediate_size, self._hyper_hidden_size),
        )

    def load_expert_projection(
        self,
        loaded_weight: torch.Tensor,
        projection: str,
        global_expert: int,
    ) -> bool:
        """Load one official per-expert projection when this leaf owns it."""
        local_expert = int(global_expert) - self.global_expert_start
        if not 0 <= local_expert < self.local_expert_count:
            return False
        if tuple(self.w13_weight.shape[1:]) != (
            2 * self._hyper_intermediate_size,
            self._hyper_hidden_size,
        ) or tuple(self.w2_weight.shape[1:]) != (
            self._hyper_hidden_size,
            self._hyper_intermediate_size,
        ):
            raise RuntimeError(
                "Per-expert checkpoint loading requires canonical pre-process FusedMoE layout"
            )
        if projection == "gate_proj":
            parameter = self.w13_weight[local_expert, :self._hyper_intermediate_size]
            expected_shape = (self._hyper_intermediate_size, self._hyper_hidden_size)
        elif projection == "up_proj":
            parameter = self.w13_weight[local_expert, self._hyper_intermediate_size:]
            expected_shape = (self._hyper_intermediate_size, self._hyper_hidden_size)
        elif projection == "down_proj":
            parameter = self.w2_weight[local_expert]
            expected_shape = (self._hyper_hidden_size, self._hyper_intermediate_size)
        else:
            raise ValueError(f"Unsupported expert projection {projection!r}")
        if tuple(loaded_weight.shape) != expected_shape:
            raise ValueError(
                f"{projection} has shape {tuple(loaded_weight.shape)}, "
                f"expected {expected_shape}"
            )
        with torch.no_grad():
            parameter.copy_(
                loaded_weight.to(device=parameter.device, dtype=parameter.dtype)
            )
        return True

    def ensure_physical_weight_layout(self) -> None:
        """Restore Ascend grouped-matmul layouts after load or sleep/wake."""
        physical_shapes = (
            (
                self.local_expert_count,
                self._hyper_hidden_size,
                2 * self._hyper_intermediate_size,
            ),
            (
                self.local_expert_count,
                self._hyper_intermediate_size,
                self._hyper_hidden_size,
            ),
        )
        canonical_shapes = (
            (
                self.local_expert_count,
                2 * self._hyper_intermediate_size,
                self._hyper_hidden_size,
            ),
            (
                self.local_expert_count,
                self._hyper_hidden_size,
                self._hyper_intermediate_size,
            ),
        )
        actual_shapes = (
            tuple(self.w13_weight.shape),
            tuple(self.w2_weight.shape),
        )
        if actual_shapes == physical_shapes:
            return
        if actual_shapes != canonical_shapes:
            raise ValueError(
                "FusedMoE weights have unsupported or mixed runtime layouts: "
                f"actual={actual_shapes}, physical={physical_shapes}, "
                f"canonical={canonical_shapes}"
            )
        process_weights = getattr(
            getattr(self, "quant_method", None),
            "process_weights_after_loading",
            None,
        )
        if not callable(process_weights):
            raise RuntimeError("FusedMoE quant method cannot restore physical weights")
        # The Ascend lifecycle also restores device-private NZ storage; a
        # transpose alone has the right shape but produces incorrect GEMM data.
        process_weights(self)
        restored_shapes = (tuple(self.w13_weight.shape), tuple(self.w2_weight.shape))
        if restored_shapes != physical_shapes:
            raise RuntimeError(
                "FusedMoE post-load lifecycle did not restore physical layouts: "
                f"actual={restored_shapes}, expected={physical_shapes}"
            )


def local_routed_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    *,
    router_fn: Callable[[nn.Module, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Execute EP1 routing without invoking an all-to-all collective."""
    if hidden_states.ndim != 3:
        raise ValueError("EP1 routed hidden states must have shape [B,S,H]")
    experts = getattr(module, "experts", None)
    if not isinstance(experts, HyperLocalFusedExperts):
        raise TypeError("EP1 routed module must use HyperLocalFusedExperts")
    if experts.global_expert_start != 0:
        raise ValueError("EP1 requires the local leaf to own experts starting at zero")
    topk_indices, topk_weights = router_fn(module, hidden_states)
    if topk_indices.ndim != 2 or topk_weights.shape != topk_indices.shape:
        raise ValueError("Router must return matching [T,K] indices and weights")
    flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
    if topk_indices.shape[0] != flattened.shape[0]:
        raise ValueError("Router token count differs from the hidden-state token count")
    experts_per_token = topk_indices.shape[1]
    dispatched = flattened.repeat_interleave(experts_per_token, dim=0)
    expert_output = experts(dispatched, topk_indices.reshape(-1))
    weighted = expert_output * topk_weights.reshape(-1, 1).to(expert_output.dtype)
    return weighted.view(*hidden_states.shape[:2], experts_per_token, hidden_states.shape[-1]).sum(dim=2)


__all__ = ["HyperExpertPlacement", "HyperLocalFusedExperts", "local_routed_forward", "routed_forward", "qwen3_combine"]
