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
"""Opt-in K3 LatentMoE EP components with reversible expert storage."""
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from collections.abc import Mapping
import math
import os
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.distributed.expert_parallel.experts import (
    ep_routed_dispatch, ep_routed_experts_and_combine, ep_routed_forward,
)
from hyper_parallel.distributed.expert_parallel.planned import ep_planned_forward
from hyper_parallel.distributed.recipe_spec import local_compute
from hyper_parallel.models.kimi_k3.adapter.conversion.checkpoint_mapping import (
    StackExpertWeights, UnstackExpertWeights, build_expert_weight_converters,
)
from hyper_parallel.models.kimi_k3.adapter.packed_experts import grouped_expert_slots
from hyper_parallel.components.functional.activation_situ import situ_packed
from hyper_parallel.models.replacement import module_replacement


class KimiK3EPExperts(nn.Module):
    """Complete SiTU expert matrices, sharded only along the expert axis."""

    def __init__(self, module: nn.Module) -> None:
        """Convert the source expert list before optimizer construction."""
        super().__init__()
        self.num_experts = module.num_experts
        self.local_expert_count = self.num_experts
        self.initializer_range = module.config.initializer_range
        self.beta = module.config.activation_situ_beta
        self.linear_beta = module.config.activation_situ_linear_beta
        self.situ_backend = module.config.situ_backend
        self.backend = module.config.moe_backend
        self.permute_backend = "torch"
        self._hp_reset_after_materialization = True
        for name in ("w1", "w2", "w3"):
            parameters = [getattr(expert, name).weight for expert in module.experts]
            if len(parameters) != self.num_experts or any(
                    parameter.requires_grad != parameters[0].requires_grad for parameter in parameters):
                raise ValueError("EP requires a complete expert list with uniform gradient ownership")
            self.register_parameter(name, nn.Parameter(torch.stack([value.detach() for value in parameters]),
                                                       requires_grad=parameters[0].requires_grad))
        self.train(module.experts.training)

    def reset_parameters(self) -> None:
        """Initialize materialized expert shards with K3's native normal distribution."""
        for parameter in (self.w1, self.w2, self.w3):
            nn.init.normal_(parameter, std=self.initializer_range)

    def forward(self, hidden_states: torch.Tensor, local_expert_indices: torch.Tensor) -> torch.Tensor:
        """Evaluate assigned slots, including empty experts, with exact K3 SiTU.

        Args:
            hidden_states: Dispatched slots [slots, expert_width] in latent or full
                hidden space, before applying routing weights.
            local_expert_indices: Local expert index for each dispatched slot.

        Returns:
            Unweighted expert outputs with the input shape and slot ordering.
        """
        weights = tuple(value.to_local() if isinstance(value, DTensor) else value
                        for value in (self.w1, self.w2, self.w3))
        if any(weight.ndim != 3 or weight.shape[0] != self.local_expert_count for weight in weights):
            raise ValueError("EP expert parameters must contain exactly the local complete expert matrices")
        if self.backend == "grouped":
            return grouped_expert_slots(hidden_states, local_expert_indices, weights,
                                        beta=self.beta, linear_beta=self.linear_beta,
                                        situ_backend=self.situ_backend, permute_backend=self.permute_backend)
        output = torch.zeros_like(hidden_states)
        for expert_index in range(self.local_expert_count):
            slots = torch.where(local_expert_indices == expert_index)[0]
            selected = hidden_states[slots].to(weights[0].dtype)
            gate = F.linear(selected, weights[0][expert_index])  # pylint: disable=not-callable
            up = F.linear(selected, weights[2][expert_index])  # pylint: disable=not-callable
            # Match the native expert's paired activation layout as well as its formula.
            activated = situ_packed(torch.cat((gate, up), dim=-1), beta=self.beta,
                                    linear_beta=self.linear_beta, backend=self.situ_backend)
            values = F.linear(activated, weights[1][expert_index])  # pylint: disable=not-callable
            output = output.index_add(0, slots, values.to(output.dtype))
        return output


def _validate_ep_backends(ep_backend: str, ep_payload_dtype: str,
                          expert_permute_backend: str, ep_combine_backend: str) -> None:
    """Validate execution choices independently of their combination constraints."""
    if ep_backend not in ("standard", "planned") or ep_payload_dtype not in ("float32", "bfloat16"):
        raise ValueError("K3 EP requires standard/planned execution and float32/bfloat16 payload")
    if expert_permute_backend not in ("torch", "npu"):
        raise ValueError("K3 expert permutation must be torch or npu")
    if ep_combine_backend not in ("index_add", "npu"):
        raise ValueError("K3 EP combination must be index_add or npu")


class KimiK3EPMoeBlock(nn.Module):
    """Preserve dense branch identities while exposing checkpoint-aware EP experts."""

    def __init__(self, module: nn.Module, *, overlap_shared_expert: bool = False,
                 ep_backend: str = "standard", ep_payload_dtype: str = "float32",
                 replay_routed: bool = False, expert_permute_backend: str = "torch",
                 ep_combine_backend: str = "index_add") -> None:
        """Keep dense branches unchanged and optionally overlap their shared expert."""
        super().__init__()
        if not isinstance(overlap_shared_expert, bool):
            raise TypeError("overlap_shared_expert must be a bool")
        _validate_dispatch_options()
        _validate_ep_backends(ep_backend, ep_payload_dtype, expert_permute_backend, ep_combine_backend)
        if not isinstance(replay_routed, bool):
            raise TypeError("replay_routed must be a bool")
        if ep_backend == "standard" and (replay_routed or ep_payload_dtype != "float32"
                                          or ep_combine_backend != "index_add"):
            raise ValueError("K3 routed replay, BF16 payload and native combine require ep_backend='planned'")
        if ep_backend == "planned" and overlap_shared_expert:
            raise ValueError("K3 planned EP does not yet support shared-expert overlap")
        self.ep_backend = ep_backend
        self.ep_payload_dtype = getattr(torch, ep_payload_dtype)
        self.replay_routed = replay_routed
        self.ep_combine_backend = ep_combine_backend
        self.overlap_shared_expert = overlap_shared_expert
        for name in ("config", "num_experts", "top_k", "use_latent_moe", "moe_hidden_size", "latent_moe_use_norm"):
            setattr(self, name, getattr(module, name))
        for name, child in module.named_children():
            self.add_module(name, KimiK3EPExperts(module) if name == "experts" else child)
        self.experts.permute_backend = expert_permute_backend
        self.train(module.training)

    def make_transforms(self) -> list[WeightConverter]:
        """Declare exact reversible per-expert FQN mappings for HF load/save."""
        return build_expert_weight_converters(self.num_experts)

    def forward(self, hidden_states: torch.Tensor, valid_token_mask: torch.Tensor | None = None,
                collect_router_stats: bool = True) -> torch.Tensor:
        """Provide an EP1 reference with the same model and optimizer parameter keys.

        Args:
            hidden_states: Decoder activations [batch, sequence, hidden_size].
            valid_token_mask: Optional token mask excluding padding from QB counts.
            collect_router_stats: Accumulate QB observations for this forward.

        Returns:
            Combined routed and shared expert outputs in the original decoder shape.

        Raises:
            RuntimeError: If experts are sharded and the bound EP compute function
                has not replaced this local reference entry point.
        """
        if self.experts.local_expert_count != self.num_experts:
            raise RuntimeError("Sharded K3 experts require kimi_k3_ep_compute_fn on their MoE boundary")
        return _latent_ep_forward(self, hidden_states, None, valid_token_mask, collect_router_stats)


@module_replacement
def replace_kimi_k3_ep_moe(*, module: nn.Module, module_fqn: str,
                          context: Mapping[str, Any], overlap_shared_expert: bool = False,
                          ep_backend: str = "standard", ep_payload_dtype: str = "float32",
                          replay_routed: bool = False, expert_permute_backend: str = "torch",
                          ep_combine_backend: str = "index_add") -> nn.Module:
    """Install checkpoint-aware expert storage through an explicit replacement rule.

    Args:
        module: Original K3 MoE block with a complete per-expert ModuleList.
        module_fqn: Replacement framework's module path; not needed by conversion.
        context: Replacement framework context; no additional state is consumed.
        overlap_shared_expert: Schedule the shared branch during routed dispatch
            when the bound EP execution path supports it.
        ep_backend: Existing standard dispatch or immutable planned execution.
        ep_payload_dtype: FP32 transport by default; planned execution also accepts
            BF16 payload while retaining FP32 weights and token combination.
        replay_routed: Recompute routed experts and token exchanges from one
            per-forward plan; router, QB and shared branch are not replayed.
        expert_permute_backend: Native sorting or native NPU expert permutation.
        ep_combine_backend: FP32 index-add or native NPU unpermutation for planned EP.

    Returns:
        MoE block with stacked expert weights and the original dense branches.
    """
    del module_fqn, context
    if not isinstance(getattr(module, "experts", None), nn.ModuleList):
        raise ValueError("K3 EP replacement requires the original per-expert ModuleList")
    if module.config.moe_backend not in ("eager", "grouped"):
        raise ValueError("K3 EP requires moe_backend='eager' or 'grouped'")
    return KimiK3EPMoeBlock(module, overlap_shared_expert=overlap_shared_expert,
                          ep_backend=ep_backend, ep_payload_dtype=ep_payload_dtype,
                          replay_routed=replay_routed, expert_permute_backend=expert_permute_backend,
                          ep_combine_backend=ep_combine_backend)


def _validate_dispatch_options() -> None:
    """Reject shared dispatch switches that change K3 learned-routing semantics."""
    if os.environ.get("HP_EP_STATIC_SPLITS", "0") == "1":
        raise ValueError("K3 learned routing cannot reuse HP_EP_STATIC_SPLITS; token counts change every forward")
    capacity = os.environ.get("HP_EP_CAPACITY_FACTOR", "").strip()
    if capacity:
        try:
            factor = float(capacity)
        except ValueError as error:
            raise ValueError("K3 HP_EP_CAPACITY_FACTOR must be empty or nonpositive (no token dropping)") from error
        if not math.isfinite(factor) or factor > 0:
            raise ValueError("K3 EP requires lossless routing; positive HP_EP_CAPACITY_FACTOR drops expert slots")


def configure_kimi_k3_router_groups(gate: nn.Module, mesh: Any, cp_mesh: Any) -> tuple[Any, ...]:
    """Pool each distinct DP and CP token shard once, excluding overlapping EP.

    Args:
        gate: K3 router receiving expert_bias_update_groups for its QB callback.
        mesh: Active plan submesh; its root supplies DP token-ownership axes.
        cp_mesh: Chronological context mesh, or None without context parallelism.

    Returns:
        Nontrivial orthogonal DP/CP process groups assigned to the router.
    """
    # Local compute receives the active plan submesh, which can omit DP.
    # Resolve its public parent mesh before assigning a callback override.
    token_mesh = mesh
    while getattr(token_mesh, "root_mesh", None) is not None:
        token_mesh = token_mesh.root_mesh
    names = tuple(getattr(token_mesh, "mesh_dim_names", ()) or ())
    dp_names = ("dp",) if "dp" in names else tuple(name for name in ("dp_replicate", "dp_shard") if name in names)
    axes = [token_mesh[name] for name in dp_names]
    if cp_mesh is not None:
        axes.append(cp_mesh)
    groups = tuple(axis.get_group() for axis in axes if axis.size() > 1)
    if len({id(group) for group in groups}) != len(groups):
        raise ValueError("K3 router statistics require distinct orthogonal DP and CP groups")
    gate.expert_bias_update_groups = groups
    return groups


def _latent_ep_forward(module: nn.Module, hidden_states: torch.Tensor, ep_group: Any,
                       valid_token_mask: torch.Tensor | None, collect_router_stats: bool) -> torch.Tensor:
    """Route full features, exchange latent tokens, then combine shared and routed experts."""
    indices, weights = module.gate(hidden_states, valid_token_mask, collect_router_stats)
    flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
    latent = module.routed_expert_down_proj(flattened) if module.use_latent_moe else flattened
    latent = latent.reshape(*hidden_states.shape[:-1], latent.shape[-1])

    def router_fn(target: nn.Module, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reuse routing computed from full hidden features before latent projection.

        Args:
            target: Dispatcher-provided MoE module, unused because routing is cached
                in this forward's closure.
            states: Dispatched latent features, unused for full-width routing.

        Returns:
            The current forward's [tokens, top_k] expert indices and mixture weights.
        """
        del target, states
        return indices, weights

    # The shared dispatcher casts routing weights to its transport dtype. K3
    # requires FP32 weights and accumulation, so this reference path exchanges
    # FP32 slots; experts still compute in their parameter/activation dtype.
    shared = None
    if module.ep_backend == "planned":
        combined = ep_planned_forward(module, latent, indices, weights, ep_group,
                                      payload_dtype=module.ep_payload_dtype, replay_routed=module.replay_routed,
                                      combine_backend=module.ep_combine_backend)
    elif ep_group is None:
        combined = _local_routed_forward(module, latent, indices, weights)
    elif module.overlap_shared_expert and module.config.num_shared_experts:
        pending = ep_routed_dispatch(module, latent.float(), router_fn=router_fn, ep_group=ep_group)
        shared = module.shared_experts(hidden_states)
        combined = ep_routed_experts_and_combine(module, pending, ep_group)
    else:
        combined = ep_routed_forward(module, latent.float(), router_fn=router_fn, ep_group=ep_group)
    combined = combined.to(latent.dtype).reshape(-1, latent.shape[-1])
    if module.use_latent_moe:
        if module.latent_moe_use_norm:
            combined = module.routed_expert_norm(combined)
        combined = module.routed_expert_up_proj(combined)
    combined = combined.reshape(hidden_states.shape)
    if module.config.num_shared_experts:
        combined = combined + (module.shared_experts(hidden_states) if shared is None else shared)
    return combined


def _local_routed_forward(module: nn.Module, latent: torch.Tensor, indices: torch.Tensor,
                          weights: torch.Tensor) -> torch.Tensor:
    """Run the same stacked expert parameters without communication for EP1."""
    flattened = latent.reshape(-1, latent.shape[-1])
    token_ids = torch.arange(flattened.shape[0], device=latent.device)[:, None].expand_as(indices).reshape(-1)
    expert_ids = indices.reshape(-1)
    order = expert_ids.argsort(stable=True)
    values = module.experts(flattened[token_ids[order]], expert_ids[order]).float()
    values = values * weights.reshape(-1)[order, None]
    output = torch.zeros_like(flattened, dtype=torch.float32)
    return output.index_add(0, token_ids[order], values).reshape(latent.shape)


@local_compute
def kimi_k3_ep_compute_fn(*, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any,
                          ep_mesh: Any) -> Callable:
    """Bind full-hidden routing, latent all-to-all and K3 branch composition.

    Args:
        module: KimiK3EPMoeBlock installed by the explicit replacement rule.
        mesh: Active plan mesh used to recover DP ownership for QB statistics.
        tp_mesh: Tensor-parallel mesh; only an absent or size-one mesh is supported.
        cp_mesh: Orthogonal context mesh included in QB token aggregation.
        ep_mesh: Expert mesh containing the ep axis, or None for local execution.

    Returns:
        Bound MoE computation with the selected communication group and local
        expert count; dense branches retain their original parameter ownership.
    """
    if tp_mesh is not None and tp_mesh.size() != 1:
        raise ValueError("K3 EP currently requires TP1")
    if not isinstance(module, KimiK3EPMoeBlock):
        raise ValueError("K3 EP requires replace_kimi_k3_ep_moe")
    ep_size = ep_mesh["ep"].size() if ep_mesh is not None else 1
    if module.num_experts % ep_size:
        raise ValueError("K3 global expert count must be divisible by EP degree")
    module.experts.local_expert_count = module.num_experts // ep_size
    configure_kimi_k3_router_groups(module.gate, mesh, cp_mesh)
    ep_group = ep_mesh.get_group("ep") if ep_mesh is not None else None

    def compute_fn(target: nn.Module, hidden_states: torch.Tensor,
                   valid_token_mask: torch.Tensor | None = None,
                   collect_router_stats: bool = True) -> torch.Tensor:
        """Execute model-owned branches with the bound expert communication group.

        Args:
            target: Replaced K3 MoE module being executed.
            hidden_states: Full-width local decoder activations.
            valid_token_mask: Optional local token mask for QB statistics.
            collect_router_stats: Whether this call accumulates QB observations.

        Returns:
            Routed plus shared expert outputs in the input activation shape.
        """
        return _latent_ep_forward(target, hidden_states, ep_group, valid_token_mask, collect_router_stats)

    return compute_fn


__all__ = ["KimiK3EPMoeBlock", "KimiK3EPExperts", "StackExpertWeights", "UnstackExpertWeights",
           "configure_kimi_k3_router_groups", "kimi_k3_ep_compute_fn", "replace_kimi_k3_ep_moe"]
