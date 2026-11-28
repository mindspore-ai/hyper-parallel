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
"""HP-owned expert shards executed by the vLLM worker's graph-safe EP backend."""
from typing import Any

import torch

from hyper_parallel.components.checkpoint.load_groups import local_target_tensor
from hyper_parallel.components.functional.moe_token_unpermute import moe_token_unpermute
from hyper_parallel.components.modules.module_family import ModuleFamily, VariantRule, build_family, register_module_family
from hyper_parallel.components.modules.moe.grouped_experts.ep.ep_grouped_experts import validate_ep_layout
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import (
    GroupedExpertsModule, match_grouped_experts_inputs,
)
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts_family import grouped_experts_sharding
from hyper_parallel.distributed.module_binding import ParallelBinding, ParallelModuleSpec
from hyper_parallel.models.replacement import module_replacement
from rl.unified_execution.vllm.moe.fused_experts import AscendFusedExpertsBackend


class RolloutEPGroupedExperts(GroupedExpertsModule):
    """The per-layer mlp.experts module. Own the weights and pass the router's [T,K] choices to the shared backend."""

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Let HP place the router boundary and shard complete expert matrices over EP."""
        return ParallelModuleSpec(
            "rollout.grouped_experts.ep", binding_kind="mesh",
            stages={"forward": ("hidden_states", "topk_ids", "topk_weights")}, validate=validate_ep_layout,
        )

    def bind_parallel(self, binding: ParallelBinding) -> None:
        """Borrow the same EP group that owns this layer's weight shards."""
        self.ep_group = binding.mesh_binding.axis_mesh("ep").get_group()

    def forward(self, hidden_states, topk_ids=None, topk_weights=None, **kwargs):
        """Pass this layer's live weight shards and router choices into the captured expert computation."""
        topk_ids, topk_weights = match_grouped_experts_inputs(topk_ids, topk_weights, kwargs)
        states = hidden_states.reshape(-1, hidden_states.shape[-1])
        gate_up = local_target_tensor(self.gate_up_proj).transpose(-1, -2)
        down = local_target_tensor(self.down_proj).transpose(-1, -2)
        return rollout_experts(states, topk_ids, topk_weights, gate_up, down, self.backend_key).view_as(hidden_states)


@torch.library.custom_op("hyper_rl::rollout_experts", mutates_args=())
def rollout_experts(states: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor,
                    gate_up: torch.Tensor, down: torch.Tensor, backend_key: str) -> torch.Tensor:
    """The captured graph supplies the current layer's weights and native router outputs."""
    # vLLM is optional in training-only processes.
    from vllm.forward_context import get_forward_context  # pylint: disable=import-outside-toplevel

    backend = get_forward_context().no_compile_layers[backend_key]
    local_rows = states.shape[0]
    if not local_rows and backend.source_group is None:
        return torch.empty_like(states)

    packed, counts, ids, inverse, bucket = backend.token_dispatch(states, indices)
    values = backend.compute_experts(packed, counts, gate_up, down)
    slots = backend.token_combine(values, ids, inverse, local_rows, bucket)

    # An empty source still participates in EP collectives for other ranks.
    if not local_rows:
        return slots.new_empty((0, states.shape[-1]))
    # Slots already have router order. HP training supplies its actual inverse
    # permutation to this same operator; rollout supplies the identity.
    inverse = torch.arange(indices.numel(), dtype=torch.int32, device=slots.device)
    return moe_token_unpermute(slots.flatten(0, 1), inverse, weights.contiguous())


@rollout_experts.register_fake
def _rollout_experts_fake(states, indices, weights, gate_up, down, backend_key):
    del indices, weights, gate_up, down, backend_key
    return torch.empty_like(states)


@module_replacement
def build_rollout_experts(*, module, module_fqn, context):
    return build_family(build_rollout_experts, module=module, module_fqn=module_fqn, context=context)


register_module_family(ModuleFamily(
    name="rollout_grouped_experts", baseline=build_rollout_experts, parallel_axis="ep",
    default_variant="local", variants={
        "local": GroupedExpertsModule.__module__ + ".GroupedExpertsModule",
        "ep": __name__ + ".RolloutEPGroupedExperts",
    },
    variant_rules=(VariantRule({"parallel": True}, "ep"),), sharding=grouped_experts_sharding,
))


@torch.no_grad()
def register_expert_backend(
    model: torch.nn.Module, config: Any, topology: Any, prefix: str,
) -> AscendFusedExpertsBackend:
    """Share communication state across layers; graph nodes carry explicit per-layer weights."""
    key = f"{prefix}.hyper_rl.experts" if prefix else "hyper_rl.experts"
    registry = config.compilation_config.static_forward_context
    if key in registry:
        raise ValueError(f"Rollout experts already registered at {key}")
    layers = [module for module in model.modules() if isinstance(module, RolloutEPGroupedExperts)]
    group, = {layer.ep_group for layer in layers}
    backend = AscendFusedExpertsBackend(config, group, topology)
    for layer in layers:
        for parameter in (layer.gate_up_proj, layer.down_proj):
            local = local_target_tensor(parameter)
            if local.shape[0] != backend.local_count:
                raise ValueError("Rollout expert shard differs from HP ownership")
            # Keep logical [E,O,I] and the Parameter; give GMM a contiguous
            # [E,I,O] view once, before checkpoint loading and graph capture.
            local.set_(local.transpose(-1, -2).contiguous().transpose(-1, -2))
        layer.backend_key = key
    registry[key] = backend
    return backend
