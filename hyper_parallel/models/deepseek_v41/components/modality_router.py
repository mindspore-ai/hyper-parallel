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
"""Modality-aware routing, sequence balancing and deferred expert-bias updates as one V4.1 family."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale
from hyper_parallel.components.losses.moe_aux_loss import sequence_load_balancing_loss
from hyper_parallel.components.modules.family import SelectorSpec, VariantRule
from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.components.modules.moe.moe_utils import accumulate_router_load as accumulate_expert_load
from hyper_parallel.components.modules.moe.moe_utils import (
    update_router_bias,
)
from hyper_parallel.components.modules.moe.router import (
    MegaGateModule,
    balanced_expert_indices,
    routing_scores,
    select_topk_scores,
)
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed.module_binding import ParallelBinding, ParallelModuleSpec
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.materialization import register_rebuildable_buffer
from hyper_parallel.models.replacement import module_replacement


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


def _update_modality_biases(
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


@torch.no_grad()
def update_modality_bias(bias: torch.Tensor, counts: torch.Tensor, update_rate: float,
                         *, vision_bias: torch.Tensor | None = None) -> None:
    """Reuse common bias updates for either one modality or the text/image counter pair."""
    if counts.ndim == 2:
        _update_modality_biases(bias, counts, update_rate, vision_bias=vision_bias)
    else:
        update_router_bias(bias, counts, update_rate)


def _router_scores(logits: torch.Tensor, scoring_func: str) -> torch.Tensor:
    """Convert router logits to the released nonnegative expert scores."""
    if scoring_func not in ("sqrtsoftplus", "softmax", "sigmoid"):
        raise ValueError(f"Unsupported V4.1 router scoring function: {scoring_func!r}")
    return routing_scores(logits, scoring_func)


def _apply_router_training_state(
        module: nn.Module,
        hidden_states: torch.Tensor,
        scores: torch.Tensor | None,
        weights: torch.Tensor,
        indices: torch.Tensor,
        image_mask: torch.Tensor | None,
        token_mask: torch.Tensor | None,
        sequence_partition_groups: tuple[Any, ...],
        sequence_ids: torch.Tensor | None,
        num_sequences: int | None,
) -> torch.Tensor:
    """Attach sequence auxiliary gradients and deferred modality counts."""
    if token_mask is not None:
        if token_mask.shape != hidden_states.shape[:-1] or token_mask.dtype != torch.bool:
            raise ValueError("token_mask must be boolean and match the hidden-state token dimensions")
        token_mask = token_mask.reshape(-1)
    module.last_aux_loss = None
    if module.training and module.aux_loss_coeff:
        if scores is None:
            raise RuntimeError("router scores are required when auxiliary loss is enabled")
        if sequence_ids is None:
            num_sequences = hidden_states.shape[0] if hidden_states.ndim == 3 else 1
            sequence_ids = torch.arange(num_sequences, device=scores.device).repeat_interleave(
                hidden_states.shape[-2],
            )
        elif sequence_ids.shape != hidden_states.shape[:-1] or num_sequences is None:
            raise ValueError("sequence_ids must match token dimensions and specify num_sequences")
        unbiased_indices = torch.topk(scores, module.top_k, dim=-1, sorted=False).indices
        aux_loss = sequence_load_balancing_loss(
            scores,
            unbiased_indices,
            sequence_ids.reshape(-1),
            num_sequences,
            token_mask,
            sequence_partition_groups,
        )
        module.last_aux_loss = aux_loss.detach()
        weights = aux_loss_auto_scale(weights, aux_loss * module.aux_loss_coeff)
    if module.training and module.enable_expert_bias:
        weights = accumulate_router_load(weights, indices, image_mask, token_mask, module.tokens_per_expert)
    return weights


def configure_router_statistics(gate: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any) -> tuple[Any, ...]:
    """Configure bias-update groups and return the router's token-partition groups."""
    sequence_groups = tuple(
        axis.get_group() for axis in (cp_mesh, tp_mesh)
        if axis is not None and axis.size() > 1
    )
    dp_groups = tuple(
        mesh[name].get_group() for name in (mesh.mesh_dim_names if mesh is not None else ())
        if name in ("dp", "dp_replicate", "dp_shard") and mesh[name].size() > 1
    )
    gate.expert_bias_update_groups = dp_groups + sequence_groups
    return sequence_groups


class DeepseekV41TopKRouter(nn.Module):
    """V4.1 learned router with independent text and image correction biases."""

    fp32_parameter_names = ("bias", "bias_vl")

    def __init__(self, config: Any = None, *, module: nn.Module | None = None,
                 module_fqn: str = "", context: Any = None) -> None:
        """Create the released noaux_tc routing parameter layout."""
        super().__init__()
        del module_fqn, context
        self.sequence_partition_groups = ()
        self.router_ep_size = 1
        self.fix_router = getattr(module if module is not None else config,
                                  "moe_router_force_load_balancing", getattr(module, "fix_router", False))
        if not isinstance(self.fix_router, bool):
            raise ValueError("moe_router_force_load_balancing must be bool")
        self._routing_binding = None
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        if module is not None:
            for name in ("hidden_size", "num_experts", "top_k", "scoring_func", "routed_scaling_factor",
                         "aux_loss_coeff", "expert_bias_update_rate", "enable_expert_bias",
                         "expert_bias_update_groups", "last_aux_loss"):
                setattr(self, name, getattr(module, name))
            self.initializer_range = getattr(module, "initializer_range", 0.02)
            for name, parameter in module._parameters.items():
                self.register_parameter(name, parameter)
            for name, buffer in module._buffers.items():
                self.register_buffer(name, buffer, persistent=name not in module._non_persistent_buffers_set)
            register_rebuildable_buffer(self, "tokens_per_expert",
                                       value=torch.zeros(2, module.num_experts, dtype=torch.long, device="cpu"))
            self.training = module.training
            return
        self.initializer_range = float(getattr(config, "initializer_range", 0.02))
        self.hidden_size = int(config.hidden_size)
        self.num_experts = int(config.num_local_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.scoring_func = str(config.scoring_func)
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        self.aux_loss_coeff = float(getattr(config, "router_aux_loss_coef", 0.0))
        if not math.isfinite(self.aux_loss_coeff) or self.aux_loss_coeff < 0:
            raise ValueError("router_aux_loss_coef must be finite and nonnegative")
        self.last_aux_loss: torch.Tensor | None = None
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
        self.expert_bias_update_rate = float(getattr(config, "router_bias_update_rate", 0.001))
        if not math.isfinite(self.expert_bias_update_rate) or self.expert_bias_update_rate < 0:
            raise ValueError("router_bias_update_rate must be finite and nonnegative")
        self.enable_expert_bias = self.expert_bias_update_rate > 0
        self.expert_bias_update_groups: tuple[Any, ...] | None = None
        self.register_buffer("tokens_per_expert", torch.zeros(2, self.num_experts, dtype=torch.long), persistent=False)
        register_rebuildable_buffer(
            self, "tokens_per_expert", value=torch.zeros(2, self.num_experts, dtype=torch.long, device="cpu"),
        )
        self.bias = nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32), requires_grad=False)
        if bool(getattr(config, "v41_vision_enabled", False)):
            self.bias_vl = nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32), requires_grad=False)
        else:
            self.register_parameter("bias_vl", None)

    def bind_parallel(self, binding: ParallelBinding) -> None:
        """Borrow job-owned token and data groups for auxiliary loss and deferred bias updates."""
        context = binding.mesh_binding.mesh_context
        if self._routing_binding is not None and self._routing_binding.mesh_context is not context:
            raise ValueError("Modality router already belongs to another parallel context")
        self.router_ep_size = context.ep_size
        mesh = context.device_mesh
        axes = tuple(mesh.mesh_dim_names)
        cp_mesh = mesh["cp"] if "cp" in axes else None
        tp_mesh = mesh["tp"] if "tp" in axes and context.sequence_parallel else None
        self.sequence_partition_groups = configure_router_statistics(self, mesh, tp_mesh, cp_mesh)
        self._routing_binding = binding.mesh_binding

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Bind only router statistics; sibling Experts owns all token dispatch/combine."""
        return ParallelModuleSpec(
            "deepseek_v41.modality_router.statistics", binding_kind="mesh",
            stages={"compute_routing": ("hidden_states", "image_mask")}, validate=_validate_router_layout,
        )

    def _apply(self, fn: Any, recurse: bool = True) -> nn.Module:
        """Keep correction values and their gradients FP32 during model dtype conversion."""
        protected = {id(value) for value in (self.bias, self.bias_vl) if value is not None}
        protected.update(id(value.grad) for value in (self.bias, self.bias_vl)
                         if value is not None and value.grad is not None)

        def preserve(value: torch.Tensor) -> torch.Tensor:
            converted = fn(value)
            if id(value) not in protected or converted.dtype == torch.float32:
                return converted
            return converted.float() if value.is_meta else value.to(device=converted.device, dtype=torch.float32)

        return super()._apply(preserve, recurse=recurse)

    def make_transforms(self) -> list:
        """Retain router matrix, modality bias and source key identities."""
        return []

    def reset_parameters(self) -> None:
        """Restore routing matrix and neutral text/image selection biases."""
        nn.init.normal_(self.weight, std=self.initializer_range)
        nn.init.zeros_(self.bias)
        if self.bias_vl is not None:
            nn.init.zeros_(self.bias_vl)
        self.tokens_per_expert.zero_()

    def forward(
            self,
            hidden_states: torch.Tensor,
            image_mask: torch.Tensor | None = None,
            token_mask: torch.Tensor | None = None,
            sequence_partition_groups: tuple[Any, ...] | None = None,
            sequence_ids: torch.Tensor | None = None,
            num_sequences: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw logits, routing weights, and selected experts.

        ``bias`` and ``bias_vl`` select experts only; the gathered routing
        weights intentionally use the unbiased scores, matching V4.1.

        Args:
            hidden_states: Token activations ending in the hidden dimension.
            image_mask: Optional boolean [batch, sequence] image-token mask.
            token_mask: Optional boolean mask matching the token dimensions;
                true tokens contribute to auxiliary statistics.
            sequence_partition_groups: Independent CP/TP groups partitioning
                one microbatch's token stream, supplied by the parallel adapter.
            sequence_ids: Optional logical sample IDs matching the token dimensions.
            num_sequences: Total logical samples, including samples absent on this rank.

        Returns:
            Flattened logits, scaled top-k weights, and expert indices. During
            training with a positive coefficient, weights carry the auxiliary
            gradient and ``last_aux_loss`` stores its unweighted detached value.
        """
        if self._routing_binding is not None:
            context = self._routing_binding.mesh_context
            if context.sequence_parallel and context.tp_size > 1:
                def localize(value: torch.Tensor | None) -> torch.Tensor | None:
                    """Select the TP-local routing metadata without cutting the CP window twice."""
                    if value is None or tuple(value.shape) == tuple(hidden_states.shape[:-1]):
                        return value
                    token_dim = hidden_states.ndim - 2
                    local_length = hidden_states.shape[token_dim]
                    if (value.ndim != hidden_states.ndim - 1
                            or value.shape[token_dim] != local_length * context.tp_size):
                        raise ValueError("Router metadata must cover the local or CP-local token interval")
                    return value.narrow(token_dim, context.tp_rank * local_length, local_length).contiguous()

                image_mask, token_mask, sequence_ids = map(localize, (image_mask, token_mask, sequence_ids))
        if sequence_partition_groups is None:
            sequence_partition_groups = self.sequence_partition_groups
        logits, scores, weights, indices = self.compute_routing(hidden_states, image_mask)
        if self.fix_router:
            indices = balanced_expert_indices(indices, self.num_experts, self.router_ep_size)
        weights = _apply_router_training_state(
            self,
            hidden_states,
            scores,
            weights,
            indices,
            image_mask,
            token_mask,
            sequence_partition_groups,
            sequence_ids,
            num_sequences,
        )
        return logits, weights, indices

    def compute_routing(self, hidden_states: torch.Tensor,
                        image_mask: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        """Project and select experts; deferred training state is owned by the complete router."""
        flattened = hidden_states.reshape(-1, self.hidden_size)
        logits = functional.linear(  # pylint: disable=not-callable
            flattened.float(), self.weight.float()
        )
        scores = _router_scores(logits, self.scoring_func)
        correction_bias = self.bias
        if image_mask is not None:
            if image_mask.shape != hidden_states.shape[:-1] or image_mask.dtype != torch.bool:
                raise ValueError("image_mask must be boolean and match the hidden-state token dimensions")
            if self.bias_vl is not None:
                correction_bias = torch.where(
                    image_mask.reshape(-1, 1),
                    self.bias_vl.unsqueeze(0),
                    self.bias.unsqueeze(0),
                )
        weights, indices = select_topk_scores(
            scores, top_k=self.top_k, correction_bias=correction_bias,
            routed_scaling_factor=self.routed_scaling_factor, normalize_topk=self.top_k > 1,
        )
        return logits, scores, weights, indices

    @torch.no_grad()

    def update_expert_bias(self, lr: float | None = None, num_recomputations: int = 1) -> None:
        """Update text/image selection biases once after the optimizer step.

        The shared monitor synchronizes counts first. Backward counts each
        microbatch once, so checkpoint recomputation needs no correction.

        Args:
            lr: Optional step size, defaulting to this router's configured rate.
            num_recomputations: Shared monitor compatibility argument; unused.
        """
        del num_recomputations
        if lr is None:
            lr = self.expert_bias_update_rate
        update_modality_bias(self.bias, self.tokens_per_expert[0], lr)
        if self.bias_vl is not None:
            update_modality_bias(self.bias_vl, self.tokens_per_expert[1], lr)
        self.tokens_per_expert.zero_()


class FusedModalityTopKRouter(DeepseekV41TopKRouter):
    """Compose public fused routing with the same modality statistics and auxiliary gradient."""

    def compute_routing(self, hidden_states: torch.Tensor,
                        image_mask: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        """Use native sqrt-softplus on NPU and the source's declared math on other devices."""
        if image_mask is not None and (
                image_mask.shape != hidden_states.shape[:-1] or image_mask.dtype != torch.bool):
            raise ValueError("image_mask must be boolean and match the hidden-state token dimensions")
        if hidden_states.device.type != "npu" or self.scoring_func != "sqrtsoftplus" or hidden_states.numel() == 0:
            return super().compute_routing(hidden_states, image_mask)
        # The source owns dual modality counters; reuse only the shared native execution phase.
        logits, weights, indices = MegaGateModule._native_forward(  # pylint: disable=protected-access
            self, hidden_states, image_mask,
        )
        scores = _router_scores(logits, self.scoring_func) if self.training and self.aux_loss_coeff else None
        return logits, scores, weights, indices


def _router_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Own replicated routing weights and correction state independently of expert tables."""
    del module_fqn, member
    placements = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    return ModuleShardingSpec(params={name: dict(placements) for name, _ in module.named_parameters()},
                              in_src={}, in_dst={}, is_boundary=False)


def _validate_router_layout(*, module: nn.Module, sharding: Any, context: Any) -> None:
    """Routing observes local token partitions using replicated matrices and statistics state."""
    del module, context
    if any(not isinstance(placement, Replicate) for axes in sharding.params.values() for placement in axes.values()):
        raise ValueError("Modality router requires replicated parameters")


@module_replacement
def ModalityTopKRouter(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "",
    context: Any = None, mesh: Any = None, use_fused_ops: bool = False,
) -> nn.Module:
    """Build the complete modality-aware router through the common family lifecycle."""
    return build_family(ModalityTopKRouter, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, use_fused_ops=use_fused_ops)


register_module_family(ModuleFamily(
    name="modality_topk_router", baseline=ModalityTopKRouter,
    variants={"local": "hyper_parallel.models.deepseek_v41.components.modality_router.DeepseekV41TopKRouter",
              "fused": "hyper_parallel.models.deepseek_v41.components.modality_router.FusedModalityTopKRouter"},
    default_variant="local", sharding=_router_sharding, selector_specs={"use_fused_ops": SelectorSpec(bool, False)},
    variant_rules=(VariantRule({"use_fused_ops": True, "parallel": False}, "fused"),),
))
