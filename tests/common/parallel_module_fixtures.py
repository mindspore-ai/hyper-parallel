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
"""Complete small attention family for protocol, FSDP2 and checkpoint acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from hyper_parallel.components.modules.module_family import (
    ModuleFamily, VariantRule, build_family, get_module_family, register_module_family,
)
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.context_parallel.cp_context import SequencePartition
from hyper_parallel.distributed.module_binding import ParallelModuleSpec
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.replacement import module_replacement


@dataclass(frozen=True)
class AttentionConfig:
    """Global dimensions of the small dense causal algorithm."""

    width: int = 2


@dataclass(frozen=True)
class AttentionOperands:
    """Explicit prepared operands and their sequence-coordinate receipt."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    partition: SequencePartition


class LocalAttention(nn.Module):
    """Normal algorithm with identity exchange/restore stages."""

    def __init__(self, *, config: AttentionConfig | None = None, module: nn.Module | None = None,
                 context: Mapping[str, Any] | None = None) -> None:
        """Register full logical weights before layout or borrow a source parameter."""
        super().__init__()
        del context
        self.weight = (module.weight if module is not None
                       else nn.Parameter(torch.full((config.width, config.width), 0.2)))
        self.cp_context = None

    def bind_parallel(self, binding: Any) -> None:
        """Bind only resources consumed by already-defined class stages.

        Args:
            binding: Common parallel resource bundle.
        """
        self.cp_context = binding.cp_context

    def exchange_attention(self, operands: AttentionOperands) -> AttentionOperands:
        """Preserve prepared operands on the local algorithm path.

        Args:
            operands: Projected Q/K/V, after any model-owned preprocessing.
        """
        return operands

    def restore_attention(self, output: torch.Tensor) -> torch.Tensor:
        """Preserve local output ownership for the gather-KV algorithm.

        Args:
            output: Local-query attention output.
        """
        return output

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run the common projection/attention algorithm through explicit stages.

        Args:
            tokens: Local tokens [B,S,H].
        """
        projected = tokens @ self.weight
        operands = self.exchange_attention(AttentionOperands(
            projected, projected, projected, SequencePartition((tokens.shape[1],), 0)))
        scores = operands.query @ operands.key.transpose(-1, -2) / (tokens.shape[-1] ** 0.5)
        query_positions = torch.arange(tokens.shape[1], device=tokens.device) + operands.partition.offset
        key_positions = torch.arange(operands.key.shape[1], device=tokens.device)
        scores = scores.masked_fill(key_positions[None, :] > query_positions[:, None], float("-inf"))
        return self.restore_attention(scores.softmax(dim=-1) @ operands.value)

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Keep the local class forward intact when applying a parameter layout."""
        return ParallelModuleSpec("test.causal_attention.local")

    def state_contract(self) -> dict[str, str]:
        """Declare representation and numerical semantics across local/CP members."""
        return {"representation": "dense_projection_v1", "attention": "causal"}


class CPAttention(LocalAttention):
    """Only the operand-exchange stage adds CP communication."""

    def exchange_attention(self, operands: AttentionOperands) -> AttentionOperands:
        """Gather K/V while retaining local Q and explicit global coordinates.

        Args:
            operands: Local projected Q/K/V.
        """
        key, partition = self.cp_context.gather_sequence(operands.key)
        value, _ = self.cp_context.gather_sequence(operands.value)
        return AttentionOperands(operands.query, key, value, partition)

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Declare the tested class stages and their supported parameter layouts."""
        return ParallelModuleSpec("test.causal_attention.cp_gather", "cp",
                                  {"exchange_attention": ("operands",), "restore_attention": ("output",)},
                                  validate=validate_attention_layout)


def validate_attention_layout(*, module: nn.Module, sharding: ModuleShardingSpec, context: Any) -> None:
    """Reject unsupported combinations before parameter mutation."""
    del module
    if context.axis_sizes["tp"] != 1 or sharding.params != {"weight": {"cp": Replicate()}}:
        raise ValueError("test CP attention supports TP=1 and CP-replicated projection weights")


def attention_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Declare complete logical parameter ownership independently of templates."""
    del module, module_fqn, member, context
    return ModuleShardingSpec(params={"weight": {"cp": Replicate()}}, in_src={}, in_dst={})


# Public family entries retain constructor-style names for stable target paths.
@module_replacement
def AttentionFamily(*, config: AttentionConfig | None = None, module: nn.Module | None = None,  # pylint: disable=invalid-name
                    module_fqn: str = "", context: Mapping[str, Any] | None = None) -> nn.Module:
    """Provide one public construction interface for local and CP members."""
    return build_family(AttentionFamily, config=config, module=module, module_fqn=module_fqn, context=context)


def register_attention_family() -> None:
    """Register only this acceptance fixture, without production attention imports."""
    if get_module_family(AttentionFamily) is not None:
        return
    prefix = __name__
    register_module_family(ModuleFamily(
        "test_causal_attention", AttentionFamily,
        variants={"local": f"{prefix}.LocalAttention", "cp": f"{prefix}.CPAttention"},
        default_variant="local", parallel_axis="cp", variant_rules=(VariantRule({"parallel": True}, "cp"),),
        sharding=attention_sharding,
    ))


def attention_fsdp_units(model: nn.Module) -> tuple[str, ...]:
    """Name a real child Module call as the FSDP execution boundary.

    Args:
        model: Final model tree.
    """
    del model
    return ("attention",)


MODEL_SPEC = ModelAdapterSpec("TestAttentionModel", "test_attention_model", fsdp_wrap_modules=attention_fsdp_units)


class AttentionModel(nn.Module):
    """A directly constructed family model with an ordinary child Module call."""

    def __init__(self, config: AttentionConfig, *, context: Mapping[str, Any]) -> None:
        """Build the final member from complete config and constructor context.

        Args:
            config: Complete global algorithm facts.
        """
        super().__init__()
        self.attention = AttentionFamily(config=config, context=context)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Invoke the actual expert/attention unit through __call__.

        Args:
            tokens: Local sequence input.
        """
        return self.attention(tokens)


def create_attention_model(config: AttentionConfig, *, context: Mapping[str, Any]) -> AttentionModel:
    """Propagate one explicit construction context through a complete model.

    Args:
        config: Complete global algorithm facts.
    """
    register_attention_family()
    return AttentionModel(config, context=context)


@dataclass(frozen=True)
class ExpertConfig:
    """Global expert count and hidden dimensions for the transport acceptance family."""

    num_experts: int = 4
    width: int = 4


@dataclass(frozen=True)
class ExpertOperands:
    """Identity dispatch on the ordinary three-Tensor algorithm path."""

    hidden_states: torch.Tensor
    expert_indices: torch.Tensor
    routing_weights: torch.Tensor


class LocalScaleExperts(nn.Module):
    """Complete routed expert computation; routing remains outside the module."""

    def __init__(self, *, config: ExpertConfig | None = None, module: nn.Module | None = None,
                 context: Mapping[str, Any] | None = None) -> None:
        """Register full logical expert weights before layout."""
        super().__init__()
        del context
        self.num_experts = config.num_experts if module is None else module.num_experts
        self.weight = (module.weight if module is not None else nn.Parameter(torch.arange(
            config.num_experts * config.width, dtype=torch.float32).reshape(
                config.num_experts, config.width) / 50 + 0.1))
        self.ep_context = None

    def bind_parallel(self, binding: Any) -> None:
        """Bind the transport context used by existing stages.

        Args:
            binding: Shared resource bundle.
        """
        self.ep_context = binding.ep_context

    def dispatch(self, hidden_states: torch.Tensor, expert_indices: torch.Tensor,
                 routing_weights: torch.Tensor) -> ExpertOperands:
        """Return the unchanged three Tensor operands on the ordinary path.

        Args:
            hidden_states: Source token activations.
            expert_indices: Already selected global expert IDs.
            routing_weights: Final model-provided routing probabilities.
        """
        return ExpertOperands(hidden_states, expert_indices, routing_weights)

    def group_experts(self, batch: Any) -> torch.Tensor:
        """Apply the same expert math to ordinary or expert-major slots.

        Args:
            batch: Local three-Tensor operands or transported expert-major operands.
        """
        hidden, indices, weights = batch.hidden_states, batch.expert_indices, batch.routing_weights
        if indices.ndim == 2:
            values = torch.tanh(hidden[:, None, :] * self.weight[indices]) * weights[:, :, None]
            return values.sum(dim=1)
        return torch.tanh(hidden * self.weight[indices]) * weights[:, None]

    def combine(self, values: torch.Tensor, batch: Any) -> torch.Tensor:
        """Preserve computed local token outputs without communication.

        Args:
            values: Local expert outputs.
            batch: Original operand record.
        """
        del batch
        return values

    def forward(self, hidden_states: torch.Tensor, expert_indices: torch.Tensor,
                routing_weights: torch.Tensor) -> torch.Tensor:
        """Use normal Module execution around the explicit expert stages.

        Args:
            hidden_states: Source token activations.
            expert_indices: Already selected global expert IDs.
            routing_weights: Final routing probabilities.
        """
        batch = self.dispatch(hidden_states, expert_indices, routing_weights)
        return self.combine(self.group_experts(batch), batch)

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Declare that the ordinary forward owns its complete execution."""
        return ParallelModuleSpec("test.scale_experts.local")

    def state_contract(self) -> dict[str, str]:
        """Identify the common representation and probability application position."""
        return {"representation": "expert_major_scale_v1", "probability": "output"}


class EPScaleExperts(LocalScaleExperts):
    """Only dispatch/combine change the inherited expert computation path."""

    def dispatch(self, hidden_states: torch.Tensor, expert_indices: torch.Tensor,
                 routing_weights: torch.Tensor) -> Any:
        """Transport already-selected slots.

        Args:
            hidden_states: Source token activations.
            expert_indices: Global expert IDs.
            routing_weights: Final probabilities with gradients.
        """
        return self.ep_context.dispatch(hidden_states, expert_indices, routing_weights)

    def combine(self, values: torch.Tensor, batch: Any) -> torch.Tensor:
        """Return weighted expert slots to source tokens.

        Args:
            values: Expert-major slot outputs.
            batch: Dispatch result with its per-call receipt.
        """
        return self.ep_context.combine(values, batch.receipt)

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Declare balanced expert ownership and explicit stage signatures."""
        return ParallelModuleSpec("test.scale_experts.ep", "ep",
                                  {"dispatch": ("hidden_states", "expert_indices", "routing_weights"),
                                   "combine": ("values", "batch")}, validate=validate_expert_layout,
                                  options={"num_experts": self.num_experts})


def validate_expert_layout(*, module: nn.Module, sharding: ModuleShardingSpec, context: Any) -> None:
    """Require the tested expert-only computation layout."""
    del module
    if context.axis_sizes["tp"] != 1 or sharding.params != {"weight": {"ep": Shard(0)}}:
        raise ValueError("test experts support TP=1 and EP Shard(0)")


def expert_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Declare expert ownership directly instead of using an outer MoE template."""
    del module, module_fqn, member
    return ModuleShardingSpec(params={"weight": {"ep": Shard(0) if context.axis_sizes["ep"] > 1 else Replicate()}},
                              in_src={}, in_dst={})


# Public family entries retain constructor-style names for stable target paths.
@module_replacement
def ScaleExpertsFamily(*, config: ExpertConfig | None = None, module: nn.Module | None = None, module_fqn: str = "",  # pylint: disable=invalid-name
                       context: Mapping[str, Any] | None = None) -> nn.Module:
    """Build one complete ordinary or EP expert member from global facts."""
    return build_family(ScaleExpertsFamily, config=config, module=module, module_fqn=module_fqn, context=context)


class ExpertModel(nn.Module):
    """A model-owned routing seam calls the final expert Module normally."""

    def __init__(self, config: ExpertConfig, *, context: Mapping[str, Any]) -> None:
        """Construct the expert family before layout/FSDP/optimizer.

        Args:
            config: Global expert facts.
        """
        super().__init__()
        self.routed_bank = ScaleExpertsFamily(config=config, context=context)

    def forward(self, hidden_states: torch.Tensor, expert_indices: torch.Tensor,
                routing_weights: torch.Tensor) -> torch.Tensor:
        """Keep routing outside the normal expert Module call.

        Args:
            hidden_states: Source token activations.
            expert_indices: Already selected global expert IDs.
            routing_weights: Model-owned final routing probabilities.
        """
        return self.routed_bank(hidden_states, expert_indices, routing_weights)


def create_expert_model(config: ExpertConfig, *, context: Mapping[str, Any]) -> ExpertModel:
    """Register and construct the acceptance family using explicit EP context.

    Args:
        config: Global expert facts.
    """
    if get_module_family(ScaleExpertsFamily) is None:
        prefix = __name__
        register_module_family(ModuleFamily(
            "test_scale_experts", ScaleExpertsFamily,
            variants={"local": f"{prefix}.LocalScaleExperts", "ep": f"{prefix}.EPScaleExperts"},
            default_variant="local", parallel_axis="ep", variant_rules=(VariantRule({"parallel": True}, "ep"),),
            sharding=expert_sharding,
        ))
    return ExpertModel(config, context=context)
