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
"""Gate and score-corrected grouped Top-K router module families."""
from __future__ import annotations

from collections.abc import Mapping
from math import isfinite, prod, sqrt
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.components.checkpoint.weight_conversion import ConversionOps, WeightConverter, WeightRenaming
from hyper_parallel.components.losses.aux_loss import aux_loss_auto_scale, sequence_load_balancing_loss
from hyper_parallel.components.ops import moe_router
from hyper_parallel.components.modules.family import SelectorSpec, VariantRule
from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.components.modules.moe.moe_utils import (
    accumulate_router_load, local_router_metadata, router_sharding,
    update_router_bias, validate_router_layout,
)
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.module_binding import ParallelBinding, ParallelModuleSpec
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.materialization import register_rebuildable_buffer
from hyper_parallel.models.replacement import module_replacement


def _setting(sources: tuple, names: tuple[str, ...], default: Any) -> Any:
    for source in sources:
        for name in names:
            value = source.get(name) if isinstance(source, Mapping) else getattr(source, name, None)
            if value is not None:
                return value
    return default


class _InitialRouterBias(ConversionOps):
    """Derive training-only correction bias when importing a bias-free HF router."""

    def __init__(self, reverse: bool = False) -> None:
        self.reverse = reverse

    def convert(self, input_dict: dict[str, Any], source_patterns: list[str],
                target_patterns: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Preserve the projection and initialize only newly introduced FP32 bias.

        Args:
            input_dict: Native projection on import, family projection/bias on export.
            source_patterns: Collected source keys in the router scope.
            target_patterns: Projection key and, on import, correction-bias key.
            **kwargs: Additional checkpoint-loader context, unused.

        Returns:
            Unchanged projection plus zero FP32 bias on import; projection on HF export.
        """
        del kwargs
        weight = input_dict[source_patterns[0]]
        weight = weight[0] if isinstance(weight, list) else weight
        result = {target_patterns[0]: weight}
        if not self.reverse:
            result[target_patterns[1]] = torch.zeros(weight.shape[0], device=weight.device, dtype=torch.float32)
        return result

    @property
    def reverse_op(self) -> ConversionOps:
        """Exclude derived correction state only when exporting a native HF checkpoint."""
        return _InitialRouterBias(reverse=not self.reverse)


def _router_transforms(module: nn.Module, projection: str) -> list[WeightRenaming | WeightConverter]:
    """Keep native state and derive correction bias only when it has no source key."""
    parameters = dict(module.named_parameters())
    if "bias" in parameters and not module.source_has_correction_bias:
        return [WeightConverter(source_patterns=projection, target_patterns=[projection, "bias"],
                                operations=[_InitialRouterBias()]),
                *(WeightRenaming(source_patterns=name, target_patterns=name)
                  for name in parameters if name not in (projection, "bias"))]
    return [WeightRenaming(source_patterns=name, target_patterns=name) for name in parameters]


def routing_scores(logits: torch.Tensor, scoring_func: str) -> torch.Tensor:
    """Apply nonnegative expert scoring without correction bias.

    Args:
        logits: Projection outputs ending in the expert dimension.
        scoring_func: sqrtsoftplus, softmax or sigmoid.
    """
    if scoring_func == "sqrtsoftplus":
        return F.softplus(logits).sqrt()  # pylint: disable=not-callable
    if scoring_func == "softmax":
        return logits.softmax(dim=-1)
    if scoring_func == "sigmoid":
        return logits.sigmoid()
    raise ValueError(f"unsupported router scoring_func: {scoring_func!r}")


def select_topk_scores(scores: torch.Tensor, *, top_k: int, correction_bias: torch.Tensor | None = None,
                       routed_scaling_factor: float = 1.0, normalize_topk: bool = True,
                       sorted_indices: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Select with correction bias, then gather unbiased weights and optionally normalize.

    Args:
        scores: Floating unbiased expert scores [tokens, experts].
        top_k: Selected experts per token.
        correction_bias: Optional selection-only [experts] or [tokens, experts] bias.
        routed_scaling_factor: Multiplier applied to selected weights.
        normalize_topk: Normalize selected weights to sum to one before scaling.
        sorted_indices: Sort selected experts by selection score.
    """
    if scores.ndim != 2 or not scores.is_floating_point():
        raise ValueError("scores must be a floating [tokens, experts] tensor")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= scores.shape[-1]:
        raise ValueError("top_k must be an integer in [1, num_experts]")
    if (isinstance(routed_scaling_factor, bool) or not isinstance(routed_scaling_factor, (int, float))
            or not isfinite(routed_scaling_factor)):
        raise ValueError("routed_scaling_factor must be finite")
    if not all((isinstance(normalize_topk, bool), isinstance(sorted_indices, bool))):
        raise TypeError("normalize_topk and sorted_indices must be bool")
    selection = scores
    if correction_bias is not None:
        if (correction_bias.device != scores.device or not correction_bias.is_floating_point()
                or tuple(correction_bias.shape) not in (
                    (scores.shape[-1],), (1, scores.shape[-1]), tuple(scores.shape))):
            raise ValueError("correction_bias must share the device and broadcast as [experts] or [tokens, experts]")
        selection = scores + correction_bias.float()
    indices = selection.topk(top_k, dim=-1, sorted=sorted_indices).indices
    weights = scores.gather(1, indices)
    if normalize_topk:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * routed_scaling_factor, indices


def balanced_expert_indices(indices: torch.Tensor, num_experts: int, ep_size: int = 1) -> torch.Tensor:
    """Replace destinations with the existing deterministic owner-major routing pattern.

    Args:
        indices: Learned integer expert IDs [tokens, top_k]; only shape/device/dtype are used.
        num_experts: Global expert count, divisible by the physical EP owner count.
        ep_size: Physical expert owners, including the TP extension of EP.

    Returns:
        Balanced global expert IDs with the same shape, dtype and device.
        Learned routing weights are deliberately left unchanged by callers.
    """
    if indices.ndim != 2 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must be integer [tokens, top_k]")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in (num_experts, ep_size)) or num_experts % ep_size:
        raise ValueError("num_experts must be positive and divisible by the positive EP owner count")
    if not 1 <= indices.shape[1] <= num_experts:
        raise ValueError("top_k must be in [1, num_experts]")
    slots = torch.arange(indices.numel(), device=indices.device)
    local_experts = num_experts // ep_size
    fixed = (slots % ep_size) * local_experts + (slots // ep_size) % local_experts
    return fixed.reshape_as(indices).to(indices.dtype)


class GateModule(nn.Module):
    """Own common gate configuration, projection state, initialization and routing math.

    Projection uses FP32 operands under the enclosing Torch autocast policy.
    The same module owns optional balancing state and parallel statistics;
    fused execution belongs to the MegaGate variant.
    """

    fp32_parameter_names = ("bias",)

    def __init__(self, *, config: Any = None, module: nn.Module | None = None,
                 context: Mapping[str, Any] | None = None) -> None:
        """Construct a gate or retain a public member and its optional training state."""
        super().__init__()
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        if module is not None:
            self._validate_source(module)
        source = config if config is not None else module
        self.source_has_correction_bias = module is not None and getattr(module, "bias", None) is not None
        self._set_routing_settings(source)
        if module is None:
            self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
            # Derived gates create their correction state after the common projection.
            GateModule.reset_parameters(self)
        else:
            weight = getattr(module, "weight", None)
            if (not isinstance(weight, nn.Parameter) or not weight.is_floating_point()
                    or tuple(weight.shape) != (self.num_experts, self.hidden_size)):
                raise ValueError("source weight must be a floating Parameter [num_experts, hidden_size]")
            self.weight = weight
            self.train(module.training)
        self._init_router_training(source, context, self.weight, module)

    def _set_routing_settings(self, source: Any) -> None:
        """Validate dimensions, scoring, TopK normalization and initialization policy."""
        for name in ("hidden_size", "num_experts", "top_k"):
            value = getattr(source, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, value)
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        self.scoring_func = source.scoring_func
        if self.scoring_func not in ("sqrtsoftplus", "softmax", "sigmoid"):
            raise ValueError("unsupported scoring_func")
        self.routed_scaling_factor = source.routed_scaling_factor
        if (isinstance(self.routed_scaling_factor, bool)
                or not isinstance(self.routed_scaling_factor, (int, float))
                or not isfinite(self.routed_scaling_factor)):
            raise ValueError("routed_scaling_factor must be finite")
        self.normalize_topk = getattr(source, "normalize_topk", self.top_k > 1)
        self.sorted_indices = getattr(source, "sorted_indices", False)
        if not all((isinstance(self.normalize_topk, bool), isinstance(self.sorted_indices, bool))):
            raise TypeError("normalize_topk and sorted_indices must be bool")
        self.initializer_range = source.initializer_range
        if (isinstance(self.initializer_range, bool) or not isinstance(self.initializer_range, (int, float))
                or not isfinite(self.initializer_range) or self.initializer_range < 0):
            raise ValueError("initializer_range must be finite and nonnegative")

    @classmethod
    def _validate_source(cls, module: nn.Module) -> None:
        if module.__class__ is not cls:
            raise TypeError("model routers own extra state; reuse routing math rather than replacing the whole router")

    def reset_parameters(self) -> None:
        """Initialize the projection after direct/meta construction."""
        nn.init.normal_(self.weight, std=self.initializer_range)
        bias = getattr(self, "bias", None)
        if bias is not None:
            nn.init.zeros_(bias)

    def make_transforms(self) -> list[WeightRenaming | WeightConverter]:
        """Map native projection/state and derive optional training-only correction bias.

        Returns:
            Native state mappings and, when required, a zero FP32 bias initializer.
        """
        return _router_transforms(self, "weight")

    def forward(self, hidden_states: torch.Tensor,
                correction_bias: torch.Tensor | None = None, *,
                token_mask: torch.Tensor | None = None,
                sequence_ids: torch.Tensor | None = None, num_sequences: int | None = None,
                sequence_partition_groups: tuple | None = None) -> tuple[torch.Tensor, ...]:
        """Return logits [T,E], learned weights [T,K] and actual dispatch IDs [T,K].

        Args:
            hidden_states: Source activations [..., hidden_size].
            correction_bias: Optional additional selection-only expert bias.
            token_mask: Boolean token metadata excluding padding from balancing statistics.
            sequence_ids: Int64 logical sample IDs for packed/partitioned sequences.
            num_sequences: Number of logical samples when sequence_ids is provided.
            sequence_partition_groups: Override existing token-partition groups for aux-loss.
        """
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must end in hidden_size and include a token dimension")
        token_mask, sequence_ids = self._metadata(hidden_states, token_mask, sequence_ids)
        own_bias = getattr(self, "bias", None)
        if own_bias is not None:
            correction_bias = own_bias if correction_bias is None else own_bias + correction_bias
        logits = F.linear(  # pylint: disable=not-callable
            hidden_states.reshape(-1, self.hidden_size).float(), self.weight.float())
        scores = routing_scores(logits, self.scoring_func)
        weights, indices = select_topk_scores(
            scores, top_k=self.top_k, correction_bias=correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            normalize_topk=self.normalize_topk, sorted_indices=self.sorted_indices)
        weights, indices = self._finish_routing(
            hidden_states, logits, weights, indices, token_mask, sequence_ids,
            num_sequences, sequence_partition_groups, scores=scores,
        )
        return logits, weights, indices

    def _finish_routing(self, hidden: torch.Tensor, logits: torch.Tensor, weights: torch.Tensor,
                        indices: torch.Tensor, token_mask: torch.Tensor | None,
                        sequence_ids: torch.Tensor | None, num_sequences: int | None,
                        sequence_partition_groups: tuple | None, *, scores: torch.Tensor | None = None,
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply training to the final expert IDs and their learned weights."""
        if self.fix_router:
            indices = balanced_expert_indices(indices, self.num_experts, self.router_ep_size)
        if scores is None:
            scores = routing_scores(logits, self.scoring_func) if self.training and self.aux_loss_coeff else logits
        weights = self._apply_router_training(
            hidden, scores, weights, indices, token_mask, sequence_ids, num_sequences, sequence_partition_groups,
        )
        return weights, indices

    def _init_router_training(self, source: Any, context: Mapping[str, Any] | None,
                              weight: nn.Parameter, module: nn.Module | None = None) -> None:
        """Resolve router options and install state for backward accounting."""
        sources = (*tuple((context or {}).get("module_configs", ())), source)
        self.fix_router = _setting(
            sources, ("moe_router_force_load_balancing", "fix_router", "router_fix_router"), False,
        )
        if not isinstance(self.fix_router, bool):
            raise ValueError("moe_router_force_load_balancing must be bool")
        for name, aliases in (
                ("aux_loss_coeff", (
                    "moe_aux_loss_coeff", "aux_loss_coeff", "router_aux_loss_coeff",
                )),
                ("expert_bias_update_rate", (
                    "moe_router_bias_update_rate", "expert_bias_update_rate", "router_bias_update_rate",
                ))):
            value = _setting(sources, aliases, 0.0)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
                raise ValueError(f"{aliases[0]} must be finite and nonnegative")
            setattr(self, name, float(value))
        self.enable_expert_bias = self.expert_bias_update_rate > 0
        self.last_aux_loss = None
        self.router_ep_size = 1
        self.router_tp_mesh = None
        self.sequence_partition_groups = ()
        self.expert_bias_update_groups = None
        counts = getattr(module, "tokens_per_expert", None)
        if module is not None and counts is None and not self.enable_expert_bias:
            # A source without bias balancing has no registered counter to preserve.
            self.tokens_per_expert = None
            return
        if counts is None:
            counts = torch.zeros(self.num_experts, dtype=torch.int64, device=weight.device)
        if not all((counts.shape == (self.num_experts,), counts.dtype == torch.int64, counts.device == weight.device)):
            raise ValueError("source tokens_per_expert must be int64 [num_experts] on the weight device")
        self.register_buffer("tokens_per_expert", counts, persistent=False)
        register_rebuildable_buffer(self, "tokens_per_expert", value=torch.zeros(
            self.num_experts, dtype=torch.int64, device="cpu",
        ))
        if self.enable_expert_bias:
            self._register_correction_state(module, weight)

    def _register_correction_state(self, module: nn.Module | None, weight: nn.Parameter) -> None:
        """Create or adopt the expert bias used for auxiliary-loss-free balancing."""
        parameter = getattr(module, "bias", None)
        if parameter is None:
            parameter = nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32, device=weight.device),
                                     requires_grad=False)
        else:
            if (not isinstance(parameter, nn.Parameter) or parameter.shape != (self.num_experts,)
                    or parameter.dtype != torch.float32 or parameter.device != weight.device):
                raise ValueError("source bias must be an FP32 Parameter [num_experts] on the weight device")
            parameter.requires_grad_(False)
        self.register_parameter("bias", parameter)

    def _apply(self, fn: Any, recurse: bool = True) -> Any:
        """Keep selection correction state and its gradients FP32 during dtype conversion."""
        parameters = tuple(getattr(self, name, None) for name in self.fp32_parameter_names)
        protected = {id(value) for parameter in parameters if parameter is not None
                     for value in (parameter, parameter.grad) if value is not None}

        def apply(tensor: torch.Tensor) -> torch.Tensor:
            """Apply storage conversion while preserving correction precision.

            Args:
                tensor: Parameter, gradient or buffer visited by the module conversion.
            """
            converted = fn(tensor)
            if id(tensor) not in protected or converted.dtype == torch.float32:
                return converted
            return converted.float() if tensor.is_meta else tensor.to(device=converted.device, dtype=torch.float32)

        return super()._apply(apply, recurse=recurse)

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Declare a replicated router that borrows existing statistics meshes."""
        return ParallelModuleSpec(
            protocol="moe.router.statistics", binding_kind="mesh",
            stages={"forward": ("hidden_states",)}, validate=validate_router_layout,
        )

    def bind_parallel(self, binding: ParallelBinding) -> None:
        """Bind source-token statistics independently of expert transport.

        Args:
            binding: Job-owned mesh resources; no new process groups are created.
        """
        mesh = binding.mesh_binding.mesh_context
        device_mesh = mesh.device_mesh
        names = tuple(device_mesh.mesh_dim_names or ()) if device_mesh is not None else ()
        tp_partitioned = mesh.sequence_parallel or mesh.ep_size > 1
        tp_mesh = device_mesh["tp"] if "tp" in names and tp_partitioned else None
        sequence_axes = [device_mesh["cp"]] if "cp" in names else []
        if tp_mesh is not None:
            sequence_axes.append(tp_mesh)
        sequence_groups = tuple(axis.get_group() for axis in sequence_axes if axis.size() > 1)
        dp_groups = tuple(device_mesh[name].get_group() for name in names
                          if name in ("dp", "dp_replicate", "dp_shard") and device_mesh[name].size() > 1)
        owners = binding.mesh_binding.axis_mesh("ep").size() if mesh.ep_size > 1 else 1
        if self.num_experts % owners:
            raise ValueError("router expert count must be divisible by the physical EP owner count")
        self.router_ep_size = owners
        self.router_tp_mesh = tp_mesh
        self.sequence_partition_groups = sequence_groups
        self.expert_bias_update_groups = dp_groups + sequence_groups

    def _metadata(self, hidden: torch.Tensor, token_mask: torch.Tensor | None,
                  sequence_ids: torch.Tensor | None) -> tuple:
        return tuple(local_router_metadata(value, hidden, self.router_tp_mesh)
                     for value in (token_mask, sequence_ids))

    def _apply_router_training(self, hidden: torch.Tensor, scores: torch.Tensor, weights: torch.Tensor,
                               indices: torch.Tensor, token_mask: torch.Tensor | None,
                               sequence_ids: torch.Tensor | None,
                               num_sequences: int | None, groups: tuple | None) -> torch.Tensor:
        """Attach auxiliary gradients and count the routes consumed by backward."""
        if token_mask is not None and (token_mask.shape != hidden.shape[:-1] or token_mask.dtype != torch.bool
                                       or token_mask.device != hidden.device):
            raise ValueError("token_mask must be boolean and match local token shape and device")
        valid = None if token_mask is None else token_mask.reshape(-1)
        self.last_aux_loss = None
        if self.training and self.aux_loss_coeff:
            auxiliary = self._sequence_auxiliary(
                hidden, scores, indices, valid, sequence_ids, num_sequences, groups,
            )
            self.last_aux_loss = auxiliary.detach()
            weights = aux_loss_auto_scale(weights, auxiliary * self.aux_loss_coeff)
        if self.training and self.enable_expert_bias:
            weights = accumulate_router_load(weights, indices, token_mask, self.tokens_per_expert)
        return weights

    def _sequence_auxiliary(self, hidden: torch.Tensor, scores: torch.Tensor, indices: torch.Tensor,
                            valid: torch.Tensor | None, sequence_ids: torch.Tensor | None,
                            num_sequences: int | None, groups: tuple | None) -> torch.Tensor:
        """Resolve token ownership and evaluate sequence load balancing."""
        if sequence_ids is None:
            num_sequences = prod(hidden.shape[:-2])
            sequence_ids = torch.arange(num_sequences, device=hidden.device).repeat_interleave(hidden.shape[-2])
            num_sequences = max(1, num_sequences)
        elif (sequence_ids.shape != hidden.shape[:-1] or sequence_ids.dtype != torch.int64
              or sequence_ids.device != hidden.device or num_sequences is None):
            raise ValueError("sequence_ids must be local int64 token metadata with num_sequences")
        return sequence_load_balancing_loss(
            scores, indices, sequence_ids.reshape(-1), num_sequences, valid,
            self.sequence_partition_groups if groups is None else groups,
        )

    @torch.no_grad()
    def update_expert_bias(self, lr: float | None = None, num_recomputations: int = 1) -> None:
        """Update selection state once after the step's consumed backwards.

        Args:
            lr: Override update speed, otherwise use the configured rate.
            num_recomputations: Compatibility argument; backward counters need no replay correction.
        """
        del num_recomputations
        rate = self.expert_bias_update_rate if lr is None else lr
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not isfinite(rate) or rate < 0:
            raise ValueError("lr must be finite and nonnegative")
        rate = float(rate)
        if self.enable_expert_bias:
            update_router_bias(self.bias, self.tokens_per_expert, rate)
        if self.tokens_per_expert is not None:
            self.tokens_per_expert.zero_()


class MegaGateModule(GateModule):
    """Extend the ordinary gate with correction biases and a fused NPU route.

    The Gate base owns the projection and common routing contract. HyperMulticore
    supplies the native function and plan, without owning this module's state.
    """

    fp32_parameter_names = ("bias", "bias_vl")

    def __init__(self, *, config: Any = None, module: nn.Module | None = None,
                 module_fqn: str = "", context: Mapping[str, Any] | None = None) -> None:
        """Construct or adopt the public MegaGate without accepting model-specific subclasses."""
        del module_fqn
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        if module is not None:
            self._validate_source(module)
        source = config if config is not None else module
        if (getattr(source, "normalize_topk", source.top_k > 1) != (source.top_k > 1)
                or getattr(source, "sorted_indices", False)):
            raise ValueError("MegaGate requires normalize_topk=(top_k > 1) and unsorted indices")
        vision_enabled = (getattr(source, "vision_enabled", False)
                          if config is not None else getattr(source, "bias_vl", None) is not None)
        if not isinstance(vision_enabled, bool):
            raise TypeError("vision_enabled must be bool")
        super().__init__(config=config, module=module, context=context)
        self._initialize_correction_state(module, vision_enabled)

    def _initialize_correction_state(self, module: nn.Module | None, vision_enabled: bool) -> None:
        """Initialize or adopt the fused backend's text and optional image biases."""
        vision_enabled = vision_enabled or getattr(self, "bias_vl", None) is not None
        for name in ("bias", "bias_vl"):
            if name == "bias_vl" and not vision_enabled:
                self.register_parameter(name, None)
                continue
            existing = getattr(self, name, None)
            if existing is not None:
                continue
            if module is None:
                parameter = nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32, device=self.weight.device))
            else:
                parameter = getattr(module, name, None)
                if (not isinstance(parameter, nn.Parameter) or tuple(parameter.shape) != (self.num_experts,)
                        or parameter.device != self.weight.device or parameter.dtype != torch.float32):
                    raise ValueError(f"source {name} must be an FP32 Parameter [num_experts] on the weight device")
            setattr(self, name, parameter)

    def reset_parameters(self) -> None:
        """Reset common projection/bias and the fused variant's optional vision bias."""
        super().reset_parameters()
        if getattr(self, "bias_vl", None) is not None:
            nn.init.zeros_(self.bias_vl)

    @classmethod
    def _validate_source(cls, module: nn.Module) -> None:
        if module.__class__ is cls:
            return
        if not module.__class__.__module__.startswith("hyper_multicore.modules.mega_gate"):
            raise TypeError("MegaGate family accepts only the public MegaGate; use a model adapter for other routers")
        # Existing HyperMulticore modules remain valid sources; this is not a base class.
        from hyper_multicore.modules.mega_gate import (  # pylint: disable=import-outside-toplevel
            MegaGate as SourceMegaGate,
        )

        if module.__class__ is not SourceMegaGate:
            raise TypeError("MegaGate family accepts only the public MegaGate; use a model adapter for other routers")

    def _torch_forward(self, hidden_states: torch.Tensor,
                       image_mask: torch.Tensor | None, **metadata: Any) -> tuple[torch.Tensor, ...]:
        """Keep fused vision selection local to this variant; training counts stay standard."""
        if image_mask is None:
            return super().forward(hidden_states, **metadata)
        correction_bias = torch.where(image_mask.reshape(-1, 1), self.bias_vl, self.bias)
        logits = F.linear(  # pylint: disable=not-callable
            hidden_states.reshape(-1, self.hidden_size).float(), self.weight.float())
        scores = routing_scores(logits, self.scoring_func)
        weights, indices = select_topk_scores(
            scores, top_k=self.top_k, correction_bias=correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            normalize_topk=self.normalize_topk, sorted_indices=self.sorted_indices)
        weights, indices = self._finish_routing(hidden_states, logits, weights, indices, scores=scores, **metadata)
        return logits, weights, indices

    def _native_forward(self, hidden_states: torch.Tensor,
                        image_mask: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        return moe_router(
            hidden_states, self.weight, self.bias, self.bias_vl if image_mask is not None else None,
            image_mask, self.top_k, self.scoring_func, self.routed_scaling_factor,
        )

    # The fused NPU API takes a modality mask in the second argument.
    def forward(self, hidden_states: torch.Tensor,  # pylint: disable=arguments-renamed
                image_mask: torch.Tensor | None = None, *, token_mask: torch.Tensor | None = None,
                sequence_ids: torch.Tensor | None = None, num_sequences: int | None = None,
                sequence_partition_groups: tuple | None = None) -> tuple[torch.Tensor, ...]:
        """Use shared Gate math or the native sqrt-softplus NPU route.

        Args:
            hidden_states: Source activations [..., hidden_size].
            image_mask: Boolean local token metadata for vision correction bias.
            token_mask: Boolean local token metadata excluding padding from statistics.
            sequence_ids: Int64 local logical sample IDs.
            num_sequences: Logical sample count when sequence_ids is provided.
            sequence_partition_groups: Override token-partition groups for aux-loss.
        """
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must end in hidden_size and include a token dimension")
        token_mask, sequence_ids = self._metadata(hidden_states, token_mask, sequence_ids)
        image_mask = local_router_metadata(image_mask, hidden_states, self.router_tp_mesh)
        if image_mask is not None:
            if self.bias_vl is None:
                raise ValueError("image_mask requires vision_enabled=True when constructing MegaGate")
            if (image_mask.shape != hidden_states.shape[:-1] or image_mask.dtype != torch.bool
                    or image_mask.device != hidden_states.device):
                raise ValueError("image_mask must be BOOL [batch, sequence] on the hidden_states device")
        logits, weights, indices = self._native_forward(hidden_states, image_mask)
        weights, indices = self._finish_routing(
            hidden_states, logits, weights, indices, token_mask, sequence_ids,
            num_sequences, sequence_partition_groups,
        )
        return logits, weights, indices

    @classmethod
    def from_module(cls, module: nn.Module, **configuration: Any) -> MegaGateModule:
        """Retain declared settings without configuration overrides.

        Args:
            module: Existing public MegaGate source.
            **configuration: Overrides are rejected; construct a new configured gate instead.
        """
        if configuration:
            raise ValueError("family source construction preserves source settings; overrides are unsupported")
        return cls(module=module)


@module_replacement
def Gate(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "",
         context: Any = None, mesh: Any = None, implementation: str = "local") -> nn.Module:
    """Build a local gate or the compatible MegaGate execution variant."""
    return build_family(Gate, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, implementation=implementation)


register_module_family(ModuleFamily(
    name="gate", baseline=Gate,
    variants={"local": "hyper_parallel.components.modules.moe.router.GateModule",
              "mega": "hyper_parallel.components.modules.moe.router.MegaGateModule"},
    default_variant="local", sharding=router_sharding,
    selector_specs={"implementation": SelectorSpec(str, "local")},
    variant_rules=(VariantRule({"implementation": "mega", "parallel": False}, "mega"),),
))


class ScoreCorrectedTopKRouterModule(nn.Module):
    """Own selection biases, source matrix identities and the FP32 unbiased routing weights."""

    def __init__(
        self,
        *,
        config: Any = None,
        module: nn.Module | None = None,
        module_fqn: str = "",
        context: Any = None,
        source_contract: str = "ids_weights",
    ) -> None:
        """Construct explicit routing settings or adopt a separate matrix/bias source."""
        super().__init__()
        del module_fqn, context
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        if source_contract not in ("ids_weights", "logits_weights_ids"):
            raise ValueError("Unsupported corrected-router source contract")
        source = config if module is None else module
        self.source_contract = source_contract
        self.hidden_size = config.hidden_size if module is None else module.weight.shape[1]
        self.num_experts = config.num_experts if module is None else module.weight.shape[0]
        self.top_k = source.top_k
        self.scoring_func = getattr(source, "moe_router_activation_func", getattr(source, "scoring_func", None))
        self.num_groups = getattr(source, "num_expert_group", getattr(source, "num_groups", 1))
        self.groups_selected = getattr(source, "topk_group", getattr(source, "groups_selected", 1))
        self.normalize_topk = getattr(source, "moe_renormalize", getattr(source, "normalize_topk", True))
        self.routed_scaling_factor = source.routed_scaling_factor
        self._validate_settings()
        if module is None:
            self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
            self.e_score_correction_bias = nn.Parameter(torch.zeros(self.num_experts), requires_grad=False)
            self.reset_parameters()
        else:
            if set(dict(module.named_parameters(recurse=False))) != {"weight", "e_score_correction_bias"}:
                raise ValueError("Corrected router source must own its matrix and correction Parameter separately")
            if list(module.named_children()) or module.e_score_correction_bias.shape != (self.num_experts,):
                raise ValueError("Unsupported corrected router source structure")
            self.weight, self.e_score_correction_bias = module.weight, module.e_score_correction_bias
            self.training = module.training

    def _validate_settings(self) -> None:
        """Validate grouped top-k coverage, scoring and normalization before weight allocation."""
        for name in ("hidden_size", "num_experts", "top_k", "num_groups", "groups_selected"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.num_experts % self.num_groups
            or self.groups_selected > self.num_groups
            or self.top_k > self.groups_selected * (self.num_experts // self.num_groups)
        ):
            raise ValueError("Router groups do not cover the requested top-k experts")
        if self.num_groups > self.groups_selected and self.num_experts // self.num_groups < 2:
            raise ValueError("Top-two group scoring needs at least two experts in each group")
        if self.scoring_func not in ("softmax", "sigmoid") or not isinstance(self.normalize_topk, bool):
            raise ValueError("Corrected routing requires sigmoid/softmax and a boolean normalization policy")
        if not isinstance(self.routed_scaling_factor, (int, float)) or not isfinite(self.routed_scaling_factor):
            raise ValueError("routed_scaling_factor must be finite")

    def reset_parameters(self) -> None:
        """Restore the source constructor matrix law and a defined neutral selection bias."""
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        nn.init.zeros_(self.e_score_correction_bias)

    def make_transforms(self) -> list:
        """Retain both source keys and their independent parameter identities."""
        return []

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Select groups with biased scores and return normalized unbiased expert weights."""
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("Router inputs must end in hidden_size and include token dimensions")
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            # Torch exposes this C-extension callable without a Python signature.
            logits = F.linear(  # pylint: disable=not-callable
                hidden_states.reshape(-1, self.hidden_size).float(), self.weight.float()
            )
            scores = routing_scores(logits, self.scoring_func)
        selection = scores + self.e_score_correction_bias.float().unsqueeze(0)
        if self.num_groups > self.groups_selected:
            group_scores = selection.view(-1, self.num_groups, self.num_experts // self.num_groups)
            group_scores = group_scores.topk(2, dim=-1).values.sum(dim=-1)
            selected = group_scores.topk(self.groups_selected, dim=-1, sorted=False).indices
            mask = torch.zeros_like(group_scores).scatter_(1, selected, 1)
            mask = mask.unsqueeze(-1).expand(-1, -1, self.num_experts // self.num_groups).reshape_as(selection)
            selection = selection.masked_fill(~mask.bool(), float("-inf"))
        indices = selection.topk(self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        if self.top_k > 1 and self.normalize_topk:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        return (indices, weights) if self.source_contract == "ids_weights" else (logits, weights, indices)


def corrected_router_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Replicate router matrices and preserve the source's local token ownership."""
    del module_fqn, member
    axes = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    tokens = dict(axes)
    rows = dict(axes)
    for axis in ("cp", "tp"):
        if context.axis_sizes[axis] > 1 and (axis == "cp" or context.sequence_parallel):
            tokens[axis], rows[axis] = Shard(1), Shard(0)
    outputs = ("indices", "weights") if module.source_contract == "ids_weights" else ("logits", "weights", "indices")
    return ModuleShardingSpec(
        params={name: dict(axes) for name, _ in module.named_parameters()},
        in_src={"hidden_states": dict(tokens)},
        in_dst={"hidden_states": dict(tokens)},
        out_src={name: dict(rows) for name in outputs},
        out_dst={name: dict(rows) for name in outputs},
        out_names=outputs,
        is_boundary=True,
    )


@module_replacement
def ScoreCorrectedTopKRouter(  # pylint: disable=invalid-name
    *,
    config: Any = None,
    module: nn.Module | None = None,
    module_fqn: str = "",
    context: Any = None,
    mesh: Any = None,
    source_contract: str = "ids_weights",
) -> nn.Module:
    """Build complete group-aware corrected routing through the shared family lifecycle."""
    return build_family(
        ScoreCorrectedTopKRouter,
        config=config,
        module=module,
        module_fqn=module_fqn,
        context=context,
        mesh=mesh,
        source_contract=source_contract,
    )


register_module_family(
    ModuleFamily(
        name="score_corrected_topk_router",
        baseline=ScoreCorrectedTopKRouter,
        variants={"local": "hyper_parallel.components.modules.moe.router.ScoreCorrectedTopKRouterModule"},
        default_variant="local",
        sharding=corrected_router_sharding,
    )
)


__all__ = [
    "Gate", "GateModule", "MegaGateModule", "ScoreCorrectedTopKRouter", "ScoreCorrectedTopKRouterModule",
    "balanced_expert_indices", "routing_scores", "select_topk_scores",
]
