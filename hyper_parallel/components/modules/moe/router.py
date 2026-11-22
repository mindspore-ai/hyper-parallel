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
"""One Gate family owning routing and optional training state."""

from __future__ import annotations
from collections.abc import Mapping
from functools import lru_cache
from math import isfinite, prod
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.components.modules.family import SelectorSpec, VariantRule
from hyper_parallel.components.modules.moe.moe_utils import (
    local_router_metadata,
    router_sharding,
    validate_router_layout,
)
from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale
from hyper_parallel.components.losses.moe_aux_loss import sequence_load_balancing_loss
from hyper_parallel.distributed.module_binding import ParallelBinding, ParallelModuleSpec


def _setting(sources: tuple, names: tuple[str, ...], default: Any) -> Any:
    for source in sources:
        for name in names:
            value = source.get(name) if isinstance(source, Mapping) else getattr(source, name, None)
            if value is not None:
                return value
    return default



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

    Projection still follows the enclosing Torch autocast policy, as the model
    source does. Optional training state is shared with the token-choice Router;
    fused execution belongs to the MegaGate variant.
    """

    def __init__(self, *, config: Any = None, module: nn.Module | None = None,
                 context: Mapping[str, Any] | None = None) -> None:
        """Construct a gate or retain a public member and its optional training state."""
        super().__init__()
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        if module is not None:
            self._validate_source(module)
        source = config if config is not None else module
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

    def make_transforms(self) -> list:
        """Declare that this member keeps parameter names and layouts unchanged.

        Returns:
            An empty converter list, matching members whose initialization is
            fully owned by ``reset_parameters``.
        """
        return []

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
        """Resolve router options and initialize parallel routing state."""
        sources = (*tuple((context or {}).get("module_configs", ())), source)
        self.fix_router = _setting(
            sources, ("moe_router_force_load_balancing", "fix_router", "router_fix_router"), False,
        )
        if not isinstance(self.fix_router, bool):
            raise ValueError("moe_router_force_load_balancing must be bool")
        for name, aliases in (
                ("aux_loss_coeff", ("moe_aux_loss_coeff", "aux_loss_coeff", "router_aux_loss_coeff")),
        ):
            value = _setting(sources, aliases, 0.0)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
                raise ValueError(f"{aliases[0]} must be finite and nonnegative")
            setattr(self, name, float(value))
        self.last_aux_loss = None
        self.router_ep_size = 1
        self.router_tp_mesh = None
        self.sequence_partition_groups = ()
        self.expert_bias_update_groups = None


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
        """Attach auxiliary gradients to the selected routes."""
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



@lru_cache(maxsize=None)
def _plan_for_device(device: torch.device) -> Any:
    # HyperMulticore is optional and its runtime is needed only for native NPU calls.
    from hyper_multicore.modules.mega_gate.plan import build_mega_gate_plan  # pylint: disable=import-outside-toplevel

    return build_mega_gate_plan(device)



class MegaGateModule(GateModule):
    """Extend the ordinary gate with correction biases and a fused NPU route.

    The Gate base owns the projection and common routing contract. HyperMulticore
    supplies the native function and plan, without owning this module's state.
    """


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
        for name in ("bias", "bias_vl"):
            if name == "bias_vl" and not vision_enabled:
                self.register_parameter(name, None)
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
        """Reset the common projection and MegaGate-specific correction state."""
        super().reset_parameters()
        nn.init.zeros_(self.bias)
        if self.bias_vl is not None:
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

    def make_transforms(self) -> list:
        """Declare that this member keeps parameter names and layouts unchanged.

        Returns:
            An empty converter list, matching members whose initialization is
            fully owned by ``reset_parameters``.
        """
        return []


    def _apply(self, fn: Any, recurse: bool = True) -> MegaGateModule:
        """Preserve declared FP32 parameters and gradients during conversion."""
        # Keep correction values and their gradients FP32 through model dtype conversion.
        protected = {id(parameter) for parameter in (self.bias, self.bias_vl) if parameter is not None}
        protected.update(id(parameter.grad) for parameter in self.parameters() if parameter.grad is not None
                         and id(parameter) in protected)

        def apply(tensor: torch.Tensor) -> torch.Tensor:
            converted = fn(tensor)
            if id(tensor) not in protected or converted.dtype == torch.float32:
                return converted
            return converted.float() if tensor.is_meta else tensor.to(device=converted.device, dtype=torch.float32)

        return super()._apply(apply, recurse=recurse)


    def _torch_forward(self, hidden_states: torch.Tensor,
                       image_mask: torch.Tensor | None, **metadata: Any) -> tuple[torch.Tensor, ...]:
        """Keep fused vision selection local to this variant; training counts stay standard."""
        if image_mask is None:
            return super().forward(hidden_states, correction_bias=self.bias, **metadata)
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
        # Only the fused execution path requires the optional native package.
        from hyper_multicore.modules.mega_gate.function import mega_gate  # pylint: disable=import-outside-toplevel

        return mega_gate(hidden_states, self.weight, self.bias, _plan_for_device(hidden_states.device),
                         vision_bias=self.bias_vl if image_mask is not None else None,
                         image_mask=None if image_mask is None else image_mask.view(-1),
                         top_k=self.top_k, routed_scaling_factor=self.routed_scaling_factor)

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
        if self.scoring_func != "sqrtsoftplus" or hidden_states.device.type != "npu" or hidden_states.numel() == 0:
            return self._torch_forward(hidden_states, image_mask, token_mask=token_mask,
                                       sequence_ids=sequence_ids, num_sequences=num_sequences,
                                       sequence_partition_groups=sequence_partition_groups)
        if image_mask is not None and not image_mask.is_contiguous():
            raise ValueError("image_mask must be contiguous for native MegaGate")
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



__all__ = [
    "Gate", "GateModule", "MegaGateModule",
    "balanced_expert_indices", "routing_scores", "select_topk_scores",
]
