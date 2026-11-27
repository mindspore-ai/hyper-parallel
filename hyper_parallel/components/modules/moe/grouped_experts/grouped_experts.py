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
"""Expert parameter construction and staged algorithm phases, independent of EP resources."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.activations import SiLUActivation

from hyper_parallel.components.checkpoint.weight_conversion import Transpose, WeightConverter
from hyper_parallel.components.functional.swiglu import swiglu_activation
from hyper_parallel.components.ops import grouped_matmul, moe_token_permute, moe_token_unpermute, swiglu
from hyper_parallel.core.dtensor.dtensor import DTensor


def _normalize_swiglu_limit(limit: Any) -> float | None:
    """Validate a source/config clamp and normalize disabled values once at construction."""
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit):
        raise ValueError("swiglu_limit/limit must be None or a finite number")
    if limit > 0:
        return limit
    warnings.warn(
        f"swiglu_limit/limit={limit} <= 0 disables clamping; using unclamped SwiGLU.",
        UserWarning, stacklevel=4,
    )
    return None


@dataclass(frozen=True)
class GroupedExpertsConfig:
    """Bias-free SwiGLU dimensions, probability placement and model initialization.

    Attributes:
        num_experts: Global expert count, before any EP parameter sharding.
        hidden_size: Input and output width.
        intermediate_size: Width of each gate/up half.
        routing_weight_mode: Apply final model-selected probabilities at this stage only.
        init_method_std: Base std for normal initialization, matching Megatron's 0.02.
        num_layers: Required global Transformer depth for FC2 scaling, before PP partitioning.
        activation: Optional gate/up activation supplied by the model; defaults to SwiGLU.
        top_k: Static routing width required by fused executors; optional for ordinary GMM.
        swiglu_limit: Positive pre-SiLU clamp in FP32; None disables it, as do nonpositive values with a warning.
        activation_dtype: SwiGLU precision when no clamp or custom activation is configured.
    """

    num_experts: int
    hidden_size: int
    intermediate_size: int
    routing_weight_mode: Literal["output", "before_fc2"] = "before_fc2"
    init_method_std: float = 0.02
    num_layers: int | None = None
    activation: Callable[[Tensor, Tensor], Tensor] | None = None
    top_k: int | None = None
    swiglu_limit: float | None = None
    activation_dtype: Literal["input", "float32"] = "input"

    def __post_init__(self) -> None:
        if self.num_layers is None:
            raise ValueError(
                "num_layers is required for depth-scaled expert initialization; pass "
                "GroupedExpertsConfig(num_layers=...) or context={'num_layers': ...} when adopting a module. "
                "Use num_layers=1 explicitly for a standalone layer."
            )
        for name in ("num_experts", "hidden_size", "intermediate_size", "num_layers"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.routing_weight_mode not in ("output", "before_fc2"):
            raise ValueError("routing_weight_mode must be output or before_fc2")
        if self.activation_dtype not in ("input", "float32"):
            raise ValueError("activation_dtype must be input or float32")
        top_k_type = type(self.top_k)
        if self.top_k is not None and (top_k_type is not int or not 1 <= self.top_k <= self.num_experts):
            raise ValueError("top_k must be a positive integer no greater than num_experts")
        object.__setattr__(self, "swiglu_limit", _normalize_swiglu_limit(self.swiglu_limit))
        if self.activation is not None and self.swiglu_limit is not None:
            raise ValueError("declare either a custom activation or swiglu_limit")


_EXPERT_CONFIG_KEYS = {
    "num_experts": ("num_experts", "num_local_experts", "n_routed_experts"),
    "hidden_size": ("hidden_dim", "hidden_size"),
    "intermediate_size": ("intermediate_dim", "moe_intermediate_size", "intermediate_size"),
    "init_method_std": ("init_method_std", "initializer_range"),
    "hidden_act": ("hidden_act", "hidden_activation", "mlp_hidden_act"),
    "top_k": ("num_experts_per_tok", "top_k", "moe_topk"),
    "swiglu_limit": ("swiglu_limit", "limit"),
}
_EXPERT_INPUT_KEYS = {
    "top_k_index": "topk_ids", "topk_indices": "topk_ids", "selected_experts": "topk_ids",
    "router_indices": "topk_ids", "topk_idx": "topk_ids",
    "top_k_weights": "topk_weights", "routing_weights": "topk_weights", "topk_weight": "topk_weights",
}


def _match_expert_config_value(configs: tuple[Any, ...], name: str, default: Any = None) -> Any:
    """Read recognized source aliases in nearest-config order without copying config objects."""
    for config in configs:
        for key in _EXPERT_CONFIG_KEYS[name]:
            value = config.get(key) if isinstance(config, Mapping) else getattr(config, key, None)
            if value is not None:
                return value
    return default


def match_grouped_experts_inputs(
    topk_ids: Tensor | None, topk_weights: Tensor | None, kwargs: Mapping[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """Normalize recognized top-k keyword aliases without changing routing semantics.

    Args:
        topk_ids: Expert IDs passed positionally or under the canonical name.
        topk_weights: Final probabilities passed positionally or under the canonical name.
        kwargs: Alternate ID/probability names from supported source forward contracts.

    Returns:
        Expert IDs and final probabilities; both retain their original Tensor identity.

    Raises:
        TypeError: An input is missing, duplicated or uses an unknown keyword.
    """
    values = {"topk_ids": topk_ids, "topk_weights": topk_weights}
    seen = set()
    for key, value in kwargs.items():
        name = _EXPERT_INPUT_KEYS.get(key)
        if name is None:
            raise TypeError(f"unexpected grouped expert input keyword {key!r}")
        if values[name] is not None or name in seen:
            raise TypeError(f"multiple values for grouped expert input {name!r}")
        values[name] = value
        seen.add(name)
    if any(value is None for value in values.values()):
        raise TypeError("grouped experts require both top-k expert IDs and routing weights")
    return values["topk_ids"], values["topk_weights"]


def _packed_parameters(module, module_fqn, configs, activation):
    """Validate packed source parameters and their activation contract."""
    parameters = dict(module.named_parameters())
    if set(parameters) != {"gate_up_proj", "down_proj"} or list(module.named_buffers()):
        raise ValueError(f"{module_fqn}: expected canonical experts or two bias-free packed projections")
    if any(weight.ndim != 3 or isinstance(weight, DTensor) or weight.grad is not None
           for weight in parameters.values()):
        raise ValueError(f"{module_fqn}: packed experts must be converted before sharding and backward")
    act_fn = getattr(module, "act_fn", None)
    hidden_act = _match_expert_config_value(configs, "hidden_act", "silu")
    if activation is None and (hidden_act != "silu" or (
            act_fn is not None and type(act_fn) not in (nn.SiLU, SiLUActivation) and act_fn is not F.silu)):
        raise ValueError(f"{module_fqn}: supply the activation for packed experts with non-SwiGLU math")
    return parameters


def _packed_dimensions(module, module_fqn, parameters, configs):
    """Infer expert dimensions and validate the source projection layouts."""
    gate_up, down = parameters["gate_up_proj"], parameters["down_proj"]
    experts, hidden, width = down.shape
    sources = (module, *configs)
    source_dimensions = (
        _match_expert_config_value(sources, "num_experts", experts),
        _match_expert_config_value(sources, "hidden_size", hidden),
        _match_expert_config_value(sources, "intermediate_size", width),
    )
    if source_dimensions != (experts, hidden, width) or gate_up.shape != (experts, 2 * width, hidden):
        raise ValueError(f"{module_fqn}: expected packed gate_up_proj [E,2I,H] and down_proj [E,H,I]")
    return experts, hidden, width


class GroupedExpertsModule(nn.Module):
    """Internal family member using selectable grouped-GEMM and activation operators.

    Local transport is identity. EP members reuse this math over owner-local slots;
    router, shared experts, auxiliary loss and model masks remain outside this module.
    """

    shared_expert_overlap_supported = True

    def __init__(
        self, *, config: GroupedExpertsConfig | None = None, module: nn.Module | None = None,
        module_fqn: str = "", context: Mapping[str, Any] | None = None,
        activation: Callable[[Tensor, Tensor], Tensor] | None = None,
        activation_dtype: str = "input", swiglu_limit: float | None = None,
    ) -> None:
        """Create global parameters or adopt an unsharded expert source.

        Args:
            config: Global algorithm facts; device/dtype follow the caller's Torch context.
            module: Existing family member or compatible packed expert source.
            module_fqn: Source module path used in diagnostics.
            context: Job facts and source/ancestor configuration; the replacement builder
                supplies global num_layers for FC2 initialization. Standalone source adoption
                requires an explicit context['num_layers'] when the source has no canonical config.
            activation: Model-owned gate/up activation for a packed source.
            activation_dtype: SwiGLU precision when adopting a packed source without a canonical config.
            swiglu_limit: Explicit clamp for a packed source without a canonical config; None reads the source.

        Note:
            Fresh construction keeps nn.Module's default training=True. Source
            construction explicitly preserves module.training, including eval mode.
        """
        super().__init__()
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        if module is not None:
            source_config = getattr(module, "config", None)
            if not isinstance(source_config, GroupedExpertsConfig):
                configs = tuple((context or {}).get("module_configs", ())) or (source_config,)
                parameters = _packed_parameters(module, module_fqn, configs, activation)
                experts, hidden, intermediate = _packed_dimensions(module, module_fqn, parameters, configs)
                if swiglu_limit is None and activation is None:
                    swiglu_limit = _match_expert_config_value((module, *configs), "swiglu_limit")
                source_config = GroupedExpertsConfig(
                    experts, hidden, intermediate, routing_weight_mode="output", activation=activation,
                    activation_dtype=activation_dtype,
                    num_layers=(context or {}).get("num_layers"),
                    init_method_std=_match_expert_config_value(configs, "init_method_std", 0.02),
                    top_k=_match_expert_config_value((module, *configs), "top_k"),
                    swiglu_limit=swiglu_limit,
                )
            self.config = source_config
            self.gate_up_proj = module.gate_up_proj
            self.down_proj = module.down_proj
            self.training = module.training
            for name, child in module.named_children():
                self.add_module(name, child)
        else:
            if not isinstance(config, GroupedExpertsConfig):
                raise ValueError("config must be GroupedExpertsConfig")
            self.config = config
            self.gate_up_proj = nn.Parameter(torch.empty(config.num_experts, 2 * config.intermediate_size,
                                                         config.hidden_size))
            self.down_proj = nn.Parameter(torch.empty(config.num_experts, config.hidden_size,
                                                      config.intermediate_size))
            self.reset_parameters()

    @property
    def limit(self) -> float:
        """Expose the resolved source clamp to model-owned expert composition."""
        return self.config.swiglu_limit or 0.0

    def apply_activation(self, gate: Tensor, up: Tensor, dtype: torch.dtype) -> Tensor:
        """Apply the custom activation or configured SwiGLU to the first projection."""
        if self.config.activation is not None:
            return self.config.activation(gate, up).to(dtype)
        precision = "float32" if self.config.swiglu_limit is not None else self.config.activation_dtype
        return swiglu_activation(gate, up, activation_dtype=precision, limit=self.config.swiglu_limit or 0.0).to(dtype)

    def reset_parameters(self) -> None:
        """Initialize both projections normally, using Megatron's depth scaling for FC2."""
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=self.config.init_method_std)
        output_std = self.config.init_method_std / math.sqrt(2.0 * self.config.num_layers)
        nn.init.normal_(self.down_proj, mean=0.0, std=output_std)

    def make_transforms(self) -> list[WeightConverter]:
        """Normalize HF expert projections to ``[E,2I,H]`` and ``[E,H,I]``.

        Model-owned HF mappings first merge individual gate/up/down experts
        (including DeepSeek's w1/w3/w2 names) and normalize legacy Qwen-VL
        layouts. Replacement conversion preserves that canonical layout and
        transposes input-major packed projections only when shapes differ.
        Already-canonical weights retain their values and storage.

        Returns:
            Converters for the packed projections; the builder assigns their module scope.
        """
        return [
            WeightConverter(
                source_patterns=name, target_patterns=name,
                operations=[Transpose(dim0=-2, dim1=-1, check_dims=True)],
            )
            for name in ("gate_up_proj", "down_proj")
        ]

    def validate_and_flatten_inputs(self, hidden_states: Tensor, topk_ids: Tensor, topk_weights: Tensor) -> Tensor:
        """Validate shapes/devices/dtypes and flatten tokens without detaching router weights.

        IDs must be in [0,E); fixed-size slot counting rejects out-of-range/drop IDs
        on the input device. Zero probability still represents an executed slot.

        Args:
            hidden_states: Original floating-point tokens [...,H].
            topk_ids: Global expert IDs [T,K].
            topk_weights: Final differentiable model probabilities [T,K].

        Returns:
            Flattened source tokens [T,H]; routes remain in the caller's local variables.
        """
        if hidden_states.ndim < 1 or hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError("hidden_states must have shape [..., hidden_size]")
        hidden = hidden_states.reshape(-1, self.config.hidden_size)
        if topk_ids.ndim != 2 or topk_ids.shape[0] != hidden.shape[0] or topk_ids.shape[1] < 1:
            raise ValueError("topk_ids must have shape [flattened_tokens, positive_top_k]")
        if topk_weights.shape != topk_ids.shape:
            raise ValueError("topk_weights must have the same [T,K] shape as topk_ids")
        if topk_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("topk_ids must have integer int32/int64 dtype")
        if not hidden_states.is_floating_point() or not topk_weights.is_floating_point():
            raise ValueError("hidden_states and topk_weights must have floating-point dtype")
        if hidden.device != topk_ids.device or hidden.device != topk_weights.device:
            raise ValueError("tokens, expert IDs and routing weights must be on the same device")
        return hidden

    def permute(
        self, hidden_states: Tensor, topk_ids: Tensor, topk_weights: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor]:
        """Group tokens by expert and align slot metadata through the shared inverse map.

        Args:
            hidden_states: Validated source tokens [T,H].
            topk_ids: Global selected expert IDs [T,K].
            topk_weights: Final differentiable routing weights [T,K].

        Returns:
            Expert-major hidden [S,H], IDs [S], counts [E], optional weights [S],
            and slot positions [S], where S=T*K.
        """
        ids = topk_ids.reshape(-1).to(torch.int64)
        # Fixed-size counting also rejects invalid IDs without host scalar reads.
        counts = ids.new_zeros(self.config.num_experts).scatter_add_(0, ids, torch.ones_like(ids))
        hidden, order, unpermute_indices = moe_token_permute(hidden_states.contiguous(), topk_ids.contiguous())
        ids = ids[order]
        weights = topk_weights.reshape(-1)[order] if self.config.routing_weight_mode == "before_fc2" else None
        return hidden, ids, counts, weights, unpermute_indices

    def dispatch(
        self, hidden_states: Tensor, expert_ids: Tensor, tokens_per_expert: Tensor,
        routing_weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, list[int] | None, list[int] | None, Tensor | None]:
        """Keep source slots and counts for local execution.

        Args:
            hidden_states: Source expert-major activations [S,H].
            expert_ids: Global expert IDs [S], unused by local transport.
            tokens_per_expert: Source expert counts [E].
            routing_weights: Optional before_fc2 probabilities [S].

        Returns:
            Activations, counts, probabilities, then three None values for absent EP transport.
        """
        del expert_ids
        return hidden_states, tokens_per_expert, routing_weights, None, None, None

    def compute_experts(
        self, hidden_states: Tensor, tokens_per_expert: Tensor, gate_up_proj: Tensor, down_proj: Tensor,
        routing_weights: Tensor | None = None,
    ) -> Tensor:
        """Apply shared SwiGLU mathematics to an explicitly supplied compute table.

        Args:
            hidden_states: Expert-major activations [R,H].
            tokens_per_expert: Counts [C] indexing the supplied weight rows.
            gate_up_proj: Materialized FC1 weights [C,2I,H], including any per-call replicas.
            down_proj: Materialized FC2 weights [C,H,I] with matching row order.
            routing_weights: Optional differentiable before_fc2 probabilities [R].

        Returns:
            One hidden-size output per slot; transport and replica gradient return are external.
        """
        gate_up, down = gate_up_proj, down_proj
        # GMM consumes local tensors; storage unshard/prefetch is owned by FSDP hooks.
        gate_up = gate_up.to_local() if isinstance(gate_up, DTensor) else gate_up
        down = down.to_local() if isinstance(down, DTensor) else down
        if (gate_up.ndim != 3 or gate_up.shape[0] < 1
                or gate_up.shape[1:] != (2 * self.config.intermediate_size, self.config.hidden_size)
                or down.shape != (gate_up.shape[0], self.config.hidden_size, self.config.intermediate_size)):
            raise ValueError("expert compute requires complete [C,2I,H] and [C,H,I] matrices; ETP is unsupported")
        if tokens_per_expert.shape != (gate_up.shape[0],):
            raise ValueError("tokens_per_expert must describe the current expert compute table")
        hidden = hidden_states
        group_list = tokens_per_expert.to(device=hidden.device, dtype=torch.int64).cumsum(0)
        # The kernel uses input-major projections while checkpoints store linear weights.
        gate_up, down = gate_up.transpose(-2, -1), down.transpose(-2, -1)
        # Empty ranks avoid unsupported empty GMM calls while retaining gradients to both weights.
        fc1 = (grouped_matmul(hidden, gate_up, group_list=group_list)  # pylint: disable=not-callable
               if hidden.shape[0] else hidden @ gate_up[0])
        if (self.config.swiglu_limit is not None or self.config.activation is not None
                or self.config.activation_dtype != "input"):
            activated = self.apply_activation(*fc1.chunk(2, dim=-1), hidden.dtype)
        else:
            activated = swiglu(fc1)
        if self.config.routing_weight_mode == "before_fc2":
            if routing_weights is None:
                raise ValueError("before_fc2 requires differentiable slot routing weights")
            activated = (activated * routing_weights.unsqueeze(-1)).to(activated.dtype)
        return (grouped_matmul(activated, down, group_list=group_list)  # pylint: disable=not-callable
                if hidden.shape[0] else activated @ down[0])

    def combine(
        self, values: Tensor, send_splits: list[int] | None, recv_splits: list[int] | None, recv_inverse: Tensor | None,
    ) -> Tensor:
        """Keep the same output slots for local execution.

        Args:
            values: Computed expert outputs in slot order.
            send_splits: Unused local transport argument.
            recv_splits: Unused local transport argument.
            recv_inverse: Unused local transport argument.

        Returns:
            The original slot outputs [S,H].
        """
        del send_splits, recv_splits, recv_inverse
        return values

    def unpermute(
        self, values: Tensor, unpermute_indices: Tensor, topk_weights: Tensor, output_shape: tuple[int, ...],
    ) -> Tensor:
        """Restore tokens and sum top-k slots, applying output probabilities once.

        Args:
            values: Outputs restored to source slot order by combine.
            unpermute_indices: This call's inverse NPU mapping [S].
            topk_weights: Original token/top-k routing weights [T,K].
            output_shape: Original hidden-state shape [...,H].

        Returns:
            Top-k combined output in the original token order and shape [...,H].
        """
        if values.shape != (unpermute_indices.numel(), self.config.hidden_size):
            raise ValueError("combine must return one hidden-size output per source slot")
        if not values.shape[0]:
            if self.config.routing_weight_mode == "output":
                values = values * topk_weights.to(values.dtype).reshape(-1, 1)
            return values.reshape(output_shape)
        # Ones retain top-k summation without reweighting an already weighted FC2 output.
        probs = topk_weights if self.config.routing_weight_mode == "output" else torch.ones_like(topk_weights)
        output = moe_token_unpermute(
            values.contiguous(), unpermute_indices, probs.contiguous(),
        )
        return output.to(values.dtype).reshape(output_shape)

    def forward(
        self, hidden_states: Tensor, topk_ids: Tensor | None = None, topk_weights: Tensor | None = None,
        **kwargs: Tensor,
    ) -> Tensor:
        """Consume model-selected routes; router, shared experts and aux loss stay with MoE.

        Args:
            hidden_states: Source tokens [...,H].
            topk_ids: Global already-selected expert IDs [T,K].
            topk_weights: Final differentiable routing weights [T,K].
            **kwargs: Recognized source aliases, including top_k_index/top_k_weights.

        Returns:
            Routed output [...,H] in the source token order.
        """
        topk_ids, topk_weights = match_grouped_experts_inputs(topk_ids, topk_weights, kwargs)
        output_shape = tuple(hidden_states.shape)
        hidden_states = self.validate_and_flatten_inputs(hidden_states, topk_ids, topk_weights)
        hidden, ids, counts, weights, unpermute_indices = self.permute(hidden_states, topk_ids, topk_weights)
        hidden, counts, weights, send_splits, recv_splits, recv_inverse = self.dispatch(
            hidden, ids, counts, weights,
        )
        values = self.compute_experts(hidden, counts, self.gate_up_proj, self.down_proj, weights)
        values = self.combine(values, send_splits, recv_splits, recv_inverse)
        return self.unpermute(values, unpermute_indices, topk_weights, output_shape)
