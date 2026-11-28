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
"""Low-precision grouped-expert members reusing staged GroupedExpertsModule, EP, and MoonEP."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hyper_parallel.components.modules.moe.grouped_experts.ep.ep_grouped_experts import EPGroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts.ep.moonep_grouped_experts import MoonEPGroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import (
    GroupedExpertsConfig,
    GroupedExpertsModule,
)
from hyper_parallel.components.quantization.config import (
    GROUPED_NDIM,
    GROUP_LIST_COUNTS,
    LowPrecisionConfig,
    LowPrecisionDtypeScheme,
    MATRIX_NDIM,
    MX_BLOCK_SIZE_32,
)
from hyper_parallel.components.quantization.functional import build_low_precision_strategy
from hyper_parallel.components.quantization.functional.base_gmm_func import (
    GroupedLinear,
    _GroupedLinearFunction,
)
from hyper_parallel.components.quantization.functional.fake_w4a8_gmm_func import FakeW4A8GroupedLinear
from hyper_parallel.components.quantization.functional.hifloat8_gmm_func import HiFloat8GroupedLinear
from hyper_parallel.components.quantization.functional.mxfp8_fused_grouped_swiglu import fused_grouped_swiglu
from hyper_parallel.components.quantization.functional.mxfp8_projection import mxfp8_projection
from hyper_parallel.components.quantization.functional.strategy_factory import _validate_tile_alignment
from hyper_parallel.components.quantization.ops.npu_mxfp8 import (
    LowPrecisionCapabilityError,
    validate_npu_gmm_runtime,
    validate_npu_swiglu_runtime,
)
from hyper_parallel.components.quantization.ops.npu_w4a8 import W4A8CapabilityError
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.models.replacement import module_replacement

_EXPERT_PARAMETER_NAMES = ("gate_up_proj", "down_proj")
_GATE_UP_BRANCHES = 2
_LP_MODES = ("mxfp8", "fused_mxfp8", "w8a8", "w4a8", "fake_w4a8", "hif8")


def _scheme_for_mode(lp_mode: str, context: Mapping[str, Any]) -> LowPrecisionDtypeScheme:
    """Resolve the LowPrecisionDtypeScheme corresponding to a named family variant."""
    configured = context.get("low_precision")
    block_size = getattr(configured, "block_size", MX_BLOCK_SIZE_32)
    if lp_mode == "w8a8":
        return LowPrecisionDtypeScheme(weight_format="mxfp8", act_format="mxfp8", block_size=block_size)
    if lp_mode == "w4a8":
        return LowPrecisionDtypeScheme(weight_format="mxfp4", act_format="mxfp8", block_size=block_size)
    if lp_mode == "fake_w4a8":
        return LowPrecisionDtypeScheme(
            weight_format="mxfp4", act_format="mxfp8", is_fake_quantize=True, block_size=block_size,
        )
    if lp_mode == "hif8":
        return LowPrecisionDtypeScheme(weight_format="hif8", act_format="hif8")
    if isinstance(configured, LowPrecisionDtypeScheme):
        return configured
    raise ValueError(f"Unsupported low-precision grouped-expert mode: {lp_mode!r}")


def _is_deepseek_v4_experts(module: nn.Module) -> bool:
    """Identify DeepSeek-V4/V4.1 packed experts without eagerly importing Transformers models."""
    cls = type(module)
    return cls.__name__ == "DeepseekV4Experts" and cls.__module__.startswith("transformers.models.deepseek_v4.")


def _validate_custom_gate_source(module: nn.Module | None, prefix: str) -> None:
    """Reject non-DeepSeek custom _apply_gate hooks before adopting source parameters."""
    if module is None or isinstance(module, GroupedExpertsModule) or _is_deepseek_v4_experts(module):
        return
    apply_gate = getattr(module, "_apply_gate", None)
    apply_gate_fn = getattr(apply_gate, "__func__", apply_gate)
    if apply_gate_fn is not None and getattr(apply_gate_fn, "__name__", "") != "_default_apply_gate":
        raise ValueError(f"{prefix}: custom _apply_gate requires a dedicated replacement")


class _LowPrecisionExpertsMixin:
    """Override only ``compute_experts`` while reusing staged routing, EP, MoonEP, and Overlap."""

    config: GroupedExpertsConfig
    gate_up_proj: nn.Parameter
    down_proj: nn.Parameter
    grouped_linear: GroupedLinear | None
    quantizer: MXFP8Quantizer | None
    clamp_limit: float | None
    fqn: str
    _lp_mode: str
    _preserve_policy_routing: bool

    @property
    def _expert_cfg(self) -> GroupedExpertsConfig:
        """Return canonical expert dimensions/options even when Policy preserves source.config."""
        return getattr(self, "_policy_expert_config", None) or self.config

    @staticmethod
    def _validate_grouped_linear(grouped_linear: GroupedLinear | None) -> None:
        """Require a non-Module GroupedLinear strategy so parameter registration stays clean."""
        if isinstance(grouped_linear, nn.Module) or not isinstance(grouped_linear, GroupedLinear):
            raise TypeError("grouped_linear must be a non-Module GroupedLinear instance")

    def _init_low_precision_backend(
        self,
        *,
        lp_mode: str,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        strategy: GroupedLinear | None = None,
    ) -> None:
        """Validate shape/runtime constraints and bind the selected low-precision compute backend."""
        effective_context = context or {}
        prefix = f"{module_fqn or '<root>'}: {type(self).__name__}"
        cfg = self._expert_cfg
        self.fqn = module_fqn
        self._lp_mode = lp_mode
        self._preserve_policy_routing = False
        self.clamp_limit = cfg.swiglu_limit
        self.activation_func = self._apply_swiglu
        tile_shapes = (tuple(self.gate_up_proj.shape), tuple(self.down_proj.shape))
        block_size = getattr(effective_context.get("low_precision"), "block_size", MX_BLOCK_SIZE_32)

        if lp_mode in ("mxfp8", "fused_mxfp8"):
            if cfg.activation is not None:
                raise ValueError(f"{prefix} requires standard or clamped SwiGLU activation")
            if lp_mode == "fused_mxfp8" and cfg.routing_weight_mode != "output":
                raise ValueError(f"{prefix}: fused MXFP8 SwiGLU requires routing_weight_mode='output'")
            _validate_tile_alignment("mxfp8", block_size, tile_shapes)
            try:
                validate_npu_gmm_runtime()
                if lp_mode == "fused_mxfp8":
                    validate_npu_swiglu_runtime(clamp_limit=self.clamp_limit)
            except RuntimeError as exc:
                raise RuntimeError(f"{prefix}: {exc}") from exc
            self.quantizer = MXFP8Quantizer()
            self.grouped_linear = None
            return

        if strategy is None:
            scheme = _scheme_for_mode(lp_mode, effective_context)
            strategy = build_low_precision_strategy(scheme, tile_shapes=tile_shapes)
        else:
            self._validate_grouped_linear(strategy)
            if not isinstance(strategy, HiFloat8GroupedLinear):
                fmt_name = "fake_w4a8" if isinstance(strategy, FakeW4A8GroupedLinear) else strategy.FORMAT_NAME
                _validate_tile_alignment(fmt_name, block_size, tile_shapes)
        self.grouped_linear = strategy
        self.quantizer = None

    def _apply_swiglu(self, projected: Tensor) -> Tensor:
        """Apply SwiGLU in projection dtype with optional DeepSeek pre-SiLU clamping."""
        cfg = self._expert_cfg
        gate, up = projected.chunk(_GATE_UP_BRANCHES, dim=-1)
        if self.clamp_limit is not None:
            gate = gate.clamp(max=self.clamp_limit)
            up = up.clamp(min=-self.clamp_limit, max=self.clamp_limit)
        if cfg.activation is not None:
            return cfg.activation(gate, up).to(projected.dtype)
        act_fn = getattr(self, "act_fn", None)
        if self._preserve_policy_routing and callable(act_fn):
            return act_fn(gate) * up  # pylint: disable=not-callable
        return F.silu(gate) * up

    def compute_experts(
        self,
        hidden_states: Tensor,
        tokens_per_expert: Tensor,
        gate_up_proj: Tensor,
        down_proj: Tensor,
        routing_weights: Tensor | None = None,
    ) -> Tensor:
        """Execute quantized grouped-expert projections on owner-local or replicated slots.

        Args:
            hidden_states: Expert-major BF16/FP activations ``[R, H]``.
            tokens_per_expert: Per-expert slot counts ``[C]`` matching ``gate_up_proj``.
            gate_up_proj: Canonical ``[C, 2I, H]`` weights (unwrapped if ``DTensor``).
            down_proj: Canonical ``[C, H, I]`` weights (unwrapped if ``DTensor``).
            routing_weights: Optional differentiable ``before_fc2`` slot probabilities ``[R]``.

        Returns:
            Expert-major outputs ``[R, H]`` in ``hidden_states.dtype``.
        """
        cfg = self._expert_cfg
        gate_up = gate_up_proj.to_local() if isinstance(gate_up_proj, DTensor) else gate_up_proj
        down = down_proj.to_local() if isinstance(down_proj, DTensor) else down_proj
        if (
            gate_up.ndim != GROUPED_NDIM
            or gate_up.shape[0] < 1
            or gate_up.shape[1:] != (2 * cfg.intermediate_size, cfg.hidden_size)
            or down.shape != (gate_up.shape[0], cfg.hidden_size, cfg.intermediate_size)
        ):
            raise ValueError("expert compute requires complete [C,2I,H] and [C,H,I] matrices; ETP is unsupported")
        if tokens_per_expert.ndim != 1 or tokens_per_expert.shape != (gate_up.shape[0],):
            raise ValueError(f"{self.fqn}: counts must contain one count per local expert")
        if tokens_per_expert.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{self.fqn}: counts must be integers")
        if tokens_per_expert.device.type == "cpu":
            if bool((tokens_per_expert < 0).any()) or int(tokens_per_expert.sum()) != hidden_states.shape[0]:
                raise ValueError(f"{self.fqn}: counts must be non-negative and sum to token count")
        counts = tokens_per_expert.to(device=hidden_states.device)

        if self._lp_mode == "fused_mxfp8":
            return fused_grouped_swiglu(
                hidden_states,
                gate_up.transpose(-2, -1),
                down.transpose(-2, -1),
                counts,
                self.quantizer,  # type: ignore[arg-type]
                clamp_limit=self.clamp_limit,
            )

        if self._lp_mode == "mxfp8":
            projected = mxfp8_projection(
                hidden_states,
                gate_up.transpose(-2, -1),
                counts,
                self.quantizer,  # type: ignore[arg-type]
            )
            activated = self._apply_swiglu(projected)
            if cfg.routing_weight_mode == "before_fc2":
                if routing_weights is None:
                    raise ValueError("before_fc2 requires differentiable slot routing weights")
                activated = (activated * routing_weights.unsqueeze(-1)).to(activated.dtype)
            return mxfp8_projection(
                activated,
                down.transpose(-2, -1),
                counts,
                self.quantizer,  # type: ignore[arg-type]
            )

        try:
            projected = _GroupedLinearFunction.apply(
                hidden_states, gate_up, counts, self.grouped_linear, GROUP_LIST_COUNTS,
            )
            activated = self._apply_swiglu(projected)
            if cfg.routing_weight_mode == "before_fc2":
                if routing_weights is None:
                    raise ValueError("before_fc2 requires differentiable slot routing weights")
                activated = (activated * routing_weights.unsqueeze(-1)).to(activated.dtype)
            return _GroupedLinearFunction.apply(
                activated, down, counts, self.grouped_linear, GROUP_LIST_COUNTS,
            )
        except (LowPrecisionCapabilityError, W4A8CapabilityError) as exc:
            if not self._preserve_policy_routing:
                raise
            target = self.fqn or "<unknown>"
            raise LowPrecisionCapabilityError(
                f"Low-precision grouped-expert target {target!r} cannot run: {exc}"
            ) from exc

    def forward_expert_major(
        self,
        x: Tensor,
        num_tokens_per_expert: Tensor,
        scores: Tensor | None = None,
    ) -> Tensor:
        """Delegate expert-major calls directly to ``compute_experts``."""
        return self.compute_experts(
            x, num_tokens_per_expert, self.gate_up_proj, self.down_proj, scores,
        )


class LowPrecisionGroupedExperts(_LowPrecisionExpertsMixin, GroupedExpertsModule):
    """Local (EP=1) low-precision experts member and canonical QAT Policy adapter."""

    def __init__(
        self,
        num_experts: int | None = None,
        hidden_dim: int | None = None,
        intermediate_dim: int | None = None,
        *,
        grouped_linear: GroupedLinear | None = None,
        fqn: str = "",
        config: GroupedExpertsConfig | None = None,
        module: nn.Module | None = None,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        activation: Callable[[Tensor, Tensor], Tensor] | None = None,
        lp_mode: str = "w8a8",
        strategy: GroupedLinear | None = None,
    ) -> None:
        """Construct via staged ModuleFamily (config/module) or canonical Policy dimensions."""
        if grouped_linear is not None or any(
            size is not None for size in (num_experts, hidden_dim, intermediate_dim)
        ):
            if config is not None or module is not None or strategy is not None:
                raise TypeError("Cannot combine Policy dimension arguments with Family config/module/strategy")
            if any(size is None for size in (num_experts, hidden_dim, intermediate_dim)):
                raise TypeError("Provide all three dimensions (num_experts, hidden_dim, intermediate_dim)")
            self._validate_grouped_linear(grouped_linear)
            nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
            self._policy_expert_config = GroupedExpertsConfig(
                num_experts=num_experts,  # type: ignore[arg-type]
                hidden_size=hidden_dim,  # type: ignore[arg-type]
                intermediate_size=intermediate_dim,  # type: ignore[arg-type]
                routing_weight_mode="output",
                num_layers=1,
            )
            self.config = self._policy_expert_config
            self.gate_up_proj = nn.Parameter(
                torch.empty(num_experts, _GATE_UP_BRANCHES * intermediate_dim, hidden_dim)  # type: ignore[arg-type]
            )
            self.down_proj = nn.Parameter(
                torch.empty(num_experts, hidden_dim, intermediate_dim)  # type: ignore[arg-type]
            )
            self.act_fn = nn.SiLU()
            self._bind_policy_state(grouped_linear, fqn=fqn, limit=None)  # type: ignore[arg-type]
            return

        _validate_custom_gate_source(module, f"{module_fqn or '<root>'}: LowPrecisionGroupedExperts")
        GroupedExpertsModule.__init__(
            self,
            config=config,
            module=module,
            module_fqn=module_fqn,
            context=context,
            activation=activation,
        )
        self._init_low_precision_backend(
            lp_mode="strategy" if strategy is not None else lp_mode,
            module_fqn=module_fqn,
            context=context,
            strategy=strategy,
        )

    def _bind_policy_state(
        self,
        grouped_linear: GroupedLinear,
        *,
        fqn: str,
        limit: float | None,
    ) -> None:
        """Configure metadata shared by ``from_module`` and dimension Policy constructors."""
        cfg = self._expert_cfg
        self._preserve_policy_routing = True
        self._lp_mode = "policy"
        self.grouped_linear = grouped_linear
        self.quantizer = None
        self.fqn = fqn
        self.clamp_limit = limit
        self.activation_limit = limit
        self.activation_func = self._apply_swiglu
        self.num_experts = cfg.num_experts
        self.hidden_dim = cfg.hidden_size
        self.intermediate_dim = cfg.intermediate_size

    @classmethod
    def from_module(
        cls,
        source: nn.Module,
        *,
        fqn: str,
        grouped_linear: GroupedLinear,
    ) -> "LowPrecisionGroupedExperts":
        """Create a zero-allocation Policy shell retaining source parameters and buffers."""
        cls._validate_grouped_linear(grouped_linear)
        parameter_names = tuple(source._parameters)  # pylint: disable=protected-access
        if parameter_names != _EXPERT_PARAMETER_NAMES:
            raise TypeError(
                f"{fqn!r} must register packed expert parameters "
                f"{_EXPERT_PARAMETER_NAMES}, but got {parameter_names}."
            )
        gate_up_proj = source._parameters["gate_up_proj"]  # pylint: disable=protected-access
        down_proj = source._parameters["down_proj"]  # pylint: disable=protected-access
        if gate_up_proj is None or down_proj is None:
            raise ValueError(f"{fqn!r} packed expert parameters cannot be None.")
        if gate_up_proj.ndim != GROUPED_NDIM or down_proj.ndim != GROUPED_NDIM:
            raise ValueError(f"{fqn!r} packed expert weights must be three-dimensional.")
        expected_down_shape = (
            gate_up_proj.shape[0],
            gate_up_proj.shape[2],
            gate_up_proj.shape[1] // _GATE_UP_BRANCHES,
        )
        if gate_up_proj.shape[1] % _GATE_UP_BRANCHES or tuple(down_proj.shape) != expected_down_shape:
            raise ValueError(
                f"{fqn!r} has incompatible gate/up and down expert shapes: "
                f"gate_up={tuple(gate_up_proj.shape)}, down={tuple(down_proj.shape)}."
            )

        converted = cls.__new__(cls)  # pylint: disable=no-value-for-parameter
        nn.Module.__init__(converted)  # pylint: disable=unnecessary-dunder-call
        for name, parameter in source._parameters.items():  # pylint: disable=protected-access
            converted.register_parameter(name, parameter)
        for name, buffer in source._buffers.items():  # pylint: disable=protected-access
            converted.register_buffer(
                name,
                buffer,
                persistent=name not in source._non_persistent_buffers_set,  # pylint: disable=protected-access
            )
        for name, child in source._modules.items():  # pylint: disable=protected-access
            converted.add_module(name, child)
        converted.act_fn = getattr(source, "act_fn", nn.SiLU())
        limit = getattr(source, "limit", None)
        converted._policy_expert_config = GroupedExpertsConfig(  # pylint: disable=protected-access
            num_experts=gate_up_proj.shape[0],
            hidden_size=gate_up_proj.shape[2],
            intermediate_size=gate_up_proj.shape[1] // _GATE_UP_BRANCHES,
            routing_weight_mode="output",
            num_layers=1,
            swiglu_limit=limit,
        )
        converted._bind_policy_state(  # pylint: disable=protected-access
            grouped_linear, fqn=fqn, limit=converted._policy_expert_config.swiglu_limit,  # pylint: disable=protected-access
        )
        if hasattr(source, "config"):
            converted.config = source.config
        else:
            converted.config = converted._policy_expert_config  # pylint: disable=protected-access
        converted.training = source.training
        return converted

    def _grouped_forward(self, sorted_inputs: Tensor, tokens_per_expert: Tensor) -> Tensor:
        """Run Policy-routed expert-major tokens through the shared compute_experts stage."""
        return self.compute_experts(
            sorted_inputs, tokens_per_expert, self.gate_up_proj, self.down_proj,
        )

    def make_transforms(self) -> list:
        """Policy adapters retain source shapes verbatim; Family members use canonical transforms."""
        return [] if self._preserve_policy_routing else super().make_transforms()

    def forward(
        self,
        hidden_states: Tensor,
        topk_ids: Tensor | None = None,
        topk_weights: Tensor | None = None,
        **kwargs: Tensor,
    ) -> Tensor:
        """Execute staged Family routing or hardware-independent Policy routing."""
        if not self._preserve_policy_routing:
            return super().forward(hidden_states, topk_ids, topk_weights, **kwargs)

        top_k_index = kwargs.pop("top_k_index", topk_ids)
        top_k_weights = kwargs.pop("top_k_weights", topk_weights)
        if kwargs or top_k_index is None or top_k_weights is None:
            raise TypeError("Policy GroupedExperts requires hidden_states, top_k_index, and top_k_weights")
        if hidden_states.ndim != MATRIX_NDIM:
            raise ValueError(
                "Grouped experts require two-dimensional hidden_states, "
                f"but got shape {tuple(hidden_states.shape)}."
            )
        if top_k_index.shape != top_k_weights.shape or top_k_index.ndim != MATRIX_NDIM:
            raise ValueError("top_k_index and top_k_weights must have the same two-dimensional shape.")
        if top_k_index.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "The routed token count must match hidden_states: "
                f"routes={top_k_index.shape[0]}, tokens={hidden_states.shape[0]}."
            )

        input_dtype = hidden_states.dtype
        token_count, top_k = top_k_index.shape
        source_token_indices = (
            torch.arange(token_count, device=hidden_states.device)
            .unsqueeze(1)
            .expand(-1, top_k)
            .reshape(-1)
        )
        flattened_expert_indices = top_k_index.reshape(-1)
        expert_order = flattened_expert_indices.argsort()
        sorted_inputs = hidden_states[source_token_indices[expert_order]]
        tokens_per_expert = torch.bincount(flattened_expert_indices, minlength=self.num_experts)
        if tokens_per_expert.numel() != self.num_experts:
            raise ValueError(f"Expert indices must be in [0, {self.num_experts}).")
        sorted_outputs = self._grouped_forward(sorted_inputs, tokens_per_expert)
        sorted_weights = top_k_weights.reshape(-1)[expert_order]
        sorted_outputs = sorted_outputs * sorted_weights.unsqueeze(-1)
        inverse_order = torch.empty_like(expert_order)
        inverse_order[expert_order] = torch.arange(expert_order.shape[0], device=expert_order.device)
        return (
            sorted_outputs[inverse_order]
            .view(token_count, top_k, self.hidden_dim)
            .sum(dim=1)
            .to(dtype=input_dtype)
        )


class EPLowPrecisionGroupedExperts(_LowPrecisionExpertsMixin, EPGroupedExperts):
    """EP>1 low-precision experts member reusing BF16 All-to-All and SharedExpert overlap."""

    def __init__(
        self,
        *,
        config: GroupedExpertsConfig | None = None,
        module: nn.Module | None = None,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        overlap_shared_expert: bool = True,
        lp_mode: str = "w8a8",
        strategy: GroupedLinear | None = None,
    ) -> None:
        """Construct EP-capable parameters and bind the local low-precision compute backend."""
        _validate_custom_gate_source(module, f"{module_fqn or '<root>'}: EPLowPrecisionGroupedExperts")
        EPGroupedExperts.__init__(
            self,
            config=config,
            module=module,
            module_fqn=module_fqn,
            context=context,
            overlap_shared_expert=overlap_shared_expert,
        )
        self._init_low_precision_backend(
            lp_mode="strategy" if strategy is not None else lp_mode,
            module_fqn=module_fqn,
            context=context,
            strategy=strategy,
        )


class MoonEPLowPrecisionGroupedExperts(_LowPrecisionExpertsMixin, MoonEPGroupedExperts):
    """MoonEP low-precision experts member reusing dynamic BF16 expert replicas and overlap."""

    def __init__(
        self,
        *,
        config: GroupedExpertsConfig | None = None,
        module: nn.Module | None = None,
        module_fqn: str = "",
        context: Any = None,
        moonep_replica_capacity: int | None = None,
        overlap_shared_expert: bool = True,
        lp_mode: str = "w8a8",
        strategy: GroupedLinear | None = None,
    ) -> None:
        """Construct MoonEP-capable parameters and bind the local low-precision compute backend."""
        _validate_custom_gate_source(module, f"{module_fqn or '<root>'}: MoonEPLowPrecisionGroupedExperts")
        MoonEPGroupedExperts.__init__(
            self,
            config=config,
            module=module,
            module_fqn=module_fqn,
            context=context,
            moonep_replica_capacity=moonep_replica_capacity,
            overlap_shared_expert=overlap_shared_expert,
        )
        self._init_low_precision_backend(
            lp_mode="strategy" if strategy is not None else lp_mode,
            module_fqn=module_fqn,
            context=context,
            strategy=strategy,
        )


def replace_grouped_experts_with_policy(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
    scheme: LowPrecisionDtypeScheme | LowPrecisionConfig | None,
) -> LowPrecisionGroupedExperts:
    """Replace canonical packed experts via the non-Family QAT Policy entry point."""
    active_model_parallel_axes = [
        axis.upper() for axis in ("tp", "cp", "ep", "pp") if context.get(axis)
    ]
    if active_model_parallel_axes:
        raise NotImplementedError(
            "Low-precision grouped experts policy replacement requires TP=CP=EP=PP=1; "
            f"active axes: {active_model_parallel_axes}; use hyper_parallel.components.modules.GroupedExperts for EP."
        )
    parameters = tuple(module.parameters(recurse=False))
    strategy = build_low_precision_strategy(
        scheme,
        tile_shapes=tuple(tuple(parameter.shape) for parameter in parameters),
    )
    return LowPrecisionGroupedExperts.from_module(
        module,
        fqn=module_fqn,
        grouped_linear=strategy,
    )


def _ensure_standalone_num_layers(module: nn.Module | None, context: Mapping[str, Any] | None) -> dict[str, Any]:
    """Populate fallback num_layers=1 when standalone replacement factories are invoked outside a model tree."""
    merged = dict(context or {})
    if module is not None and merged.get("num_layers") is None:
        cfg = getattr(module, "config", None)
        inferred = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "num_layers", None) or 1
        merged["num_layers"] = inferred
    return merged


@module_replacement
def replace_mxfp8_grouped_experts(
    *, module: nn.Module, module_fqn: str = "", context: Mapping[str, Any] | None = None,
) -> LowPrecisionGroupedExperts:
    """Explicit local native W8A8 MXFP8 replacement factory."""
    ctx = _ensure_standalone_num_layers(module, context)
    return LowPrecisionGroupedExperts(
        module=module, module_fqn=module_fqn, context={**ctx, "low_precision": LowPrecisionDtypeScheme()},
        lp_mode="w8a8",
    )


@module_replacement
def replace_w4a8_grouped_experts(
    *, module: nn.Module, module_fqn: str = "", context: Mapping[str, Any] | None = None,
) -> LowPrecisionGroupedExperts:
    """Explicit local native W4A8 replacement factory."""
    ctx = _ensure_standalone_num_layers(module, context)
    block_size = getattr(ctx.get("low_precision"), "block_size", MX_BLOCK_SIZE_32)
    scheme = LowPrecisionDtypeScheme(weight_format="mxfp4", act_format="mxfp8", block_size=block_size)
    return LowPrecisionGroupedExperts(
        module=module, module_fqn=module_fqn, context={**ctx, "low_precision": scheme}, lp_mode="w4a8",
    )


@module_replacement
def replace_fake_w4a8_grouped_experts(
    *, module: nn.Module, module_fqn: str = "", context: Mapping[str, Any] | None = None,
) -> LowPrecisionGroupedExperts:
    """Explicit local fake-quantized W4A8 replacement factory."""
    ctx = _ensure_standalone_num_layers(module, context)
    block_size = getattr(ctx.get("low_precision"), "block_size", MX_BLOCK_SIZE_32)
    scheme = LowPrecisionDtypeScheme(
        weight_format="mxfp4", act_format="mxfp8", is_fake_quantize=True, block_size=block_size,
    )
    return LowPrecisionGroupedExperts(
        module=module, module_fqn=module_fqn, context={**ctx, "low_precision": scheme}, lp_mode="fake_w4a8",
    )


def _make_family_member_builder(target_cls: type[nn.Module], lp_mode: str) -> Callable[..., nn.Module]:
    """Create a signature-preserving ModuleFamily member constructor for one (topology, lp_mode) pair."""
    init_sig = inspect.signature(target_cls.__init__)
    excluded = {
        "self", "num_experts", "hidden_dim", "intermediate_dim",
        "grouped_linear", "fqn", "lp_mode", "strategy",
    }
    allowed_params = [param for name, param in init_sig.parameters.items() if name not in excluded]

    def builder(**kwargs: Any) -> nn.Module:
        return target_cls(lp_mode=lp_mode, **kwargs)

    builder.__signature__ = inspect.Signature(parameters=allowed_params)  # type: ignore[attr-defined]
    builder.__name__ = f"build_{target_cls.__name__}_{lp_mode}"
    builder.__qualname__ = builder.__name__
    builder.__doc__ = f"Build {target_cls.__name__} bound to low-precision mode {lp_mode!r}."
    return builder


for _topology_prefix, _cls in (
    ("local", LowPrecisionGroupedExperts),
    ("ep", EPLowPrecisionGroupedExperts),
    ("moonep", MoonEPLowPrecisionGroupedExperts),
):
    for _mode in _LP_MODES:
        _fn_name = f"build_{_topology_prefix}_{_mode}_experts"
        globals()[_fn_name] = _make_family_member_builder(_cls, _mode)

__all__ = [
    "EPLowPrecisionGroupedExperts",
    "LowPrecisionGroupedExperts",
    "MoonEPLowPrecisionGroupedExperts",
    "replace_fake_w4a8_grouped_experts",
    "replace_grouped_experts_with_policy",
    "replace_mxfp8_grouped_experts",
    "replace_w4a8_grouped_experts",
]
