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
"""Local Gated DeltaNet training with reversible grouped QKV and convolution rows."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.components.checkpoint import InterleaveQKV
from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter
from hyper_parallel.components.functional.gated_delta_net import chunk_gated_delta_rule
from hyper_parallel.components.functional.linear_attention_convolution import (
    validate_causal_convolution,
    run_causal_short_conv, validate_conv_backend,
)
from hyper_parallel.components.modules.attention.linear_attention_utils.layer_adapter import (
    adopt_source_parameters, copy_unbound_linear_attention, prepare_linear_attention_input,
)
from hyper_parallel.components.modules.attention.linear_attention_utils.sequence_layout import PackedSequenceInfo


class GatedDeltaNetRMSNorm(nn.Module):
    """Qwen-style per-head RMSNorm followed by SiLU, retaining its rounding order."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        """Create a direct-scale normalization weight."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Normalize in FP32, round, scale and then apply the FP32 output gate.

        Args:
            hidden_states: Attention output with per-head channels in the last dimension.
            gate: Output-gate logits applied after RMS normalization.
        """
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(dtype)
        return (hidden_states * F.silu(gate.float())).to(dtype)


class GatedDeltaNetModule(nn.Module):
    """Local GDN projections, convolution, GVA, gates, recurrence and output."""

    def __init__(self, *, config: Any = None, module: nn.Module | None = None,
                 backend: str = "eager", conv_backend: str | None = None,
                 chunk_size: int = 64, input_layout: str = "BSH") -> None:
        """Construct local parameters or convert source QKV rows, retaining checkpoint names."""
        super().__init__()
        if (config is None) == (module is None):
            raise ValueError("GDN requires exactly one of config or module")
        self.config = config if module is None else getattr(module, "config", None)
        self.backend = backend
        self.conv_backend = validate_conv_backend(backend if conv_backend is None else conv_backend)
        self.chunk_size, self.input_layout = chunk_size, input_layout
        self._config_initialization = module is None or getattr(module, "_config_initialization", False)
        if module is None:
            self._initialize_parameters(config)
        else:
            self._adopt_source(module)
        validate_causal_convolution(self.conv1d)
        if self.num_v_heads % self.num_k_heads:
            raise ValueError("GDN value heads must be divisible by query heads")
        self._configure_qkv_layout(module)
        if self.backend not in ("eager", "triton", "ascendc"):
            raise ValueError("GDN backend must be eager, triton or ascendc")
        if self.backend != "eager" and (self.head_k_dim != 128 or self.head_v_dim != 128 or chunk_size != 64):
            raise ValueError("Fused GDN requires K=V=128 and chunk_size=64")
        if self.activation not in ("silu", "swish"):
            raise NotImplementedError("GDN family implements SiLU/Swish convolution")

    def __deepcopy__(self, memo: dict) -> nn.Module:
        """Retain pre-binding model-copy behavior without copying process groups."""
        return copy_unbound_linear_attention(self, memo)

    def _initialize_parameters(self, config: Any) -> None:
        """Create source-compatible projections, gates, convolution and output norm."""
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = getattr(config, "linear_conv_kernel_dim", 4)
        self.activation = getattr(config, "hidden_act", "silu")
        self.layer_idx = getattr(config, "layer_idx", 0)
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, self.conv_kernel_size,
                               padding=self.conv_kernel_size - 1, groups=self.conv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        # A_log is the source model checkpoint key.
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads).uniform_(0, 16).log())  # pylint: disable=invalid-name
        self.norm = GatedDeltaNetRMSNorm(self.head_v_dim, getattr(config, "rms_norm_eps", 1e-6))
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def _adopt_source(self, module: nn.Module) -> None:
        """Adopt source parameters and dimensions without changing checkpoint keys."""
        owners = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "conv1d", "A_log", "dt_bias",
                  "norm", "out_proj", "out_proj_input")
        dimensions = ("num_k_heads", "num_v_heads", "head_k_dim", "head_v_dim", "key_dim", "value_dim", "conv_dim")
        missing = [name for name in owners[:-1] + dimensions if not hasattr(module, name)]
        if missing:
            raise TypeError(f"GDN source is missing {missing}")
        adopt_source_parameters(self, module, owners)
        for name in dimensions:
            setattr(self, name, getattr(module, name))
        self.hidden_size = self.in_proj_qkv.in_features
        self.layer_idx = getattr(module, "layer_idx", None)
        self.activation = getattr(module, "activation", "silu")
        self.conv_kernel_size = self.conv1d.kernel_size[0]

    def _configure_qkv_layout(self, module: nn.Module | None) -> None:
        """Keep global conversion dimensions when a grouped local member becomes CP."""
        self.num_value_groups = self.num_v_heads // self.num_k_heads
        self.qkv_group_width = 2 * self.head_k_dim + self.num_value_groups * self.head_v_dim
        self._source_qkv_heads = getattr(module, "_source_qkv_heads", self.num_k_heads)
        self._convert_source_qkv = module is not None and getattr(module, "_convert_source_qkv", True)
        if module is not None and not isinstance(module, GatedDeltaNetModule):
            self._group_source_parameters()

    def _qkv_conversion(self) -> InterleaveQKV:
        """Group one Q/K head with all value heads sharing that key."""
        return InterleaveQKV(self._source_qkv_heads, 1, self.head_k_dim,
                             self.num_value_groups * self.head_v_dim, source_is_fused=True)

    def _group_source_parameters(self) -> None:
        """Convert loaded source rows without mutating the original projection or convolution."""
        conversion = self._qkv_conversion()
        for name in ("in_proj_qkv", "conv1d"):
            child = deepcopy(getattr(self, name))
            for suffix in ("weight", "bias"):
                parameter = getattr(child, suffix, None)
                if parameter is not None:
                    grouped = conversion.convert({suffix: parameter}, [suffix], [suffix])[suffix]
                    with torch.no_grad():
                        parameter.copy_(grouped)
            setattr(self, name, child)

    def make_transforms(self) -> list[WeightConverter]:
        """Declare reversible source-QKV and depthwise-convolution row conversion.

        Returns:
            Source-checkpoint transforms; native construction already uses grouped rows.
        """
        if not self._convert_source_qkv:
            return []
        names = [f"{owner}.{suffix}" for owner in ("in_proj_qkv", "conv1d")
                 for suffix in ("weight", "bias") if getattr(getattr(self, owner), suffix, None) is not None]
        return [WeightConverter(source_patterns=name, target_patterns=name,
                                operations=[self._qkv_conversion()]) for name in names]

    def reset_parameters(self) -> None:
        """Initialize direct/meta parameters without resetting converted sources."""
        if not self._config_initialization:
            return
        for child in self.children():
            if isinstance(child, (nn.Linear, nn.Conv1d)):
                child.reset_parameters()
        with torch.no_grad():
            self.A_log.uniform_(0, 16).log_()
            self.dt_bias.fill_(1)
            self.norm.weight.fill_(1)

    def project_inputs(self, hidden_states: torch.Tensor) -> tuple:
        """Keep output-gate tokens local while producing raw recurrence operands.

        Args:
            hidden_states: Local hidden activations in BSH or flat TH layout.
        """
        mixed_qkv = self.in_proj_qkv(hidden_states)
        gate = self.in_proj_z(hidden_states).reshape(*hidden_states.shape[:2], self.num_v_heads, self.head_v_dim)
        return (mixed_qkv, self.in_proj_b(hidden_states), self.in_proj_a(hidden_states)), gate

    def prepare_core_inputs(self, mixed_qkv: torch.Tensor, beta_logits: torch.Tensor,
                            decay_logits: torch.Tensor, degree: int = 1,
                            a_log: torch.Tensor | None = None, dt_bias: torch.Tensor | None = None) -> tuple:
        """Apply GVA and source gate arithmetic after convolution/layout exchange.

        Args:
            mixed_qkv: Packed query, key and value projection output.
            beta_logits: Raw delta-rule update logits.
            decay_logits: Raw decay projection output.
            degree: Number of ranks participating in the head/sequence exchange.
            a_log: Log decay-rate parameter.
            dt_bias: Per-key time-step bias.
        """
        grouped = mixed_qkv.unflatten(-1, (self.num_k_heads // degree, self.qkv_group_width))
        query, key, value = grouped.split((self.head_k_dim, self.head_k_dim,
                                          self.num_value_groups * self.head_v_dim), dim=-1)
        batch, tokens = mixed_qkv.shape[:2]
        query = query.reshape(batch, tokens, self.num_k_heads // degree, self.head_k_dim)
        key = key.reshape_as(query)
        value = value.reshape(batch, tokens, self.num_v_heads // degree, self.head_v_dim)
        a_log = self.A_log if a_log is None else a_log
        dt_bias = self.dt_bias if dt_bias is None else dt_bias
        beta = beta_logits.sigmoid()
        decay = -a_log.float().exp() * F.softplus(decay_logits.float() + dt_bias)
        groups = self.num_v_heads // self.num_k_heads
        if groups > 1 and self.backend != "ascendc":
            # The native GVA kernels index shared Q/K directly. Existing eager
            # and Triton state kernels still require matched head counts.
            query = query.repeat_interleave(groups, dim=2)
            key = key.repeat_interleave(groups, dim=2)
        return query, key, value, decay, beta

    def run_local_core(self, operands: tuple, info: PackedSequenceInfo | None) -> torch.Tensor:
        """Call the explicit local backend with immutable document metadata.

        Args:
            operands: Projected inputs consumed by this attention stage.
            info: Immutable packed-sequence metadata, or None for dense input.
        """
        offsets = None
        if info is not None:
            offsets = (info.local_offsets if self.backend in ("eager", "ascendc") else
                       info.materialize(operands[0].device, self.chunk_size).cu_seqlens)
        return chunk_gated_delta_rule(*operands, initial_state=None, output_final_state=False,
                                      chunk_size=self.chunk_size, use_qk_l2norm_in_kernel=True,
                                      backend=self.backend, cu_seqlens=offsets)[0]

    def execute_attention_pipeline(self, operands: tuple, info: PackedSequenceInfo | None) -> torch.Tensor:
        """Execute the local convolution and state recurrence.

        Args:
            operands: Projected inputs consumed by this attention stage.
            info: Immutable packed-sequence metadata, or None for dense input.
        """
        mixed, beta, decay = operands
        mixed = run_causal_short_conv(mixed, self.conv1d, sequence_info=info, backend=self.conv_backend)
        return self.run_local_core(self.prepare_core_inputs(mixed, beta, decay), info)

    def project_output(self, output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Preserve the source norm and optional output-input adapter.

        Args:
            output: Local attention output before its final projection.
            gate: Output-gate logits applied after normalization.
        """
        batch, tokens = output.shape[:2]
        output = self.norm(output.reshape(-1, self.head_v_dim), gate.reshape(-1, self.head_v_dim))
        output = output.reshape(batch, tokens, self.value_dim)
        if hasattr(self, "out_proj_input"):
            output = self.out_proj_input(output)
        return self.out_proj(output)

    def prepare_input(self, hidden_states: torch.Tensor, options: dict) -> tuple:
        """Adapt local dense/flat input; CP members resolve their own token interval.

        Args:
            hidden_states: Local dense or flat hidden activations.
            options: Explicit backend options or invocation document metadata.
        """
        return prepare_linear_attention_input(hidden_states, options)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Accept the source model's decorated positional and keyword call contract.

        Args:
            *args: Hidden states, optional cache and optional padding mask.
            **kwargs: Named inputs and per-invocation packed metadata.
        """
        # HF's accelerate wrapper exposes only (*args, **kwargs). The typed
        # training implementation still rejects missing or extra arguments.
        return self._forward_training(*args, **kwargs)

    def _forward_training(self, hidden_states: torch.Tensor, cache_params: Any = None,
                          attention_mask: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        """Run training GDN on dense or flat packed tokens.

        Args:
            hidden_states: Local hidden activations in BSH or flat TH layout.
            cache_params: Inference cache; training-family execution requires None.
            attention_mask: Optional source-model mask; packed boundaries carry document resets.
        """
        if cache_params is not None or kwargs.get("use_cache", False):
            raise NotImplementedError("GDN family supports training without recurrent cache")
        hidden_states, info, is_flat_input = self.prepare_input(hidden_states, kwargs)
        if attention_mask is not None:
            if attention_mask.ndim != 2 or tuple(attention_mask.shape) != tuple(hidden_states.shape[:2]):
                raise ValueError("GDN expects a local two-dimensional padding mask")
            hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)
        operands, gate = self.project_inputs(hidden_states)
        del hidden_states
        output = self.execute_attention_pipeline(operands, info)
        del operands
        output = self.project_output(output, gate)
        return output.squeeze(0) if is_flat_input else output
