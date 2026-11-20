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
"""Complete training-time KDA algorithm, independent of CP resource binding."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.components.functional.kimi_delta_attention import (
    chunk_kda,
)
from hyper_parallel.components.modules.attention.linear_attention.sequence_layout import (
    PackedSequenceInfo,
)
from hyper_parallel.components.functional.linear_attention_convolution import (
    validate_causal_convolution,
    run_causal_short_conv,
)
from hyper_parallel.components.functional.linear_attention_convolution import validate_conv_backend
from hyper_parallel.components.modules.attention.linear_attention.layer_adapter import (
    adopt_source_parameters, copy_unbound_linear_attention, prepare_linear_attention_input,
)

class KimiRMSNormGated(nn.Module):
    """Per-head RMSNorm followed by the sigmoid output gate used by KDA."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """Initialize the per-head normalization weight and epsilon."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Normalize in FP32 and apply a sigmoid gate in the input dtype.

        Args:
            hidden_states: Attention output with per-head channels in the last dimension.
            gate: Output-gate logits applied after RMS normalization.
        """
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.variance_epsilon)
        normalized = normalized * self.weight.float()
        return (normalized * gate.float().sigmoid()).to(input_dtype)


def _make_short_convolution(channels: int, kernel_size: int) -> nn.Conv1d:
    """Create the depthwise causal convolution used by the training layer."""
    convolution = nn.Conv1d(
        channels,
        channels,
        kernel_size=kernel_size,
        groups=channels,
        padding=kernel_size - 1,
        bias=False,
    )
    convolution.activation = "silu"
    return convolution


class KimiDeltaAttentionModule(nn.Module):
    """KDA projections, convolution, gated recurrence and output in one local algorithm."""

    def __init__(self, *, config: Any = None, module: nn.Module | None = None,
                 backend: str = "eager", conv_backend: str | None = None,
                 chunk_size: int = 64, input_layout: str = "BSH") -> None:
        """Construct local parameters or adopt source weights without renaming them."""
        super().__init__()
        if (config is None) == (module is None):
            raise ValueError("KDA requires exactly one of config or module")
        self.config = config if module is None else getattr(module, "config", None)
        self.backend = backend
        self.conv_backend = validate_conv_backend(backend if conv_backend is None else conv_backend)
        self.chunk_size = chunk_size
        self.input_layout = input_layout
        self._config_initialization = module is None or getattr(module, "_config_initialization", False)
        self._returns_tuple = False
        if module is None:
            self._initialize_parameters(config)
        else:
            self._adopt_source(module)
        self._validate_local_contract()

    def _validate_local_contract(self) -> None:
        """Reject unsupported source semantics before running any local kernels."""
        if self.use_short_conv:
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                validate_causal_convolution(getattr(self, name), require_activation=True)
        if self.backend not in ("eager", "triton", "ascendc"):
            raise ValueError("KDA backend must be eager, triton or ascendc")
        if self.backend != "eager" and (self.head_k_dim != 128 or self.head_v_dim != 128 or self.chunk_size != 64):
            raise ValueError("Fused KDA requires K=V=128 and chunk_size=64")
        if getattr(self, "allow_neg_eigval", False):
            raise NotImplementedError("KDA family does not support allow_neg_eigval=True")
        if not self.safe_gate or self.lower_bound is None or self.lower_bound >= 0:
            raise ValueError("KDA family requires its lower-bounded gate")

    def __deepcopy__(self, memo: dict) -> nn.Module:
        """Retain pre-binding model-copy behavior without copying process groups."""
        return copy_unbound_linear_attention(self, memo)

    def _initialize_parameters(self, config: Any) -> None:
        """Keep the established standalone KDA parameter initialization."""
        self.hidden_size = getattr(config, "hidden_size", None)
        self.num_heads = getattr(config, "num_heads", 32)
        self.num_k_heads = self.num_heads
        self.num_v_heads = getattr(config, "num_v_heads", None)
        self.head_k_dim = getattr(config, "head_k_dim", 128)
        self.head_v_dim = getattr(config, "head_v_dim", 128)
        self.conv_kernel_size = getattr(config, "conv_kernel_size", 4)
        self.gate_lower_bound = getattr(config, "gate_lower_bound", -5.0)
        self.use_full_rank_gate = getattr(config, "use_full_rank_gate", True)
        self.layer_idx = getattr(config, "layer_idx", 0)
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        if self.hidden_size is None or self.hidden_size <= 0 or self.num_heads <= 0:
            raise ValueError("hidden_size and num_heads must be positive.")
        if self.num_v_heads is None:
            self.num_v_heads = self.num_heads
        if self.num_v_heads <= 0 or self.num_v_heads % self.num_heads:
            raise ValueError("num_v_heads must be positive and divisible by num_heads.")
        if self.head_k_dim <= 0 or self.head_v_dim <= 0:
            raise ValueError("KDA head dimensions must be positive.")
        if self.conv_kernel_size <= 0 or self.chunk_size <= 0:
            raise ValueError("conv_kernel_size and chunk_size must be positive.")
        if self.gate_lower_bound >= 0:
            raise ValueError("gate_lower_bound must be negative.")

        self.head_dim = self.head_v_dim
        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_size = self.conv_kernel_size
        self.lower_bound = self.gate_lower_bound
        self.safe_gate = True
        self.allow_neg_eigval = False
        self.use_short_conv = True

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.q_conv1d = _make_short_convolution(self.key_dim, self.conv_kernel_size)
        self.k_conv1d = _make_short_convolution(self.key_dim, self.conv_kernel_size)
        self.v_conv1d = _make_short_convolution(self.value_dim, self.conv_kernel_size)

        # A_log is the source model checkpoint key.
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))  # pylint: disable=invalid-name
        self.dt_bias = nn.Parameter(
            torch.zeros(self.num_v_heads * self.head_k_dim, dtype=torch.float32)
        )
        with torch.no_grad():
            self.A_log.copy_(torch.empty_like(self.A_log).uniform_(1, 16).log())

        self.f_a_proj = nn.Linear(self.hidden_size, self.head_k_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_k_dim, self.num_v_heads * self.head_k_dim, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        else:
            self.g_a_proj = nn.Linear(self.hidden_size, self.head_v_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_v_dim, self.value_dim, bias=False)
        self.o_norm = KimiRMSNormGated(self.head_v_dim, eps=rms_norm_eps)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)


    def _adopt_source(self, module: nn.Module) -> None:
        """Adopt source parameters and dimensions without changing checkpoint keys."""
        required = ("q_proj", "k_proj", "v_proj", "b_proj", "o_norm", "o_proj", "A_log", "dt_bias",
                    "num_heads", "head_k_dim")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise TypeError(f"KDA source is missing {missing}")
        if not (hasattr(module, "f_proj") or all(hasattr(module, name) for name in ("f_a_proj", "f_b_proj"))):
            raise TypeError("KDA source requires f_proj or f_a_proj/f_b_proj")
        if not (hasattr(module, "g_proj") or all(hasattr(module, name) for name in ("g_a_proj", "g_b_proj"))):
            raise TypeError("KDA source requires g_proj or g_a_proj/g_b_proj")
        owners = ("q_proj", "k_proj", "v_proj", "q_conv1d", "k_conv1d", "v_conv1d", "A_log", "dt_bias",
                  "f_proj", "f_a_proj", "f_b_proj", "b_proj", "g_proj", "g_a_proj", "g_b_proj", "o_norm", "o_proj")
        adopt_source_parameters(self, module, owners)
        self.num_heads = self.num_k_heads = module.num_heads
        self.num_v_heads = getattr(module, "num_v_heads", self.num_heads)
        self.head_k_dim = module.head_k_dim
        self.head_v_dim = getattr(module, "head_v_dim", getattr(module, "head_dim", None))
        self.head_dim = self.head_v_dim
        if self.num_v_heads % self.num_heads:
            raise ValueError("KDA value heads must be divisible by query heads")
        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.hidden_size = self.q_proj.in_features
        self.layer_idx = getattr(module, "layer_idx", None)
        self.lower_bound = getattr(module, "lower_bound", getattr(module, "gate_lower_bound", None))
        self.gate_lower_bound = self.lower_bound
        self.safe_gate = getattr(module, "safe_gate", self.lower_bound is not None)
        self.allow_neg_eigval = getattr(module, "allow_neg_eigval", False)
        self.use_short_conv = getattr(module, "use_short_conv", hasattr(module, "q_conv1d"))
        self._returns_tuple = getattr(module, "_returns_tuple", not hasattr(module, "f_a_proj"))

    @staticmethod
    def make_transforms() -> list:
        """Keep source checkpoint names and shapes unchanged during replacement.

        Returns:
            An empty conversion list; source Parameter objects are adopted directly.
        """
        return []

    def reset_parameters(self) -> None:
        """Reset native/meta construction, without resetting adopted source weights."""
        if not self._config_initialization:
            return
        for child in self.children():
            if isinstance(child, (nn.Linear, nn.Conv1d)):
                child.reset_parameters()
        nn.init.uniform_(self.A_log, 1, 16)
        with torch.no_grad():
            self.A_log.log_()
        nn.init.zeros_(self.dt_bias)
        nn.init.ones_(self.o_norm.weight)

    def project_inputs(self, hidden_states: torch.Tensor) -> tuple:
        """Project Q/K/V in the source order, before convolution or head exchange.

        Args:
            hidden_states: Local hidden activations in BSH or flat TH layout.
        """
        return self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

    def project_state_gates(self, hidden_states: torch.Tensor) -> tuple:
        """Produce raw decay and beta logits after the scheduled Q/K/V stage.

        Args:
            hidden_states: Local hidden activations in BSH or flat TH layout.
        """
        gate = self.f_proj(hidden_states) if hasattr(self, "f_proj") else self.f_b_proj(self.f_a_proj(hidden_states))
        return gate, self.b_proj(hidden_states)

    def project_output_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Keep source ordering of the output gate's shared-hidden gradient accumulation.

        Args:
            hidden_states: Local hidden activations in BSH or flat TH layout.
        """
        gate = self.g_proj(hidden_states) if hasattr(self, "g_proj") else self.g_b_proj(self.g_a_proj(hidden_states))
        return gate.reshape(*hidden_states.shape[:2], self.num_v_heads, self.head_v_dim)

    def convolve_inputs(self, operands: tuple, info: PackedSequenceInfo | None) -> tuple:
        """Apply local convolutions before shaping heads.

        Args:
            operands: Projected inputs consumed by this attention stage.
            info: Immutable packed-sequence metadata, or None for dense input.
        """
        query, key, value = operands
        if self.use_short_conv:
            query, key, value = (run_causal_short_conv(tensor, conv, sequence_info=info, backend=self.conv_backend)
                                 for tensor, conv in zip((query, key, value),
                                                         (self.q_conv1d, self.k_conv1d, self.v_conv1d)))
        else:
            query, key, value = (F.silu(tensor) for tensor in (query, key, value))
        return query, key, value

    def shape_inputs(self, operands: tuple, degree: int = 1) -> tuple:
        """Expose head axes without changing raw gate or beta semantics.

        Args:
            operands: Projected inputs consumed by this attention stage.
            degree: Number of ranks participating in the head/sequence exchange.
        """
        query, key, value, gate, beta = operands
        batch, tokens = query.shape[:2]
        kh, vh = self.num_heads // degree, self.num_v_heads // degree
        return (query.reshape(batch, tokens, kh, self.head_k_dim),
                key.reshape(batch, tokens, kh, self.head_k_dim),
                value.reshape(batch, tokens, vh, self.head_v_dim),
                gate.reshape(batch, tokens, vh, self.head_k_dim), beta.reshape(batch, tokens, vh))

    def run_local_core(self, operands: tuple, info: PackedSequenceInfo | None,
                      a_log: torch.Tensor | None = None, dt_bias: torch.Tensor | None = None) -> torch.Tensor:
        """Invoke the selected complete local autograd without CP transport.

        Args:
            operands: Projected inputs consumed by this attention stage.
            info: Immutable packed-sequence metadata, or None for dense input.
            a_log: Log decay-rate parameter.
            dt_bias: Per-key time-step bias.
        """
        offsets = None
        if info is not None:
            offsets = (info.local_offsets if self.backend in ("eager", "ascendc") else
                       info.materialize(operands[0].device, self.chunk_size).cu_seqlens)
        return chunk_kda(*operands, a_log=self.A_log if a_log is None else a_log,
                         dt_bias=self.dt_bias if dt_bias is None else dt_bias,
                         initial_state=None, output_final_state=False, chunk_size=self.chunk_size,
                         use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                         use_beta_sigmoid_in_kernel=True, lower_bound=self.lower_bound,
                         backend=self.backend, cu_seqlens=offsets)[0]

    def execute_attention_pipeline(self, hidden_states: torch.Tensor,
                                   info: PackedSequenceInfo | None) -> tuple:
        """Keep local convolution, state gates and output-gate ordering unchanged.

        Args:
            hidden_states: Local hidden activations in BSH or flat TH layout.
            info: Immutable packed-sequence metadata, or None for dense input.
        """
        projected = self.convolve_inputs(self.project_inputs(hidden_states), info)
        gate, beta = self.project_state_gates(hidden_states)
        output = self.run_local_core(self.shape_inputs((*projected, gate, beta)), info)
        return output, self.project_output_gate(hidden_states)

    def project_output(self, output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Apply the source normalization/gate and output projection in token layout.

        Args:
            output: Local attention output before its final projection.
            gate: Output-gate logits applied after normalization.
        """
        output = self.o_norm(output, gate)
        return self.o_proj(output.reshape(*output.shape[:2], self.value_dim))

    def prepare_input(self, hidden_states: torch.Tensor, options: dict) -> tuple:
        """Adapt local dense/flat input; CP members resolve their own token interval.

        Args:
            hidden_states: Local dense or flat hidden activations.
            options: Explicit backend options or invocation document metadata.
        """
        return prepare_linear_attention_input(hidden_states, options)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
                past_key_values: Any = None, use_cache: bool = False, output_attentions: bool = False,
                **kwargs: Any) -> Any:
        """Run dense or packed training KDA, retaining the source output structure.

        Args:
            hidden_states: Local hidden activations in BSH or flat TH layout.
            attention_mask: Optional source-model mask; packed boundaries carry document resets.
            past_key_values: Source-model inference cache; training-family execution requires None.
            use_cache: Inference-cache request; unsupported by this training entry.
            output_attentions: Source-model attention-weight request; KDA returns no attention matrix.
        """
        if use_cache or past_key_values is not None or kwargs.get("cache_params") is not None:
            raise NotImplementedError("KDA family supports training without recurrent cache")
        if output_attentions:
            raise NotImplementedError("KDA does not materialize attention weights")
        if attention_mask is not None and not bool(attention_mask.bool().all().item()):
            raise NotImplementedError("Padded KDA batches are not implemented")
        hidden_states, info, is_flat_input = self.prepare_input(hidden_states, kwargs)
        output, gate = self.execute_attention_pipeline(hidden_states, info)
        del hidden_states
        output = self.project_output(output, gate)
        if is_flat_input:
            output = output.squeeze(0)
        return (output, None, None) if self._returns_tuple else output
