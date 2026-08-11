# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
# Standalone MLP Module - Extracted from Sophon-Pytorch
# This module contains the core MLP algorithms with hardware acceleration
# features preserved, but with all distributed/memory optimization logic removed.

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Conditional NPU imports
try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

if TYPE_CHECKING:
    from hyper_parallel.models.vl_moe.model import (
        VLTextConfig,
    )


# ============================================================================
# Utility functions
# ============================================================================

def _get_act_fn(hidden_act):
    """Map activation function name to callable."""
    _ACT_NAME_TO_FN = {
        "silu": F.silu,
        "gelu": F.gelu,
        "relu": F.relu,
        "gelu_new": F.gelu,
        "gelu_pytorch_tanh": F.gelu,
        "tanh": torch.tanh,
    }
    return _ACT_NAME_TO_FN.get(hidden_act, F.silu)


class LinearWithMatmul(nn.Linear):
    """Aligned with ColumnParallelLinear (single device, no parallelism).

    name: ColumnParallelLinear
    forward path: LinearWithGradAccumulationAndAsyncCommunication.forward
      → output = torch.matmul(input, weight.t())
      → output = output + bias  (bias explicitly added separately, not addmm fused)

    Used for: linear_fc1, linear_fc2

    skip_bias_add=True: don't add bias, return (output, bias) for external addition.
    skip_bias_add=False: bias explicitly added in forward, return (output, None).

    NPU bf16 note: matmul runs in @custom_fwd autograd Function context,
    NPU automatically selects the same kernel for non-contiguous weight.t() as weight.t().contiguous().
    calling directly in forward does not trigger this behavior, so explicit .contiguous() is needed.
    """

    def __init__(self, in_features, out_features, bias=True, skip_bias_add=True):
        super().__init__(in_features, out_features, bias=bias)
        self.skip_bias_add = skip_bias_add

    def forward(self, input):
        output = torch.matmul(input, self.weight.t().contiguous())
        if self.skip_bias_add:
            return output, self.bias
        if self.bias is not None:
            output = output + self.bias
        return output, None


class LinearWithFusedOps(nn.Linear):
    """Aligned with SequenceParallelLinear (single device, no parallelism).

    name: SequenceParallelLinear
    forward path:
      - When skip_bias_add=False and gradient_accumulation_fusion is enabled:
        goes through LinearWithGradAccumulation → F.linear(input, weight, bias)  (addmm fused)
      - Otherwise: F.linear(input, weight, bias)  (addmm fused)
    Both paths use F.linear (addmm fused), different from ColumnParallelLinear's matmul+bias.

    Used for: linear_qkv (MLA/DSA scenario), index_linear_k, prev_proj

    Key difference: uses F.linear (addmm fused) instead of torch.matmul + bias added separately.
    Under NPU bf16, addmm fusion and matmul+bias have different rounding results.
    """

    def __init__(self, in_features, out_features, bias=True, skip_bias_add=True):
        super().__init__(in_features, out_features, bias=bias)
        self.skip_bias_add = skip_bias_add

    def forward(self, input):
        if self.skip_bias_add:
            # Aligned with: F.linear(input, weight), without bias
            output = F.linear(input, self.weight)
            return output, self.bias
        # Aligned with: F.linear(input, weight, bias) — addmm fused
        output = F.linear(input, self.weight, self.bias)
        return output, None


# ============================================================================
# TextMLP
# ============================================================================

class TextMLP(nn.Module):

    def __init__(
        self,
        config: "VLTextConfig",
        ffn_fc1_factor: Optional[int] = None,
        activation_func: Optional[Callable] = None,
    ):
        super().__init__()
        self.config = config

        # Cache the activation function mapped from config.hidden_act
        self._config_act_fn = _get_act_fn(config.hidden_act)
        self.ffn_fc1_factor = ffn_fc1_factor or (2 if config.gated_linear_unit else 1)

        self.linear_fc1 = LinearWithMatmul(
            config.hidden_size,
            config.intermediate_size * self.ffn_fc1_factor,
            bias=config.attention_bias,
        )
        self.linear_fc1._init_role = "input"
        if config.perform_initialization:
            config._standalone_init_weights(self.linear_fc1)

        if activation_func is not None:
            self.activation_func = activation_func
        elif config.gated_linear_unit:
            if self._config_act_fn is F.silu and config.use_fused_swiglu:
                self.activation_func = self._swiglu_fallback
            else:
                def glu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    return self._config_act_fn(x[0]) * x[1]
                self.activation_func = glu
        else:
            self.activation_func = self._config_act_fn

        self.linear_fc2 = LinearWithMatmul(
            config.intermediate_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.linear_fc2._init_role = "output"
        if config.perform_initialization:
            config._standalone_init_weights(self.linear_fc2)

    @staticmethod
    def _swiglu_fallback(x):
        x = torch.chunk(x, 2, dim=-1)
        return F.silu(x[0]) * x[1]

    def get_sandwich_post_norm_scale(self):
        return self.config.ffn_post_norm_scale

    def forward(self, hidden_states):
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
        if bias_parallel is not None:
            intermediate_parallel = intermediate_parallel + bias_parallel
        # npu_swiglu only works on NPU device; fall back to CPU implementation
        if (self.config.use_fused_swiglu
                and self.config.gated_linear_unit
                and self._config_act_fn is F.silu
                and HAS_NPU
                and hidden_states.device.type != 'cpu'):
            intermediate_parallel = torch_npu.npu_swiglu(intermediate_parallel, dim=-1)
        else:
            intermediate_parallel = self.activation_func(intermediate_parallel)
        output, output_bias = self.linear_fc2(intermediate_parallel)
        return output, output_bias
