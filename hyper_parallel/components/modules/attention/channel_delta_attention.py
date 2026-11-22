# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Complete channel-wise delta-attention family with fused convolution and unbounded negative-softplus decay."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.components.functional.kimi_delta_attention import chunk_kda as chunk_channel_delta_rule
from hyper_parallel.components.functional.linear_attention_convolution import packed_causal_conv1d
from hyper_parallel.components.modules.attention.linear_attention_utils.sequence_layout import resolve_packed_sequence
from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.replacement import module_replacement


class ChannelForgetGate(nn.Module):
    """Low-rank channel logits followed by the source's unbounded negative-softplus decay."""

    def __init__(self, config: Any, module: nn.Module | None = None) -> None:
        """Preserve nested gate keys, bare parameter shapes and low-rank matrix boundaries."""
        super().__init__()
        self.head_dim, self.num_heads = config.linear_head_dim, config.linear_num_heads
        width = self.head_dim * self.num_heads
        self.f_a_proj = (
            module.f_a_proj if module is not None else nn.Linear(config.hidden_size, self.head_dim, bias=False)
        )
        self.f_b_proj = module.f_b_proj if module is not None else nn.Linear(self.head_dim, width, bias=False)
        self.dt_bias = module.dt_bias if module is not None else nn.Parameter(torch.empty(width))
        self.A_log = module.A_log if module is not None else nn.Parameter(torch.empty(1, 1, self.num_heads, 1))
        if module is None:
            self.reset_parameters()
        else:
            self.training = module.training

    def reset_parameters(self) -> None:
        """Initialize only family-owned time/decay state after empty materialization."""
        with torch.no_grad():
            self.A_log.copy_(self.A_log.uniform_(1, 16).log())
            self.dt_bias.uniform_(math.log(1e-3), math.log(1e-1))
            dt = self.dt_bias.exp().clamp_min(1e-4)
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def make_transforms(self) -> list:
        """Nested source gate parameters retain their keys and shapes."""
        return []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Use FP32 channel decay without the lower-bounded K3 gate transformation."""
        logits = self.f_b_proj(self.f_a_proj(hidden_states))
        logits = (logits.float() + self.dt_bias.float().view(1, 1, -1)).view(
            *hidden_states.shape[:2], self.num_heads, self.head_dim
        )
        softplus = torch.where(logits > 20.0, logits, torch.log(1.0 + logits.exp()))
        return -self.A_log.float().exp() * softplus


class ChannelRMSNormGated(nn.Module):
    """FP32 direct-scale RMSNorm followed by a sigmoid output gate."""

    def __init__(self, config: Any, module: nn.Module | None = None) -> None:
        """Adopt the source scale and epsilon without changing its checkpoint identity."""
        super().__init__()
        self.weight = module.weight if module is not None else nn.Parameter(torch.ones(config.linear_head_dim))
        self.variance_epsilon = module.variance_epsilon if module is not None else config.rms_norm_eps
        if module is not None:
            self.training = module.training

    def reset_parameters(self) -> None:
        """Reset direct normalization scale."""
        nn.init.ones_(self.weight)

    def make_transforms(self) -> list:
        """Normalization keeps the source weight representation."""
        return []

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Apply normalization and output gating before the source dtype cast."""
        dtype = hidden_states.dtype
        values = hidden_states.float()
        values = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (values * self.weight.float() * gate.float().sigmoid()).to(dtype)


class ChannelDeltaAttentionModule(nn.Module):
    """Own source projections, fused QKV convolution, channel recurrence and low-rank output gating."""

    def __init__(
        self, *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None
    ) -> None:
        """Construct dimensions directly or adopt a complete fused-convolution source contract."""
        super().__init__()
        del module_fqn
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        if module is not None:
            nearest = tuple((context or {}).get("module_configs", ()))
            config = getattr(module, "config", None) or next(
                (item for item in nearest if hasattr(item, "linear_num_heads")), None
            )
            if config is None:
                raise ValueError("Channel delta source adoption requires its nearest linear-attention config")
        self.config = config
        self.hidden_size, self.num_heads, self.head_dim = (
            config.hidden_size,
            config.linear_num_heads,
            config.linear_head_dim,
        )
        self.qkv_dim, self.conv_kernel_size = self.num_heads * self.head_dim, config.linear_conv_kernel_dim
        self.layer_idx = getattr(module, "layer_idx", getattr(config, "layer_idx", 0))
        self._validate_source(config, module)
        for name in ("q_proj", "k_proj", "v_proj"):
            setattr(
                self,
                name,
                getattr(module, name) if module is not None else nn.Linear(self.hidden_size, self.qkv_dim, bias=False),
            )
        self.conv1d = (
            module.conv1d
            if module is not None
            else nn.Conv1d(
                3 * self.qkv_dim,
                3 * self.qkv_dim,
                self.conv_kernel_size,
                groups=3 * self.qkv_dim,
                padding=self.conv_kernel_size - 1,
                bias=False,
            )
        )
        self.forget_gate = ChannelForgetGate(config, None if module is None else module.forget_gate)
        self.b_proj = module.b_proj if module is not None else nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.g_a_proj = (
            module.g_a_proj if module is not None else nn.Linear(self.hidden_size, self.head_dim, bias=False)
        )
        self.g_b_proj = module.g_b_proj if module is not None else nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.o_norm = ChannelRMSNormGated(config, None if module is None else module.o_norm)
        self.o_proj = module.o_proj if module is not None else nn.Linear(self.qkv_dim, self.hidden_size, bias=False)
        if module is not None:
            self.training = module.training

    def _validate_source(self, config: Any, module: nn.Module | None) -> None:
        """Validate dimensions and the complete fused-convolution source state contract."""
        for name in ("hidden_size", "num_heads", "head_dim", "conv_kernel_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Channel delta {name} must be a positive integer")
        if config.hidden_act not in ("silu", "swish"):
            raise ValueError("Channel delta fused convolution requires a SiLU activation")
        if module is not None:
            expected = {
                "q_proj",
                "k_proj",
                "v_proj",
                "conv1d",
                "forget_gate",
                "b_proj",
                "g_a_proj",
                "g_b_proj",
                "o_norm",
                "o_proj",
            }
            if set(dict(module.named_children())) != expected or list(module.named_parameters(recurse=False)):
                raise ValueError(
                    "Source does not match fused-convolution, unbounded-decay, low-rank-gate channel delta"
                )

    def make_transforms(self) -> list:
        """All logical matrices, fused convolution channels and nested source keys remain unchanged."""
        return []

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Any = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute training recurrence with independent packed-document boundaries and padding."""
        if cache_params is not None:
            raise ValueError("Channel delta training member does not support recurrent/conv cache")
        flat = hidden_states.ndim == 2
        if flat:
            hidden_states = hidden_states.unsqueeze(0)
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("Channel delta expects [batch, sequence, hidden] or [tokens, hidden]")
        if attention_mask is not None:
            if attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError("Channel delta padding mask must cover the local batch/sequence")
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
        batch, sequence = hidden_states.shape[:2]
        packed = resolve_packed_sequence(kwargs, sequence)
        mixed = torch.cat((self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)), dim=-1)
        if packed is None:
            mixed = F.silu(self.conv1d(mixed.transpose(1, 2))[:, :, :sequence]).transpose(1, 2)
        else:
            mixed = packed_causal_conv1d(mixed, self.conv1d, packed)
        query, key, value = (
            item.reshape(batch, sequence, self.num_heads, self.head_dim) for item in mixed.split(self.qkv_dim, dim=-1)
        )
        output, _ = chunk_channel_delta_rule(
            query,
            key,
            value,
            self.forget_gate(hidden_states),
            self.b_proj(hidden_states).sigmoid(),
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None if packed is None else packed.local_offsets,
        )
        gate = self.g_b_proj(self.g_a_proj(hidden_states)).reshape(batch, sequence, self.num_heads, self.head_dim)
        output = self.o_norm(output, gate).reshape(batch, sequence, self.qkv_dim)
        output = self.o_proj(output)
        return output.squeeze(0) if flat else output


def validate_channel_delta(*, config: Any, module: nn.Module | None, context: Any, member: Any) -> None:
    """Reject parallel modes whose full channel-state execution has not been bound and validated."""
    del config, module, context
    if member.axis_sizes["tp"] > 1 or member.axis_sizes["cp"] > 1:
        raise NotImplementedError("Channel delta family currently implements local training only")


def replicated_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Own the local member's complete parameter tree without inferring projection names."""
    del module_fqn, member
    placements = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    return ModuleShardingSpec(
        params={name: dict(placements) for name, _ in module.named_parameters()},
        in_src={},
        in_dst={},
        out_src=None,
        out_dst=None,
        is_boundary=True,
    )


@module_replacement
def ChannelDeltaAttention(  # pylint: disable=invalid-name
    *,
    config: Any = None,
    module: nn.Module | None = None,
    module_fqn: str = "",
    context: Any = None,
    mesh: Any = None,
) -> nn.Module:
    """Construct channel-wise delta attention with its own source/signature/state contract."""
    return build_family(
        ChannelDeltaAttention, config=config, module=module, module_fqn=module_fqn, context=context, mesh=mesh
    )


register_module_family(
    ModuleFamily(
        name="channel_delta_attention",
        baseline=ChannelDeltaAttention,
        variants={"local": (
            "hyper_parallel.components.modules.attention.channel_delta_attention."
            "ChannelDeltaAttentionModule"
        )},
        default_variant="local",
        sharding=replicated_sharding,
        validate=validate_channel_delta,
    )
)
