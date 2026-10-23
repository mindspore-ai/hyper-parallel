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

"""Random-weight MoE Transformer integration example with simplified GQA and four-stream mHC."""

from __future__ import annotations

__all__ = [
    "MoeTransformerBlock",
    "MoeTransformerConfig",
    "MoeTransformerModel",
]

from dataclasses import dataclass
from typing import Any

# This is a Torch-only executable example, not a platform-agnostic library module.
import torch  # pylint: disable=forbidden-backend-import
import torch.nn.functional as F  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.rotary_embedding import apply_rotary_pos_emb
from hyper_parallel.components.modules import RMSNorm
from hyper_parallel.core.multicore import MegaMoeExperts, MhcState
from hyper_parallel.core.multicore.examples.mhc import MhcBoundary
from hyper_parallel.core.multicore.examples.moe import MoeTransformerBlock
from hyper_parallel.core.multicore.modules.mega_moe.spec import (
    _resolve_capacity_factors,
)


@dataclass(frozen=True)
class MoeTransformerConfig:
    """Synthetic MoE Transformer model configuration with four-stream mHC.

    The example intentionally models one TP1/EP8 optimizer-step workload.
    Route stress, fusion matrices, and isolated expert performance belong to
    the corresponding distributed system tests.
    """

    vocab_size: int = 32000
    hidden_size: int = 5120
    num_layers: int = 2
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    max_seq_len: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    num_experts: int = 48
    top_k: int = 8
    intermediate_size: int = 1792
    shared_expert_intermediate_size: int = 1792
    routed_scaling_factor: float = 1.0
    local_num_tokens: int = 4096
    initial_capacity_factor: float | None = None
    ep_size: int = 8
    dispatch_mode: str = "push"
    capacity_growth_factor: float | None = None

    def __post_init__(self) -> None:
        """Validate the fixed optimizer-step topology."""
        initial, growth = _resolve_capacity_factors(
            self.dispatch_mode, self.initial_capacity_factor, self.capacity_growth_factor)
        object.__setattr__(self, "initial_capacity_factor", initial)
        object.__setattr__(self, "capacity_growth_factor", growth)
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_seq_len": self.max_seq_len,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "intermediate_size": self.intermediate_size,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_size,
            "local_num_tokens": self.local_num_tokens,
            "ep_size": self.ep_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})."
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})."
            )
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("Attention head dimension must be even for rotary embeddings.")
        if self.num_experts % self.ep_size:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by ep_size ({self.ep_size})."
            )
        if self.top_k > self.num_experts:
            raise ValueError(
                f"top_k ({self.top_k}) cannot exceed num_experts ({self.num_experts})."
            )
        if self.max_seq_len > self.local_num_tokens:
            raise ValueError(
                f"max_seq_len ({self.max_seq_len}) cannot exceed local_num_tokens "
                f"({self.local_num_tokens}) in the fixed batch-one workload."
            )


class MoeTransformerAttention(nn.Module):
    """Simplified grouped-query attention for the integration example."""

    def __init__(self, config: MoeTransformerConfig) -> None:
        """Initialize QKV, output, and rotary projections."""
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = True
        inverse_frequency = config.rope_theta ** (
            -torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        )
        angles = torch.outer(torch.arange(config.max_seq_len, dtype=torch.float32), inverse_frequency)
        angles = torch.cat((angles, angles), dim=-1)
        self.register_buffer("rotary_cos", angles.cos(), persistent=False)
        self.register_buffer("rotary_sin", angles.sin(), persistent=False)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run fused causal grouped-query attention.

        Args:
            hidden_states: Input activations preserving the batch, sequence and hidden dimensions.
            position_ids: Position indices used by the rotary embedding.
        """
        batch_size, sequence_length, _ = hidden_states.shape
        query = (
            self.q_proj(hidden_states)
            .view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        key = (
            self.k_proj(hidden_states)
            .view(
                batch_size,
                sequence_length,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        value = (
            self.v_proj(hidden_states)
            .view(
                batch_size,
                sequence_length,
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        cos = self.rotary_cos[position_ids].to(hidden_states.dtype)
        sin = self.rotary_sin[position_ids].to(hidden_states.dtype)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if hidden_states.device.type == "npu":
            # Import the optional Ascend backend only when executing on NPU.
            from hyper_parallel.components.functional.npu_fusion_attention import (  # pylint: disable=C0415
                npu_fusion_attention_forward,
            )
            output, _ = npu_fusion_attention_forward(
                self, query, key, value, attention_mask=None, dropout=0.0, scaling=self.scaling,
            )
        else:
            repeats = self.num_heads // self.num_key_value_heads
            output = F.scaled_dot_product_attention(  # pylint: disable=not-callable
                query, key.repeat_interleave(repeats, dim=1), value.repeat_interleave(repeats, dim=1),
                is_causal=True, scale=self.scaling,
            ).transpose(1, 2).contiguous()
        return self.o_proj(output.reshape(batch_size, sequence_length, -1))


class MoeTransformerDecoderLayer(nn.Module):
    """One mHC attention and MoE decoder layer."""

    def __init__(
        self,
        config: MoeTransformerConfig,
        ep_group: Any | None,
        mega_kernels: tuple[str, ...],
    ) -> None:
        """Initialize one attention and MoE decoder layer."""
        super().__init__()
        self.attention_mhc = MhcBoundary(
            config.hidden_size, config.rms_norm_eps, use_mega_mhc="mhc" in mega_kernels,
        )
        self.ffn_mhc = MhcBoundary(
            config.hidden_size, config.rms_norm_eps, use_mega_mhc="mhc" in mega_kernels,
        )
        self.self_attn = MoeTransformerAttention(config)
        self.mlp = MoeTransformerBlock(config, ep_group, mega_kernels)

    def forward(self, state: MhcState, position_ids: torch.Tensor) -> MhcState:
        """Carry shifted residual mappings across attention, MoE, and decoder layers.

        Args:
            state: Explicit pending output, residual streams and mixing coefficients.
            position_ids: Position indices used by the rotary embedding.
        """
        attention = self.attention_mhc(state)
        state = attention.next_state(self.self_attn(attention.hidden_states, position_ids))
        feed_forward = self.ffn_mhc(state)
        return feed_forward.next_state(self.mlp(feed_forward.hidden_states))


class MoeTransformerModel(nn.Module):
    """Random-weight text-only MoE Transformer demo with four-stream mHC."""

    def __init__(
        self,
        config: MoeTransformerConfig,
        *,
        ep_group: Any | None = None,
        mega_kernels: tuple[str, ...] = ("moe", "gate", "mhc"),
    ) -> None:
        """Initialize all decoder layers and their shared execution group."""
        super().__init__()
        unknown = set(mega_kernels) - {"moe", "gate", "mhc"}
        if unknown:
            raise ValueError(f"Unsupported Mega kernels: {sorted(unknown)}.")
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            MoeTransformerDecoderLayer(config, ep_group, mega_kernels)
            for _ in range(config.num_layers)
        )
        MegaMoeExperts.share_execution_resources(
            layer.mlp.experts for layer in self.layers
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Return token logits and optional next-token loss.

        Args:
            input_ids: Rank-local token IDs shaped [batch, sequence].
            labels: Optional next-token targets with the same leading dimensions.
        """
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(
            hidden_states.shape[1],
            dtype=torch.int64,
            device=hidden_states.device,
        )
        state = MhcState.initialize(hidden_states)
        for layer in self.layers:
            state = layer(state, position_ids)
        hidden_states = state.finish()
        logits = self.lm_head(self.norm(hidden_states))
        loss = None
        if labels is not None:
            shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:]
            loss = F.cross_entropy(
                logits.float().view(-1, logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}

    def expert_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return local routed-expert parameters excluded from dense all-reduce."""
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.mlp.experts.parameters())
        return tuple(parameters)

    def close(self) -> None:
        """Release the shared MegaMoe runtime after all queued work finishes."""
        for layer in self.layers:
            layer.mlp.experts.close()
