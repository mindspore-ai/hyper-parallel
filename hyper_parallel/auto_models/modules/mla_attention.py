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
"""Multi-head latent attention module using the Hyper projection layout."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

# This package provides PyTorch-specific high-performance modules.
# pylint: disable=forbidden-backend-import
import torch  # pylint: disable=forbidden-backend-import
from torch import nn
from transformers.core_model_loading import WeightConverter

from hyper_parallel.auto_models.components.checkpoint import ConcatenateWithSections
from hyper_parallel.auto_models.components.model_transform import module_replacement
from hyper_parallel.auto_models.ops import apply_rotary_pos_emb, apply_rotary_pos_emb_interleave
from hyper_parallel.auto_models.ops import npu_fusion_attention_forward


@module_replacement
class MLAAttention(nn.Module):
    """Transformers-compatible MLA using a fused latent projection.

    ``q_a_proj`` and ``kv_a_proj_with_mqa`` are fused into ``linear_qkv``.
    Unchanged child modules retain their source names and checkpoint layout.
    Construction only creates the target structure; ``make_transforms``
    declares the checkpoint conversion.
    """

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        attention_interface: Callable[..., tuple[torch.Tensor, torch.Tensor | None]] = (
            npu_fusion_attention_forward
        ),
    ) -> None:
        """Build the high-performance module from a Transformers MLA module."""
        super().__init__()
        del module_fqn, context
        self.attention_interface = attention_interface
        required = (
            "q_a_proj",
            "q_a_layernorm",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_a_layernorm",
            "kv_b_proj",
            "o_proj",
        )
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise TypeError(f"MLAAttention source module is missing: {missing}")

        config = module.config
        self.config = config
        self.layer_idx = getattr(module, "layer_idx", getattr(module, "layer_number", None))
        self.num_heads = getattr(module, "num_heads", config.num_attention_heads)
        self.num_key_value_heads = getattr(
            module,
            "num_key_value_heads",
            getattr(config, "num_key_value_heads", self.num_heads),
        )
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.q_lora_rank = getattr(module, "q_lora_rank", config.q_lora_rank)
        self.kv_lora_rank = getattr(module, "kv_lora_rank", config.kv_lora_rank)
        if self.q_lora_rank is None:
            raise ValueError("MLAAttention requires the Q-LoRA projection path")
        self.qk_rope_head_dim = getattr(module, "qk_rope_head_dim", config.qk_rope_head_dim)
        self.qk_nope_head_dim = getattr(module, "qk_nope_head_dim", config.qk_nope_head_dim)
        self.v_head_dim = getattr(module, "v_head_dim", config.v_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = getattr(module, "scaling", self.qk_head_dim**-0.5)
        self.attention_dropout = getattr(module, "attention_dropout", 0.0)
        if isinstance(self.attention_dropout, nn.Dropout):
            self.attention_dropout = self.attention_dropout.p
        self.is_causal = getattr(module, "is_causal", True)
        self.sliding_window = getattr(
            module, "sliding_window", getattr(config, "sliding_window", None)
        )
        self.rotary_interleaved = bool(
            getattr(module, "rotary_interleaved", getattr(config, "rope_interleave", False))
        )
        if getattr(module, "param_sink_number", 0) > 0:
            raise ValueError(
                "MLAAttention only implements standard MLA; parameter-sink attention "
                "must use the sink high-performance interfaces."
            )
        self.use_mome = bool(getattr(module, "use_mome", False))
        if self.use_mome:
            raise NotImplementedError(
                "MLAAttention only implements standard MLA; MOME must use the "
                "aggregate_hidden high-performance path."
            )

        q_a = module.q_a_proj
        kv_a = module.kv_a_proj_with_mqa
        self._q_latent_output_size = q_a.out_features
        self._kv_latent_output_size = kv_a.out_features
        can_fuse = (
            q_a.in_features == kv_a.in_features
            and q_a.weight.requires_grad == kv_a.weight.requires_grad
            and (q_a.bias is None) == (kv_a.bias is None)
            and (q_a.bias is None or q_a.bias.requires_grad == kv_a.bias.requires_grad)
        )
        if not can_fuse:
            raise ValueError(
                "Q and KV latent projections cannot be represented by one fused projection"
            )
        self.linear_qkv = nn.Linear(
            q_a.in_features,
            q_a.out_features + kv_a.out_features,
            bias=q_a.bias is not None,
            device=q_a.weight.device,
            dtype=q_a.weight.dtype,
        )
        self.linear_qkv.weight.requires_grad_(q_a.weight.requires_grad)
        if q_a.bias is not None:
            self.linear_qkv.bias.requires_grad_(q_a.bias.requires_grad)

        self.q_a_layernorm = module.q_a_layernorm
        self.kv_a_layernorm = module.kv_a_layernorm
        self.q_b_proj = module.q_b_proj
        self.kv_b_proj = module.kv_b_proj
        self.o_proj = module.o_proj

        self.train(module.training)

    def make_transforms(self) -> list[WeightConverter]:
        """Describe reversible source-checkpoint to high-performance conversion."""
        transforms: list[WeightConverter] = [
            WeightConverter(
                source_patterns=[
                    "q_a_proj.weight",
                    "kv_a_proj_with_mqa.weight",
                ],
                target_patterns="linear_qkv.weight",
                operations=[
                    ConcatenateWithSections(
                        sections=(
                            self._q_latent_output_size,
                            self._kv_latent_output_size,
                        ),
                        dim=0,
                    )
                ],
            ),
        ]
        if self.linear_qkv.bias is not None:
            transforms.insert(
                1,
                WeightConverter(
                    source_patterns=[
                        "q_a_proj.bias",
                        "kv_a_proj_with_mqa.bias",
                    ],
                    target_patterns="linear_qkv.bias",
                    operations=[
                        ConcatenateWithSections(
                            sections=(
                                self._q_latent_output_size,
                                self._kv_latent_output_size,
                            ),
                            dim=0,
                        )
                    ],
                ),
            )
        return transforms

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        actual_seq_len: torch.Tensor | Sequence[int] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run MLA with the same external contract as Transformers attention."""
        batch_size, seq_length = hidden_states.shape[:-1]
        latent_states = self.linear_qkv(hidden_states)
        q_latent, kv_nope, k_rot = torch.split(
            latent_states,
            (self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim),
            dim=-1,
        )
        q_resid = self.q_a_layernorm(q_latent)
        q_states = self.q_b_proj(q_resid).view(
            batch_size,
            seq_length,
            self.num_heads,
            self.qk_head_dim,
        )
        q_pass, q_rot = torch.split(
            q_states,
            (self.qk_nope_head_dim, self.qk_rope_head_dim),
            dim=-1,
        )

        kv_nope = self.kv_a_layernorm(kv_nope).view(
            batch_size, 1, seq_length, self.kv_lora_rank
        )
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        q_rot = q_rot.transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            if self.rotary_interleaved:
                q_rot, k_rot = apply_rotary_pos_emb_interleave(
                    q_rot, k_rot, cos, sin
                )
            else:
                q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        if past_key_values is not None:
            kv_nope, k_rot = past_key_values.update(
                kv_nope,
                k_rot,
                self.layer_idx,
            )

        kv_seq_length = kv_nope.shape[2]
        kv_states = self.kv_b_proj(kv_nope).view(
            batch_size,
            kv_seq_length,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        ).transpose(1, 2)
        k_nope, value_states = torch.split(
            kv_states,
            (self.qk_nope_head_dim, self.v_head_dim),
            dim=-1,
        )
        k_rot = k_rot.expand(-1, self.num_heads, -1, -1)
        query_states = torch.cat((q_pass.transpose(1, 2), q_rot), dim=-1)
        key_states = torch.cat((k_nope, k_rot), dim=-1)

        attn_output, attn_weights = self.attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            actual_seq_len=actual_seq_len,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
