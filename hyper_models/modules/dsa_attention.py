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
"""Model-compatible DSA modules using the Hyper high-performance operators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# This package provides PyTorch-specific high-performance modules.
# pylint: disable=forbidden-backend-import
import torch  # pylint: disable=forbidden-backend-import
from torch import nn
from torch.nn import functional as F
from transformers.core_model_loading import WeightConverter

from hyper_models.components.checkpoint import ConcatenateWithSections
from hyper_models.ops import aggregate_hidden
from hyper_models.ops import apply_rotary_pos_emb, apply_rotary_pos_emb_interleave
from hyper_models.ops import (
    dsa_indexer,
    dsa_kl_loss,
    dsa_sparse_attention,
    dsa_sparse_attention_rescale,
)


def apply_mome(
    hidden_states: torch.Tensor,
    mome_mask: torch.Tensor | None,
    convolution: nn.Conv1d,
    *,
    fused: bool,
) -> torch.Tensor:
    """Apply Pangu's masked causal depthwise convolution and residual."""
    if mome_mask is None:
        raise ValueError("mome_mask is required when MOME is enabled")
    if mome_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"mome_mask must have shape {hidden_states.shape[:2]}, "
            f"but got {tuple(mome_mask.shape)}"
        )

    mome_mask = mome_mask.to(device=hidden_states.device, dtype=torch.bool)
    padding = convolution.kernel_size[0] - 1
    padded_states = F.pad(hidden_states, (0, 0, padding, 0))
    if fused:
        padded_mask = F.pad(mome_mask, (padding, 0), value=False)
        weight = convolution.weight.squeeze(1).transpose(0, 1)
        mixed_states = aggregate_hidden(
            padded_states.transpose(0, 1).contiguous(),
            weight,
            padded_mask,
        )
        mixed_states = mixed_states[padding:].transpose(0, 1).contiguous()
    else:
        mixed_states = convolution(padded_states.transpose(1, 2)).transpose(1, 2)
        mixed_states = mixed_states * mome_mask.unsqueeze(-1).to(mixed_states.dtype)
    return hidden_states + mixed_states


class _AuxLossAutoScaler(torch.autograd.Function):
    """Inject the auxiliary DSA loss gradient without changing output."""

    main_loss_backward_scale = torch.tensor(1.0)

    @staticmethod
    def forward(ctx: Any, output: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
        """Return the main output and retain the auxiliary loss for backward."""
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pass through the main gradient and inject the auxiliary gradient."""
        (aux_loss,) = ctx.saved_tensors
        scale = _AuxLossAutoScaler.main_loss_backward_scale
        return grad_output, torch.ones_like(aux_loss) * scale

    @staticmethod
    def set_loss_scale(scale: torch.Tensor) -> None:
        """Set the loss scale used for the injected auxiliary gradient."""
        _AuxLossAutoScaler.main_loss_backward_scale = scale


class DeepseekV32DSAAttention(nn.Module):
    """NPU DSA replacement for Transformers ``DeepseekV32Attention``.

    It supports causal training and packed sequences. Packed boundaries must
    be supplied through ``actual_seq_len``. The optional ``attention_mask`` is
    assumed to be the standard causal mask created by Transformers; padding,
    custom attention masks, dropout, and KV cache are not supported.
    """

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Build the high-performance module from a Transformers DSA module."""
        super().__init__()
        del module_fqn, context
        required = (
            "q_a_proj",
            "q_a_layernorm",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_a_layernorm",
            "kv_b_proj",
            "o_proj",
            "indexer",
        )
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise TypeError(f"DeepseekV32DSAAttention source module is missing: {missing}")
        indexer_required = ("wq_b", "wk", "k_norm", "weights_proj")
        indexer_missing = [
            name for name in indexer_required if not hasattr(module.indexer, name)
        ]
        if indexer_missing:
            raise TypeError(
                f"DeepseekV32DSAAttention source indexer is missing: {indexer_missing}"
            )

        config = module.config
        self.config = config
        if hasattr(config, "indexer_types"):
            raise TypeError(
                "DeepseekV32DSAAttention supports the DeepSeek-V3.2 DSA contract; "
                "GLM-MOE-DSA shared indexers and three-value forward result require "
                "a dedicated replacement"
            )
        self.layer_idx = getattr(module, "layer_idx", getattr(module, "layer_number", None))
        self.num_heads = getattr(module, "num_heads", config.num_attention_heads)
        self.q_lora_rank = getattr(module, "q_lora_rank", config.q_lora_rank)
        self.kv_lora_rank = getattr(module, "kv_lora_rank", config.kv_lora_rank)
        self.qk_rope_head_dim = getattr(
            module, "qk_rope_head_dim", config.qk_rope_head_dim
        )
        self.qk_nope_head_dim = getattr(
            module, "qk_nope_head_dim", config.qk_nope_head_dim
        )
        self.v_head_dim = getattr(module, "v_head_dim", config.v_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.q_lora_rank is None:
            raise ValueError("DeepseekV32DSAAttention requires the Q-LoRA projection path")
        self.scaling = getattr(module, "scaling", self.qk_head_dim**-0.5)
        attention_dropout = getattr(
            module, "attention_dropout", getattr(config, "attention_dropout", 0.0)
        )
        if isinstance(attention_dropout, nn.Dropout):
            attention_dropout = attention_dropout.p
        if attention_dropout != 0.0:
            raise ValueError(
                "DeepseekV32DSAAttention requires attention_dropout=0 because "
                "the sparse-attention operator does not apply dropout"
            )
        source_rotary_interleaved = getattr(module, "rotary_interleaved", None)
        if source_rotary_interleaved is None:
            source_rotary_interleaved = getattr(config, "rope_interleave", None)
        if source_rotary_interleaved is None:
            source_rotary_interleaved = getattr(config, "rope_interleaved", None)
        # Transformers DSA uses interleaved RoPE for the main MLA path when
        # the source does not expose a mode, while its indexer remains half-split.
        self.rotary_interleaved = bool(
            True if source_rotary_interleaved is None else source_rotary_interleaved
        )
        self.index_rotary_interleaved = bool(
            getattr(
                module.indexer,
                "rotary_interleaved",
                getattr(config, "index_rope_interleaved", False),
            )
        )
        self.index_head_dim = getattr(module.indexer, "head_dim", config.index_head_dim)
        self.num_index_heads = getattr(
            module.indexer,
            "n_heads",
            getattr(
                config,
                "index_n_heads",
                getattr(config, "index_num_attention_heads", None),
            ),
        )
        if self.num_index_heads is None:
            raise ValueError("DeepseekV32DSAAttention requires the number of indexer heads")
        self.index_topk = getattr(module.indexer, "index_topk", config.index_topk)
        self.dsa_loss_coeff = getattr(config, "dsa_loss_coeff", 0.0)
        self.freeze_dsa = bool(
            getattr(config, "freeze_DSA", getattr(config, "freeze_dsa", False))
        )
        if getattr(module, "param_sink_number", 0) > 0:
            raise ValueError(
                "DeepseekV32DSAAttention does not support the parameter-sink path"
            )
        if getattr(module, "use_mome", False):
            raise ValueError("DeepseekV32DSAAttention does not support the MOME path")

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
        self.indexer = module.indexer
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
                    source_patterns=["q_a_proj.bias", "kv_a_proj_with_mqa.bias"],
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

    @staticmethod
    def _get_actual_seq_len(
        actual_seq_len: torch.Tensor | Sequence[int] | None,
        batch_size: int,
        seq_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalize cumulative sequence lengths for the TND custom operators."""
        if isinstance(actual_seq_len, torch.Tensor):
            return actual_seq_len.to(device=device, dtype=torch.int32)
        if actual_seq_len is not None:
            return torch.tensor(actual_seq_len, device=device, dtype=torch.int32)
        return torch.arange(
            seq_length,
            (batch_size + 1) * seq_length,
            seq_length,
            dtype=torch.int32,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        position_ids: torch.Tensor | None = None,
        actual_seq_len: torch.Tensor | Sequence[int] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Run causal or packed DSA with the NPU sparse-attention kernels."""
        batch_size, seq_length = hidden_states.shape[:-1]
        if past_key_values is not None:
            raise NotImplementedError("DeepseekV32DSAAttention does not support KV cache")
        if position_ids is not None and (
            position_ids.ndim != 2
            or position_ids.shape[0] not in (1, batch_size)
            or position_ids.shape[1] != seq_length
        ):
            raise ValueError(
                "position_ids must have shape [1 or batch_size, sequence_length]"
            )
        if attention_mask is not None:
            expected_shape = (
                hidden_states.shape[0],
                1,
                hidden_states.shape[1],
                hidden_states.shape[1],
            )
            if tuple(attention_mask.shape) != expected_shape:
                raise ValueError(
                    "DeepseekV32DSAAttention supports only a Transformers 4D causal mask; "
                    "padding and custom masks are not supported"
                )
        output_attentions = kwargs.pop("output_attentions", False)
        if output_attentions:
            raise NotImplementedError(
                "DeepseekV32DSAAttention does not expose attention weights"
            )
        latent_states = self.linear_qkv(hidden_states)
        q_latent, kv_nope, k_rot = torch.split(
            latent_states,
            (self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim),
            dim=-1,
        )
        q_resid = self.q_a_layernorm(q_latent)
        q_states = self.q_b_proj(q_resid).view(
            batch_size, seq_length, self.num_heads, self.qk_head_dim
        )
        q_pass, q_rot = torch.split(
            q_states, (self.qk_nope_head_dim, self.qk_rope_head_dim), dim=-1
        )

        kv_weight = self.kv_b_proj.weight.view(
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        key_up_weight = kv_weight[:, : self.qk_nope_head_dim]
        absorbed_query_states = q_pass.permute(2, 0, 1, 3).reshape(
            self.num_heads, batch_size * seq_length, self.qk_nope_head_dim
        )
        absorbed_query_states = torch.bmm(
            absorbed_query_states, key_up_weight
        ).view(
            self.num_heads, batch_size, seq_length, self.kv_lora_rank
        ).permute(1, 2, 0, 3)

        kv_nope = self.kv_a_layernorm(kv_nope).view(
            batch_size, seq_length, 1, self.kv_lora_rank
        )
        k_rot = k_rot.view(batch_size, seq_length, 1, self.qk_rope_head_dim)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            if self.rotary_interleaved:
                q_rot, k_rot = apply_rotary_pos_emb_interleave(
                    q_rot, k_rot, cos, sin, unsqueeze_dim=2
                )
            else:
                q_rot, k_rot = apply_rotary_pos_emb(
                    q_rot, k_rot, cos, sin, unsqueeze_dim=2
                )

        index_query = self.indexer.wq_b(q_resid.detach()).view(
            batch_size,
            seq_length,
            self.num_index_heads,
            self.index_head_dim,
        )
        index_key = self.indexer.k_norm(
            self.indexer.wk(hidden_states.detach())
        ).unsqueeze(2)
        merge_weight = self.indexer.weights_proj(hidden_states.detach())
        merge_weight = (
            merge_weight
            * self.num_index_heads**-0.5
            * self.index_head_dim**-0.5
        )
        if position_embeddings is not None:
            index_query_rot, index_query_pass = torch.split(
                index_query,
                (self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim),
                dim=-1,
            )
            index_key_rot, index_key_pass = torch.split(
                index_key,
                (self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim),
                dim=-1,
            )
            if self.index_rotary_interleaved:
                index_query_rot, index_key_rot = apply_rotary_pos_emb_interleave(
                    index_query_rot,
                    index_key_rot,
                    cos,
                    sin,
                    unsqueeze_dim=2,
                )
            else:
                index_query_rot, index_key_rot = apply_rotary_pos_emb(
                    index_query_rot,
                    index_key_rot,
                    cos,
                    sin,
                    unsqueeze_dim=2,
                )
            index_query = torch.cat((index_query_rot, index_query_pass), dim=-1)
            index_key = torch.cat((index_key_rot, index_key_pass), dim=-1)

        actual_seq_len = self._get_actual_seq_len(
            actual_seq_len,
            batch_size,
            seq_length,
            hidden_states.device,
        )
        topk_indices, index_query_tnd, index_key_tnd, merge_weight_tnd = dsa_indexer(
            index_query,
            index_key,
            merge_weight,
            actual_seq_len,
            actual_seq_len,
            self.index_topk,
        )
        sparse_scale = self.scaling
        attn_output, softmax_max, softmax_sum = dsa_sparse_attention(
            absorbed_query_states,
            kv_nope,
            q_rot,
            k_rot,
            topk_indices,
            sparse_scale,
            actual_seq_len,
            actual_seq_len,
        )
        if self.training and not self.freeze_dsa and self.dsa_loss_coeff:
            query_tnd, key_tnd, q_rot_tnd, k_rot_tnd = (
                tensor.reshape(-1, tensor.shape[2], tensor.shape[3])
                for tensor in (absorbed_query_states, kv_nope, q_rot, k_rot)
            )
            aux_loss = dsa_kl_loss(
                index_query_tnd,
                index_key_tnd,
                merge_weight_tnd,
                query_tnd,
                key_tnd,
                topk_indices,
                softmax_max,
                softmax_sum,
                q_rot_tnd,
                k_rot_tnd,
                actual_seq_len,
                actual_seq_len,
                sparse_scale,
                self.dsa_loss_coeff,
            )
            attn_output = _AuxLossAutoScaler.apply(attn_output, aux_loss)

        attn_output = attn_output[..., : self.kv_lora_rank]
        value_up_weight = kv_weight[:, self.qk_nope_head_dim :].transpose(1, 2)
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(
            self.num_heads, batch_size * seq_length, self.kv_lora_rank
        )
        attn_output = torch.bmm(attn_output, value_up_weight)
        attn_output = attn_output.view(
            self.num_heads, batch_size, seq_length, self.v_head_dim
        ).permute(1, 2, 0, 3).reshape(batch_size, seq_length, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class DSAAttention(nn.Module):
    """NPU replacement for a Pangu-compatible DSA attention module.

    Packed boundaries are supplied through ``actual_seq_len``. Attention masks,
    KV reuse, and KV cache are not supported. The source parameter layout is
    preserved, so no checkpoint conversion is required.
    """

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Build the replacement from a compatible DSA attention module."""
        super().__init__()
        del module_fqn, context
        required = (
            "linear_qkv",
            "q_layernorm",
            "k_layernorm",
            "linear_qb",
            "linear_kvb",
            "linear_proj",
            "index_linear_qb",
            "index_linear_k",
            "index_k_layernorm",
            "linear_merge_weight",
        )
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise TypeError(f"DSAAttention source module is missing: {missing}")

        unsupported = {
            "mla_mm_split": bool(getattr(module, "mla_mm_split", False)),
            "dense warm-up": bool(getattr(module, "dsa_dense_warm_up", False)),
        }
        enabled = [name for name, value in unsupported.items() if value]
        if enabled:
            raise NotImplementedError(
                f"DSAAttention does not yet support: {', '.join(enabled)}"
            )

        self.config = module.config
        self.layer_number = module.layer_number
        self.num_heads = module.num_heads
        self.q_lora_rank = module.q_lora_rank
        self.kv_lora_rank = module.kv_lora_rank
        self.qk_rope_head_dim = module.qk_rope_head_dim
        self.qk_nope_head_dim = module.qk_nope_head_dim
        self.qk_head_dim = module.qk_head_dim
        self.v_head_dim = module.v_head_dim
        self.index_head_dim = module.index_head_dim
        self.num_index_heads = module.num_index_heads
        self.index_topk = module.index_topk
        self.rotary_interleaved = module.rotary_interleaved
        self.dsa_loss_coeff = module.dsa_loss_coeff
        self.freeze_dsa = module.freeze_dsa
        self.use_flash_attn = bool(getattr(module, "use_flash_attn", True))
        self.use_mome = bool(getattr(module, "use_mome", False))
        self.use_fused_mome = bool(getattr(module, "use_fused_mome", False))
        self.param_sink_number = int(getattr(module, "param_sink_number", 0))
        self.param_sink_scalar = getattr(module, "param_sink_scalar", None)
        self.apply_FA_rescale = bool(getattr(module, "apply_FA_rescale", False))
        self.attention_dropout = getattr(module, "attention_dropout", nn.Dropout(0.0))
        if self.param_sink_number > 0:
            if self.param_sink_scalar:
                raise NotImplementedError(
                    "DSAAttention does not support scalar parameter sink"
                )
            if not self.apply_FA_rescale:
                raise NotImplementedError(
                    "DSAAttention supports parameter sink through "
                    "apply_FA_rescale only"
                )

        self.linear_qkv = module.linear_qkv
        self.q_layernorm = module.q_layernorm
        self.k_layernorm = module.k_layernorm
        self.linear_qb = module.linear_qb
        self.linear_kvb = module.linear_kvb
        self.linear_proj = module.linear_proj
        self.index_linear_qb = module.index_linear_qb
        self.index_linear_k = module.index_linear_k
        self.index_k_layernorm = module.index_k_layernorm
        self.linear_merge_weight = module.linear_merge_weight
        if self.use_mome:
            self.qa_conv = module.qa_conv
            self.compresskv_conv = module.compresskv_conv
            self.o_conv = module.o_conv
        if self.param_sink_number > 0:
            self.param_sink_k_pe = module.param_sink_k_pe
            self.param_sink_compressed_kv = module.param_sink_compressed_kv
        self.train(module.training)

    def make_transforms(self) -> list[WeightConverter]:
        """Return no transforms because all source parameter names are preserved."""
        return []

    @staticmethod
    def _get_actual_seq_len(
        actual_seq_len: torch.Tensor | Sequence[int] | None,
        batch_size: int,
        seq_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalize cumulative sequence lengths for the TND custom operators."""
        if actual_seq_len is None:
            return torch.arange(
                seq_length,
                (batch_size + 1) * seq_length,
                seq_length,
                dtype=torch.int32,
                device=device,
            )
        return torch.as_tensor(actual_seq_len, dtype=torch.int32, device=device)

    def _prepare_param_sink(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct the DSA sink key and value in BSND layout."""
        sink_rotary_key = self.param_sink_k_pe.unsqueeze(0).unsqueeze(2).expand(
            batch_size, -1, -1, -1
        )
        sink_latent = self.param_sink_compressed_kv.unsqueeze(0).unsqueeze(2).expand(
            batch_size, -1, -1, -1
        )
        sink_latent = self.k_layernorm(sink_latent)
        sink_value = (
            F.pad(sink_latent, (0, self.qk_rope_head_dim))
            if self.use_flash_attn and self.qk_rope_head_dim > 0
            else sink_latent
        )
        sink_key = torch.cat((sink_latent, sink_rotary_key), dim=-1)
        return sink_key, sink_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        cache_position: torch.Tensor | None = None,
        actual_seq_len: torch.Tensor | Sequence[int] | None = None,
        kv_reuse_states: Any | None = None,
        output_attentions: bool = False,
        return_bias: bool = False,
        mome_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run DSA with its configured MOME and parameter-sink paths."""
        if attention_mask is not None:
            raise ValueError(
                "DSAAttention does not consume attention_mask; pass packed sequence "
                "boundaries through actual_seq_len"
            )
        if kv_reuse_states is not None:
            raise NotImplementedError("DSAAttention does not support KV reuse")
        if past_key_values is not None or cache_position is not None:
            raise NotImplementedError("DSAAttention does not support KV cache")
        if output_attentions:
            raise NotImplementedError("DSAAttention does not expose attention weights")

        batch_size, seq_length = hidden_states.shape[:-1]
        latent_states, _ = self.linear_qkv(hidden_states)
        q_latent, kv_nope, k_rot = torch.split(
            latent_states,
            (self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim),
            dim=-1,
        )
        if self.use_mome:
            q_latent = apply_mome(
                q_latent, mome_mask, self.qa_conv, fused=False
            )
            kv_nope = apply_mome(
                kv_nope, mome_mask, self.compresskv_conv, fused=False
            )
        q_resid = self.q_layernorm(q_latent)
        q_states, _ = self.linear_qb(q_resid.contiguous())
        q_states = q_states.view(
            batch_size, seq_length, self.num_heads, self.qk_head_dim
        )
        q_pass, q_rot = torch.split(
            q_states, (self.qk_nope_head_dim, self.qk_rope_head_dim), dim=-1
        )

        kv_weight = self.linear_kvb.weight.view(
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        key_up_weight = kv_weight[:, : self.qk_nope_head_dim]
        absorbed_query_states = q_pass.permute(2, 0, 1, 3).reshape(
            self.num_heads, batch_size * seq_length, self.qk_nope_head_dim
        )
        absorbed_query_states = torch.bmm(
            absorbed_query_states, key_up_weight
        ).view(
            self.num_heads, batch_size, seq_length, self.kv_lora_rank
        ).permute(1, 2, 0, 3)

        kv_nope = self.k_layernorm(kv_nope).view(
            batch_size, seq_length, 1, self.kv_lora_rank
        )
        k_rot = k_rot.view(batch_size, seq_length, 1, self.qk_rope_head_dim)

        index_query, _ = self.index_linear_qb(q_resid.detach())
        index_query = index_query.view(
            batch_size, seq_length, self.num_index_heads, self.index_head_dim
        )
        index_key, _ = self.index_linear_k(hidden_states.detach())
        index_key = self.index_k_layernorm(index_key.unsqueeze(2))
        merge_weight, _ = self.linear_merge_weight(hidden_states.detach())
        merge_weight = (
            merge_weight
            * self.num_index_heads**-0.5
            * self.index_head_dim**-0.5
        )

        if position_embeddings is not None:
            cos, sin = position_embeddings
            if self.rotary_interleaved:
                q_rot, k_rot = apply_rotary_pos_emb_interleave(
                    q_rot, k_rot, cos, sin, unsqueeze_dim=2
                )
            else:
                q_rot, k_rot = apply_rotary_pos_emb(
                    q_rot, k_rot, cos, sin, unsqueeze_dim=2
                )

            index_query_rot, index_query_pass = torch.split(
                index_query,
                (self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim),
                dim=-1,
            )
            index_key_rot, index_key_pass = torch.split(
                index_key,
                (self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim),
                dim=-1,
            )
            if self.rotary_interleaved:
                index_query_rot, index_key_rot = apply_rotary_pos_emb_interleave(
                    index_query_rot,
                    index_key_rot,
                    cos,
                    sin,
                    unsqueeze_dim=2,
                )
            else:
                index_query_rot, index_key_rot = apply_rotary_pos_emb(
                    index_query_rot,
                    index_key_rot,
                    cos,
                    sin,
                    unsqueeze_dim=2,
                )
            index_query = torch.cat((index_query_rot, index_query_pass), dim=-1)
            index_key = torch.cat((index_key_rot, index_key_pass), dim=-1)

        actual_seq_len = self._get_actual_seq_len(
            actual_seq_len, batch_size, seq_length, hidden_states.device
        )
        topk_indices, index_query_tnd, index_key_tnd, merge_weight_tnd = dsa_indexer(
            index_query,
            index_key,
            merge_weight,
            actual_seq_len,
            actual_seq_len,
            self.index_topk,
        )
        sparse_scale = self.qk_head_dim**-0.5
        if self.param_sink_number > 0:
            sink_key, sink_value = self._prepare_param_sink(batch_size)
            attn_output, softmax_max, softmax_sum = dsa_sparse_attention_rescale(
                absorbed_query_states,
                kv_nope,
                q_rot,
                k_rot,
                sink_key,
                sink_value,
                topk_indices,
                batch_size,
                seq_length,
                self.num_heads,
                sparse_scale,
                1 - self.attention_dropout.p,
                actual_seq_len,
                actual_seq_len,
            )
        else:
            attn_output, softmax_max, softmax_sum = dsa_sparse_attention(
                absorbed_query_states,
                kv_nope,
                q_rot,
                k_rot,
                topk_indices,
                sparse_scale,
                actual_seq_len,
                actual_seq_len,
            )
        if self.training and not self.freeze_dsa and self.dsa_loss_coeff:
            query_tnd, key_tnd, q_rot_tnd, k_rot_tnd = (
                tensor.reshape(-1, tensor.shape[2], tensor.shape[3])
                for tensor in (absorbed_query_states, kv_nope, q_rot, k_rot)
            )
            aux_loss = dsa_kl_loss(
                index_query_tnd,
                index_key_tnd,
                merge_weight_tnd,
                query_tnd,
                key_tnd,
                topk_indices,
                softmax_max,
                softmax_sum,
                q_rot_tnd,
                k_rot_tnd,
                actual_seq_len,
                actual_seq_len,
                sparse_scale,
                self.dsa_loss_coeff,
            )
            attn_output = _AuxLossAutoScaler.apply(attn_output, aux_loss)

        attn_output = attn_output[..., : self.kv_lora_rank]
        value_up_weight = kv_weight[:, self.qk_nope_head_dim :].transpose(1, 2)
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(
            self.num_heads, batch_size * seq_length, self.kv_lora_rank
        )
        attn_output = torch.bmm(attn_output, value_up_weight)
        attn_output = attn_output.view(
            self.num_heads, batch_size, seq_length, self.v_head_dim
        ).permute(1, 2, 0, 3).reshape(batch_size, seq_length, -1)
        if self.use_mome:
            attn_output = apply_mome(
                attn_output,
                mome_mask,
                self.o_conv,
                fused=self.use_fused_mome,
            )
        output, bias = self.linear_proj(attn_output)
        return (output, bias) if return_bias else (output, None)
