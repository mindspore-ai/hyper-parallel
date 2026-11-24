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
"""DeepSeek sparse-attention implementations and shared projection mathematics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.distributed._functional_collectives as funcol
from torch import nn

from hyper_parallel.components.checkpoint import ConcatenateWithSections
from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter
from hyper_parallel.components.losses.aux_loss import aux_loss_auto_scale
from hyper_parallel.components.functional.dsa_indexer import dsa_indexer
from hyper_parallel.components.functional.dsa_kl_loss import dsa_kl_loss
from hyper_parallel.components.functional.dsa_sparse_attention import dsa_sparse_attention
from hyper_parallel.components.functional.npu_fusion_attention import resolve_packed_sequence_lengths
from hyper_parallel.distributed.module_binding import ParallelBinding
from hyper_parallel.components.modules.attention.deepseek_sparse_attention.indexer_share import (
    IndexShareContext, IndexShareState, resolve_index_share_plan,
)
from hyper_parallel.components.modules.attention.deepseek_sparse_attention.mtp_index_share import (
    MTPIndexShareContext, MTPIndexShareState,
)


@dataclass(frozen=True)
class SparseAttentionSequenceMetadata:
    """Kernel endpoints and the consumed KV prefix for one invocation."""

    kv_prefix_length: int
    query_ends: torch.Tensor
    key_ends: torch.Tensor

    @property
    def sequence_lengths(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cumulative int32 Q/K endpoints for the existing CANN kernels."""
        return self.query_ends, self.key_ends

    @classmethod
    def build(
        cls, *, batch_size: int, local_seq_len: int, device: torch.device,
        actual_seq_len: torch.Tensor | Sequence[int] | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> SparseAttentionSequenceMetadata:
        """Resolve local cumulative endpoints through the shared packed parser.

        Args:
            batch_size: Number of batch-major rows.
            local_seq_len: Token count per row.
            device: Kernel metadata device.
            actual_seq_len: Cumulative packed endpoints without leading zero.
            kwargs: Supported packed aliases, including packed_seq_params.
        """
        total = batch_size * local_seq_len
        packed = dict(kwargs or {})
        if actual_seq_len is not None:
            packed["actual_seq_len"] = actual_seq_len
        queries, keys = resolve_packed_sequence_lengths(packed, total, total)
        if queries is None:
            queries = keys = [(row + 1) * local_seq_len for row in range(batch_size)]
        if len(queries) != len(keys):
            raise ValueError("Sparse attention requires the same number of packed Q and KV documents")
        return cls(
            local_seq_len,
            torch.tensor(queries, dtype=torch.int32, device=device),
            torch.tensor(keys, dtype=torch.int32, device=device),
        )


def _apply_attention_rope(
    query: torch.Tensor,
    key: torch.Tensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the configured RoPE convention to an attention query/key pair."""
    if position_embeddings is None:
        return query, key
    # The rotary backend imports torch_npu; defer it until positional math is requested.
    from hyper_parallel.components.functional import (  # pylint: disable=import-outside-toplevel
        apply_rotary_pos_emb, apply_rotary_pos_emb_interleave,
    )

    cos, sin = position_embeddings
    rotary_op = apply_rotary_pos_emb_interleave if interleaved else apply_rotary_pos_emb
    if key is None:
        # The backend rotates Q and K together. Reuse it for Q without allowing
        # uninitialized temporary values to enter the kernel.
        dummy_key = torch.zeros((*query.shape[:2], 1, query.shape[-1]), device=query.device, dtype=query.dtype)
        query, _ = rotary_op(query, dummy_key, cos, sin, unsqueeze_dim=2)
        return query, None
    return rotary_op(query, key, cos, sin, unsqueeze_dim=2)


def _apply_index_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    rope_head_dim: int,
    index_head_dim: int,
    interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to the rotary portion of an indexer query/key pair."""
    if position_embeddings is None:
        return query, key
    # The rotary backend is optional for construction and CPU metadata/projection tests.
    from hyper_parallel.components.functional import (  # pylint: disable=import-outside-toplevel
        apply_rotary_pos_emb, apply_rotary_pos_emb_interleave,
    )

    cos, sin = position_embeddings
    split_sizes = (rope_head_dim, index_head_dim - rope_head_dim)
    query_rot, query_pass = torch.split(query, split_sizes, dim=-1)
    key_rot, key_pass = torch.split(key, split_sizes, dim=-1)
    rotary_op = apply_rotary_pos_emb_interleave if interleaved else apply_rotary_pos_emb
    query_rot, key_rot = rotary_op(query_rot, key_rot, cos, sin, unsqueeze_dim=2)
    return torch.cat((query_rot, query_pass), dim=-1), torch.cat((key_rot, key_pass), dim=-1)


def _absorb_query_projection(
    query: torch.Tensor,
    key_up_weight: torch.Tensor,
    *,
    num_heads: int,
    batch_size: int,
    seq_length: int,
    input_dim: int,
    output_dim: int,
) -> torch.Tensor:
    """Absorb the key up-projection weight into query states."""
    query = query.permute(2, 0, 1, 3).reshape(num_heads, batch_size * seq_length, input_dim)
    query = torch.bmm(query, key_up_weight).view(num_heads, batch_size, seq_length, output_dim)
    return query.permute(1, 2, 0, 3)


def _restore_attention_projection(
    output: torch.Tensor,
    value_up_weight: torch.Tensor,
    *,
    num_heads: int,
    batch_size: int,
    seq_length: int,
    kv_lora_rank: int,
    value_head_dim: int,
) -> torch.Tensor:
    """Apply the absorbed value up-projection and restore the hidden layout."""
    output = output[..., :kv_lora_rank]
    output = output.permute(2, 0, 1, 3).reshape(num_heads, batch_size * seq_length, kv_lora_rank)
    output = torch.bmm(output, value_up_weight)
    return output.view(num_heads, batch_size, seq_length, value_head_dim).permute(1, 2, 0, 3).reshape(
        batch_size, seq_length, -1
    )


def prepare_indexer_teacher(
    query: torch.Tensor, key: torch.Tensor,
    query_rope: torch.Tensor, key_rope: torch.Tensor,
    maximum: torch.Tensor, total: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Detach the supplied teacher tensors without changing their local layout.

    Args:
        query: Main-attention TND query.
        key: Dense per-head key or sparse shared latent key in TND.
        query_rope: TND query positional band.
        key_rope: TND key positional band.
        maximum: Dense N,T,1 or sparse 1,T,N softmax maximum.
        total: Corresponding softmax exponential sum.

    Returns:
        Detached Q, K, Q-rope, K-rope, maximum and sum for the existing KL kernel.
    """
    return tuple(tensor.detach() for tensor in (query, key, query_rope, key_rope, maximum, total))


@dataclass(frozen=True)
class SparseAttentionInputs:
    """Invocation-local operands and the resolved dense/sparse training stage."""

    index_states: tuple[torch.Tensor, ...]
    attention_states: tuple[torch.Tensor, ...]
    metadata: SparseAttentionSequenceMetadata
    use_dense: bool
    mtp_topk_indices: torch.Tensor | None = None


def _dense_flash_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sequence_lengths: tuple[torch.Tensor, torch.Tensor],
    *,
    need_stats: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run one TND FA and return dense-indexer statistics."""
    query_ends, key_ends = (lengths.tolist() for lengths in sequence_lengths)
    q_flat, k_flat, v_flat = (tensor.flatten(0, 1) for tensor in (query, key, value))
    num_heads = q_flat.shape[1]
    causal_mask = torch.ones((2048, 2048), dtype=torch.bool, device=query.device).triu(diagonal=1)
    import torch_npu  # pylint: disable=import-outside-toplevel

    result = torch_npu.npu_fusion_attention(
        q_flat.contiguous(), k_flat.contiguous(), v_flat.contiguous(), num_heads, "TND",
        pse=None, padding_mask=None, atten_mask=causal_mask, scale=module.scaling,
        pre_tockens=getattr(module, "pre_tockens", 1048576),
        next_tockens=getattr(module, "next_tockens", 0), keep_prob=1.0, inner_precise=0,
        actual_seq_qlen=query_ends, actual_seq_kvlen=key_ends, sparse_mode=3,
    )
    output = result[0].reshape(*query.shape[:3], value.shape[-1])
    if not need_stats:
        return output, None, None

    # TND FA stores each packed segment as [head, token, 8], while exposing a
    # physical [token, head, 8] shape. Dense KL consumes [head, token, 1].
    statistics = []
    starts = [0, *query_ends[:-1]]
    for statistic in result[1:3]:
        rows = statistic.reshape(-1, statistic.shape[-1])
        segments = [
            rows[start * num_heads : end * num_heads].reshape(
                num_heads, end - start, statistic.shape[-1]
            )[..., :1]
            for start, end in zip(starts, query_ends)
        ]
        statistics.append(torch.cat(segments, dim=1).detach())
    return output, statistics[0], statistics[1]


def _deepseek_v32_dense_attention(
    module: nn.Module,
    index_states: tuple[torch.Tensor, ...],
    attention_states: tuple[torch.Tensor, ...],
    sequence_lengths: tuple[torch.Tensor, torch.Tensor],
    *, aux_loss_coefficient: float | None = None,
) -> torch.Tensor:
    """Run MLA FA and dense Indexer KL with the effective auxiliary-loss scale.

    ``aux_loss_coefficient`` overrides the configured ``dsa_loss_coeff`` for
    this layer, which Indexer Share uses to divide ``served`` supervision.
    """
    query, compressed_kv, query_rope, key_rope = attention_states
    projected_kv = module.kv_b_proj(compressed_kv.squeeze(2)).view(
        *compressed_kv.shape[:2], query.shape[2], module.qk_nope_head_dim + module.v_head_dim,
    )
    key, value = projected_kv.split((module.qk_nope_head_dim, module.v_head_dim), dim=-1)
    key_rope = key_rope.expand(-1, -1, query.shape[2], -1)
    coefficient = module.dsa_loss_coeff if aux_loss_coefficient is None else aux_loss_coefficient
    need_stats = bool(module.training and not module.freeze_dsa and coefficient)
    output, maximum, total = _dense_flash_attention(
        module, torch.cat((query, query_rope), dim=-1), torch.cat((key, key_rope), dim=-1),
        value, sequence_lengths, need_stats=need_stats,
    )
    if need_stats:
        # Dense CANN operators are optional for sparse and Omni execution.
        # pylint: disable=import-outside-toplevel
        from hyper_parallel.components.functional.dsa_dense import (
            dsa_dense_indexer_kl_loss,
            dsa_dense_indexer_lse,
        )
        # pylint: enable=import-outside-toplevel

        query_lengths, key_lengths = sequence_lengths
        index_query, index_key, weights = (tensor.flatten(0, 1) for tensor in index_states)
        indexer_stats = dsa_dense_indexer_lse(
            index_query, index_key, weights, query_lengths=query_lengths, key_lengths=key_lengths,
        )
        teacher_query, teacher_key, teacher_qrope, teacher_krope, maximum, total = module.prepare_teacher(
            query.flatten(0, 1), key.flatten(0, 1),
            query_rope.flatten(0, 1), key_rope.flatten(0, 1), maximum, total,
            dense=True,
        )
        loss = dsa_dense_indexer_kl_loss(
            index_query, index_key, weights, teacher_query, teacher_key,
            maximum, total, indexer_stats, scale=module.scaling,
            query_rope=teacher_qrope, key_rope=teacher_krope,
            query_lengths=query_lengths, key_lengths=key_lengths, loss_coefficient=coefficient,
        )
        output = aux_loss_auto_scale(output, loss)
    return output


class DeepseekSparseAttentionModule(nn.Module):
    """DeepSeek-V3.2 sparse attention with native configuration and source-module construction.

    It supports causal training and packed sequences. Packed boundaries must
    be supplied through ``actual_seq_len``. ``attention_mask`` is a compatibility
    argument whose values are not consumed; causal masking comes from DSA kernels
    and sequence metadata. Padding/custom masks, dropout, and KV cache are unsupported.
    """

    def bind_parallel(self, binding: ParallelBinding) -> None:
        """Borrow the model's existing TP group for teacher assembly."""
        mesh = binding.mesh_binding.mesh_context.device_mesh
        names = getattr(mesh, "mesh_dim_names", None)
        if self._tp_size > 1 or (isinstance(names, (list, tuple)) and "tp" in names):
            tp = binding.mesh_binding.axis_mesh("tp")
            bound_size = tp.size()
            if self._tp_size not in (1, bound_size):
                raise ValueError(
                    f"Sparse-attention TP size {self._tp_size} does not match bound mesh size {bound_size}"
                )
            self._tp_size = bound_size
            self._tp_group = tp.get_group() if bound_size > 1 else None

    def _gather_tp(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """Concatenate one head shard across the borrowed TP group."""
        if self._tp_size == 1:
            return tensor
        if self._tp_group is None:
            raise RuntimeError("parallel DeepSeek sparse attention must be bound before forward")
        gathered = funcol.all_gather_tensor(tensor.contiguous(), dim, self._tp_group)
        return funcol.wait_tensor(gathered)

    def gather_teacher_heads(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_rope: torch.Tensor,
        key_rope: torch.Tensor,
        maximum: torch.Tensor,
        total: torch.Tensor,
        *,
        dense: bool,
    ) -> tuple[torch.Tensor, ...]:
        """Assemble the global-head detached teacher expected by the fused KL op."""
        query = self._gather_tp(query, 1)
        query_rope = self._gather_tp(query_rope, 1)
        if dense:
            key = self._gather_tp(key, 1)
            key_rope = self._gather_tp(key_rope, 1)
        stats_dim = 0 if dense else 2
        maximum = self._gather_tp(maximum, stats_dim)
        total = self._gather_tp(total, stats_dim)
        return query, key, query_rope, key_rope, maximum, total

    def prepare_teacher(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_rope: torch.Tensor,
        key_rope: torch.Tensor,
        maximum: torch.Tensor,
        total: torch.Tensor,
        *,
        dense: bool,
    ) -> tuple[torch.Tensor, ...]:
        """Prepare full-head detached operands for the indexer KL."""
        teacher = prepare_indexer_teacher(query, key, query_rope, key_rope, maximum, total)
        return self.gather_teacher_heads(*teacher, dense=dense)

    def _align_tokens(self, tensor: torch.Tensor, dim: int, tokens: int) -> torch.Tensor:
        """Gather sequence-sharded metadata when attention inputs are TP-gathered."""
        local = tensor.shape[dim]
        if local == tokens:
            return tensor
        if local * self._tp_size != tokens:
            raise ValueError("Position length must match local or TP-gathered tokens")
        return self._gather_tp(tensor, dim % tensor.ndim)

    def prepare_positions(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Align position tables with the tokens consumed by attention."""
        if position_embeddings is None:
            return None
        cosine, sine = position_embeddings
        return tuple(self._align_tokens(table, -2, hidden_states.shape[1]) for table in (cosine, sine))

    def prepare_position_ids(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Align optional position IDs with TP-gathered attention tokens."""
        if position_ids is None:
            return None
        if position_ids.ndim != 2:
            raise ValueError("position_ids must have BS layout")
        return self._align_tokens(position_ids, -1, hidden_states.shape[1])

    def _initialize_dimensions(self, module: nn.Module | None) -> None:
        """Read main-attention and indexer dimensions from a source or config."""
        config = self.config
        indexer = getattr(module, "indexer", None)
        self.layer_idx = getattr(
            module, "layer_idx", getattr(module, "layer_number", getattr(config, "layer_idx", None)),
        )
        self.num_heads = getattr(module, "num_heads", config.num_attention_heads)
        self.q_lora_rank = getattr(module, "q_lora_rank", config.q_lora_rank)
        self.kv_lora_rank = getattr(module, "kv_lora_rank", config.kv_lora_rank)
        self.qk_rope_head_dim = getattr(module, "qk_rope_head_dim", config.qk_rope_head_dim)
        self.qk_nope_head_dim = getattr(module, "qk_nope_head_dim", config.qk_nope_head_dim)
        self.v_head_dim = getattr(module, "v_head_dim", config.v_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.q_lora_rank is None:
            raise ValueError("DeepseekSparseAttention requires the Q-LoRA projection path")
        self.index_head_dim = getattr(indexer, "head_dim", config.index_head_dim)
        self.num_index_heads = getattr(
            indexer,
            "n_heads",
            getattr(config, "index_n_heads", getattr(config, "index_num_attention_heads", None)),
        )
        if self.num_index_heads is None:
            raise ValueError("DeepseekSparseAttention requires the number of indexer heads")
        self.index_topk = getattr(indexer, "index_topk", config.index_topk)

    def _initialize_options(self, module: nn.Module | None) -> None:
        """Read sparse attention options while preserving source scaling and RoPE conventions."""
        config = self.config
        self.scaling = getattr(module, "scaling", None)
        if self.scaling is None:
            self.scaling = self._config_scaling(config)
        attention_dropout = getattr(module, "attention_dropout", getattr(config, "attention_dropout", 0.0))
        if isinstance(attention_dropout, nn.Dropout):
            attention_dropout = attention_dropout.p
        if attention_dropout != 0.0:
            raise ValueError(
                "DeepseekSparseAttention requires attention_dropout=0 because "
                "the sparse-attention operator does not apply dropout"
            )
        source_rotary_interleaved = getattr(module, "rotary_interleaved", getattr(config, "rotary_interleaved", None))
        if source_rotary_interleaved is None:
            source_rotary_interleaved = getattr(config, "rope_interleave", None)
        if source_rotary_interleaved is None:
            source_rotary_interleaved = getattr(config, "rope_interleaved", None)
        self.rotary_interleaved = bool(True if source_rotary_interleaved is None else source_rotary_interleaved)
        self.index_rotary_interleaved = bool(
            getattr(
                getattr(module, "indexer", None), "rotary_interleaved",
                getattr(config, "index_rope_interleaved", False),
            )
        )
        self.dsa_loss_coeff = getattr(config, "dsa_loss_coeff", 0.0)
        self.freeze_dsa = bool(getattr(config, "freeze_DSA", getattr(config, "freeze_dsa", False)))
        self.dsa_dense_warm_up = bool(
            getattr(module, "dsa_dense_warm_up", getattr(config, "dsa_dense_warm_up", False))
        )
        self.index_share_for_mtp_iteration = getattr(config, "index_share_for_mtp_iteration", False)
        if not isinstance(self.index_share_for_mtp_iteration, bool):
            raise ValueError("index_share_for_mtp_iteration must be a boolean")
        layer_count = getattr(config, "num_hidden_layers", 0)
        self.is_mtp_iteration = type(self.layer_idx) is int and self.layer_idx >= layer_count
        self.mtp_index_share_enabled = self.index_share_for_mtp_iteration and self.is_mtp_iteration
        self.mtp_index_share_key = ""
        if getattr(module, "param_sink_number", getattr(config, "param_sink_number", 0)) > 0:
            raise ValueError("DeepseekSparseAttention does not support the parameter-sink path")
        if getattr(module, "use_mome", getattr(config, "use_mome", False)):
            raise ValueError("DeepseekSparseAttention does not support the MOME path")

    def _config_scaling(self, config: Any) -> float:
        """Apply the source-compatible YaRN factor for native construction."""
        scaling = self.qk_head_dim**-0.5
        rope = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default" and rope.get("mscale_all_dim", 0):
            factor = rope["factor"]
            mscale = 1.0 if factor <= 1 else 1.0 + 0.1 * rope["mscale_all_dim"] * math.log(factor)
            scaling *= mscale * mscale
        return getattr(config, "scaling", scaling)

    def _initialize_config_layers(self, config: Any) -> None:
        """Allocate the native parameter tree without constructing a Transformers model."""
        dtype = getattr(config, "dtype", None)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.removeprefix("torch."))
        factory_kwargs = {"device": getattr(config, "device", None), "dtype": dtype}
        bias = getattr(config, "attention_bias", False)
        self._q_latent_output_size = self.q_lora_rank
        self._kv_latent_output_size = self.kv_lora_rank + self.qk_rope_head_dim
        self.linear_qkv = nn.Linear(
            config.hidden_size, self._q_latent_output_size + self._kv_latent_output_size,
            bias=bias, **factory_kwargs,
        )
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=eps, **factory_kwargs)
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=eps, **factory_kwargs)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False, **factory_kwargs,
        )
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False, **factory_kwargs,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, config.hidden_size, bias=bias, **factory_kwargs,
        )
        self.indexer = None
        if not self.is_index_leader:
            return
        self.indexer = nn.Module()
        self.indexer.n_heads, self.indexer.head_dim = self.num_index_heads, self.index_head_dim
        self.indexer.index_topk = self.index_topk
        self.indexer.wq_b = nn.Linear(
            self.q_lora_rank, self.num_index_heads * self.index_head_dim, bias=False, **factory_kwargs,
        )
        self.indexer.wk = nn.Linear(config.hidden_size, self.index_head_dim, bias=False, **factory_kwargs)
        self.indexer.k_norm = nn.LayerNorm(self.index_head_dim, eps=1e-6, **factory_kwargs)
        self.indexer.weights_proj = nn.Linear(config.hidden_size, self.num_index_heads, bias=False, **factory_kwargs)

    def _initialize_source_layers(self, module: nn.Module) -> None:
        """Fuse loaded down-projection values and retain the remaining Parameters."""
        q_a, kv_a = module.q_a_proj, module.kv_a_proj_with_mqa
        self._q_latent_output_size, self._kv_latent_output_size = q_a.out_features, kv_a.out_features
        can_fuse = (
            q_a.in_features == kv_a.in_features
            and q_a.weight.requires_grad == kv_a.weight.requires_grad
            and q_a.weight.device == kv_a.weight.device
            and q_a.weight.dtype == kv_a.weight.dtype
            and (q_a.bias is None) == (kv_a.bias is None)
            and (q_a.bias is None or q_a.bias.requires_grad == kv_a.bias.requires_grad)
        )
        if not can_fuse:
            raise ValueError("Q and KV latent projections cannot be represented by one fused projection")
        self.linear_qkv = nn.Linear(
            q_a.in_features, q_a.out_features + kv_a.out_features, bias=q_a.bias is not None,
            device=q_a.weight.device, dtype=q_a.weight.dtype,
        )
        self.linear_qkv.weight.requires_grad_(q_a.weight.requires_grad)
        if self.linear_qkv.bias is not None:
            self.linear_qkv.bias.requires_grad_(q_a.bias.requires_grad)
        if not q_a.weight.is_meta:
            with torch.no_grad():
                self.linear_qkv.weight.copy_(torch.cat((q_a.weight, kv_a.weight), dim=0))
                if self.linear_qkv.bias is not None:
                    self.linear_qkv.bias.copy_(torch.cat((q_a.bias, kv_a.bias), dim=0))
        self.linear_qkv._is_hf_initialized = True  # pylint: disable=protected-access
        for name in ("q_a_layernorm", "kv_a_layernorm", "q_b_proj"):
            setattr(self, name, getattr(module, name))
        if module.kv_b_proj.bias is not None:
            raise ValueError("Absorbed KV projection requires a bias-free Linear")
        self.kv_b_proj = module.kv_b_proj
        self.o_proj = module.o_proj
        self.indexer = module.indexer if self.is_index_leader else None
        self._owns_indexer = self.is_index_leader

    def _reuse_local_member(self, module: DeepseekSparseAttentionModule) -> None:
        """Promote a loaded unbound member without re-fusing or reinitializing values."""
        if getattr(module, "_hp_parallel_binding", None) is not None:
            raise ValueError("DeepSeek-V3.2 promotion must precede parallel binding and FSDP")
        fields = (
            "config", "layer_idx", "num_heads", "q_lora_rank", "kv_lora_rank",
            "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim", "qk_head_dim", "index_head_dim",
            "num_index_heads", "index_topk", "scaling", "rotary_interleaved", "index_rotary_interleaved",
            "dsa_loss_coeff", "freeze_dsa", "dsa_dense_warm_up",
            "index_share_plan", "is_index_leader",
            "is_mtp_iteration", "index_share_enabled", "_owns_indexer",
            "index_share_for_mtp_iteration", "mtp_index_share_enabled", "mtp_index_share_key",
            "_q_latent_output_size", "_kv_latent_output_size", "_tp_size", "_tp_group",
        )
        for name in fields:
            setattr(self, name, getattr(module, name))
        for name in ("linear_qkv", "q_a_layernorm", "kv_a_layernorm", "q_b_proj", "kv_b_proj", "o_proj"):
            setattr(self, name, getattr(module, name))
        self._modules.pop("indexer", None)
        if module._owns_indexer:
            self.indexer = module.indexer
        else:
            object.__setattr__(self, "indexer", module.indexer)

    def __init__(
        self, *, config: Any = None, module: nn.Module | None = None,
        module_fqn: str = "", context: Mapping[str, Any] | None = None,
        kv_cp_size: int = 1, kv_cp_strategy: str = "allgather",
        cp_comm_overlap: bool = False, dense_warmup: bool | None = None,
    ) -> None:
        """Construct native sparse attention parameters, convert a source, or promote a local member.

        Args:
            config: Native dimensions/options, exclusive with module; optional dtype/device
                select parameter placement. Otherwise the active Torch defaults apply.
            module: Source attention or an unbound concrete family member.
            module_fqn: Source path supplied by replacement.
            context: Logical construction context, without execution resources.
            kv_cp_size: KV domain size retained for deferred family promotion.
            kv_cp_strategy: KV transport, currently allgather.
            cp_comm_overlap: Submit key-bank gathers asynchronously when enabled.
            dense_warmup: None follows the mutable training stage; bool pins it.
        """
        super().__init__()
        del module_fqn
        if (config is None) == (module is None):
            raise ValueError("DeepSeek-V3.2 sparse attention requires exactly one of config or module")
        if not isinstance(kv_cp_size, int) or isinstance(kv_cp_size, bool) or kv_cp_size < 1:
            raise ValueError("kv_cp_size must be a positive integer")
        if kv_cp_strategy != "allgather":
            raise ValueError("DeepSeek sparse attention supports only kv_cp_strategy='allgather'")
        self._config_initialization = module is None or getattr(module, "_config_initialization", False)
        self._tp_size = (context or {}).get("tp_size", 1)
        self._tp_group = None
        self.kv_cp_size, self.kv_cp_strategy = kv_cp_size, kv_cp_strategy
        self.cp_comm_overlap = cp_comm_overlap
        self._dense_warmup_override = dense_warmup
        if isinstance(module, DeepseekSparseAttentionModule):
            self._reuse_local_member(module)
        else:
            self.config = config if module is None else module.config
            if hasattr(self.config, "indexer_types"):
                raise TypeError(
                    "DeepSeek-V3.2 sparse attention does not support shared-indexer or three-value output contracts"
                )
            self._initialize_dimensions(module)
            self._initialize_options(module)
            self.index_share_plan = resolve_index_share_plan(self.config)
            if self.index_share_plan.enabled and getattr(self.config, "model_type", None) != "deepseek_v32":
                raise ValueError("Indexer sharing is supported only for DeepSeek-V3.2")
            layer_count = getattr(self.config, "num_hidden_layers", 0)
            self.is_mtp_iteration = type(self.layer_idx) is int and self.layer_idx >= layer_count
            if self.index_share_plan.enabled and not self.is_mtp_iteration and (
                type(self.layer_idx) is not int or not 0 <= self.layer_idx < len(self.index_share_plan.leaders)
            ):
                raise ValueError("Indexer sharing requires a zero-based global layer_idx within num_hidden_layers")
            self.index_share_enabled = self.index_share_plan.enabled and not self.is_mtp_iteration
            self.is_index_leader = (
                not self.index_share_enabled or self.index_share_plan.leaders[self.layer_idx] == self.layer_idx
            )
            self._owns_indexer = self.is_index_leader
            if module is None:
                self._initialize_config_layers(config)
            else:
                self._initialize_source_layers(module)
        if module is not None:
            self.train(module.training)

    def reset_parameters(self) -> None:
        """Initialize native/meta parameters without resetting converted source weights."""
        if not self._config_initialization:
            return
        for child in (
            self.linear_qkv, self.q_a_layernorm, self.kv_a_layernorm, self.q_b_proj,
            self.kv_b_proj, self.o_proj,
        ):
            child.reset_parameters()
        if self.indexer is not None and self._owns_indexer:
            for child in self.indexer.children():
                child.reset_parameters()

    def make_transforms(self) -> list[WeightConverter]:
        """Describe reversible source-checkpoint to high-performance conversion."""
        if self._config_initialization:
            return []
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
    def _invalid_position_ids(
        position_ids: torch.Tensor | None,
        batch_size: int,
        seq_length: int,
    ) -> bool:
        """Return whether position IDs violate the supported dense layout."""
        if position_ids is None:
            return False
        if position_ids.ndim != 2:
            return True
        return position_ids.shape[0] not in (1, batch_size) or position_ids.shape[1] != seq_length

    def _validate_forward_inputs(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Any | None,
        position_ids: torch.Tensor | None,
        kwargs: dict[str, Any],
    ) -> None:
        """Validate inputs supported by the DeepSeek sparse-attention path."""
        del attention_mask
        if hidden_states.ndim != 3:
            raise ValueError("DeepSeek-V3.2 hidden_states must have shape [batch, sequence, hidden]")
        batch_size, seq_length = hidden_states.shape[:-1]
        if past_key_values is not None:
            raise NotImplementedError("DeepseekSparseAttention does not support KV cache")
        if DeepseekSparseAttentionModule._invalid_position_ids(position_ids, batch_size, seq_length):
            raise ValueError("position_ids must have shape [1 or batch_size, sequence_length]")
        if kwargs.pop("output_attentions", False):
            raise NotImplementedError("DeepseekSparseAttention does not expose attention weights")

    def _project_attention_states(
        self, hidden_states: torch.Tensor, *, use_dense: bool = False, skip_kv: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, ...]]:
        """Project local states inside the attention/block FSDP boundary."""
        batch_size, seq_length = hidden_states.shape[:-1]
        if skip_kv:
            bias = self.linear_qkv.bias
            q_latent = torch.nn.functional.linear(  # pylint: disable=not-callable
                hidden_states, self.linear_qkv.weight[:self.q_lora_rank],
                None if bias is None else bias[:self.q_lora_rank],
            )
            latent_kv = key_rope = None
        else:
            q_latent, latent_kv, key_rope = torch.split(
                self.linear_qkv(hidden_states),
                (self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim), dim=-1,
            )
        q_resid = self.q_a_layernorm(q_latent)
        projected = self.q_b_proj(q_resid)
        heads = projected.shape[-1] // self.qk_head_dim
        query, query_rope = projected.view(
            batch_size, seq_length, heads, self.qk_head_dim,
        ).split((self.qk_nope_head_dim, self.qk_rope_head_dim), dim=-1)
        if not use_dense:
            weight = self.kv_b_proj.weight.view(
                heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank,
            )
            query = _absorb_query_projection(
                query, weight[:, :self.qk_nope_head_dim], num_heads=heads,
                batch_size=batch_size, seq_length=seq_length,
                input_dim=self.qk_nope_head_dim, output_dim=self.kv_lora_rank,
            )
        if latent_kv is not None:
            latent_kv = self.kv_a_layernorm(latent_kv).view(batch_size, seq_length, 1, self.kv_lora_rank)
            key_rope = key_rope.view(batch_size, seq_length, 1, self.qk_rope_head_dim)
        return q_resid, (query, latent_kv, query_rope, key_rope)

    def _project_index_states(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build indexer query, key, and merge-weight states."""
        batch_size, seq_length = hidden_states.shape[:-1]
        index_query = self.indexer.wq_b(q_resid.detach()).view(
            batch_size, seq_length, self.num_index_heads, self.index_head_dim
        )
        index_key = self.indexer.k_norm(self.indexer.wk(hidden_states.detach())).unsqueeze(2)
        merge_weight = self.indexer.weights_proj(hidden_states.detach())
        # Avoid intermediate low-precision rounding at the TopK boundary.
        weight_scale = (self.num_index_heads * self.index_head_dim)**-0.5
        merge_weight = (merge_weight.float() * weight_scale).to(merge_weight.dtype)
        index_query, index_key = _apply_index_rope(
            index_query,
            index_key,
            position_embeddings,
            rope_head_dim=self.qk_rope_head_dim,
            index_head_dim=self.index_head_dim,
            interleaved=self.index_rotary_interleaved,
        )
        return index_query, index_key, merge_weight

    def _apply_auxiliary_loss(
        self,
        attn_output: torch.Tensor,
        index_states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attention_states: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        topk_indices: torch.Tensor,
        softmax_stats: tuple[torch.Tensor, torch.Tensor],
        actual_q_len: torch.Tensor,
        actual_kv_len: torch.Tensor,
        *, aux_loss_coefficient: float | None = None,
    ) -> torch.Tensor:
        """Attach Indexer KL using the effective per-layer auxiliary-loss scale."""
        coefficient = self.dsa_loss_coeff if aux_loss_coefficient is None else aux_loss_coefficient
        if not (self.training and not self.freeze_dsa and coefficient):
            return attn_output
        query_tnd, key_tnd, q_rot_tnd, k_rot_tnd = (
            tensor.reshape(-1, tensor.shape[2], tensor.shape[3])
            for tensor in attention_states
        )
        query_tnd, key_tnd, q_rot_tnd, k_rot_tnd, maximum, total = self.prepare_teacher(
            query_tnd, key_tnd, q_rot_tnd, k_rot_tnd, *softmax_stats,
            dense=False,
        )
        aux_loss = dsa_kl_loss(
            *index_states, query_tnd, key_tnd,
            topk_indices, maximum, total, q_rot_tnd, k_rot_tnd,
            actual_q_len, actual_kv_len, self.scaling, coefficient,
            backend="cann",
        )
        return aux_loss_auto_scale(attn_output, aux_loss)

    def use_dense_attention(self) -> bool:
        """Read the current training stage unless an explicit override pins it."""
        return self.dsa_dense_warm_up if self._dense_warmup_override is None else self._dense_warmup_override

    def preflight_attention(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None,
        past_key_values: Any | None, position_ids: torch.Tensor | None,
        actual_seq_len: torch.Tensor | Sequence[int] | None,
        kwargs: dict[str, Any], *, use_dense: bool,
    ) -> SparseAttentionSequenceMetadata:
        """Validate local inputs and construct packed geometry before projection."""
        del use_dense
        self._validate_forward_inputs(hidden_states, attention_mask, past_key_values, position_ids, kwargs)
        return SparseAttentionSequenceMetadata.build(
            batch_size=hidden_states.shape[0], local_seq_len=hidden_states.shape[1],
            device=hidden_states.device, actual_seq_len=actual_seq_len, kwargs=kwargs,
        )

    def prepare_attention(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        *, use_dense: bool, reuse_mtp_kv: bool = False,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor | None, ...]]:
        """Run projections and both independent RoPE conventions on local tokens."""
        q_resid, states = self._project_attention_states(
            hidden_states, use_dense=use_dense, skip_kv=reuse_mtp_kv,
        )
        query, latent_kv, query_rope, key_rope = states
        query_rope, key_rope = _apply_attention_rope(
            query_rope, key_rope, position_embeddings, interleaved=self.rotary_interleaved,
        )
        index_states = () if reuse_mtp_kv else (
            self._project_index_states(hidden_states, q_resid, position_embeddings) if self.is_index_leader else ()
        )
        return index_states, (query, latent_kv, query_rope, key_rope)

    def prepare_attention_inputs(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        metadata: SparseAttentionSequenceMetadata, *, use_dense: bool, reuse_mtp_kv: bool = False,
    ) -> SparseAttentionInputs:
        """Prepare local operands while retaining one invocation's resolved mode."""
        index_states, attention_states = self.prepare_attention(
            hidden_states, position_embeddings, use_dense=use_dense, reuse_mtp_kv=reuse_mtp_kv,
        )
        return SparseAttentionInputs(index_states, attention_states, metadata, use_dense)

    def exchange_attention(self, inputs: SparseAttentionInputs) -> SparseAttentionInputs:
        """Return ready operands; local attention needs no sequence exchange."""
        return inputs

    def run_attention(
        self, index_states: tuple[torch.Tensor, ...], attention_states: tuple[torch.Tensor, ...],
        sequence_lengths: tuple[torch.Tensor, torch.Tensor], *, use_dense: bool,
        index_share_context: IndexShareContext | None = None,
        mtp_index_share_context: MTPIndexShareContext | None = None,
        mtp_iteration: int | None = None,
        mtp_query_positions: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
        kv_prefix_length: int | None = None,
    ) -> torch.Tensor:
        """Run selection once per group and supervise with each served layer's teacher."""
        aux_loss_coefficient = self.dsa_loss_coeff
        plan = self.index_share_plan
        if self.index_share_enabled:
            if not isinstance(index_share_context, IndexShareContext):
                raise ValueError("Indexer sharing requires a fresh IndexShareContext for each model forward")
            leader = plan.leaders[self.layer_idx]
            if plan.loss_mode == "served":
                aux_loss_coefficient /= plan.group_size(self.layer_idx)
            elif not self.is_index_leader:
                aux_loss_coefficient = 0.0
            if not self.is_index_leader:
                state = index_share_context.groups.get(leader)
                if state is None:
                    raise ValueError(f"Shared layer {self.layer_idx} requires leader {leader} to run first")
                if (state.use_dense != use_dense or state.index_topk != self.index_topk
                        or state.index_states[0].shape[:2] != attention_states[0].shape[:2]
                        or state.index_states[0].device != attention_states[0].device):
                    raise ValueError("Shared Indexer requires matching stage, Top-K, batch/sequence and device")
                index_states, topk = state.index_states, state.topk
        if mtp_topk_indices is not None:
            topk = mtp_topk_indices
            aux_loss_coefficient = 0.0
        elif self.is_index_leader:
            if use_dense:
                topk = None
            else:
                topk, *_ = dsa_indexer(
                    *index_states, *sequence_lengths, self.index_topk, backend="cann",
                )
            if self.index_share_enabled:
                index_share_context.groups[self.layer_idx] = IndexShareState(
                    index_states, None if topk is None else topk.detach(), use_dense, self.index_topk,
                )
            if (self.mtp_index_share_enabled and mtp_index_share_context is not None
                    and mtp_iteration == 0 and not use_dense):
                if not isinstance(topk, torch.Tensor):
                    raise RuntimeError("MTP IndexShare producer did not compute Top-K indices")
                query_ends, key_ends = sequence_lengths
                positions = None
                if mtp_query_positions is not None:
                    if mtp_query_positions.shape[0] == 1 and attention_states[0].shape[0] > 1:
                        mtp_query_positions = mtp_query_positions.expand(attention_states[0].shape[0], -1)
                    positions = tuple(int(value) for value in mtp_query_positions.detach().reshape(-1).cpu().tolist())
                mtp_index_share_context.publish(
                    self.mtp_index_share_key,
                    MTPIndexShareState(
                        topk.detach(), attention_states[1], attention_states[3], query_ends, key_ends,
                        positions, kv_prefix_length if kv_prefix_length is not None else int(key_ends[-1]),
                        self.index_topk,
                    ),
                )
        if use_dense:
            return _deepseek_v32_dense_attention(
                self, index_states, attention_states, sequence_lengths,
                aux_loss_coefficient=aux_loss_coefficient,
            ).flatten(2)
        flattened = tuple(tensor.flatten(0, 1) for tensor in index_states)
        output, maximum, total = dsa_sparse_attention(
            *attention_states, topk, self.scaling, *sequence_lengths, backend="cann",
        )
        output = self._apply_auxiliary_loss(
            output, tuple(flattened), attention_states, topk, (maximum, total), *sequence_lengths,
            aux_loss_coefficient=aux_loss_coefficient,
        )
        heads = output.shape[2]
        weight = self.kv_b_proj.weight.view(
            heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank,
        )
        return _restore_attention_projection(
            output, weight[:, self.qk_nope_head_dim:].transpose(1, 2), num_heads=heads,
            batch_size=output.shape[0], seq_length=output.shape[1],
            kv_lora_rank=self.kv_lora_rank, value_head_dim=self.v_head_dim,
        )

    def project_output(self, output: torch.Tensor) -> torch.Tensor:
        """Run the output Linear before the shared TP boundary."""
        return self.o_proj(output)

    def forward(
        self, hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None, past_key_values: Any | None = None,
        position_ids: torch.Tensor | None = None,
        actual_seq_len: torch.Tensor | Sequence[int] | None = None,
        index_share_context: IndexShareContext | None = None,
        mtp_index_share_context: MTPIndexShareContext | None = None,
        mtp_iteration: int | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Run causal or packed sparse attention while retaining the Transformers output ABI."""
        use_dense = self.use_dense_attention()
        position_ids = self.prepare_position_ids(hidden_states, position_ids)
        if self.index_share_enabled and not isinstance(index_share_context, IndexShareContext):
            raise ValueError("Indexer sharing requires a fresh IndexShareContext for each model forward")
        if self.mtp_index_share_enabled:
            if not isinstance(mtp_index_share_context, MTPIndexShareContext):
                raise ValueError("MTP IndexShare requires a fresh MTPIndexShareContext for each MTP forward")
            if isinstance(mtp_iteration, bool) or not isinstance(mtp_iteration, int) or mtp_iteration < 0:
                raise ValueError("MTP IndexShare requires a nonnegative mtp_iteration")
            if not self.mtp_index_share_key:
                raise ValueError("MTP IndexShare attention must be registered by MultiTokenPrediction")
        if (self.index_share_enabled and self.training and self.dsa_loss_coeff and not self.freeze_dsa
                and not torch.is_grad_enabled()):
            raise ValueError("Indexer sharing training requires grad-enabled forward and non-reentrant checkpointing")
        metadata = self.preflight_attention(
            hidden_states, attention_mask, past_key_values, position_ids, actual_seq_len, kwargs,
            use_dense=use_dense,
        )
        if (self.mtp_index_share_enabled and getattr(self, "_cp_context", None) is not None
                and getattr(metadata, "is_packed", False)):
            raise NotImplementedError(
                "MTP IndexShare with CP currently requires non-packed, equal-length batch rows; "
                "packed/TND alignment is supported only by the local attention path"
            )
        mtp_state = None
        mtp_topk = None
        mtp_view = None
        if self.mtp_index_share_enabled:
            if getattr(self, "_cp_context", None) is not None and position_ids is None:
                raise ValueError("MTP IndexShare under CP requires position_ids for every step")
            if use_dense:
                mtp_index_share_context.mark_dense(self.mtp_index_share_key)
            elif mtp_iteration > 0:
                mtp_state = mtp_index_share_context.consumer_state(self.mtp_index_share_key, mtp_iteration)
            if mtp_state is not None:
                mtp_view = mtp_state.view_for_query(
                    metadata.query_ends, query_positions=position_ids,
                    query_batch_size=hidden_states.shape[0], query_seq_length=hidden_states.shape[1],
                    index_topk=self.index_topk,
                )
                mtp_topk = mtp_view.topk_indices
        inputs = self.prepare_attention_inputs(
            hidden_states, self.prepare_positions(hidden_states, position_embeddings), metadata, use_dense=use_dense,
            reuse_mtp_kv=mtp_state is not None,
        )
        if mtp_state is not None:
            query, _, query_rope, _ = inputs.attention_states
            shared_metadata = SparseAttentionSequenceMetadata(
                mtp_view.kv_prefix_length, metadata.query_ends, mtp_view.key_ends,
            )
            inputs = SparseAttentionInputs(
                (), (query, mtp_view.latent_kv, query_rope, mtp_view.key_rope), shared_metadata,
                use_dense, mtp_view.topk_indices,
            )
        inputs = self.exchange_attention(inputs)
        output = self.run_attention(
            inputs.index_states, inputs.attention_states, inputs.metadata.sequence_lengths, use_dense=inputs.use_dense,
            index_share_context=index_share_context,
            mtp_index_share_context=mtp_index_share_context, mtp_iteration=mtp_iteration,
            mtp_query_positions=position_ids, mtp_topk_indices=inputs.mtp_topk_indices,
            kv_prefix_length=inputs.metadata.kv_prefix_length,
        )
        del inputs
        return self.project_output(output), None
