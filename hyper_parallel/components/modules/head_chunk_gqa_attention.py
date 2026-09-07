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
"""Head-staged GQA using the current packed/interleaved QKV layout."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

# This module is an explicitly PyTorch high-performance component.
# pylint: disable=forbidden-backend-import,not-callable
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable

from hyper_parallel.components.functional import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
    npu_fusion_attention_forward,
    rms_norm,
)
from hyper_parallel.components.modules.gqa_attention import GQAAttention, rotate_half
from hyper_parallel.distributed.context_parallel.collectives import (
    ulysses_head_to_seq,
    ulysses_seq_to_head,
)
from hyper_parallel.models.replacement import module_replacement


_NPU_CAUSAL_MASKS: dict[str, torch.Tensor] = {}
_FA_SAVED_TENSOR_COUNT = 5
_PACKED_SEQUENCE_ARGUMENTS = (
    "actual_q_len",
    "actual_kv_len",
    "actual_seq_qlen",
    "actual_seq_kvlen",
    "cu_seq_lens",
    "cu_seq_lens_q",
    "cu_seq_lens_k",
    "cu_seq_lens_kv",
    "cu_seqlens_q",
    "cu_seqlens_k",
    "cu_seqlens_kv",
    "packed_seq_params",
)
_UNSUPPORTED_ATTENTION_ARGUMENTS = (
    "indices",
    "pre_tokens",
    "next_tokens",
    "pre_tockens",
    "next_tockens",
    "sparse_mode",
    "sliding_window",
)


def _compressed_causal_mask(reference: torch.Tensor) -> torch.Tensor:
    """Return the reused compressed causal mask required by NPU FA v3."""
    key = str(reference.device)
    mask = _NPU_CAUSAL_MASKS.get(key)
    if mask is None:
        mask = torch.triu(
            torch.ones(2048, 2048, dtype=torch.bool, device=reference.device),
            diagonal=1,
        )
        _NPU_CAUSAL_MASKS[key] = mask
    return mask


def _fusion_attention_forward_saved(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Run NPU causal attention and retain exactly its backward statistics."""
    # torch-npu is optional outside the explicitly NPU-only execution path.
    import torch_npu  # pylint: disable=import-outside-toplevel

    output, softmax_max, softmax_sum, _, seed, offset = (
        torch_npu.npu_fusion_attention_v3(
            query,
            key,
            value,
            query.shape[1],
            "BNSD",
            atten_mask=_compressed_causal_mask(query),
            scale=scale,
            sparse_mode=3,
            pre_tockens=2147483647,
            next_tockens=0,
        )
    )
    return output, (output, softmax_max, softmax_sum, seed, offset)


def _fusion_attention_backward_saved(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output: torch.Tensor,
    scale: float,
    saved: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run NPU causal attention grad from one stage's forward statistics."""
    # torch-npu is optional outside the explicitly NPU-only execution path.
    import torch_npu  # pylint: disable=import-outside-toplevel

    if len(saved) != _FA_SAVED_TENSOR_COUNT:
        raise RuntimeError(
            f"Head Chunk FA expected {_FA_SAVED_TENSOR_COUNT} saved tensors, "
            f"got {len(saved)}"
        )
    attention_in, softmax_max, softmax_sum, seed, offset = saved
    return torch_npu.npu_fusion_attention_grad_v3(
        query,
        key,
        value,
        grad_output,
        query.shape[1],
        "BNSD",
        atten_mask=_compressed_causal_mask(query),
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        attention_in=attention_in,
        scale_value=scale,
        sparse_mode=3,
        pre_tockens=2147483647,
        next_tockens=0,
        seed=seed,
        offset=offset,
    )[:3]


def _seq_to_head(tensor: torch.Tensor, cp_mesh: Any | None) -> torch.Tensor:
    """Gather sequence and shard heads, or return the CP=1 tensor."""
    if cp_mesh is None or cp_mesh.size() <= 1:
        return tensor
    return ulysses_seq_to_head(tensor, 2, 1, cp_mesh)


def _head_to_seq(tensor: torch.Tensor, cp_mesh: Any | None) -> torch.Tensor:
    """Shard sequence and gather heads, or return the CP=1 tensor."""
    if cp_mesh is None or cp_mesh.size() <= 1:
        return tensor
    return ulysses_head_to_seq(tensor, 2, 1, cp_mesh)


def _project_stage_tensors(
    owner: Any,
    hidden_states: torch.Tensor,
    stage_weight: torch.Tensor,
    query_norm_weight: torch.Tensor,
    key_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one packed stage's local Q/K/V graph in BNSD layout."""
    packed_states = F.linear(hidden_states, stage_weight).view(
        *hidden_states.shape[:-1],
        kv_count,
        owner.qkv_group_width,
    )
    query_states, key_states, value_states = torch.split(
        packed_states,
        owner.qkv_split_sizes,
        dim=-1,
    )
    query_count = kv_count * owner.num_key_value_groups
    query_states = query_states.reshape(
        *hidden_states.shape[:-1],
        query_count,
        owner.qk_head_dim,
    )
    query_states = rms_norm(
        query_states,
        query_norm_weight,
        owner.query_norm_eps,
    )
    key_states = rms_norm(
        key_states,
        key_norm_weight,
        owner.key_norm_eps,
    )
    query_states, key_states = owner._apply_position_embeddings(
        query_states,
        key_states,
        cos,
        sin,
    )
    query_states = query_states.transpose(1, 2).contiguous()
    key_states = key_states.transpose(1, 2).contiguous()
    value_states = value_states.transpose(1, 2).contiguous()
    if owner.expand_kv_heads:
        key_states = key_states.repeat_interleave(
            owner.num_key_value_groups,
            dim=1,
        )
        value_states = value_states.repeat_interleave(
            owner.num_key_value_groups,
            dim=1,
        )
    return query_states, key_states, value_states


class _SavedStatsHeadChunk(torch.autograd.Function):
    """Head-stage projection recomputation with direct NPU FA backward."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        packed_weight: torch.Tensor,
        output_weight: torch.Tensor,
        query_norm_weight: torch.Tensor,
        key_norm_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        owner: Any,
    ) -> torch.Tensor:
        """Run bounded stages and save only NPU attention statistics."""
        batch_size, local_sequence, hidden_size = hidden_states.shape
        output = hidden_states.new_zeros(batch_size * local_sequence, hidden_size)
        saved_attention_tensors: list[torch.Tensor] = []
        cp_mesh = owner.head_chunk_cp_mesh
        for kv_start in range(0, owner.num_key_value_heads, owner.kv_chunk_size):
            kv_count = min(
                owner.kv_chunk_size,
                owner.num_key_value_heads - kv_start,
            )
            row_start = kv_start * owner.qkv_group_width
            row_end = (kv_start + kv_count) * owner.qkv_group_width
            query_local, key_local, value_local = _project_stage_tensors(
                owner,
                hidden_states,
                packed_weight[row_start:row_end],
                query_norm_weight,
                key_norm_weight,
                cos,
                sin,
                kv_count,
            )
            query_stage = _seq_to_head(query_local, cp_mesh).contiguous()
            key_stage = _seq_to_head(key_local, cp_mesh).contiguous()
            value_stage = _seq_to_head(value_local, cp_mesh).contiguous()
            attention_stage, attention_saved = _fusion_attention_forward_saved(
                query_stage,
                key_stage,
                value_stage,
                owner.scaling,
            )
            saved_attention_tensors.extend(attention_saved)

            attention_local = _head_to_seq(
                attention_stage,
                cp_mesh,
            ).transpose(1, 2).contiguous()
            query_start = kv_start * owner.num_key_value_groups
            query_count = kv_count * owner.num_key_value_groups
            column_start = query_start * owner.v_head_dim
            column_end = (query_start + query_count) * owner.v_head_dim
            output.addmm_(
                attention_local.reshape(
                    batch_size * local_sequence,
                    query_count * owner.v_head_dim,
                ),
                output_weight[:, column_start:column_end].transpose(0, 1),
            )

        ctx.owner = owner
        ctx.save_for_backward(
            hidden_states,
            packed_weight,
            output_weight,
            query_norm_weight,
            key_norm_weight,
            cos,
            sin,
            *saved_attention_tensors,
        )
        return output.reshape(batch_size, local_sequence, hidden_size)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Recompute projections/A2A and consume saved NPU FA statistics."""
        (
            hidden_states,
            packed_weight,
            output_weight,
            query_norm_weight,
            key_norm_weight,
            cos,
            sin,
            *saved_attention_tensors,
        ) = ctx.saved_tensors
        owner = ctx.owner
        expected_saved = owner.stage_count * _FA_SAVED_TENSOR_COUNT
        if len(saved_attention_tensors) != expected_saved:
            raise RuntimeError(
                "Head Chunk backward saved-tensor count mismatch: "
                f"expected {expected_saved}, got {len(saved_attention_tensors)}"
            )

        batch_size, local_sequence, hidden_size = hidden_states.shape
        grad_output_2d = grad_output.reshape(batch_size * local_sequence, hidden_size)
        grad_hidden = torch.zeros_like(hidden_states)
        grad_packed = torch.zeros_like(packed_weight)
        grad_output_weight = torch.zeros_like(output_weight)
        grad_query_norm = torch.zeros_like(query_norm_weight, dtype=torch.float32)
        grad_key_norm = torch.zeros_like(key_norm_weight, dtype=torch.float32)
        cp_mesh = owner.head_chunk_cp_mesh
        saved_offset = 0

        for kv_start in range(0, owner.num_key_value_heads, owner.kv_chunk_size):
            kv_count = min(
                owner.kv_chunk_size,
                owner.num_key_value_heads - kv_start,
            )
            row_start = kv_start * owner.qkv_group_width
            row_end = (kv_start + kv_count) * owner.qkv_group_width
            query_start = kv_start * owner.num_key_value_groups
            query_count = kv_count * owner.num_key_value_groups
            column_start = query_start * owner.v_head_dim
            column_end = (query_start + query_count) * owner.v_head_dim
            stage_saved = tuple(
                saved_attention_tensors[
                    saved_offset:saved_offset + _FA_SAVED_TENSOR_COUNT
                ]
            )
            saved_offset += _FA_SAVED_TENSOR_COUNT

            hidden_leaf = hidden_states.detach().requires_grad_(True)
            packed_leaf = packed_weight[row_start:row_end].detach().requires_grad_(True)
            query_norm_leaf = query_norm_weight.detach().requires_grad_(True)
            key_norm_leaf = key_norm_weight.detach().requires_grad_(True)
            with torch.enable_grad():
                query_local, key_local, value_local = _project_stage_tensors(
                    owner,
                    hidden_leaf,
                    packed_leaf,
                    query_norm_leaf,
                    key_norm_leaf,
                    cos,
                    sin,
                    kv_count,
                )

            with torch.no_grad():
                query_stage = _seq_to_head(query_local.detach(), cp_mesh).contiguous()
                key_stage = _seq_to_head(key_local.detach(), cp_mesh).contiguous()
                value_stage = _seq_to_head(value_local.detach(), cp_mesh).contiguous()
                attention_local = _head_to_seq(
                    stage_saved[0],
                    cp_mesh,
                ).transpose(1, 2).contiguous()
                attention_local_2d = attention_local.reshape(
                    batch_size * local_sequence,
                    query_count * owner.v_head_dim,
                )
                grad_output_weight[:, column_start:column_end].copy_(
                    torch.mm(grad_output_2d.transpose(0, 1), attention_local_2d)
                )
                grad_attention_local = torch.mm(
                    grad_output_2d,
                    output_weight[:, column_start:column_end],
                ).reshape(
                    batch_size,
                    local_sequence,
                    query_count,
                    owner.v_head_dim,
                ).transpose(1, 2).contiguous()
                grad_attention_stage = _seq_to_head(
                    grad_attention_local,
                    cp_mesh,
                ).contiguous()
                grad_query_stage, grad_key_stage, grad_value_stage = (
                    _fusion_attention_backward_saved(
                        query_stage,
                        key_stage,
                        value_stage,
                        grad_attention_stage,
                        owner.scaling,
                        stage_saved,
                    )
                )
                grad_query_local = _head_to_seq(
                    grad_query_stage,
                    cp_mesh,
                ).contiguous()
                grad_key_local = _head_to_seq(
                    grad_key_stage,
                    cp_mesh,
                ).contiguous()
                grad_value_local = _head_to_seq(
                    grad_value_stage,
                    cp_mesh,
                ).contiguous()

            (
                grad_hidden_stage,
                grad_packed_stage,
                grad_query_norm_stage,
                grad_key_norm_stage,
            ) = torch.autograd.grad(
                (query_local, key_local, value_local),
                (hidden_leaf, packed_leaf, query_norm_leaf, key_norm_leaf),
                (grad_query_local, grad_key_local, grad_value_local),
            )
            grad_hidden.add_(grad_hidden_stage)
            grad_packed[row_start:row_end].copy_(grad_packed_stage)
            grad_query_norm.add_(grad_query_norm_stage.float())
            grad_key_norm.add_(grad_key_norm_stage.float())

        if saved_offset != expected_saved:
            raise RuntimeError("Head Chunk backward did not consume all saved tensors")
        needs = ctx.needs_input_grad
        return (
            grad_hidden if needs[0] else None,
            grad_packed if needs[1] else None,
            grad_output_weight if needs[2] else None,
            grad_query_norm.to(query_norm_weight.dtype) if needs[3] else None,
            grad_key_norm.to(key_norm_weight.dtype) if needs[4] else None,
            None,
            None,
            None,
        )


@module_replacement
class HeadChunkGQAAttention(GQAAttention):
    """Run bounded packed-QKV stages with direct saved-stat FA backward.

    Unlike the historical projection-head implementation, this class keeps
    exactly the same ``linear_qkv`` parameter, checkpoint transforms, output
    projection, forward signature, and return tuple as :class:`GQAAttention`.
    A stage always owns whole KV groups, so no packed parameter row is split.
    """

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        head_chunk_size: int,
        expand_kv_heads: bool = False,
        attention_interface: Callable[..., tuple[torch.Tensor, torch.Tensor | None]] = (
            npu_fusion_attention_forward
        ),
    ) -> None:
        """Build a packed GQA replacement and configure its query-head stage.

        Args:
            module: Source Transformers-style GQA module.
            module_fqn: Fully qualified source name supplied by replacement.
            context: Replacement context supplied by model construction.
            head_chunk_size: Maximum global query heads computed per stage.
            expand_kv_heads: Repeat each stage's K/V heads to query-head count.
                This permits Pure Ulysses when a stage contains fewer KV heads
                than the CP degree; autograd sums repeated K/V gradients.
            attention_interface: Attention kernel retained for the CP=1
                evaluation/no-grad parent fallback. The chunked training path
                deliberately uses NPU FA v3 directly because its custom
                backward consumes that operator's saved statistics.

        Raises:
            NotImplementedError: If tensor parallelism is active.
            ValueError: If the source or stage violates the first-version
                training contract.
        """
        if context is not None and context.get("tp", False):
            raise NotImplementedError(
                "Head Chunk GQA version one does not support tensor parallelism"
            )
        super().__init__(
            module=module,
            module_fqn=module_fqn,
            context=context,
            attention_interface=attention_interface,
        )
        if (
            not isinstance(head_chunk_size, int)
            or isinstance(head_chunk_size, bool)
            or head_chunk_size <= 0
        ):
            raise ValueError(
                "head_chunk_size must be a positive integer, got "
                f"{head_chunk_size!r}"
            )
        if not isinstance(expand_kv_heads, bool):
            raise ValueError(
                "expand_kv_heads must be a boolean, got "
                f"{expand_kv_heads!r}"
            )
        if head_chunk_size % self.num_key_value_groups:
            raise ValueError(
                "head_chunk_size must contain whole KV groups: "
                f"{head_chunk_size} is not divisible by "
                f"num_key_value_groups={self.num_key_value_groups}"
            )
        if head_chunk_size > self.num_heads or self.num_heads % head_chunk_size:
            raise ValueError(
                "head_chunk_size must divide num_attention_heads without a "
                f"tail stage; got {head_chunk_size} and {self.num_heads}"
            )
        if self.linear_qkv.bias is not None or self.o_proj.bias is not None:
            raise ValueError("Head Chunk GQA version one supports bias-free projections only")
        if self.attention_dropout != 0:
            raise ValueError("Head Chunk GQA version one requires attention_dropout=0")
        if self.is_causal is not True:
            raise ValueError("Head Chunk GQA version one requires causal attention")
        if self.sliding_window is not None:
            raise ValueError("Head Chunk GQA version one does not support sliding-window attention")
        if self.qk_head_dim != self.v_head_dim:
            raise ValueError("Head Chunk GQA version one requires equal QK and V head dimensions")
        if self.q_norm is None or not isinstance(getattr(self.q_norm, "weight", None), nn.Parameter):
            raise ValueError("Head Chunk GQA requires a trainable Q RMSNorm weight")
        if self.k_norm is None or not isinstance(getattr(self.k_norm, "weight", None), nn.Parameter):
            raise ValueError("Head Chunk GQA requires a trainable K RMSNorm weight")
        query_norm_eps = getattr(
            self.q_norm,
            "variance_epsilon",
            getattr(self.q_norm, "eps", None),
        )
        key_norm_eps = getattr(
            self.k_norm,
            "variance_epsilon",
            getattr(self.k_norm, "eps", None),
        )
        if query_norm_eps is None or key_norm_eps is None:
            raise ValueError("Head Chunk GQA Q/K norms must expose eps")
        self.query_norm_eps = float(query_norm_eps)
        self.key_norm_eps = float(key_norm_eps)
        self.head_chunk_size = head_chunk_size
        self.kv_chunk_size = head_chunk_size // self.num_key_value_groups
        self.expand_kv_heads = expand_kv_heads
        self.stage_count = self.num_heads // self.head_chunk_size
        self.head_chunk_cp_mesh = None

    def _apply_position_embeddings(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the same RoPE variants as the current full GQA module."""
        if cos is None or sin is None:
            return query_states, key_states
        if self.rotary_interleaved:
            return apply_rotary_pos_emb_interleave(
                query_states,
                key_states,
                cos,
                sin,
                unsqueeze_dim=2,
            )
        if cos.shape[-1] < self.qk_head_dim:
            rotary_dim = cos.shape[-1]
            query_rot, query_pass = (
                query_states[..., :rotary_dim],
                query_states[..., rotary_dim:],
            )
            key_rot, key_pass = (
                key_states[..., :rotary_dim],
                key_states[..., rotary_dim:],
            )
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)
            query_states = torch.cat(
                (query_rot * cos + rotate_half(query_rot) * sin, query_pass),
                dim=-1,
            )
            key_states = torch.cat(
                (key_rot * cos + rotate_half(key_rot) * sin, key_pass),
                dim=-1,
            )
            return query_states, key_states
        return apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            unsqueeze_dim=2,
        )

    def _validate_chunk_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
        past_key_values: Any | None,
        actual_seq_len: torch.Tensor | Sequence[int] | None,
        kwargs: Mapping[str, Any],
    ) -> None:
        """Reject inputs whose semantics are not implemented by version one."""
        if hidden_states.dim() != 3:
            raise ValueError(
                "Head Chunk GQA requires hidden_states shaped [B, S, H], got "
                f"{tuple(hidden_states.shape)}"
            )
        if hidden_states.device.type != "npu":
            raise ValueError("Head Chunk GQA training requires Ascend NPU tensors")
        if hidden_states.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Head Chunk GQA training requires float16 or bfloat16")
        if position_embeddings is None or len(position_embeddings) != 2:
            raise ValueError("Head Chunk GQA requires precomputed cosine and sine tensors")
        cos, sin = position_embeddings
        if not isinstance(cos, torch.Tensor) or not isinstance(sin, torch.Tensor):
            raise ValueError("Head Chunk GQA cosine and sine values must be tensors")
        if cos.device != hidden_states.device or sin.device != hidden_states.device:
            raise ValueError("Head Chunk GQA position embeddings must share the input device")
        if cos.dtype != hidden_states.dtype or sin.dtype != hidden_states.dtype:
            raise ValueError("Head Chunk GQA position embeddings must share the input dtype")
        if cos.requires_grad or sin.requires_grad:
            raise ValueError("Head Chunk GQA does not return position-embedding gradients")
        if attention_mask is not None:
            raise ValueError("Head Chunk GQA version one requires implicit causal attention")
        if past_key_values is not None:
            raise ValueError("Head Chunk GQA training does not support KV cache")
        if actual_seq_len is not None:
            raise ValueError("Head Chunk GQA version one does not support packed sequences")
        packed_arguments = [
            name for name in _PACKED_SEQUENCE_ARGUMENTS if kwargs.get(name) is not None
        ]
        if packed_arguments:
            raise ValueError(
                "Head Chunk GQA version one does not support packed sequences; "
                f"received {packed_arguments}"
            )
        unsupported_arguments = [
            name
            for name in _UNSUPPORTED_ATTENTION_ARGUMENTS
            if kwargs.get(name) is not None
        ]
        if unsupported_arguments:
            raise ValueError(
                "Head Chunk GQA version one uses fixed dense causal FA options; "
                f"received unsupported arguments {unsupported_arguments}"
            )
        if kwargs.get("is_causal") is not None and kwargs["is_causal"] is not True:
            raise ValueError("Head Chunk GQA version one requires is_causal=true")
        if kwargs.get("output_attentions", False):
            raise ValueError("Head Chunk GQA does not support output_attentions=true")
        weights = (
            self.linear_qkv.weight,
            self.o_proj.weight,
            self.q_norm.weight,
            self.k_norm.weight,
        )
        if any(weight.device != hidden_states.device for weight in weights):
            raise ValueError("Head Chunk GQA inputs and parameters must share a device")
        if any(weight.dtype != hidden_states.dtype for weight in weights):
            raise ValueError("Head Chunk GQA inputs and parameters must share a dtype")
        cp_size = 1 if self.head_chunk_cp_mesh is None else self.head_chunk_cp_mesh.size()
        if self.head_chunk_size % cp_size:
            raise ValueError("each query-head stage must be divisible by CP size")
        if not self.expand_kv_heads and self.kv_chunk_size % cp_size:
            raise ValueError(
                "each KV stage must be divisible by CP size unless "
                "expand_kv_heads=true"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        actual_seq_len: torch.Tensor | Sequence[int] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run bounded query-head stages with saved-stat FA backward."""
        if (
            self.head_chunk_cp_mesh is None
            and (not self.training or not torch.is_grad_enabled())
        ):
            return super().forward(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                actual_seq_len=actual_seq_len,
                **kwargs,
            )

        self._validate_chunk_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            actual_seq_len,
            kwargs,
        )
        cos, sin = position_embeddings
        output = _SavedStatsHeadChunk.apply(
            hidden_states,
            self.linear_qkv.weight,
            self.o_proj.weight,
            self.q_norm.weight,
            self.k_norm.weight,
            cos,
            sin,
            self,
        )
        return output, None


# Compatibility name for users of the historical PR. The implementation is
# now the packed GQA replacement above, not the old raw-ProcessGroup module.
ProjectionHeadChunkAttention = HeadChunkGQAAttention


__all__ = ["HeadChunkGQAAttention", "ProjectionHeadChunkAttention"]
