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
"""DSA sparse-attention rescaling for parameter-sink attention."""

from typing import Any, Tuple

import torch  # pylint: disable=forbidden-backend-import
import torch.nn.functional as F  # pylint: disable=forbidden-backend-import
import torch_npu
from einops import rearrange
import omni_training_custom_ops  # noqa: F401  # pylint: disable=unused-import


class _SparseAttentionRescale(torch.autograd.Function):
    """Autograd bridge for sparse attention with separate sink parameters."""

    @staticmethod
    def forward(
        ctx: Any,
        query_nope: torch.Tensor,
        compressed_kv: torch.Tensor,
        query_rope: torch.Tensor,
        key_rope: torch.Tensor,
        sink_key: torch.Tensor,
        sink_value: torch.Tensor,
        topk_indices: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        num_heads: int,
        scale: float,
        keep_prob: float,
        actual_seq_qlen: torch.Tensor,
        actual_seq_kvlen: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run sparse and sink attention and rescale their outputs."""
        query_nope_tnd, compressed_kv_tnd, query_rope_tnd, key_rope_tnd = [
            rearrange(tensor, "b s n d -> (b s) n d")
            for tensor in (query_nope, compressed_kv, query_rope, key_rope)
        ]
        output, softmax_max, softmax_sum = torch.ops.custom.npu_sparse_flash_attention_enhance(
            query_nope_tnd,
            compressed_kv_tnd,
            compressed_kv_tnd,
            topk_indices,
            scale,
            block_table=None,
            actual_seq_lengths_query=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_kvlen,
            query_rope=query_rope_tnd,
            key_rope=key_rope_tnd,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="TND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=True,
        )
        if query_rope.size(-1) > 0:
            output = F.pad(output, [0, query_rope.size(-1)])
        output = rearrange(output, "(b s) n d -> b s n d", b=batch_size, s=sequence_length)

        query = torch.cat([query_nope, query_rope], dim=-1)
        sink_query = rearrange(query, "b s n d -> s b (n d)")
        sink_key_sbh = rearrange(sink_key, "b s n d -> s b (n d)")
        sink_value_sbh = rearrange(sink_value, "b s n d -> s b (n d)")
        sink_output, sink_softmax_max, sink_softmax_sum = torch_npu.npu_fusion_attention(
            sink_query,
            sink_key_sbh,
            sink_value_sbh,
            num_heads,
            "SBH",
            pse=None,
            padding_mask=None,
            atten_mask=None,
            scale=scale,
            keep_prob=keep_prob,
            inner_precise=0,
            sparse_mode=0,
            actual_seq_qlen=None,
            actual_seq_kvlen=None,
        )[:3]

        sink_output = rearrange(sink_output, "s b (n d) -> b s n d", n=num_heads)
        softmax_max_rescale = softmax_max.squeeze(0).view(batch_size, sequence_length, num_heads)
        softmax_sum_rescale = softmax_sum.squeeze(0).view(batch_size, sequence_length, num_heads)
        sink_softmax_max_rescale = sink_softmax_max[:, :, :, 0].transpose(1, 2)
        sink_softmax_sum_rescale = sink_softmax_sum[:, :, :, 0].transpose(1, 2)
        combined_max = torch.maximum(softmax_max_rescale, sink_softmax_max_rescale)
        output_sum = softmax_sum_rescale * torch.exp(softmax_max_rescale - combined_max)
        sink_sum = sink_softmax_sum_rescale * torch.exp(sink_softmax_max_rescale - combined_max)
        combined_sum = output_sum + sink_sum
        output_scale = (output_sum / combined_sum).unsqueeze(-1)
        sink_scale = (sink_sum / combined_sum).unsqueeze(-1)
        rescaled_output = output * output_scale + sink_output * sink_scale
        rescaled_output = rescaled_output.to(dtype=output.dtype)

        ctx.save_for_backward(
            query_nope,
            compressed_kv,
            query_rope,
            key_rope,
            sink_key,
            sink_value,
            topk_indices,
            softmax_max,
            softmax_sum,
            sink_softmax_max,
            sink_softmax_sum,
            rescaled_output,
            output_scale,
            sink_scale,
        )
        ctx.params = (batch_size, sequence_length, num_heads, scale, keep_prob, actual_seq_qlen, actual_seq_kvlen)
        return rescaled_output, softmax_max, softmax_sum

    @staticmethod
    def backward(
        ctx: Any,
        grad_rescaled_output: torch.Tensor,
        _grad_softmax_max: torch.Tensor,
        _grad_softmax_sum: torch.Tensor,
    ) -> tuple:
        """Run the explicit sparse- and fusion-attention backward operators."""
        (
            query_nope,
            compressed_kv,
            query_rope,
            key_rope,
            sink_key,
            sink_value,
            topk_indices,
            softmax_max,
            softmax_sum,
            sink_softmax_max,
            sink_softmax_sum,
            rescaled_output,
            output_scale,
            sink_scale,
        ) = ctx.saved_tensors
        batch_size, sequence_length, num_heads, scale, keep_prob, actual_seq_qlen, actual_seq_kvlen = ctx.params
        query_nope_tnd, compressed_kv_tnd, query_rope_tnd, key_rope_tnd = [
            rearrange(tensor, "b s n d -> (b s) n d")
            for tensor in (query_nope, compressed_kv, query_rope, key_rope)
        ]
        grad_output = rearrange(output_scale * grad_rescaled_output, "b s n d -> (b s) n d")
        rescaled_output_tnd = rearrange(rescaled_output, "b s n d -> (b s) n d")
        grad_output = grad_output[:, :, :-query_rope.size(-1)]
        rescaled_output_tnd = rescaled_output_tnd[:, :, :-query_rope.size(-1)]

        grad_query_nope, grad_key, grad_value, grad_query_rope, grad_key_rope = (
            torch.ops.custom.npu_sparse_flash_attention_grad_enhance(
                query_nope_tnd,
                compressed_kv_tnd,
                compressed_kv_tnd,
                topk_indices,
                grad_output.to(sink_key.dtype),
                rescaled_output_tnd,
                softmax_max,
                softmax_sum,
                scale,
                sparse_block_size=1,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                query_rope=query_rope_tnd,
                key_rope=key_rope_tnd,
                layout="TND",
                sparse_mode=3,
                attention_mode=2,
                deterministic=torch.are_deterministic_algorithms_enabled(),
            )
        )
        sink_grad_output = rearrange(sink_scale * grad_rescaled_output, "b s n d -> s b (n d)")
        query = torch.cat([query_nope, query_rope], dim=-1)
        sink_query = rearrange(query, "b s n d -> s b (n d)")
        sink_key_sbh = rearrange(sink_key, "b s n d -> s b (n d)")
        sink_value_sbh = rearrange(sink_value, "b s n d -> s b (n d)")
        sink_grad_query, sink_grad_key, sink_grad_value, *_ = torch_npu.npu_fusion_attention_grad(
            sink_query,
            sink_key_sbh,
            sink_value_sbh,
            sink_grad_output.to(sink_key.dtype),
            num_heads,
            "SBH",
            pse=None,
            padding_mask=None,
            atten_mask=None,
            softmax_max=sink_softmax_max,
            softmax_sum=sink_softmax_sum,
            attention_in=rearrange(rescaled_output, "b s n d -> s b (n d)"),
            scale_value=scale,
            inner_precise=0,
            keep_prob=keep_prob,
            actual_seq_qlen=None,
            actual_seq_kvlen=None,
            sparse_mode=0,
        )
        grad_query_nope, grad_key, grad_value, grad_query_rope, grad_key_rope = [
            rearrange(tensor, "(b s) n d -> b s n d", b=batch_size, s=sequence_length)
            for tensor in (grad_query_nope, grad_key, grad_value, grad_query_rope, grad_key_rope)
        ]
        sink_grad_query = rearrange(sink_grad_query, "s b (n d) -> b s n d", n=num_heads)
        sink_grad_key = rearrange(sink_grad_key, "s b (n d) -> b s n d", n=sink_key.size(2))
        sink_grad_value = rearrange(sink_grad_value, "s b (n d) -> b s n d", n=sink_value.size(2))
        sink_grad_query_nope, sink_grad_query_rope = torch.split(
            sink_grad_query,
            [query_nope.size(-1), query_rope.size(-1)],
            dim=-1,
        )
        return (
            grad_query_nope + sink_grad_query_nope,
            grad_key + grad_value,
            grad_query_rope + sink_grad_query_rope,
            grad_key_rope,
            sink_grad_key,
            sink_grad_value,
            *((None,) * 8),
        )


def dsa_sparse_attention_rescale(
    query_nope: torch.Tensor,
    compressed_kv: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    sink_key: torch.Tensor,
    sink_value: torch.Tensor,
    topk_indices: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    num_heads: int,
    scale: float,
    keep_prob: float,
    actual_seq_qlen: torch.Tensor,
    actual_seq_kvlen: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine DSA sparse attention and parameter-sink attention.

    The four DSA states and both sink states use BSND layout. ``topk_indices``
    is the int32 output from :func:`dsa_indexer`; sequence lengths are
    cumulative int32 NPU tensors. The returned BSND output is followed by the
    sparse-attention softmax maximum and sum used by :func:`dsa_kl_loss`.
    """
    return _SparseAttentionRescale.apply(
        query_nope,
        compressed_kv,
        query_rope,
        key_rope,
        sink_key,
        sink_value,
        topk_indices,
        batch_size,
        sequence_length,
        num_heads,
        scale,
        keep_prob,
        actual_seq_qlen,
        actual_seq_kvlen,
    )
