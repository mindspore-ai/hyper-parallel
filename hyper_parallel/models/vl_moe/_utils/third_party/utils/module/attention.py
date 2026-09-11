# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .mlp import (
    LinearWithMatmul,
    LinearWithFusedOps,
)
from .aux_loss import AuxLossAutoScaler


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

try:
    import torch_npu

    HAS_NPU = True
except Exception:
    HAS_NPU = False

try:
    import omni_training_custom_ops  # noqa: F401
except Exception:
    pass


class FusedRMSNorm(nn.Module):
    """RMS normalization with optional NPU fused kernel."""

    def __init__(self, hidden_size, eps=1e-5, use_fused_rmsnorm=False):
        super().__init__()
        self.eps = eps
        self.use_fused_rmsnorm = use_fused_rmsnorm
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if self.use_fused_rmsnorm and HAS_NPU and x.device.type != 'cpu':
            return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ============================================================================
# RoPE helpers
# ============================================================================

def _rotate_half(x, rotary_interleaved=False):
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    dim = x.shape[-1]
    index1 = np.ones(dim)
    index1[::2] = 0
    index2 = np.zeros(dim)
    index2[::2] = -1
    rotation_matrix = np.eye(dim, k=1) * index1 + np.eye(dim, k=-1) * index2
    rotation_matrix = torch.from_numpy(rotation_matrix[None, None, :, :]).to(x.dtype).to(x.device)
    return torch.matmul(x, rotation_matrix)


def apply_rotary_pos_emb(t, cos, sin, rotary_interleaved=False, use_fused_rotary_pos_emb=False):
    """Apply rotary embeddings using precomputed cosine and sine tensors."""
    rot_dim = cos.shape[-1]
    t_dim = t.shape[-1]
    t_pass = None
    if rot_dim != t_dim:
        t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    cos = cos.to(t.dtype)
    sin = sin.to(t.dtype)

    while cos.ndim < t.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

    if use_fused_rotary_pos_emb and HAS_NPU and t.device.type != "cpu":
        rotary_mode = "interleave" if rotary_interleaved else "half"
        sequence_length, batch_size, num_heads, head_dim = t.shape
        if rotary_interleaved and batch_size > 1 and sequence_length > 1:
            t = torch_npu.npu_rotary_mul(
                t.reshape(batch_size * sequence_length, 1, num_heads, head_dim),
                cos.reshape(batch_size * sequence_length, 1, cos.shape[-2], cos.shape[-1]),
                sin.reshape(batch_size * sequence_length, 1, sin.shape[-2], sin.shape[-1]),
                rotary_mode=rotary_mode,
            ).reshape(sequence_length, batch_size, num_heads, head_dim)
        else:
            t = torch_npu.npu_rotary_mul(t.clone(), cos, sin, rotary_mode=rotary_mode)
    else:
        t = (t * cos) + (_rotate_half(t, rotary_interleaved) * sin)

    if rot_dim != t_dim:
        return torch.cat((t, t_pass), dim=-1)
    return t


class _FusedMOMEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, weight, mask):
        ctx.save_for_backward(input_tensor, weight, mask)
        output = torch.ops.custom.npu_aggregate_hidden(
            input_tensor, weight, mask=mask
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight, mask = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input, grad_weight = torch.ops.custom.npu_aggregate_hidden_grad(
            grad_output, input_tensor, weight, mask=mask,
        )
        return grad_input, grad_weight, None


def _apply_mome(hidden_states, mome_mask, conv, use_fused):
    """Apply masked causal depthwise convolution with a residual connection."""
    if mome_mask is None:
        raise ValueError("mome_mask is required when MOME is enabled")
    if mome_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"mome_mask must have shape {hidden_states.shape[:2]}, but got {tuple(mome_mask.shape)}"
        )

    mome_mask = mome_mask.to(device=hidden_states.device, dtype=torch.bool)
    padding = conv.kernel_size[0] - 1
    padded_states = F.pad(hidden_states, (0, 0, padding, 0))

    if use_fused:
        padded_mask = F.pad(mome_mask, (padding, 0), value=False)
        weight = conv.weight.squeeze(1).transpose(0, 1)
        mixed_states = _FusedMOMEFunction.apply(
            padded_states.transpose(0, 1).contiguous(),
            weight,
            padded_mask,
        )
        mixed_states = mixed_states[padding:].transpose(0, 1).contiguous()
    else:
        mixed_states = conv(padded_states.transpose(1, 2)).transpose(1, 2)
        mixed_states = mixed_states * mome_mask.unsqueeze(-1).to(mixed_states.dtype)

    return hidden_states + mixed_states


# ============================================================================
# FA rescale functions
# ============================================================================

def apply_FA_rescale_forward(
    output, softmax_max, softmax_sum,
    output_sink, softmax_max_sink, softmax_sum_sink,
    seq_length, bsz, n_head
):
    output_orig_reshaped = rearrange(output, '(b s) n d -> s b n d', b=bsz, s=seq_length)
    output_sink_reshaped = rearrange(output_sink, 's b (n d) -> s b n d', n=n_head)

    softmax_sum_orig_reshaped = softmax_sum[:, :, 0].view(bsz, seq_length, n_head).transpose(0, 1)
    softmax_max_orig_reshaped = softmax_max[:, :, 0].view(bsz, seq_length, n_head).transpose(0, 1)

    softmax_sum_sink_reshaped = rearrange(softmax_sum_sink[:, :, :, 0], 'b n s -> s b n')
    softmax_max_sink_reshaped = rearrange(softmax_max_sink[:, :, :, 0], 'b n s -> s b n')

    softmax_max_full = torch.maximum(softmax_max_orig_reshaped, softmax_max_sink_reshaped)
    softmax_sum_orig_add_max = softmax_sum_orig_reshaped * torch.exp(softmax_max_orig_reshaped - softmax_max_full)
    softmax_sum_sink_add_max = softmax_sum_sink_reshaped * torch.exp(softmax_max_sink_reshaped - softmax_max_full)

    softmax_sum_full = softmax_sum_orig_add_max + softmax_sum_sink_add_max
    rescale_orig = softmax_sum_orig_add_max / softmax_sum_full
    rescale_sink = softmax_sum_sink_add_max / softmax_sum_full

    rescale_orig_expanded = rescale_orig.unsqueeze(3)
    rescale_sink_expanded = rescale_sink.unsqueeze(3)

    core_attn_out_reshaped = output_orig_reshaped * rescale_orig_expanded + \
                                output_sink_reshaped * rescale_sink_expanded

    core_attn_out_reshaped = core_attn_out_reshaped.to(dtype=output.dtype)
    return core_attn_out_reshaped, rescale_orig_expanded, rescale_sink_expanded


class FArescale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, param_sink_key, param_sink_value,
                attention_mask, bsz, seq_length, n_head, scale,
                pre_tockens, next_tockens, keep_prob, sparse_mode, actual_q_len, actual_kv_len):
        output, softmax_max, softmax_sum = torch_npu.npu_fusion_attention(
            query, key, value, n_head, "TND",
            pse=None, padding_mask=None, atten_mask=attention_mask,
            scale=scale, pre_tockens=pre_tockens, next_tockens=next_tockens,
            keep_prob=keep_prob, inner_precise=0, sparse_mode=sparse_mode,
            actual_seq_qlen=actual_q_len, actual_seq_kvlen=actual_kv_len,
            softmax_layout="TND",
        )[:3]

        query_for_sink = rearrange(query, '(b s) n d -> s b (n d)', b=bsz, s=seq_length, n=n_head)

        output_sink, softmax_max_sink, softmax_sum_sink = torch_npu.npu_fusion_attention(
            query_for_sink, param_sink_key, param_sink_value,
            n_head, "SBH",
            pse=None, padding_mask=None, atten_mask=None,
            scale=scale, keep_prob=keep_prob, inner_precise=0, sparse_mode=0,
            actual_seq_qlen=None, actual_seq_kvlen=None,
        )[:3]

        rescaled_output, rescale_orig_expanded, rescale_sink_expanded = apply_FA_rescale_forward(
            output, softmax_max, softmax_sum,
            output_sink, softmax_max_sink, softmax_sum_sink,
            seq_length, bsz, n_head
        )
        ctx.save_for_backward(
            query, key, value, param_sink_key, param_sink_value, attention_mask,
            softmax_max, softmax_sum,
            softmax_max_sink, softmax_sum_sink,
            rescaled_output, rescale_orig_expanded, rescale_sink_expanded
        )
        ctx.params = (
            bsz, seq_length, n_head, scale, pre_tockens, next_tockens,
            keep_prob, sparse_mode, actual_q_len, actual_kv_len
        )
        return rescaled_output, softmax_max

    @staticmethod
    def backward(ctx, grad_rescaled_output, grad_softmax_max):
        (
            query, key, value, param_sink_key, param_sink_value, attention_mask,
            softmax_max, softmax_sum,
            softmax_max_sink, softmax_sum_sink,
            rescaled_output, rescale_orig_expanded, rescale_sink_expanded
        ) = ctx.saved_tensors
        (
            bsz, seq_length, n_head, scale, pre_tockens, next_tockens,
            keep_prob, sparse_mode, actual_q_len, actual_kv_len
        ) = ctx.params
        dtype = param_sink_key.dtype
        grad_output = rearrange(rescale_orig_expanded * grad_rescaled_output, 's b n d -> (b s) n d')
        dq, dk, dv, *_ = torch_npu.npu_fusion_attention_grad(
            query, key, value, grad_output.to(dtype), n_head, "TND",
            pse=None, padding_mask=None, atten_mask=attention_mask,
            softmax_max=softmax_max, softmax_sum=softmax_sum,
            attention_in=rearrange(rescaled_output, 's b n d -> (b s) n d'),
            scale_value=scale, pre_tockens=pre_tockens, next_tockens=next_tockens,
            inner_precise=0, keep_prob=keep_prob,
            actual_seq_qlen=actual_q_len, actual_seq_kvlen=actual_kv_len,
            sparse_mode=sparse_mode, softmax_layout="TND",
        )
        grad_output_sink = rearrange(rescale_sink_expanded * grad_rescaled_output, 's b n d -> s b (n d)')
        query_for_sink = rearrange(query, '(b s) n d -> s b (n d)', b=bsz, s=seq_length, n=n_head)
        dq_sink, dk_sink, dv_sink, *_ = torch_npu.npu_fusion_attention_grad(
            query_for_sink, param_sink_key, param_sink_value, grad_output_sink.to(dtype), n_head, "SBH",
            pse=None, padding_mask=None, atten_mask=None,
            softmax_max=softmax_max_sink, softmax_sum=softmax_sum_sink,
            attention_in=rearrange(rescaled_output, 's b n d -> s b (n d)'),
            scale_value=scale, inner_precise=0, keep_prob=keep_prob,
            actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0
        )
        dq_sink_tnd = rearrange(dq_sink, 's b (n d) -> (b s) n d', b=bsz, s=seq_length, n=n_head)
        grad_returns = (dq + dq_sink_tnd, dk, dv, dk_sink, dv_sink)
        none_returns = (None,) * 11
        return (*grad_returns, *none_returns)


class SFArescale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_nope, compressed_kv_norm, q_pe, k_pe,
                param_sink_key, param_sink_value,
                topk_indices, bsz, seq_length, n_head, scale,
                keep_prob, npu_actual_q_len, npu_actual_kv_len):
        q_nope_tnd, compressed_kv_tnd, q_pe_tnd, k_pe_tnd = [
            rearrange(tensor, "b s n d -> (b s) n d")
            for tensor in (q_nope, compressed_kv_norm, q_pe, k_pe)
        ]
        output, softmax_max, softmax_sum = torch.ops.custom.npu_sparse_flash_attention_enhance(
            q_nope_tnd, compressed_kv_tnd, compressed_kv_tnd, topk_indices, scale, block_table=None,
            actual_seq_lengths_query=npu_actual_q_len,
            actual_seq_lengths_kv=npu_actual_kv_len,
            query_rope=q_pe_tnd, key_rope=k_pe_tnd, sparse_block_size=1,
            layout_query="TND", layout_kv="TND", sparse_mode=3,
            attention_mode=2, return_softmax_lse=True)
        qk_rope_head_dim = q_pe.size(-1)
        if qk_rope_head_dim > 0:
            output = F.pad(output, [0, qk_rope_head_dim])
        output = rearrange(output, "(b s) n d -> b s n d", b=bsz, s=seq_length)

        query = torch.cat([q_nope, q_pe], dim=-1)
        query_for_sink = rearrange(query, "b s n d -> s b (n d)")
        param_sink_key_sbh = rearrange(param_sink_key, "b s n d -> s b (n d)")
        param_sink_value_sbh = rearrange(param_sink_value, "b s n d -> s b (n d)")
        output_sink, softmax_max_sink, softmax_sum_sink = torch_npu.npu_fusion_attention(
            query_for_sink, param_sink_key_sbh, param_sink_value_sbh,
            n_head, "SBH",
            pse=None, padding_mask=None, atten_mask=None,
            scale=scale, keep_prob=keep_prob, inner_precise=0, sparse_mode=0,
            actual_seq_qlen=None, actual_seq_kvlen=None,
        )[:3]

        output_sink = rearrange(output_sink, "s b (n d) -> b s n d", n=n_head)
        softmax_max_orig = softmax_max.squeeze(0).view(bsz, seq_length, n_head)
        softmax_sum_orig = softmax_sum.squeeze(0).view(bsz, seq_length, n_head)
        softmax_max_sink_rescale = softmax_max_sink[:, :, :, 0].transpose(1, 2)
        softmax_sum_sink_rescale = softmax_sum_sink[:, :, :, 0].transpose(1, 2)
        softmax_max_full = torch.maximum(softmax_max_orig, softmax_max_sink_rescale)
        softmax_sum_orig = softmax_sum_orig * torch.exp(softmax_max_orig - softmax_max_full)
        softmax_sum_sink_rescale = softmax_sum_sink_rescale * torch.exp(
            softmax_max_sink_rescale - softmax_max_full
        )
        softmax_sum_full = softmax_sum_orig + softmax_sum_sink_rescale
        rescale_orig = (softmax_sum_orig / softmax_sum_full).unsqueeze(-1)
        rescale_sink = (softmax_sum_sink_rescale / softmax_sum_full).unsqueeze(-1)
        rescaled_output = output * rescale_orig + output_sink * rescale_sink
        rescaled_output = rescaled_output.to(dtype=output.dtype)

        ctx.save_for_backward(
            q_nope, compressed_kv_norm, q_pe, k_pe, param_sink_key, param_sink_value, topk_indices,
            softmax_max, softmax_sum,
            softmax_max_sink, softmax_sum_sink,
            rescaled_output, rescale_orig, rescale_sink
        )
        ctx.params = (
            bsz, seq_length, n_head, scale, keep_prob, npu_actual_q_len, npu_actual_kv_len
        )
        return rescaled_output, softmax_max, softmax_sum

    @staticmethod
    def backward(ctx, grad_rescaled_output, grad_softmax_max, grad_softmax_sum):
        (
            q_nope, compressed_kv_norm, q_pe, k_pe, param_sink_key, param_sink_value, topk_indices,
            softmax_max, softmax_sum,
            softmax_max_sink, softmax_sum_sink,
            rescaled_output, rescale_orig, rescale_sink
        ) = ctx.saved_tensors
        (
            bsz, seq_length, n_head, scale, keep_prob, npu_actual_q_len, npu_actual_kv_len
        ) = ctx.params
        dtype = param_sink_key.dtype
        q_nope_tnd, compressed_kv_tnd, q_pe_tnd, k_pe_tnd = [
            rearrange(tensor, "b s n d -> (b s) n d")
            for tensor in (q_nope, compressed_kv_norm, q_pe, k_pe)
        ]
        grad_output = rearrange(rescale_orig * grad_rescaled_output, "b s n d -> (b s) n d")
        rescaled_output_tnd = rearrange(rescaled_output, "b s n d -> (b s) n d")

        grad_output = grad_output[:, :, :-q_pe.size(-1)]
        rescaled_output_tnd = rescaled_output_tnd[:, :, :-q_pe.size(-1)]

        dq_nope, dk, dv, dq_pe, dk_pe = torch.ops.custom.npu_sparse_flash_attention_grad_enhance(
            q_nope_tnd, compressed_kv_tnd, compressed_kv_tnd, topk_indices,
            grad_output.to(dtype), rescaled_output_tnd,
            softmax_max, softmax_sum,
            scale, sparse_block_size=1,
            actual_seq_qlen=npu_actual_q_len,
            actual_seq_kvlen=npu_actual_kv_len,
            query_rope=q_pe_tnd, key_rope=k_pe_tnd,
            layout="TND", sparse_mode=3,
            attention_mode=2, deterministic=torch.are_deterministic_algorithms_enabled())
        grad_output_sink = rearrange(
            rescale_sink * grad_rescaled_output, "b s n d -> s b (n d)"
        )
        query = torch.cat([q_nope, q_pe], dim=-1)
        query_for_sink = rearrange(query, "b s n d -> s b (n d)")
        param_sink_key_sbh = rearrange(param_sink_key, "b s n d -> s b (n d)")
        param_sink_value_sbh = rearrange(param_sink_value, "b s n d -> s b (n d)")
        dq_sink, dk_sink, dv_sink, *_ = torch_npu.npu_fusion_attention_grad(
            query_for_sink, param_sink_key_sbh, param_sink_value_sbh,
            grad_output_sink.to(dtype), n_head, "SBH",
            pse=None, padding_mask=None, atten_mask=None,
            softmax_max=softmax_max_sink, softmax_sum=softmax_sum_sink,
            attention_in=rearrange(rescaled_output, "b s n d -> s b (n d)"),
            scale_value=scale, inner_precise=0, keep_prob=keep_prob,
            actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0
        )
        dq_nope, dk, dv, dq_pe, dk_pe = [
            rearrange(tensor, "(b s) n d -> b s n d", b=bsz, s=seq_length)
            for tensor in (dq_nope, dk, dv, dq_pe, dk_pe)
        ]
        dq_sink = rearrange(dq_sink, "s b (n d) -> b s n d", n=n_head)
        dk_sink = rearrange(dk_sink, "s b (n d) -> b s n d", n=param_sink_key.size(2))
        dv_sink = rearrange(dv_sink, "s b (n d) -> b s n d", n=param_sink_value.size(2))
        dq_nope_sink, dq_pe_sink = torch.split(dq_sink, [q_nope.size(-1), q_pe.size(-1)], dim=-1)
        grad_returns = (dq_nope + dq_nope_sink, dk + dv, dq_pe + dq_pe_sink, dk_pe, dk_sink, dv_sink)
        none_returns = (None,) * 9
        return (*grad_returns, *none_returns)


# ============================================================================
# DSA KL Loss
# ============================================================================

class SparseLightningIndexerKLLossTrainFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, index_query, index_key, merge_weight, query, key,
                topk_indices, softmax_max, softmax_sum, query_rope, key_rope,
                actual_seq_qlen, actual_seq_klen, scale, loss_coeff):
        d_index_query, d_index_key, d_merge_weight, inner_loss = \
            torch.ops.custom.npu_sparse_lightning_indexer_grad_kl_loss_enhance(
                query=query, key=key, query_index=index_query, key_index=index_key, weights=merge_weight,
                sparse_indices=topk_indices,
                softmax_max=softmax_max, softmax_sum=softmax_sum,
                scale_value=scale, query_rope=query_rope, key_rope=key_rope,
                actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen,
                layout='TND', sparse_mode=3,
                deterministic=torch.are_deterministic_algorithms_enabled(), sparse_block_size=1)

        ctx.save_for_backward(d_index_query / query.size(0) * loss_coeff,
            d_index_key / query.size(0) * loss_coeff, d_merge_weight / query.size(0) * loss_coeff)

        return inner_loss.squeeze() / query.size(0) * loss_coeff

    @staticmethod
    def backward(ctx, grad_loss):
        d_index_query, d_index_key, d_merge_weight = ctx.saved_tensors
        grad_returns = (d_index_query * grad_loss, d_index_key * grad_loss, d_merge_weight * grad_loss)
        none_returns = (None,) * 11
        return (*grad_returns, *none_returns)


_DSA_DENSE_WARMUP_CHUNK_SIZE = 128


def _dsa_build_attention_mask(index_query, actual_q_len, actual_kv_len, device,
                              mem_buffer=128 * 1024 * 1024):
    """Build the causal mask used by DSA dense warm-up."""
    query_len = actual_q_len[-1].item()
    kv_len = actual_kv_len[-1].item()
    bsz = index_query.shape[0]
    if bsz > 1:
        raise AssertionError("DSA dense warm up does not support mbs > 1")
    attention_mask_dsa = torch.ones((bsz, 1, query_len, kv_len), dtype=torch.bool, device=device)
    start_indices_q = torch.cat([torch.tensor([0], device=device), actual_q_len[:-1]])
    end_indices_q = actual_q_len
    start_indices_kv = torch.cat([torch.tensor([0], device=device), actual_kv_len[:-1]])
    end_indices_kv = actual_kv_len

    for (start_q, end_q, start_kv, end_kv) in zip(start_indices_q, end_indices_q,
                                                  start_indices_kv, end_indices_kv):
        s_q, e_q = start_q.item(), end_q.item()
        s_kv, e_kv = start_kv.item(), end_kv.item()

        query_size = e_q - s_q
        kv_size = e_kv - s_kv

        if query_size > 0 and kv_size > 0:
            offset = kv_size - query_size
            chunk_rows = max(1, mem_buffer // query_size)
            for r_start in range(0, query_size, chunk_rows):
                r_end = min(r_start + chunk_rows, query_size)
                row_indices = torch.arange(r_start, r_end, device=device).view(-1, 1)
                col_indices = torch.arange(kv_size, device=device).view(1, -1)
                attention_mask_dsa[0, 0, s_q + r_start:s_q + r_end, s_kv:e_kv] = \
                    (col_indices > (row_indices + offset))
    return attention_mask_dsa


class _DSADenseLightningIndexer(torch.autograd.Function):
    """Compute the dense DSA indexer KL loss and its explicit gradients."""

    @staticmethod
    def forward(ctx, index_query, index_key, merge_weight, query, key, attention_mask,
                loss_coeff, scale, chunk_size, training, is_grad_enabled):
        b, s, _, _ = index_query.shape
        device = index_query.device
        dtype = torch.float32
        total_loss = torch.tensor(0.0, dtype=dtype, device=device)
        if not is_grad_enabled:
            return total_loss

        k_idx_all = index_key.to(dtype)
        k_t = key.permute(0, 2, 3, 1).float()

        grad_index_query = torch.zeros_like(index_query, dtype=dtype)
        grad_index_key = torch.zeros_like(index_key, dtype=dtype)
        grad_merge_weight = torch.zeros_like(merge_weight, dtype=dtype)

        with torch.no_grad():
            for i in range(0, s, chunk_size):
                end_i = min(i + chunk_size, s)

                q_idx_chunk = index_query[:, i:end_i].to(dtype)
                S_idx_chunk = torch.einsum("bcnd,btnd->bnct", q_idx_chunk, k_idx_all)
                S_idx_chunk_relu = F.relu(S_idx_chunk)

                W_chunk = merge_weight[:, i:end_i].permute(0, 2, 1).unsqueeze(-1)
                I_chunk = (W_chunk * S_idx_chunk_relu).sum(dim=1, keepdim=True)

                mask_chunk = attention_mask[:, :, i:end_i, :]
                I_use_chunk = I_chunk.masked_fill(mask_chunk, -1e9)

                if training and is_grad_enabled:
                    q_chunk = query[:, i:end_i].float()
                    q_t = q_chunk.permute(0, 2, 1, 3)

                    if scale is None:
                        scale = q_chunk.size(-1) ** -0.5

                    scores = torch.matmul(q_t * scale, k_t)
                    scores.masked_fill_(mask_chunk, -1e9)
                    torch.softmax(scores, dim=-1, dtype=torch.float32, out=scores)
                    P_chunk = scores.sum(dim=1, keepdim=True)
                    P_use_chunk = P_chunk / P_chunk.sum(dim=-1, keepdim=True)

                    y_softmax = F.log_softmax(I_use_chunk, dim=-1, dtype=torch.float32)
                    chunk_loss = F.kl_div(y_softmax, P_use_chunk, reduction='batchmean')
                    total_loss += chunk_loss * loss_coeff

                    grad_scalar = loss_coeff / s
                    dI_chunk = (torch.exp(y_softmax) - P_use_chunk) * grad_scalar

                    mask_pos = (S_idx_chunk > 0).to(dtype)
                    grad_S_idx_chunk = W_chunk * (dI_chunk * mask_pos)

                    dW_contrib = (dI_chunk.squeeze(1) * S_idx_chunk_relu).sum(dim=-1)
                    grad_merge_weight[:, i:end_i] = dW_contrib.permute(0, 2, 1)

                    grad_index_query_chunk = torch.einsum("bnct,btnd->bcnd", grad_S_idx_chunk, k_idx_all)
                    grad_index_query[:, i:end_i] = grad_index_query_chunk

                    grad_index_key_chunk = torch.einsum("bcnd,bnct->btnd", q_idx_chunk, grad_S_idx_chunk)
                    grad_index_key += grad_index_key_chunk.sum(dim=2, keepdim=True)

        if is_grad_enabled:
            ctx.save_for_backward(grad_index_query, grad_index_key, grad_merge_weight)

        return total_loss / s

    @staticmethod
    def backward(ctx, grad_loss):
        grad_index_query, grad_index_key, grad_merge_weight = ctx.saved_tensors
        grad_returns = (grad_index_query * grad_loss,
                        grad_index_key * grad_loss,
                        grad_merge_weight * grad_loss)
        none_returns = (None,) * 8
        return (*grad_returns, *none_returns)


def _attach_dense_indexer_loss(
    attn_output,
    query_li,
    key_li,
    index_query_li,
    index_key_li,
    merge_weight_li,
    actual_q_len,
    actual_kv_len,
    qk_nope_head_dim,
    qk_rope_head_dim,
    dsa_loss_coeff,
):
    """Attach the dense DSA indexer KL loss to the attention output."""
    device = query_li.device
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    scale = qk_head_dim**-0.5

    npu_actual_q_len = torch.tensor(actual_q_len, dtype=torch.int32, device=device)
    npu_actual_kv_len = torch.tensor(actual_kv_len, dtype=torch.int32, device=device)

    attention_mask_no_sink = _dsa_build_attention_mask(
        index_query_li, npu_actual_q_len, npu_actual_kv_len, device)

    is_grad_enabled = torch.is_grad_enabled()
    loss = _DSADenseLightningIndexer.apply(
        index_query_li.float(),
        index_key_li.float(),
        merge_weight_li,
        query_li,
        key_li,
        attention_mask_no_sink,
        dsa_loss_coeff,
        scale,
        _DSA_DENSE_WARMUP_CHUNK_SIZE,
        is_grad_enabled,
        is_grad_enabled,
    )
    return AuxLossAutoScaler.apply(attn_output, loss)


# ============================================================================
# Attention backend registry
# ============================================================================

ATTENTION_FUNCTIONS = {}


class AttentionBackendRegistry:
    """Registry for attention backend functions."""

    _backends = ATTENTION_FUNCTIONS

    @classmethod
    def register(cls, name):
        """Decorator that registers a backend function under *name*."""
        def decorator(fn):
            cls._backends[name] = fn
            return fn
        return decorator

    @classmethod
    def get(cls, name):
        return cls._backends[name]

    @classmethod
    def list_backends(cls):
        return list(cls._backends.keys())


def _native_attention_layout(module):
    if module.attention_type == "dsa" and module.dsa_dense_warm_up:
        return "TND"

    input_layout = "SBH" if module.use_fused_sink_fa or module.apply_FA_rescale else "BSND"
    config = module.config
    if config.mask_type == "causal" and config.mask_compress and config.reset_attention_mask:
        return "TND"
    if config.mask_type == "full" and (config.reset_attention_mask or config.mask_compress):
        return "TND"
    return input_layout


def _native_sequence_lengths(module, input_layout, batch_size, sequence_length, actual_q_len, actual_kv_len):
    if module.attention_type == "dsa" and module.dsa_dense_warm_up:
        return actual_q_len, actual_kv_len
    if input_layout != "TND":
        return actual_q_len, actual_kv_len
    if actual_q_len is None:
        lengths = [sequence_length] * batch_size
        return lengths, lengths
    if module.param_sink_number > 0 and not (module.use_fused_sink_fa or module.apply_FA_rescale):
        actual_q_len = [length + module.param_sink_number for length in actual_q_len]
        actual_kv_len = [length + module.param_sink_number for length in actual_kv_len]
    return actual_q_len, actual_kv_len


def _convert_bsnd_qkv_for_backend(query, key, value, input_layout):
    """Convert BSND QKV to the layout required by a native attention kernel."""
    if query.ndim != 4:
        return query, key, value
    if input_layout == "BSND":
        return query, key, value
    if input_layout == "TND":
        return [rearrange(tensor, "b s n d -> (b s) n d") for tensor in (query, key, value)]
    if input_layout == "SBH":
        return [rearrange(tensor, "b s n d -> s b (n d)") for tensor in (query, key, value)]
    raise ValueError(f"Unsupported attention backend input layout: {input_layout}")


def _restore_backend_output_to_bsnd(output, input_layout, batch_size, sequence_length, num_heads):
    if input_layout == "TND":
        return rearrange(output, "(b s) n d -> b s n d", b=batch_size, s=sequence_length)
    if input_layout == "SBH":
        return rearrange(output, "s b (n d) -> b s n d", n=num_heads)
    if input_layout == "BSND":
        return output if output.ndim == 4 else output.view(batch_size, sequence_length, num_heads, -1)
    raise ValueError(f"Unsupported attention backend output layout: {input_layout}")


def _extend_param_sink_attention_mask(attention_mask, param_sink_number, q_len, kv_len):
    """Extend a prepared additive dense mask for the parameter-sink Q/K prefix."""
    if attention_mask is None or attention_mask.dtype == torch.bool or param_sink_number == 0:
        return attention_mask
    if attention_mask.shape[-2:] == (q_len, kv_len):
        return attention_mask

    sequence_length = attention_mask.shape[-1]
    expected_length = sequence_length + param_sink_number
    if attention_mask.shape[-2] != sequence_length or (q_len, kv_len) != (expected_length, expected_length):
        raise ValueError("Prepared attention mask is incompatible with the parameter-sink Q/K prefix.")

    prefix_shape = (*attention_mask.shape[:-2], param_sink_number, sequence_length)
    sink_queries = torch.zeros(prefix_shape, dtype=attention_mask.dtype, device=attention_mask.device)
    sink_keys = torch.full(
        (*attention_mask.shape[:-2], q_len, param_sink_number),
        torch.finfo(attention_mask.dtype).min,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    ).triu(diagonal=1)

    attention_mask = torch.cat([sink_queries, attention_mask], dim=-2)
    return torch.cat([sink_keys, attention_mask], dim=-1)


@AttentionBackendRegistry.register("npu_fused_sink_fa")
def npu_fused_sink_fa_forward(module, query, key, value, attention_mask, **kwargs):
    """NPU fused_sink_fa backend (npu_flash_attention_score_enhance)."""
    input_layout = _native_attention_layout(module)
    actual_q_len, actual_kv_len = _native_sequence_lengths(
        module,
        input_layout,
        kwargs["bsz"],
        kwargs["seq_length"],
        kwargs["actual_q_len"],
        kwargs["actual_kv_len"],
    )
    query, key, value = _convert_bsnd_qkv_for_backend(query, key, value, input_layout)
    output, softmax_max, softmax_sum = torch.ops.custom.npu_flash_attention_score_enhance(
        query, key, value, kwargs["n_head"],
        pse=None, padding_mask=None, atten_mask=attention_mask,
        scale=kwargs["scale"], keep_prob=1 - module.attention_dropout.p,
        input_layout=input_layout,
        actual_seq_qlen=actual_q_len,
        actual_seq_kvlen=actual_kv_len,
        pre_tokens=module.pre_tockens, next_tokens=module.next_tockens,
        inner_precise=0, sparse_mode=module.sparse_mode,
        prefix=[], sink_num=module.sink_num,
    )[:3]
    return _restore_backend_output_to_bsnd(
        output,
        input_layout,
        kwargs["bsz"],
        kwargs["seq_length"],
        kwargs["n_head"],
    )


@AttentionBackendRegistry.register("npu_fusion_attention")
def npu_fusion_attention_forward(module, query, key, value, attention_mask, **kwargs):
    """NPU npu_fusion_attention backend."""
    input_layout = _native_attention_layout(module)
    actual_q_len, actual_kv_len = _native_sequence_lengths(
        module,
        input_layout,
        kwargs["bsz"],
        kwargs["seq_length"],
        kwargs["actual_q_len"],
        kwargs["actual_kv_len"],
    )
    query, key, value = _convert_bsnd_qkv_for_backend(query, key, value, input_layout)
    output, softmax_max, softmax_sum = torch_npu.npu_fusion_attention(
        query, key, value, kwargs["n_head"], input_layout,
        pse=None, padding_mask=None, atten_mask=attention_mask,
        scale=kwargs["scale"],
        pre_tockens=module.pre_tockens, next_tockens=module.next_tockens,
        keep_prob=1 - module.attention_dropout.p, inner_precise=0,
        sparse_mode=module.sparse_mode,
        actual_seq_qlen=actual_q_len,
        actual_seq_kvlen=actual_kv_len,
    )[:3]
    has_prepended_sink = module.param_sink_number > 0 and not (
        module.use_fused_sink_fa
        or module.apply_FA_rescale
        or (module.attention_type == "dsa" and module.dsa_dense_warm_up)
    )
    output = _restore_backend_output_to_bsnd(
        output,
        input_layout,
        kwargs["bsz"],
        kwargs["seq_length"],
        kwargs["n_head"],
    )
    if has_prepended_sink:
        output = output[:, module.param_sink_number:]
    return output


@AttentionBackendRegistry.register("cpu_sdpa")
def cpu_sdpa_forward(module, query, key, value, attention_mask, **kwargs):
    """CPU fallback backend using PyTorch scaled_dot_product_attention."""
    n_head = kwargs["n_head"]
    head_dim = kwargs["head_dim"]
    bsz = kwargs["bsz"]
    scale = kwargs["scale"]
    input_layout = kwargs.get("input_layout")
    if query.ndim == 4:
        input_layout = "BSND"
    elif input_layout is None:
        input_layout = _native_attention_layout(module)
    is_bsnd = input_layout == "BSND"

    if is_bsnd:
        # [B, S, N, D] -> SDPA's [B, N, S, D].
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
    elif input_layout == "SBH":
        q = rearrange(query, "s b (h d) -> b h s d", h=n_head, d=head_dim)
        kv_dim = key.shape[-1]
        if kv_dim % head_dim == 0:
            kv_heads = kv_dim // head_dim
            k_head_dim = head_dim
        else:
            kv_heads = module.num_key_value_heads
            k_head_dim = kv_dim // kv_heads
        k = rearrange(key, "s b (h d) -> b h s d", h=kv_heads, d=k_head_dim)
        v_dim = value.shape[-1]
        v_head_dim_actual = v_dim // kv_heads
        v = rearrange(value, "s b (h d) -> b h s d", h=kv_heads, d=v_head_dim_actual)
    elif input_layout == "TND":
        seq_len = query.shape[0] // bsz
        q = rearrange(query, "(b s) h d -> b h s d", b=bsz, s=seq_len)
        k = rearrange(key, "(b s) h d -> b h s d", b=bsz, s=seq_len)
        v = rearrange(value, "(b s) h d -> b h s d", b=bsz, s=seq_len)
    else:
        raise ValueError(f"Unsupported attention backend input layout: {input_layout}")

    # q/k/v are now BHSD; k/v may still have fewer heads for GQA.
    # GQA: expand k/v heads to match q heads
    n_rep = q.shape[1] // k.shape[1]
    if n_rep > 1:
        k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(
            q.shape[0], q.shape[1], k.shape[2], k.shape[3])
        v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(
            q.shape[0], q.shape[1], v.shape[2], v.shape[3])

    if attention_mask is not None:
        # masks broadcast over heads: [Q, K], [B, Q, K], or [B, 1, Q, K].
        attention_mask = attention_mask[..., : q.shape[-2], : k.shape[-2]]
        if attention_mask.dtype == torch.bool:
            # uses True=blocked; SDPA boolean masks use the opposite convention.
            attention_mask = torch.zeros_like(attention_mask, dtype=q.dtype).masked_fill_(
                attention_mask, float("-inf")
            )
        else:
            attention_mask = attention_mask.to(dtype=q.dtype)

    output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        scale=scale,
        is_causal=attention_mask is None and module.sparse_mode in (2, 4),
    )

    output = output.transpose(1, 2)

    if module.param_sink_number > 0 and not (
        module.use_fused_sink_fa
        or module.apply_FA_rescale
        or (module.attention_type == "dsa" and module.dsa_dense_warm_up)
    ):
        output = output[:, module.param_sink_number:]

    return output


@AttentionBackendRegistry.register("npu_fa_rescale")
def npu_fa_rescale_forward(module, query, key, value, attention_mask, **kwargs):
    """NPU FArescale backend (FArescale autograd function)."""
    native_layout = _native_attention_layout(module)
    actual_q_len, actual_kv_len = _native_sequence_lengths(
        module,
        native_layout,
        kwargs["bsz"],
        kwargs["seq_length"],
        kwargs["actual_q_len"],
        kwargs["actual_kv_len"],
    )
    param_sink_key = kwargs["param_sink_key"]
    param_sink_value = kwargs["param_sink_value"]
    query, key, value = _convert_bsnd_qkv_for_backend(query, key, value, "TND")
    param_sink_key, param_sink_value = [
        rearrange(x, "b s n d -> s b (n d)") for x in [param_sink_key, param_sink_value]
    ]
    rescaled_output, softmax_max = FArescale.apply(
        query, key, value, param_sink_key, param_sink_value,
        attention_mask, kwargs["bsz"], kwargs["seq_length"],
        kwargs["n_head"], kwargs["scale"],
        module.pre_tockens, module.next_tockens, 1 - module.attention_dropout.p,
        module.sparse_mode, actual_q_len, actual_kv_len,
    )
    return rearrange(rescaled_output, "s b n d -> b s n d")


def dsa_lightning_indexer_forward(
    module,
    index_query,
    index_key,
    merge_weight,
    actual_q_len,
    actual_kv_len,
):
    index_query_tnd = rearrange(index_query, "b s n d -> (b s) n d")
    index_key_tnd = rearrange(index_key, "b s n d -> (b s) n d")
    merge_weight_tnd = rearrange(merge_weight, "b s n -> (b s) n")
    topk_indices, _ = torch.ops.custom.npu_lightning_indexer_enhance(
        index_query_tnd,
        index_key_tnd,
        merge_weight_tnd,
        actual_seq_lengths_query=actual_q_len,
        actual_seq_lengths_key=actual_kv_len,
        block_table=None,
        layout_query="TND",
        layout_key="TND",
        sparse_count=module.index_topk,
        sparse_mode=3,
        return_value=False,
    )
    return topk_indices, index_query_tnd, index_key_tnd, merge_weight_tnd


@AttentionBackendRegistry.register("dsa_sparse_attention")
def dsa_sparse_attention_forward(module, query, key, value, attention_mask, **kwargs):
    del value, attention_mask
    query_tnd = rearrange(query, "b s n d -> (b s) n d")
    key_tnd = rearrange(key, "b s n d -> (b s) n d")
    q_pe = kwargs["q_pe"]
    k_pe = kwargs["k_pe"]
    q_pe_tnd = rearrange(q_pe, "b s n d -> (b s) n d")
    k_pe_tnd = rearrange(k_pe, "b s n d -> (b s) n d")
    topk_indices = kwargs["topk_indices"]
    actual_q_len = kwargs["actual_q_len"]
    actual_kv_len = kwargs["actual_kv_len"]
    scale = kwargs["scale"]

    param_sink_key = kwargs["param_sink_key"]
    param_sink_value = kwargs["param_sink_value"]
    if param_sink_key is None or param_sink_key.nelement() == 0:
        attn_output, softmax_max, softmax_sum = torch.ops.custom.npu_sparse_flash_attention_enhance(
            query_tnd,
            key_tnd,
            key_tnd,
            topk_indices,
            scale,
            block_table=None,
            actual_seq_lengths_query=actual_q_len,
            actual_seq_lengths_kv=actual_kv_len,
            query_rope=q_pe_tnd,
            key_rope=k_pe_tnd,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="TND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=True,
        )
        if q_pe.size(-1) > 0:
            attn_output = F.pad(attn_output, [0, q_pe.size(-1)])
        attn_output = rearrange(attn_output, "(b s) n d -> b s n d", b=query.shape[0])
    else:
        attn_output, softmax_max, softmax_sum = SFArescale.apply(
            query,
            key,
            q_pe,
            k_pe,
            param_sink_key,
            param_sink_value,
            topk_indices,
            query.shape[0],
            query.shape[1],
            module.num_heads,
            scale,
            1 - module.attention_dropout.p,
            actual_q_len,
            actual_kv_len,
        )
    return attn_output, softmax_max, softmax_sum


def _config_dtype(config):
    params_dtype = getattr(config, "dtype", None)
    if not isinstance(params_dtype, torch.dtype):
        params_dtype = getattr(config, "torch_dtype", None)
    return params_dtype if isinstance(params_dtype, torch.dtype) else torch.bfloat16


def _layer_numbers(layers, offset=0):
    if not layers:
        return set()
    if isinstance(layers, str):
        return set(map(int, layers.split(","))) if layers.strip() else set()
    return {int(layer) + offset for layer in layers}


def _initialize_parameter(config, parameter, role="input"):
    if config.perform_initialization:
        config._standalone_init_weights(parameter, init_role=role)


class GQAAttention(nn.Module):
    """Grouped-query attention with a Transformers-style BSH interface."""

    def __init__(self, config, layer_number: int):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.attention_type = "gqa"
        self.params_dtype = _config_dtype(config)

        self.use_flash_attn = config.use_flash_attn
        self.use_fused_sink_fa = config.use_fused_sink_fa
        self.apply_FA_rescale = config.apply_FA_rescale
        if self.use_fused_sink_fa:
            self.attn_implementation = "npu_fused_sink_fa"
        elif self.apply_FA_rescale:
            self.attn_implementation = "npu_fa_rescale"
        else:
            self.attn_implementation = "npu_fusion_attention"
        self.rotary_interleaved = config.rope_interleaved
        self.use_mome = config.use_mome
        self.use_fused_mome = config.use_fused_mome
        self.kv_reuse_mapping = getattr(config, "kv_reuse_mapping", None)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

        self.sparse_mode = 0
        self.pre_tockens = 1048576
        self.next_tockens = 0
        self.sink_num = config.param_sink_number // 64
        backend_layer_number = max(1, self.layer_number)
        # Config stores SWA layer indices as zero-based, while standalone layers
        # use one-based layer_number (matching the pre-refactor implementation).
        swa_layers = _layer_numbers(config.swa_layers, offset=1)
        self.use_swa = backend_layer_number in swa_layers
        self.swa_sliding_window = None
        if self.use_swa:
            swa_sliding_window = config.sliding_window_list
            swa_windows = (
                list(map(int, swa_sliding_window.split(",")))
                if isinstance(swa_sliding_window, str)
                else list(map(int, swa_sliding_window))
            )
            if len(swa_windows) == 1:
                self.swa_sliding_window = swa_windows[0]
            else:
                swa_layer_list = sorted(swa_layers)
                self.swa_sliding_window = swa_windows[swa_layer_list.index(backend_layer_number)]
        if config.mask_type == "causal" and config.mask_compress:
            self.sparse_mode = 2
            if self.use_swa and config.swa_attention_sink == 0:
                self.sparse_mode = 4
                self.pre_tockens = self.swa_sliding_window
        elif config.mask_type == "general":
            self.sparse_mode = 1

        self.param_sink_number = config.param_sink_number
        self.param_sink_with_value = config.param_sink_with_value
        self.param_sink_scalar = config.param_sink_scalar
        self.param_sink_of_head_num = config.param_sink_of_head_num

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.qk_head_dim = config.head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.augs_gqa = False
        if (
            self.qk_rope_head_dim is not None
            and self.qk_nope_head_dim is not None
            and self.qk_rope_head_dim + self.qk_nope_head_dim != self.v_head_dim
        ):
            self.augs_gqa = True
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        use_q_norm = bool(getattr(config, "qk_layernorm", False))
        use_k_norm = use_q_norm or bool(getattr(config, "k_layernorm", False)) or bool(
            getattr(config, "use_k_norm", False)
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qkv_split_sizes = (
            self.num_key_value_groups * self.qk_head_dim,
            self.qk_head_dim,
            self.v_head_dim if self.augs_gqa else self.qk_head_dim,
        )
        self.qkv_group_width = sum(self.qkv_split_sizes)
        qkv_projection_size = self.num_key_value_heads * self.qkv_group_width
        norm_factor = math.sqrt(self.qk_head_dim)
        rope_scaling = config.rope_scaling or {}
        if rope_scaling.get("rope_type") is not None and rope_scaling.get("mscale_all_dim", 0):
            mscale = yarn_get_mscale(rope_scaling.get("factor"), rope_scaling["mscale_all_dim"])
            norm_factor /= mscale * mscale
        self.scaling = 1.0 / norm_factor

        self.attn_groupnorm = config.attn_groupnorm
        self.attn_elementwise_gate = config.attn_elementwise_gate
        self.gqa_qkv_clone_for_mem = config.gqa_qkv_clone_for_mem
        if self.use_mome:
            mome_window = config.router_sliding_window
            conv_hidden = self.v_head_dim * self.num_heads
            self.o_conv = nn.Conv1d(
                conv_hidden,
                conv_hidden,
                mome_window,
                groups=conv_hidden,
                bias=False,
            )
        if self.attn_groupnorm:
            self.groupnorm = nn.LayerNorm(self.qk_head_dim, eps=config.rms_norm_eps)
        if self.attn_elementwise_gate:
            self.attention_gate = LinearWithMatmul(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)

        norm_kwargs = {"eps": config.rms_norm_eps, "use_fused_rmsnorm": config.use_fused_rmsnorm}
        self.q_layernorm = (
            FusedRMSNorm(hidden_size=self.qk_head_dim, **norm_kwargs) if use_q_norm else None
        )
        self.k_layernorm = (
            FusedRMSNorm(hidden_size=self.qk_head_dim, **norm_kwargs) if use_k_norm else None
        )

        bias = config.attention_bias or config.add_qkv_bias
        self.linear_qkv = LinearWithMatmul(config.hidden_size, qkv_projection_size, bias=bias, skip_bias_add=False)
        _initialize_parameter(config, self.linear_qkv.weight)
        proj_input_size = self.v_head_dim * self.num_heads if self.augs_gqa else self.num_heads * self.qk_head_dim
        self.linear_proj = LinearWithMatmul(proj_input_size, config.hidden_size, bias=config.attention_bias)
        _initialize_parameter(config, self.linear_proj.weight, "output")
        value_dim = self.v_head_dim if self.augs_gqa else self.qk_head_dim
        self._init_param_sink_parameters(config, value_dim)

    def _init_param_sink_parameters(self, config, value_dim):
        if self.param_sink_number <= 0:
            return
        self.register_buffer(
            "param_sink_query",
            torch.zeros(
                self.param_sink_number,
                self.num_heads,
                self.qk_head_dim,
                dtype=self.params_dtype,
            ),
            persistent=False
        )
        if self.param_sink_of_head_num:
            self.param_sink_num_heads = self.num_heads
        else:
            self.param_sink_num_heads = self.num_key_value_heads
        if self.param_sink_scalar:
            self.register_buffer(
                "param_sink_key_zero_pad",
                torch.zeros(
                    self.param_sink_number,
                    self.param_sink_num_heads,
                    self.param_sink_scalar - 1,
                    dtype=self.params_dtype,
                ),
                persistent=False
            )
            self.param_sink_key = nn.Parameter(
                torch.empty(self.param_sink_number, self.param_sink_num_heads, dtype=self.params_dtype)
            )
        else:
            self.param_sink_key = nn.Parameter(
                torch.empty(
                    self.param_sink_number,
                    self.param_sink_num_heads,
                    self.qk_head_dim,
                    dtype=self.params_dtype,
                )
            )
        _initialize_parameter(config, self.param_sink_key)
        if self.param_sink_scalar:
            sink_value_shape = (self.param_sink_number, self.param_sink_num_heads, self.v_head_dim)
            if self.param_sink_with_value:
                self.param_sink_value = nn.Parameter(torch.empty(sink_value_shape, dtype=self.params_dtype))
                _initialize_parameter(config, self.param_sink_value)
            else:
                self.register_buffer(
                    "param_sink_value",
                    torch.zeros(sink_value_shape, dtype=self.params_dtype),
                    persistent=False,
                )

    def _prepare_param_sink(self, batch_size, query_states, key_states, value_states):
        if self.param_sink_scalar:
            param_sink_key = torch.cat(
                [self.param_sink_key_zero_pad, self.param_sink_key.unsqueeze(-1)],
                dim=-1,
            )
            param_sink_key = param_sink_key.unsqueeze(0).expand(batch_size, -1, -1, -1)
            value_dim = self.v_head_dim if self.augs_gqa else self.qk_head_dim
            param_sink_value = F.pad(self.param_sink_value, [0, self.param_sink_scalar - value_dim])
            param_sink_value = param_sink_value.unsqueeze(0).expand(batch_size, -1, -1, -1)
        else:
            param_sink_key = self.param_sink_key.unsqueeze(0).expand(batch_size, -1, -1, -1)
            if self.k_layernorm is not None:
                param_sink_key = self.k_layernorm(param_sink_key)
            param_sink_value = (
                F.pad(self.param_sink_value, [0, self.qk_head_dim - self.v_head_dim])
                if self.augs_gqa
                else self.param_sink_value
            )
            param_sink_value = param_sink_value.unsqueeze(0).expand(batch_size, -1, -1, -1)
        if self.param_sink_of_head_num:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, 2)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, 2)
        if self.param_sink_scalar:
            padding = self.param_sink_scalar - self.qk_head_dim
            key_states = F.pad(key_states, [0, padding], value=0)
            value_states = F.pad(value_states, [0, padding], value=0)
        if not self.apply_FA_rescale:
            key_states = torch.cat([param_sink_key, key_states], dim=1)
            value_states = torch.cat([param_sink_value, value_states], dim=1)
        if not (self.use_fused_sink_fa or self.apply_FA_rescale):
            param_sink_query = self.param_sink_query.unsqueeze(0).expand(batch_size, -1, -1, -1)
            query_states = torch.cat([param_sink_query, query_states], dim=1)
            if self.param_sink_scalar:
                query_states = F.pad(query_states, [0, padding], value=1)
        return query_states, key_states, value_states, param_sink_key, param_sink_value

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        actual_seq_len=None,
        kv_reuse_states=None,
        output_attentions=False,
        return_bias=False,
        mome_mask=None,
    ):
        if past_key_values is not None or cache_position is not None:
            raise NotImplementedError("GQA cache support has not been implemented yet.")
        if output_attentions:
            raise NotImplementedError("GQA does not expose attention weights for its current backends.")

        input_shape = hidden_states.shape[:-1]
        batch_size = input_shape[0]
        query_shape = (*input_shape, self.num_heads, self.qk_head_dim)

        qkv_states, _ = self.linear_qkv(hidden_states)
        qkv_states = qkv_states.view(*input_shape, self.num_key_value_heads, self.qkv_group_width)
        query_states, key_states, value_states = torch.split(qkv_states, self.qkv_split_sizes, dim=-1)
        query_states = query_states.reshape(query_shape)
        if self.gqa_qkv_clone_for_mem:
            query_states, key_states, value_states = (
                query_states.clone(),
                key_states.clone(),
                value_states.clone(),
            )
        if self.q_layernorm is not None:
            query_states = self.q_layernorm(query_states)
        if self.k_layernorm is not None:
            key_states = self.k_layernorm(key_states)
        if self.use_flash_attn and self.augs_gqa and not self.use_fused_sink_fa:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states = apply_rotary_pos_emb(query_states, cos, sin, self.rotary_interleaved)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, self.rotary_interleaved)

        param_sink_key, param_sink_value = None, None
        if self.param_sink_number > 0:
            query_states, key_states, value_states, param_sink_key, param_sink_value = self._prepare_param_sink(
                batch_size,
                query_states,
                key_states,
                value_states,
            )
            attention_mask = _extend_param_sink_attention_mask(
                attention_mask,
                self.param_sink_number,
                query_states.shape[1],
                key_states.shape[1],
            )
        if self.attn_elementwise_gate:
            gate_score = self.attention_gate(hidden_states)
        if isinstance(actual_seq_len, torch.Tensor):
            actual_seq_len = [int(item) for item in actual_seq_len.tolist()]
        if self.kv_reuse_mapping:
            kv_reuse_layer = max(1, self.layer_number)
            if kv_reuse_states is None:
                raise RuntimeError("kv_reuse_states must be provided when KV reuse is enabled")
            kv_reuse_states[kv_reuse_layer] = key_states, value_states
            source_layer = self.kv_reuse_mapping.get(
                kv_reuse_layer,
                self.kv_reuse_mapping.get(str(kv_reuse_layer)),
            )
            if source_layer is not None:
                key_states, value_states = kv_reuse_states[int(source_layer)]

        backend_sequence_length = query_states.shape[1]
        attention_interface = ATTENTION_FUNCTIONS["cpu_sdpa"]
        if HAS_NPU and query_states.device.type != "cpu":
            attention_interface = ATTENTION_FUNCTIONS[self.attn_implementation]
        attn_output = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            n_head=query_states.shape[2],
            head_dim=self.qk_head_dim,
            bsz=batch_size,
            scale=self.scaling,
            seq_length=backend_sequence_length,
            actual_q_len=actual_seq_len,
            actual_kv_len=actual_seq_len,
            param_sink_key=param_sink_key,
            param_sink_value=param_sink_value,
        )
        if self.use_flash_attn and self.augs_gqa and not self.use_fused_sink_fa:
            attn_output = attn_output[..., : self.v_head_dim]
        elif self.use_flash_attn and self.param_sink_scalar:
            attn_output = attn_output[..., : self.qk_head_dim]
        attn_output = attn_output.flatten(2)
        if self.use_mome:
            attn_output = _apply_mome(
                hidden_states=attn_output,
                mome_mask=mome_mask,
                conv=self.o_conv,
                use_fused=self.use_fused_mome,
            )
        if self.attn_groupnorm:
            attn_output = attn_output.reshape(-1, self.num_heads, self.qk_head_dim)
            attn_output = self.groupnorm(attn_output).reshape(*input_shape, -1)
        if self.attn_elementwise_gate:
            attn_output = attn_output * F.sigmoid(gate_score)
        output, bias = self.linear_proj(attn_output)
        return (output, bias) if return_bias else (output, None)


class MLAAttention(nn.Module):
    """Multi-head latent attention with a Transformers-style BSH interface."""

    def __init__(self, config, layer_number: int):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.attention_type = "mla"
        self.params_dtype = _config_dtype(config)

        self.use_flash_attn = config.use_flash_attn
        self.use_fused_sink_fa = config.use_fused_sink_fa
        self.apply_FA_rescale = config.apply_FA_rescale
        if self.use_fused_sink_fa:
            self.attn_implementation = "npu_fused_sink_fa"
        elif self.apply_FA_rescale:
            self.attn_implementation = "npu_fa_rescale"
        else:
            self.attn_implementation = "npu_fusion_attention"
        self.rotary_interleaved = config.rope_interleaved
        self.use_mome = config.use_mome
        self.use_fused_mome = config.use_fused_mome
        self.kv_reuse_mapping = getattr(config, "kv_reuse_mapping", None)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

        self.sparse_mode = 0
        self.pre_tockens = 1048576
        self.next_tockens = 0
        self.sink_num = config.param_sink_number // 64
        backend_layer_number = max(1, self.layer_number)
        swa_layers = _layer_numbers(config.swa_layers, offset=1)
        self.use_swa = backend_layer_number in swa_layers
        self.swa_sliding_window = None
        if self.use_swa:
            swa_sliding_window = config.sliding_window_list
            swa_windows = (
                list(map(int, swa_sliding_window.split(",")))
                if isinstance(swa_sliding_window, str)
                else list(map(int, swa_sliding_window))
            )
            if len(swa_windows) == 1:
                self.swa_sliding_window = swa_windows[0]
            else:
                swa_layer_list = sorted(swa_layers)
                self.swa_sliding_window = swa_windows[swa_layer_list.index(backend_layer_number)]
        if config.mask_type == "causal" and config.mask_compress:
            self.sparse_mode = 2
            if self.use_swa and config.swa_attention_sink == 0:
                self.sparse_mode = 4
                self.pre_tockens = self.swa_sliding_window
        elif config.mask_type == "general":
            self.sparse_mode = 1

        self.param_sink_number = config.param_sink_number
        self.param_sink_with_value = config.param_sink_with_value
        self.param_sink_scalar = config.param_sink_scalar
        self.param_sink_of_head_num = config.param_sink_of_head_num

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.v_head_dim = config.v_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        reference_dim = self.v_head_dim if self.v_head_dim is not None else config.head_dim
        if self.qk_rope_head_dim is None and self.qk_nope_head_dim is None:
            self.qk_nope_head_dim = reference_dim
            self.qk_rope_head_dim = 0
        elif self.qk_rope_head_dim is not None and self.qk_nope_head_dim is None:
            self.qk_nope_head_dim = reference_dim - self.qk_rope_head_dim
        elif self.qk_nope_head_dim is not None and self.qk_rope_head_dim is None:
            self.qk_rope_head_dim = reference_dim - self.qk_nope_head_dim

        self.mla_mm_split = config.mla_mm_split
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        if self.q_lora_rank is None:
            raise ValueError("MLA requires q_lora_rank; the original implementation has no no-Q-LoRA path.")
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        norm_factor = math.sqrt(self.qk_head_dim)
        rope_scaling = config.rope_scaling or {}
        if rope_scaling.get("rope_type") is not None and rope_scaling.get("mscale_all_dim", 0):
            mscale = yarn_get_mscale(rope_scaling.get("factor"), rope_scaling["mscale_all_dim"])
            norm_factor /= mscale * mscale
        self.scaling = 1.0 / norm_factor

        if self.use_mome:
            mome_window = config.router_sliding_window
            conv_hidden = self.v_head_dim * self.num_heads
            self.o_conv = nn.Conv1d(
                conv_hidden,
                conv_hidden,
                mome_window,
                groups=conv_hidden,
                bias=False,
            )

        norm_kwargs = {"eps": config.rms_norm_eps, "use_fused_rmsnorm": config.use_fused_rmsnorm}
        self.q_layernorm = FusedRMSNorm(hidden_size=self.q_lora_rank, **norm_kwargs)
        self.k_layernorm = FusedRMSNorm(hidden_size=self.kv_lora_rank, **norm_kwargs)
        bias = config.attention_bias or config.add_qkv_bias
        if self.mla_mm_split:
            self.linear_qk_nope = LinearWithMatmul(
                self.q_lora_rank,
                self.num_heads * self.qk_nope_head_dim,
                bias=bias,
            )
            self.linear_qk_rope = LinearWithMatmul(
                self.q_lora_rank,
                self.num_heads * self.qk_rope_head_dim,
                bias=bias,
            )
            _initialize_parameter(config, self.linear_qk_nope.weight)
            _initialize_parameter(config, self.linear_qk_rope.weight)
        else:
            self.linear_qb = LinearWithMatmul(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=bias,
            )
            _initialize_parameter(config, self.linear_qb.weight)
        self.linear_qkv = LinearWithFusedOps(
            config.hidden_size,
            self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
            bias=bias,
            skip_bias_add=False
        )
        _initialize_parameter(config, self.linear_qkv.weight)
        if self.use_mome:
            self.qa_conv = nn.Conv1d(
                self.q_lora_rank,
                self.q_lora_rank,
                mome_window,
                groups=self.q_lora_rank,
                bias=False,
            )
            self.compresskv_conv = nn.Conv1d(
                self.kv_lora_rank,
                self.kv_lora_rank,
                mome_window,
                groups=self.kv_lora_rank,
                bias=False,
            )
        if self.mla_mm_split:
            self.linear_kv_nope = LinearWithMatmul(
                self.kv_lora_rank,
                self.num_heads * self.qk_nope_head_dim,
                bias=bias,
            )
            self.linear_v = LinearWithMatmul(
                self.kv_lora_rank,
                self.num_heads * self.v_head_dim,
                bias=bias,
            )
            _initialize_parameter(config, self.linear_kv_nope.weight)
            _initialize_parameter(config, self.linear_v.weight)
        else:
            kvb_head_dim = self.qk_head_dim - self.qk_rope_head_dim + self.v_head_dim
            self.linear_kvb = LinearWithMatmul(
                self.kv_lora_rank,
                self.num_heads * kvb_head_dim,
                bias=bias,
            )
            _initialize_parameter(config, self.linear_kvb.weight)
        self.linear_proj = LinearWithMatmul(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        _initialize_parameter(config, self.linear_proj.weight, "output")
        self._init_param_sink_parameters(config)

    def _init_param_sink_parameters(self, config):
        if self.param_sink_number <= 0:
            return
        self.register_buffer(
            "param_sink_query",
            torch.zeros(
                self.param_sink_number,
                self.num_heads,
                self.qk_head_dim,
                dtype=self.params_dtype,
            ),
            persistent=False
        )
        if self.param_sink_of_head_num:
            self.param_sink_num_heads = self.num_heads
        else:
            self.param_sink_num_heads = self.num_key_value_heads
        if self.param_sink_scalar:
            self.register_buffer(
                "param_sink_key_zero_pad",
                torch.zeros(
                    self.param_sink_number,
                    self.param_sink_num_heads,
                    self.param_sink_scalar - 1,
                    dtype=self.params_dtype,
                ),
                persistent=False
            )
            self.param_sink_key = nn.Parameter(
                torch.empty(self.param_sink_number, self.param_sink_num_heads, dtype=self.params_dtype)
            )
            _initialize_parameter(config, self.param_sink_key)
        else:
            self.param_sink_k_pe = nn.Parameter(
                torch.empty(self.param_sink_number, self.qk_rope_head_dim, dtype=self.params_dtype)
            )
            self.param_sink_compressed_kv = nn.Parameter(
                torch.empty(self.param_sink_number, self.kv_lora_rank, dtype=self.params_dtype)
            )
            _initialize_parameter(config, self.param_sink_k_pe)
            _initialize_parameter(config, self.param_sink_compressed_kv)
        sink_value_shape = (self.param_sink_number, self.param_sink_num_heads, self.v_head_dim)
        if self.param_sink_with_value:
            self.param_sink_value = nn.Parameter(torch.empty(sink_value_shape, dtype=self.params_dtype))
            _initialize_parameter(config, self.param_sink_value)
        else:
            self.register_buffer("param_sink_value", torch.zeros(sink_value_shape, dtype=self.params_dtype), persistent=False)

    def _prepare_param_sink(self, batch_size, query_states, key_states, value_states):
        if self.param_sink_scalar:
            param_sink_key = torch.cat(
                [self.param_sink_key_zero_pad, self.param_sink_key.unsqueeze(-1)],
                dim=-1,
            )
            param_sink_key = param_sink_key.unsqueeze(0).expand(batch_size, -1, -1, -1)
            param_sink_value = F.pad(self.param_sink_value, [0, self.param_sink_scalar - self.v_head_dim])
            param_sink_value = param_sink_value.unsqueeze(0).expand(batch_size, -1, -1, -1)
        else:
            param_sink_k_pe = self.param_sink_k_pe.unsqueeze(0).unsqueeze(2)
            param_sink_k_pe = param_sink_k_pe.expand(batch_size, -1, self.num_heads, -1)
            compressed_kv = self.param_sink_compressed_kv.unsqueeze(0).expand(batch_size, -1, -1)
            compressed_kv = self.k_layernorm(compressed_kv)
            if self.mla_mm_split:
                param_sink_k_nope, _ = self.linear_kv_nope(compressed_kv)
                param_sink_value, _ = self.linear_v(compressed_kv)
                param_sink_k_nope = param_sink_k_nope.view(
                    batch_size,
                    self.param_sink_number,
                    self.num_heads,
                    self.qk_nope_head_dim,
                )
                param_sink_value = param_sink_value.view(
                    batch_size,
                    self.param_sink_number,
                    self.num_heads,
                    self.v_head_dim,
                )
            else:
                param_sink_kv, _ = self.linear_kvb(compressed_kv.contiguous())
                param_sink_kv = param_sink_kv.view(
                    batch_size,
                    self.param_sink_number,
                    self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                )
                param_sink_k_nope, param_sink_value = torch.split(
                    param_sink_kv,
                    [self.qk_nope_head_dim, self.v_head_dim],
                    dim=-1,
                )
            if self.use_flash_attn and not self.use_fused_sink_fa:
                param_sink_value = F.pad(param_sink_value, [0, self.qk_head_dim - self.v_head_dim])
            param_sink_key = torch.cat([param_sink_k_nope, param_sink_k_pe], dim=-1)
        if self.param_sink_of_head_num:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, 2)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, 2)
        if self.param_sink_scalar:
            padding = self.param_sink_scalar - self.qk_head_dim
            key_states = F.pad(key_states, [0, padding], value=0)
            value_states = F.pad(value_states, [0, padding], value=0)
        if not self.apply_FA_rescale:
            key_states = torch.cat([param_sink_key, key_states], dim=1)
            value_states = torch.cat([param_sink_value, value_states], dim=1)
        if not (self.use_fused_sink_fa or self.apply_FA_rescale):
            param_sink_query = self.param_sink_query.unsqueeze(0).expand(batch_size, -1, -1, -1)
            query_states = torch.cat([param_sink_query, query_states], dim=1)
            if self.param_sink_scalar:
                query_states = F.pad(query_states, [0, padding], value=1)
        return query_states, key_states, value_states, param_sink_key, param_sink_value

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        actual_seq_len=None,
        kv_reuse_states=None,
        output_attentions=False,
        return_bias=False,
        mome_mask=None,
    ):
        if past_key_values is not None or cache_position is not None:
            raise NotImplementedError("MLA cache support has not been implemented yet.")
        if output_attentions:
            raise NotImplementedError("MLA does not expose attention weights for its current backends.")

        batch_size, sequence_length = hidden_states.shape[:-1]

        latent_states, _ = self.linear_qkv(hidden_states)
        q_a, compressed_kv, k_pe = torch.split(
            latent_states,
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        if self.use_mome:
            q_a = _apply_mome(
                hidden_states=q_a,
                mome_mask=mome_mask,
                conv=self.qa_conv,
                use_fused=self.use_fused_mome,
            )
            compressed_kv = _apply_mome(
                hidden_states=compressed_kv,
                mome_mask=mome_mask,
                conv=self.compresskv_conv,
                use_fused=self.use_fused_mome,
            )

        q_a = self.q_layernorm(q_a)
        head_shape = (batch_size, sequence_length, self.num_heads)
        if self.mla_mm_split:
            q_nope, _ = self.linear_qk_nope(q_a)
            q_pe, _ = self.linear_qk_rope(q_a)
            q_nope = q_nope.view(*head_shape, self.qk_nope_head_dim)
            q_pe = q_pe.view(*head_shape, self.qk_rope_head_dim)
        else:
            query_states, _ = self.linear_qb(q_a.contiguous())
            query_states = query_states.view(*head_shape, self.qk_head_dim)
            q_nope, q_pe = torch.split(
                query_states,
                [self.qk_nope_head_dim, self.qk_rope_head_dim],
                dim=-1,
            )

        k_pe = k_pe.view(batch_size, sequence_length, 1, self.qk_rope_head_dim)
        compressed_kv = self.k_layernorm(compressed_kv)
        if self.mla_mm_split:
            k_nope, _ = self.linear_kv_nope(compressed_kv)
            value_states, _ = self.linear_v(compressed_kv)
            k_nope = k_nope.view(*head_shape, self.qk_nope_head_dim)
            value_states = value_states.view(*head_shape, self.v_head_dim)
        else:
            key_value_states, _ = self.linear_kvb(compressed_kv.contiguous())
            key_value_states = key_value_states.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            k_nope, value_states = torch.split(
                key_value_states,
                [self.qk_nope_head_dim, self.v_head_dim],
                dim=-1,
            )
        if self.use_flash_attn and not self.use_fused_sink_fa:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q_pe = apply_rotary_pos_emb(
                q_pe.transpose(0, 1),
                cos.transpose(0, 1),
                sin.transpose(0, 1),
                self.rotary_interleaved,
                use_fused_rotary_pos_emb=True,
            ).transpose(0, 1)
            k_pe = apply_rotary_pos_emb(
                k_pe.transpose(0, 1),
                cos.transpose(0, 1),
                sin.transpose(0, 1),
                self.rotary_interleaved,
                use_fused_rotary_pos_emb=True,
            ).transpose(0, 1)
        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat(
            [k_nope, k_pe.expand(*k_pe.shape[:2], self.num_heads, k_pe.shape[3])],
            dim=-1,
        )

        param_sink_key, param_sink_value = None, None
        if self.param_sink_number > 0:
            query_states, key_states, value_states, param_sink_key, param_sink_value = self._prepare_param_sink(
                batch_size,
                query_states,
                key_states,
                value_states,
            )
            attention_mask = _extend_param_sink_attention_mask(
                attention_mask,
                self.param_sink_number,
                query_states.shape[1],
                key_states.shape[1],
            )
        if isinstance(actual_seq_len, torch.Tensor):
            actual_seq_len = [int(item) for item in actual_seq_len.tolist()]
        if self.kv_reuse_mapping:
            kv_reuse_layer = max(1, self.layer_number)
            if kv_reuse_states is None:
                raise RuntimeError("kv_reuse_states must be provided when KV reuse is enabled")
            kv_reuse_states[kv_reuse_layer] = key_states, value_states
            source_layer = self.kv_reuse_mapping.get(
                kv_reuse_layer,
                self.kv_reuse_mapping.get(str(kv_reuse_layer)),
            )
            if source_layer is not None:
                key_states, value_states = kv_reuse_states[int(source_layer)]

        backend_sequence_length = query_states.shape[1]
        attention_interface = ATTENTION_FUNCTIONS["cpu_sdpa"]
        if HAS_NPU and query_states.device.type != "cpu":
            attention_interface = ATTENTION_FUNCTIONS[self.attn_implementation]
        attn_output = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            n_head=query_states.shape[2],
            head_dim=self.qk_head_dim,
            bsz=batch_size,
            scale=self.scaling,
            seq_length=backend_sequence_length,
            actual_q_len=actual_seq_len,
            actual_kv_len=actual_seq_len,
            param_sink_key=param_sink_key,
            param_sink_value=param_sink_value,
        )
        if self.use_flash_attn and not self.use_fused_sink_fa:
            attn_output = attn_output[..., : self.v_head_dim]
        elif self.use_flash_attn and self.param_sink_scalar:
            attn_output = attn_output[..., : self.qk_head_dim]
        attn_output = attn_output.flatten(2)
        if self.use_mome:
            attn_output = _apply_mome(
                hidden_states=attn_output,
                mome_mask=mome_mask,
                conv=self.o_conv,
                use_fused=self.use_fused_mome,
            )
        output, bias = self.linear_proj(attn_output.transpose(0, 1))
        output = output.transpose(0, 1)
        return (output, bias) if return_bias else (output, None)


class DSAAttention(nn.Module):
    """Dynamic sparse attention with a Transformers-style BSH interface."""

    def __init__(
        self,
        config,
        layer_number: int,
    ):
        super().__init__()
        if not config.use_mla:
            raise ValueError("DSA layers require MLA configuration.")
        self.config = config
        self.layer_number = layer_number
        self.attention_type = "dsa"
        self.params_dtype = _config_dtype(config)

        self.use_flash_attn = config.use_flash_attn
        self.use_fused_sink_fa = config.use_fused_sink_fa
        self.apply_FA_rescale = config.apply_FA_rescale
        self.attn_implementation = "npu_fusion_attention"
        self.rotary_interleaved = config.rope_interleaved
        self.use_mome = config.use_mome
        self.use_fused_mome = config.use_fused_mome
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.dsa_loss_coeff = config.dsa_loss_coeff
        self.freeze_dsa = config.freeze_DSA
        self.index_head_dim = config.index_head_dim
        self.num_index_heads = config.index_num_attention_heads
        self.index_topk = config.index_topk

        self.sparse_mode = 0
        self.pre_tockens = 1048576
        self.next_tockens = 0
        self.sink_num = config.param_sink_number // 64
        backend_layer_number = max(1, self.layer_number)
        swa_layers = _layer_numbers(config.swa_layers, offset=1)
        self.use_swa = backend_layer_number in swa_layers
        self.swa_sliding_window = None
        if self.use_swa:
            swa_sliding_window = config.sliding_window_list
            swa_windows = (
                list(map(int, swa_sliding_window.split(",")))
                if isinstance(swa_sliding_window, str)
                else list(map(int, swa_sliding_window))
            )
            if len(swa_windows) == 1:
                self.swa_sliding_window = swa_windows[0]
            else:
                swa_layer_list = sorted(swa_layers)
                self.swa_sliding_window = swa_windows[swa_layer_list.index(backend_layer_number)]
        if config.mask_type == "causal" and config.mask_compress:
            self.sparse_mode = 2
            if self.use_swa and config.swa_attention_sink == 0:
                self.sparse_mode = 4
                self.pre_tockens = self.swa_sliding_window
        elif config.mask_type == "general":
            self.sparse_mode = 1

        self.param_sink_number = config.param_sink_number
        self.param_sink_with_value = config.param_sink_with_value
        self.param_sink_scalar = config.param_sink_scalar
        self.param_sink_of_head_num = config.param_sink_of_head_num

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.v_head_dim = config.v_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim

        reference_dim = self.v_head_dim if self.v_head_dim is not None else config.head_dim
        if self.qk_rope_head_dim is None and self.qk_nope_head_dim is None:
            self.qk_nope_head_dim = reference_dim
            self.qk_rope_head_dim = 0
        elif self.qk_rope_head_dim is not None and self.qk_nope_head_dim is None:
            self.qk_nope_head_dim = reference_dim - self.qk_rope_head_dim
        elif self.qk_nope_head_dim is not None and self.qk_rope_head_dim is None:
            self.qk_rope_head_dim = reference_dim - self.qk_nope_head_dim

        self.mla_mm_split = config.mla_mm_split
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        if self.q_lora_rank is None:
            raise ValueError("DSA requires q_lora_rank; the original implementation has no no-Q-LoRA path.")
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        norm_factor = math.sqrt(self.qk_head_dim)
        rope_scaling = config.rope_scaling or {}
        rope_scaling_type = rope_scaling.get("rope_type")
        rope_scaling_factor = rope_scaling.get("factor")
        mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
        if rope_scaling_type is not None:
            if mscale_all_dim:
                mscale = yarn_get_mscale(rope_scaling_factor, mscale_all_dim)
                norm_factor /= mscale * mscale
        self.scaling = 1.0 / norm_factor

        if self.use_mome:
            mome_window = config.router_sliding_window
            conv_hidden = self.v_head_dim * self.num_heads
            self.o_conv = nn.Conv1d(
                conv_hidden,
                conv_hidden,
                mome_window,
                groups=conv_hidden,
                bias=False,
            )

        norm_kwargs = {
            "eps": config.rms_norm_eps,
            "use_fused_rmsnorm": config.use_fused_rmsnorm,
        }
        self.q_layernorm = FusedRMSNorm(hidden_size=self.q_lora_rank, **norm_kwargs)
        self.k_layernorm = FusedRMSNorm(hidden_size=self.kv_lora_rank, **norm_kwargs)

        bias = config.attention_bias or config.add_qkv_bias
        if self.mla_mm_split:
            self.linear_qk_nope = LinearWithMatmul(
                self.q_lora_rank,
                self.num_heads * self.qk_nope_head_dim,
                bias=bias,
            )
            self.linear_qk_rope = LinearWithMatmul(
                self.q_lora_rank,
                self.num_heads * self.qk_rope_head_dim,
                bias=bias,
            )
            _initialize_parameter(config, self.linear_qk_nope.weight)
            _initialize_parameter(config, self.linear_qk_rope.weight)
        else:
            self.linear_qb = LinearWithMatmul(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=bias,
            )
            _initialize_parameter(config, self.linear_qb.weight)

        self.linear_qkv = LinearWithMatmul(
            config.hidden_size,
            self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
            bias=bias,
        )
        _initialize_parameter(config, self.linear_qkv.weight)

        if self.use_mome:
            self.qa_conv = nn.Conv1d(
                self.q_lora_rank,
                self.q_lora_rank,
                mome_window,
                groups=self.q_lora_rank,
                bias=False,
            )
            self.compresskv_conv = nn.Conv1d(
                self.kv_lora_rank,
                self.kv_lora_rank,
                mome_window,
                groups=self.kv_lora_rank,
                bias=False,
            )
        if self.mla_mm_split:
            self.linear_kv_nope = LinearWithMatmul(
                self.kv_lora_rank,
                self.num_heads * self.qk_nope_head_dim,
                bias=bias,
            )
            self.linear_v = LinearWithMatmul(
                self.kv_lora_rank,
                self.num_heads * self.v_head_dim,
                bias=bias,
            )
            _initialize_parameter(config, self.linear_kv_nope.weight)
            _initialize_parameter(config, self.linear_v.weight)
        else:
            kvb_head_dim = self.qk_head_dim - self.qk_rope_head_dim + self.v_head_dim
            self.linear_kvb = LinearWithMatmul(
                self.kv_lora_rank,
                self.num_heads * kvb_head_dim,
                bias=bias,
            )
            _initialize_parameter(config, self.linear_kvb.weight)

        self.linear_proj = LinearWithMatmul(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        _initialize_parameter(config, self.linear_proj.weight, "output")

        self._init_param_sink_parameters(config)

        self.index_linear_qb = LinearWithMatmul(
            self.q_lora_rank,
            config.index_head_dim * config.index_num_attention_heads,
            bias=bias,
        )
        _initialize_parameter(config, self.index_linear_qb.weight)
        self.index_linear_k = LinearWithFusedOps(config.hidden_size, config.index_head_dim, bias=bias, skip_bias_add=False)
        _initialize_parameter(config, self.index_linear_k.weight)
        self.index_k_layernorm = FusedRMSNorm(
            config.index_head_dim,
            config.rms_norm_eps,
            config.use_fused_rmsnorm,
        )
        self.linear_merge_weight = LinearWithMatmul(
            config.hidden_size,
            config.index_num_attention_heads,
            bias=bias,
        )
        _initialize_parameter(config, self.linear_merge_weight.weight)
        self.dsa_dense_warm_up = config.dsa_dense_warm_up

    def _init_param_sink_parameters(self, config):
        if self.param_sink_number <= 0:
            return

        self.register_buffer(
            "param_sink_query",
            torch.zeros(
                (self.param_sink_number, self.num_heads, self.qk_head_dim),
                dtype=self.params_dtype,
            ),
            persistent=False,
        )

        if self.param_sink_of_head_num:
            self.param_sink_num_heads = self.num_heads
        else:
            self.param_sink_num_heads = self.num_key_value_heads

        if self.param_sink_scalar:
            self.register_buffer(
                "param_sink_key_zero_pad",
                torch.zeros(
                    (self.param_sink_number, self.param_sink_num_heads, self.param_sink_scalar - 1),
                    dtype=self.params_dtype,
                ),
                persistent=False,
            )
            self.param_sink_key = nn.Parameter(
                torch.empty(
                    (self.param_sink_number, self.param_sink_num_heads),
                    dtype=self.params_dtype,
                )
            )
            _initialize_parameter(config, self.param_sink_key)
        else:
            self.param_sink_k_pe = nn.Parameter(
                torch.empty(
                    (self.param_sink_number, self.qk_rope_head_dim),
                    dtype=self.params_dtype,
                )
            )
            _initialize_parameter(config, self.param_sink_k_pe)
            self.param_sink_compressed_kv = nn.Parameter(
                torch.empty(
                    (self.param_sink_number, self.kv_lora_rank),
                    dtype=self.params_dtype,
                )
            )
            _initialize_parameter(config, self.param_sink_compressed_kv)


    def _prepare_param_sink(self, batch_size):
        if self.param_sink_scalar:
            raise NotImplementedError("DSA does not support scalar parameter sink.")
        param_sink_k_pe = self.param_sink_k_pe.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, -1, -1)
        compressed_kv = self.param_sink_compressed_kv.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, -1, -1)
        compressed_kv = self.k_layernorm(compressed_kv)
        param_sink_value = (
            F.pad(compressed_kv, [0, self.qk_rope_head_dim])
            if self.use_flash_attn and self.qk_rope_head_dim > 0
            else compressed_kv
        )
        param_sink_key = torch.cat([compressed_kv, param_sink_k_pe], dim=-1)
        return param_sink_key, param_sink_value

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        actual_seq_len=None,
        kv_reuse_states=None,
        output_attentions=False,
        return_bias=False,
        mome_mask=None,
    ):
        if past_key_values is not None or cache_position is not None:
            raise NotImplementedError("DSA cache support has not been implemented yet.")
        if output_attentions:
            raise NotImplementedError("DSA does not expose attention weights for its current backends.")

        batch_size, sequence_length = hidden_states.shape[:-1]

        mixed_x_layer, _ = self.linear_qkv(hidden_states)
        q_a, compressed_kv, k_pe = torch.split(
            mixed_x_layer,
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        if self.use_mome:
            q_a = _apply_mome(
                hidden_states=q_a,
                mome_mask=mome_mask,
                conv=self.qa_conv,
                use_fused=False,
            )
            compressed_kv = _apply_mome(
                hidden_states=compressed_kv,
                mome_mask=mome_mask,
                conv=self.compresskv_conv,
                use_fused=False,
            )
        if self.q_layernorm is not None:
            q_a = self.q_layernorm(q_a)

        if not self.mla_mm_split:
            query_states, _ = self.linear_qb(q_a.contiguous())
            query_states = query_states.view(batch_size, sequence_length, self.num_heads, self.qk_head_dim)
            q_nope, q_pe = torch.split(
                query_states,
                [self.qk_nope_head_dim, self.qk_rope_head_dim],
                dim=-1,
            )
            rearranged_weight = rearrange(
                self.linear_kvb.weight,
                "(N M) L -> N M L",
                N=self.num_heads,
            )
            key_up_proj_weight, _ = torch.split(
                rearranged_weight,
                [self.qk_nope_head_dim, self.v_head_dim],
                dim=1,
            )
            q_nope = rearrange(q_nope, "B S N P -> N (B S) P", N=self.num_heads)
        else:
            q_nope, _ = self.linear_qk_nope(q_a)
            q_pe, _ = self.linear_qk_rope(q_a)
            q_pe = q_pe.view(batch_size, sequence_length, self.num_heads, self.qk_rope_head_dim)
            key_up_proj_weight = rearrange(
                self.linear_kv_nope.weight,
                "(N P) L -> N P L",
                N=self.num_heads,
            )
            q_nope = rearrange(q_nope, "B S (N P) -> N (B S) P", N=self.num_heads)
        q_nope = torch.bmm(q_nope, key_up_proj_weight)
        q_nope = rearrange(q_nope, "N (B S) L -> B S N L", B=batch_size)

        compressed_kv_norm = self.k_layernorm(compressed_kv)
        k_pe = k_pe.view(batch_size, sequence_length, 1, self.qk_rope_head_dim)
        compressed_kv_norm = compressed_kv_norm.view(batch_size, sequence_length, 1, self.kv_lora_rank)

        if self.param_sink_number > 0:
            param_sink_key, param_sink_value = self._prepare_param_sink(batch_size)
        else:
            param_sink_key, param_sink_value = None, None

        index_query, _ = self.index_linear_qb(q_a.detach())
        index_query = index_query.view(batch_size, sequence_length, -1, self.index_head_dim)
        index_key, _ = self.index_linear_k(hidden_states.detach())
        merge_weight, _ = self.linear_merge_weight(hidden_states.detach())
        merge_weight = merge_weight * (self.num_index_heads**-0.5) * (self.index_head_dim**-0.5)
        index_key = index_key.unsqueeze(2)
        if self.index_k_layernorm is not None:
            index_key = self.index_k_layernorm(index_key)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q_pe = apply_rotary_pos_emb(
                q_pe,
                cos,
                sin,
                self.rotary_interleaved,
                use_fused_rotary_pos_emb=True,
            )
            k_pe = apply_rotary_pos_emb(
                k_pe,
                cos,
                sin,
                self.rotary_interleaved,
                use_fused_rotary_pos_emb=True,
            )
            index_query = apply_rotary_pos_emb(
                index_query,
                cos,
                sin,
                self.rotary_interleaved,
                use_fused_rotary_pos_emb=True,
            )
            index_key = apply_rotary_pos_emb(
                index_key,
                cos,
                sin,
                self.rotary_interleaved,
                use_fused_rotary_pos_emb=True,
            )

        if isinstance(actual_seq_len, torch.Tensor):
            actual_seq_len = [int(item) for item in actual_seq_len.tolist()]

        if self.dsa_dense_warm_up:
            query_li = torch.cat([q_nope, q_pe], dim=-1)
            key_li = torch.cat([compressed_kv_norm, k_pe], dim=-1)
            index_query_li = index_query.clone()
            index_key_li = index_key.clone()
            merge_weight_li = merge_weight.clone()

            query_states = query_li
            key_states = key_li
            value_states = F.pad(compressed_kv_norm, [0, self.qk_rope_head_dim])
            attention_interface = ATTENTION_FUNCTIONS["cpu_sdpa"]
            if HAS_NPU and query_states.device.type != "cpu":
                attention_interface = ATTENTION_FUNCTIONS[self.attn_implementation]
            attention_scale = self.scaling
            actual_q_len = actual_seq_len
            actual_kv_len = actual_seq_len
            topk_indices = None
        else:
            actual_q_len = actual_seq_len
            actual_kv_len = actual_seq_len
            if isinstance(actual_seq_len, list):
                actual_q_len = torch.tensor(actual_seq_len, dtype=torch.int32, device=q_nope.device)
                actual_kv_len = actual_q_len
            topk_indices, index_query_tnd, index_key_tnd, merge_weight_tnd = dsa_lightning_indexer_forward(
                self,
                index_query,
                index_key,
                merge_weight,
                actual_q_len,
                actual_kv_len,
            )
            query_states = q_nope
            key_states = compressed_kv_norm
            value_states = compressed_kv_norm
            attention_interface = ATTENTION_FUNCTIONS["dsa_sparse_attention"]
            attention_scale = self.qk_head_dim**-0.5

        attn_result = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            n_head=query_states.shape[2],
            head_dim=self.qk_head_dim,
            bsz=batch_size,
            scale=attention_scale,
            seq_length=sequence_length,
            actual_q_len=actual_q_len,
            actual_kv_len=actual_kv_len,
            param_sink_key=param_sink_key,
            param_sink_value=param_sink_value,
            q_pe=q_pe,
            k_pe=k_pe,
            topk_indices=topk_indices,
        )

        if self.dsa_dense_warm_up:
            attn_output = _attach_dense_indexer_loss(
                attn_result,
                query_li,
                key_li,
                index_query_li,
                index_key_li,
                merge_weight_li,
                actual_q_len,
                actual_kv_len,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.dsa_loss_coeff,
            )
        else:
            attn_output, softmax_max, softmax_sum = attn_result
            if self.training and not self.freeze_dsa:
                q_nope_tnd, compressed_kv_tnd, q_pe_tnd, k_pe_tnd = [
                    rearrange(tensor, "b s n d -> (b s) n d")
                    for tensor in (q_nope, compressed_kv_norm, q_pe, k_pe)
                ]
                dsa_loss_scalar = SparseLightningIndexerKLLossTrainFunction.apply(
                    index_query_tnd,
                    index_key_tnd,
                    merge_weight_tnd,
                    q_nope_tnd,
                    compressed_kv_tnd,
                    topk_indices,
                    softmax_max,
                    softmax_sum,
                    q_pe_tnd,
                    k_pe_tnd,
                    actual_q_len,
                    actual_kv_len,
                    attention_scale,
                    self.dsa_loss_coeff,
                )
                attn_output = AuxLossAutoScaler.apply(attn_output, dsa_loss_scalar)

        if self.use_flash_attn:
            attn_output = attn_output[..., : self.kv_lora_rank]
        attn_output = rearrange(attn_output, "B S N L -> N (B S) L")
        if not self.mla_mm_split:
            rearranged_weight = rearrange(
                self.linear_kvb.weight,
                "(N M) L -> N L M",
                N=self.num_heads,
            )
            _, value_weight = torch.split(
                rearranged_weight,
                [self.qk_nope_head_dim, self.v_head_dim],
                dim=2,
            )
        else:
            value_weight = rearrange(
                self.linear_v.weight,
                "(N V) L -> N L V",
                N=self.num_heads,
            )
        attn_output = torch.bmm(attn_output, value_weight)
        attn_output = rearrange(
            attn_output,
            "N (S B) V -> B S (N V)",
            S=sequence_length,
            B=batch_size,
        )
        if self.use_mome:
            attn_output = _apply_mome(
                hidden_states=attn_output,
                mome_mask=mome_mask,
                conv=self.o_conv,
                use_fused=self.use_fused_mome,
            )
        output, bias = self.linear_proj(attn_output)
        return (output, bias) if return_bias else (output, None)


ATTENTION_CLASSES = {
    "gqa": GQAAttention,
    "mla": MLAAttention,
    "dsa": DSAAttention,
}
