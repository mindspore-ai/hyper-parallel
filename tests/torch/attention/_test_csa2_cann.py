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
"""Single-card CSA2 CANN forward, backward and KL parity checks."""

import copy
import os
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=unused-import
from torch import nn

from hyper_parallel.components.modules import csa2_cann
from hyper_parallel.components.modules import shared_compressed_dsa_attention as csa
from hyper_parallel.components.modules.csa2_cann_target import CANNSharedCompressedDSAAttention
from hyper_parallel.models.deepseek_v41.adapter.distributed.shared_attention_context_parallel import (
    deepseek_v41_shared_attention_parallel_wrapper,
)


class _ToyCompressor(nn.Module):
    """Produce a compressed KV stream without external model dependencies."""

    def __init__(self, compress_ratio: int = 4) -> None:
        """Create the compressed-key projection."""
        super().__init__()
        self.compress_ratio = compress_ratio
        self.projection = nn.Linear(512, 512, bias=False)

    def forward(self, hidden: torch.Tensor, position_embeddings: tuple) -> tuple:
        """Return the shared latent and compressed KV."""
        del position_embeddings
        latent = self.projection(hidden.reshape(1, -1, self.compress_ratio, 512).mean(dim=2))
        return latent, latent


class _ToyIndexer(nn.Module):
    """Expose the full Indexer projection contract to the shared module."""

    def __init__(self, compress_ratio: int = 4) -> None:
        """Create the projections and selection geometry."""
        super().__init__()
        self.q_b_proj = nn.Linear(512, 32 * 128, bias=False)
        self.weights_proj = nn.Linear(512, 32, bias=False)
        self.wk = nn.Linear(512, 128, bias=False)
        self.k_norm = nn.LayerNorm(128, bias=False)
        self.compress_ratio = compress_ratio
        self.num_heads = 32
        self.head_dim = 128
        self.index_topk = 512
        self.owns_key = True
        self.is_candidate_source = False
        self.uses_candidates = False
        self.candidate_topk_blocks = 0
        self.candidate_block_size = 1
        self.loss_coeff = 0.001
        self.query_chunk_size = 64


class _ToyAttention(nn.Module):
    """Expose the full CSA2 layer contract with production kernel dimensions."""

    def __init__(self, compress_ratio: int = 4) -> None:
        """Create a complete attention parameter tree for parity testing."""
        super().__init__()
        self.q_a_proj = nn.Linear(512, 512, bias=False)
        self.q_a_norm = nn.LayerNorm(512, bias=False)
        self.q_b_proj = nn.Linear(512, 4 * 512, bias=False)
        self.kv_proj = nn.Linear(512, 512, bias=False)
        self.kv_norm = nn.LayerNorm(512, bias=False)
        self.o_a_proj = nn.Linear(512, 512, bias=False)
        self.o_b_proj = nn.Linear(512, 512, bias=False)
        self.compressor = _ToyCompressor(compress_ratio)
        self.indexer = _ToyIndexer(compress_ratio)
        self.sinks = nn.Parameter(torch.zeros(4))
        self.config = SimpleNamespace(o_groups=4, qk_rope_head_dim=64)
        self.layer_idx = 0
        self.num_heads = 4
        self.num_key_value_groups = 1
        self.compress_ratio = compress_ratio
        self.is_kv_source = True
        self.is_index_source = True
        self.kv_source_layer_idx = 0
        self.index_source_layer_idx = 0
        self.candidate_source_layer_idx = None
        self.head_dim = 512
        self.sliding_window = 128
        self.scaling = 512**-0.5


def _selected_indices(
        length: int, compressed_length: int, device: torch.device, compress_ratio: int = 4,
) -> torch.Tensor:
    """Make sorted causal indices including empty and padded rows."""
    positions = torch.arange(512, device=device, dtype=torch.int32)
    counts = (torch.arange(length, device=device, dtype=torch.int32) + 1) // compress_ratio
    return torch.where(
        positions.unsqueeze(0) < counts.unsqueeze(1),
        positions.unsqueeze(0).expand(length, -1),
        -1,
    ).clamp_max(compressed_length - 1).unsqueeze(0).contiguous()


def _check_attention(device: torch.device) -> None:
    """Check the joint window/compressed/sink operator and four gradients."""
    length, compressed_length, heads, head_dim = 2048, 512, 4, 512
    torch.manual_seed(432)
    query = (torch.randn(1, heads, length, head_dim, device=device) * 0.03).to(torch.bfloat16).requires_grad_()
    raw_key = (torch.randn(1, 1, length, head_dim, device=device) * 0.03).to(torch.bfloat16).requires_grad_()
    compressed_key = (
        (torch.randn(1, compressed_length, head_dim, device=device) * 0.03)
        .to(torch.bfloat16).requires_grad_()
    )
    sinks = torch.randn(heads, device=device, dtype=torch.float32).requires_grad_()
    indices = _selected_indices(length, compressed_length, device)
    if not csa2_cann.supports_cann_attention(
            query, raw_key, compressed_key, indices,
            compress_ratio=4, sliding_window=128,
    ):
        raise AssertionError("the production CSA geometry must select the CANN attention path")

    actual = csa2_cann.cann_sparse_attention(
        query, raw_key, compressed_key, indices, sinks, head_dim**-0.5,
    )
    window = csa.build_sliding_window_indices(1, length, 128, device)
    compressed_indices = (indices.long() + length).masked_fill(indices < 0, -1)
    reference = csa._reference_sparse_attention(  # pylint: disable=protected-access
        query,
        torch.cat((raw_key, compressed_key.unsqueeze(1)), dim=2),
        torch.cat((window, compressed_indices), dim=-1),
        sinks,
        head_dim**-0.5,
    )
    torch.testing.assert_close(actual.float(), reference.float(), rtol=0, atol=2.5e-4)

    grad_output = torch.randn_like(actual)
    operands = (query, raw_key, compressed_key, sinks)
    actual_grads = torch.autograd.grad(actual, operands, grad_outputs=grad_output, retain_graph=True)
    reference_grads = torch.autograd.grad(reference, operands, grad_outputs=grad_output)
    tolerances = ((1.0e-5, 0), (1.6e-2, 1.0e-2), (1.6e-2, 1.0e-2), (5.0e-4, 0))
    for actual_grad, reference_grad, (atol, rtol) in zip(actual_grads, reference_grads, tolerances):
        torch.testing.assert_close(actual_grad.float(), reference_grad.float(), rtol=rtol, atol=atol)


def _check_indexer_kl(
        device: torch.device, index_heads: int, compress_ratio: int, query_offset: int = 0,
) -> None:
    """Check the unchanged scalar loss and every Indexer gradient."""
    length = 1024 if query_offset else 2048
    compressed_length = (query_offset + length) // compress_ratio
    index_dim, attention_heads, attention_dim = 128, 4, 512
    torch.manual_seed(700)
    index_query = torch.randn(
        1, length, index_heads, index_dim, device=device, dtype=torch.bfloat16, requires_grad=True,
    )
    index_key = torch.randn(
        1, compressed_length, index_dim, device=device, dtype=torch.bfloat16, requires_grad=True,
    )
    merge_weight = (
        (torch.randn(1, length, index_heads, device=device) * 0.01)
        .to(torch.bfloat16).requires_grad_()
    )
    attention_query = (
        (torch.randn(1, attention_heads, length, attention_dim, device=device) * 0.03)
        .to(torch.bfloat16)
    )
    compressed_key = (
        (torch.randn(1, compressed_length, attention_dim, device=device) * 0.03)
        .to(torch.bfloat16)
    )
    sinks = torch.randn(attention_heads, device=device)
    indices = _selected_indices(
        query_offset + length, compressed_length, device, compress_ratio,
    )[:, query_offset:]
    indices[:, :256] = -1
    if not csa2_cann.supports_cann_indexer_kl(
            index_query, index_key, merge_weight, indices,
            compress_ratio=compress_ratio, query_offset=query_offset,
    ):
        raise AssertionError("the production CSA geometry must select the CANN Indexer KL path")
    args = (index_query, index_key, merge_weight, attention_query, compressed_key, indices, sinks)
    coefficient = 0.001
    chunk_size = 64
    reference = csa._SharedCompressedIndexerKLLoss.apply(  # pylint: disable=protected-access
        *args, attention_dim**-0.5, coefficient, chunk_size, None,
    )
    actual = csa2_cann.cann_indexer_kl_loss(
        *args, attention_scale=attention_dim**-0.5,
        loss_coeff=coefficient, query_chunk_size=chunk_size,
        compress_ratio=compress_ratio, query_offset=query_offset,
    )
    torch.testing.assert_close(actual, reference, rtol=0, atol=1.0e-7)
    operands = (index_query, index_key, merge_weight)
    reference_grads = torch.autograd.grad(reference, operands, retain_graph=True)
    actual_grads = torch.autograd.grad(actual, operands)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad.float(), reference_grad.float(), rtol=1.0e-2, atol=1.0e-8)


def _check_indexer_forward(
        device: torch.device, length: int, compressed_length: int, offset: int, compress_ratio: int,
) -> None:
    """Compare normal Top-K and validate causal keys under exact score ties."""
    heads, head_dim, topk = 32, 128, 512
    torch.manual_seed(737 + offset)
    query = torch.randn(1, length, heads, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, compressed_length, head_dim, device=device, dtype=torch.bfloat16)
    weight = torch.randn(1, length, heads, device=device, dtype=torch.bfloat16).float() * 0.01
    if not csa2_cann.supports_cann_indexer_forward(
            query, key, weight, compress_ratio=compress_ratio, sparse_count=topk, query_offset=offset,
    ):
        raise AssertionError("the full-key CSA2 geometry must select CANN Indexer forward")
    expected = csa.compressed_causal_topk(
        query, key, weight, compress_ratio=compress_ratio, sparse_count=topk,
        query_offset=offset, query_chunk_size=64,
    )
    actual = csa2_cann.cann_indexer_forward(
        query, key, weight, compress_ratio=compress_ratio, query_offset=offset,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    if length == 4096 and offset == 0:
        tied = csa2_cann.cann_indexer_forward(
            torch.zeros_like(query), key, weight,
            compress_ratio=compress_ratio, query_offset=offset,
        )
        final_ids = tied[0, -1].cpu().tolist()
        if len(set(final_ids)) != topk or any(index < 0 or index >= compressed_length for index in final_ids):
            raise AssertionError("CANN tied-score Top-K must contain 512 distinct causal keys")


def _check_indexer_teacher(device: torch.device) -> None:
    """Keep every teacher probability identical after removing key gathers."""
    length, compressed_length, selected_count = 4096, 1024, 512
    attention_heads, head_dim, chunk_size = 4, 512, 64
    torch.manual_seed(713)
    query = (torch.randn(1, attention_heads, length, head_dim, device=device) * 0.03).bfloat16()
    key = (torch.randn(1, compressed_length, head_dim, device=device) * 0.03).bfloat16()
    sinks = torch.randn(attention_heads, device=device)
    visible = (torch.arange(length, device=device) + 1) // 4
    positions = visible.unsqueeze(-1) - selected_count + torch.arange(selected_count, device=device)
    indices = positions.masked_fill(positions < 0, -1).to(torch.int32).unsqueeze(0)
    indices[:, 1000, 0] = compressed_length - 1

    expected_chunks = []
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        selected = indices[:, start:end]
        valid = selected >= 0
        selected_key = key.index_select(1, selected.clamp_min(0).long().reshape(-1))
        selected_key = selected_key.reshape(1, end - start, selected_count, head_dim).float()
        scores = torch.einsum("bhcd,bckd->bhck", query[:, :, start:end].float(), selected_key)
        scores = scores * head_dim**-0.5
        scores.masked_fill_(~valid.unsqueeze(1), -1.0e9)
        sink_logits = sinks.float().view(1, -1, 1, 1).expand(1, -1, end - start, -1)
        target = torch.cat((scores, sink_logits), dim=-1).softmax(dim=-1)
        target = target[..., :-1].masked_fill(~valid.unsqueeze(1), 0.0).sum(dim=1)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        expected_chunks.append(target)
    expected = torch.cat(expected_chunks, dim=1)
    actual = csa2_cann._indexer_target(query, key, indices, sinks, head_dim**-0.5, chunk_size)  # pylint: disable=protected-access
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _check_complete_layer(device: torch.device, compress_ratio: int = 4) -> None:
    """Check selection, attention, auxiliary loss and projections together."""
    length = 2048
    torch.manual_seed(811)
    source = _ToyAttention(compress_ratio)
    reference = csa.SharedCompressedDSAAttentionBase(
        copy.deepcopy(source), replace_indexer=True, use_optimized_sparse_attention=False,
    ).to(device=device, dtype=torch.bfloat16)
    actual = CANNSharedCompressedDSAAttention(
        copy.deepcopy(source),
    ).to(device=device, dtype=torch.bfloat16)
    hidden = (torch.randn(1, length, 512, device=device) * 0.03).to(torch.bfloat16)
    cos = torch.ones(1, length, 32, dtype=torch.bfloat16, device=device)
    sin = torch.zeros_like(cos)
    positions = torch.arange(length, device=device).unsqueeze(0)
    embeddings = {"main": (cos, sin), "compress": (cos, sin)}

    def run(layer: nn.Module) -> tuple:
        """Return one layer's output, gradients and discrete selection."""
        inputs = hidden.detach().clone().requires_grad_()
        state = csa.SharedCompressedAttentionState()
        output, _ = layer(
            inputs, embeddings, positions, None, shared_attention_state=state,
        )
        selection = state.require_topk_indices(0, 0).clone()
        output.float().square().mean().backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in layer.named_parameters() if parameter.grad is not None
        }
        return output.detach(), inputs.grad.detach(), selection, gradients

    reference_output, reference_input_grad, reference_indices, reference_grads = run(reference)
    with mock.patch.object(
            csa2_cann, "cann_indexer_forward", wraps=csa2_cann.cann_indexer_forward,
    ) as fused_indexer, mock.patch.object(
            csa2_cann, "cann_sparse_attention", wraps=csa2_cann.cann_sparse_attention,
    ) as fused_attention, mock.patch.object(
            csa2_cann, "cann_indexer_kl_loss", wraps=csa2_cann.cann_indexer_kl_loss,
    ) as fused_kl:
        actual_output, actual_input_grad, actual_indices, actual_grads = run(actual)
    expected_attention_calls = 1 if compress_ratio == 4 else 0
    if (
            fused_indexer.call_count != 1
            or fused_attention.call_count != expected_attention_calls
            or fused_kl.call_count != 1
    ):
        raise AssertionError("the full layer did not execute the supported CSA2 CANN kernels")
    torch.testing.assert_close(actual_indices, reference_indices, rtol=0, atol=0)
    torch.testing.assert_close(actual_output.float(), reference_output.float(), rtol=1.0e-2, atol=4.0e-3)
    torch.testing.assert_close(actual_input_grad.float(), reference_input_grad.float(), rtol=2.0e-2, atol=1.0e-6)
    for name in (
            "sinks", "q_b_proj.weight", "compressor.projection.weight",
            "indexer.q_b_proj.weight", "indexer.weights_proj.weight",
    ):
        torch.testing.assert_close(
            actual_grads[name].float(), reference_grads[name].float(),
            rtol=2.0e-2, atol=1.0e-6, msg=name,
        )


def _check_window_only_layer(device: torch.device) -> None:
    """Verify ratio-zero replacement uses only the raw sliding-window KV."""
    length = 256
    torch.manual_seed(812)
    source = _ToyAttention(0)
    del source.compressor
    del source.indexer
    source.is_kv_source = False
    source.is_index_source = False
    source.kv_source_layer_idx = None
    source.index_source_layer_idx = None
    reference = csa.SharedCompressedDSAAttentionBase(
        copy.deepcopy(source), replace_indexer=True, use_optimized_sparse_attention=False,
    ).to(device=device, dtype=torch.bfloat16)
    actual = CANNSharedCompressedDSAAttention(copy.deepcopy(source)).to(device=device, dtype=torch.bfloat16)
    hidden = (torch.randn(1, length, 512, device=device) * 0.03).to(torch.bfloat16)
    cos = torch.ones(1, length, 32, dtype=torch.bfloat16, device=device)
    sin = torch.zeros_like(cos)
    positions = torch.arange(length, device=device).unsqueeze(0)
    embeddings = {"main": (cos, sin), "compress": (cos, sin)}

    def run(layer: nn.Module) -> tuple[torch.Tensor, torch.Tensor, csa.SharedCompressedAttentionState]:
        """Return output, input gradient and published state."""
        inputs = hidden.detach().clone().requires_grad_()
        state = csa.SharedCompressedAttentionState()
        output, _ = layer(inputs, embeddings, positions, None, shared_attention_state=state)
        output.float().square().mean().backward()
        return output.detach(), inputs.grad.detach(), state

    reference_output, reference_grad, _ = run(reference)
    with mock.patch.object(
            csa, "npu_sparse_attention_with_scalar_sink",
            wraps=csa.npu_sparse_attention_with_scalar_sink,
    ) as sparse_attention, mock.patch.object(
            csa2_cann, "cann_sparse_attention", wraps=csa2_cann.cann_sparse_attention,
    ) as fused_attention, mock.patch.object(
            csa2_cann, "cann_indexer_forward", wraps=csa2_cann.cann_indexer_forward,
    ) as fused_indexer:
        actual_output, actual_grad, state = run(actual)
    if sparse_attention.call_count != 1 or fused_attention.call_count or fused_indexer.call_count:
        raise AssertionError("ratio-zero layer did not use the sliding-window sparse attention path")
    query, key_value, indices = sparse_attention.call_args.args[:3]
    if query.shape[2] != length or key_value.shape[2] != length or indices.shape != (1, length, 128):
        raise AssertionError("ratio-zero layer included compressed KV or Top-K indices")
    if state.compressed_kv_by_source or state.topk_indices_by_source:
        raise AssertionError("ratio-zero layer published compressed KV or Indexer selection")
    torch.testing.assert_close(actual_output.float(), reference_output.float(), rtol=1.0e-2, atol=4.0e-3)
    torch.testing.assert_close(actual_grad.float(), reference_grad.float(), rtol=2.0e-2, atol=3.0e-6)


def _check_cp_kernel_offsets(
        device: torch.device,
        global_length: int,
        query_offset: int,
        local_length: int = 1024,
) -> None:
    """Check one CP rank's local queries against globally gathered keys."""
    compressed_length, attention_heads, attention_dim = global_length // 4, 4, 512
    torch.manual_seed(923)
    query = torch.randn(1, attention_heads, local_length, attention_dim, device=device) * 0.03
    query = query.to(torch.bfloat16).requires_grad_()
    raw_key = torch.randn(1, 1, global_length, attention_dim, device=device) * 0.03
    raw_key = raw_key.to(torch.bfloat16).requires_grad_()
    compressed_key = torch.randn(1, compressed_length, attention_dim, device=device) * 0.03
    compressed_key = compressed_key.to(torch.bfloat16).requires_grad_()
    sinks = torch.randn(attention_heads, device=device).requires_grad_()
    if global_length == 2048:
        indices = _selected_indices(global_length, compressed_length, device)[:, query_offset:]
    else:
        positions = torch.arange(query_offset, query_offset + local_length, device=device)
        visible = (positions + 1) // 4
        recent = (visible - 512).unsqueeze(-1) + torch.arange(512, device=device)
        indices = recent.masked_fill(recent >= visible.unsqueeze(-1), -1).to(torch.int32).unsqueeze(0)
    if not csa2_cann.supports_cann_attention(
            query, raw_key, compressed_key, indices,
            compress_ratio=4, sliding_window=128, query_offset=query_offset,
    ):
        raise AssertionError("CP-local CSA geometry must select CANN attention")
    actual = csa2_cann.cann_sparse_attention(
        query, raw_key, compressed_key, indices, sinks,
        attention_dim**-0.5, query_offset=query_offset,
    )
    window = csa.build_sliding_window_indices(
        1, local_length, 128, device,
        query_offset=query_offset, key_length=global_length,
    )
    sparse_indices = torch.cat((
        window,
        (indices.long() + global_length).masked_fill(indices < 0, -1),
    ), dim=-1)
    reference = csa._reference_sparse_attention(  # pylint: disable=protected-access
        query, torch.cat((raw_key, compressed_key.unsqueeze(1)), dim=2),
        sparse_indices, sinks, attention_dim**-0.5,
    )
    torch.testing.assert_close(actual.float(), reference.float(), rtol=0, atol=2.5e-4)
    grad_output = torch.randn_like(actual)
    operands = (query, raw_key, compressed_key, sinks)
    actual_grads = torch.autograd.grad(actual, operands, grad_outputs=grad_output, retain_graph=True)
    reference_grads = torch.autograd.grad(reference, operands, grad_outputs=grad_output)
    for actual_grad, reference_grad, atol in zip(
            actual_grads, reference_grads, (1.0e-5, 1.6e-2, 1.6e-2, 5.0e-4),
    ):
        torch.testing.assert_close(actual_grad.float(), reference_grad.float(), rtol=1.0e-2, atol=atol)

    index_heads, index_dim = 32, 128
    index_query = torch.randn(
        1, local_length, index_heads, index_dim,
        device=device, dtype=torch.bfloat16, requires_grad=True,
    )
    index_key = torch.randn(
        1, compressed_length, index_dim,
        device=device, dtype=torch.bfloat16, requires_grad=True,
    )
    merge_weight = torch.randn(1, local_length, index_heads, device=device) * 0.01
    merge_weight = merge_weight.to(torch.bfloat16).requires_grad_()
    if not csa2_cann.supports_cann_indexer_kl(
            index_query, index_key, merge_weight, indices,
            compress_ratio=4, query_offset=query_offset,
    ):
        raise AssertionError("CP-local Indexer geometry must select fused KL")
    args = (index_query, index_key, merge_weight, query.detach(), compressed_key.detach(), indices, sinks.detach())
    expected_loss = csa._SharedCompressedIndexerKLLoss.apply(  # pylint: disable=protected-access
        *args, attention_dim**-0.5, 0.001, 64, None,
    )
    actual_loss = csa2_cann.cann_indexer_kl_loss(
        *args, attention_scale=attention_dim**-0.5,
        loss_coeff=0.001, query_chunk_size=64,
        compress_ratio=4, query_offset=query_offset,
    )
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=1.0e-7)
    operands = (index_query, index_key, merge_weight)
    expected_grads = torch.autograd.grad(expected_loss, operands, retain_graph=True)
    actual_grads = torch.autograd.grad(actual_loss, operands)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad.float(), expected_grad.float(), rtol=1.0e-2, atol=1.0e-8)


def test_csa2_cann_forward_backward_parity() -> None:
    """Run the validated CANN shapes on one Ascend card."""
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    _check_attention(device)
    for compress_ratio in (1, 2, 4):
        _check_indexer_kl(device, 32, compress_ratio)
    _check_indexer_kl(device, 64, 4)
    _check_indexer_kl(device, 32, 2, query_offset=1024)
    _check_indexer_forward(device, 1024, 1024, 0, 1)
    _check_indexer_forward(device, 2048, 1024, 0, 2)
    _check_indexer_forward(device, 1024, 1024, 1024, 2)
    _check_indexer_forward(device, 4096, 1024, 0, 4)
    _check_indexer_forward(device, 1024, 1024, 3072, 4)
    _check_indexer_teacher(device)
    _check_complete_layer(device, 4)
    _check_complete_layer(device, 2)
    _check_complete_layer(device, 1)
    _check_window_only_layer(device)
    _check_cp_kernel_offsets(device, 2048, 1024)
    _check_cp_kernel_offsets(device, 4096, 3072)
    _check_cp_kernel_offsets(device, 4096, 2048, local_length=2048)


class _CPMesh:
    """Expose the initialized two-rank group to the production CP wrapper."""

    @staticmethod
    def size() -> int:
        """Return the world size."""
        return dist.get_world_size()

    @staticmethod
    def get_local_rank() -> int:
        """Return the local CP rank."""
        return dist.get_rank()

    @staticmethod
    def get_group() -> dist.ProcessGroup:
        """Return the production all-gather process group."""
        return dist.group.WORLD


def test_csa2_cann_two_rank_cp_parity() -> None:
    """Compare the complete target with the existing CP implementation."""
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)
    dist.init_process_group("hccl")
    try:
        if dist.get_world_size() != 2:
            raise AssertionError("the CSA2 CP test requires exactly two ranks")
        local_length = 1024
        torch.manual_seed(1007)
        source = _ToyAttention()
        reference = csa.SharedCompressedDSAAttention(copy.deepcopy(source)).to(
            device=device, dtype=torch.bfloat16,
        )
        reference.use_optimized_sparse_attention = False
        actual = CANNSharedCompressedDSAAttention(copy.deepcopy(source)).to(
            device=device, dtype=torch.bfloat16,
        )
        global_hidden = torch.randn(1, local_length * 2, 512) * 0.03
        hidden = global_hidden[:, rank * local_length:(rank + 1) * local_length].to(
            device=device, dtype=torch.bfloat16,
        )
        positions = torch.arange(
            rank * local_length, (rank + 1) * local_length, device=device,
        ).unsqueeze(0)
        cos = torch.ones(1, local_length, 32, device=device, dtype=torch.bfloat16)
        embeddings = {"main": (cos, torch.zeros_like(cos)), "compress": (cos, torch.zeros_like(cos))}
        def run(layer: nn.Module) -> tuple:
            """Run the same local batch through one CP implementation."""
            inputs = hidden.detach().clone().requires_grad_()
            state = csa.SharedCompressedAttentionState()
            request = deepseek_v41_shared_attention_parallel_wrapper(
                layer, None, None, _CPMesh(), None,
            )
            if request.target is not layer:
                raise AssertionError("the CP wrapper must bind the selected attention target")
            output, _ = request.forward(
                inputs, embeddings, positions, None,
                shared_attention_state=state,
            )
            selection = state.require_topk_indices(0, 0).clone()
            output.float().square().mean().backward()
            gradients = {
                name: parameter.grad.detach().clone()
                for name, parameter in layer.named_parameters() if parameter.grad is not None
            }
            return output.detach(), inputs.grad.detach(), selection, gradients

        expected_output, expected_input_grad, expected_indices, expected_grads = run(reference)
        with mock.patch.object(
                csa2_cann, "cann_indexer_forward", wraps=csa2_cann.cann_indexer_forward,
        ) as fused_indexer, mock.patch.object(
                csa2_cann, "cann_sparse_attention", wraps=csa2_cann.cann_sparse_attention,
        ) as fused_attention, mock.patch.object(
                csa2_cann, "cann_indexer_kl_loss", wraps=csa2_cann.cann_indexer_kl_loss,
        ) as fused_kl:
            actual_output, actual_input_grad, actual_indices, actual_grads = run(actual)
        if fused_indexer.call_count != 1 or fused_attention.call_count != 1 or fused_kl.call_count != 1:
            raise AssertionError("all three CANN kernels must run on every CP rank")
        torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
        torch.testing.assert_close(actual_output.float(), expected_output.float(), rtol=1.0e-2, atol=4.0e-3)
        torch.testing.assert_close(actual_input_grad.float(), expected_input_grad.float(), rtol=2.0e-2, atol=1.0e-6)
        for name in (
                "sinks", "q_b_proj.weight", "compressor.projection.weight",
                "indexer.q_b_proj.weight", "indexer.weights_proj.weight",
        ):
            torch.testing.assert_close(
                actual_grads[name].float(), expected_grads[name].float(),
                rtol=2.0e-2, atol=1.0e-6, msg=f"rank {rank}: {name}",
            )
    finally:
        dist.destroy_process_group()
