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
"""Distributed CPU parity worker for shared compressed DSA CP."""

from copy import deepcopy
from typing import Any

import torch
import torch.distributed as dist
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)

from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedPackedSequence,
    SharedCompressedAttentionState,
    SharedCompressedDSAAttention,
    compressed_candidate_topk,
    compressed_causal_topk_and_candidates,
    shared_compressed_indexer_kl_loss,
)
from hyper_parallel.models.deepseek_v41.adapter.context_parallel import (
    _build_shared_attention_cp_context,
    _build_shared_attention_tp_context,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41AttentionPlaceholder,
)


class _CPMesh:
    """Minimal mesh interface backed by the initialized world group."""

    @staticmethod
    def size() -> int:
        """Return the CP world size."""
        return dist.get_world_size()

    @staticmethod
    def get_local_rank() -> int:
        """Return this process's CP rank."""
        return dist.get_rank()

    @staticmethod
    def get_group() -> Any:
        """Return the raw process group expected by the collective API."""
        return dist.group.WORLD


_TPMesh = _CPMesh


def _config() -> DeepseekV4Config:
    """Create a small V4.1 attention configuration."""
    config = DeepseekV4Config(  # pylint: disable=unexpected-keyword-arg
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=16,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        max_position_embeddings=128,
        layer_types=["sliding_attention"] * 4,
        mlp_layer_types=["moe"] * 4,
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 2},
        compress_rope_theta=10000.0,
        sliding_window=8,
        o_groups=2,
        o_lora_rank=16,
        index_n_heads=4,
        index_head_dim=8,
        index_topk=2,
        rms_norm_eps=1.0e-6,
        use_cache=False,
        partial_rotary_factor=0.5,
    )
    config.v41_compress_ratios = [0, 0, 2, 2]
    config.v41_kv_source_layer_ids = [2]
    config.v41_index_source_layer_ids = [2]
    return config


def _position_embeddings(position_ids: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Build deterministic half-width interleaved RoPE factors."""
    frequencies = torch.tensor([1.0, 0.1])
    angles = position_ids.float().unsqueeze(-1) * frequencies
    embeddings = (angles.cos(), angles.sin())
    return {"main": embeddings, "compress": embeddings}


def test_deepseek_v41_dsa_cp_gloo():
    """Async CP matches output, hidden gradient, and replicated parameter gradient."""
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() == 2
        torch.manual_seed(29)
        reference = SharedCompressedDSAAttention(
            DeepseekV41AttentionPlaceholder(_config(), layer_idx=2)
        )
        with torch.no_grad():
            for name, parameter in reference.named_parameters():
                if name == "sinks":
                    parameter.zero_()
                elif parameter.ndim == 1:
                    parameter.fill_(1.0)
                else:
                    parameter.normal_(mean=0.0, std=0.1)
        parallel = deepcopy(reference)

        torch.manual_seed(31)
        full_hidden = torch.randn(1, 8, 32, requires_grad=True)
        local_hidden = full_hidden.detach()[:, rank * 4:(rank + 1) * 4].clone().requires_grad_(True)
        full_positions = torch.arange(8).unsqueeze(0)
        local_positions = full_positions[:, rank * 4:(rank + 1) * 4].contiguous()

        reference_output, _ = reference(
            full_hidden,
            position_embeddings=_position_embeddings(full_positions),
            position_ids=full_positions,
            attention_mask=None,
            shared_attention_state=SharedCompressedAttentionState(),
            packed_seq_params=SharedCompressedPackedSequence(
                cu_seq_lens=torch.tensor([0, 4, 8], dtype=torch.int32),
                local_query_start=0,
                local_query_length=8,
                global_sequence_length=8,
            ),
        )
        parallel_output, _ = parallel(
            local_hidden,
            position_embeddings=_position_embeddings(local_positions),
            position_ids=local_positions,
            attention_mask=None,
            shared_attention_state=SharedCompressedAttentionState(),
            shared_attention_cp_context=_build_shared_attention_cp_context(_CPMesh()),
            packed_seq_params=SharedCompressedPackedSequence(
                cu_seq_lens=torch.tensor([0, 4, 8], dtype=torch.int32),
                local_query_start=rank * 4,
                local_query_length=4,
                global_sequence_length=8,
            ),
        )
        expected_output = reference_output[:, rank * 4:(rank + 1) * 4]
        torch.testing.assert_close(parallel_output, expected_output, rtol=1.0e-5, atol=1.0e-6)

        token_weights = torch.cat((torch.ones(4), torch.full((4,), 2.0))).view(1, 8, 1)
        (reference_output * token_weights).sum().backward()
        (parallel_output * float(rank + 1)).sum().backward()
        expected_hidden_grad = full_hidden.grad[:, rank * 4:(rank + 1) * 4]
        torch.testing.assert_close(local_hidden.grad, expected_hidden_grad, rtol=1.0e-5, atol=1.0e-6)

        reference_parameters = dict(reference.named_parameters())
        for name, parameter in parallel.named_parameters():
            expected = reference_parameters[name].grad
            if parameter.grad is None or expected is None:
                assert parameter.grad is expected
                continue
            dist.all_reduce(parameter.grad)
            torch.testing.assert_close(parameter.grad, expected, rtol=1.0e-5, atol=1.0e-6)
    finally:
        dist.destroy_process_group()


def test_deepseek_v41_indexer_tp_gloo():
    """Head-sharded Indexer matches Full/Reindex selection and KL gradients."""
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert world_size == 2
        torch.manual_seed(37)
        index_query = torch.randn(1, 8, 4, 3)
        index_key = torch.randn(1, 4, 3)
        merge_weight = torch.randn(1, 8, 4)
        local_head_slice = slice(rank * 2, (rank + 1) * 2)
        tp_context = _build_shared_attention_tp_context(_TPMesh())

        expected_topk, expected_candidates = compressed_causal_topk_and_candidates(
            index_query,
            index_key,
            merge_weight,
            compress_ratio=2,
            sparse_count=2,
            topk_blocks=1,
            block_size=2,
            query_chunk_size=2,
        )
        actual_topk, actual_candidates = compressed_causal_topk_and_candidates(
            index_query[:, :, local_head_slice],
            index_key,
            merge_weight[:, :, local_head_slice],
            compress_ratio=2,
            sparse_count=2,
            topk_blocks=1,
            block_size=2,
            query_chunk_size=2,
            reduce_sum=tp_context.reduce_sum,
        )
        torch.testing.assert_close(actual_topk, expected_topk)
        torch.testing.assert_close(actual_candidates, expected_candidates)
        actual_reindex = compressed_candidate_topk(
            index_query[:, :, local_head_slice],
            index_key,
            merge_weight[:, :, local_head_slice],
            actual_candidates,
            compress_ratio=2,
            sparse_count=2,
            block_size=2,
            query_chunk_size=2,
            reduce_sum=tp_context.reduce_sum,
        )
        expected_reindex = compressed_candidate_topk(
            index_query,
            index_key,
            merge_weight,
            expected_candidates,
            compress_ratio=2,
            sparse_count=2,
            block_size=2,
            query_chunk_size=2,
        )
        torch.testing.assert_close(actual_reindex, expected_reindex)

        full_inputs = [
            index_query.clone().requires_grad_(),
            index_key.clone().requires_grad_(),
            merge_weight.clone().requires_grad_(),
        ]
        local_inputs = [
            index_query[:, :, local_head_slice].clone().requires_grad_(),
            index_key.clone().requires_grad_(),
            merge_weight[:, :, local_head_slice].clone().requires_grad_(),
        ]
        attention_query = torch.randn(1, 4, 8, 5)
        compressed_key = torch.randn(1, 4, 5)
        sinks = torch.randn(4)
        expected_loss = shared_compressed_indexer_kl_loss(
            *full_inputs,
            attention_query,
            compressed_key,
            expected_topk,
            sinks,
            attention_scale=5**-0.5,
            loss_coeff=0.1,
            query_chunk_size=2,
        )
        actual_loss = shared_compressed_indexer_kl_loss(
            *local_inputs,
            attention_query[:, local_head_slice],
            compressed_key,
            actual_topk,
            sinks[local_head_slice],
            attention_scale=5**-0.5,
            loss_coeff=0.1,
            query_chunk_size=2,
            tp_context=tp_context,
        )
        expected_loss.backward()
        actual_loss.backward()

        torch.testing.assert_close(actual_loss, expected_loss, rtol=1.0e-5, atol=1.0e-6)
        torch.testing.assert_close(
            local_inputs[0].grad,
            full_inputs[0].grad[:, :, local_head_slice],
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            local_inputs[2].grad,
            full_inputs[2].grad[:, :, local_head_slice],
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        dist.all_reduce(local_inputs[1].grad)
        torch.testing.assert_close(
            local_inputs[1].grad,
            full_inputs[1].grad,
            rtol=1.0e-5,
            atol=1.0e-6,
        )
    finally:
        dist.destroy_process_group()
