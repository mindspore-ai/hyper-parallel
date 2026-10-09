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
"""Distributed CPU parity worker for the shared compressed DSA indexer."""

from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    compressed_candidate_topk,
    compressed_causal_topk_and_candidates,
    shared_compressed_indexer_kl_loss,
)
from hyper_parallel.models.deepseek_v41.adapter.distributed.shared_attention_context_parallel import (
    _build_shared_attention_tp_context,
)


class _TPMesh:
    """Minimal TP mesh interface backed by the initialized world group."""

    @staticmethod
    def size() -> int:
        """Return the TP world size."""
        return dist.get_world_size()

    @staticmethod
    def get_local_rank() -> int:
        """Return this process's TP rank."""
        return dist.get_rank()

    @staticmethod
    def get_group() -> Any:
        """Return the raw process group expected by the collective API."""
        return dist.group.WORLD


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
