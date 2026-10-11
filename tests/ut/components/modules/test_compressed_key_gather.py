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
"""Tests for gathering compressed keys in the CSA selection chains.

The candidate chain and the Indexer KL loss gather compressed keys at selected
positions. Advanced indexing with two index tensors has no AI Core
implementation, so ``_gather_keys`` folds the batch offset into one index and
uses a single ``index_select``. These tests pin that the substitution is the
same gather, including for a batch larger than one, where the folded offset is
what could go wrong.
"""
import torch

import hyper_parallel.components.modules.shared_compressed_dsa_attention as csa
from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    _gather_keys,
    compressed_candidate_topk,
    compressed_causal_topk_and_candidates,
)


def _advanced_index(key: torch.Tensor, batch_indices: torch.Tensor,
                    positions: torch.Tensor) -> torch.Tensor:
    """The advanced indexing the helper replaces, kept as the reference."""
    return key[batch_indices, positions]


def _case(batch: int, compressed: int, rows: int, width: int, dim: int = 4):
    """Key, batch ids and positions shaped the way the chains build them."""
    torch.manual_seed(3)
    key = torch.randn(batch, compressed, dim)
    batch_indices = torch.arange(batch).view(-1, 1, 1)
    positions = torch.randint(0, compressed, (batch, rows, width))
    return key, batch_indices, positions


def test_gather_matches_advanced_indexing_for_one_batch():
    """The single-batch gather is unchanged."""
    key, batch_indices, positions = _case(1, 16, 5, 7)
    assert torch.equal(_gather_keys(key, batch_indices, positions),
                       _advanced_index(key, batch_indices, positions))


def test_gather_matches_advanced_indexing_across_batches():
    """With batch > 1 the folded batch offset must pick the right rows."""
    key, batch_indices, positions = _case(3, 16, 5, 7)
    assert torch.equal(_gather_keys(key, batch_indices, positions),
                       _advanced_index(key, batch_indices, positions))


def test_gather_does_not_mix_batches():
    """Each batch entry gathers only from its own key rows."""
    key = torch.zeros(2, 4, 3)
    key[1] = 1.0
    batch_indices = torch.arange(2).view(-1, 1, 1)
    positions = torch.zeros(2, 1, 2, dtype=torch.int64)
    gathered = _gather_keys(key, batch_indices, positions)
    assert torch.equal(gathered[0], torch.zeros(1, 2, 3))
    assert torch.equal(gathered[1], torch.ones(1, 2, 3))


def test_gather_shape_and_dtype():
    """The result is positions.shape + (head_dim,) in the key's dtype."""
    key, batch_indices, positions = _case(2, 8, 3, 5, dim=6)
    gathered = _gather_keys(key, batch_indices, positions)
    assert gathered.shape == (2, 3, 5, 6)
    assert gathered.dtype == key.dtype


def test_gather_handles_a_non_contiguous_key():
    """A transposed key still gathers correctly."""
    base = torch.randn(2, 4, 8)
    key = base.transpose(1, 2)
    batch_indices = torch.arange(2).view(-1, 1, 1)
    positions = torch.randint(0, 8, (2, 3, 2))
    assert torch.equal(_gather_keys(key, batch_indices, positions),
                       _advanced_index(key, batch_indices, positions))


def test_gather_handles_repeated_positions():
    """Repeats are allowed; each occurrence gathers the same row."""
    key, batch_indices, _ = _case(1, 8, 1, 1)
    positions = torch.full((1, 2, 3), 5, dtype=torch.int64)
    gathered = _gather_keys(key, batch_indices, positions)
    assert torch.equal(gathered, key[0, 5].expand(1, 2, 3, key.shape[-1]))


def _chain_inputs(seq: int = 16, comp_ratio: int = 2, heads: int = 2, dim: int = 4,
                  batch: int = 1):
    """Small CPU tensors shaped the way the chains expect."""
    torch.manual_seed(5)
    query = torch.randn(batch, seq, heads, dim)
    key = torch.randn(batch, seq // comp_ratio, dim)
    merge_weight = torch.randn(batch, seq, heads)
    return query, key, merge_weight


def test_candidate_topk_is_unchanged_by_the_gather(monkeypatch):
    """The reindex chain selects the same positions either way."""
    query, key, merge_weight = _chain_inputs()
    candidates = compressed_causal_topk_and_candidates(
        query, key, merge_weight, compress_ratio=2, sparse_count=4, topk_blocks=2,
        block_size=2, query_chunk_size=8)[1]
    args = {"compress_ratio": 2, "sparse_count": 4, "block_size": 2,
            "query_chunk_size": 8}
    fixed = compressed_candidate_topk(query, key, merge_weight, candidates, **args)
    monkeypatch.setattr(csa, "_gather_keys", _advanced_index)
    before = compressed_candidate_topk(query, key, merge_weight, candidates, **args)
    assert torch.equal(fixed, before)


def test_candidate_topk_is_unchanged_with_a_larger_batch(monkeypatch):
    """Batch > 1 exercises the folded offset through the whole chain."""
    query, key, merge_weight = _chain_inputs(batch=3)
    candidates = compressed_causal_topk_and_candidates(
        query, key, merge_weight, compress_ratio=2, sparse_count=4, topk_blocks=2,
        block_size=2, query_chunk_size=8)[1]
    args = {"compress_ratio": 2, "sparse_count": 4, "block_size": 2,
            "query_chunk_size": 8}
    fixed = compressed_candidate_topk(query, key, merge_weight, candidates, **args)
    monkeypatch.setattr(csa, "_gather_keys", _advanced_index)
    before = compressed_candidate_topk(query, key, merge_weight, candidates, **args)
    assert torch.equal(fixed, before)
