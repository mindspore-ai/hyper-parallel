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
"""Tests for sorting compressed-key positions in the CSA selection chains.

The chains sort the positions ``topk`` returns so a consumer sees them in
ascending order. Sorting those integer indices directly has no AI Core
implementation and falls back to AI CPU, so ``_sorted_positions`` routes the
sort through float32. These tests pin the property that makes the substitution
safe: for positions bounded well below ``2**24`` the result is the integer sort
bit for bit, in all three chains.
"""
import torch

import hyper_parallel.components.modules.shared_compressed_dsa_attention as csa
from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    _sorted_positions,
    compressed_candidate_topk,
    compressed_causal_topk,
    compressed_causal_topk_and_candidates,
)


def _integer_sort(indices: torch.Tensor) -> torch.Tensor:
    """The integer sort the chains used before, kept as the reference."""
    return indices.sort(dim=-1).values


def test_sorted_positions_matches_the_integer_sort():
    """Random positions sort identically through float32."""
    torch.manual_seed(11)
    indices = torch.randint(0, 32768, (2, 64, 512), dtype=torch.int64)
    assert torch.equal(_sorted_positions(indices), _integer_sort(indices))


def test_sorted_positions_keeps_the_input_dtype():
    """int64 in, int64 out; int32 in, int32 out."""
    for dtype in (torch.int64, torch.int32):
        indices = torch.randint(0, 2048, (3, 16), dtype=dtype)
        assert _sorted_positions(indices).dtype == dtype


def test_sorted_positions_handles_the_sentinel_and_negative_fill():
    """The chains fill invalid slots with the compressed length or -1."""
    indices = torch.tensor([[2048, 5, 2048, 0], [-1, 7, -1, 3]], dtype=torch.int64)
    assert torch.equal(_sorted_positions(indices), _integer_sort(indices))


def test_sorted_positions_is_exact_up_to_the_float32_integer_limit():
    """Values just under 2**24 still round-trip exactly."""
    limit = (1 << 24) - 1
    indices = torch.tensor([[limit, limit - 1, 0, 1 << 23]], dtype=torch.int64)
    assert torch.equal(_sorted_positions(indices), _integer_sort(indices))


def test_sorted_positions_returns_a_new_tensor():
    """One call site mutates the result in place, so it must not alias input."""
    indices = torch.tensor([[3, 1, 2]], dtype=torch.int64)
    out = _sorted_positions(indices)
    out[0, 0] = 99
    assert indices.tolist() == [[3, 1, 2]]


def _chain_inputs(seq: int = 16, comp_ratio: int = 2, heads: int = 2, dim: int = 4):
    """Small CPU tensors shaped the way the chains expect."""
    torch.manual_seed(5)
    query = torch.randn(1, seq, heads, dim)
    key = torch.randn(1, seq // comp_ratio, dim)
    merge_weight = torch.randn(1, seq, heads)
    return query, key, merge_weight


def _with_integer_sort(monkeypatch):
    """Point the module's helper back at the integer sort."""
    monkeypatch.setattr(csa, "_sorted_positions", _integer_sort)


def test_causal_topk_is_unchanged_by_the_float32_sort(monkeypatch):
    """The causal chain selects the same positions either way."""
    query, key, merge_weight = _chain_inputs()
    args = {"compress_ratio": 2, "sparse_count": 4, "query_chunk_size": 8}
    fixed = compressed_causal_topk(query, key, merge_weight, **args)
    _with_integer_sort(monkeypatch)
    before = compressed_causal_topk(query, key, merge_weight, **args)
    assert torch.equal(fixed, before)


def test_topk_and_candidates_is_unchanged_by_the_float32_sort(monkeypatch):
    """The source chain returns the same positions and candidate blocks."""
    query, key, merge_weight = _chain_inputs()
    args = {"compress_ratio": 2, "sparse_count": 4, "topk_blocks": 2,
            "block_size": 2, "query_chunk_size": 8}
    fixed = compressed_causal_topk_and_candidates(query, key, merge_weight, **args)
    _with_integer_sort(monkeypatch)
    before = compressed_causal_topk_and_candidates(query, key, merge_weight, **args)
    assert torch.equal(fixed[0], before[0])
    assert torch.equal(fixed[1], before[1])


def test_candidate_topk_is_unchanged_by_the_float32_sort(monkeypatch):
    """The reindex chain selects the same positions either way."""
    query, key, merge_weight = _chain_inputs()
    candidates = compressed_causal_topk_and_candidates(
        query, key, merge_weight, compress_ratio=2, sparse_count=4, topk_blocks=2,
        block_size=2, query_chunk_size=8)[1]
    args = {"compress_ratio": 2, "sparse_count": 4, "block_size": 2,
            "query_chunk_size": 8}
    fixed = compressed_candidate_topk(query, key, merge_weight, candidates, **args)
    _with_integer_sort(monkeypatch)
    before = compressed_candidate_topk(query, key, merge_weight, candidates, **args)
    assert torch.equal(fixed, before)


def test_selected_positions_are_ascending_with_an_invalid_tail():
    """Valid positions form an ascending prefix and the -1 fill trails it.

    The sort puts the sentinel (the compressed length, the largest value) last,
    and the fill then turns that tail into -1.
    """
    query, key, merge_weight = _chain_inputs()
    selected = compressed_causal_topk(query, key, merge_weight, compress_ratio=2,
                                      sparse_count=4, query_chunk_size=8)
    assert selected.dtype == torch.int32
    rows = selected.reshape(-1, selected.shape[-1]).tolist()
    for row in rows:
        valid = [value for value in row if value >= 0]
        assert row[:len(valid)] == valid, row
        assert set(row[len(valid):]) <= {-1}, row
        assert valid == sorted(valid), row
