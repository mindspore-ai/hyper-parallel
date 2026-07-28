# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Unit tests for platform-independent multi-stream PP group construction."""

import pytest

from hyper_parallel.platform.platform import _build_p2p_edge_rank_lists
from tests.common.mark_utils import arg_mark


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_build_linear_p2p_edges_preserves_pipeline_adjacency() -> None:
    """
    Feature: Platform-independent pipeline edge construction.
    Description: Build edges for an ordered, non-interleaved PP rank list.
    Expectation: Only consecutive logical ranks form normalized two-rank edges.
    """
    edges = _build_p2p_edge_rank_lists([0, 2, 4, 6])
    assert edges == [(0, 2), (2, 4), (4, 6)], f"Unexpected linear PP edges: {edges}"


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_build_interleaved_p2p_edges_adds_wrap_once() -> None:
    """
    Feature: Interleaved pipeline edge construction.
    Description: Add the physical last-to-first edge used between virtual pipeline chunks.
    Expectation: The wrap edge appears once and a two-rank PP group remains deduplicated.
    """
    edges = _build_p2p_edge_rank_lists([0, 2, 4, 6], include_wrap=True)
    assert edges == [(0, 2), (0, 6), (2, 4), (4, 6)], f"Unexpected wrapped PP edges: {edges}"
    assert _build_p2p_edge_rank_lists([0, 1], include_wrap=True) == [(0, 1)]


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("invalid_ranks", [[0, 1, 1], [0, True], "0,1"])
def test_build_p2p_edges_rejects_invalid_rank_lists(invalid_ranks: object) -> None:
    """
    Feature: Pipeline edge rank-list validation.
    Description: Pass duplicate, boolean, and non-sequence rank specifications.
    Expectation: Invalid or ambiguous inputs raise ValueError before group creation.
    """
    with pytest.raises(ValueError):
        _build_p2p_edge_rank_lists(invalid_ranks)
