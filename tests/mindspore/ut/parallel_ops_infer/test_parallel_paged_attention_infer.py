# Copyright 2025 Huawei Technologies Co., Ltd
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
"""paged_attention distributed infer layout tests"""

import pytest
from hyper_parallel import Layout
from hyper_parallel.core.shard.ops.parallel_paged_attention import PagedAttentionDistributedOp

op = PagedAttentionDistributedOp("PagedAttention")


def _build_layout(mesh_shape, alias_name, rank_list, query_map, cache_map, block_map):
    query_layout = Layout(mesh_shape, alias_name, rank_list)(*query_map)
    key_cache_layout = Layout(mesh_shape, alias_name, rank_list)(*cache_map)
    value_cache_layout = Layout(mesh_shape, alias_name, rank_list)(*cache_map)
    block_tables_layout = Layout(mesh_shape, alias_name, rank_list)(*block_map)
    context_lens_layout = Layout(mesh_shape, alias_name, rank_list)("None",)
    return query_layout, key_cache_layout, value_cache_layout, block_tables_layout, context_lens_layout


def test_paged_attention_layout_basic():
    """
    Feature: PagedAttention infer layout
    Description: Allow num_tokens/q_head_num sharding and propagate to output
    Expectation: Success
    """
    mesh_shape = (2, 4)
    alias_name = ("dp", "mp")
    rank_list = list(range(8))

    query_layout, key_cache_layout, value_cache_layout, block_tables_layout, context_lens_layout = _build_layout(
        mesh_shape,
        alias_name,
        rank_list,
        ("dp", "mp", "None"),
        ("None", "None", "mp", "None"),
        ("dp", "None"),
    )

    output_layout = op.infer_layout(
        (query_layout, key_cache_layout, value_cache_layout, block_tables_layout, context_lens_layout),
        (),
    )
    assert output_layout.tensor_map == (1, 0, -1)


def test_paged_attention_disallow_block_shard():
    """
    Feature: PagedAttention infer layout
    Description: Disallow sharding on block_size/num_blocks
    Expectation: Raise ValueError
    """
    mesh_shape = (2, 4)
    alias_name = ("dp", "mp")
    rank_list = list(range(8))

    query_layout, key_cache_layout, value_cache_layout, block_tables_layout, context_lens_layout = _build_layout(
        mesh_shape,
        alias_name,
        rank_list,
        ("dp", "mp", "None"),
        ("dp", "None", "mp", "None"),
        ("dp", "None"),
    )

    with pytest.raises(ValueError):
        _ = op.infer_layout(
            (query_layout, key_cache_layout, value_cache_layout, block_tables_layout, context_lens_layout),
            (),
        )


def test_paged_attention_disallow_head_dim_shard():
    """
    Feature: PagedAttention infer layout
    Description: Disallow sharding on head_dim
    Expectation: Raise ValueError
    """
    mesh_shape = (2, 4)
    alias_name = ("dp", "mp")
    rank_list = list(range(8))

    query_layout, key_cache_layout, value_cache_layout, block_tables_layout, context_lens_layout = _build_layout(
        mesh_shape,
        alias_name,
        rank_list,
        ("dp", "None", "mp"),
        ("None", "None", "None", "None"),
        ("dp", "None"),
    )

    with pytest.raises(ValueError):
        _ = op.infer_layout(
            (query_layout, key_cache_layout, value_cache_layout, block_tables_layout, context_lens_layout),
            (),
        )


def test_paged_attention_mismatch_head_shard():
    """
    Feature: PagedAttention infer layout
    Description: Require kv_head_num shard same as query q_head_num
    Expectation: Raise ValueError
    """
    mesh_shape = (2, 4)
    alias_name = ("dp", "mp")
    rank_list = list(range(8))

    query_layout, key_cache_layout, value_cache_layout, block_tables_layout, context_lens_layout = _build_layout(
        mesh_shape,
        alias_name,
        rank_list,
        ("dp", "mp", "None"),
        ("None", "None", "dp", "None"),
        ("dp", "None"),
    )

    with pytest.raises(ValueError):
        _ = op.infer_layout(
            (query_layout, key_cache_layout, value_cache_layout, block_tables_layout, context_lens_layout),
            (),
        )


def test_paged_attention_disallow_block_table_second_dim():
    """
    Feature: PagedAttention infer layout
    Description: Disallow sharding on block_tables second dim
    Expectation: Raise ValueError
    """
    mesh_shape = (2, 4)
    alias_name = ("dp", "mp")
    rank_list = list(range(8))

    query_layout, key_cache_layout, value_cache_layout, block_tables_layout, context_lens_layout = _build_layout(
        mesh_shape,
        alias_name,
        rank_list,
        ("dp", "mp", "None"),
        ("None", "None", "mp", "None"),
        ("dp", "mp"),
    )

    with pytest.raises(ValueError):
        _ = op.infer_layout(
            (query_layout, key_cache_layout, value_cache_layout, block_tables_layout, context_lens_layout),
            (),
        )


def test_paged_attention_q_seq_lens_unsharded():
    """
    Feature: PagedAttention infer layout
    Description: Allow q_seq_lens when unsharded
    Expectation: Success
    """
    mesh_shape = (2, 4)
    alias_name = ("dp", "mp")
    rank_list = list(range(8))

    query_layout, key_cache_layout, value_cache_layout, block_tables_layout, context_lens_layout = _build_layout(
        mesh_shape,
        alias_name,
        rank_list,
        ("dp", "mp", "None"),
        ("None", "None", "mp", "None"),
        ("dp", "None"),
    )
    q_seq_lens_layout = Layout(mesh_shape, alias_name, rank_list)("None",)

    output_layout = op.infer_layout(
        (query_layout, key_cache_layout, value_cache_layout, block_tables_layout, context_lens_layout,
         q_seq_lens_layout),
        (),
    )
    assert output_layout.tensor_map == (1, 0, -1)


def test_paged_attention_q_seq_lens_sharded():
    """
    Feature: PagedAttention infer layout
    Description: Disallow q_seq_lens sharding
    Expectation: Raise ValueError
    """
    mesh_shape = (2, 4)
    alias_name = ("dp", "mp")
    rank_list = list(range(8))

    query_layout, key_cache_layout, value_cache_layout, block_tables_layout, context_lens_layout = _build_layout(
        mesh_shape,
        alias_name,
        rank_list,
        ("dp", "mp", "None"),
        ("None", "None", "mp", "None"),
        ("dp", "None"),
    )
    q_seq_lens_layout = Layout(mesh_shape, alias_name, rank_list)("dp",)

    with pytest.raises(ValueError):
        _ = op.infer_layout(
            (query_layout, key_cache_layout, value_cache_layout, block_tables_layout, context_lens_layout,
             q_seq_lens_layout),
            (),
        )


def test_paged_attention_optional_layouts_smoke():
    """
    Feature: PagedAttention infer layout
    Description: Optional layouts (attn_mask/quant) should not break infer
    Expectation: Success
    """
    mesh_shape = (2, 4)
    alias_name = ("dp", "mp")
    rank_list = list(range(8))

    query_layout, key_cache_layout, value_cache_layout, block_tables_layout, context_lens_layout = _build_layout(
        mesh_shape,
        alias_name,
        rank_list,
        ("dp", "mp", "None"),
        ("None", "None", "mp", "None"),
        ("dp", "None"),
    )
    q_seq_lens_layout = Layout(mesh_shape, alias_name, rank_list)("None",)
    attn_mask_layout = Layout(mesh_shape, alias_name, rank_list)("None", "None")
    quant_scale_layout = Layout(mesh_shape, alias_name, rank_list)("None", "None")

    output_layout = op.infer_layout(
        (query_layout, key_cache_layout, value_cache_layout, block_tables_layout, context_lens_layout,
         q_seq_lens_layout, attn_mask_layout, quant_scale_layout),
        (),
    )
    assert output_layout.tensor_map == (1, 0, -1)
