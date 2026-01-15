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
"""Distributed implementation for PagedAttention operator."""

from hyper_parallel.core.layout import Layout
from .parallel_ops import DistributedOp


class PagedAttentionDistributedOp(DistributedOp):
    """Distributed implementation for PagedAttention operator."""

    @staticmethod
    def _ensure_same_mesh(left_layout, right_layout, left_name, right_name):
        if left_layout.mesh_shape != right_layout.mesh_shape:
            raise ValueError(
                f"{left_name} and {right_name} must have same mesh_shape, got "
                f"{left_layout.mesh_shape} and {right_layout.mesh_shape}."
            )

    @staticmethod
    def _check_not_sharded(layout, dim, name):
        if layout.alias_tensor_map[dim] != "None":
            raise ValueError(f"{name} dim {dim} can not be sharded, layout: {layout}")

    @staticmethod
    def _check_replicated_or_match(layout, dim, name, expect):
        mapping = layout.alias_tensor_map[dim]
        if isinstance(mapping, tuple):
            raise ValueError(f"{name} dim {dim} can not be sharded with multiple dims, layout: {layout}")
        if mapping not in ("None", expect):
            raise ValueError(
                f"{name} dim {dim} must be replicated or match query shard '{expect}', "
                f"got {mapping}, layout: {layout}"
            )

    def infer_layout(self, layouts, extra_args):
        print(f"[paged_attention] infer_layout called, layout_num={len(layouts)}")
        """
        Infer output layout for PagedAttention operator.

        Minimal support:
        - Allow sharding on query num_tokens and q_head_num dims.
        - Require key_cache/value_cache kv_head_num to shard consistently with query q_head_num.
        - Disallow sharding on block_size/num_blocks/max_num_blocks_per_query/head_dim.
        - Output layout inherits query layout.
        - context_lens/q_seq_lens may be replicated or shard with query num_tokens.
        """
        if len(layouts) < 4:
            raise ValueError(
                f"PagedAttention requires at least 4 input layouts (query/key_cache/value_cache/block_tables), "
                f"got {len(layouts)}"
            )

        query_layout = layouts[0]
        key_cache_layout = layouts[1]
        value_cache_layout = layouts[2]
        block_tables_layout = layouts[3]
        context_lens_layout = layouts[4] if len(layouts) > 4 else None
        q_seq_lens_layout = layouts[5] if len(layouts) > 5 else None

        if not query_layout or not key_cache_layout or not value_cache_layout or not block_tables_layout:
            raise ValueError("PagedAttention requires query/key_cache/value_cache/block_tables layouts.")

        self._ensure_same_mesh(query_layout, key_cache_layout, "query", "key_cache")
        self._ensure_same_mesh(query_layout, value_cache_layout, "query", "value_cache")
        self._ensure_same_mesh(query_layout, block_tables_layout, "query", "block_tables")
        if context_lens_layout:
            self._ensure_same_mesh(query_layout, context_lens_layout, "query", "context_lens")
        if q_seq_lens_layout:
            self._ensure_same_mesh(query_layout, q_seq_lens_layout, "query", "q_seq_lens")

        if len(query_layout.alias_tensor_map) != 3:
            raise ValueError(f"query layout rank should be 3, got {len(query_layout.alias_tensor_map)}")
        if len(key_cache_layout.alias_tensor_map) != 4:
            raise ValueError("key_cache layout rank should be 4")
        if len(value_cache_layout.alias_tensor_map) != 4:
            raise ValueError("value_cache layout rank should be 4")
        if len(block_tables_layout.alias_tensor_map) != 2:
            raise ValueError("block_tables layout rank should be 2")
        if context_lens_layout and len(context_lens_layout.alias_tensor_map) != 1:
            raise ValueError("context_lens layout rank should be 1")
        if q_seq_lens_layout and len(q_seq_lens_layout.alias_tensor_map) != 1:
            raise ValueError("q_seq_lens layout rank should be 1")

        query_map = query_layout.alias_tensor_map
        key_cache_map = key_cache_layout.alias_tensor_map
        value_cache_map = value_cache_layout.alias_tensor_map
        block_tables_map = block_tables_layout.alias_tensor_map

        # query: (num_tokens, q_head_num, head_dim)
        self._check_not_sharded(query_layout, 2, "query head_dim")

        # key/value cache: (num_blocks, block_size, kv_head_num, head_dim)
        self._check_not_sharded(key_cache_layout, 0, "key_cache num_blocks")
        self._check_not_sharded(key_cache_layout, 1, "key_cache block_size")
        self._check_not_sharded(value_cache_layout, 0, "value_cache num_blocks")
        self._check_not_sharded(value_cache_layout, 1, "value_cache block_size")
        if len(key_cache_map) >= 4:
            self._check_not_sharded(key_cache_layout, 3, "key_cache head_dim")
        if len(value_cache_map) >= 4:
            self._check_not_sharded(value_cache_layout, 3, "value_cache head_dim")

        # kv_head_num and q_head_num must shard consistently.
        if key_cache_map[2] != value_cache_map[2] or key_cache_map[2] != query_map[1]:
            raise ValueError(
                "kv_head_num and q_head_num must shard consistently, got "
                f"query: {query_map}, key_cache: {key_cache_map}, value_cache: {value_cache_map}."
            )

        # block_tables: (num_tokens, max_num_blocks_per_query)
        if block_tables_map[0] != query_map[0]:
            raise ValueError(
                "block_tables num_tokens dim must match query num_tokens sharding, "
                f"got block_tables: {block_tables_map}, query: {query_map}."
            )
        self._check_not_sharded(block_tables_layout, 1, "block_tables max_num_blocks_per_query")

        if context_lens_layout:
            self._check_replicated_or_match(context_lens_layout, 0, "context_lens", query_map[0])
        if q_seq_lens_layout:
            self._check_replicated_or_match(q_seq_lens_layout, 0, "q_seq_lens", query_map[0])

        output_layout = Layout(
            mesh_shape=query_layout.mesh_shape,
            alias_name=query_layout.alias_name,
            rank_list=query_layout.rank_list
        )
        return output_layout(*query_map)
