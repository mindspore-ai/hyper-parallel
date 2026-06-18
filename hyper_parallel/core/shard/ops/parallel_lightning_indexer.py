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
"""Distributed implementation for lightning_indexer operator."""
import copy
from typing import Callable, Optional, Tuple

from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.platform import get_platform
from .parallel_ops import DistributedOp
from .parallel_npu_dense_lightning_indexer_softmax_lse import (
    _adjust_bsnd_key,
    _adjust_tnd_seq_lens,
    _to_local_seq_len,
)

platform = get_platform()

_MAX_INT64 = 9223372036854775807

# Maps layout_str -> tensor role -> {dim_index: dim_label} for replicated-dim checks.
# 'q' = query, 'k' = key, 'w' = weights.
_REPLICATED_DIMS = {
    'BSND': {
        'q': {2: 'N1', 3: 'D'},
        'k': {1: 'S2', 2: 'N2', 3: 'D'},
        'w': {2: 'N1'},
    },
    'TND': {
        'q': {1: 'N1', 2: 'D'},
        'k': {1: 'N2', 2: 'D'},
        'w': {1: 'N1'},
    },
}


def _normalize_lightning_indexer_args(
        query,
        key,
        weights,
        actual_seq_lengths_query=None,
        actual_seq_lengths_key=None,
        block_table=None,
        layout_query='BSND',
        layout_key='BSND',
        sparse_count=2048,
        sparse_mode=3,
        pre_tokens=_MAX_INT64,
        next_tokens=_MAX_INT64,
        return_value=False):
    """Normalize positional and keyword arguments into a canonical positional tuple.

    Args:
        query: Query tensor.
        key: Key tensor.
        weights: Weight tensor.
        actual_seq_lengths_query: Cumulative query sequence lengths (TND only).
        actual_seq_lengths_key: Cumulative key sequence lengths (TND only).
        block_table: Block table for PageAttention (optional).
        layout_query: Input layout string for query, 'BSND' or 'TND'.
        layout_key: Input layout string for key, 'BSND', 'TND', or 'PA_BSND'.
        sparse_count: Number of top-k blocks to retain.
        sparse_mode: Sparse attention mode (0=defaultMask, 3=rightDownCausal).
        pre_tokens: Sparse pre-tokens count.
        next_tokens: Sparse next-tokens count.
        return_value: Whether to output sparse_values.

    Returns:
        tuple: (positional_args_tuple, keyword_args_dict)
    """
    local_args = (query, key, weights)
    local_kwargs = {
        'actual_seq_lengths_query': actual_seq_lengths_query,
        'actual_seq_lengths_key': actual_seq_lengths_key,
        'block_table': block_table,
        'layout_query': layout_query,
        'layout_key': layout_key,
        'sparse_count': sparse_count,
        'sparse_mode': sparse_mode,
        'pre_tokens': pre_tokens,
        'next_tokens': next_tokens,
        'return_value': return_value,
    }
    return local_args, local_kwargs


class LightningIndexerDistributedOp(DistributedOp):
    """Distributed operator for MindSpore built-in lightning_indexer.

    LightningIndexer computes the top-k most relevant key positions for each query token
    in sparse attention. It is a MindSpore built-in op (accessed via
    ``ops.lightning_indexer``), not a custom op, so only the distributed sharding
    logic is implemented here.

    Supports BSND and TND input layouts on both MindSpore and PyTorch platforms.

    Output shapes:
      - BSND: query (B, S1, N1, D) → outputs (B, S1, N2, sparse_count)
      - TND:  query (T1, N1, D)    → outputs (T1, N2, sparse_count)

    Context parallelism (CP) is handled in ``get_expand_impl``:
      - BSND+CP: key S2 is sliced to the causal window for each rank.
      - TND+CP:  actual_seq_qlen / actual_seq_klen are adjusted per rank.

    """

    @staticmethod
    def _infer_output_layout(q_layout: Layout, layout_str: str) -> Layout:
        """Build the output layout for both sparse outputs from the query layout.

        BSND: input (B, S1, N1, D) → output (B, S1, N2, sparse_count)
              tensor_map: (q_tm[0], q_tm[1], -1, -1)
        TND:  input (T1, N1, D)    → output (T1, N2, sparse_count)
              tensor_map: (q_tm[0], -1, -1)

        N2 is always replicated (key's head dimension constraint).
        sparse_count is always replicated (int scalar attribute).

        Args:
            q_layout: Layout of the query input.
            layout_str: 'BSND' or 'TND'.

        Returns:
            Layout for the output tensors.
        """
        q_tm = q_layout.tensor_map
        out_layout = Layout.from_device_mesh(q_layout.mesh)
        if layout_str == 'BSND':
            out_tm = (q_tm[0], q_tm[1], -1, -1)
        else:
            out_tm = (q_tm[0], -1, -1)
        out_layout.set_tensor_map(out_tm)
        out_layout.tensor_map_to_placement()
        return out_layout

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """Extract local tensors and build the layout cache.

        Args:
            args: Positional arguments (may contain DTensors).
            kwargs: Keyword arguments.

        Returns:
            tuple: (local_args, local_kwargs, cache_values) where cache_values is
                [q_layout, k_layout, w_layout, layout_str].
        """
        norm_args, local_kwargs = _normalize_lightning_indexer_args(*args, **kwargs)

        query_index, key_index, weights = norm_args[0], norm_args[1], norm_args[2]
        layout_str = local_kwargs['layout_query']  # layout_query

        local_kwargs['actual_seq_lengths_query'] = _to_local_seq_len(
            local_kwargs.get('actual_seq_lengths_query'))
        local_kwargs['actual_seq_lengths_key'] = _to_local_seq_len(
            local_kwargs.get('actual_seq_lengths_key'))

        local_args = (query_index.to_local(), key_index.to_local(), weights.to_local())

        cache_values = [query_index.layout, key_index.layout, weights.layout, layout_str]
        return local_args, local_kwargs, cache_values

    @staticmethod
    def _validate_input_layouts(
            q_layout: Layout,
            k_layout: Layout,
            w_layout: Layout,
            layout_str: str,
    ) -> None:
        """Validate sharding constraints for all input tensors.

        BSND rules (query/key/weights shapes: (B,S1,N1,D) / (B,S2,N2,D) / (B,S1,N1)):
          - N1 (dim 2) and D (dim 3) of query must be replicated.
          - S2 (dim 1), N2 (dim 2), D (dim 3) of key must be replicated.
          - B sharding of query and key must be identical.
          - B and S1 sharding of weights must match query; N1 must be replicated.

        TND rules (query/key/weights shapes: (T1,N1,D) / (T2,N2,D) / (T1,N1)):
          - N1 (dim 1) and D (dim 2) of query must be replicated.
          - N2 (dim 1) and D (dim 2) of key must be replicated.
          - T1 sharding of weights must match query; N1 must be replicated.

        Args:
            q_layout: Layout of query.
            k_layout: Layout of key.
            w_layout: Layout of weights.
            layout_str: 'BSND' or 'TND'.

        Raises:
            ValueError: If any constraint is violated.
        """
        op = "lightning_indexer"
        q_tm = q_layout.tensor_map
        k_tm = k_layout.tensor_map
        w_tm = w_layout.tensor_map
        tms = {'q': (q_tm, 'query'), 'k': (k_tm, 'key'), 'w': (w_tm, 'weights')}
        for role, dims in _REPLICATED_DIMS.get(layout_str, {}).items():
            tm, tensor_name = tms[role]
            for dim, label in dims.items():
                if tm[dim] != -1:
                    raise ValueError(
                        f"For {op}, {label} (dim {dim}) of {tensor_name} should be replicated, "
                        f"but got tensor_map={tm}"
                    )
        if layout_str == 'BSND':
            if q_tm[0] != k_tm[0]:
                raise ValueError(
                    f"For {op}, B (dim 0) sharding of query and key should match, "
                    f"but got query={q_tm[0]}, key={k_tm[0]}"
                )
            if w_tm[0] != q_tm[0]:
                raise ValueError(
                    f"For {op}, B (dim 0) sharding of weights should match query, "
                    f"but got weights={w_tm[0]}, query={q_tm[0]}"
                )
            if w_tm[1] != q_tm[1]:
                raise ValueError(
                    f"For {op}, S1 (dim 1) sharding of weights should match query, "
                    f"but got weights={w_tm[1]}, query={q_tm[1]}"
                )
        else:  # TND
            if w_tm[0] != q_tm[0]:
                raise ValueError(
                    f"For {op}, T1 (dim 0) sharding of weights should match query, "
                    f"but got weights={w_tm[0]}, query={q_tm[0]}"
                )

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """Infer output layouts for sparse_indices and sparse_values outputs.

        Rules:
            1. No Partial inputs are allowed on any of the three input tensors.
            2. Input sharding constraints are validated per layout_str (see
               ``_validate_input_layouts`` for the full rule set).
            3. Output tensor shape depends on layout_str:
               - BSND: query (B, S1, N1, D) → outputs (B, S1, N2, sparse_count).
                 B and S1 sharding are inherited from query;
                 N2 and sparse_count are always replicated.
               - TND: query (T1, N1, D) → outputs (T1, N2, sparse_count).
                 T1 sharding is inherited from query;
                 N2 and sparse_count are always replicated.
            4. Both sparse_indices and sparse_values outputs share the same layout
               (independent deep copies so callers can mutate them safely).

        Args:
            cache_values: [q_layout, k_layout, w_layout, layout_str]

        Returns:
            tuple: ((indices_layout, values_layout), None)

        Raises:
            ValueError: If any input has Partial status, or sharding constraints
                are violated.
        """
        q_layout = cache_values[0]
        k_layout = cache_values[1]
        w_layout = cache_values[2]
        layout_str = cache_values[3]

        self._check_partial_inputs([q_layout, k_layout, w_layout])
        self._validate_input_layouts(q_layout, k_layout, w_layout, layout_str)

        out_layout = self._infer_output_layout(q_layout, layout_str)
        return (out_layout, copy.deepcopy(out_layout)), None

    def get_expand_impl(  # pylint: disable=W0237
            self,
            func: Optional[Callable],
            infer_result: tuple,
            cache_values: list,
            extra_args: Optional[tuple] = None,
    ) -> Optional[Callable]:
        """Return a custom callable if context-parallel adjustments are needed.

        BSND+CP: wraps ``func`` to slice key's S2 to the causal window.
        TND+CP:  wraps ``func`` to adjust actual_seq_qlen/klen per rank.
        No CP:   returns None (dispatcher calls ``func`` directly).

        Args:
            func: The underlying op callable.
            infer_result: Output from ``infer_layout``.
            cache_values: [q_layout, k_layout, w_layout, layout_str].
            extra_args: Unused; kept for interface compatibility.

        Returns:
            Callable wrapper or None.
        """
        q_layout = cache_values[0]
        k_layout = cache_values[1]
        layout_str = cache_values[3]

        if layout_str == 'BSND':
            # S1 is dim 1 of query; if not sharded, no CP adjustment needed.
            if q_layout.tensor_map[1] == -1:
                return None
            split_id = q_layout.get_split_id(1)

            def _bsnd_cp_impl(*args, **kwargs):
                local_q, local_k = args[0], args[1]
                sliced_k = _adjust_bsnd_key(local_k, local_q.shape[1], split_id)
                return func(local_q, sliced_k, *args[2:], **kwargs)

            return _bsnd_cp_impl

        # TND: DP always requires seq_len adjustment; CP additionally
        # requires token-level offset adjustment.
        dp_size = k_layout.get_dim_split_num(0)  # DP splits on k's T2
        split_id = q_layout.get_split_id(0)
        cp_size = (q_layout.get_dim_split_num(0) // dp_size
                   if dp_size > 0 else 1)
        cp_rank = split_id % cp_size if cp_size > 1 else 0

        def _tnd_impl(*args, **kwargs):
            local_q, local_k = args[0], args[1]

            qlen_tensor = kwargs.get('actual_seq_lengths_query')
            klen_tensor = kwargs.get('actual_seq_lengths_key')

            if qlen_tensor is None or klen_tensor is None:
                return func(*args, **kwargs)

            adj_q, adj_k = _adjust_tnd_seq_lens(
                local_q, local_k, qlen_tensor, klen_tensor,
                cp_rank=cp_rank,
            )

            return func(*args, **{**kwargs, 'actual_seq_lengths_query': adj_q,
                                 'actual_seq_lengths_key': adj_k})

        return _tnd_impl
