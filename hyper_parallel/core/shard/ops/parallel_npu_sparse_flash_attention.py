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
"""Distributed implementation for npu_sparse_flash_attention operator."""
import copy
from typing import Callable, Optional, Tuple

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp
from .parallel_npu_dense_lightning_indexer_softmax_lse import _adjust_bsnd_key, _adjust_tnd_seq_lens

_MAX_INT64 = 9223372036854775807

# Maps layout_str -> tensor role -> {dim_index: dim_label} for replicated-dim checks.
# 'q' = query, 'k' = key, 'v' = value, 'si' = sparse_indices.
# N1 (head num of query) is forbidden from sharding due to severe performance impact.
_REPLICATED_DIMS = {
    'BSND': {
        'q': {2: 'N1', 3: 'D'},
        'k': {1: 'S2', 2: 'N2', 3: 'D'},
        'v': {1: 'S2', 2: 'N2', 3: 'D'},
        'si': {2: 'N2', 3: 'sparse_size'},
    },
    'TND': {
        'q': {1: 'N1', 2: 'D'},
        'k': {1: 'N2', 2: 'D'},
        'v': {1: 'N2', 2: 'D'},
        'si': {1: 'N2', 2: 'sparse_size'},
    },
}


def _normalize_sfa_args(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        block_table=None,
        actual_seq_lengths_query=None,
        actual_seq_lengths_kv=None,
        query_rope=None,
        key_rope=None,
        sparse_block_size=1,
        layout_query='BSND',
        layout_kv='BSND',
        sparse_mode=3,
        pre_tokens=_MAX_INT64,
        next_tokens=_MAX_INT64,
        attention_mode=2,
        return_softmax_lse=False):
    """Normalize positional and keyword arguments into a canonical positional tuple.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        sparse_indices: Sparse index tensor (int32).
        scale_value: Scaling factor (float).
        block_table: Optional PageAttention block mapping table.
        actual_seq_lengths_query: Actual query sequence lengths per batch.
        actual_seq_lengths_kv: Actual KV sequence lengths per batch.
        query_rope: Optional MLA query rope tensor.
        key_rope: Optional MLA key rope tensor.
        sparse_block_size: Block size for sparse computation.
        layout_query: Query layout string ('BSND' or 'TND').
        layout_kv: KV layout string ('BSND', 'TND', or 'PA_BSND').
        sparse_mode: Sparse attention mode.
        pre_tokens: Preceding token window size.
        next_tokens: Following token window size.
        attention_mode: Attention mode (0 or 2 for MLA-absorb).
        return_softmax_lse: Whether to return softmax max/sum.

    Returns:
        tuple: (positional_args_tuple, empty_kwargs_dict)
    """
    return (
        query, key, value, sparse_indices, scale_value,
        block_table, actual_seq_lengths_query, actual_seq_lengths_kv,
        query_rope, key_rope, sparse_block_size,
        layout_query, layout_kv, sparse_mode,
        pre_tokens, next_tokens, attention_mode, return_softmax_lse,
    ), {}


def _to_local(t):
    """Extract the local tensor from a DTensor input (None passes through).

    Every input except actual_seq_lengths must be a DTensor so its layout can be
    inferred; passing a plain tensor raises AttributeError here by design.
    """
    if t is None:
        return None
    return t.to_local()


def _to_local_seq_len(t):
    """Extract the local tensor from an actual_seq_lengths input.

    These are built inside the network and are not guaranteed to be DTensors, so
    a plain tensor (or None) is passed through unchanged.
    """
    if isinstance(t, DTensor):
        return t.to_local()
    return t


class SparseFlashAttentionDistributedOp(DistributedOp):
    """Distributed operator for npu_sparse_flash_attention.

    Supports BSND and TND input layouts on both MindSpore
    and PyTorch / torch_npu backends.

    Both frameworks provide built-in forward and backward implementations;
    this class handles only the distributed dispatch (layout inference and
    optional TND+CP sequence-length adjustment).

    Output shapes relative to inputs:
      - BSND: query (B, S1, N1, D) → attention_out (B, S1, N1, D),
              softmax_max/sum (B, N2, S1, N1/N2)
      - TND:  query (T1, N1, D)    → attention_out (T1, N1, D),
              softmax_max/sum (N2, T1, N1/N2)

    Sharding constraints:
      - N1 (query head dim) must be replicated — TP on this dim is forbidden
        due to severe performance impact.
      - Key/value S2 (or T2), N2, and D dims must be replicated.
      - sparse_indices N2 and sparse_size dims must be replicated.
      - PA_BSND layout is not supported in distributed mode.

    Context parallelism:
      - BSND+CP: k, v, and key_rope are sliced to the causal window
        ``[:, :S1_local*(split_id+1), :, :]`` before calling the kernel, matching the
        MindFormers adjust_bsnd_input logic.  sparse_indices from lightning_indexer are
        generated with the same truncation, so they remain valid for the sliced k.
      - TND+CP: adjusts actual_seq_lengths_query/kv per rank using
        _adjust_tnd_seq_lens (same logic as dsa_attention.py).
    """

    @staticmethod
    def _infer_softmax_layout(q_layout: Layout, layout_str: str) -> Layout:
        """Build the output layout for softmax_max and softmax_sum.

        BSND: query (B, S1, N1, D) → softmax (B, N2, S1, N1/N2)
              tensor_map: (q_tm[0], -1, q_tm[1], -1)
        TND:  query (T1, N1, D)    → softmax (N2, T1, N1/N2)
              tensor_map: (-1, q_tm[0], -1)

        N2 and N1/N2 are always replicated because N2=1 and N1 is forbidden
        from sharding.

        Args:
            q_layout: Layout of the query input.
            layout_str: 'BSND' or 'TND'.

        Returns:
            Layout for softmax_max / softmax_sum.
        """
        q_tm = q_layout.tensor_map
        out_layout = Layout.from_device_mesh(q_layout.mesh)
        if layout_str == 'BSND':
            out_tm = (q_tm[0], -1, q_tm[1], -1)
        else:
            out_tm = (-1, q_tm[0], -1)
        out_layout.set_tensor_map(out_tm)
        out_layout.tensor_map_to_placement()
        return out_layout

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """Extract local tensors and build the layout cache.

        Args:
            args: Positional arguments (may contain DTensors).
            kwargs: Keyword arguments.

        Returns:
            tuple: (local_args, local_kwargs, cache_values) where
                local_args = (query_local, key_local, value_local,
                              sparse_indices_local, scale_value),
                local_kwargs contains all remaining arguments,
                cache_values = [q_layout, k_layout, v_layout, si_layout, layout_query_str].
        """
        norm_args, _ = _normalize_sfa_args(*args, **kwargs)
        query = norm_args[0]
        key = norm_args[1]
        value = norm_args[2]
        sparse_indices = norm_args[3]
        scale_value = norm_args[4]
        layout_query_str = norm_args[11]

        local_args = (
            _to_local(query),
            _to_local(key),
            _to_local(value),
            _to_local(sparse_indices),
            scale_value,
        )
        local_kwargs = {
            'block_table': _to_local(norm_args[5]),
            'actual_seq_lengths_query': _to_local_seq_len(norm_args[6]),
            'actual_seq_lengths_kv': _to_local_seq_len(norm_args[7]),
            'query_rope': _to_local(norm_args[8]),
            'key_rope': _to_local(norm_args[9]),
            'sparse_block_size': norm_args[10],
            'layout_query': norm_args[11],
            'layout_kv': norm_args[12],
            'sparse_mode': norm_args[13],
            'pre_tokens': norm_args[14],
            'next_tokens': norm_args[15],
            'attention_mode': norm_args[16],
            'return_softmax_lse': norm_args[17],
        }

        cache_values = [
            query.layout,
            key.layout,
            value.layout,
            sparse_indices.layout,
            layout_query_str,
        ]
        return local_args, local_kwargs, cache_values

    @staticmethod
    def _validate_input_layouts(
            q_layout: Layout,
            k_layout: Layout,
            v_layout: Layout,
            si_layout: Layout,
            layout_str: str,
    ) -> None:
        """Validate sharding constraints for all input tensors.

        BSND rules (shapes: (B,S1,N1,D) / (B,S2,N2,D) / (B,S2,N2,D) / (B,S1,N2,sparse_size)):
          - N1 (dim 2) and D (dim 3) of query must be replicated.
          - S2 (dim 1), N2 (dim 2), D (dim 3) of key and value must be replicated.
          - N2 (dim 2) and sparse_size (dim 3) of sparse_indices must be replicated.
          - B sharding of key, value, and sparse_indices must match query.
          - S1 sharding of sparse_indices must match query.

        TND rules (shapes: (T1,N1,D) / (T2,N2,D) / (T2,N2,D) / (T1,N2,sparse_size)):
          - N1 (dim 1) and D (dim 2) of query must be replicated.
          - N2 (dim 1) and D (dim 2) of key and value must be replicated.
          - N2 (dim 1) and sparse_size (dim 2) of sparse_indices must be replicated.
          - T2 sharding of key and value must match.
          - T1 sharding of sparse_indices must match query.

        PA_BSND is not supported in distributed mode.

        Args:
            q_layout: Layout of query.
            k_layout: Layout of key.
            v_layout: Layout of value.
            si_layout: Layout of sparse_indices.
            layout_str: 'BSND' or 'TND'.

        Raises:
            ValueError: If layout_str is 'PA_BSND', if any required dimension is
                sharded, or if batch/sequence consistency constraints are violated.
        """
        if layout_str == 'PA_BSND':
            raise ValueError(
                "For npu_sparse_flash_attention, PA_BSND layout is not supported "
                "in distributed mode."
            )

        op = "npu_sparse_flash_attention"
        q_tm = q_layout.tensor_map
        k_tm = k_layout.tensor_map
        v_tm = v_layout.tensor_map
        si_tm = si_layout.tensor_map
        tms = {
            'q': (q_tm, 'query'),
            'k': (k_tm, 'key'),
            'v': (v_tm, 'value'),
            'si': (si_tm, 'sparse_indices'),
        }
        for role, dims in _REPLICATED_DIMS.get(layout_str, {}).items():
            tm_entry = tms.get(role)
            if tm_entry is None:
                continue
            tm, tensor_name = tm_entry
            for dim, label in dims.items():
                if tm[dim] != -1:
                    raise ValueError(
                        f"For {op}, {label} (dim {dim}) of {tensor_name} should be replicated, "
                        f"but got tensor_map={tm}"
                    )

        if layout_str == 'BSND':
            if q_tm[0] != k_tm[0]:
                raise ValueError(
                    f"For {op}, B (dim 0) sharding of key should match query, "
                    f"but got query={q_tm[0]}, key={k_tm[0]}"
                )
            if q_tm[0] != v_tm[0]:
                raise ValueError(
                    f"For {op}, B (dim 0) sharding of value should match query, "
                    f"but got query={q_tm[0]}, value={v_tm[0]}"
                )
            if q_tm[0] != si_tm[0]:
                raise ValueError(
                    f"For {op}, B (dim 0) sharding of sparse_indices should match query, "
                    f"but got query={q_tm[0]}, sparse_indices={si_tm[0]}"
                )
            if q_tm[1] != si_tm[1]:
                raise ValueError(
                    f"For {op}, S1 (dim 1) sharding of sparse_indices should match query, "
                    f"but got query={q_tm[1]}, sparse_indices={si_tm[1]}"
                )
        else:  # TND
            if k_tm[0] != v_tm[0]:
                raise ValueError(
                    f"For {op}, T2 (dim 0) sharding of value should match key, "
                    f"but got key={k_tm[0]}, value={v_tm[0]}"
                )
            if q_tm[0] != si_tm[0]:
                raise ValueError(
                    f"For {op}, T1 (dim 0) sharding of sparse_indices should match query, "
                    f"but got query={q_tm[0]}, sparse_indices={si_tm[0]}"
                )

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """Infer output layouts for all three outputs.

        Rules:
            1. PA_BSND layout is rejected.
            2. Partial inputs are not allowed on any of the four primary tensors.
            3. Sharding constraints are validated (see _validate_input_layouts).
            4. attention_out inherits query layout (deep copy).
            5. softmax_max and softmax_sum share the same layout derived from
               query layout with N2 and N1/N2 dims always replicated.
            6. All three output layouts are independent deep copies.

        Args:
            cache_values: [q_layout, k_layout, v_layout, si_layout, layout_str]

        Returns:
            tuple: ((attn_layout, softmax_max_layout, softmax_sum_layout), None)

        Raises:
            ValueError: If PA_BSND layout, any input has Partial status, or
                sharding constraints are violated.
        """
        q_layout = cache_values[0]
        k_layout = cache_values[1]
        v_layout = cache_values[2]
        si_layout = cache_values[3]
        layout_str = cache_values[4]

        self._check_partial_inputs([q_layout, k_layout, v_layout, si_layout])
        self._validate_input_layouts(q_layout, k_layout, v_layout, si_layout, layout_str)

        attn_layout = copy.deepcopy(q_layout)
        softmax_layout = self._infer_softmax_layout(q_layout, layout_str)
        return (attn_layout, softmax_layout, copy.deepcopy(softmax_layout)), None

    def get_expand_impl(  # pylint: disable=W0237
            self,
            func: Optional[Callable],
            infer_result: tuple,
            cache_values: list,
            extra_args: Optional[tuple] = None,
    ) -> Optional[Callable]:
        """Return a custom callable if context-parallel adjustment is needed.

        BSND (S1 not sharded): returns None — k/v are Replicated; sparse_indices
                 reference the full k directly.
        BSND+CP (S1 sharded): wraps func to slice k, v, and key_rope to the
                 causal window ``k[:, :S1_local*(split_id+1), :, :]`` before calling
                 the kernel.  Mirrors MindFormers adjust_bsnd_input logic, ensuring
                 that sparse_indices produced by lightning_indexer (which applies the
                 same truncation) remain valid.
        TND+CP: wraps func to adjust actual_seq_lengths_query/kv per rank,
                using the same algorithm as dsa_attention._sparse_flash_attention_forward.
        TND (no CP): wraps func to clamp seq_lens to local T1 slice.

        Args:
            func: The underlying op callable.
            infer_result: Output from infer_layout.
            cache_values: [q_layout, k_layout, v_layout, si_layout, layout_str].
            extra_args: Unused; kept for interface compatibility.

        Returns:
            Callable wrapper or None.
        """
        q_layout = cache_values[0]
        k_layout = cache_values[1]
        layout_str = cache_values[4]

        if layout_str == 'BSND':
            if q_layout.tensor_map[1] == -1:
                # S1 not sharded: pure DP or fully replicated.
                # k/v are Replicate on the CP dimension, so sparse_indices reference
                # the full k directly; no truncation needed.
                return None
            split_id = q_layout.get_split_id(1)

            def _bsnd_cp_impl(*args, **kwargs):
                local_q, local_k, local_v = args[0], args[1], args[2]
                s1_local = local_q.shape[1]
                sliced_k = _adjust_bsnd_key(local_k, s1_local, split_id)
                sliced_v = _adjust_bsnd_key(local_v, s1_local, split_id)
                key_rope = kwargs.get('key_rope')
                new_kwargs = (
                    {**kwargs, 'key_rope': _adjust_bsnd_key(key_rope, s1_local, split_id)}
                    if key_rope is not None else kwargs
                )
                return func(local_q, sliced_k, sliced_v, *args[3:], **new_kwargs)

            return _bsnd_cp_impl

        # TND: CP applies when q's T1 is sharded more finely than k's T2.
        q_split = q_layout.get_dim_split_num(0)
        k_split = k_layout.get_dim_split_num(0)
        split_id = q_layout.get_split_id(0) if q_split > k_split else 0
        cp_size = q_split // k_split if k_split > 0 else 1
        cp_rank = split_id % cp_size if cp_size > 1 else 0

        def _tnd_cp_impl(*args, **kwargs):
            local_q, local_k = args[0], args[1]
            qlen_tensor = kwargs.get('actual_seq_lengths_query')
            klen_tensor = kwargs.get('actual_seq_lengths_kv')
            if qlen_tensor is None or klen_tensor is None:
                return func(*args, **kwargs)
            adj_q, adj_k = _adjust_tnd_seq_lens(
                local_q, local_k, qlen_tensor, klen_tensor,
                cp_rank=cp_rank,
            )
            return func(*args, **{
                **kwargs,
                'actual_seq_lengths_query': adj_q,
                'actual_seq_lengths_kv': adj_k,
            })

        return _tnd_cp_impl
