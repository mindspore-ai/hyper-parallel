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
"""Distributed implementation for npu_dense_lightning_indexer_softmax_lse operator."""
import copy
from typing import Callable, Optional, Tuple

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType
from .parallel_ops import DistributedOp

platform = get_platform()

_MAX_INT64 = 9223372036854775807


def _to_local_seq_len(t):
    """Extract the local tensor from an actual_seq_lengths input.

    These are built inside the network and are not guaranteed to be DTensors, so
    a plain tensor (or None) is passed through unchanged. All other inputs must be
    DTensors (they go through ``.to_local()`` directly) so their layout can be inferred.
    """
    if isinstance(t, DTensor):
        return t.to_local()
    return t

# Maps layout_str -> tensor role -> {dim_index: dim_label} for replicated-dim checks.
# 'q' = query_index, 'k' = key_index, 'w' = weights.
_REPLICATED_DIMS = {
    'BSND': {
        'q': {2: 'N1index', 3: 'D'},
        'k': {1: 'S2', 2: 'N2index', 3: 'D'},
        'w': {2: 'N1index'},
    },
    'TND': {
        'q': {1: 'N1index', 2: 'D'},
        'k': {1: 'N2index', 2: 'D'},
        'w': {1: 'N1index'},
    },
}


def _normalize_softmax_lse_args(
        query_index,
        key_index,
        weights,
        actual_seq_qlen=None,
        actual_seq_klen=None,
        layout='BSND',
        sparse_mode=3,
        pre_tokens=_MAX_INT64,
        next_tokens=_MAX_INT64):
    """Normalize positional and keyword arguments into a canonical positional tuple.

    Args:
        query_index: Query index tensor.
        key_index: Key index tensor.
        weights: Weight tensor.
        actual_seq_qlen: Cumulative query sequence lengths (TND only).
        actual_seq_klen: Cumulative key sequence lengths (TND only).
        layout: Input layout string, 'BSND' or 'TND'.
        sparse_mode: Sparse attention mode (only mode 3 is supported).
        pre_tokens: Sparse pre-tokens count.
        next_tokens: Sparse next-tokens count.

    Returns:
        tuple: (positional_args_tuple, empty_kwargs_dict)
    """
    return (
        query_index, key_index, weights,
        actual_seq_qlen, actual_seq_klen,
        layout, sparse_mode, pre_tokens, next_tokens,
    ), {}


def _adjust_bsnd_key(local_k, local_q_s1: int, split_id: int):
    """Slice key's S2 dimension to the causal window for BSND+CP.

    For rightDownCausal (sparse_mode=3) with context parallelism, rank ``split_id``
    holds the ``split_id``-th slice of S1. Its valid key window is
    ``k[:, :S1_local*(split_id+1), :, :]``.

    Args:
        local_k: Local key tensor of shape (B, S2, N2index, D).
        local_q_s1: Local S1 length of query on this rank.
        split_id: This rank's position in the CP group along the S dimension.

    Returns:
        Sliced key tensor of shape (B, local_q_s1*(split_id+1), N2index, D).
    """
    return local_k[:, :local_q_s1 * (split_id + 1), :, :]


def _adjust_tnd_seq_lens(
        local_q,
        local_k,
        actual_seq_qlen,
        actual_seq_klen,
        cp_rank: int = 0,
) -> Tuple:
    """Adjust actual_seq_qlen/klen for TND layout with CP token-level offset.

    DP batch slicing is handled by DTensor Shard(0) — the caller already
    calls ``.to_local()`` to obtain the DP-local slice before invoking this
    function.

    Uses Tensor native methods (clamp, roll, cumsum) where both frameworks
    agree, and ``platform.relu`` where the APIs differ.

    Args:
        local_q: Local query tensor (T1_local, N1index, D).
        local_k: Local key tensor (T2_local, N2index, D).
        actual_seq_qlen: DP-local cumulative query sequence lengths (int32 Tensor).
        actual_seq_klen: DP-local cumulative key sequence lengths (int32 Tensor).
        cp_rank: CP rank index (0-based, for token offset within batch).

    Returns:
        tuple[Tensor, Tensor]: (adj_qlen, adj_klen) for this rank's local shard.
    """
    slice_tq = local_q.shape[0]
    slice_tk = local_k.shape[0]
    offset_q = slice_tq * cp_rank

    new_actual_seq_qlen = platform.tensor_type_cast(
        (actual_seq_qlen - offset_q).clamp(0, slice_tq), 'int32')

    new_actual_seq_klen = (
        actual_seq_klen
        - platform.relu(actual_seq_qlen - offset_q)
        + new_actual_seq_qlen
    )

    prev_seq_klen = new_actual_seq_klen.roll(1, 0)
    prev_seq_klen[0] = 0
    new_actual_seq_klen = platform.relu(new_actual_seq_klen - prev_seq_klen).cumsum(0)
    new_actual_seq_klen[-1] = slice_tk

    return new_actual_seq_qlen, new_actual_seq_klen


class NpuDenseLightningIndexerSoftmaxLseDistributedOp(DistributedOp):
    """Distributed operator for npu_dense_lightning_indexer_softmax_lse.

    Supports BSND and TND input layouts on both MindSpore (DFunction path)
    and PyTorch (torch_npu path).

    Output shapes differ from inputs:
      - BSND: query (B, S1, N1, D) → outputs (B, N2index, S1)
      - TND:  query (T1, N1, D)    → outputs (N2index, T1)

    Context parallelism (CP) is handled in ``get_expand_impl``:
      - BSND+CP: key S2 is sliced to the causal window for each rank.
      - TND+CP:  actual_seq_qlen / actual_seq_klen are adjusted per rank.

    Platform differences in ``preprocess``:
      - MindSpore: all arguments must be positional (no kwargs).
      - PyTorch: optional arguments are passed as keyword arguments.
    """

    @staticmethod
    def _infer_output_layout(q_layout: Layout, layout_str: str) -> Layout:
        """Build the output layout for both softmax outputs from the query layout.

        BSND: input (B, S1, N1, D) → output (B, N2index, S1)
              tensor_map: (q_tm[0], -1, q_tm[1])
        TND:  input (T1, N1, D)    → output (N2index, T1)
              tensor_map: (-1, q_tm[0])

        N2index is always replicated; batch/sequence sharding is inherited.

        Args:
            q_layout: Layout of the query_index input.
            layout_str: 'BSND' or 'TND'.

        Returns:
            Layout for the output tensors.
        """
        q_tm = q_layout.tensor_map
        out_layout = Layout.from_device_mesh(q_layout.mesh)
        if layout_str == 'BSND':
            out_tm = (q_tm[0], -1, q_tm[1])
        else:
            out_tm = (-1, q_tm[0])
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
        norm_args, _ = _normalize_softmax_lse_args(*args, **kwargs)
        query_index, key_index, weights = norm_args[0], norm_args[1], norm_args[2]
        layout_str = norm_args[5]

        qlen_arg = _to_local_seq_len(norm_args[3])
        klen_arg = _to_local_seq_len(norm_args[4])

        if platform.platform_type == PlatformType.MINDSPORE:
            local_args = (
                query_index.to_local(),
                key_index.to_local(),
                weights.to_local(),
                qlen_arg,
                klen_arg,
                *norm_args[5:],
            )
            local_kwargs = {}
        else:
            local_args = (query_index.to_local(), key_index.to_local(), weights.to_local())
            local_kwargs = {
                'actual_seq_qlen': qlen_arg,
                'actual_seq_klen': klen_arg,
                'layout': norm_args[5],
                'sparse_mode': norm_args[6],
                'pre_tokens': norm_args[7],
                'next_tokens': norm_args[8],
            }

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
          - N1index (dim 2) and D (dim 3) of query_index must be replicated.
          - S2 (dim 1), N2index (dim 2), D (dim 3) of key_index must be replicated.
          - B sharding of query_index and key_index must be identical.
          - B and S1 sharding of weights must match query_index; N1index must be replicated.

        TND rules (query/key/weights shapes: (T1,N1,D) / (T2,N2,D) / (T1,N1)):
          - N1index (dim 1) and D (dim 2) of query_index must be replicated.
          - N2index (dim 1) and D (dim 2) of key_index must be replicated.
          - T1 sharding of weights must match query_index; N1index must be replicated.

        Args:
            q_layout: Layout of query_index.
            k_layout: Layout of key_index.
            w_layout: Layout of weights.
            layout_str: 'BSND' or 'TND'.

        Raises:
            ValueError: If any constraint is violated.
        """
        op = "npu_dense_lightning_indexer_softmax_lse"
        q_tm = q_layout.tensor_map
        k_tm = k_layout.tensor_map
        w_tm = w_layout.tensor_map
        tms = {'q': (q_tm, 'query_index'), 'k': (k_tm, 'key_index'), 'w': (w_tm, 'weights')}
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
                    f"For {op}, B (dim 0) sharding of query_index and key_index should match, "
                    f"but got query_index={q_tm[0]}, key_index={k_tm[0]}"
                )
            if w_tm[0] != q_tm[0]:
                raise ValueError(
                    f"For {op}, B (dim 0) sharding of weights should match query_index, "
                    f"but got weights={w_tm[0]}, query_index={q_tm[0]}"
                )
            if w_tm[1] != q_tm[1]:
                raise ValueError(
                    f"For {op}, S1 (dim 1) sharding of weights should match query_index, "
                    f"but got weights={w_tm[1]}, query_index={q_tm[1]}"
                )
        else:  # TND
            if w_tm[0] != q_tm[0]:
                raise ValueError(
                    f"For {op}, T1 (dim 0) sharding of weights should match query_index, "
                    f"but got weights={w_tm[0]}, query_index={q_tm[0]}"
                )

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """Infer output layouts for both softmax outputs.

        Rules:
            1. No Partial inputs are allowed on any of the three input tensors.
            2. Input sharding constraints are validated per layout_str (see
               ``_validate_input_layouts`` for the full rule set).
            3. Output tensor shape depends on layout_str:
               - BSND: query (B, S1, N1, D) → outputs (B, N2index, S1).
                 B and S1 sharding are inherited from query_index;
                 N2index is always replicated.
               - TND: query (T1, N1, D) → outputs (N2index, T1).
                 T1 sharding is inherited from query_index;
                 N2index is always replicated.
            4. Both softmax_max and softmax_sum outputs share the same layout
               (independent deep copies so callers can mutate them safely).

        Args:
            cache_values: [q_layout, k_layout, w_layout, layout_str]

        Returns:
            tuple: ((softmax_max_layout, softmax_sum_layout), None)

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
            if len(args) > 3:
                qlen_tensor = args[3]
                klen_tensor = args[4]
            else:
                qlen_tensor = kwargs.get('actual_seq_qlen')
                klen_tensor = kwargs.get('actual_seq_klen')

            if qlen_tensor is None or klen_tensor is None:
                return func(*args, **kwargs)

            adj_q, adj_k = _adjust_tnd_seq_lens(
                local_q, local_k, qlen_tensor, klen_tensor,
                cp_rank=cp_rank,
            )

            if len(args) > 3:  # MindSpore
                return func(local_q, local_k, args[2], adj_q, adj_k, *args[5:], **kwargs)
            return func(*args, **{**kwargs, 'actual_seq_qlen': adj_q,
                                 'actual_seq_klen': adj_k})

        return _tnd_impl
