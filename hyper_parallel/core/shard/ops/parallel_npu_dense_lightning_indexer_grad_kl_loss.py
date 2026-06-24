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
"""Distributed implementation for npu_dense_lightning_indexer_grad_kl_loss operator."""
import copy
from typing import Callable, Optional, Tuple

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType
from .parallel_ops import DistributedOp
from .parallel_npu_dense_lightning_indexer_softmax_lse import (
    _adjust_bsnd_key,
    _adjust_tnd_seq_lens,
)

platform = get_platform()

_MAX_INT64 = 9223372036854775807

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


def _normalize_dense_grad_kl_loss_args(
        query,
        key,
        query_index,
        key_index,
        weights,
        softmax_max,
        softmax_sum,
        softmax_max_index,
        softmax_sum_index,
        scale_value,
        query_rope=None,
        key_rope=None,
        actual_seq_qlen=None,
        actual_seq_klen=None,
        layout='BSND',
        sparse_mode=3,
        pre_tokens=_MAX_INT64,
        next_tokens=_MAX_INT64):
    """Normalize positional and keyword arguments into a canonical positional tuple.

    Args:
        query: Main attention query (Q).
        key: Main attention key (K).
        query_index: Lightning Indexer query input (Q̃).
        key_index: Lightning Indexer key input (K̃).
        weights: Lightning Indexer weight coefficient (W).
        softmax_max: Attention softmax max values (float32).
        softmax_sum: Attention softmax sum values (float32).
        softmax_max_index: Index attention softmax max (from softmax_lse, float32).
        softmax_sum_index: Index attention softmax sum (from softmax_lse, float32).
        scale_value: Scaling factor (float32 scalar).
        query_rope: Optional MLA query rope tensor.
        key_rope: Optional MLA key rope tensor.
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
        query, key, query_index, key_index, weights,
        softmax_max, softmax_sum, softmax_max_index, softmax_sum_index,
        scale_value,
        query_rope, key_rope,
        actual_seq_qlen, actual_seq_klen,
        layout, sparse_mode, pre_tokens, next_tokens,
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


class NpuDenseLightningIndexerGradKlLossDistributedOp(DistributedOp):
    """Distributed operator for npu_dense_lightning_indexer_grad_kl_loss.

    Supports BSND and TND input layouts on both MindSpore (DFunction path)
    and PyTorch (torch_npu path).

    Output shapes mirror corresponding input shapes:
      - d_query_index: same as query_index
      - d_key_index: same as key_index
      - d_weights: same as weights
      - loss: replicated 1-D scalar tensor (float32)

    Context parallelism (CP) is handled in ``get_expand_impl``:
      - BSND+CP: key, key_index, and key_rope S2 dimensions are sliced to the
        causal window for each rank.
      - TND+CP: actual_seq_qlen / actual_seq_klen are adjusted per rank.

    Platform differences in ``preprocess``:
      - MindSpore: all 18 arguments are positional (no kwargs).
      - PyTorch: the 10 mandatory positional tensor args are positional; optional
        tensors and scalars are passed as keyword arguments.
    """

    @staticmethod
    def _infer_output_layouts(
            q_index_layout: Layout,
            k_index_layout: Layout,
            w_layout: Layout,
            layout_str: str = 'BSND',
    ) -> Tuple[Layout, Layout, Layout, Layout]:
        """Build layouts for the four outputs.

        d_query_index inherits query_index layout; d_key_index inherits
        key_index layout but gains a CP Partial when context parallelism is
        active (each rank contributes only the gradient for key tokens in its
        causal window; an AllReduce across the CP group is required);
        d_weights inherits weights layout; loss is a 1-D scalar that requires
        partial reduction when batch or CP is sharded.

        Args:
            q_index_layout: Layout of query_index.
            k_index_layout: Layout of key_index.
            w_layout: Layout of weights.
            layout_str: Input layout format ('BSND' or 'TND').

        Returns:
            tuple: (d_query_index_layout, d_key_index_layout,
                    d_weights_layout, loss_layout)
        """
        loss_layout = Layout.from_device_mesh(q_index_layout.mesh)
        loss_layout.set_tensor_map((-1,))
        loss_layout.tensor_map_to_placement()
        mesh_dim_names = q_index_layout.mesh.mesh_dim_names
        ndim = len(mesh_dim_names)
        if k_index_layout.tensor_map[0] != -1:
            dp_axis_name = mesh_dim_names[ndim - 1 - k_index_layout.tensor_map[0]]
            loss_layout.set_partial_by_dev_axis(dp_axis_name, "sum")
        cp_axis_name = None
        if layout_str == 'BSND':
            if q_index_layout.tensor_map[1] != -1:
                cp_axis_name = mesh_dim_names[ndim - 1 - q_index_layout.tensor_map[1]]
                loss_layout.set_partial_by_dev_axis(cp_axis_name, "sum")
        else:
            # TND: CP applies when q's T1 split is finer than k's T1 split.
            # In TND, both DP and CP share the T1 dimension (T1 = total tokens).
            # tensor_map[1] is always N1index (replicated), so S1-based detection
            # used for BSND cannot be reused here.
            q_split = q_index_layout.get_dim_split_num(0)
            k_split = k_index_layout.get_dim_split_num(0)
            if q_split > k_split:
                tm0 = q_index_layout.tensor_map[0]
                if isinstance(tm0, tuple):
                    # T1 sharded on dp+cp; k only on dp → the other element is cp.
                    k_tm0 = k_index_layout.tensor_map[0]
                    for idx in tm0:
                        if idx != k_tm0:
                            cp_axis_name = mesh_dim_names[ndim - 1 - idx]
                            break
                elif tm0 != -1:
                    cp_axis_name = mesh_dim_names[ndim - 1 - tm0]
                if cp_axis_name is not None:
                    loss_layout.set_partial_by_dev_axis(cp_axis_name, "sum")
        d_k_index_layout = copy.deepcopy(k_index_layout)
        if cp_axis_name is not None:
            # Each CP rank only covers its causal window; sum across CP group
            # to obtain the complete d_key_index gradient.
            d_k_index_layout.set_partial_by_dev_axis(cp_axis_name, "sum")
        return (
            copy.deepcopy(q_index_layout),
            d_k_index_layout,
            copy.deepcopy(w_layout),
            loss_layout,
        )

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """Extract local tensors and build the layout cache.

        Args:
            args: Positional arguments (may contain DTensors).
            kwargs: Keyword arguments.

        Returns:
            tuple: (local_args, local_kwargs, cache_values) where cache_values is
                [q_index_layout, k_index_layout, w_layout, layout_str].
        """
        norm_args, _ = _normalize_dense_grad_kl_loss_args(*args, **kwargs)
        query_index = norm_args[2]
        key_index = norm_args[3]
        weights = norm_args[4]
        scale_value = norm_args[9]
        layout_str = norm_args[14]

        if platform.platform_type == PlatformType.MINDSPORE:
            local_args = (
                _to_local(norm_args[0]), _to_local(norm_args[1]),
                _to_local(query_index), _to_local(key_index), _to_local(weights),
                _to_local(norm_args[5]), _to_local(norm_args[6]),
                _to_local(norm_args[7]), _to_local(norm_args[8]),
                scale_value,
                _to_local(norm_args[10]), _to_local(norm_args[11]),
                _to_local_seq_len(norm_args[12]), _to_local_seq_len(norm_args[13]),
                layout_str, norm_args[15], norm_args[16], norm_args[17],
            )
            local_kwargs = {}
        else:
            local_args = (
                _to_local(norm_args[0]), _to_local(norm_args[1]),
                _to_local(query_index), _to_local(key_index), _to_local(weights),
                _to_local(norm_args[5]), _to_local(norm_args[6]),
                _to_local(norm_args[7]), _to_local(norm_args[8]),
                scale_value,
            )
            local_kwargs = {
                'query_rope': _to_local(norm_args[10]),
                'key_rope': _to_local(norm_args[11]),
                'actual_seq_qlen': _to_local_seq_len(norm_args[12]),
                'actual_seq_klen': _to_local_seq_len(norm_args[13]),
                'layout': layout_str,
                'sparse_mode': norm_args[15],
                'pre_tokens': norm_args[16],
                'next_tokens': norm_args[17],
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
        """Validate sharding constraints for query_index, key_index, and weights.

        BSND rules (shapes: (B,S1,N1index,D) / (B,S2,N2index,D) / (B,S1,N1index)):
          - N1index (dim 2) and D (dim 3) of query_index must be replicated.
          - S2 (dim 1), N2index (dim 2), D (dim 3) of key_index must be replicated.
          - B sharding of query_index and key_index must be identical.
          - B (dim 0) and S1 (dim 1) of weights must match query_index;
            N1index (dim 2) must be replicated.

        TND rules (shapes: (T1,N1index,D) / (T2,N2index,D) / (T1,N1index)):
          - N1index (dim 1) and D (dim 2) of query_index must be replicated.
          - N2index (dim 1) and D (dim 2) of key_index must be replicated.
          - T1 (dim 0) of weights must match query_index; N1index (dim 1) replicated.

        Args:
            q_layout: Layout of query_index.
            k_layout: Layout of key_index.
            w_layout: Layout of weights.
            layout_str: 'BSND' or 'TND'.

        Raises:
            ValueError: If any constraint is violated.
        """
        op = "npu_dense_lightning_indexer_grad_kl_loss"
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
        """Infer output layouts for all four outputs.

        Rules:
            1. No Partial inputs allowed on query_index, key_index, or weights.
            2. Input sharding constraints are validated per layout_str.
            3. d_query_index, d_key_index, d_weights inherit their corresponding
               input layouts (independent deep copies).
            4. loss is a fully replicated 1-D scalar layout.

        Args:
            cache_values: [q_index_layout, k_index_layout, w_layout, layout_str]

        Returns:
            tuple: ((d_query_index_layout, d_key_index_layout,
                     d_weights_layout, loss_layout), None)

        Raises:
            ValueError: If any input has Partial status or sharding constraints
                are violated.
        """
        q_index_layout = cache_values[0]
        k_index_layout = cache_values[1]
        w_layout = cache_values[2]
        layout_str = cache_values[3]

        self._check_partial_inputs([q_index_layout, k_index_layout, w_layout])
        self._validate_input_layouts(q_index_layout, k_index_layout, w_layout, layout_str)

        return self._infer_output_layouts(q_index_layout, k_index_layout, w_layout, layout_str), None

    def get_expand_impl(  # pylint: disable=W0237
            self,
            func: Optional[Callable],
            infer_result: tuple,
            cache_values: list,
            extra_args: Optional[tuple] = None,
    ) -> Optional[Callable]:
        """Return a custom callable if context-parallel adjustments are needed.

        BSND+CP: wraps ``func`` to slice key, key_index, and key_rope S2
                 dimensions to the causal window for this rank's S1 slice, then
                 zero-pads d_key_index back to the full S2 size so that each
                 rank's partial gradient can be AllReduced across the CP group.
        TND+CP:  wraps ``func`` to adjust actual_seq_qlen/klen per rank.
        No CP:   returns None (dispatcher calls ``func`` directly).

        Args:
            func: The underlying op callable.
            infer_result: Output from ``infer_layout``.
            cache_values: [q_index_layout, k_index_layout, w_layout, layout_str].
            extra_args: Unused; kept for interface compatibility.

        Returns:
            Callable wrapper or None.
        """
        q_layout = cache_values[0]
        k_layout = cache_values[1]
        layout_str = cache_values[3]

        if layout_str == 'BSND':
            if q_layout.tensor_map[1] == -1:
                return None
            split_id = q_layout.get_split_id(1)

            def _bsnd_cp_impl(*args, **kwargs):
                local_q = args[0]
                s1_local = local_q.shape[1]
                s2_full = args[3].shape[1]
                sliced_k = _adjust_bsnd_key(args[1], s1_local, split_id)
                sliced_k_idx = _adjust_bsnd_key(args[3], s1_local, split_id)
                if len(args) > 10:  # MindSpore: all 18 positional args
                    local_k_rope = args[11]
                    sliced_k_rope = (
                        _adjust_bsnd_key(local_k_rope, s1_local, split_id)
                        if local_k_rope is not None else None
                    )
                    outs = func(
                        args[0], sliced_k, args[2], sliced_k_idx,
                        *args[4:11], sliced_k_rope, *args[12:], **kwargs
                    )
                else:
                    # PyTorch: 10 positional, optional tensors in kwargs
                    new_args = list(args)
                    new_args[1] = sliced_k
                    new_args[3] = sliced_k_idx
                    new_kwargs = dict(kwargs)
                    if new_kwargs.get('key_rope') is not None:
                        new_kwargs['key_rope'] = _adjust_bsnd_key(
                            new_kwargs['key_rope'], s1_local, split_id
                        )
                    outs = func(*new_args, **new_kwargs)
                d_q_idx, d_k_idx, d_w, loss = outs
                pad_len = s2_full - d_k_idx.shape[1]
                if pad_len > 0:
                    pad_shape = list(d_k_idx.shape)
                    pad_shape[1] = pad_len
                    zero_pad = platform.zeros(
                        tuple(pad_shape), dtype=d_k_idx.dtype, device=d_k_idx.device
                    )
                    d_k_idx = platform.cat([d_k_idx, zero_pad], dim=1)
                return d_q_idx, d_k_idx, d_w, loss

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
            if len(args) > 10:
                qlen_tensor = args[12]
                klen_tensor = args[13]
            else:
                qlen_tensor = kwargs.get('actual_seq_qlen')
                klen_tensor = kwargs.get('actual_seq_klen')

            if qlen_tensor is None or klen_tensor is None:
                return func(*args, **kwargs)

            adj_q, adj_k = _adjust_tnd_seq_lens(
                local_q, local_k, qlen_tensor, klen_tensor,
                cp_rank=cp_rank,
            )

            if len(args) > 10:  # MindSpore
                return func(*args[:12], adj_q, adj_k, *args[14:], **kwargs)
            return func(*args, **{**kwargs, 'actual_seq_qlen': adj_q,
                                 'actual_seq_klen': adj_k})

        return _tnd_impl
