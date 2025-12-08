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
"""
Distributed implementation for MatMul operator.
"""

from mindspore import nn
from mindspore import Tensor
from mindspore.ops import flash_attention_score
from dist_parallel.layout import Layout
from dist_parallel.local_func import custom_shard


class ParallelFlashAttention(nn.Cell):
    """
    Parallel FlashAttention.

    This class implements the FlashAttention mechanism for fast and memory-efficient attention computation.
    It supports multiple attention types, mask modes, and is optimized for parallel training including
    tensor and context parallelism.

    Reference:
        "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
        https://arxiv.org/abs/2205.14135

    Args:
        head_num (int): The head num of `query`, equal to N1.
        keep_prob (double, optional): The keep probability of dropout. Value range is (0.0, 1.0]. When `keep_prob`
            is 1.0, `drop_mask` should be None.
            Default: ``1.0``.
        scalar_value (double, optional): The scale value indicating the scale coefficient, which is used as the
            scalar of Muls in the calculation. Generally, the value is 1.0 / (D ** 0.5).
            Default: ``1.0``.
        pre_tokens (int, optional): Parameter for sparse computation, represents how many tokens are counted forward.
            When `sparse_mode` is set to 1, 2, 3, or 5, this parameter does not take effect.
            Default: ``2147483647``.
        next_tokens (int, optional): Parameter for sparse computation, represents how many tokens are counted backward.
            When `sparse_mode` is set to 1, 2, 3, or 5, this parameter does not take effect. Default: ``2147483647``.
            The value of `pre_tokens` corresponds to S1, and the value of `next_tokens` corresponds to S2.
            They define the valid area on the `attn_mask` matrix. It must ensure that the band is not empty.
            The following values are not allowed:

            - pre_tokens < 0 and next_tokens < 0.
            - (pre_tokens < 0 and next_tokens >= 0) and (next_tokens < abs(pre_tokens) or abs(pre_tokens) >= S2).
            - (pre_tokens >= 0 and next_tokens < 0) and (abs(next_tokens) > pre_tokens or abs(next_tokens) >= S1).

        inner_precise (int, optional): The parameter is reserved and not implemented yet. Default:``0``.
        input_layout (str, optional): Specifies the layout of input `query`, `key` and `value`. The value can be
            "BSH", "BNSD", "SBH", "BSND" or "TND". "TND" is an experimental format. Default: ``"BSH"``.
            When input_layout is "TND", the following restrictions must be met.
            Assume there are two lists that represent the length of the input sequence: list_seq_q and list_seq_k. Each
            value in the list indicates the length of the sequence in the batch. For example, list_seq_q = [4, 2, 6],
            list_seq_k = [10, 3, 9]. The element of list indicate S. T1 is sum(list_seq_q) = 12, T2 is
            sum(list_seq_k) = 22.
            max_seqlen_q = max(list_seq_q), max_seqlen_k = max(list_seq_k).
            qk_pointer = sum(list_seq_q * list_seq_k), which is the sum of the element multiplication.

            - The lengths of two lists must be the same, and size of list is batch. batch is less than or equal to
              1024.
            - When `input_layout` is "TND", `actual_seq_qlen` and `actual_seq_kvlen` must be not none.
              Otherwise, they are none.
            - The `actual_seq_qlen` and `actual_seq_kvlen` are the cumulative sum of sequence of key/value, so they must
              be non-decreasing.
            - If `real_shift` is not none, list_seq_q and list_seq_k must be same. The maximum value of list_seq_q and
              list_seq_k is greater than 1024. `real_shift` should be :math:`(B, N1, 1024, S2)` and
              :math:`(1, N1, 1024, S2)`, and S2 is equal to max_seqlen_k.
            - `attn_mask` must be a lower trianglar matrix, so `sparse_mode` should be 2 or 3. The shape of `attn_mask`
              should be :math:`(2048, 2048)`.
            - The shape of `drop_mask` is :math:`(qk_pointer * N1 // 8,)`.
            - `prefix` is none.
            - `next_tokens` is 0, and `pre_tokens` is not less than max_seqlen_q.
            - When `sparse_mode` is 3, S1 of each batch should be less than or equal to S2.
            - 0 should not exist in list_seq_k.

        sparse_mode (int, optional): Indicates sparse mode. Default: ``0``.

            - 0: Indicates the defaultMask mode. If `attn_mask` is not passed, the mask operation is not performed,
              `next_tokens` and `pre_tokens` (internally assigned as INT_MAX) are ignored. If passed in, the full
              `attn_mask` matrix (S1 * S2) needs to be passed in, indicating that the part between `next_tokens` and
              `pre_tokens` needs to be calculated.
            - 1: Represents allMask, that is, passing in the complete `attn_mask` matrix.
            - 2: Representing the leftUpCausal mode corresponds to the lower triangle scenario divided by the left
              vertex, and the optimized `attn_mask` matrix (2048*2048) is required.
            - 3: Representing the rightDownCausal model corresponds to the lower triangle scene divided by the lower
              right vertex, and the optimized `attn_mask` matrix (2048*2048) is required.
            - 4: Represents the band scenario, that is, the part between counting `next_tokens` and `pre_tokens`,
              and the optimized `attn_mask` matrix (2048*2048) is required.
            - 5: Represents the prefix scenario, that is, on the basis of rightDownCasual, a matrix with length S1 and
              width N is added to the left side. The value of N is obtained by the new input prefix, and the N value
              of each Batch axis is different, not implemented yet.
            - 6: Represents the global scenario, not implemented yet.
            - 7: Represents the dilated scenario, not implemented yet.
            - 8: Represents the block_local scenario, not implemented yet.

    Inputs:
        - **query** (Tensor): The query tensor with shape (B, S1, H1) or (B, N1, S1, D).
        - **key** (Tensor): The key tensor with shape (B, S2, H2) or (B, N2, S2, D).
        - **value** (Tensor): The value tensor with shape (B, S2, H2) or (B, N2, S2, D).
        - **attn_mask** (Tensor, optional): Attention mask. A value of 0 keeps the element;
          a value of 1 masks it out. Shape can vary based on attention mode.

    Outputs:
        - **attention_out** (Tensor): The attention output tensor with the same shape and type as `query`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self,
                 head_num,
                 keep_prob=1.0,
                 scalar_value=1.0,
                 pre_tokens=2147483647,
                 next_tokens=2147483647,
                 inner_precise=0,
                 input_layout='BSH',
                 sparse_mode=0
                 ):
        super().__init__()
        self._in_layouts = None
        self._out_layouts = None
        if input_layout.find('T') != -1:
            raise ValueError("Do not support FlashAttentionVarLen currently.")

        self._batch_dim, self._seq_dim, self._head_num_dim = self._infer_dim_by_layout(input_layout)

        self._head_num = head_num
        self._real_shift = None
        self._drop_mask = None
        self._padding_mask = None
        self._prefix = None
        self._actual_seq_qlen = None
        self._actual_seq_kvlen = None
        self._keep_prob = keep_prob
        self._scalar_value = scalar_value
        self._pre_tokens = pre_tokens
        self._next_tokens = next_tokens
        self._inner_precise = inner_precise
        self._input_layout = input_layout
        self._sparse_mode = sparse_mode

        self._head_num = head_num
        self._wrap_func = None

    def construct(self, query, key, value, attn_mask):
        # TODO: actual_seq_qlen/kv_len is tuple type, cannot be described by dtensor
        if self._in_layouts is None or self._out_layouts is None:
            raise ValueError("Please call the shard function first.")
        _, head_num_split_num, _ = self._infer_split_dim_by_in_strategy()
        input_args = (query, key, value, self._head_num // head_num_split_num, self._real_shift, self._drop_mask,
                      self._padding_mask, attn_mask, self._prefix, self._actual_seq_qlen, self._actual_seq_kvlen,
                      self._keep_prob, self._scalar_value, self._pre_tokens, self._next_tokens, self._inner_precise,
                      self._input_layout, self._sparse_mode)
        in_layout_with_non_tensor = self._insert_none_for_non_tensor_arg(self._in_layouts, input_args)
        self._wrap_func = custom_shard(flash_attention_score, self._out_layouts, in_layout_with_non_tensor)
        return self._wrap_func(*input_args)

    def shard(self, in_strategy, out_strategy):
        if self._in_layouts is not None or self._out_layouts is not None:
            raise ValueError(f"For {self.__class__.__name__}, the shard method cannot be called repeatedly to set the "
                             f"sharding strategy.")

        if not isinstance(in_strategy, tuple) or any(not isinstance(ele, Layout) for ele in in_strategy):
            raise ValueError("The type of in_startegy must be Tuple[Layout].")
        if not isinstance(out_strategy, tuple) or any(not isinstance(ele, Layout) for ele in out_strategy):
            raise ValueError("The type of out_strategy must be Tuple[Layout].")
        self._in_layouts = in_strategy
        self._out_layouts = out_strategy
        super().shard(in_strategy, out_strategy)

    @staticmethod
    def _insert_none_for_non_tensor_arg(layouts, args):
        """_insert_none_for_non_tensor_arg"""
        if not layouts:
            raise ValueError("Flash attention need config in_layouts.")
        ret_layouts = [None] * len(args)
        tensor_indexes = [i for i in range(len(args)) if isinstance(args[i], Tensor)]
        if len(tensor_indexes) != len(layouts):
            raise ValueError(f"The size of layouts must be equal to tensor input num, but got {len(layouts)} "
                             f"and {len(tensor_indexes)}.")
        for i, tensor_index in enumerate(tensor_indexes):
            ret_layouts[tensor_index] = layouts[i]
        return ret_layouts

    @staticmethod
    def _infer_dim_by_layout(input_layout):
        """Infer batch, seq, head_num dim by input layout."""
        batch_dim = input_layout.find('B')
        head_num_dim = input_layout.find('H')
        if head_num_dim == -1:
            head_num_dim = input_layout.find('N')
        seq_dim = input_layout.find('S')
        assert batch_dim != -1, f"Cannot found batch dim by input_layout: {input_layout}"
        assert seq_dim != -1, f"Cannot found seq dim dim by input_layout: {input_layout}"
        assert head_num_dim != -1, f"Cannot found head num dim by input_layout: {input_layout}"
        return batch_dim, seq_dim, head_num_dim

    def _infer_split_dim_by_in_strategy(self):
        query_layout = self._in_layouts[0]
        batch_split_num = self._get_split_num(query_layout, self._batch_dim)
        head_num_split_num = self._get_split_num(query_layout, self._head_num_dim)
        seq_split_num = self._get_split_num(query_layout, self._seq_dim)
        assert seq_split_num == 1, "The sharding along sequence dimension is not support currently."
        return batch_split_num, head_num_split_num, seq_split_num

    @staticmethod
    def _get_split_num(layout: Layout, dim: int):
        """Get split num along specify dimension."""
        map_dev = layout.alias_tensor_map[dim]
        split_num = 1
        if isinstance(map_dev, tuple):
            for axis_name in map_dev:
                if axis_name != 'None':
                    split_num *= layout.mesh.get_device_num_along_axis(axis_name)
        elif isinstance(map_dev, str):
            split_num *= layout.mesh.get_device_num_along_axis(map_dev)
        else:
            raise ValueError(f"Failed to get split num along dim {dim}, corresponding map is {map_dev}")
        return split_num
