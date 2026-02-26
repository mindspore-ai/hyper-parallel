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

# pylint: disable=unused-argument
"""FlashAttentionScore Distributed Operator for MindSpore"""

import copy
import warnings

from typing import List, Tuple, Optional, Any
from hyper_parallel.core.layout import Layout
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_ops_register import register_distributed_op
from hyper_parallel.platform import get_platform

platform = get_platform()
Tensor = platform.Tensor

SPARSE_DEFAULT_MASK = 0
SPARSE_ALL_MASK = 1
SPARSE_LEFT_UP_CAUSAL = 2
SPARSE_RIGHT_DOWN_CAUSAL = 3
SPARSE_BAND = 4

LEFT_UP_TO_LEFT_UP = 0
LEFT_UP_TO_RIGHT_DOWN = 1
RIGHT_DOWN_TO_RIGHT_DOWN = 2

SPARSE_MODE_UPDATE_MAP = {
    SPARSE_DEFAULT_MASK: LEFT_UP_TO_LEFT_UP,
    SPARSE_ALL_MASK: LEFT_UP_TO_LEFT_UP,
    SPARSE_LEFT_UP_CAUSAL: LEFT_UP_TO_RIGHT_DOWN,
    SPARSE_RIGHT_DOWN_CAUSAL: RIGHT_DOWN_TO_RIGHT_DOWN,
    SPARSE_BAND: RIGHT_DOWN_TO_RIGHT_DOWN,
}

INPUT_LAYOUT_INT_TO_STR = {
    0: "BSH",
    1: "BNSD",
    2: "SBH",
    3: "BSND",
    4: "TND",
}

# Indices into extra_args list extracted by _process_args_and_kwargs.
# pyboost flat arg order: [q, k, v, real_shift, drop_mask, padding_mask,
#     attn_mask, prefix, actual_seq_qlen, actual_seq_kvlen,
#     head_num, keep_prob, scale_value, pre_tokens, next_tokens,
#     inner_precise, input_layout, sparse_mode]
# The first 10 are tensors/None (go to input_layouts); the last 8 are scalars
# (go to extra_args in order).
# Negative indices are used because optional tensor args (e.g. attn_mask) that
# are plain Tensors (not DTensors) also end up in extra_args before the scalars.
EA_HEAD_NUM = -8
EA_KEEP_PROB = -7
EA_SCALE_VALUE = -6
EA_PRE_TOKENS = -5
EA_NEXT_TOKENS = -4
EA_INNER_PRECISE = -3
EA_INPUT_LAYOUT = -2
EA_SPARSE_MODE = -1


def _resolve_input_layout(input_layout) -> str:
    """Resolve input_layout from either string or integer enum to string."""
    if isinstance(input_layout, str):
        return input_layout
    if isinstance(input_layout, int) and input_layout in INPUT_LAYOUT_INT_TO_STR:
        return INPUT_LAYOUT_INT_TO_STR[input_layout]
    return str(input_layout)


class FlashAttentionScoreDistributedOp:
    """Distributed operator for mindspore.ops.flash_attention_score."""

    def __init__(self, op_name: str):
        self.op_name = op_name
        register_distributed_op(op_name, self)

        self._layout_dims = {
            "BSH": {"batch": 0, "seq": 1, "hidden": 2},
            "BNSD": {"batch": 0, "head": 1, "seq": 2, "dim": 3},
            "SBH": {"seq": 0, "batch": 1, "hidden": 2},
            "BSND": {"batch": 0, "seq": 1, "head": 2, "dim": 3},
            "TND": {"total": 0, "head": 1, "dim": 2},
        }

    @staticmethod
    def _to_python_scalar(value):
        """Convert a value that may be a scalar Tensor to a Python native type.

        Scalar parameters from pyboost may arrive as Tensor objects after
        passing through the C++ dispatch layer. Arithmetic and comparison
        operations on such Tensor scalars would trigger device kernel calls
        (e.g. aclnnRemainder on Ascend), which is both unnecessary and
        error-prone. This helper converts them back to plain Python types.
        """
        if isinstance(value, Tensor):
            return value.item()
        return value

    def _tensor_map_to_placements(self, base_layout: Layout, tensor_map: tuple) -> tuple:
        """Convert tensor_map to placements."""
        mesh_ndim = len(base_layout.mesh_shape)
        placements = []

        for mesh_dim_idx in range(mesh_ndim):
            is_sharded = False

            for tensor_dim_idx, tensor_dim_map in enumerate(tensor_map):
                if tensor_dim_map == -1:
                    continue

                if isinstance(tensor_dim_map, tuple):
                    if mesh_dim_idx in tensor_dim_map:
                        placements.append(Shard(tensor_dim_idx))
                        is_sharded = True
                        break
                elif tensor_dim_map == mesh_dim_idx:
                    placements.append(Shard(tensor_dim_idx))
                    is_sharded = True
                    break

            if not is_sharded:
                placements.append(Replicate())

        return tuple(placements)

    def _is_dynamic_shape(self, tensor: Tensor, dim: int) -> bool:
        """Check if tensor has dynamic shape at given dimension."""
        try:
            shape_val = tensor.shape[dim]
            if isinstance(shape_val, int):
                return shape_val == -1
            return not isinstance(shape_val, int)
        except (IndexError, AttributeError):
            return False

    def _get_dynamic_shape_info(self, query: Tensor, key: Tensor, input_layout: str) -> dict:
        """Get dynamic shape information for query and key tensors."""
        dims = self._layout_dims.get(input_layout, {})

        seq_dim_idx = None
        if 'seq' in dims:
            seq_dim_idx = dims['seq']
        elif 'total' in dims:
            seq_dim_idx = dims['total']

        if seq_dim_idx is None:
            return {'is_dynamic': False}

        q_is_dynamic = self._is_dynamic_shape(query, seq_dim_idx)
        kv_is_dynamic = self._is_dynamic_shape(key, seq_dim_idx)

        return {
            'is_dynamic': q_is_dynamic or kv_is_dynamic,
            'q_seq_dim': seq_dim_idx,
            'kv_seq_dim': seq_dim_idx,
            'q_batch_dim': dims.get('batch', dims.get('total')),
        }

    def _is_attn_mask_compressed(self, sparse_mode: int) -> bool:
        """Check if attention mask is compressed for given sparse mode."""
        return sparse_mode in (
            SPARSE_LEFT_UP_CAUSAL,
            SPARSE_RIGHT_DOWN_CAUSAL,
            SPARSE_BAND,
        )

    def _compute_sparse_params(
        self,
        sparse_mode: int,
        pre_tokens: int,
        next_tokens: int,
        split_id: int,
        split_num: int,
        local_q_len: int,
        global_q_len: int,
        global_kv_len: int,
    ) -> Tuple[int, int, int]:
        """Calculate adjusted sparse parameters for static shape."""
        if sparse_mode not in SPARSE_MODE_UPDATE_MAP:
            return sparse_mode, pre_tokens, next_tokens

        if sparse_mode == SPARSE_ALL_MASK:
            return sparse_mode, pre_tokens, next_tokens

        if sparse_mode in (SPARSE_DEFAULT_MASK, SPARSE_BAND):
            new_pre_tokens = pre_tokens
            new_next_tokens = next_tokens
        else:
            new_pre_tokens = global_kv_len
            new_next_tokens = 0

        new_sparse_mode = SPARSE_BAND if sparse_mode != SPARSE_DEFAULT_MASK else sparse_mode
        update_mode = SPARSE_MODE_UPDATE_MAP[sparse_mode]

        if update_mode == LEFT_UP_TO_LEFT_UP:
            new_pre_tokens += -split_id * local_q_len
            new_next_tokens += split_id * local_q_len
        elif update_mode == LEFT_UP_TO_RIGHT_DOWN:
            offset = global_kv_len - (split_id + 1) * local_q_len
            new_pre_tokens += offset
            new_next_tokens += -offset
        elif update_mode == RIGHT_DOWN_TO_RIGHT_DOWN:
            offset = (split_num - split_id - 1) * local_q_len
            new_pre_tokens += offset
            new_next_tokens += -offset

        return new_sparse_mode, new_pre_tokens, new_next_tokens

    def _compute_sparse_params_dynamic(
        self,
        query: Tensor,
        key: Tensor,
        sparse_mode: int,
        pre_tokens: int,
        next_tokens: int,
        split_id: int,
        split_num: int,
        seq_dim_idx: int,
    ) -> Tuple:
        """Calculate adjusted sparse parameters for dynamic shape."""
        if sparse_mode not in SPARSE_MODE_UPDATE_MAP:
            return sparse_mode, pre_tokens, next_tokens

        if sparse_mode == SPARSE_ALL_MASK:
            return sparse_mode, pre_tokens, next_tokens

        query_seq_length = query.shape[seq_dim_idx]
        key_seq_length = key.shape[seq_dim_idx]
        local_q_len_symbolic = query_seq_length // split_num

        if sparse_mode in (SPARSE_DEFAULT_MASK, SPARSE_BAND):
            new_pre_tokens = pre_tokens
            new_next_tokens = next_tokens
        else:
            new_pre_tokens = key_seq_length
            new_next_tokens = 0

        new_sparse_mode = SPARSE_BAND if sparse_mode != SPARSE_DEFAULT_MASK else sparse_mode
        update_mode = SPARSE_MODE_UPDATE_MAP[sparse_mode]

        if update_mode == LEFT_UP_TO_LEFT_UP:
            offset = -split_id * local_q_len_symbolic
            new_pre_tokens = new_pre_tokens + offset
            new_next_tokens = new_next_tokens - offset
        elif update_mode == LEFT_UP_TO_RIGHT_DOWN:
            offset = key_seq_length - (split_id + 1) * local_q_len_symbolic
            new_pre_tokens = new_pre_tokens + offset
            new_next_tokens = new_next_tokens - offset
        elif update_mode == RIGHT_DOWN_TO_RIGHT_DOWN:
            offset = (split_num - split_id - 1) * local_q_len_symbolic
            new_pre_tokens = new_pre_tokens + offset
            new_next_tokens = new_next_tokens - offset

        return new_sparse_mode, new_pre_tokens, new_next_tokens

    def _adjust_actual_seq_len_for_tnd_cp(
        self,
        query: Tensor,
        key: Tensor,
        actual_seq_qlen: List[int],
        actual_seq_kvlen: List[int],
        split_id: int,
        kv_is_sharded: bool,
    ) -> Tuple[List[int], List[int]]:
        """Adjust actual_seq_qlen and actual_seq_kvlen for TND layout with context parallel."""
        slice_tq = query.shape[0]
        slice_tk = key.shape[0]
        is_dynamic = self._is_dynamic_shape(query, 0) or self._is_dynamic_shape(key, 0)

        return self._adjust_actual_seq_len(
            slice_tq, slice_tk,
            actual_seq_qlen, actual_seq_kvlen,
            split_id, kv_is_sharded,
            is_dynamic,
        )

    def _adjust_actual_seq_len(
        self,
        slice_tq,
        slice_tk,
        actual_seq_qlen: List[int],
        actual_seq_kvlen: List[int],
        split_id: int,
        kv_is_sharded: bool,
        is_dynamic: bool,
    ) -> Tuple[List[int], List[int]]:
        """Adjust actual_seq_len for both static and dynamic shapes."""
        import mindspore  # pylint: disable=import-outside-toplevel
        from mindspore import mint  # pylint: disable=import-outside-toplevel
        from mindspore import ops  # pylint: disable=import-outside-toplevel

        offset_q = slice_tq * split_id

        actual_seq_qlen_tensor = mindspore.tensor(actual_seq_qlen, dtype=mindspore.int64)
        actual_seq_kvlen_tensor = mindspore.tensor(actual_seq_kvlen, dtype=mindspore.int64)

        qlen_offset = actual_seq_qlen_tensor - offset_q
        new_actual_seq_qlen = mint.clamp(qlen_offset, min=0, max=slice_tq)

        if kv_is_sharded:
            offset_kv = slice_tk * split_id
            kvlen_offset = actual_seq_kvlen_tensor - offset_kv
            new_actual_seq_kvlen = mint.clamp(kvlen_offset, min=0, max=slice_tk)
        else:
            relu_result = ops.clamp(qlen_offset.float(), min=0).long()
            kvlen_offset = relu_result - new_actual_seq_qlen
            new_actual_seq_kvlen = actual_seq_kvlen_tensor - kvlen_offset

            if len(new_actual_seq_kvlen) > 0:
                last_idx = len(new_actual_seq_kvlen) - 1
                if is_dynamic:
                    mask = actual_seq_kvlen_tensor[last_idx] == slice_tk
                    new_actual_seq_kvlen[last_idx] = mint.where(
                        mask, slice_tk, new_actual_seq_kvlen[last_idx]
                    )
                else:
                    if actual_seq_kvlen_tensor[last_idx].item() == slice_tk:
                        new_actual_seq_kvlen[last_idx] = slice_tk

        return new_actual_seq_qlen.tolist(), new_actual_seq_kvlen.tolist()

    def _validate_attn_mask(
        self,
        attn_mask: Optional[Tensor],
        sparse_mode: int,
        input_layout: str,
        is_varlen: bool = False
    ) -> None:
        """Validate attention mask shape and configuration for given sparse mode."""
        if attn_mask is None:
            if sparse_mode == SPARSE_ALL_MASK:
                raise ValueError(
                    "sparse_mode=1 (allMask) requires attn_mask to be provided"
                )
            return

        mask_shape = attn_mask.shape

        if len(mask_shape) not in (2, 4):
            raise ValueError(
                f"attn_mask only supports 2D or 4D format, but got {len(mask_shape)}D"
            )

        if is_varlen:
            if len(mask_shape) != 2:
                raise ValueError(
                    f"Varlen scenario only supports 2D attn_mask (maxSq, maxSkv), "
                    f"but got {len(mask_shape)}D"
                )

        if self._is_attn_mask_compressed(sparse_mode):
            expected_shape = (2048, 2048)
            if mask_shape[-2:] != expected_shape:
                warnings.warn(
                    f"sparse_mode={sparse_mode} uses compressed mask, "
                    f"expected shape {expected_shape} but got {mask_shape[-2:]}"
                )

    def _validate_real_shift_configuration(
        self,
        real_shift: Optional[Tensor],
        sparse_mode: int
    ) -> None:
        """Validate real_shift (positional encoding) configuration."""
        if real_shift is None:
            return

        real_shift_shape = real_shift.shape

        if len(real_shift_shape) not in (3, 4):
            raise ValueError(
                f"real_shift only supports 3D or 4D format, "
                f"but got {len(real_shift_shape)}D"
            )

        if len(real_shift_shape) == 4 and real_shift_shape[2] == 1024:
            warnings.warn("Detected Alibi positional encoding compression scenario")

    def infer_layout(
        self, input_layouts: List[Optional[Layout]], extra_args: List[Any]
    ) -> Tuple[Layout, ...]:
        """Infer output layouts from input layouts and scalar params in extra_args."""
        query_layout = input_layouts[0]
        if query_layout is None:
            raise ValueError("Query layout cannot be None")

        input_layout_str = _resolve_input_layout(
            self._to_python_scalar(extra_args[EA_INPUT_LAYOUT])
        )

        attention_out_layout = copy.deepcopy(query_layout)
        if attention_out_layout.placements is None and attention_out_layout.tensor_map is not None:
            attention_out_placements = self._tensor_map_to_placements(
                attention_out_layout, attention_out_layout.tensor_map
            )
            attention_out_layout.set_placements(attention_out_placements)

        if input_layout_str in self._layout_dims:
            softmax_layout = self._infer_softmax_layout_by_input_layout(
                query_layout, input_layout_str
            )
        else:
            softmax_layout = self._infer_softmax_layout_conservatively(query_layout)

        softmax_max_layout = softmax_layout
        softmax_sum_layout = copy.deepcopy(softmax_layout)
        softmax_out_layout = self._create_replicated_scalar_layout(query_layout)
        if softmax_out_layout.placements is None and softmax_out_layout.tensor_map is not None:
            softmax_out_placements = self._tensor_map_to_placements(
                softmax_out_layout, softmax_out_layout.tensor_map
            )
            softmax_out_layout.set_placements(softmax_out_placements)

        # Output order must match C++ kernel: softmax_max, softmax_sum,
        # softmax_out, attention_out.
        return (
            softmax_max_layout,
            softmax_sum_layout,
            softmax_out_layout,
            attention_out_layout,
        )

    def _infer_softmax_layout_conservatively(self, query_layout: Layout) -> Layout:
        """Conservative fallback for softmax layout inference."""
        softmax_layout = Layout.from_device_mesh(query_layout.mesh)
        query_tm = query_layout.tensor_map

        if query_tm is None or len(query_tm) == 0:
            softmax_tensor_map = (-1, -1, -1, -1)
        else:
            softmax_tm = [
                query_tm[0] if len(query_tm) > 0 else -1,
                -1,
                -1,
                -1,
            ]
            softmax_tensor_map = tuple(softmax_tm)

            warnings.warn(
                f"Using conservative softmax layout inference due to "
                f"missing/invalid input_layout.\n"
                f"Query tensor_map: {query_tm}\n"
                f"Inferred softmax tensor_map: {softmax_tensor_map}\n"
                f"This may not be optimal. Please provide explicit input_layout."
            )

        softmax_layout.set_tensor_map(softmax_tensor_map)
        softmax_placements = self._tensor_map_to_placements(
            softmax_layout, softmax_tensor_map
        )
        softmax_layout.set_placements(softmax_placements)

        return softmax_layout

    def _infer_softmax_layout_by_input_layout(
        self,
        query_layout: Layout,
        input_layout_str: str,
    ) -> Layout:
        """Infer softmax layout based on input_layout."""
        softmax_tensor_map = self._build_softmax_tensor_map(
            query_layout, input_layout_str
        )

        softmax_layout = Layout.from_device_mesh(query_layout.mesh)
        softmax_layout.set_tensor_map(softmax_tensor_map)
        softmax_placements = self._tensor_map_to_placements(
            softmax_layout, softmax_tensor_map
        )
        softmax_layout.set_placements(softmax_placements)

        return softmax_layout

    def _build_softmax_tensor_map(
        self,
        query_layout: Layout,
        input_layout_str: str,
    ) -> tuple:
        """Build softmax tensor_map based on query layout and input_layout."""
        dims = self._layout_dims.get(input_layout_str, {})
        query_tm = query_layout.tensor_map

        if query_tm is None:
            return (-1, -1, -1, -1)

        softmax_tm = [-1, -1, -1, -1]

        if input_layout_str == "TND":
            softmax_tm[0] = query_tm[0] if len(query_tm) > 0 else -1
            softmax_tm[1] = query_tm[1] if len(query_tm) > 1 else -1
        else:
            if "batch" in dims:
                batch_idx = dims["batch"]
                if batch_idx < len(query_tm):
                    softmax_tm[0] = query_tm[batch_idx]

            if "head" in dims:
                head_idx = dims["head"]
                if head_idx < len(query_tm):
                    softmax_tm[1] = query_tm[head_idx]
            elif "hidden" in dims:
                hidden_idx = dims["hidden"]
                if hidden_idx < len(query_tm):
                    softmax_tm[1] = query_tm[hidden_idx]

            if "seq" in dims:
                seq_idx = dims["seq"]
                if seq_idx < len(query_tm):
                    softmax_tm[2] = query_tm[seq_idx]

        softmax_tm[3] = -1

        return tuple(softmax_tm)

    def _validate_sharding_consistency(
        self,
        query_layout: Layout,
        key_layout: Optional[Layout],
        input_layout: str
    ):
        """Validate Q/K/V sharding consistency."""
        if key_layout is None or not hasattr(key_layout, 'tensor_map'):
            return

        dims = self._layout_dims.get(input_layout, {})
        q_tm = query_layout.tensor_map
        k_tm = key_layout.tensor_map

        if q_tm is None or k_tm is None:
            return

        self._check_batch_consistency(dims, q_tm, k_tm, input_layout)
        self._check_hidden_consistency(dims, q_tm, k_tm, input_layout)
        self._check_dim_consistency(dims, q_tm, k_tm, input_layout)

    def _check_batch_consistency(self, dims, q_tm, k_tm, input_layout):
        """Check batch dimension sharding consistency."""
        if "batch" not in dims:
            return

        batch_idx = dims["batch"]
        if batch_idx >= len(q_tm) or batch_idx >= len(k_tm):
            return

        q_batch_shard = self._normalize_dim_map(q_tm[batch_idx])
        k_batch_shard = self._normalize_dim_map(k_tm[batch_idx])

        if q_batch_shard != k_batch_shard:
            raise ValueError(
                f"Query and Key/Value must have identical batch sharding strategy.\n"
                f"Input layout: {input_layout}\n"
                f"Query batch sharding (dim {batch_idx}): {q_batch_shard}\n"
                f"Key/Value batch sharding (dim {batch_idx}): {k_batch_shard}\n"
                f"Query tensor_map: {q_tm}\n"
                f"Key tensor_map: {k_tm}"
            )

    def _check_hidden_consistency(self, dims, q_tm, k_tm, input_layout):
        """Check hidden dimension sharding consistency."""
        if "hidden" not in dims:
            return

        hidden_idx = dims["hidden"]
        if hidden_idx >= len(q_tm) or hidden_idx >= len(k_tm):
            return

        q_hidden_shard = self._normalize_dim_map(q_tm[hidden_idx])
        k_hidden_shard = self._normalize_dim_map(k_tm[hidden_idx])

        if q_hidden_shard != k_hidden_shard:
            raise ValueError(
                f"Query and Key/Value must have identical hidden sharding strategy.\n"
                f"Input layout: {input_layout}\n"
                f"Query hidden sharding (dim {hidden_idx}): {q_hidden_shard}\n"
                f"Key/Value hidden sharding (dim {hidden_idx}): {k_hidden_shard}\n"
                f"Query tensor_map: {q_tm}\n"
                f"Key tensor_map: {k_tm}\n"
                f"Note: This checks sharding strategy, not tensor size.\n"
                f"GQA (different head counts) is supported when strategies match."
            )

    def _check_dim_consistency(self, dims, q_tm, k_tm, input_layout):
        """Check dim dimension sharding consistency."""
        if "dim" not in dims:
            return

        dim_idx = dims["dim"]
        if dim_idx >= len(q_tm) or dim_idx >= len(k_tm):
            return

        q_dim_shard = self._normalize_dim_map(q_tm[dim_idx])
        k_dim_shard = self._normalize_dim_map(k_tm[dim_idx])

        if q_dim_shard != k_dim_shard:
            raise ValueError(
                f"Query and Key/Value must have identical dim sharding strategy.\n"
                f"Input layout: {input_layout}\n"
                f"Query dim sharding (dim {dim_idx}): {q_dim_shard}\n"
                f"Key/Value dim sharding (dim {dim_idx}): {k_dim_shard}\n"
                f"Query tensor_map: {q_tm}\n"
                f"Key tensor_map: {k_tm}"
            )

    def _check_seq_sharding_compatibility(
        self,
        query_layout: Layout,
        key_layout: Optional[Layout],
        input_layout: str,
        seq_dim_idx: int,
        seq_split_num: int,
        kv_seq_split_num: int
    ):
        """Check sequence dimension sharding compatibility."""
        if key_layout is None:
            return

        q_tm = query_layout.tensor_map
        k_tm = key_layout.tensor_map

        if q_tm is None or k_tm is None:
            return

        if seq_dim_idx >= len(q_tm) or seq_dim_idx >= len(k_tm):
            return

        if input_layout != "TND" and kv_seq_split_num > 1:
            raise NotImplementedError(
                f"KV sequence sharding is not supported for layout '{input_layout}' "
                f"without Ring Attention.\n"
                f"Query sequence split num: {seq_split_num}\n"
                f"Key/Value sequence split num: {kv_seq_split_num}\n"
                f"Supported scenarios:\n"
                f"  - Query sequence sharding + KV not sharded (Ulysses-style)\n"
                f"  - Query and KV both not sharded\n"
                f"Unsupported scenario:\n"
                f"  - KV sequence sharding (requires Ring Attention)"
            )

        q_seq_shard = self._normalize_dim_map(q_tm[seq_dim_idx])
        k_seq_shard = self._normalize_dim_map(k_tm[seq_dim_idx])

        if q_seq_shard != k_seq_shard:
            if input_layout == "TND":
                pass
            elif kv_seq_split_num > 1:
                raise NotImplementedError(
                    f"Ring Attention (KV sequence sharding with different strategy) "
                    f"is not supported.\n"
                    f"Input layout: {input_layout}\n"
                    f"Query sequence split num: {seq_split_num}\n"
                    f"Key/Value sequence split num: {kv_seq_split_num}\n"
                    f"Query seq sharding strategy: {q_seq_shard}\n"
                    f"Key/Value seq sharding strategy: {k_seq_shard}"
                )

    def _truncate_result(self, result):
        """Truncate operator result to first 4 outputs."""
        if isinstance(result, (tuple, list)) and len(result) >= 4:
            return result[:4]
        return result

    def _adjust_head_num(self, head_num: int, head_split_num: int) -> int:
        """Validate and adjust head_num for head parallelism."""
        if head_split_num <= 0:
            raise ValueError(f"Invalid head_split_num={head_split_num}")
        if head_num % head_split_num != 0:
            raise ValueError(
                f"head_num({head_num}) not divisible by "
                f"head_split_num({head_split_num})"
            )
        return head_num // head_split_num

    def _compute_adjusted_sparse_params(
        self,
        query, key,
        sparse_mode: int,
        pre_tokens: int,
        next_tokens: int,
        split_id: int,
        seq_split_num: int,
        seq_dim_idx: int,
        kv_seq_split_num: int,
        is_dynamic: bool,
    ) -> Tuple[int, int, int]:
        """Compute adjusted sparse parameters based on dynamic or static shape."""
        if is_dynamic:
            return self._compute_sparse_params_dynamic(
                query, key,
                sparse_mode, pre_tokens, next_tokens,
                split_id, seq_split_num, seq_dim_idx,
            )

        local_q_len = query.shape[seq_dim_idx]
        global_q_len = local_q_len * seq_split_num
        local_kv_len = (
            key.shape[seq_dim_idx]
            if hasattr(key, "shape") and len(key.shape) > seq_dim_idx
            else local_q_len
        )
        global_kv_len = local_kv_len * kv_seq_split_num

        return self._compute_sparse_params(
            sparse_mode, pre_tokens, next_tokens,
            split_id, seq_split_num, local_q_len, global_q_len, global_kv_len,
        )

    def _adjust_tnd_layout_params(
        self,
        query, key,
        query_layout: Layout,
        key_layout: Optional[Layout],
        input_layout: str,
        sparse_mode: int,
        pre_tokens: int,
        next_tokens: int,
        actual_seq_qlen: Optional[List[int]],
        actual_seq_kvlen: Optional[List[int]],
        seq_split_num: int,
        split_id: int,
        kv_seq_split_num: int,
        is_dynamic: bool,
    ) -> Tuple:
        """Adjust parameters for TND layout including CP and DP modes."""
        batch_split_num, s1_split_num = self._calculate_tnd_split_params(
            query_layout, key_layout, input_layout
        )

        if s1_split_num > 1:
            if is_dynamic:
                if sparse_mode != SPARSE_RIGHT_DOWN_CAUSAL:
                    raise ValueError(
                        f"TND layout with context parallelism "
                        f"(s1_split_num={s1_split_num} > 1) requires "
                        f"sparse_mode={SPARSE_RIGHT_DOWN_CAUSAL}, "
                        f"but got {sparse_mode}"
                    )
            else:
                query_global_t = query.shape[0] * seq_split_num
                key_global_t = key.shape[0] * kv_seq_split_num
                self._validate_tnd_cp_requirements(
                    query_layout, key_layout, input_layout,
                    sparse_mode,
                    batch_split_num, s1_split_num,
                    (query_global_t, *query.shape[1:]),
                    (key_global_t, *key.shape[1:])
                )
        else:
            if not is_dynamic and query.shape[0] != key.shape[0]:
                raise ValueError(
                    f"TND layout with DP-only (s1_split_num=1) requires "
                    f"Query and Key to have the same local T-dimension, "
                    f"but got:\n"
                    f"Query local T-dim: {query.shape[0]}\n"
                    f"Key local T-dim: {key.shape[0]}\n"
                    f"batch_split_num: {batch_split_num}, "
                    f"s1_split_num: {s1_split_num}"
                )

        if actual_seq_qlen is None or actual_seq_kvlen is None:
            raise ValueError(
                "When using TND layout with sequence parallelism, "
                "actual_seq_qlen and actual_seq_kvlen must be provided."
            )

        kv_is_sharded = kv_seq_split_num > 1
        adjusted_actual_seq_qlen, adjusted_actual_seq_kvlen = (
            self._adjust_actual_seq_len_for_tnd_cp(
                query, key, actual_seq_qlen, actual_seq_kvlen,
                split_id, kv_is_sharded
            )
        )

        return (sparse_mode, pre_tokens, next_tokens,
                adjusted_actual_seq_qlen, adjusted_actual_seq_kvlen)

    def get_expand_impl(
        self,
        func: callable,
        output_layouts: Tuple[Layout, ...],
        input_layouts: List[Optional[Layout]],
        extra_args: List[Any],
    ) -> Optional[callable]:
        """Create expanded implementation.

        extra_args contains 8 scalar params extracted by _process_args_and_kwargs:
            [head_num, keep_prob, scale_value, pre_tokens, next_tokens,
             inner_precise, input_layout, sparse_mode]
        The returned expanded_impl receives 18 positional args matching the
        pyboost flat arg list (10 tensor args + 8 scalar args).
        """
        query_layout = input_layouts[0]
        if query_layout is None:
            return None

        if len(input_layouts) >= 3:
            key_layout_check = input_layouts[1]
            value_layout_check = input_layouts[2]

            if (key_layout_check is not None and value_layout_check is not None and
                hasattr(key_layout_check, 'tensor_map') and
                hasattr(value_layout_check, 'tensor_map')):
                if key_layout_check.tensor_map != value_layout_check.tensor_map:
                    raise ValueError(
                        f"Key and Value must have identical sharding strategies.\n"
                        f"Key tensor_map: {key_layout_check.tensor_map}\n"
                        f"Value tensor_map: {value_layout_check.tensor_map}"
                    )

        head_num = int(self._to_python_scalar(extra_args[EA_HEAD_NUM]))
        pre_tokens = int(self._to_python_scalar(extra_args[EA_PRE_TOKENS]))
        next_tokens = int(self._to_python_scalar(extra_args[EA_NEXT_TOKENS]))
        sparse_mode = int(self._to_python_scalar(extra_args[EA_SPARSE_MODE]))
        input_layout = _resolve_input_layout(
            self._to_python_scalar(extra_args[EA_INPUT_LAYOUT])
        )

        def expanded_impl(
            query, key, value,
            real_shift, drop_mask, padding_mask, attn_mask, prefix,
            actual_seq_qlen, actual_seq_kvlen,
            p_head_num, p_keep_prob, p_scale_value,
            p_pre_tokens, p_next_tokens, p_inner_precise,
            p_input_layout, p_sparse_mode,
        ):
            key_layout = input_layouts[1]
            self._validate_sharding_consistency(
                query_layout, key_layout, input_layout
            )

            is_varlen = input_layout == "TND" and actual_seq_qlen is not None
            self._validate_attn_mask(attn_mask, sparse_mode, input_layout, is_varlen)
            self._validate_real_shift_configuration(real_shift, sparse_mode)

            split_info = self._get_split_info(query_layout, input_layout)
            head_split_num = split_info["head"]
            seq_split_num = split_info["seq"]

            if head_split_num == 1 and seq_split_num == 1:
                result = func(
                    query, key, value,
                    real_shift, drop_mask, padding_mask, attn_mask, prefix,
                    actual_seq_qlen, actual_seq_kvlen,
                    p_head_num, p_keep_prob, p_scale_value,
                    p_pre_tokens, p_next_tokens, p_inner_precise,
                    p_input_layout, p_sparse_mode,
                )
                return self._truncate_result(result)

            adjusted_head_num = self._adjust_head_num(head_num, head_split_num)

            adjusted_sparse_mode = sparse_mode
            adjusted_pre_tokens = pre_tokens
            adjusted_next_tokens = next_tokens
            adjusted_actual_seq_qlen = actual_seq_qlen
            adjusted_actual_seq_kvlen = actual_seq_kvlen

            if seq_split_num > 1:
                dynamic_info = self._get_dynamic_shape_info(
                    query, key, input_layout
                )
                is_dynamic = dynamic_info.get('is_dynamic', False)

                split_id = self._get_split_id(query_layout, input_layout)
                seq_dim_idx = self._get_seq_dim_idx(
                    self._layout_dims.get(input_layout, {})
                )

                if seq_dim_idx is None:
                    raise ValueError(
                        f"Cannot infer seq/total dim for "
                        f"input_layout={input_layout}"
                    )

                kv_seq_split_num = 1
                if key_layout is not None:
                    kv_split_info = self._get_split_info(
                        key_layout, input_layout
                    )
                    kv_seq_split_num = kv_split_info["seq"]

                self._check_seq_sharding_compatibility(
                    query_layout, key_layout, input_layout,
                    seq_dim_idx, seq_split_num, kv_seq_split_num
                )

                (adjusted_sparse_mode,
                 adjusted_pre_tokens,
                 adjusted_next_tokens) = self._compute_adjusted_sparse_params(
                    query, key,
                    sparse_mode, pre_tokens, next_tokens,
                    split_id, seq_split_num, seq_dim_idx,
                    kv_seq_split_num, is_dynamic,
                )

                if input_layout == "TND":
                    (adjusted_sparse_mode,
                     adjusted_pre_tokens,
                     adjusted_next_tokens,
                     adjusted_actual_seq_qlen,
                     adjusted_actual_seq_kvlen) = self._adjust_tnd_layout_params(
                        query, key, query_layout, key_layout,
                        input_layout, sparse_mode,
                        pre_tokens, next_tokens,
                        actual_seq_qlen, actual_seq_kvlen,
                        seq_split_num, split_id,
                        kv_seq_split_num, is_dynamic,
                    )

            result = func(
                query, key, value,
                real_shift, drop_mask, padding_mask, attn_mask, prefix,
                adjusted_actual_seq_qlen, adjusted_actual_seq_kvlen,
                int(adjusted_head_num), p_keep_prob, p_scale_value,
                int(adjusted_pre_tokens), int(adjusted_next_tokens), p_inner_precise,
                p_input_layout, int(adjusted_sparse_mode),
            )

            return self._truncate_result(result)

        return expanded_impl

    def _get_seq_dim_idx(self, dims: dict) -> Optional[int]:
        """Get the sequence dimension index."""
        if "seq" in dims:
            return dims["seq"]
        if "total" in dims:
            return dims["total"]
        return None

    def _normalize_dim_map(self, dim_map):
        """Normalize dim_map."""
        if dim_map is None:
            return "None"
        return dim_map

    def _get_split_info(self, layout: Layout, input_layout_str: str):
        """Extract split information from layout."""
        dims = self._layout_dims.get(input_layout_str, {})
        result = {"batch": 1, "head": 1, "seq": 1}

        if getattr(layout, "alias_tensor_map", None) is None:
            return result

        if "batch" in dims:
            result["batch"] = self._get_dim_split_num(layout, dims["batch"])

        if "head" in dims:
            result["head"] = self._get_dim_split_num(layout, dims["head"])
        elif "hidden" in dims:
            result["head"] = self._get_dim_split_num(layout, dims["hidden"])

        if "seq" in dims:
            result["seq"] = self._get_dim_split_num(layout, dims["seq"])
        elif "total" in dims:
            result["seq"] = self._get_dim_split_num(layout, dims["total"])

        return result

    def _calculate_tnd_split_params(
        self,
        query_layout: Layout,
        key_layout: Layout,
        input_layout: str
    ) -> Tuple[int, int]:
        """Calculate batch_split_num and s1_split_num for TND layout."""
        if input_layout != "TND":
            return 1, 1

        query_split_info = self._get_split_info(query_layout, input_layout)
        key_split_info = self._get_split_info(key_layout, input_layout)

        query_seq_split = query_split_info["seq"]
        key_seq_split = key_split_info["seq"]

        batch_split_num = key_seq_split

        if batch_split_num == 0:
            s1_split_num = query_seq_split
        else:
            s1_split_num = query_seq_split // batch_split_num

        return batch_split_num, s1_split_num

    def _get_dim_split_num(self, layout: Layout, dim_idx: int) -> int:
        """Get split number along a tensor dimension."""
        if getattr(layout, "alias_tensor_map", None) is None:
            return 1

        if dim_idx >= len(layout.alias_tensor_map):
            return 1

        dim_map = self._normalize_dim_map(layout.alias_tensor_map[dim_idx])

        if dim_map == "None":
            return 1

        if isinstance(dim_map, str):
            return layout.mesh.get_device_num_along_axis(dim_map)

        if isinstance(dim_map, tuple):
            total = 1
            for axis_name in dim_map:
                axis_name = self._normalize_dim_map(axis_name)
                if axis_name != "None":
                    total *= layout.mesh.get_device_num_along_axis(axis_name)
            return total

        return 1

    def _get_split_id(self, layout: Layout, input_layout_str: str) -> int:
        """Get split ID along the sequence dimension."""
        dims = self._layout_dims.get(input_layout_str, {})
        seq_dim_idx = self._get_seq_dim_idx(dims)

        if seq_dim_idx is None or getattr(layout, "alias_tensor_map", None) is None:
            return 0

        if seq_dim_idx >= len(layout.alias_tensor_map):
            return 0

        dim_map = self._normalize_dim_map(layout.alias_tensor_map[seq_dim_idx])

        if dim_map == "None":
            return 0

        if isinstance(dim_map, str):
            rank = platform.get_rank()
            rank_list = layout.mesh.get_rank_list_along_axis(dim_map)
            if rank in rank_list:
                return rank_list.index(rank)
            return 0

        if isinstance(dim_map, tuple):
            non_none_axes = [
                ax for ax in dim_map if self._normalize_dim_map(ax) != "None"
            ]
            if len(non_none_axes) == 0:
                return 0
            if len(non_none_axes) > 1:
                warnings.warn(
                    f"Seq dim is sharded by multiple axes {non_none_axes}. "
                    f"Using the last axis for split_id calculation."
                )
            axis_name = non_none_axes[-1]
            rank = platform.get_rank()
            rank_list = layout.mesh.get_rank_list_along_axis(axis_name)
            if rank in rank_list:
                return rank_list.index(rank)

        return 0

    def _validate_tnd_cp_requirements(
        self,
        query_layout: Layout,
        key_layout: Layout,
        input_layout: str,
        sparse_mode: int,
        batch_split_num: int,
        s1_split_num: int,
        query_global_shape: Tuple[int, ...],
        key_global_shape: Tuple[int, ...]
    ):
        """Validate requirements for TND+CP mode."""
        if input_layout != "TND" or s1_split_num <= 1:
            return

        if sparse_mode != SPARSE_RIGHT_DOWN_CAUSAL:
            raise ValueError(
                f"TND layout with context parallelism "
                f"(s1_split_num={s1_split_num} > 1) "
                f"requires sparse_mode={SPARSE_RIGHT_DOWN_CAUSAL} "
                f"(rightDownCausal), but got sparse_mode={sparse_mode}.\n"
                f"This is required by MindSpore for correct attention "
                f"mask partitioning."
            )

        if query_global_shape[0] != key_global_shape[0]:
            raise ValueError(
                f"TND layout with context parallelism requires Query and Key "
                f"to have the same global T-dimension, but got:\n"
                f"Query global T-dim: {query_global_shape[0]}\n"
                f"Key global T-dim: {key_global_shape[0]}\n"
                f"Note: This checks the global shape before sharding, "
                f"not local shape."
            )

        query_split_info = self._get_split_info(query_layout, input_layout)
        key_split_info = self._get_split_info(key_layout, input_layout)

        query_seq_split = query_split_info["seq"]
        key_seq_split = key_split_info["seq"]

        if query_seq_split != key_seq_split * s1_split_num:
            raise ValueError(
                f"TND layout requires Query split number to be an integer "
                f"multiple of Key split number.\n"
                f"Query T-dimension split: {query_seq_split}\n"
                f"Key T-dimension split: {key_seq_split}\n"
                f"s1_split_num: {s1_split_num}\n"
                f"Expected: {query_seq_split} == {key_seq_split} x "
                f"{s1_split_num} = {key_seq_split * s1_split_num}\n"
                f"This ensures proper alignment for context parallel."
            )

    def _create_replicated_scalar_layout(self, query_layout: Layout) -> Layout:
        """Create a fully replicated layout for scalar tensor."""
        layout = Layout.from_device_mesh(query_layout.mesh)
        mesh_ndim = len(query_layout.mesh_shape)
        replicated_placements = tuple(Replicate() for _ in range(mesh_ndim))
        layout.set_placements(replicated_placements)
        layout.set_tensor_map(())
        return layout
