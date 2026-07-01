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
"""ScaledDotProductAttention Distributed Operator"""

import copy
import warnings

from typing import Tuple, Optional
from hyper_parallel.core.shard.ops.parallel_npu_flash_attention_score import (  # pylint: disable=C0415
    _get_lb_override,
)
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.shard.ops.parallel_ops import DistributedOp
from hyper_parallel.platform import get_platform

platform = get_platform()
Tensor = platform.Tensor


def _normalize_sdpa_args(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
                         enable_gqa=False):
    return (query, key, value, attn_mask, dropout_p, is_causal, scale), {'enable_gqa': enable_gqa}


class ScaledDotProductAttentionDistributedOp(DistributedOp):
    """Distributed operator for torch.nn.functional.scaled_dot_product_attention.

    Input shape: [B, N, S, D] (4D) or [N, S, D] (3D).
    Output: single Tensor with the same shape as query.

    Supported parallelism:
      - DP: Shard batch dimension (4D only)
      - MP: Shard head dimension
      - SP: Shard Q sequence dimension, KV replicated
      - Combinations: DP+MP, SP+MP, DP+SP+MP

    Note:
        The ``enable_gqa`` flag is passed through to the underlying SDPA call,
        but distributed GQA correctness (Q_heads % KV_heads == 0, per-rank
        local-head grouping integrity, and Q/K head-shard correspondence) is
        **not** validated by the current layout inference.
    """

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for ScaledDotProductAttention operator.

        Args:
            args (tuple): Positional arguments (query, key, value, ...).
            kwargs (dict): Keyword arguments.

        Returns:
            tuple: (local_args, local_kwargs, cache_values) where local_args contains
                local tensors and runtime scalars, and cache_values contains Layout objects.
        """
        args, kwargs = _normalize_sdpa_args(*args, **kwargs)
        query, key, value, attn_mask, dropout_p, is_causal, scale = args
        enable_gqa = kwargs['enable_gqa']

        if hasattr(attn_mask, '_layout'):
            raise NotImplementedError(
                f"For {self.op_name}, DTensor attn_mask is not supported yet."
            )

        local_args = (
            query.to_local() if hasattr(query, '_layout') else query,
            key.to_local() if hasattr(key, '_layout') else key,
            value.to_local() if hasattr(value, '_layout') else value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
        )
        local_kwargs = {'enable_gqa': enable_gqa}

        cache_values = [
            query.layout if hasattr(query, '_layout') else None,
            key.layout if hasattr(key, '_layout') else None,
            value.layout if hasattr(value, '_layout') else None,
        ]
        return local_args, local_kwargs, cache_values

    @staticmethod
    def _normalize_dim_map(dim_map):
        """Normalize dim_map to string representation."""
        if dim_map is None:
            return "None"
        return dim_map

    def _get_dims(self, query_layout: Layout) -> dict:
        """Determine dimension mapping based on tensor rank.

        4D [B, N, S, D]: batch=0, head=1, seq=2, dim=3
        3D [N, S, D]:    head=0, seq=1, dim=2
        """
        tm = query_layout.alias_tensor_map
        if tm is not None and len(tm) == 3:
            return {"head": 0, "seq": 1, "dim": 2}
        return {"batch": 0, "head": 1, "seq": 2, "dim": 3}

    def _get_dim_split_num(self, layout: Layout, dim_idx: int) -> int:
        """Get split number along a tensor dimension."""
        if getattr(layout, "alias_tensor_map", None) is None:
            return 1
        if dim_idx >= len(layout.alias_tensor_map):
            return 1

        dim_map = ScaledDotProductAttentionDistributedOp._normalize_dim_map(layout.alias_tensor_map[dim_idx])

        if dim_map == "None":
            return 1

        if isinstance(dim_map, str):
            return layout.mesh.get_device_num_along_axis(dim_map)

        if isinstance(dim_map, tuple):
            total = 1
            for axis_name in dim_map:
                axis_name = ScaledDotProductAttentionDistributedOp._normalize_dim_map(axis_name)
                if axis_name != "None":
                    total *= layout.mesh.get_device_num_along_axis(axis_name)
            return total

        return 1

    def _get_split_info(self, layout: Layout, dims: dict) -> dict:
        """Extract split information from layout."""
        result = {"batch": 1, "head": 1, "seq": 1}

        if getattr(layout, "alias_tensor_map", None) is None:
            return result

        if "batch" in dims:
            result["batch"] = self._get_dim_split_num(layout, dims["batch"])
        result["head"] = self._get_dim_split_num(layout, dims["head"])
        result["seq"] = self._get_dim_split_num(layout, dims["seq"])

        return result

    def _get_split_id(self, layout: Layout, dims: dict) -> int:
        """Get split ID along the sequence dimension."""
        seq_dim_idx = dims["seq"]

        if getattr(layout, "alias_tensor_map", None) is None:
            return 0
        if seq_dim_idx >= len(layout.alias_tensor_map):
            return 0

        dim_map = ScaledDotProductAttentionDistributedOp._normalize_dim_map(layout.alias_tensor_map[seq_dim_idx])

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
                ax for ax in dim_map if ScaledDotProductAttentionDistributedOp._normalize_dim_map(ax) != "None"
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

    def _validate_sharding_consistency(
        self,
        query_layout: Layout,
        key_layout: Optional[Layout],
        dims: dict,
    ):
        """Validate Q/K/V sharding consistency on non-sequence dimensions.

        Note:
            When GQA is enabled (``enable_gqa=True``), Q and K may have different
            numbers of heads while still sharing the same mesh axis mapping.
            Distributed GQA correctness (Q_heads % KV_heads == 0, per-rank local-head
            grouping integrity) is not yet validated.
        """
        if key_layout is None or not hasattr(key_layout, 'tensor_map'):
            return

        q_tm = query_layout.alias_tensor_map
        k_tm = key_layout.alias_tensor_map

        if q_tm is None or k_tm is None:
            return

        for dim_name in ("batch", "head", "dim"):
            if dim_name not in dims:
                continue

            dim_idx = dims[dim_name]
            if dim_idx >= len(q_tm) or dim_idx >= len(k_tm):
                continue

            q_shard = ScaledDotProductAttentionDistributedOp._normalize_dim_map(q_tm[dim_idx])
            k_shard = ScaledDotProductAttentionDistributedOp._normalize_dim_map(k_tm[dim_idx])

            if q_shard != k_shard:
                raise ValueError(
                    f"Query and Key/Value must have identical {dim_name} "
                    f"sharding strategy.\n"
                    f"Query {dim_name} sharding (dim {dim_idx}): {q_shard}\n"
                    f"Key/Value {dim_name} sharding (dim {dim_idx}): {k_shard}\n"
                    f"Query alias_tensor_map: {q_tm}\n"
                    f"Key alias_tensor_map: {k_tm}"
                )

    @staticmethod
    def _build_causal_mask_for_chunk(
        local_q_len: int,
        kv_len: int,
        split_id: int,
        device,
    ) -> Tensor:
        """Build causal attention mask for a local Q chunk.

        For global Q position (split_id * local_q_len + i), causal mask allows
        attending to KV positions [0, split_id * local_q_len + i].
        Returns a bool mask where True means allow attention.
        """
        import torch   # pylint: disable=import-outside-toplevel

        offset = split_id * local_q_len
        q_positions = torch.arange(local_q_len, device=device).unsqueeze(1) + offset
        kv_positions = torch.arange(kv_len, device=device).unsqueeze(0)
        return kv_positions <= q_positions

    def _adjust_attn_mask_for_sp(
        self,
        attn_mask: Optional[Tensor],
        is_causal: bool,
        key: Tensor,
        value: Tensor,
        split_id: int,
        local_q_len: int,
        seq_split_num: int,
        global_kv_len: int,
        seq_dim: int,
        device,
    ) -> Tuple[Optional[Tensor], bool, Tensor, Tensor]:
        """Adjust attn_mask, is_causal, and KV tensors for sequence parallelism.

        For is_causal=True: truncates KV to the causally relevant range via
        narrow(). For split_id=0 the truncated KV length equals Q length, so
        is_causal=True is preserved and the kernel uses its built-in fast path.
        For split_id>0, an explicit mask over the truncated KV range is built,
        which is smaller than the original full-length mask.

        For explicit attn_mask with global Q dimension: slices to local Q range.

        Returns (adjusted_attn_mask, adjusted_is_causal, adjusted_key, adjusted_value).
        """
        if is_causal:
            kv_end = min((split_id + 1) * local_q_len, global_kv_len)
            key = key.narrow(seq_dim, 0, kv_end)
            value = value.narrow(seq_dim, 0, kv_end)

            if split_id == 0:
                return None, True, key, value

            causal_mask = ScaledDotProductAttentionDistributedOp._build_causal_mask_for_chunk(
                local_q_len, kv_end, split_id, device,
            )
            return causal_mask, False, key, value

        if attn_mask is not None:
            global_q_len = local_q_len * seq_split_num
            if attn_mask.shape[-2] == global_q_len:
                offset = split_id * local_q_len
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask[offset:offset + local_q_len, :]
                elif attn_mask.dim() == 4:
                    attn_mask = attn_mask[:, :, offset:offset + local_q_len, :]

        return attn_mask, is_causal, key, value

    @staticmethod
    def _validate_input_layouts(query_layout, key_layout, value_layout, op_name):
        """Validate basic input constraints (Rules 1-3).

        Rule 1: Query layout must not be None.
        Rule 2: Query, Key, Value must not have Partial status.
        Rule 3: Only 3D or 4D inputs are supported; Q/K/V must have the same rank.

        Args:
            query_layout: Query Layout (must not be None).
            key_layout: Key Layout or None.
            value_layout: Value Layout or None.
            op_name: Operator name for error messages.

        Returns:
            int: query_ndim (number of dimensions of the query tensor).

        Raises:
            ValueError: If any rule is violated.
        """
        if query_layout is None:
            raise ValueError(
                f"For {op_name}, query layout should not be None, but got None."
            )

        query_ndim = len(query_layout.alias_tensor_map)
        if query_ndim not in (3, 4):
            raise ValueError(
                f"For {op_name}, only 3D or 4D inputs are supported, "
                f"but got query ndim={query_ndim}."
            )
        if key_layout is not None and len(key_layout.alias_tensor_map) != query_ndim:
            raise ValueError(
                f"For {op_name}, Query, Key and Value must have the same rank.\n"
                f"Query ndim: {query_ndim}\n"
                f"Key ndim: {len(key_layout.alias_tensor_map)}"
            )
        if value_layout is not None and len(value_layout.alias_tensor_map) != query_ndim:
            raise ValueError(
                f"For {op_name}, Query, Key and Value must have the same rank.\n"
                f"Query ndim: {query_ndim}\n"
                f"Value ndim: {len(value_layout.alias_tensor_map)}"
            )

        return query_ndim

    def _validate_tensor_strategy(self, query_layout, key_layout, value_layout, dims):
        """Validate mixed DTensor/plain Tensor configuration (Rule 4).

        When K/V are plain Tensors, Query batch/head must not be sharded.
        Key and Value must both be DTensors or both be plain Tensors.

        Args:
            query_layout: Query Layout.
            key_layout: Key Layout or None.
            value_layout: Value Layout or None.
            dims: Dimension mapping dict from _get_dims.

        Raises:
            ValueError: If the strategy is invalid.
        """
        if key_layout is None and value_layout is None:
            split_info = self._get_split_info(query_layout, dims)
            batch_split = split_info.get("batch", 1)
            head_split = split_info["head"]
            if batch_split > 1 or head_split > 1:
                raise ValueError(
                    f"For {self.op_name}, when Query is a DTensor but Key/Value are plain Tensors, "
                    f"Query batch and head dimensions must not be sharded.\n"
                    f"Query batch split: {batch_split}\n"
                    f"Query head split: {head_split}"
                )
        elif (key_layout is None) != (value_layout is None):
            raise ValueError(
                f"For {self.op_name}, Key and Value must both be DTensors or both be plain Tensors.\n"
                f"Key is {'DTensor' if key_layout is not None else 'plain Tensor'}\n"
                f"Value is {'DTensor' if value_layout is not None else 'plain Tensor'}"
            )

    @staticmethod
    def _validate_mesh_kv_identity(query_layout, key_layout, value_layout, op_name):
        """Validate mesh identity and KV sharding consistency (Rules 5, 7).

        Rule 5: Query, Key, Value must belong to the same mesh.
        Rule 7: Key and Value must have identical sharding strategies.

        Args:
            query_layout: Query Layout.
            key_layout: Key Layout or None.
            value_layout: Value Layout or None.
            op_name: Operator name for error messages.

        Raises:
            ValueError: If mesh mismatch or KV sharding mismatch detected.
        """
        q_mesh_hash = query_layout.mesh.to_hash()
        for name, layout in [("Key", key_layout), ("Value", value_layout)]:
            if layout is not None and layout.mesh.to_hash() != q_mesh_hash:
                raise ValueError(
                    f"For {op_name}, {name} mesh must match Query mesh.\n"
                    f"Query mesh: {query_layout.mesh}\n"
                    f"{name} mesh: {layout.mesh}"
                )

        key_map = getattr(key_layout, "alias_tensor_map", None) if key_layout is not None else None
        value_map = getattr(value_layout, "alias_tensor_map", None) if value_layout is not None else None
        if key_map != value_map:
            raise ValueError(
                f"For {op_name}, Key and Value must have identical sharding strategies.\n"
                f"Key alias_tensor_map: {key_map}\n"
                f"Value alias_tensor_map: {value_map}"
            )

    def _validate_sharding_forbidden(self, query_layout, key_layout, dims):
        """Validate forbidden sharding patterns (Rules 8, 9).

        Rule 8: KV sequence sharding is not allowed (no Ring Attention support).
        Rule 9: Sharding the last embedding dimension is not supported.

        Args:
            query_layout: Query Layout.
            key_layout: Key Layout or None.
            dims: Dimension mapping dict from _get_dims.

        Raises:
            NotImplementedError: If a forbidden sharding pattern is detected.
        """
        if key_layout is not None:
            kv_seq_split_num = self._get_split_info(key_layout, dims)["seq"]
            if kv_seq_split_num > 1:
                raise NotImplementedError(
                    f"For {self.op_name}, KV sequence sharding is not supported "
                    f"without Ring Attention.\n"
                    f"Key/Value sequence split num: {kv_seq_split_num}"
                )

        dim_split = self._get_dim_split_num(query_layout, dims["dim"])
        if dim_split > 1:
            raise NotImplementedError(
                f"For {self.op_name}, sharding the last embedding dimension is not supported.\n"
                f"Dim split num: {dim_split}"
            )

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for ScaledDotProductAttention operator.

        Rules:
            1. Query layout must not be None.
            2. Query, Key, Value must not have Partial status.
            3. Only 3D or 4D inputs are supported; Q/K/V must have the same rank.
            4. When K/V are plain Tensors, Query batch/head must not be sharded.
               Key and Value must both be DTensors or both be plain Tensors.
            5. Query, Key, Value must belong to the same mesh.
            6. Query and Key must have identical sharding on batch, head, and dim axes.
            7. Key and Value must have identical sharding strategies.
            8. KV sequence sharding is not allowed (no Ring Attention support).
            9. Sharding the last embedding dimension is not supported.
            10. Output layout is a deepcopy of query layout.

        Args:
            cache_values (list): [query_layout, key_layout, value_layout]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If any rule above is violated.
            NotImplementedError: If an unsupported sharding pattern is detected.
        """
        query_layout, key_layout, value_layout = cache_values

        # Rules 1-3: Basic input validation
        self._validate_input_layouts(query_layout, key_layout, value_layout, self.op_name)

        # Rule 2 continuation: Check Partial status
        if not self._allow_partial_inputs:
            self._check_partial_inputs(
                [layout for layout in (query_layout, key_layout, value_layout) if layout is not None]
            )

        dims = self._get_dims(query_layout)

        # Rule 4: Mixed DTensor/plain Tensor configuration
        self._validate_tensor_strategy(query_layout, key_layout, value_layout, dims)

        # Rules 5, 7: Mesh identity and KV sharding consistency
        self._validate_mesh_kv_identity(query_layout, key_layout, value_layout, self.op_name)

        # Rule 6: Query/Key sharding consistency on non-sequence dimensions
        self._validate_sharding_consistency(query_layout, key_layout, dims)

        # Rules 8, 9: Forbidden sharding patterns
        self._validate_sharding_forbidden(query_layout, key_layout, dims)

        # Rule 10: Output layout is a deepcopy of query layout
        attention_out_layout = copy.deepcopy(query_layout)
        if attention_out_layout.placements is None and attention_out_layout.tensor_map is not None:
            attention_out_layout.tensor_map_to_placement()

        return ((attention_out_layout,), None)

    # pylint: disable=W0237
    def get_expand_impl(self, func, infer_result, cache_values):
        """Create expanded implementation with sequence parallelism support.

        Args:
            func: Original operator callable.
            infer_result (tuple): ((output_layout,), None) from infer_layout.
            cache_values (list): [query_layout, key_layout, value_layout]

        Returns:
            callable | None: expanded_impl closure, or None if query_layout is None.
        """
        query_layout = cache_values[0]
        if query_layout is None:
            return None

        dims = self._get_dims(query_layout)

        def _expanded_impl(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            enable_gqa=False,
        ):
            split_info = self._get_split_info(query_layout, dims)
            seq_split_num = split_info["seq"]

            lb_split_id, lb_split_num = _get_lb_override()

            adjusted_attn_mask = attn_mask
            adjusted_is_causal = is_causal

            if seq_split_num > 1 or lb_split_id is not None:
                if lb_split_id is not None:
                    if lb_split_num is None:
                        raise ValueError("lb_split_num must not be None when lb_split_id is set")
                    split_id = lb_split_id
                    seq_split_num = lb_split_num
                else:
                    split_id = self._get_split_id(query_layout, dims)
                local_q_len = query.shape[dims["seq"]]
                global_kv_len = key.shape[dims["seq"]]

                adjusted_attn_mask, adjusted_is_causal, key, value = (
                    self._adjust_attn_mask_for_sp(
                        attn_mask, is_causal, key, value,
                        split_id, local_q_len, seq_split_num,
                        global_kv_len, dims["seq"], query.device,
                    )
                )

            return func(
                query, key, value,
                attn_mask=adjusted_attn_mask,
                dropout_p=dropout_p,
                is_causal=adjusted_is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

        return _expanded_impl
