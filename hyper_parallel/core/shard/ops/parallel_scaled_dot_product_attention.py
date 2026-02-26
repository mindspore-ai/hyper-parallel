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

from typing import List, Tuple, Optional, Any
from hyper_parallel.core.layout import Layout
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_ops_register import register_distributed_op
from hyper_parallel.platform import get_platform

platform = get_platform()
Tensor = platform.Tensor


class ScaledDotProductAttentionDistributedOp:
    """Distributed operator for torch.nn.functional.scaled_dot_product_attention.

    Input shape: [B, N, S, D] (4D) or [N, S, D] (3D).
    Output: single Tensor with the same shape as query.

    Supported parallelism:
      - DP: Shard batch dimension (4D only)
      - MP: Shard head dimension
      - SP: Shard Q sequence dimension, KV replicated
      - Combinations: DP+MP, SP+MP, DP+SP+MP
    """

    def __init__(self, op_name: str):
        self.op_name = op_name
        register_distributed_op(op_name, self)

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

    def _normalize_dim_map(self, dim_map):
        """Normalize dim_map to string representation."""
        if dim_map is None:
            return "None"
        return dim_map

    def _get_dims(self, query_layout: Layout) -> dict:
        """Determine dimension mapping based on tensor rank.

        4D [B, N, S, D]: batch=0, head=1, seq=2, dim=3
        3D [N, S, D]:    head=0, seq=1, dim=2
        """
        tm = query_layout.tensor_map
        if tm is not None and len(tm) == 3:
            return {"head": 0, "seq": 1, "dim": 2}
        return {"batch": 0, "head": 1, "seq": 2, "dim": 3}

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

    def _validate_sharding_consistency(
        self,
        query_layout: Layout,
        key_layout: Optional[Layout],
        dims: dict,
    ):
        """Validate Q/K/V sharding consistency on non-sequence dimensions."""
        if key_layout is None or not hasattr(key_layout, 'tensor_map'):
            return

        q_tm = query_layout.tensor_map
        k_tm = key_layout.tensor_map

        if q_tm is None or k_tm is None:
            return

        for dim_name in ("batch", "head", "dim"):
            if dim_name not in dims:
                continue

            dim_idx = dims[dim_name]
            if dim_idx >= len(q_tm) or dim_idx >= len(k_tm):
                continue

            q_shard = self._normalize_dim_map(q_tm[dim_idx])
            k_shard = self._normalize_dim_map(k_tm[dim_idx])

            if q_shard != k_shard:
                raise ValueError(
                    f"Query and Key/Value must have identical {dim_name} "
                    f"sharding strategy.\n"
                    f"Query {dim_name} sharding (dim {dim_idx}): {q_shard}\n"
                    f"Key/Value {dim_name} sharding (dim {dim_idx}): {k_shard}\n"
                    f"Query tensor_map: {q_tm}\n"
                    f"Key tensor_map: {k_tm}"
                )

    def _build_causal_mask_for_chunk(
        self,
        local_q_len: int,
        global_kv_len: int,
        split_id: int,
        device,
    ) -> Tensor:
        """Build causal attention mask for a local Q chunk against full KV.

        For global Q position (split_id * local_q_len + j), causal mask allows
        attending to KV positions [0, split_id * local_q_len + j].
        Returns a bool mask where True means allow attention.
        """
        import torch   # pylint: disable=import-outside-toplevel

        offset = split_id * local_q_len
        q_positions = torch.arange(local_q_len, device=device).unsqueeze(1) + offset
        kv_positions = torch.arange(global_kv_len, device=device).unsqueeze(0)
        return kv_positions <= q_positions

    def _adjust_attn_mask_for_sp(
        self,
        attn_mask: Optional[Tensor],
        is_causal: bool,
        split_id: int,
        local_q_len: int,
        seq_split_num: int,
        global_kv_len: int,
        device,
    ) -> Tuple[Optional[Tensor], bool]:
        """Adjust attn_mask and is_causal for sequence parallelism.

        For is_causal=True: constructs explicit causal mask for the local Q chunk,
        since the standard lower-triangular pattern no longer applies after splitting.

        For explicit attn_mask with global Q dimension: slices to local Q range.

        Returns (adjusted_attn_mask, adjusted_is_causal).
        """
        if is_causal:
            causal_mask = self._build_causal_mask_for_chunk(
                local_q_len, global_kv_len, split_id, device,
            )
            return causal_mask, False

        if attn_mask is not None:
            global_q_len = local_q_len * seq_split_num
            if attn_mask.shape[-2] == global_q_len:
                offset = split_id * local_q_len
                if attn_mask.dim() == 2:
                    return attn_mask[offset:offset + local_q_len, :], False
                if attn_mask.dim() == 4:
                    return attn_mask[:, :, offset:offset + local_q_len, :], False

        return attn_mask, is_causal

    def infer_layout(
        self, input_layouts: List[Optional[Layout]], extra_args: List[Any]
    ) -> Tuple[Layout, ...]:
        """Infer output layout. Output has the same layout as query."""
        query_layout = input_layouts[0]
        if query_layout is None:
            raise ValueError("Query layout cannot be None")

        attention_out_layout = copy.deepcopy(query_layout)
        if attention_out_layout.placements is None and attention_out_layout.tensor_map is not None:
            placements = self._tensor_map_to_placements(
                attention_out_layout, attention_out_layout.tensor_map
            )
            attention_out_layout.set_placements(placements)

        return attention_out_layout

    def get_expand_impl(
        self,
        func: callable,
        output_layouts: Tuple[Layout, ...],
        input_layouts: List[Optional[Layout]],
        extra_args: List[Any],
    ) -> Optional[callable]:
        """Create expanded implementation."""
        query_layout = input_layouts[0]
        if query_layout is None:
            return None

        if len(input_layouts) >= 3:
            key_layout = input_layouts[1]
            value_layout = input_layouts[2]

            if (key_layout is not None and value_layout is not None and
                hasattr(key_layout, 'tensor_map') and hasattr(value_layout, 'tensor_map')):
                if key_layout.tensor_map != value_layout.tensor_map:
                    raise ValueError(
                        f"Key and Value must have identical sharding strategies.\n"
                        f"Key tensor_map: {key_layout.tensor_map}\n"
                        f"Value tensor_map: {value_layout.tensor_map}"
                    )

        dims = self._get_dims(query_layout)

        def expanded_impl(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            enable_gqa=False,
        ):
            key_layout = input_layouts[1] if len(input_layouts) > 1 else None
            self._validate_sharding_consistency(query_layout, key_layout, dims)

            split_info = self._get_split_info(query_layout, dims)
            seq_split_num = split_info["seq"]

            adjusted_attn_mask = attn_mask
            adjusted_is_causal = is_causal

            if seq_split_num > 1:
                kv_seq_split_num = 1
                if key_layout is not None:
                    kv_split_info = self._get_split_info(key_layout, dims)
                    kv_seq_split_num = kv_split_info["seq"]

                if kv_seq_split_num > 1:
                    raise NotImplementedError(
                        "KV sequence sharding is not supported for SDPA "
                        "without Ring Attention.\n"
                        f"Query sequence split num: {seq_split_num}\n"
                        f"Key/Value sequence split num: {kv_seq_split_num}"
                    )

                split_id = self._get_split_id(query_layout, dims)
                local_q_len = query.shape[dims["seq"]]
                global_kv_len = key.shape[dims["seq"]]

                adjusted_attn_mask, adjusted_is_causal = self._adjust_attn_mask_for_sp(
                    attn_mask, is_causal,
                    split_id, local_q_len, seq_split_num,
                    global_kv_len, query.device,
                )

            return func(
                query, key, value,
                attn_mask=adjusted_attn_mask,
                dropout_p=dropout_p,
                is_causal=adjusted_is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

        return expanded_impl
