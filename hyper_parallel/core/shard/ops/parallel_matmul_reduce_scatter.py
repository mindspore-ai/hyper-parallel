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
"""Distributed implementation for MatmulReduceScatter (MC2 fusion) operator."""
import copy
from typing import Tuple

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp


def _normalize_mrs_args(
        x1,
        x2,
        group,
        world_size,
        reduce_op='sum',
        bias=None,
        comm_turn=0,
        trans_input=False,
        trans_x2=False,
) -> Tuple[tuple, dict]:
    """Normalize positional and keyword arguments into a canonical positional tuple.

    Args:
        x1: Input tensor. Physical shape (m, k); trans_input=False only (MS constraint).
        x2: Weight tensor with shape (k, n) if trans_x2=False, or (n, k) if trans_x2=True.
        group: Communication group name string.
        world_size: Number of ranks in the communication group.
        reduce_op: Reduce operation string (only 'sum' is supported).
        bias: Must be None (MS constraint; bias is not supported).
        comm_turn: Communication turn (only 0 is supported).
        trans_input: Must be False (MS constraint).
        trans_x2: If True, x2 physical shape is (n, k); MindSpore transposes x2 before CANN call.

    Returns:
        tuple: (positional_args_tuple, empty_kwargs_dict)
    """
    return (x1, x2, group, world_size, reduce_op, bias, comm_turn, trans_input, trans_x2), {}


class MatmulReduceScatterDistributedOp(DistributedOp):
    """Distributed operator for mindspore.ops.matmul_reduce_scatter (MC2 fusion).

    The CANN MatmulReduceScatter kernel handles communication (ReduceScatter)
    internally. HyperParallel's role is to:
    1. Extract local tensors from DTensor inputs via to_local().
    2. Infer output DTensor layouts so downstream operators can correctly
       understand the distribution state of the output.

    No Partial state is needed because CANN internally completes the ReduceScatter;
    each rank receives the correctly reduced and scattered result slice.

    Shape transformation (logical, after any transpose):
        x1 (m, k_local), x2 (k_local, n) —[matmul]→ partial (m, n)
        —[CANN ReduceScatter on m]→ output (m_local = m / world_size, n)

    Sharding constraints:
        - x1 physical (m, k): dim 1 (k) must be Shard (TP/comm axis);
          dim 0 (m) may be Replicate or Shard (DP axis).
        - x2's k-dim must be Shard and match x1's k-dim placement exactly.
        - trans_x2=False: x2 k-dim is dim 0; trans_x2=True: x2 k-dim is dim 1.
        - trans_input=False, bias=None only (current MS constraints).
        - Partial inputs are not allowed.
    """

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """Extract local tensors and build the layout cache.

        Args:
            args: Positional arguments (DTensors for x1 and x2).
            kwargs: Keyword arguments.

        Returns:
            tuple: (local_args, local_kwargs, cache_values) where
                local_args contains extracted local tensors,
                local_kwargs contains keyword arguments,
                cache_values = [x1_layout, x2_layout, trans_x2].
        """
        norm_args, _ = _normalize_mrs_args(*args, **kwargs)
        x1 = norm_args[0]
        x2 = norm_args[1]
        trans_x2 = norm_args[8]

        # MindSpore matmul_reduce_scatter: only (input, x2, group, world_size) are
        # positional; reduce_op, bias, comm_turn, trans_input, trans_x2 are
        # keyword-only (after the '*' separator).
        local_args = (
            x1.to_local(),
            x2.to_local(),
            norm_args[2],   # group
            norm_args[3],   # world_size
        )
        local_kwargs = {
            'reduce_op': norm_args[4],
            'bias': norm_args[5],
            'comm_turn': norm_args[6],
            'trans_input': norm_args[7],
            'trans_x2': trans_x2,
        }

        cache_values = [
            x1.layout,
            x2.layout,
            trans_x2,
        ]
        return local_args, local_kwargs, cache_values

    @staticmethod
    def _validate_input_layouts(
            x1_layout: Layout,
            x2_layout: Layout,
            trans_x2: bool,
    ) -> None:
        """Validate sharding constraints for MatmulReduceScatter inputs.

        Args:
            x1_layout: Layout of x1. Physical (m, k); trans_input=False only.
            x2_layout: Layout of x2 (k, n) if trans_x2=False, or (n, k) if trans_x2=True.
            trans_x2: Whether x2 is transposed.

        Raises:
            ValueError: If x1's k dimension is Replicate, x2's k dimension layout does not
                match x1's k dimension layout, or any input has Partial status.
        """
        op = "matmul_reduce_scatter"
        x1_tm = x1_layout.tensor_map
        x2_tm = x2_layout.tensor_map

        # trans_input=False only: x1 physical (m, k) — k is dim 1.
        x1_k_dim = 1

        if x1_tm[x1_k_dim] == -1:
            raise ValueError(
                f"For {op}, x1 k-dim (dim {x1_k_dim}) must be "
                f"Shard (not Replicate), because ReduceScatter requires k to be sharded. "
                f"Got tensor_map={x1_tm}"
            )

        x2_k_dim = 1 if trans_x2 else 0
        if x2_tm[x2_k_dim] != x1_tm[x1_k_dim]:
            raise ValueError(
                f"For {op}, x2 dim {x2_k_dim} (k) layout must match x1 k-dim (dim {x1_k_dim}) "
                f"layout (trans_x2={trans_x2}), "
                f"but got x1_k={x1_tm[x1_k_dim]}, x2_k={x2_tm[x2_k_dim]}"
            )

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """Infer output layout for MatmulReduceScatter.

        ReduceScatter converts the k-dim sharding (TP comm axis) into m-dim sharding.
        trans_input=False only: x1 physical (m, k).

          comm_mesh_dim = x1_tm[1]   (k → TP / ReduceScatter scatter axis)
          dp_mesh_dim   = x1_tm[0]   (m → DP, or -1 if Replicate)

          If dp_mesh_dim == -1 (pure TP):
              output_tm[0] = comm_mesh_dim
          Else (TP + DP joint sharding):
              output_tm[0] = (dp_mesh_dim, comm_mesh_dim)
              dp_mesh_dim is the outer pre-existing sharding; comm_mesh_dim is the
              inner scatter added by ReduceScatter (within each DP group).

          output_tm[1] = x2's n dimension placement
            - trans_x2=False: x2 dim 1 (n) → x2_tm[1]
            - trans_x2=True:  x2 dim 0 (n) → x2_tm[0]

        Args:
            cache_values: [x1_layout, x2_layout, trans_x2]

        Returns:
            tuple: (output_layout, None)

        Raises:
            ValueError: If any input has Partial status or sharding constraints are violated.
        """
        x1_layout = cache_values[0]
        x2_layout = cache_values[1]
        trans_x2 = cache_values[2]

        self._check_partial_inputs([x1_layout, x2_layout])
        self._validate_input_layouts(x1_layout, x2_layout, trans_x2)

        x1_tm = x1_layout.tensor_map
        x2_tm = x2_layout.tensor_map

        # trans_input=False: k is dim 1 (comm axis), m is dim 0 (DP axis or Replicate).
        comm_mesh_dim = x1_tm[1]
        dp_mesh_dim = x1_tm[0]

        if dp_mesh_dim == -1:
            output_m = comm_mesh_dim
        else:
            output_m = (dp_mesh_dim, comm_mesh_dim)

        n_placement = x2_tm[0] if trans_x2 else x2_tm[1]

        output_layout = Layout.from_device_mesh(x1_layout.mesh)
        output_layout.set_tensor_map((output_m, n_placement))
        output_layout.tensor_map_to_placement()

        return copy.deepcopy(output_layout), None
