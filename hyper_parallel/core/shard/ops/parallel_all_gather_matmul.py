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
"""Distributed implementation for AllGatherMatmul (MC2 fusion) operator."""
import copy
from typing import Tuple

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp


def _normalize_agm_args(
        x1,
        x2,
        group,
        world_size,
        bias=None,
        gather_index=0,
        gather_output=True,
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
        bias: Must be None (MS constraint; bias is not supported).
        gather_index: Index for gather operation (only 0 is supported).
        gather_output: Whether to return the gathered intermediate tensor.
        comm_turn: Communication turn (only 0 is supported).
        trans_input: Must be False (MS constraint).
        trans_x2: If True, x2 physical shape is (n, k); MindSpore transposes x2 before CANN call.

    Returns:
        tuple: (positional_args_tuple, empty_kwargs_dict)
    """
    return (x1, x2, group, world_size, bias, gather_index, gather_output, comm_turn, trans_input, trans_x2), {}


class AllGatherMatmulDistributedOp(DistributedOp):
    """Distributed operator for mindspore.ops.all_gather_matmul (MC2 fusion).

    The CANN AllGatherMatmul kernel handles communication (AllGather) internally.
    HyperParallel's role is to:
    1. Extract local tensors from DTensor inputs via to_local().
    2. Infer output DTensor layouts so downstream operators can correctly
       understand the distribution state of the output.

    Shape transformation (logical, after any transpose):
        x1 (m_local, k_local) —[CANN AllGather on m]→ (m_global, k_local) —[matmul]→ output (m_global, n_local)
        gather_out (m_global, k_local) — valid only when gather_output=True

    Sharding constraints:
        - x1 physical (m, k): dim 0 is m (AllGather consumes m); dim 1 (k) may be Replicate or Shard.
        - x1's m-dim tensor_map must not be a tuple (joint sharding across multiple mesh dims unsupported).
        - x1 k-dim and x2 k-dim must share the same placement (both Replicate, or both sharded on the
          same mesh axis). trans_x2=False: x2 k is dim 0; trans_x2=True: x2 k is dim 1.
        - When k is sharded, output carries Partial(sum) status on the k-dim mesh axis; the caller is
          responsible for applying AllReduce to obtain the correct full result.
        - gather_out k-dim follows x1's k-dim placement.
        - Partial inputs are not allowed.
        - gather_index=0, trans_input=False, bias=None only (current MS constraints).
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
                cache_values = [x1_layout, x2_layout, trans_x2, gather_output].
        """
        norm_args, _ = _normalize_agm_args(*args, **kwargs)
        x1 = norm_args[0]
        x2 = norm_args[1]
        gather_output = norm_args[6]
        trans_x2 = norm_args[9]

        # MindSpore all_gather_matmul: only (input, x2, group, world_size) are
        # positional; bias, gather_index, gather_output, comm_turn, trans_input,
        # trans_x2 are keyword-only (after the '*' separator).
        local_args = (
            x1.to_local(),
            x2.to_local(),
            norm_args[2],   # group
            norm_args[3],   # world_size
        )
        local_kwargs = {
            'bias': norm_args[4],
            'gather_index': norm_args[5],
            'gather_output': gather_output,
            'comm_turn': norm_args[7],
            'trans_input': norm_args[8],
            'trans_x2': trans_x2,
        }

        cache_values = [
            x1.layout,
            x2.layout,
            trans_x2,
            gather_output,
        ]
        return local_args, local_kwargs, cache_values

    @staticmethod
    def _set_partial_from_k(output_layout: Layout, k_placement, op: str = 'sum') -> None:
        """Set Partial on output_layout for the mesh axes corresponding to k_placement.

        Args:
            output_layout: Layout to mark as Partial.
            k_placement: Tensor_map value for the k dimension (integer or tuple of integers).
            op: Reduction operation, default 'sum'.
        """
        alias = output_layout.alias_name
        n = len(alias)
        if isinstance(k_placement, tuple):
            for v in k_placement:
                output_layout.set_partial_by_dev_axis(alias[n - 1 - v], op)
        else:
            output_layout.set_partial_by_dev_axis(alias[n - 1 - k_placement], op)

    @staticmethod
    def _validate_input_layouts(
            x1_layout: Layout,
            x2_layout: Layout,
            trans_x2: bool,
    ) -> None:
        """Validate sharding constraints for AllGatherMatmul inputs.

        Args:
            x1_layout: Layout of x1. Physical (m, k); trans_input=False only.
            x2_layout: Layout of x2 (k, n) if trans_x2=False, or (n, k) if trans_x2=True.
            trans_x2: Whether x2 is transposed.

        Raises:
            ValueError: If x1's m-dim tensor_map is a tuple, the k-dim placements of x1 and x2
                do not match, or any input has Partial status.
        """
        op = "all_gather_matmul"
        x1_tm = x1_layout.tensor_map
        x2_tm = x2_layout.tensor_map

        # trans_input=False only: x1 physical (m, k) — k is dim 1, m is dim 0.
        x1_m_dim = 0

        if isinstance(x1_tm[x1_m_dim], tuple):
            raise ValueError(
                f"For {op}, x1 m-dim (dim {x1_m_dim}) "
                f"with tensor_map={x1_tm[x1_m_dim]} is jointly sharded across multiple "
                f"mesh dims, which is not supported in this version."
            )

        # k-dim placement must match between x1 and x2.
        x2_k_dim = 1 if trans_x2 else 0
        if x1_tm[1] != x2_tm[x2_k_dim]:
            raise ValueError(
                f"For {op}, x1 k-dim (dim 1) placement {x1_tm[1]} must match "
                f"x2 k-dim (dim {x2_k_dim}) placement {x2_tm[x2_k_dim]} "
                f"(trans_x2={trans_x2})."
            )

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """Infer output layouts for (output, gather_out).

        AllGather on m dim: output dim 0 is always -1 (Replicate), because
        AllGather unconditionally makes the m dimension global.

        n dim: follows x2's n placement.
          - trans_x2=False: n is x2 dim 1 → output_tm[1] = x2_tm[1]
          - trans_x2=True:  n is x2 dim 0 → output_tm[1] = x2_tm[0]

        k dim (contraction): when k is sharded, output carries Partial(sum) on the
          k-dim mesh axis; the caller must apply AllReduce to get the correct result.

        gather_out layout: m is Replicate (-1); k follows x1's k-dim placement.

        Args:
            cache_values: [x1_layout, x2_layout, trans_x2, gather_output]

        Returns:
            tuple: ((output_layout, gather_out_layout), None)

        Raises:
            ValueError: If any input has Partial status or sharding constraints are violated.
        """
        x1_layout = cache_values[0]
        x2_layout = cache_values[1]
        trans_x2 = cache_values[2]
        gather_output = cache_values[3]

        self._check_partial_inputs([x1_layout, x2_layout])
        self._validate_input_layouts(x1_layout, x2_layout, trans_x2)

        x1_tm = x1_layout.tensor_map
        x2_tm = x2_layout.tensor_map
        k_placement = x1_tm[1]
        n_placement = x2_tm[0] if trans_x2 else x2_tm[1]

        # output: m is Replicate (-1) because AllGather consumed the m sharding;
        # n inherits from x2's n dim.
        output_layout = Layout.from_device_mesh(x1_layout.mesh)
        output_layout.set_tensor_map((-1, n_placement))
        output_layout.tensor_map_to_placement()

        # When k is sharded, output is a partial sum; mark Partial so the framework
        # can insert AllReduce downstream.
        if k_placement != -1:
            self._set_partial_from_k(output_layout, k_placement)

        # gather_out: gather_output=True → m Replicate (-1), k follows x1's k placement.
        # gather_output=False → CANN returns a 1-D empty tensor; force all-Replicate so
        # the layout is compatible with any tensor rank returned by the kernel.
        gather_out_layout = Layout.from_device_mesh(x1_layout.mesh)
        gather_k = k_placement if gather_output else -1
        gather_out_layout.set_tensor_map((-1, gather_k))
        gather_out_layout.tensor_map_to_placement()

        return (copy.deepcopy(output_layout), copy.deepcopy(gather_out_layout)), None
