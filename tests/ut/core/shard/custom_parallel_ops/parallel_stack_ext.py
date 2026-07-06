# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
Distributed implementation for StackExt operator.
"""

from typing import Tuple

from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.shard.ops.parallel_ops import DistributedOp


def _normalize_stack_ext_args(tensors, axis=0, **kwargs):
    """Normalize arguments for StackExt.

    Args:
        tensors: Tuple/list of tensors to stack.
        axis: The axis along which to stack (default 0).

    Returns:
        tuple: ((tensors,), {'axis': axis})
    """
    return (tensors,), {'axis': axis}


class StackExtDistributedOp(DistributedOp):
    """Distributed implementation for StackExt operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """Preprocess arguments for StackExt.

        Calls ``_normalize_stack_ext_args`` to unify positional and keyword
        argument forms, then builds local_args and cache_values.

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
                cache_values: [layouts, axis]
        """
        args, kwargs = _normalize_stack_ext_args(*args, **kwargs)
        tensors = args[0]
        axis = kwargs['axis']

        local_tensors = tuple(
            tensor.to_local() if hasattr(tensor, "_layout") else tensor
            for tensor in tensors
        )
        layouts = tuple(
            tensor.layout if hasattr(tensor, "_layout") else None
            for tensor in tensors
        )

        local_args = (local_tensors, axis)
        cache_values = [layouts, axis]
        return local_args, {}, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """Infer output layout for StackExt.

        Args:
            cache_values: [layouts, axis] where ``layouts`` is a tuple of Layout
                objects and ``axis`` is the insertion dimension index.

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If layouts are inconsistent, axis is out of range, or
                input tensor sequence is empty.
        """
        layouts, axis = cache_values

        if not self._allow_partial_inputs:
            self._check_partial_inputs(layouts)

        valid_layouts = tuple(l for l in layouts if l is not None)

        if not valid_layouts:
            raise ValueError(
                f"Operation {self.op_name}: at least one input must be a DTensor"
            )

        base_layout = valid_layouts[0]
        in_rank = len(base_layout.tensor_map)
        out_rank = in_rank + 1

        if axis < -out_rank or axis >= out_rank:
            raise ValueError(
                f"Operation {self.op_name}: axis {axis} is out of range "
                f"for output rank {out_rank}"
            )

        if axis < 0:
            axis += out_rank

        for layout in valid_layouts[1:]:
            if layout.mesh_shape != base_layout.mesh_shape:
                raise ValueError(
                    f"Operation {self.op_name}: inputs must have the same mesh_shape"
                )

            if layout.tensor_map != base_layout.tensor_map:
                raise ValueError(
                    f"Operation {self.op_name}: inputs must have the same tensor_map"
                )

        out_alias_map = (
            list(base_layout.alias_tensor_map[:axis])
            + ["None"]
            + list(base_layout.alias_tensor_map[axis:])
        )

        output_layout = Layout(
            base_layout.mesh_shape,
            base_layout.alias_name,
            base_layout.rank_list,
        )(*out_alias_map)

        return (output_layout,), None
