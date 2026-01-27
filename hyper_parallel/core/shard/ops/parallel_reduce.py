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
Distributed implementation for Reduce operator.
"""

from typing import Sequence, Union, Tuple, List
from hyper_parallel.core.layout import Layout
from hyper_parallel.platform import get_platform
from .parallel_ops import DistributedOp
platform = get_platform()
Tensor = platform.Tensor


StrOrTuple = Union[str, Tuple["StrOrTuple", ...], List["StrOrTuple"]]


class ReduceExtDistributedOpBase(DistributedOp):
    """
    Base class for distributed reduce operators.

    Args:
        op_name (str): Name of the operator to register.
        partial_type (list): List of the operator for allreduce.
    """

    def __init__(self, op_name, partial_type=None):
        super().__init__(op_name)
        if partial_type is None:
            partial_type = ["sum"]
        self.partial_type = partial_type

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for reduce operator.

        Args:
            layouts (tuple): Layouts of input tensor.
            extra_args (dict): Additional arguments (dim, keepdim).

        Returns:
            tuple: Layout for output tensor.
        """
        if not layouts:
            raise ValueError(f"{self.__class__.__name__} requires at least one input layout")

        x_layout = layouts[0]

        if x_layout.mesh_shape is None:
            raise ValueError("Input layouts cannot be None.")

        # [dim, keepdim]
        if not extra_args:
            dim = None
            keepdim = False
        elif len(extra_args) == 1:
            dim = None
            keepdim = extra_args[0]
        else:
            dim, keepdim = extra_args

        if isinstance(dim, Tensor):
            raise TypeError(
                "The `dim` argument should not be a `Tensor`. Instead, use one of the following types: "
                "`None`, `int`, `tuple[int]`, or `list[int]`."
            )

        # Infer the output shape based on dim and keepdim
        output_layout = self._infer_output_layout(x_layout, dim, keepdim)

        return output_layout

    def _infer_output_layout(self, x_layout, dim, keepdim):
        """Infer output layout for reduce operator."""
        # Case 1: Handle dim as an empty tuple, meaning reduce all dimensions
        if dim is None:
            return self._handle_all_axis_reduce(x_layout, keepdim)

        # Case 2: Handle dim as int, tuple, or list, with keepdim True or False
        output_layout = Layout(
            mesh_shape=x_layout.mesh_shape,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list
        )
        x_map = x_layout.alias_tensor_map
        reduce_alias, x_map = self.replace_axis_with_none(dim, x_layout, keepdim)
        output_layout = output_layout(*x_map)
        self._apply_partial(output_layout, reduce_alias)
        return output_layout

    def _handle_all_axis_reduce(self, x_layout, keepdim):
        """Handle the case where dim is empty, meaning reduce all dimensions."""
        layout = Layout(
            mesh_shape=x_layout.mesh_shape,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list
        )

        if not keepdim:
            output_layout = layout()
        else:
            tensor_map = tuple(["None"] * len(x_layout.alias_tensor_map))
            output_layout = layout(*tensor_map)

        self._apply_partial(output_layout, x_layout.alias_tensor_map)
        return output_layout

    def replace_axis_with_none(self, dim, x_layout, keepdim):
        """Replace or drop dimensions depending on keepdim."""
        if not isinstance(dim, (tuple, list)):
            dim = [dim]
        else:
            dim = list(dim)

        rank = len(x_layout.alias_tensor_map)
        for i, axis_id in enumerate(dim):
            if axis_id < 0:
                dim[i] = rank + axis_id
            if not isinstance(axis_id, int) or dim[i] >= rank or dim[i] < 0:
                raise ValueError(f"Invalid reduce axis index {axis_id} at position {i}.")

        alias_tensor_map = x_layout.alias_tensor_map
        reduce_alias = [alias_tensor_map[axis_id] for axis_id in dim if
                        alias_tensor_map[axis_id] is not None and alias_tensor_map[axis_id] != "None"]
        reduce_alias = self._flatten_aliases(reduce_alias)

        if keepdim:
            return self._replace_keepdim(alias_tensor_map, reduce_alias)
        return self._replace_dropdim(alias_tensor_map, reduce_alias, dim)

    def _flatten_aliases(self, reduce_alias):
        """Flatten reduce_alias into a list of atomic alias strings."""
        flat = []
        for alias in reduce_alias:
            if isinstance(alias, (tuple, list)):
                flat.extend(alias)
            else:
                flat.append(alias)
        return flat

    def _replace_keepdim(self, alias_tensor_map, reduce_alias):
        """keepdim, replace reduce alias with 'None'."""
        new_alias_map = []
        for alias in alias_tensor_map:
            if isinstance(alias, (tuple, list)):
                new_alias = tuple("None" if item in reduce_alias else item for item in alias)
                new_alias_map.append(new_alias)
            else:
                if alias in reduce_alias:
                    new_alias_map.append("None")
                else:
                    new_alias_map.append(alias)
        new_alias_map = self._compact_tensor_map(new_alias_map)
        return reduce_alias, tuple(new_alias_map)

    def _replace_dropdim(self, alias_tensor_map, reduce_alias, dim):
        """Compress reduce dim."""
        new_alias_map = []
        for i, alias in enumerate(alias_tensor_map):
            if i in dim:
                continue
            if isinstance(alias, (tuple, list)):
                new_alias = tuple(item for item in alias if item not in reduce_alias)
                if new_alias:
                    new_alias_map.append(new_alias)
            else:
                if alias in reduce_alias:
                    continue
                new_alias_map.append(alias)
        new_alias_map = self._compact_tensor_map(new_alias_map)
        return reduce_alias, tuple(new_alias_map)

    def _compact_tensor_map(self, alias_map: Sequence[StrOrTuple]) -> Tuple[StrOrTuple, ...]:
        """Extend tensor map of 'None'."""

        def _compress(elem: StrOrTuple) -> StrOrTuple:
            if isinstance(elem, (list, tuple)):
                compressed = tuple(_compress(e) for e in elem)
                if len(compressed) == 1:
                    return compressed[0]
                if all(x == 'None' for x in compressed):
                    return 'None'
                return compressed
            return elem

        return tuple(_compress(elem) for elem in alias_map)

    def _apply_partial(self, out_layout, alias):
        """Apply all partial to given alias (string, tuple, list)."""
        if alias == "None":
            return
        if isinstance(alias, (tuple, list)):
            for elem in alias:
                self._apply_partial(out_layout, elem)
        else:
            for ops in self.partial_type:
                out_layout.set_partial_by_dev_axis(alias, ops)


class SumExtDistributedOp(ReduceExtDistributedOpBase):
    """Distributed implementation for SumExt operator."""

    def __init__(self, op_name="SumExt"):
        super().__init__(op_name, partial_type=["sum"])


class MeanExtDistributedOp(ReduceExtDistributedOpBase):
    """Distributed implementation for MeanExt operator."""

    def __init__(self, op_name="MeanExt"):
        super().__init__(op_name, partial_type=["avg"])


class ReduceMaxDistributedOp(ReduceExtDistributedOpBase):
    """Distributed implementation for ReduceMax operator."""

    def __init__(self, op_name="ReduceMax"):
        super().__init__(op_name, partial_type=["max"])

class ProdExtDistributedOp(ReduceExtDistributedOpBase):
    """
    Distributed implementation for ProdExt operator (product of all elements or along a dim).
    Compatible with torch.prod arguments.
    """

    def __init__(self, op_name="prod"):
        super().__init__(op_name, partial_type=["prod"])
