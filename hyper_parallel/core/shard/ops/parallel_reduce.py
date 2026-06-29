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
Distributed implementation for Reduce operator.
"""

from copy import deepcopy
from typing import Sequence, Union, Tuple, List

from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.platform import get_platform
from .parallel_ops import DistributedOp

platform = get_platform()
Tensor = platform.Tensor


StrOrTuple = Union[str, Tuple["StrOrTuple", ...], List["StrOrTuple"]]


def _normalize_reduce_args(input_tensor, dim=None, keepdim=False, dtype=None):
    """Normalize reduce-family arguments to consistent positional form.

    Handles torch.sum / torch.mean / torch.prod / torch.all and MindSpore
    SumExt / MeanExt / ReduceMax / MaxDim where *dim* and *keepdim* are
    ordinary positional parameters. SumExt and MeanExt also pass a trailing
    dtype slot positionally; normalize it here while keeping layout inference
    based only on ``input_layout``, ``dim``, and ``keepdim``.

    Args:
        input_tensor: The input tensor (DTensor or Tensor).
        dim: Dimension(s) to reduce. None means all dimensions, () means no reduction.
        keepdim: Whether to retain reduced dimensions.
        dtype: Optional output dtype.

    Returns:
        tuple: ``((input_tensor, dim, keepdim, dtype), {})`` — all-positional, empty kwargs.
    """
    return (input_tensor, dim, keepdim, dtype), {}


class ReduceExtDistributedOpBase(DistributedOp):
    """
    Base class for distributed reduce operators.

    Args:
        op_name (str): Name of the operator to register.
        partial_type (list): List of the operator for allreduce.
    """

    _TORCH_DTYPE_OP_NAMES = frozenset({"sum", "mean", "prod"})
    _MS_POSITIONAL_DTYPE_OP_NAMES = frozenset({"SumExt", "MeanExt"})

    def __init__(self, op_name, partial_type=None):
        super().__init__(op_name)
        if partial_type is None:
            partial_type = ["sum"]
        self.partial_type = partial_type
        self._allow_partial_inputs = True

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for reduce operators.

        Normalizes ``(input, dim, keepdim)`` across torch and MindSpore call
        sites, converts DTensor inputs to local tensors, and builds
        ``cache_values`` for layout inference.

        Args:
            args (tuple): Positional arguments passed to the operator call.
            kwargs (dict): Keyword arguments passed to the operator call.

        Returns:
            tuple: ``(local_args, local_kwargs, cache_values)`` where
                ``local_args`` is ``(input_local, dim, keepdim)`` and
                ``cache_values`` is ``[input_layout, dim, keepdim]``.
        """
        has_dtype_arg = len(args) >= 4 or "dtype" in kwargs
        normalized_args, _ = _normalize_reduce_args(*args, **kwargs)
        input_tensor, dim, keepdim, dtype = normalized_args
        local_args = (input_tensor.to_local(), dim, keepdim)
        local_kwargs = {}
        if has_dtype_arg:
            if self.op_name in self._TORCH_DTYPE_OP_NAMES:
                local_kwargs["dtype"] = dtype
            elif self.op_name in self._MS_POSITIONAL_DTYPE_OP_NAMES:
                local_args += (dtype,)
            else:
                raise TypeError(
                    f"For {self.op_name}, the `dtype` argument is not supported."
                )
        cache_values = [input_tensor.layout, dim, keepdim]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for reduce operators.

        Rules:
            1. Input must have a valid mesh_shape.
            2. ``dim`` must be ``None``, ``int``, ``tuple[int]``, or ``list[int]`` —
               never a ``Tensor``.
            3. ``dim`` values must be valid axis indices for the input tensor.
            4. Reduce axes are replaced with ``"None"`` (keepdim) or dropped;
               Partial status is applied on sharded reduced axes.

        Args:
            cache_values (list): ``[input_layout, dim, keepdim]``.

        Returns:
            tuple: ``((output_layout,), None)``

        Raises:
            ValueError: If the input layout is missing or ``mesh_shape`` is ``None``.
            TypeError: If ``dim`` is a ``Tensor``.
            ValueError: If a ``dim`` axis index is out of range.
        """
        x_layout = cache_values[0]
        dim = cache_values[1]
        keepdim = cache_values[2]

        if x_layout is None or x_layout.mesh_shape is None:
            raise ValueError(
                f"For {self.op_name}, input layout cannot be None."
            )

        # Check partial inputs
        if not self._allow_partial_inputs:
            self._check_partial_inputs([x_layout])

        if dim is not None and not isinstance(dim, (int, tuple, list)):
            raise TypeError(
                f"For {self.op_name}, the `dim` argument should be `None`, `int`, "
                f"`tuple[int]`, or `list[int]`, but got {type(dim).__name__}."
            )

        # Infer the output shape based on dim and keepdim
        output_layout = self._infer_output_layout(x_layout, dim, keepdim)

        return ((output_layout,), None)

    def _infer_output_layout(self, x_layout, dim, keepdim):
        """Infer output layout for reduce operator."""
        # Case 1: dim is None — reduce all dimensions (scalar output or all-None when keepdim).
        if dim is None:
            return self._handle_all_axis_reduce(x_layout, keepdim)

        # Case 2: dim is an empty tuple/list — reduce no dimensions, output layout equals input.
        if isinstance(dim, (tuple, list)) and len(dim) == 0:
            output_layout = Layout(
                mesh_shape=x_layout.mesh_shape,
                alias_name=x_layout.alias_name,
                rank_list=x_layout.rank_list
            )
            return output_layout(*x_layout.alias_tensor_map)

        # Case 3: dim is int, tuple, or list with at least one element.
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
        """Handle the case where dim is None, meaning reduce all dimensions."""
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


class AllExtDistributedOp(ReduceExtDistributedOpBase):
    """
    Distributed implementation for All operator
    Returns the cumulative sum of elements of input in the dimension dim.
    """

    def __init__(self, op_name="all"):
        super().__init__(op_name, partial_type=["all"])


class MaxDistributedOp(ReduceExtDistributedOpBase):
    """
    Distributed implementation for Pytorch style Max operator.

    Supports three Pytorch behaviors:
    1. torch.max(input) -> Global reduction (returns single Tensor)
    2. torch.max(input, dim, keepdim=False) -> Dimension reduction (returns (values, indices))
    3. torch.max(input, other) -> Element-wise max (returns single Tensor)
    """

    def __init__(self, op_name="max"):
        super().__init__(op_name, partial_type=["max"])

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for Max/Min operators.

        Routes between element-wise mode (two DTensor inputs) and reduction
        mode (input, dim, keepdim).  All parameters are positional with empty
        kwargs for MindSpore Primitive compatibility.

        Args:
            args (tuple): Positional arguments passed to the operator call.
            kwargs (dict): Keyword arguments passed to the operator call.

        Returns:
            tuple: ``(local_args, local_kwargs, cache_values)``.
        """
        # Element-wise mode: torch.max(a, b) — second arg is a DTensor.
        if len(args) >= 2 and hasattr(args[1], "_layout"):
            input_tensor = args[0]
            other_tensor = args[1]
            local_args = (input_tensor.to_local(), other_tensor.to_local())
            local_kwargs = {}
            cache_values = [input_tensor.layout, other_tensor.layout]
            return local_args, local_kwargs, cache_values

        # Reduction mode: torch.max(input, dim, keepdim) / MaxDim(input, dim, keepdim).
        has_dtype_arg = len(args) >= 4 or "dtype" in kwargs
        args, kwargs = _normalize_reduce_args(*args, **kwargs)
        input_tensor, dim, keepdim, _ = args
        if has_dtype_arg:
            raise TypeError(
                f"For {self.op_name}, the `dtype` argument is not supported."
            )
        # torch.max(x) global reduction: only pass the tensor so the call
        # remains torch.max(local_x), not torch.max(local_x, None, False).
        if dim is None:
            local_args = (input_tensor.to_local(),)
        else:
            local_args = (input_tensor.to_local(), dim, keepdim)
        local_kwargs = {}

        cache_values = [input_tensor.layout, dim, keepdim]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layouts for Max/Min operators.

        Rules:
            1. Element-wise mode (two Layouts in *cache_values*): output layout
               equals the first input layout.
            2. Reduction mode: same rules as ``ReduceExtDistributedOpBase``.
            3. When ``dim`` is ``None`` (global reduction), returns a single layout.
            4. When ``dim`` is specified, returns ``(values_layout, indices_layout)``.
            5. ``dim`` must be ``None``, ``int``, ``tuple[int]``, or ``list[int]``.

        Args:
            cache_values (list): ``[input_layout, dim, keepdim]`` for reduction,
                or ``[input_layout, other_layout]`` for element-wise.

        Returns:
            tuple: ``((output_layout,), None)`` or ``((values_layout, indices_layout), None)``.

        Raises:
            ValueError: If the input layout is missing or ``mesh_shape`` is ``None``.
            TypeError: If ``dim`` is not ``None``, ``int``, ``tuple``, or ``list``.
        """
        # Element-wise mode: two Layout objects in cache_values.
        if len(cache_values) == 2 and hasattr(cache_values[1], "mesh_shape"):
            # Check partial inputs
            if not self._allow_partial_inputs:
                self._check_partial_inputs(cache_values)
            return ((deepcopy(cache_values[0]),), None)

        x_layout = cache_values[0]
        dim = cache_values[1]
        keepdim = cache_values[2]

        if x_layout is None or x_layout.mesh_shape is None:
            raise ValueError(
                f"For {self.op_name}, input layout cannot be None."
            )

        # Check partial inputs
        if not self._allow_partial_inputs:
            self._check_partial_inputs([x_layout])

        if dim is not None and not isinstance(dim, (int, tuple, list)):
            raise TypeError(
                f"For {self.op_name}, the `dim` argument should be `None`, `int`, "
                f"`tuple[int]`, or `list[int]`, but got {type(dim).__name__}."
            )

        values_layout = self._infer_output_layout(x_layout, dim, keepdim)

        if dim is None:
            # torch.max(input) -> Single Tensor
            return ((values_layout,), None)

        # torch.max(input, dim) -> (values, indices)
        indices_layout = deepcopy(values_layout)
        return ((values_layout, indices_layout), None)


class MinDistributedOp(MaxDistributedOp):
    """
    Distributed implementation for Pytorch style Min operator.
    
    Supports three Pytorch behaviors:
    1. torch.min(input) -> Global reduction (returns single Tensor)
    2. torch.min(input, dim, keepdim=False) -> Dimension reduction (returns (values, indices))
    3. torch.min(input, other) -> Element-wise min (returns single Tensor)
    """

    def __init__(self, op_name="min"):
        # Call the parent class (MaxDistributedOp) initialization
        super().__init__(op_name=op_name)
        # Override the partial_type to use "min" instead of "max" for the underlying communication
        self.partial_type = ["min"]
