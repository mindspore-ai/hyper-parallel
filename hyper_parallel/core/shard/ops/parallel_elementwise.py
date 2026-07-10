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
"""
Distributed implementation for Element-wise operator.
"""

import copy
from typing import Callable, Optional, Tuple

from .parallel_ops import DistributedOp


_INPLACE_ELEMENTWISE_OPS = frozenset({
    "add_", "sub_", "InplaceAddExt", "InplaceSubExt",
})


def _partial_signature(layout, mesh_ndim: int) -> tuple:
    """Return one Partial entry per output mesh dimension."""
    if layout is None:
        return (None,) * mesh_ndim
    partial = tuple(layout.partial)
    if len(partial) != mesh_ndim:
        raise ValueError(
            f"Input and output mesh dimensions must match, but got "
            f"input={len(partial)} and output={mesh_ndim}."
        )
    return partial


def _contributes_to_partial_output(input_layout, output_layout) -> bool:
    """Return whether this rank contributes an input to a Partial output.

    An input that is not Partial on one of the output's Partial axes is
    replicated along that axis. Only coordinate zero may contribute that
    replicated value, otherwise the eventual reduction would count it once
    per rank on the added axis.
    """
    output_partial = tuple(output_layout.partial)
    input_partial = _partial_signature(input_layout, len(output_partial))
    for mesh_dim, output_partial_type in enumerate(output_partial):
        if (
            output_partial_type is not None
            and input_partial[mesh_dim] is None
            and output_layout.mesh.get_local_rank(mesh_dim) != 0
        ):
            return False
    return True


def _zero_contribution(value):
    """Create a strict zero while retaining a floating tensor's grad edge."""
    is_complex = getattr(value, "is_complex", None)
    if callable(is_complex) and is_complex():
        return value * 0
    if hasattr(value, "clamp"):
        # ``value * 0`` turns +/-Inf into NaN. Clamp first so the forward value
        # is finite, then multiply by zero to make the derivative zero even at
        # the clamp boundary.
        return value.clamp(0, 0) * 0
    if isinstance(value, (bool, int, float, complex)):
        return type(value)(0)
    return value * 0


def _unwrap_local_value(value):
    """Convert DTensor-like values to local tensors while preserving containers."""
    if hasattr(value, "_layout"):
        return value.to_local()
    if isinstance(value, tuple):
        return tuple(_unwrap_local_value(item) for item in value)
    if isinstance(value, list):
        return [_unwrap_local_value(item) for item in value]
    return value


def _collect_layout_and_shape(value):
    """Collect layout and shape from one argument for layout inference cache."""
    layout = value.layout if hasattr(value, "_layout") else None
    shape = value.shape if hasattr(value, "shape") else None
    return layout, shape


def _build_elementwise_cache_values(args, kwargs):
    """Build cache_values from the real element-wise input arguments."""
    input_layouts = []
    input_shapes = []
    for value in args:
        layout, shape = _collect_layout_and_shape(value)
        input_layouts.append(layout)
        input_shapes.append(shape)
    for value in kwargs.values():
        layout, shape = _collect_layout_and_shape(value)
        input_layouts.append(layout)
        input_shapes.append(shape)
    return [*input_layouts, input_shapes]


class ElementWiseDistributedOp(DistributedOp):
    """
    Base class for distributed element-wise operators.

    Supports broadcasting following broadcasting rules and handles
    distributed tensor layouts with proper sharding strategy inference.

    Args:
        op_name (str): Name of the operator to register.
    """

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for element-wise operators.

        NOTE: aclop packed-args normalization (for MindSpore aclop operators
        like Mod, StopGradient that pack args as ``(prim, name, (real_args...))``)
        is handled upstream in ``OpDispatcher._dispatch_layout_infer`` via
        ``_normalize_aclop_args``. This method receives clean unpacked args.

        Args:
            args (tuple): Positional arguments passed to the operator.
            kwargs (dict): Keyword arguments passed to the operator.

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        local_kwargs = {key: _unwrap_local_value(value) for key, value in kwargs.items()}

        local_args = tuple(_unwrap_local_value(arg) for arg in args)
        cache_values = _build_elementwise_cache_values(args, kwargs)
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:  # pylint: disable=W0221
        """
        Infer output layouts for element-wise operations with broadcasting support.

        Rules:
            1. Inputs must not have Partial status unless the operator explicitly allows it.
            2. Input shapes must be broadcast-compatible when shape information is available.
            3. Broadcasting dimensions cannot be sharded.
            4. Non-broadcast sharding patterns must be compatible across inputs.
            5. Output layout uses the merged sharding strategy and merged Partial status.

        Args:
            cache_values (list): [input_layout_0, ..., input_layout_n, input_shapes].

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If input layouts are not compatible for broadcasting.
        """
        layouts = tuple(cache_values[:-1])
        input_shapes = cache_values[-1] if cache_values else None

        if not layouts:
            return None

        valid_layouts = [layout for layout in layouts if layout is not None]

        if not valid_layouts:
            return None

        if not self._allow_partial_inputs:
            self._check_partial_inputs(layouts)

        if len(valid_layouts) == 1:
            return ((copy.deepcopy(valid_layouts[0]),), None)

        if not input_shapes:
            return ((self._handle_no_input_shapes(valid_layouts),), None)

        aligned_layouts, aligned_shapes = self._align_layouts_and_shapes(layouts, input_shapes)

        if len(aligned_layouts) <= 1 or len(aligned_layouts) != len(aligned_shapes):
            return ((copy.deepcopy(valid_layouts[0]),), None)

        output_shape = self._compute_output_shape(aligned_shapes)
        merged_tensor_map, merged_partial = self._merge_all_layouts(
            aligned_layouts,
            aligned_shapes,
            output_shape,
            layouts
        )

        self._check_all_inputs_broadcasts_and_partial(aligned_layouts, aligned_shapes, output_shape)

        output_layout = self._create_output_layout(aligned_layouts[0], merged_tensor_map, merged_partial)
        return ((output_layout,), None)

    def _handle_no_input_shapes(self, valid_layouts):
        """
        Handle the case when input shapes are not available.
        """
        first_layout = valid_layouts[0]
        first_alias_map = first_layout.alias_tensor_map
        for layout in valid_layouts[1:]:
            if layout.alias_tensor_map != first_alias_map:
                raise ValueError(
                    f"For {self.op_name}, cannot infer layout without shapes: "
                    f"mismatched alias_tensor_map {first_alias_map} vs {layout.alias_tensor_map}."
                )
        return copy.deepcopy(first_layout)

    def _align_layouts_and_shapes(self, layouts, input_shapes):
        """
        Align layouts with shapes by position, skipping None layouts.
        """
        aligned_layouts = []
        aligned_shapes = []
        for layout, shape in zip(layouts, input_shapes):
            if layout is None:
                continue
            aligned_layouts.append(layout)
            aligned_shapes.append(shape)
        return aligned_layouts, aligned_shapes

    def _compute_output_shape(self, aligned_shapes):
        """
        Compute broadcasted output shape from all input shapes.
        """
        output_shape = aligned_shapes[0]
        for shape in aligned_shapes[1:]:
            output_shape = self._broadcast_shapes(output_shape, shape)
        return output_shape

    def _merge_all_layouts(self, aligned_layouts, aligned_shapes, output_shape, layouts):
        """
        Merge all input layouts sequentially to get final tensor_map and partial status.
        """
        base_layout = aligned_layouts[0]

        merged_tensor_map = self._merge_tensor_maps_for_broadcast(
            aligned_layouts[0],
            aligned_layouts[1],
            aligned_shapes[0],
            aligned_shapes[1],
            output_shape
        )

        merged_partial = self._merge_partial_status(
            base_layout.partial,
            aligned_layouts[1].partial,
            merged_tensor_map,
            aligned_layouts[0].tensor_map if aligned_layouts[0].tensor_map else tuple(),
            aligned_layouts[1].tensor_map if aligned_layouts[1].tensor_map else tuple(),
            layouts
        )

        for i in range(2, len(aligned_layouts)):
            temp_layout = self._create_output_layout(base_layout, merged_tensor_map, merged_partial)
            merged_tensor_map = self._merge_tensor_maps_for_broadcast(
                temp_layout,
                aligned_layouts[i],
                output_shape,
                aligned_shapes[i],
                output_shape
            )
            merged_partial = self._merge_partial_status(
                merged_partial,
                aligned_layouts[i].partial,
                merged_tensor_map,
                temp_layout.tensor_map if temp_layout.tensor_map else tuple(),
                aligned_layouts[i].tensor_map if aligned_layouts[i].tensor_map else tuple(),
                layouts
            )

        return merged_tensor_map, merged_partial

    def _merge_partial_status(self, partial1, partial2, merged_tensor_map, tensor_map1, tensor_map2, layouts):
        """
        Merge partial status from two inputs.

        Rules:
        1. Both None → None
        2. One None → Use the other
        3. Both not None and same → Use it
        4. Both not None and different → Error
        5. Check Shard + Partial conflicts for each input

        Args:
            partial1: Partial status list from first input
            partial2: Partial status list from second input
            merged_tensor_map: Merged tensor map for output
            tensor_map1: Tensor map of first input
            tensor_map2: Tensor map of second input

        Returns:
            List: Merged partial status

        Raises:
            ValueError: If partial operations conflict or Shard+Partial conflict found
        """
        # Check Shard + Partial conflicts for input1
        self._check_shard_partial_conflict(tensor_map1, partial1, layouts)

        # Check Shard + Partial conflicts for input2
        self._check_shard_partial_conflict(tensor_map2, partial2, layouts)

        # Determine mesh dimension from partial lists
        mesh_dim = max(len(partial1) if partial1 else 0, len(partial2) if partial2 else 0)

        merged_partial = [None] * mesh_dim

        for i in range(mesh_dim):
            op1 = partial1[i] if partial1 and i < len(partial1) else None
            op2 = partial2[i] if partial2 and i < len(partial2) else None

            # Both have partial status with different operations
            if op1 is not None and op2 is not None and op1 != op2:
                raise ValueError(
                    f"For {self.op_name}, partial operations should be same for device axis {i}, "
                    f"but got {op1} and {op2}"
                )

            # Merge: prefer non-None, or either if both same
            if op1 is not None:
                merged_partial[i] = op1
            elif op2 is not None:
                merged_partial[i] = op2

        # Check final output for Shard + Partial conflicts
        self._check_shard_partial_conflict(merged_tensor_map, merged_partial, layouts)

        return merged_partial

    def _check_shard_partial_conflict(self, tensor_map, partial_list, layouts):
        """
        Check for conflicts between Shard and Partial on same device axis.

        Args:
            tensor_map: Tensor map to check
            partial_list: Partial status list

        Raises:
            ValueError: If Shard and Partial conflict found
        """
        if not partial_list:
            return

        mesh_dim = len(partial_list)

        # Collect all device axis used for sharding
        sharded_axis = set()
        if tensor_map:
            for map_val in tensor_map:
                if isinstance(map_val, tuple):
                    for sub_val in map_val:
                        if sub_val != -1:
                            # Convert to device axis index
                            axis_idx = mesh_dim - 1 - sub_val
                            sharded_axis.add(axis_idx)
                elif map_val != -1:
                    axis_idx = mesh_dim - 1 - map_val
                    sharded_axis.add(axis_idx)

        # Check if any sharded axis has partial status
        for axis_idx in sharded_axis:
            if 0 <= axis_idx < len(partial_list) and partial_list[axis_idx] is not None:
                raise ValueError(
                    f"For {self.op_name}, Shard and Partial should not coexist on same device axis "
                    f"{axis_idx}, but got Partial({partial_list[axis_idx]}). "
                    f"Please check layouts: {layouts}."
                )

    def _check_all_inputs_broadcasts_and_partial(self, layouts, input_shapes, output_shape):
        """
        Check if any input broadcasts and has Partial status.
        """
        for i, (layout, input_shape) in enumerate(zip(layouts, input_shapes)):
            if layout is None:
                continue

            input_name = f"input{i+1}"

            input_len = len(input_shape)
            output_len = len(output_shape)

            if input_len < output_len:
                aligned_input_shape = (1,) * (output_len - input_len) + tuple(input_shape)
            else:
                aligned_input_shape = input_shape

            broadcasts = False
            for in_dim, out_dim in zip(aligned_input_shape, output_shape):
                if in_dim == 1 and out_dim > 1:
                    broadcasts = True
                    break

            if broadcasts and layout.is_partial():
                raise ValueError(
                    f"For {self.op_name}, {input_name} has Partial status and broadcasts. "
                    f"Should be without Partial status for broadcasting without communication"
                )

    def _broadcast_shapes(self, shape1, shape2):
        """
        Calculate the broadcasted shape of two shapes according to broadcasting rules.

        Broadcasting rules:
        1. If two arrays have different numbers of dimensions, pad the shape of the
           lower-dimensional array with 1s on the left until both shapes have the same length.
        2. If two arrays have the same number of dimensions but different lengths in some
           dimensions, dimensions with length 1 will be expanded to match the other array's
           dimension length.
        3. If two arrays have the same number of dimensions but any dimension has different
           lengths and neither is 1, raise an error.

        Args:
            shape1 (tuple): Shape of the first tensor, e.g., (3, 1, 5)
            shape2 (tuple): Shape of the second tensor, e.g., (4, 5)

        Returns:
            tuple: Broadcasted shape, e.g., (3, 4, 5)

        Raises:
            ValueError: If shapes cannot be broadcast together.
        """
        # Rule 1: Right-align, pad with 1s on the left to make dimensions equal
        len1, len2 = len(shape1), len(shape2)
        max_len = max(len1, len2)

        padded_shape1 = (1,) * (max_len - len1) + tuple(shape1)
        padded_shape2 = (1,) * (max_len - len2) + tuple(shape2)

        # Rules 2 and 3: Check if each dimension can be broadcast
        result_shape = []
        for dim1, dim2 in zip(padded_shape1, padded_shape2):
            if dim1 == dim2:
                # Dimensions are the same, use directly
                result_shape.append(dim1)
            elif dim1 == 1:
                # First shape has 1 in this dimension, expand to dim2
                result_shape.append(dim2)
            elif dim2 == 1:
                # Second shape has 1 in this dimension, expand to dim1
                result_shape.append(dim1)
            else:
                # Rule 3: Dimensions are different and neither is 1, cannot broadcast
                raise ValueError(
                    f"For {self.op_name}, shapes {shape1} and {shape2} cannot be broadcast together. "
                    f"Dimension mismatch: {dim1} vs {dim2}"
                )

        return tuple(result_shape)

    def _align_tensor_maps_for_broadcast(self, layout1, layout2, shape1, shape2):
        """
        Align tensor_maps of two layouts to support broadcasting.

        When two tensors have different dimensions, the tensor_map of the
        lower-dimensional tensor is padded with -1 (indicating no sharding) at the front.

        Args:
            layout1: Layout of the first tensor
            layout2: Layout of the second tensor
            shape1 (tuple): Global shape of the first tensor
            shape2 (tuple): Global shape of the second tensor

        Returns:
            tuple: (aligned_map1, aligned_map2) - Aligned tensor_maps
        """
        len1, len2 = len(shape1), len(shape2)
        max_len = max(len1, len2)

        map1 = layout1.tensor_map if layout1.tensor_map else tuple([-1] * len1)
        map2 = layout2.tensor_map if layout2.tensor_map else tuple([-1] * len2)

        aligned_map1 = (-1,) * (max_len - len1) + map1
        aligned_map2 = (-1,) * (max_len - len2) + map2

        return aligned_map1, aligned_map2

    def _normalize_tensor_map_element(self, map_element):
        """
        Normalize a tensor_map element to a tuple of device axis for unified processing.

        Args:
            map_element: Element from tensor_map, can be:
                        - int: -1 (no sharding) or device axis index
                        - tuple: multiple device axis

        Returns:
            tuple: Tuple of device axis (empty tuple if not sharded)
        """
        if map_element == -1:
            return ()
        if isinstance(map_element, int):
            return (map_element,)
        if isinstance(map_element, tuple):
            return tuple(dim for dim in map_element if dim != -1)
        return ()

    def _denormalize_tensor_map_element(self, device_axis_tuple):
        """
        Convert a tuple of device axis back to tensor_map element format.

        Args:
            device_axis_tuple (tuple): Tuple of device axis

        Returns:
            int or tuple: -1 if empty, single int if one element, tuple if multiple elements
        """
        if not device_axis_tuple:
            return -1
        if len(device_axis_tuple) == 1:
            return device_axis_tuple[0]
        return device_axis_tuple

    def _merge_tensor_maps_for_broadcast(self, layout1, layout2, shape1, shape2, output_shape):
        """
        Merge tensor_maps of two inputs to generate output tensor_map.

        This method handles both simple int-type and complex tuple-type tensor_map elements,
        ensuring correct sharding strategy for the broadcasted output.

        Args:
            layout1: Layout of the first input
            layout2: Layout of the second input
            shape1 (tuple): Global shape of the first input
            shape2 (tuple): Global shape of the second input
            output_shape (tuple): Global shape of the output

        Returns:
            tuple: Merged tensor_map for the output

        Raises:
            ValueError: If sharding strategies conflict or broadcasting dimension is sharded
        """
        map1, map2 = self._align_tensor_maps_for_broadcast(layout1, layout2, shape1, shape2)

        len1, len2 = len(shape1), len(shape2)
        max_len = len(output_shape)
        padded_shape1 = (1,) * (max_len - len1) + tuple(shape1)
        padded_shape2 = (1,) * (max_len - len2) + tuple(shape2)

        merged_map = []
        for i, (dim1, dim2, out_dim) in enumerate(zip(padded_shape1, padded_shape2, output_shape)):
            m1, m2 = map1[i], map2[i]

            m1_axis = self._normalize_tensor_map_element(m1)
            m2_axis = self._normalize_tensor_map_element(m2)

            m1_axis_for_compare = frozenset(m1_axis)
            m2_axis_for_compare = frozenset(m2_axis)

            m1_is_sharded = bool(m1_axis)
            m2_is_sharded = bool(m2_axis)

            if not m1_is_sharded and not m2_is_sharded:
                merged_map.append(-1)

            elif not m1_is_sharded:
                if dim2 == 1 and out_dim > 1:
                    raise ValueError(
                        f"For {self.op_name}, dimension {i} of second input has size 1 "
                        f"but is sharded on device axis {m2_axis}. "
                        f"Broadcasting dimension cannot be sharded."
                    )
                merged_map.append(self._denormalize_tensor_map_element(m2_axis))

            elif not m2_is_sharded:
                if dim1 == 1 and out_dim > 1:
                    raise ValueError(
                        f"For {self.op_name}, dimension {i} of first input has size 1 "
                        f"but is sharded on device axis {m1_axis}. "
                        f"Broadcasting dimension cannot be sharded."
                    )
                merged_map.append(self._denormalize_tensor_map_element(m1_axis))

            else:
                if m1_axis_for_compare != m2_axis_for_compare:
                    raise ValueError(
                        f"For {self.op_name}, inputs should have same sharding pattern, "
                        f"but got confilcting sharding at dimension {i}, "
                        f"input1 shaded on {m1_axis} and input2 shaded on {m2_axis}."
                    )

                if (dim1 == 1 or dim2 == 1) and dim1 != dim2:
                    raise ValueError(
                        f"For {self.op_name}, dimension {i} is broadcast from size 1 "
                        f"to {out_dim} but is sharded on device axis {m1_axis}. "
                        f"Broadcasting dimension cannot be sharded."
                    )

                merged_map.append(self._denormalize_tensor_map_element(m1_axis))

        return tuple(merged_map)

    def _create_output_layout(self, base_layout, output_tensor_map, partial_list=None):
        """
        Create output layout based on input layout.

        Args:
            base_layout: Base layout (usually from the first input)
            output_tensor_map (tuple): Tensor_map for the output
            partial_list (list): Partial status list for the output

        Returns:
            Layout: New Layout object with updated tensor_map and alias_tensor_map
        """
        new_layout = copy.deepcopy(base_layout)
        new_layout.set_tensor_map(output_tensor_map)

        alias_tensor_map = []
        for tensor_dim in output_tensor_map:
            if tensor_dim == -1:
                alias_tensor_map.append("None")
            elif isinstance(tensor_dim, tuple):
                alias_tuple = tuple(
                    base_layout.alias_name[len(base_layout.alias_name) - 1 - dim]
                    for dim in tensor_dim
                    if dim != -1
                )
                alias_tensor_map.append(alias_tuple if alias_tuple else "None")
            else:
                alias_tensor_map.append(
                    base_layout.alias_name[len(base_layout.alias_name) - 1 - tensor_dim]
                )

        new_layout.set_alias_tensor_map(tuple(alias_tensor_map))

        # Set partial status if provided
        if partial_list:
            for i, partial_op in enumerate(partial_list):
                if partial_op is not None and i < len(new_layout.alias_name):
                    new_layout.set_partial_by_dev_axis(new_layout.alias_name[i], partial_op)

        return new_layout


class ElementWiseWithPartialDistributedOp(ElementWiseDistributedOp):
    """
    Base class for elementwise operations that support partial status propagation.
    """
    def __init__(self, op_name):
        super().__init__(op_name)
        self._allow_partial_inputs = True


class AddDistributedOp(ElementWiseWithPartialDistributedOp):
    """
    Distributed implementation for Add operator.

    This operator supports partial status propagation from inputs to output,
    which is useful for operations like gradient accumulation where partial
    results need to be preserved through the computation graph.
    """

    @staticmethod
    def _canonicalize_binary_operands(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """Move semantic ``input`` and ``other`` operands before option values."""
        local_kwargs = dict(kwargs)
        if args:
            if "input" in local_kwargs:
                raise ValueError("Binary elementwise input was provided both positionally and by keyword.")
            lhs = args[0]
            trailing_args = args[1:]
        elif "input" in local_kwargs:
            lhs = local_kwargs.pop("input")
            trailing_args = ()
        else:
            return args, local_kwargs

        if trailing_args:
            if "other" in local_kwargs:
                raise ValueError("Binary elementwise other was provided both positionally and by keyword.")
            rhs = trailing_args[0]
            trailing_args = trailing_args[1:]
        elif "other" in local_kwargs:
            rhs = local_kwargs.pop("other")
        else:
            return args, local_kwargs
        return (lhs, rhs, *trailing_args), local_kwargs

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """Canonicalize binary operands before local unwrapping and layout caching.

        Args:
            args: Positional backend arguments.
            kwargs: Keyword backend arguments, potentially including
                ``input``, ``other``, and options such as ``alpha``.

        Returns:
            Local arguments, local keyword arguments, and layout cache values
            with the semantic left and right operands in the first two slots.
        """
        canonical_args, canonical_kwargs = self._canonicalize_binary_operands(args, kwargs)
        return super().preprocess(canonical_args, canonical_kwargs)

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:  # pylint: disable=W0221
        """
        Infer output layout for Add operator.

        Rules:
            1. Follow element-wise broadcasting and sharding merge rules.
            2. Partial status is allowed and propagated.
            3. Propagated Partial status must be "sum" or None.

        Args:
            cache_values (list): [input_layout_0, ..., input_layout_n, input_shapes].

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If propagated Partial status is not "sum" or None.
        """
        layouts = tuple(cache_values[:-1])
        if self.op_name in _INPLACE_ELEMENTWISE_OPS and len(layouts) >= 2:
            mesh_ndim = max(
                (len(layout.partial) for layout in layouts[:2] if layout is not None),
                default=0,
            )
            partial_signatures = tuple(
                _partial_signature(layout, mesh_ndim)
                for layout in layouts[:2]
            )
            if partial_signatures[0] != partial_signatures[1]:
                raise ValueError(
                    f"For {self.op_name}, input Partial placements should be identical "
                    f"for in-place execution, but got {partial_signatures}."
                )

        infer_result = super().infer_layout(cache_values)
        if infer_result is None:
            return infer_result

        output_layout = infer_result[0][0]
        for i, partial_type in enumerate(output_layout.partial):
            if partial_type is not None and partial_type != "sum":
                raise ValueError(
                    f"For {self.op_name}, inputs partial status should be 'sum' or None, "
                    f"but got {partial_type} at index {i}."
                )

        return infer_result

    def get_expand_impl(self, func: Optional[Callable], infer_result: tuple,  # pylint: disable=W0221
                        cache_values: list) -> Optional[Callable]:
        """Build a rank-local implementation that avoids repeated values.

        Args:
            func: Local backend add or subtract implementation.
            infer_result: Inferred output layout and auxiliary result.
            cache_values: Input layouts followed by input shapes.

        Returns:
            A contribution-aware local implementation, or ``None`` when both
            inputs already have identical Partial placements.
        """
        layouts = tuple(cache_values[:-1])
        x1_layout = layouts[0]
        x2_layout = layouts[1]
        output_layout = infer_result[0][0]
        output_mesh_ndim = len(output_layout.partial)
        x1_partial = _partial_signature(x1_layout, output_mesh_ndim)
        x2_partial = _partial_signature(x2_layout, output_mesh_ndim)

        if x1_partial == x2_partial:
            return None

        x1_contributes = _contributes_to_partial_output(x1_layout, output_layout)
        x2_contributes = _contributes_to_partial_output(x2_layout, output_layout)

        # ``*args``/``**kwargs`` forward add-specific extras such as ``alpha``.
        # A strict zero retains a valid zero-gradient autograd edge.
        def _expand_impl(x1, x2, *args, **kwargs):
            local_x1 = x1 if x1_contributes else _zero_contribution(x1)
            local_x2 = x2 if x2_contributes else _zero_contribution(x2)
            return func(local_x1, local_x2, *args, **kwargs)

        return _expand_impl


class SubDistributedOp(AddDistributedOp):
    """Distributed subtraction with correct replicated-minus-partial signs.

    When the first operand is replicated and the second is Partial(sum), only
    the first Partial rank may contribute the replicated value. Every other
    rank must contribute ``0 - x2_local`` so reducing the output reconstructs
    ``x1 - sum(x2_local)``.
    """
