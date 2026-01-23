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
from .parallel_ops import DistributedOp


class ElementWiseDistributedOp(DistributedOp):
    """
    Base class for distributed element-wise operators.

    Supports broadcasting following broadcasting rules and handles
    distributed tensor layouts with proper sharding strategy inference.

    Args:
        op_name (str): Name of the operator to register.
    """

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layouts for element-wise operations with broadcasting support.

        For element-wise operations:
        - Supports broadcasting following NumPy broadcasting rules
        - All inputs must have compatible shapes for broadcasting
        - Output will have the broadcasted shape and appropriate sharding strategy
        - Handles both simple and complex sharding patterns (including tuple-type tensor_maps)

        Args:
            layouts (tuple): Tuple of layouts for input tensors
            extra_args: Extra arguments for the operation. It can be:
                        - dict containing 'input_shapes'
                        - list/tuple where the last element is input_shapes (WithShape path)

        Returns:
            Layout: Layout for output tensor with merged sharding strategy.

        Raises:
            ValueError: If input layouts are not compatible for broadcasting.
        """
        if not layouts:
            return None

        valid_layouts = [layout for layout in layouts if layout is not None]

        if not valid_layouts:
            return None

        # Check partial inputs - ElementWiseDistributedOp does not support partial by default
        # This check is performed after basic layout validation
        if not self._allow_partial_inputs:
            self._check_partial_inputs(layouts)

        if len(valid_layouts) == 1:
            return valid_layouts[0]

        input_shapes = self._extract_input_shapes(extra_args)

        if not input_shapes:
            return self._handle_no_input_shapes(valid_layouts)

        aligned_layouts, aligned_shapes = self._align_layouts_and_shapes(layouts, input_shapes)

        if len(aligned_layouts) <= 1 or len(aligned_layouts) != len(aligned_shapes):
            return valid_layouts[0]

        output_shape = self._compute_output_shape(aligned_shapes)
        merged_tensor_map, merged_partial = self._merge_all_layouts(
            aligned_layouts,
            aligned_shapes,
            output_shape,
            layouts
        )

        self._check_all_inputs_broadcasts_and_partial(aligned_layouts, aligned_shapes, output_shape)

        return self._create_output_layout(aligned_layouts[0], merged_tensor_map, merged_partial)

    def _handle_no_input_shapes(self, valid_layouts):
        """
        Handle the case when input shapes are not available.
        """
        first_layout = valid_layouts[0]
        for layout in valid_layouts[1:]:
            if layout.tensor_map != first_layout.tensor_map:
                raise ValueError(
                    f"For {self.op_name}, cannot infer layout without shapes: "
                    f"mismatched tensor_map {first_layout.tensor_map} vs {layout.tensor_map}."
                )
        return first_layout

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

    def _extract_input_shapes(self, extra_args):
        """
        Extract input_shapes from extra_args.

        Compatible with:
          - dict: {"input_shapes": [...]}
          - list/tuple (WithShape dispatcher): extra_args = [..., input_shapes]
        """
        if isinstance(extra_args, dict):
            return extra_args.get("input_shapes", None)

        if isinstance(extra_args, (list, tuple)) and extra_args:
            maybe_shapes = extra_args[-1]
            if isinstance(maybe_shapes, (list, tuple)):
                return maybe_shapes

        return None

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

    def _merge_tensor_maps_without_shape(self, layout1, layout2):
        """
        Merge tensor_maps without shape information (for broadcasting scenarios).

        Merging rules without shape:
        - If both dimensions are not sharded: use -1
        - If one is sharded and one is not: use the sharded one (assume broadcasting)
        - If both are sharded: they must be identical, otherwise raise error

        Args:
            layout1: Layout of the first input
            layout2: Layout of the second input

        Returns:
            tuple: Merged tensor_map

        Raises:
            ValueError: If sharding strategies conflict
        """
        map1 = layout1.tensor_map if layout1.tensor_map else tuple()
        map2 = layout2.tensor_map if layout2.tensor_map else tuple()

        # Align ranks by padding with -1
        max_len = max(len(map1), len(map2))
        padded_map1 = (-1,) * (max_len - len(map1)) + map1
        padded_map2 = (-1,) * (max_len - len(map2)) + map2

        merged_map = []
        for i, (m1, m2) in enumerate(zip(padded_map1, padded_map2)):
            m1_axis = self._normalize_tensor_map_element(m1)
            m2_axis = self._normalize_tensor_map_element(m2)

            m1_axis_for_compare = frozenset(m1_axis)
            m2_axis_for_compare = frozenset(m2_axis)

            m1_is_sharded = bool(m1_axis)
            m2_is_sharded = bool(m2_axis)

            if not m1_is_sharded and not m2_is_sharded:
                merged_map.append(-1)
            elif not m1_is_sharded:
                merged_map.append(self._denormalize_tensor_map_element(m2_axis))
            elif not m2_is_sharded:
                merged_map.append(self._denormalize_tensor_map_element(m1_axis))
            else:
                if m1_axis_for_compare != m2_axis_for_compare:
                    raise ValueError(
                        f"For {self.op_name}, inputs should have same sharding pattern, "
                        f"but got confilcting sharding at dimension {i}, "
                        f"input1 shaded on {m1_axis} and input2 shaded on {m2_axis}."
                    )
                merged_map.append(self._denormalize_tensor_map_element(m1_axis))

        return tuple(merged_map)

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

    def get_expand_impl(self, func, output_layout, layouts, extra_args):
        """
        Get expand implementation for the operator
        """
        x1_layout = layouts[0]
        x2_layout = layouts[1]
        x1_partial = x1_layout.is_partial() if x1_layout is not None else None
        x2_partial = x2_layout.is_partial() if x2_layout is not None else None

        if x1_partial != x2_partial:
            scaling_factor = 1
            for i, partial_type in enumerate(output_layout.partial):
                if partial_type == "sum":
                    scaling_factor *= output_layout.mesh_shape[i]
                elif partial_type is not None:
                    raise ValueError(
                        f"For {self.op_name}, inputs partial status should be 'sum' or None, "
                        f"but got {partial_type} at index {i}."
                    )

            # use expand_impl only when one of x1 and x2 is with partial placement.
            def expand_impl1(x1, x2):
                add_out = func(x1 / scaling_factor, x2)
                return add_out

            def expand_impl2(x1, x2):
                add_out = func(x1, x2 / scaling_factor)
                return add_out
            return expand_impl1 if not x1_partial else expand_impl2
        return None
