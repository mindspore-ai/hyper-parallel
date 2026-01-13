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

    def infer_layout_with_shape(self, layouts, input_shapes):
        """
        Infer output layouts for element-wise operations with shape information.

        Note:
            The framework's WithShape dispatch path still calls `infer_layout(layouts, extra_args)`
            where extra_args is a list and the last element is `input_shapes`. This helper is kept
            for compatibility but may not be invoked by the dispatcher.
        """
        extra_args = {"input_shapes": input_shapes}
        return self.infer_layout(layouts, extra_args)

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

        if len(valid_layouts) == 1:
            return valid_layouts[0]

        try:
            input_shapes = self._extract_input_shapes(extra_args)

            if not input_shapes:
                # If any real tensor layouts exist, they must match exactly.
                # Otherwise we cannot safely infer broadcasting behavior.
                first_layout = valid_layouts[0]
                for layout in valid_layouts[1:]:
                    if layout.tensor_map != first_layout.tensor_map:
                        raise ValueError(
                            f"Element-wise operation {self.op_name} cannot infer layout without shapes: "
                            f"mismatched tensor_map {first_layout.tensor_map} vs {layout.tensor_map}."
                        )
                return first_layout

            # Align shapes with layouts by position (layouts may include None for scalars / non-dtensor args)
            aligned_layouts = []
            aligned_shapes = []
            for layout, shape in zip(layouts, input_shapes):
                if layout is None:
                    continue
                aligned_layouts.append(layout)
                aligned_shapes.append(shape)

            # Defensive fallback: if alignment is unexpected, use legacy path
            if len(aligned_layouts) <= 1 or len(aligned_layouts) != len(aligned_shapes):
                return valid_layouts[0]

            # Compute broadcasted output shape
            output_shape = aligned_shapes[0]
            for shape in aligned_shapes[1:]:
                output_shape = self._broadcast_shapes(output_shape, shape)

            base_layout = aligned_layouts[0]
            merged_tensor_map = self._merge_tensor_maps_for_broadcast(
                aligned_layouts[0],
                aligned_layouts[1],
                aligned_shapes[0],
                aligned_shapes[1],
                output_shape,
            )

            for i in range(2, len(aligned_layouts)):
                temp_layout = self._create_output_layout(base_layout, merged_tensor_map)
                merged_tensor_map = self._merge_tensor_maps_for_broadcast(
                    temp_layout,
                    aligned_layouts[i],
                    output_shape,
                    aligned_shapes[i],
                    output_shape,
                )

            return self._create_output_layout(base_layout, merged_tensor_map)

        except ValueError as e:
            raise ValueError(
                f"Element-wise operation {self.op_name} failed to infer layout: {str(e)}"
            ) from e

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
            m1_axes = self._normalize_tensor_map_element(m1)
            m2_axes = self._normalize_tensor_map_element(m2)

            m1_is_sharded = bool(m1_axes)
            m2_is_sharded = bool(m2_axes)

            if not m1_is_sharded and not m2_is_sharded:
                merged_map.append(-1)
            elif not m1_is_sharded:
                merged_map.append(self._denormalize_tensor_map_element(m2_axes))
            elif not m2_is_sharded:
                merged_map.append(self._denormalize_tensor_map_element(m1_axes))
            else:
                # Both sharded, must be identical
                if m1_axes != m2_axes:
                    raise ValueError(
                        f"Conflicting sharding at dimension {i}: "
                        f"layout1 sharded on device axes {m1_axes}, "
                        f"layout2 sharded on device axes {m2_axes}"
                    )
                merged_map.append(self._denormalize_tensor_map_element(m1_axes))

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
                    f"Shapes {shape1} and {shape2} cannot be broadcast together. "
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
        Normalize a tensor_map element to a set of device axes for unified processing.

        Args:
            map_element: Element from tensor_map, can be:
                        - int: -1 (no sharding) or device axis index
                        - tuple: multiple device axes

        Returns:
            set: Set of device axes (empty set if not sharded)
        """
        if map_element == -1:
            return set()
        if isinstance(map_element, int):
            return {map_element}
        if isinstance(map_element, tuple):
            return {dim for dim in map_element if dim != -1}
        return set()

    def _denormalize_tensor_map_element(self, device_axes_set):
        """
        Convert a set of device axes back to tensor_map element format.

        Args:
            device_axes_set (set): Set of device axes

        Returns:
            int or tuple: -1 if empty, single int if one element, tuple if multiple elements
        """
        if not device_axes_set:
            return -1
        if len(device_axes_set) == 1:
            return next(iter(device_axes_set))
        return tuple(sorted(device_axes_set))

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

            m1_axes = self._normalize_tensor_map_element(m1)
            m2_axes = self._normalize_tensor_map_element(m2)

            m1_is_sharded = bool(m1_axes)
            m2_is_sharded = bool(m2_axes)

            if not m1_is_sharded and not m2_is_sharded:
                merged_map.append(-1)

            elif not m1_is_sharded:
                if dim2 == 1 and out_dim > 1:
                    raise ValueError(
                        f"Dimension {i} of second input has size 1 but is sharded on device axes {m2_axes}. "
                        f"Broadcasting dimension cannot be sharded."
                    )
                merged_map.append(self._denormalize_tensor_map_element(m2_axes))

            elif not m2_is_sharded:
                if dim1 == 1 and out_dim > 1:
                    raise ValueError(
                        f"Dimension {i} of first input has size 1 but is sharded on device axes {m1_axes}. "
                        f"Broadcasting dimension cannot be sharded."
                    )
                merged_map.append(self._denormalize_tensor_map_element(m1_axes))

            else:
                if m1_axes != m2_axes:
                    raise ValueError(
                        f"Conflicting sharding at dimension {i}: "
                        f"input1 sharded on device axes {m1_axes}, "
                        f"input2 sharded on device axes {m2_axes}"
                    )

                if (dim1 == 1 or dim2 == 1) and dim1 != dim2:
                    raise ValueError(
                        f"Dimension {i} is broadcast from size 1 to {out_dim} but is sharded on device axes {m1_axes}. "
                        f"Broadcasting dimension cannot be sharded."
                    )

                merged_map.append(self._denormalize_tensor_map_element(m1_axes))

        return tuple(merged_map)

    def _create_output_layout(self, base_layout, output_tensor_map):
        """
        Create output layout based on input layout.

        Args:
            base_layout: Base layout (usually from the first input)
            output_tensor_map (tuple): Tensor_map for the output

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
        new_layout.update_compact_str()

        return new_layout


class LessEqualDistributedOp(ElementWiseDistributedOp):
    """Distributed implementation for LessEqual (<=) operator."""

    def __init__(self, op_name="LessEqual"):
        super().__init__(op_name)


class GreaterEqualDistributedOp(ElementWiseDistributedOp):
    """Distributed implementation for GreaterEqual (>=) operator."""

    def __init__(self, op_name="GreaterEqual"):
        super().__init__(op_name)


class LogicalOrDistributedOp(ElementWiseDistributedOp):
    """Distributed implementation for LogicalOr operator."""

    def __init__(self, op_name="LogicalOr"):
        super().__init__(op_name)


class MinimumDistributedOp(ElementWiseDistributedOp):
    """Distributed implementation for Minimum operator."""

    def __init__(self, op_name="Minimum"):
        super().__init__(op_name)
