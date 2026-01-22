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
Distributed implementation for Gather operator.
"""

from hyper_parallel.core.layout import Layout
from .parallel_ops import DistributedOp


class IndexSelectDistributedOp(DistributedOp):
    """Distributed implementation for Index Select operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layouts for Index Select operations.

        Args:
            layouts: Layouts of input tensors
            extra_args: extra_args of input tensors

        Returns:
            tuple: Layout for output tensor.

        Raises:
            ValueError: If input layouts are not compatible.
        """
        # Check
        if len(layouts) != 3:
            raise ValueError(f"Gather ops requires 3 layouts, but {len(layouts)}")
        if len(extra_args) != 1:
            raise ValueError(f"Gather ops requires 1 extra args, but {len(extra_args)}")

        # Parse layout info
        p_layout, i_layout = layouts[0], layouts[2]
        axis, batch_dims = extra_args[0], 0

        p_tensor_map = p_layout.alias_tensor_map
        i_tensor_map = i_layout.alias_tensor_map

        # Create output layout
        if p_tensor_map[axis] != "None":
            raise ValueError(
                f"Operation {self.op_name}: Cannot perform sharding on params along the axis"
            )

        if len(i_tensor_map) != 1:
            raise ValueError(
                f"Operation {self.op_name}: index is not a one-dimensional Tensor"
            )

        if axis < -len(p_tensor_map) or axis >= len(p_tensor_map):
            raise ValueError(
                f"Operation {self.op_name}: dim value is out of valid range"
            )

        output_tensor_map = (
            p_tensor_map[:axis] + i_tensor_map[batch_dims:] + p_tensor_map[axis + 1 :]
        )
        output_layout = i_layout
        output_layout = Layout(
            mesh_shape=output_layout.mesh_shape,
            alias_name=output_layout.alias_name,
            rank_list=output_layout.rank_list,
        )
        output_layout = output_layout(*output_tensor_map)
        return output_layout


class GatherDistributedOp(DistributedOp):
    """Distributed implementation for Gather operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layouts for Gather operations.

        Args:
            layouts: Layouts of input tensors
            extra_args: extra_args of input tensors

        Returns:
            tuple: Layout for output tensor.

        Raises:
            ValueError: If input layouts are not compatible.
        """
        # Check
        if len(layouts) != 3:
            raise ValueError(f"Gather ops requires 3 layouts, but {len(layouts)}")
        if len(extra_args) != 1:
            raise ValueError(f"Gather ops requires 1 extra args, but {len(extra_args)}")

        # Parse layout info
        p_layout, i_layout = layouts[0], layouts[2]
        axis = extra_args[0]

        p_tensor_map = p_layout.alias_tensor_map
        i_tensor_map = i_layout.alias_tensor_map

        # Create output layout
        if p_tensor_map[axis] != "None":
            raise ValueError(
                f"Operation {self.op_name}: Cannot perform sharding on params along the axis"
            )

        if len(p_tensor_map) != len(i_tensor_map):
            raise ValueError(
                f"Operation {self.op_name}: input and index must have the same number of dimensions"
            )

        if axis < -len(p_tensor_map) or axis >= len(p_tensor_map):
            raise ValueError(
                f"Operation {self.op_name}: dim value is out of valid range"
            )

        output_tensor_map = i_tensor_map
        output_layout = i_layout
        output_layout = Layout(
            mesh_shape=output_layout.mesh_shape,
            alias_name=output_layout.alias_name,
            rank_list=output_layout.rank_list,
        )
        output_layout = output_layout(*output_tensor_map)
        return output_layout


class GatherNdDistributedOp(DistributedOp):
    """Distributed implementation for GatherNd operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for GatherNd.

        For GatherNd: out.shape = indices.shape[:-1] + input_x.shape[K:], where K = indices.shape[-1].

        This implementation:
            - Inherits sharding from indices[:-1].
            - Allows sharding on input_x trailing dims input_x[K:].
            - Requires input_x[:K] to be replicated ("None") if input_layout is provided.
            - Requires indices[-1] (K dim) to be replicated ("None").

        Output Layout:
        output_tensor_map = indices_tensor_map[:-1] + input_tensor_map[K:]
        If input_layout is None, input trailing dims are treated as replicated ("None").
        """
        input_layout, indices_layout = self._parse_input_layouts(layouts)

        input_shape, indices_shape = self._get_input_shapes(extra_args)
        k, trail_rank = self._get_k_and_trailing_rank(input_shape, indices_shape)

        input_tensor_map, indices_tensor_map = self._validate_tensor_maps(
            input_layout, indices_layout, k
        )

        # Output sharding: inherit indices[:-1] + input_x[K:].
        if input_tensor_map is None:
            output_tensor_map = tuple(indices_tensor_map[:-1]) + ("None",) * trail_rank
        else:
            output_tensor_map = tuple(indices_tensor_map[:-1]) + tuple(input_tensor_map[k:])

        output_layout = Layout(
            mesh_shape=indices_layout.mesh_shape,
            alias_name=indices_layout.alias_name,
            rank_list=indices_layout.rank_list,
        )

        if output_tensor_map:
            output_layout = output_layout(*output_tensor_map)
        else:
            output_layout = output_layout("None")

        return output_layout

    def _parse_input_layouts(self, layouts):
        """Parse and validate input layouts."""
        if len(layouts) < 2:
            raise ValueError(
                f"Operation {self.op_name} requires at least 2 input layouts, but got {len(layouts)}"
            )

        input_layout, indices_layout = layouts[0], layouts[1]

        # Extra inputs are allowed only when they are non-tensor args (layout is None).
        for extra_layout in layouts[2:]:
            if extra_layout is not None:
                raise ValueError(
                    f"Operation {self.op_name} only supports 2 tensor inputs, but got extra tensor layout: "
                    f"{extra_layout}"
                )

        # For GatherNd: input_layout can be None (treated as fully replicated), but indices_layout must exist.
        if indices_layout is None or not hasattr(indices_layout, "alias_tensor_map"):
            raise ValueError(f"Operation {self.op_name}: Indices layout cannot be None")

        return input_layout, indices_layout

    def _validate_tensor_maps(self, input_layout, indices_layout, k):
        """Validate tensor maps constraints for GatherNd."""
        indices_tensor_map = indices_layout.alias_tensor_map

        # Validate: indices tensor_map must exist and last dimension cannot be split.
        if not indices_tensor_map:
            raise ValueError(f"Operation {self.op_name}: indices tensor_map cannot be empty")

        last_axis = indices_tensor_map[-1]
        if not self._is_none_axis(last_axis):
            raise ValueError(
                f"Operation {self.op_name}: The last dimension of indices cannot be split. "
                f"Got indices[-1] = {last_axis}"
            )

        # Validate input only when layout is provided.
        input_tensor_map = None
        if input_layout is not None:
            input_tensor_map = input_layout.alias_tensor_map

            if k > len(input_tensor_map):
                raise ValueError(
                    f"Operation {self.op_name}: indices last dim (K={k}) is larger than input rank "
                    f"({len(input_tensor_map)})"
                )

            # Indexed dims [0:K) must be replicated.
            for axis_name in input_tensor_map[:k]:
                if not self._is_none_axis(axis_name):
                    raise ValueError(
                        f"Operation {self.op_name}: input_x cannot be split on indexed dims [0:{k}). "
                        f"These dims must be 'None', but got tensor_map: {input_tensor_map}"
                    )

        return input_tensor_map, indices_tensor_map

    def _get_input_shapes(self, extra_args):
        """Get input and indices shapes from extra_args (WithShape suffix required)."""
        input_shapes = None
        if extra_args and hasattr(extra_args[-1], "__len__") and len(extra_args[-1]) >= 2:
            input_shapes = extra_args[-1]

        if input_shapes is None:
            raise ValueError(
                f"Operation {self.op_name}: missing input_shapes in extra_args. "
                f"Please configure yaml with infer_layout_suffix: WithShape."
            )

        input_shape = input_shapes[0]
        indices_shape = input_shapes[1]
        if input_shape is None or indices_shape is None:
            raise ValueError(f"Operation {self.op_name}: input_shapes contains None: {input_shapes}")

        input_shape = self._normalize_shape(input_shape, "input")
        indices_shape = self._normalize_shape(indices_shape, "indices")

        if len(indices_shape) < 1:
            raise ValueError(f"Operation {self.op_name}: indices shape invalid: {indices_shape}")

        return input_shape, indices_shape

    def _normalize_shape(self, shape, name):
        """Normalize shape-like object to tuple of int."""
        try:
            norm = tuple(shape)
        except TypeError as err:
            raise ValueError(f"Operation {self.op_name}: {name} shape is not iterable: {shape}") from err

        try:
            norm = tuple(int(dim) for dim in norm)
        except (TypeError, ValueError) as err:
            raise ValueError(f"Operation {self.op_name}: {name} shape contains non-integer dims: {norm}") from err

        return norm

    def _get_k_and_trailing_rank(self, input_shape, indices_shape):
        """Compute K and trailing rank = len(input_shape) - K, where K is indices_shape[-1]."""
        k = indices_shape[-1]
        try:
            k = int(k)
        except (TypeError, ValueError) as err:
            raise ValueError(f"Operation {self.op_name}: indices last dim (K) is invalid: {k}") from err

        if k <= 0:
            raise ValueError(f"Operation {self.op_name}: indices last dim (K) must be positive, but got {k}")

        trail_rank = len(input_shape) - k
        if trail_rank < 0:
            raise ValueError(
                f"Operation {self.op_name}: indices last dim (K={k}) is larger than input rank ({len(input_shape)})"
            )

        return k, trail_rank

    def _is_none_axis(self, axis_name):
        """
        Check if an axis name represents no sharding.
        """
        if axis_name == "None":
            return True

        if isinstance(axis_name, tuple):
            return all(name == "None" for name in axis_name)

        return False
