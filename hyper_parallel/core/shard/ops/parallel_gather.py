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

from typing import Tuple

from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.platform import get_platform
from .parallel_ops import DistributedOp


def _normalize_index_select_args(input_tensor, dim, index):
    return (input_tensor, dim, index), {}


def _normalize_gatherd_args(input_tensor, dim, index, **kwargs):
    """Normalize torch.gather / GatherD args, forwarding keyword-only params."""
    return (input_tensor, dim, index), kwargs


def _normalize_gathernd_args(input_tensor, indices):
    return (input_tensor, indices), {}


class IndexSelectDistributedOp(DistributedOp):
    """Distributed implementation for Index Select operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for IndexSelect operator.

        Args:
            args (tuple): Input arguments (input, dim, index).
            kwargs (dict): Keyword arguments.

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, _ = _normalize_index_select_args(*args, **kwargs)
        input_tensor, dim, index = args[0], args[1], args[2]
        local_args = (input_tensor.to_local(), dim, index.to_local())
        local_kwargs = {}
        cache_values = [input_tensor.layout, index.layout, dim]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:  # pylint: disable=W0221
        """
        Infer output layouts for Index Select operations.

        Rules:
            1. Input and index must not have Partial status.
            2. cache_values must contain [input_layout, index_layout, dim].
            3. dim must be within the valid input rank range.
            4. index must be one-dimensional.
            5. Output replaces the selected input dimension with the index layout.
            6. If the selected input dimension is sharded, output carries Partial('sum').

        Args:
            cache_values (list): [input_layout, index_layout, dim].

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If input layouts are not compatible or have partial status.
        """
        if not self._allow_partial_inputs:
            self._check_partial_inputs(cache_values[:2])

        # Parse layout info
        p_layout, i_layout, axis = cache_values[0], cache_values[1], cache_values[2]

        p_tensor_map = p_layout.alias_tensor_map
        i_tensor_map = i_layout.alias_tensor_map

        # 1. Validate the axis range before any manipulation
        if axis < -len(p_tensor_map) or axis >= len(p_tensor_map):
            raise ValueError(
                f"For {self.op_name}, dim value {axis} is out of valid range"
            )

        # 2. Convert negative axis to positive index to avoid Python slicing bugs
        if axis < 0:
            axis += len(p_tensor_map)

        if len(i_tensor_map) != 1:
            raise ValueError(
                f"For {self.op_name}, index is not a one-dimensional Tensor"
            )

        # 3. Create output layout map
        # We allow sharding on the `axis`. Since `index_select` replaces the `axis`
        # dimension with the `index` dimension, if `axis` was sharded, that mesh
        # dimension is removed from the output tensor map.
        output_tensor_map = list(p_tensor_map[:axis]) + list(i_tensor_map) + list(p_tensor_map[axis + 1 :])

        output_layout = Layout(
            mesh_shape=p_layout.mesh_shape,
            alias_name=p_layout.alias_name,
            rank_list=p_layout.rank_list,
        )
        output_layout = output_layout(*output_tensor_map)

        # 4. Implicit Communication via Partial Layout
        # If the gather axis was sharded, the local output will only be a masked partial result.
        # We set the output layout to Partial('sum') for that specific mesh dimension so the
        # OpDispatcher handles the AllReduce automatically when this tensor is used later.
        shard_mesh_dim_name = p_tensor_map[axis]
        if shard_mesh_dim_name != "None":
            # Handle possible multi-axis sharding tuple
            if isinstance(shard_mesh_dim_name, tuple):
                for dim_name in shard_mesh_dim_name:
                    if dim_name != "None":
                        output_layout.set_partial_by_dev_axis(dim_name, 'sum')
            else:
                output_layout.set_partial_by_dev_axis(shard_mesh_dim_name, 'sum')

        return ((output_layout,), None)

    def get_expand_impl(self, func, infer_result, cache_values):  # pylint: disable=W0221
        """
        Get the expanded execution implementation for Index Select.
        """
        p_layout = cache_values[0]
        axis = cache_values[2]
        if axis < 0:
            axis += len(p_layout.alias_tensor_map)

        shard_mesh_dim_name = p_layout.alias_tensor_map[axis]

        # If the axis is NOT sharded, fallback to standard execution
        if shard_mesh_dim_name == "None":
            return None

        # If the axis IS sharded, return a custom function with Masking ONLY.
        # The explicit AllReduce is completely removed.
        def expand_impl(input_tensor, dim, index, **kwargs):
            platform = get_platform()
            mesh = p_layout.mesh

            # Fetch the communication group for the sharded mesh dimension
            if isinstance(shard_mesh_dim_name, tuple):
                target_dim_name = next(d for d in shard_mesh_dim_name if d != "None")
            else:
                target_dim_name = shard_mesh_dim_name

            comm_group_info = mesh.get_comm_group_by_axis(target_dim_name)
            group = comm_group_info.group if hasattr(comm_group_info, 'group') else comm_group_info

            # Get the rank of the current device within this specific communication group
            group_rank = platform.get_group_local_rank(group=group)

            # Calculate global index boundaries for the local chunk
            local_dim_size = input_tensor.shape[dim]
            start_idx = group_rank * local_dim_size
            end_idx = start_idx + local_dim_size

            # 1. Compute mask: True for indices that belong to the current rank
            mask = (index >= start_idx) & (index < end_idx)

            # 2. Shift global indices to local indices
            safe_index = index - start_idx

            # Clamp safe_index to valid local ranges to prevent CUDA out-of-bounds
            # errors during the local index_select (invalid ones will be masked out anyway).
            safe_index = safe_index.clamp(min=0, max=local_dim_size - 1)

            # 3. Perform local index_select using tensor's built-in method
            local_out = input_tensor.index_select(dim, safe_index, **kwargs)

            # 4. Mask out the invalid indices (set them to 0)
            # Reshape the 1D mask to broadcast against the output shape
            mask_shape = [1] * local_out.ndim
            mask_shape[dim] = -1
            mask_reshaped = mask.reshape(mask_shape).to(local_out.dtype)

            local_out = local_out * mask_reshaped

            # Return the partial local tensor directly. The framework's layout engine
            # and OpDispatcher will trigger the AllReduce when this Partial tensor
            # is redistributed to a non-partial layout.
            return local_out

        return expand_impl


class GatherDDistributedOp(DistributedOp):
    """Distributed implementation for GatherD operator.
    
    GatherD gathers values along a specified axis from the input tensor using the index tensor.
    
    Signature: GatherD(input, dim, index) -> output
    
    Key constraints:
    - Input and index must have the same number of dimensions
    - Output inherits the sharding pattern of the input tensor
    """

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for GatherD operator.

        Args:
            args (tuple): Input arguments (input, dim, index).
            kwargs (dict): Keyword arguments.

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_gatherd_args(*args, **kwargs)
        input_tensor, dim, index = args[0], args[1], args[2]
        local_args = (input_tensor.to_local(), dim, index.to_local())
        local_kwargs = kwargs
        cache_values = [input_tensor.layout, index.layout, dim]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:  # pylint: disable=W0221
        """
        Infer output layouts for GatherD operations.

        Rules:
            1. Input and index must not have Partial status.
            2. cache_values must contain [input_layout, index_layout, dim].
            3. Input and index must have the same rank.
            4. dim must be within the valid input rank range.
            5. Input and index must use the same sharding on non-dim axes.
            6. Output inherits index layout and becomes Partial('sum') when dim is sharded.

        Args:
            cache_values (list): [input_layout, index_layout, dim].

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If input layouts are not compatible or have partial status.
        """
        if not self._allow_partial_inputs:
            self._check_partial_inputs(cache_values[:2])

        input_layout, index_layout, dim = cache_values[0], cache_values[1], cache_values[2]
        # Validate layouts exist
        if input_layout is None or not hasattr(input_layout, "tensor_map"):
            raise ValueError(f"For {self.op_name}, input layout cannot be None")
        if index_layout is None or not hasattr(index_layout, "tensor_map"):
            raise ValueError(f"For {self.op_name}, index layout cannot be None")
        input_tensor_map = input_layout.alias_tensor_map
        index_tensor_map = index_layout.alias_tensor_map
        # Validate same rank
        if len(input_tensor_map) != len(index_tensor_map):
            raise ValueError(
                f"For {self.op_name}, input and index must have the same number of dimensions. "
                f"Got input rank={len(input_tensor_map)}, index rank={len(index_tensor_map)}"
            )
        # Validate dim is in valid range
        rank = len(input_tensor_map)
        if dim < -rank or dim >= rank:
            raise ValueError(
                f"For {self.op_name}, dim value {dim} is out of valid range [{-rank}, {rank-1}]"
            )
        # Normalize negative dim
        if dim < 0:
            dim = dim + rank
        for axis, (input_axis_map, index_axis_map) in enumerate(zip(input_tensor_map, index_tensor_map)):
            if axis == dim:
                continue
            if input_axis_map != index_axis_map:
                raise ValueError(
                    f"For {self.op_name}, input and index must use the same sharding on non-dim axis {axis}. "
                    f"Got input tensor_map={input_tensor_map}, index tensor_map={index_tensor_map}, dim={dim}"
                )
        # Output inherits index layout
        output_layout = Layout(
            mesh_shape=index_layout.mesh_shape,
            alias_name=index_layout.alias_name,
            rank_list=index_layout.rank_list,
        )
        output_layout.set_tensor_map(index_layout.tensor_map)
        dim_axis_name = input_tensor_map[dim]
        if dim_axis_name != "None":
            # pylint: disable=protected-access
            # Inherit current partial state from index layout
            output_layout._partial = list(index_layout.partial)
            if isinstance(dim_axis_name, tuple):
                for axis_name in dim_axis_name:
                    if axis_name != "None":
                        output_layout.set_partial_by_dev_axis(axis_name, 'sum')
            else:
                output_layout.set_partial_by_dev_axis(dim_axis_name, 'sum')
        # pylint: disable=protected-access
        # Rebuild readable alias tensor map
        output_layout._alias_tensor_map = output_layout._build_readable_tensor_map()
        # pylint: disable=protected-access
        # Sync tensor_map to placement representation
        output_layout.tensor_map_to_placement()
        # Update compact string description
        output_layout.update_compact_str()
        return ((output_layout,), None)

    def get_expand_impl(self, func, infer_result, cache_values):  # pylint: disable=W0221
        """
        Returns the execution implementation wrapper for distributed GatherD.
        
        When the dim axis is sharded, each rank gathers from its local slice of the input tensor.
        The indices need to be adjusted to account for the local partition offset.
        
        Args:
            func: The original GatherD function to wrap
            infer_result: The inferred output layouts and extra info
            cache_values: [input_layout, index_layout, dim]
            
        Returns:
            Callable: Distributed implementation wrapper, or None if no sharding
        """
        input_layout = cache_values[0]
        dim = cache_values[2]
        if dim < 0:
            dim += len(input_layout.tensor_map)
        input_alias_map = input_layout.alias_tensor_map
        # Check if dim axis is sharded (enhanced MP)
        if input_alias_map[dim] == "None": # native sharding, no need for custom implementation
            return None

        dim_axis_name = input_alias_map[dim]
        if isinstance(dim_axis_name, tuple):
            dim_axis_name = next(axis for axis in dim_axis_name if axis != "None")

        def distributed_gatherd_impl(*args, **kwargs):
            """
            Distributed GatherD implementation for sharded dim axis.
            
            Each rank gathers from its local slice of input tensor.
            Indices are adjusted by subtracting the local partition offset.
            """
            input_tensor = args[0]
            index_tensor = args[2]
            # Calculate local partition offset for the dim axis
            mesh = input_layout.mesh
            mesh_dim_idx = input_layout.alias_name.index(dim_axis_name)
            # Get the coordinate of current rank along the mesh dimension
            dim_coord = mesh.get_local_rank(mesh_dim_idx)
            # Calculate the size of input tensor's dim dimension per partition
            input_dim_size = input_tensor.shape[dim]
            # Calculate the starting index of local partition
            local_start_index = int(dim_coord * input_dim_size)
            local_end_index = int(local_start_index + input_dim_size)
            # Adjust indices: subtract local_start_index to map global indices to local range
            # This is similar to how Embedding shifts indices for Row Parallelism
            adjusted_index = index_tensor - local_start_index
            # Create mask to identify out-of-bounds indices
            # Indices outside [0, local_dim_size) belong to other partitions
            mask = (index_tensor >= local_start_index) & (index_tensor < local_end_index)
            # Cross-platform cast to matching int dtype
            mask_int = mask.to(index_tensor.dtype)
            # Zero out invalid indices to prevent out-of-bounds access
            safe_index = adjusted_index * mask_int
            # Replace original index tensor with adjusted index
            new_args = list(args)
            new_args[2] = safe_index
            # Execute native GatherD with adjusted indices
            output = func(*new_args, **kwargs)
            # Zero out outputs corresponding to invalid indices
            mask_int = mask_int.to(output.dtype)
            output = output * mask_int
            return output
        return distributed_gatherd_impl


class GatherNdDistributedOp(DistributedOp):
    """Distributed implementation for GatherNd operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for GatherNd operator.

        NOTE: aclop packed-args normalization (for MindSpore aclop operators
        that pack args as ``(prim, name, (real_args...))``) is handled
        upstream in ``OpDispatcher._dispatch_layout_infer`` via
        ``_normalize_aclop_args``. This method receives clean unpacked args.

        Args:
            args (tuple): Input arguments (input, indices).
            kwargs (dict): Keyword arguments.

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, _ = _normalize_gathernd_args(*args, **kwargs)
        input_tensor, indices = args[0], args[1]
        local_input = input_tensor.to_local() if hasattr(input_tensor, "_layout") else input_tensor
        local_indices = indices.to_local()
        local_args = (local_input, local_indices)
        local_kwargs = {}
        cache_values = [
            input_tensor.layout if hasattr(input_tensor, "_layout") else None,
            indices.layout,
            input_tensor.shape,
            indices.shape,
        ]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:  # pylint: disable=W0221
        """
        Infer output layout for GatherNd.

        Rules:
            1. Input and indices must not have Partial status.
            2. cache_values must contain [input_layout_or_None, indices_layout, input_shape, indices_shape].
            3. indices[-1] (K) must be replicated.
            4. input indexed dims [0:K) must be replicated when input layout is provided.
            5. Output inherits indices[:-1] sharding plus input trailing dims input[K:].

        For GatherNd: out.shape = indices.shape[:-1] + input_x.shape[K:], where K = indices.shape[-1].

        This implementation:
            - Inherits sharding from indices[:-1].
            - Allows sharding on input_x trailing dims input_x[K:].
            - Requires input_x[:K] to be replicated ("None") if input_layout is provided.
            - Requires indices[-1] (K dim) to be replicated ("None").

        Output Layout:
        output_tensor_map = indices_tensor_map[:-1] + input_tensor_map[K:]
        If input_layout is None, input trailing dims are treated as replicated ("None").

        Args:
            cache_values (list): [input_layout_or_None, indices_layout, input_shape, indices_shape].

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If input layouts, tensor maps, or shapes violate the rules above.
        """
        input_layout, indices_layout = self._parse_input_layouts(cache_values[:2])
        if not self._allow_partial_inputs:
            self._check_partial_inputs([input_layout, indices_layout])

        input_shape, indices_shape = self._get_input_shapes(cache_values[2:])
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

        return ((output_layout,), None)

    def _parse_input_layouts(self, layouts):
        """Parse and validate input layouts."""
        if len(layouts) < 2:
            raise ValueError(
                f"For {self.op_name}, requires at least 2 input layouts, but got {len(layouts)}"
            )

        input_layout, indices_layout = layouts[0], layouts[1]

        # Extra inputs are allowed only when they are non-tensor args (layout is None).
        for extra_layout in layouts[2:]:
            if extra_layout is not None:
                raise ValueError(
                    f"For {self.op_name}, only supports 2 tensor inputs, but got extra tensor layout: "
                    f"{extra_layout}"
                )

        # For GatherNd: input_layout can be None (treated as fully replicated), but indices_layout must exist.
        if indices_layout is None or not hasattr(indices_layout, "alias_tensor_map"):
            raise ValueError(f"For {self.op_name}, indices layout cannot be None")

        return input_layout, indices_layout

    def _validate_tensor_maps(self, input_layout, indices_layout, k):
        """Validate tensor maps constraints for GatherNd."""
        indices_tensor_map = indices_layout.alias_tensor_map

        # Validate: indices tensor_map must exist and last dimension cannot be split.
        if not indices_tensor_map:
            raise ValueError(f"For {self.op_name}, indices tensor_map cannot be empty")

        last_axis = indices_tensor_map[-1]
        if not self._is_none_axis(last_axis):
            raise ValueError(
                f"For {self.op_name}, the last dimension of indices cannot be split. "
                f"Got indices[-1] = {last_axis}"
            )

        # Validate input only when layout is provided.
        input_tensor_map = None
        if input_layout is not None:
            input_tensor_map = input_layout.alias_tensor_map

            if k > len(input_tensor_map):
                raise ValueError(
                    f"For {self.op_name}, indices last dim (K={k}) is larger than input rank "
                    f"({len(input_tensor_map)})"
                )

            # Indexed dims [0:K) must be replicated.
            for axis_name in input_tensor_map[:k]:
                if not self._is_none_axis(axis_name):
                    raise ValueError(
                        f"For {self.op_name}, input_x cannot be split on indexed dims [0:{k}). "
                        f"These dims must be 'None', but got tensor_map: {input_tensor_map}"
                    )

        return input_tensor_map, indices_tensor_map

    def _get_input_shapes(self, shape_values):
        """Get input and indices shapes from cache values."""
        input_shapes = None
        if shape_values and len(shape_values) == 2:
            input_shapes = shape_values

        if input_shapes is None:
            raise ValueError(
                f"For {self.op_name}, missing input_shapes in cache_values."
            )

        input_shape = input_shapes[0]
        indices_shape = input_shapes[1]
        if input_shape is None or indices_shape is None:
            raise ValueError(f"For {self.op_name}, input_shapes contains None: {input_shapes}")

        input_shape = self._normalize_shape(input_shape, "input")
        indices_shape = self._normalize_shape(indices_shape, "indices")

        if len(indices_shape) < 1:
            raise ValueError(f"For {self.op_name}, indices shape invalid: {indices_shape}")

        return input_shape, indices_shape

    def _normalize_shape(self, shape, name):
        """Normalize shape-like object to tuple of int."""
        try:
            norm = tuple(shape)
        except TypeError as err:
            raise ValueError(f"For {self.op_name}, {name} shape is not iterable: {shape}") from err

        try:
            norm = tuple(int(dim) for dim in norm)
        except (TypeError, ValueError) as err:
            raise ValueError(f"For {self.op_name}, {name} shape contains non-integer dims: {norm}") from err

        return norm

    def _get_k_and_trailing_rank(self, input_shape, indices_shape):
        """Compute K and trailing rank = len(input_shape) - K, where K is indices_shape[-1]."""
        k = indices_shape[-1]
        try:
            k = int(k)
        except (TypeError, ValueError) as err:
            raise ValueError(f"For {self.op_name}, indices last dim (K) is invalid: {k}") from err

        if k <= 0:
            raise ValueError(f"For {self.op_name}, indices last dim (K) must be positive, but got {k}")

        trail_rank = len(input_shape) - k
        if trail_rank < 0:
            raise ValueError(
                f"For {self.op_name}, indices last dim (K={k}) is larger than input rank ({len(input_shape)})"
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
