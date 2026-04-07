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
Distributed implementation for Stack operator.
"""

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp

# pylint: disable=unused-argument
def _normalize_stack_args(tensors, dim=0, **kwargs):
    """
    Normalize arguments for torch.stack.
    """
    return (tensors,), {'dim': dim}


class StackDistributedOp(DistributedOp):
    """Distributed implementation for Stack operator."""

    def preprocess(self, args, kwargs):
        """
        Preprocess input arguments and extract local components.

        Normalizes args, explicitly extracts parameters, and prepares
        local tensors and cache values without validation logic.
        """
        args, kwargs = _normalize_stack_args(*args, **kwargs)

        # Explicit parameter extraction
        tensors = args[0]
        dim = kwargs['dim']

        # Extract local tensors and layouts
        local_tensors = tuple(t.to_local() if hasattr(t, "to_local") else t for t in tensors)
        layouts = [getattr(t, "layout", None) for t in tensors]

        # Construct local args and kwargs for the inner op execution
        local_args = (local_tensors,)
        local_kwargs = {'dim': dim}

        # Flatten layouts and append dim for caching and inference
        cache_values = layouts + [dim]

        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values):
        """
        Infer output layout based on cache values for torch.stack.

        All validation logic (e.g., empty checks, layout consistency,
        and dimension bounds) is handled here.
        """
        layouts = cache_values[:-1]
        dim = cache_values[-1]

        # 1. Validation Logic
        if not layouts:
            raise ValueError(f"Operation {self.op_name}: stack requires at least one input tensor.")

        valid_layouts = [lyt for lyt in layouts if lyt is not None]

        if not valid_layouts:
            raise ValueError(f"Operation {self.op_name}: stack requires at least one input DTensor.")

        # Reference layout to validate consistency across all input tensors
        base_layout = valid_layouts[0]
        for layout in valid_layouts[1:]:
            if layout != base_layout:
                raise ValueError(
                    f"Operation {self.op_name}: All input tensors must have the same layout. "
                    f"Expected layout: {base_layout}, Mismatched layout: {layout}"
                )

        ndim = len(base_layout.tensor_map)

        # Normalize and validate the dimension. For stack, valid range is [-ndim - 1, ndim]
        actual_dim = dim if dim >= 0 else dim + ndim + 1
        if actual_dim < 0 or actual_dim > ndim:
            raise ValueError(
                f"Operation {self.op_name}: Dimension out of range (expected to be in range of "
                f"[{-ndim - 1}, {ndim}], but got {dim})"
            )

        # 2. Layout Inference Logic
        in_tensor_map = base_layout.tensor_map

        # Insert an unsharded mapping (-1) at the newly created dimension
        output_tensor_map = list(in_tensor_map)
        output_tensor_map.insert(actual_dim, -1)

        mesh_shape = base_layout.mesh_shape
        alias_name = base_layout.alias_name
        rank_list = base_layout.rank_list

        def idx_to_alias(idx, aliases):
            """Map tensor_map index back to the alias string."""
            if idx == -1:
                return "None"
            # Reverse indexing mapped to the framework's layout design
            return aliases[len(aliases) - idx - 1]

        output_alias_map = tuple(idx_to_alias(idx, alias_name) for idx in output_tensor_map)

        # Reconstruct the output layout
        output_layout = Layout(
            mesh_shape=mesh_shape,
            alias_name=alias_name,
            rank_list=rank_list
        )

        # Apply the placement mappings via the __call__ method
        output_layout = output_layout(*output_alias_map)

        # Returns a tuple of output layouts and None for extra_args
        return ((output_layout,), None)
