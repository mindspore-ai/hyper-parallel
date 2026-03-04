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
Distributed implementation for Concat operator.
"""

from .parallel_ops import DistributedOp


class ConcatDistributedOp(DistributedOp):
    """Distributed implementation for Concat."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for Concat and normalize the concatenation dimension.
        Raises an error if the specified concatenation dimension is sharded.

        Args:
            layouts (tuple): Layouts of input tensors and scalar arguments.
            extra_args (list): Additional arguments (e.g., dim). Modified in-place
                               to store the normalized positive dimension.

        Returns:
            Layout: The inferred output layout (identical to the input layouts).
        """
        # Filter out None values which correspond to scalar arguments (e.g., dim)
        valid_layouts = [layout for layout in layouts if layout is not None]

        if not valid_layouts:
            raise ValueError(f"Operation {self.op_name}: cat requires at least one input DTensor.")

        # In this framework, we assume inputs must be aligned to the same layout
        # for concatenation. We use the first valid layout as the reference.
        base_layout = valid_layouts[0]

        # Verify consistency across all valid input layouts
        for _, layout in enumerate(valid_layouts):
            if layout != base_layout:
                raise ValueError(
                    f"Operation {self.op_name}: All input tensors must have the same layout. "
                    f"Expected layout: {base_layout}, Mismatched layout: {layout}"
                )

        # Extract dim from extra_args, assuming the framework populates it correctly
        dim = extra_args[0] if extra_args else 0

        # Convert negative dim to positive
        ndim = len(base_layout.tensor_map)
        actual_dim = dim if dim >= 0 else dim + ndim

        # Check if the concatenation dimension is sharded
        mapping = base_layout.tensor_map[actual_dim]
        mapping_list = mapping if isinstance(mapping, tuple) else (mapping,)
        is_sharded = any(m != -1 for m in mapping_list)

        if is_sharded:
            raise ValueError(
                f"Operation {self.op_name}: Concatenation along a sharded dimension "
                f"(dim={dim}, normalized_dim={actual_dim}) is not supported."
            )

        # Store the normalized actual_dim back into extra_args
        # so get_expand_impl can use it directly as a positive integer
        if extra_args:
            extra_args[0] = actual_dim
        else:
            extra_args.append(actual_dim)

        return base_layout
