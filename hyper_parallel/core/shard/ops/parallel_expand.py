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
Distributed implementation for Expand operator.
"""

from hyper_parallel.core.layout import Layout
from .parallel_ops import DistributedOp

class ExpandDistributedOp(DistributedOp):
    """Distributed implementation for torch.Tensor.expand."""

    def infer_layout(self, layouts, extra_args=None):
        """
        Infer output layout for torch.Tensor.expand.

        PyTorch semantics:
          - Expands singleton dimensions (size 1) to larger sizes
          - Passing -1 preserves the original size of that dimension
          - Only dimensions with global size 1 can be expanded
          - Existing dimensions being expanded MUST be unsharded:

        Args:
            layouts (tuple): Layouts of inputs. Expected:
                layouts[0] (Layout): Input tensor layout (required).
            extra_args (tuple): Should contain 'sizes'. Expected:
                extra_args[0] (int): One element in desired expanded sizes (required).
                ...
                extra_args[n] (int): One element in desired expanded sizes (required).

        Returns:
            Layout: Output tensor layout with:
                - New dimensions: unsharded (-1)
                - Expanded existing dimensions: unsharded (-1)
                - Preserved dimensions (-1 in sizes): original sharding preserved
        """
        if not layouts or layouts[0] is None:
            raise ValueError(
                f"Operation {self.op_name}: expand requires a valid input tensor layout."
            )
        input_layout = layouts[0]
        in_tensor_map = input_layout.tensor_map
        input_ndim = len(in_tensor_map)

        if not extra_args or len(extra_args) < 1:
            raise ValueError(
                f"Operation {self.op_name}: expand requires 'sizes' parameter in extra_args."
            )
        output_ndim = len(extra_args)

        # Normalize sizes to tuple
        sizes = []
        for i in range(output_ndim):
            if not isinstance(extra_args[i], int):
                raise ValueError(
                f"Operation {self.op_name}: elements in 'sizes' parameter must be int."
            )
            sizes.append(extra_args[i])
        sizes = tuple(sizes)

        # output_ndim = len(sizes)
        num_new_dims = output_ndim - input_ndim

        # PyTorch only allows prepending new dimensions (not inserting in middle)
        if num_new_dims < 0:
            raise ValueError(
                f"Operation {self.op_name}: Cannot reduce dimensions with expand. "
                f"Input has {input_ndim} dims, requested {output_ndim} dims."
            )

        # Build output tensor map
        output_map = []

        # Rule 1: For the new dimensions, the size cannot be set to -1.
        for i in range(num_new_dims):
            if sizes[i] == -1:
                raise ValueError(
                    f"Operation {self.op_name}: Cannot use -1 for new dimension at position {i}. "
                )
            output_map.append(-1)  # Always unsharded

        # Rule 2: Process existing dimensions
        for i in range(input_ndim):
            output_dim_idx = num_new_dims + i
            requested_size = sizes[output_dim_idx]

            if requested_size == -1:
                # keep original sharding
                output_map.append(in_tensor_map[i])
            else:
                # Cannot expand dimension which is sharded
                if in_tensor_map[i] != -1:
                    raise ValueError(
                        f"Operation {self.op_name}: Cannot expand dimension {i} which is sharded."
                    )
                # Expanded dimension becomes unsharded in output
                output_map.append(-1)

        # Construct output layout
        mesh_shape = input_layout.mesh_shape
        alias_name = input_layout.alias_name
        rank_list = input_layout.rank_list

        def idx_to_alias(idx, aliases):
            if idx == -1:
                return "None"
            return aliases[len(aliases) - idx - 1]

        output_alias_map = tuple(idx_to_alias(idx, alias_name) for idx in output_map)

        output_layout = Layout(
            mesh_shape=mesh_shape,
            alias_name=alias_name,
            rank_list=rank_list
        )
        output_layout = output_layout(*output_alias_map)
        return output_layout


class ExpandAsDistributedOp(DistributedOp):
    """Distributed implementation for torch.Tensor.expand_as."""

    def infer_layout(self, layouts, extra_args=None):
        """
        Infer output layout for expand_as.

        PyTorch semantics:
          - Only dimensions with global size == 1 can be expanded to larger sizes
          - Dimensions with size > 1 must exactly match between input and target
          - Broadcast replicates a single value across the expanded dimension

        Critical sharding constraints:
          - Input dimensions with global size == 1 MUST be unsharded (-1)
          - When expanding a dimension (size 1 → N), this dimension must be unsharded in input layout
          - Expanded dimensions become unsharded in output
          - Non-expanded dimensions preserve their input sharding pattern

        Args:
            layouts (tuple): Layouts of inputs. Expected:
                layouts[0] (Layout): Input tensor layout (required).
                layouts[1] (Layout): Target tensor layout (No need).
            extra_args (tuple): Must contain shape information. Expected:
                extra_args[0][0] (tuple of int): Input global shape.
                extra_args[0][1] (tuple of int): Target global shape.
                
        Returns:
            Layout: Output tensor layout with sharding preserved for non-expanded
                    dimensions and unsharded for expanded dimensions.
        """
        # Validate input layout
        if not layouts or layouts[0] is None:
            raise ValueError(
                f"Operation {self.op_name}: expand requires a valid input tensor layout."
            )
        input_layout = layouts[0]
        in_tensor_map = input_layout.tensor_map
        input_ndim = len(in_tensor_map)

        # Extract shape information from extra_args
        if not extra_args or extra_args[0] is None or len(extra_args[0]) < 2:
            raise ValueError(
                f"Operation {self.op_name}: expand requires (input_global_shape, target_shape) "
                f"in extra_args."
            )
        input_global_shape = extra_args[0][0]
        target_shape = extra_args[0][1]

        if not isinstance(target_shape, (tuple, list)):
            raise ValueError(
                f"Operation {self.op_name}: target_shape must be tuple/list, got {type(target_shape)}."
            )
        if not isinstance(input_global_shape, (tuple, list)):
            raise ValueError(
                f"Operation {self.op_name}: input_global_shape must be tuple/list, got {type(input_global_shape)}."
            )

        target_shape = tuple(target_shape)
        input_global_shape = tuple(input_global_shape)
        target_ndim = len(target_shape)

        # PyTorch rule: target rank cannot be smaller than input rank
        if target_ndim < input_ndim:
            raise ValueError(
                f"Operation {self.op_name}: target shape {target_shape} (ndim={target_ndim}) cannot be "
                f"smaller than input shape {input_global_shape} (ndim={input_ndim})."
            )

        # Align dimensions (input to target)
        num_leading_implicit = target_ndim - input_ndim
        aligned_input_shape = (1,) * num_leading_implicit + input_global_shape
        aligned_tensor_map = (-1,) * num_leading_implicit + in_tensor_map

        # Validate expansion rules and build output tensor_map
        output_tensor_map = []
        for i, (in_size, tgt_size, shard_spec) in enumerate(
            zip(aligned_input_shape, target_shape, aligned_tensor_map)
        ):
            if in_size == tgt_size:
                # Dimension unchanged - preserve sharding pattern
                output_tensor_map.append(shard_spec)
            elif in_size == 1 and tgt_size > 1:
                # Dimension is expanded (broadcast) - must be unsharded
                if shard_spec != -1:
                    raise ValueError(
                        f"Operation {self.op_name}: Cannot expand sharded dimension {i} which is going to broadcast "
                        f"(global size 1 → {tgt_size})."
                    )
                output_tensor_map.append(-1)
            else:
                raise ValueError(
                    f"Operation {self.op_name}: Cannot expand dimension {i} from size {in_size} "
                    f"to {tgt_size}."
                )

        # Construct output layout with same mesh configuration
        mesh_shape = input_layout.mesh_shape
        alias_name = input_layout.alias_name
        rank_list = input_layout.rank_list

        # Convert tensor_map indices to alias strings for Layout constructor
        def idx_to_alias(idx, aliases):
            if idx == -1:
                return "None"
            return aliases[len(aliases) - idx - 1]

        output_map = tuple(idx_to_alias(idx, alias_name) for idx in output_tensor_map)

        output_layout = Layout(
            mesh_shape=mesh_shape,
            alias_name=alias_name,
            rank_list=rank_list
        )
        output_layout = output_layout(*output_map)
        return output_layout
