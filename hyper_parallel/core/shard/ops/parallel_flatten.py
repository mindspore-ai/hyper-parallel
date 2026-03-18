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
Distributed implementation for Flatten operator.
"""
from hyper_parallel.core.shard.ops.parallel_reshape import ReshapeDistributedOp


class FlattenDistributedOp(ReshapeDistributedOp):
    """Distributed implementation for torch.flatten."""

    def infer_layout(self, layouts, extra_args=None):
        """
        Infer output layout for torch.flatten.

        PyTorch semantics:
          - flatten(input, start_dim=0, end_dim=-1)
          - Flattens the input tensor starting from `start_dim` to `end_dim`.

        Args:
            layouts (tuple): Layouts of inputs.
            extra_args (tuple): Contains scalar arguments (start_dim, end_dim)
                                 and the input shapes list appended by WithShape suffix.

        Returns:
            Layout: Output tensor layout.
        """
        if not layouts or layouts[0] is None:
            raise ValueError(
                f"Operation {self.op_name}: flatten requires a valid input tensor layout."
            )

        input_layout = layouts[0]
        # WithShape suffix appends a list of input_shapes as the last item in extra_args
        input_shapes = extra_args[-1]
        input_shape = input_shapes[0]

        # PyTorch flatten defaults: start_dim=0, end_dim=-1
        start_dim = 0
        end_dim = -1

        # Parse scalar args (start_dim, end_dim) from extra_args
        num_scalar_args = len(extra_args) - 1
        if num_scalar_args > 0:
            start_dim = extra_args[0]
        if num_scalar_args > 1:
            end_dim = extra_args[1]

        ndim = len(input_shape)

        # Handle 0-D tensor case specifically to avoid parent class indexing errors
        if ndim == 0:
            out_layout = layouts[0].__class__.from_device_mesh(input_layout.mesh)
            out_layout.set_placements(input_layout.placements)
            out_layout.placement_to_tensor_map(1) # Flattened 0-D becomes 1-D of shape (1,)
            return out_layout

        # Handle negative dimensions
        if start_dim < 0:
            start_dim += ndim
        if end_dim < 0:
            end_dim += ndim

        # Validate dimension bounds
        if start_dim < 0 or start_dim >= ndim or end_dim < 0 or end_dim >= ndim:
            raise ValueError(f"Dimension out of range for flatten: start_dim={start_dim}, end_dim={end_dim}")


        # If start_dim > end_dim, PyTorch returns the tensor unchanged (identity operation).
        # This also covers the case where start_dim == end_dim (e.g., flatten(1, 1))
        # because the loop `range(start_dim, end_dim + 1)` would still execute, but
        # effectively for a single dimension, resulting in `dst_shape == input_shape`.
        # However, the underlying `_merge_unshared_axis` logic in ReshapeDistributedOp
        # is not designed for an identity mapping and introduces inconsistencies.
        # Thus, for an identity flatten, we should simply return the original layout.
        if start_dim >= end_dim:
            # If flattening a single dimension or a range that results in no actual flattening,
            # the layout should remain unchanged.
            return input_layout # Directly return the input layout


        # Calculate the target flattened shape
        flattened_size = 1
        for i in range(start_dim, end_dim + 1):
            flattened_size *= input_shape[i]
        dst_shape = list(input_shape[:start_dim]) + [flattened_size] + list(input_shape[end_dim + 1:])

        # Prepare args for ReshapeDistributedOp.infer_layout
        # Since self.op_name is 'flatten' (not in ["reshape", "view"]),
        # the parent class will treat this as the MindSpore Reshape branch,
        # which expects: extra_args = [dst_shape, input_shape]
        mock_extra_args = [dst_shape, input_shape]

        # Reuse parent class logic to safely resolve Shard tensor mapping merges
        out_layout, _ = super().infer_layout(layouts, mock_extra_args)

        # Returning purely the Layout object, which is what the WithShape suffix expects
        return out_layout
