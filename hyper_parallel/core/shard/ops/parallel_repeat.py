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
Distributed implementation for Repeat operator.
"""

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp

class RepeatDistributedOp(DistributedOp):
    """Distributed implementation for torch.Tensor.repeat."""

    def infer_layout(self, layouts, extra_args=None):
        """
        Infer output layout for torch.Tensor.repeat.

        PyTorch semantics:
          - Repeats this tensor along the specified dimensions.
          - If the number of repeat dimensions is larger than the tensor dimensions,
            the tensor is implicitly unsqueezed at the front.
          - The number of repeat dimensions cannot be smaller than the tensor dimensions.
          - Dimensions being repeated (>1 or 0) MUST be unsharded.

        Args:
            layouts (tuple): Layouts of inputs. Expected:
                layouts[0] (Layout): Input tensor layout (required).
            extra_args (tuple/list): Should contain the repeat sizes.

        Returns:
            Layout: Output tensor layout with:
                - New prepended dimensions: unsharded (-1)
                - Repeated existing dimensions (size != 1): unsharded (-1)
                - Preserved existing dimensions (size == 1): original sharding preserved
        """
        if not layouts or layouts[0] is None:
            raise ValueError(
                f"Operation {self.op_name}: repeat requires a valid input tensor layout."
            )

        input_layout = layouts[0]
        in_tensor_map = input_layout.tensor_map
        input_ndim = len(in_tensor_map)

        if not extra_args or len(extra_args) < 1:
            raise ValueError(
                f"Operation {self.op_name}: repeat requires repeat sizes in extra_args."
            )

        # Robustly handle sizes unpacking (e.g., if args are packed as a single tuple)
        if len(extra_args) == 1 and isinstance(extra_args[0], (tuple, list)):
            flat_args = extra_args[0]
        else:
            flat_args = extra_args

        # Normalize repeat sizes to tuple of ints
        repeats = []
        for arg in flat_args:
            if not isinstance(arg, int):
                arg = int(arg)
            repeats.append(arg)
        repeats = tuple(repeats)
        output_ndim = len(repeats)

        num_new_dims = output_ndim - input_ndim
        output_map = []

        # Rule 1: New prepended dimensions are always unsharded
        for i in range(num_new_dims):
            output_map.append(-1)

        # Rule 2: Process existing dimensions
        for i in range(input_ndim):
            repeat_idx = num_new_dims + i
            repeat_times = repeats[repeat_idx]

            if repeat_times == 1:
                # If the dimension is not repeated, keep the original sharding
                output_map.append(in_tensor_map[i])
            else:
                # If the dimension is repeated (or zeroed), it cannot be currently sharded
                if in_tensor_map[i] != -1:
                    raise ValueError(
                        f"Operation {self.op_name}: Cannot repeat dimension {i} which is sharded. "
                        f"Please redistribute (unshard) the tensor along this dimension first."
                    )
                # Repeated dimension remains unsharded in output
                output_map.append(-1)

        # Construct output layout mapping
        mesh_shape = input_layout.mesh_shape
        alias_name = input_layout.alias_name
        rank_list = input_layout.rank_list

        def idx_to_alias(idx, aliases):
            """Convert layout index back to alias string mapping"""
            if idx == -1:
                return "None"
            return aliases[len(aliases) - idx - 1]

        output_alias_map = tuple(idx_to_alias(idx, alias_name) for idx in output_map)

        # Instantiate new layout
        output_layout = Layout(
            mesh_shape=mesh_shape,
            alias_name=alias_name,
            rank_list=rank_list
        )
        output_layout = output_layout(*output_alias_map)

        return output_layout
