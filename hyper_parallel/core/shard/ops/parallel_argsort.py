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
Distributed implementation for Argsort operator.
"""

from .parallel_ops import DistributedOp


class ArgsortDistributedOp(DistributedOp):
    """Distributed implementation for torch.argsort."""

    def infer_layout(self, layouts, extra_args=None):
        """
        Infer output layout for torch.argsort.

        PyTorch semantics:
          - Signature: torch.argsort(input, dim=-1, descending=False, stable=False)
          - Returns the indices that sort a tensor along a given dimension.
          - The output tensor has the exact same shape as the input tensor.
          - Distributed constraint: The dimension being sorted (`dim`) MUST NOT be sharded,
            as sorting requires full visibility of the elements along that axis.

        Args:
            layouts (tuple): Layouts of inputs. Expected:
                layouts[0] (Layout): Input tensor layout (required).
            extra_args (list): Additional scalar arguments. Expected:
                extra_args[0] (int): The dimension to sort along (default: -1).
                extra_args[1] (bool): descending flag.
                extra_args[2] (bool): stable flag.

        Returns:
            Layout: Output tensor layout (identical to input layout, provided the sorted
                    dimension is valid and unsharded).
        """
        if not layouts or layouts[0] is None:
            raise ValueError(
                f"Operation {self.op_name}: argsort requires a valid input tensor layout."
            )

        input_layout = layouts[0]
        in_tensor_map = input_layout.tensor_map
        input_ndim = len(in_tensor_map)

        # 1. Parse 'dim' from extra_args (default is -1 per PyTorch semantics)
        dim = -1
        if extra_args and len(extra_args) > 0:
            # We assume the first extra argument is 'dim' based on positional unpacking
            if isinstance(extra_args[0], int):
                dim = extra_args[0]
            # Fallback logic in case kwargs ordering puts booleans first
            elif isinstance(extra_args[0], bool) and len(extra_args) > 1 and isinstance(extra_args[1], int):
                dim = extra_args[1]

        # 2. Normalize negative dimensions
        actual_dim = dim
        if actual_dim < 0:
            actual_dim += input_ndim

        # 3. Validate dimension bounds
        if actual_dim < 0 or actual_dim >= input_ndim:
            raise ValueError(
                f"Operation {self.op_name}: dim {dim} is out of bounds for "
                f"tensor of dimension {input_ndim}."
            )

        # 4. Enforce Distributed Constraint: The sorting dimension cannot be sharded.
        # In tensor_map, a value of -1 means unsharded. Any value >= 0 represents
        # the device mesh axis index that shards this dimension.
        if in_tensor_map[actual_dim] != -1:
            raise ValueError(
                f"Operation {self.op_name}: Cannot perform argsort along dimension {dim} "
                f"because it is currently sharded across device mesh axis {in_tensor_map[actual_dim]}. "
                f"Please redistribute the tensor to unshard this dimension before sorting."
            )

        # 5. The shape and distribution of the indices are identical to the input
        return input_layout
