# Copyright 2025 Huawei Technologies Co., Ltd
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
Distributed implementation for TopK operator.
"""

from .parallel_ops import DistributedOp

class TopKDistributedOp(DistributedOp):
    """Distributed implementation for TopK operator."""

    def infer_layout(self, layouts, extra_args=None):
        """
        Infer output layouts for TopK operator.

        TopK: values, indices = topk(input, k, dim)

        Rules:
        1. dim = -1 if not specified.
        2. The dimension `dim` MUST be unsharded to ensure global top-k correctness.
        3. Both values and indices have same layout as input

        Args:
            layouts (tuple): Layouts of inputs. Expected:
                layouts[0] (Layout): Input tensor layout (required).
            extra_args (tuple, optional): Requires k and optionally contains dim. Expected:
                extra_args[0] (int, required): K value.
                extra_args[1] (int, optional): Dimension to compute topk. Defaults to -1.

        Returns:
            tuple: Layouts for values and indices tensors
        """
        if not layouts or layouts[0] is None:
            raise ValueError("topk requires a valid input tensor layout.")

        input_layout = layouts[0]
        in_tensor_map = input_layout.tensor_map

        dim = -1 # If dim is not given, the last dimension of the input is chosen.
        if len(extra_args) >= 2 and extra_args[1] is not None:
            dim = extra_args[1]
        input_dim = len(in_tensor_map)
        if dim < 0:
            dim = input_dim + dim # -1 represents the last dimension
        if not 0 <= dim < input_dim:
            raise ValueError(f"Dimension out of range (expected to be in [0, {input_dim}), but got {dim}).")

        # The chosen dim must NOT be sharded
        if in_tensor_map[dim] != -1:
            raise ValueError(
                f"Operation {self.op_name}: Cannot perform sharding on params along the chosen dim"
            )

        return input_layout, input_layout
