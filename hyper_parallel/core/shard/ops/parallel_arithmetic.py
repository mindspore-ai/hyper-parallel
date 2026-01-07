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
Arithmetic ops with axis distributed operator implementation.
"""

from hyper_parallel.core.layout import Layout
from .parallel_ops import DistributedOp

class ArithmeticDistributedOp(DistributedOp):
    """
    Distributed implementation for arithmetic operators (e.g., softmax).

    Inherits from DistributedOp and provides arithmetic specific implementations.
    """

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layouts for arithmetic operations.

        For arithmetic operations, all inputs should have the same layout,
        and the output will have the same layout.

        Args:
            primitive: Primitive instance
            layouts: Layouts of input tensors

        Returns:
            tuple: Layout for output tensor.

        Raises:
            ValueError: If input layouts are not compatible.
        """
        x_layout, y_layout = layouts[:2]
        if y_layout is None:
            return x_layout
        if x_layout is None:
            return y_layout
        x_dict, y_dict = x_layout.to_dict(), y_layout.to_dict()
        if x_dict == y_dict:
            return x_layout

        tensormap0 = x_layout.tensor_map
        tensormap1 = y_layout.tensor_map
        len_diff = 0
        tensormap_big, tensormap_small, output_layout = (tensormap0, tensormap1, x_layout) \
            if self.tensormap_length(tensormap0) > self.tensormap_length(tensormap1) else \
            (tensormap1, tensormap0, y_layout)
        output_tensormap = list(tensormap_big)
        len_diff = self.tensormap_length(tensormap_big) - self.tensormap_length(tensormap_small)
        for i in range(self.tensormap_length(tensormap_small)):
            if isinstance(tensormap_small[i], tuple) or isinstance(tensormap_big[i], tuple):
                self.check_tensormap(tensormap_small[i], tensormap_big[i], i)
                continue
            output_tensormap[i + len_diff] = tensormap_small[i] if abs(tensormap_big[i + len_diff]) == 1 else \
                tensormap_big[i + len_diff]
        new_output_layout = Layout(
            mesh_shape=output_layout.mesh_shape,
            alias_name=output_layout.alias_name,
            rank_list=output_layout.rank_list
        )
        output_alias_tensormap = output_layout.alias_tensor_map
        output_layout = new_output_layout(*output_alias_tensormap)
        return output_layout

    def tensormap_length(self, tensormap):
        """
        compute tensor map length
        """
        num = len(tensormap)
        for x in tensormap:
            if isinstance(x, int) and x == -1:
                num -= 1
        return num

    def check_tensormap(self, tensormap_big, tensormap_small, axis):
        """
        check_layout
        """
        if tensormap_big != tensormap_small:
            raise ValueError(
                f"Operation {self.op_name}: The tensor map ({tensormap_big}) of input0 in index {axis} "
                f"does not equal to tensor map ({tensormap_small}) of input1 in index {axis}")
