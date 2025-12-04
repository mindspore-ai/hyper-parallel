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
Activation with axis distributed operator implementation.
"""

from .parallel_ops import DistributedOp

class ActivationWithAxisDistributedOp(DistributedOp):
    """
    Distributed implementation for activation-with-axis operators (e.g., softmax).

    Inherits from DistributedOp and provides activation-with-axis specific implementations.
    """

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layouts for activation-with-axis operations.

        For activation-with-axis operations, all inputs should have the same layout,
        and the output will have the same layout.

        Args:
            primitive: Primitive instance
            layouts: Layouts of input tensors

        Returns:
            tuple: Layout for output tensor.

        Raises:
            ValueError: If input layouts are not compatible.
        """
        if not layouts:
            return None

        self.check_layout(layouts, extra_args)
        # Verify all layouts are the same
        first_layout = None
        for layout in layouts:
            if first_layout is None and layout is not None:
                first_layout = layout
            if layout is not None and first_layout is not None and layout != first_layout:
                raise ValueError(
                    f"Operation {self.op_name} requires all tensor inputs to have the same layout. "
                    f"Input a: {first_layout}, Input b: {layout}")
        return first_layout

    def check_layout(self, layouts, extra_args):
        """
        check_layout
        """
        min_slice_num = 1
        x_dict = layouts[0].to_dict()
        x_dev = x_dict["tensor_map"]
        extra_args = extra_args[0]
        if not isinstance(extra_args, (int, tuple)):
            raise ValueError(
                f"Operation {self.op_name}: The extra args should be int or tuple, but got ({type(extra_args)})")
        extra_args = (extra_args,) if isinstance(extra_args, int) else extra_args
        for axis_index in extra_args:
            tensor_map = x_dev[axis_index]
            if tensor_map == -1:
                continue
            axis_strategy = x_dict["device_matrix"][len(x_dict["device_matrix"]) - tensor_map - 1]
            if axis_strategy != min_slice_num:
                raise ValueError(
                    f"Operation {self.op_name}: The strategy corresponding to axis dimension (in dim {axis_index}, "
                    f"now set to {axis_strategy}) is not 1")
