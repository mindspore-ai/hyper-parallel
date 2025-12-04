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
Element-wise distributed operator implementation.
"""

from .parallel_ops import DistributedOp

class ElementWiseDistributedOp(DistributedOp):
    """
    Distributed implementation for element-wise operators (e.g., Add, ReLU).

    Inherits from DistributedOp and provides element-wise specific implementations.
    """
    def infer_layout(self, layouts, extra_args):
        """
        Infer output layouts for element-wise operations.

        For element-wise operations, all inputs should have the same layout,
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

        # Verify all layouts are the same
        first_layout = None
        for layout in layouts:
            if first_layout is None and layout is not None:
                first_layout = layout
            if layout is not None and first_layout is not None and layout != first_layout:
                raise ValueError(
                    f"Element-wise operation {self.op_name} requires all tensor inputs to have the same layout. "
                    f"Input a: {first_layout}, Input b: {layout}")
        return first_layout
