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


class SliceExtDistributedOp(DistributedOp):
    """Distributed implementation for SliceExt operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layouts for Split operator.

        Rules:
        1. Shared axis can not be split.

        Args:
            layouts (Layout): Layout of input tensor
            extra_args (list): split size or sections, axis, input shape

        Returns:
            tuple: Layouts for output tensors
        """

        input_layout = layouts[0]
        axis = extra_args[0]
        # Check shared axis can not be split.
        tensor_map = input_layout.tensor_map
        if tensor_map[axis] != -1:
            raise ValueError(f"Can not split tensor at sharded axis[{axis}], layout: {input_layout}")
        return input_layout
