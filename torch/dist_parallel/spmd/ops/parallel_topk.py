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

    def infer_layout(self, input_layouts, extra_args=None):
        """
        Infer output layouts for TopK operator.

        TopK: values, indices = topk(input, k)

        Rules:
        1. Output layouts must match input layout exactly
        2. Both values and indices have same layout as input

        Args:
            input_layout (Layout): Layout of input tensor
            k (int, optional): The k in topk, not affecting layout

        Returns:
            tuple: Layouts for values and indices tensors
        """

        if not input_layouts[0]:
            raise ValueError(f"input_layouts is empty: {input_layouts[0]}")

        return input_layouts[0], input_layouts[0]
