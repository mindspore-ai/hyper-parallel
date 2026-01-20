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
Distributed operator implementation.
"""

from .parallel_ops_register import register_distributed_op

class DistributedOp:
    """
    Base class for distributed operator implementations.

    This class provides default implementations for distributed operators.
    Subclasses should override methods as needed for specific operators.

    Args:
        op_name (str): Name of the operator to register.
    """
    def __init__(self, op_name):
        self.op_name = op_name
        register_distributed_op(op_name, self)

    # pylint: disable=W0613
    def infer_layout(self, layouts, extra_args):
        """
        Infer output layouts based on input layouts.

        Default implementation returns the first input layout for element-wise operations.

        Args:
            layouts (tuple): Layouts of input tensor.
            extra_args (dict): Additional arguments (dim, keepdim).

        Returns:
            tuple: Layouts for output tensors.
        """
        if layouts:
            return (layouts[0],)
        return None

    def get_expand_impl(self, func, output_layout, layouts, extra_args):
        """
        Get expand implementation for the operator
        """
        return None
