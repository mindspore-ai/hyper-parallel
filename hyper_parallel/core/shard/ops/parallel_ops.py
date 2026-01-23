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
        self._allow_partial_inputs = False

    def _check_partial_inputs(self, layouts):
        """
        Check if any input layout has partial status and raise an error if not allowed.

        This method can be called by subclasses to enforce that partial inputs
        are not supported for a particular operator. Subclasses that support
        partial inputs should not call this method.

        Args:
            layouts (tuple): Layouts of input tensor.

        Raises:
            ValueError: If any input layout has partial status.
        """
        for i, layout in enumerate(layouts):
            if layout is not None and layout.is_partial():
                raise ValueError(
                    f"For {self.op_name}, input {i} with {layout} has Partial status which is not allowed. "
                    f"Should be without Partial status for this operation."
                )

    # pylint: disable=W0613
    def infer_layout(self, layouts, extra_args):
        """
        Infer output layouts based on input layouts.

        Default implementation returns the first input layout for element-wise operations.
        Subclasses can override this method to provide custom layout inference logic.

        Args:
            layouts (tuple): Layouts of input tensor.
            extra_args (dict): Additional arguments (dim, keepdim).

        Returns:
            tuple: Layouts for output tensors.
        """
        # Check partial inputs
        if not self._allow_partial_inputs:
            self._check_partial_inputs(layouts)

        if layouts:
            return (layouts[0],)
        return None

    def get_expand_impl(self, func, output_layout, layouts, extra_args):
        """
        Get expand implementation for the operator
        """
        return None
