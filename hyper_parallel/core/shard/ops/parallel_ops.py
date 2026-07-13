# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

from typing import Optional

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
    def preprocess(self, args: tuple, kwargs: dict) -> Optional[tuple]:
        """
        Unified preprocessing: parameter parsing + to_local + cache_values construction.

        Subclasses override this to participate in the new dispatch flow.

        Args:
            args (tuple): Positional arguments passed to the operator call.
            kwargs (dict): Keyword arguments passed to the operator call.

        Returns:
            None: Fall back to legacy dispatch (default).
            tuple: (local_args, local_kwargs, cache_values)
                - local_args: Local tensor positional arguments (DTensors already to_local'd).
                - local_kwargs: Local tensor keyword arguments (DTensors already to_local'd).
                - cache_values: Values affecting layout inference (fixed order).
                    Contains Layout objects (with compact_str) and raw values (int, bool, tuple, etc.).
        """
        return None

    def infer_layout(self, cache_values: list) -> Optional[tuple]:
        """
        Infer output layouts based on cache_values built by preprocess.

        Default implementation extracts the first Layout from cache_values and
        returns it as the output layout (element-wise default). Subclasses should
        override this method to provide custom layout inference logic.

        Args:
            cache_values (list): Values built by preprocess that affect layout inference.
                Contains Layout objects and non-layout values (shapes, scalars, etc.).

        Returns:
            tuple: ((output_layouts,), None) or None if no layouts found.
        """
        if not self._allow_partial_inputs:
            self._check_partial_inputs(cache_values)

        if cache_values:
            return (cache_values[0],)
        return None

    # pylint: disable=W0613
    def get_expand_impl(
        self,
        func: Optional[callable],
        infer_result: tuple,
        cache_values: list,
    ) -> Optional[callable]:
        """
        Get expand implementation for the operator.

        Args:
            func (Optional[callable]): The underlying operator function.
            infer_result (tuple): Result returned by infer_layout (output_layouts, extra_info).
            cache_values (list): Values built by preprocess, forwarded from the dispatch layer.

        Returns:
            Optional[callable]: A closure that wraps the operator call with extra logic,
                or None if no expansion is needed.
        """
        return None

    @staticmethod
    def wrap_output(py_output, output_layouts):
        """Wrap local outputs into DTensors according to inferred layouts.

        Subclasses may override this when a specific operator needs custom
        packing semantics for certain output slots.
        """
        # pylint: disable=C0415
        from hyper_parallel.core.dtensor.dtensor import DTensor

        if isinstance(py_output, (tuple, list)):
            if len(py_output) != len(output_layouts):
                raise RuntimeError(
                    f"Output tuple size ({len(py_output)}) "
                    f"does not match layout tuple size ({len(output_layouts)})")
            # Inline the fast construction (equivalent to from_local_with_layout) to
            # avoid an extra Python call frame per output in this hot multi-output loop.
            return tuple(
                DTensor(item, layout.mesh, layout.placements, layout)
                for item, layout in zip(py_output, output_layouts)
            )

        if isinstance(output_layouts, (tuple, list)):
            if len(output_layouts) != 1:
                raise RuntimeError(
                    f"Scalar output expects a single layout, but got {len(output_layouts)} layouts"
                )
            output_layout = output_layouts[0]
        else:
            output_layout = output_layouts

        return DTensor.from_local_with_layout(py_output, output_layout)
