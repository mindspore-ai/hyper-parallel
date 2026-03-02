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
Distributed implementation for Scatter operator.
"""

from .parallel_ops import DistributedOp


class ScatterDistributedOp(DistributedOp):
    """Distributed implementation for torch.scatter."""

    def infer_layout(self, layouts, extra_args=None):
        """
        Infer output layout for scatter.

        Args:
            layouts (tuple): Layouts of inputs. Expected order for scatter(input, dim, index, src):
                layouts[0]: input (DTensor)
                layouts[1]: dim (None, as it's int)
                layouts[2]: index (DTensor)
                layouts[3]: src (DTensor or None if scalar)
            extra_args (list): Contains non-tensor arguments.
                extra_args[0]: dim (int)
                extra_args[1]: src (if src is scalar)

        Returns:
            Layout: Output has same layout as input.
        """
        # 1. Check partial status
        if not self._allow_partial_inputs:
            self._check_partial_inputs(layouts)

        if not layouts or layouts[0] is None:
            raise ValueError(
                f"Operation {self.op_name}: scatter requires a valid input tensor layout."
            )

        input_layout = layouts[0]
        input_map = input_layout.tensor_map
        ndim = len(input_map)

        # 2. Extract and Validate 'dim'
        if not extra_args or len(extra_args) < 1:
            raise ValueError(
                f"Operation {self.op_name}: scatter requires 'dim' parameter in extra_args."
            )

        dim = extra_args[0]
        if not isinstance(dim, int):
            raise ValueError(f"Operation {self.op_name}: 'dim' must be an integer.")

        # Normalize dim
        if dim < 0:
            dim += ndim

        if dim < 0 or dim >= ndim:
            raise ValueError(
                f"Operation {self.op_name}: dim {dim} is out of bounds for tensor with {ndim} dimensions."
            )

        # 3. Rule: Scatter dimension cannot be sharded
        if input_map[dim] != -1:
            raise ValueError(
                f"Operation {self.op_name}: Scatter along sharded dimension {dim} is not supported. "
                "The target dimension must be Replicated (unsharded)."
            )

        # 4. Rule: Index layout must match Input layout
        if len(layouts) > 2:
            index_layout = layouts[2]
            if index_layout is not None:
                if index_layout.tensor_map != input_map:
                    raise ValueError(
                        f"Operation {self.op_name}: Index tensor layout {index_layout.tensor_map} "
                        f"must match input tensor layout {input_map}."
                    )

        # 5. Rule: Src layout must match Input layout (if src is a tensor)
        if len(layouts) > 3:
            src_layout = layouts[3]
            if src_layout is not None:
                if src_layout.tensor_map != input_map:
                    raise ValueError(
                        f"Operation {self.op_name}: Src tensor layout {src_layout.tensor_map} "
                        f"must match input tensor layout {input_map}."
                    )

        # Output layout is the same as input layout (scatter modifies input or returns shape of input)
        return input_layout
