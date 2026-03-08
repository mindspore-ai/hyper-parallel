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
Distributed implementation for Nonzero operator.
"""

from hyper_parallel.core.layout import Layout
from .parallel_ops import DistributedOp

class NonzeroDistributedOp(DistributedOp):
    """Distributed implementation for torch.nonzero."""

    def infer_layout(self, layouts, extra_args=None):
        """
        Infer output layout for torch.nonzero.

        PyTorch semantics:
          - Returns a tensor containing the indices of all non-zero elements.
          - If as_tuple=False (default): Returns a 2-D tensor of shape (z, n),
            where z is the number of non-zero elements and n is the input dimension.
          - If as_tuple=True: Returns a tuple of 1-D tensors, one for each dimension.

        Distributed semantics:
          - Because the output shape depends dynamically on the input data,
            the input tensor MUST be fully replicated (unsharded) across the mesh.
            Otherwise, each rank would produce a different local shape.
          - The output layout will also be fully replicated.

        Args:
            layouts (tuple): Layouts of inputs. Expected:
                layouts[0] (Layout): Input tensor layout (required).
            extra_args (list): Contains additional kwargs/args like 'as_tuple'.

        Returns:
            Layout or tuple[Layout]: Replicated output layout(s) matching PyTorch's
                                     as_tuple return signature.
        """
        # =====================================================================
        # FIX: Explicitly enforce the base class's partial status guardrail
        # =====================================================================
        if not self._allow_partial_inputs:
            self._check_partial_inputs(layouts)

        if not layouts or layouts[0] is None:
            raise ValueError(
                f"Operation {self.op_name}: nonzero requires a valid input tensor layout."
            )

        input_layout = layouts[0]
        in_tensor_map = input_layout.tensor_map
        input_ndim = len(in_tensor_map)

        # Rule 1: Input must be fully replicated due to data-dependent dynamic shapes
        for dim_sharding in in_tensor_map:
            if dim_sharding != -1:
                raise ValueError(
                    f"Operation {self.op_name}: input tensor must be fully replicated "
                    f"(unsharded). nonzero produces dynamic shapes that depend on data values, "
                    f"which causes shape mismatches across ranks if the tensor is sharded."
                )

        # Rule 2: Parse 'as_tuple' from extra_args (defaults to False)
        as_tuple = False
        if extra_args:
            for arg in extra_args:
                if isinstance(arg, bool):
                    as_tuple = arg
                    break

        mesh_shape = input_layout.mesh_shape
        alias_name = input_layout.alias_name
        rank_list = input_layout.rank_list

        def _create_replicated_layout(ndim):
            """Helper to create a fully replicated layout for a given dimension."""
            layout = Layout(
                mesh_shape=mesh_shape,
                alias_name=alias_name,
                rank_list=rank_list
            )
            # Replicated layout maps all dimensions to "None"
            alias_map = tuple("None" for _ in range(ndim))
            return layout(*alias_map)

        # Rule 3: Construct the return layout based on as_tuple flag
        if as_tuple:
            # Returns a tuple of 1D tensors, one for each dimension in the input
            out_layout = _create_replicated_layout(1)
            return tuple(out_layout for _ in range(input_ndim))

        # Default: Returns a single 2D tensor of shape (z, n)
        return _create_replicated_layout(2)
