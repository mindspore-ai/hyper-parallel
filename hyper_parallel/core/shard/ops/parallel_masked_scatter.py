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
Distributed implementation for MaskedScatter operator.
"""

from .parallel_ops import DistributedOp


class MaskedScatterDistributedOp(DistributedOp):
    """Distributed implementation for torch.Tensor.masked_scatter."""

    def infer_layout(self, layouts, extra_args=None):
        """
        Infer output layout for torch.Tensor.masked_scatter.

        PyTorch semantics:
            masked_scatter_(mask, source)
            Copies elements from source into self tensor at positions where the mask is True.
            Elements from source are taken in order.

        Distributed restrictions:
            Because `masked_scatter` consumes elements from `source` sequentially based on
            the flattened index of `True` values in `mask`, sharding the input or mask
            would require a global prefix sum (scan) to determine the correct offset
            in `source` for each rank. Without this communication overhead, correct
            behavior cannot be guaranteed on sharded tensors.

            Therefore, this implementation enforces that all inputs (input, mask, source)
            must be fully Replicated (Unsharded).

        Args:
            layouts (tuple): Layouts of inputs. Expected:
                layouts[0] (Layout): Input tensor layout.
                layouts[1] (Layout): Mask tensor layout.
                layouts[2] (Layout): Source tensor layout.
            extra_args (dict): Additional arguments.

        Returns:
            Layout: Output tensor layout (same as input).

        Raises:
            ValueError: If any input tensor is sharded.
        """
        # Check partial status via base class
        if not self._allow_partial_inputs:
            self._check_partial_inputs(layouts)

        # Check strict replication for all involved distributed tensors
        for i, layout in enumerate(layouts):
            if layout is None:
                continue

            # Check tensor_map for sharding
            # -1 indicates un-sharded (Replicated).
            # Any integer >= 0 or tuple (for interleaved) indicates sharding.
            for dim_map in layout.tensor_map:
                if dim_map != -1:
                    raise ValueError(
                        f"Operation {self.op_name}: Input {i} (Layout: {layout}) is sharded. "
                        f"masked_scatter currently only supports fully Replicated (Unsharded) tensors "
                        f"due to sequential dependency on source elements."
                    )

        # Output layout follows input layout (which we verified is Replicated/None)
        # Note: Must return a single Layout object, not a tuple, because the op returns a single Tensor.
        return layouts[0]
