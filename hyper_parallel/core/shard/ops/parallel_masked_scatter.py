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

from typing import Tuple

from .parallel_ops import DistributedOp


def _normalize_masked_scatter_args(input_tensor, mask, source):
    return (input_tensor, mask, source), {}


class MaskedScatterDistributedOp(DistributedOp):
    """Distributed implementation for torch.Tensor.masked_scatter."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for MaskedScatter operator.

        torch.masked_scatter(input, mask, source) takes three positional tensor inputs
        with no keyword-only parameters.

        Args:
            args (tuple): Positional arguments (input, mask, source).
            kwargs (dict): Keyword arguments (none expected).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_masked_scatter_args(*args, **kwargs)
        input_tensor, mask, source = args[0], args[1], args[2]
        local_args = (
            input_tensor.to_local(),
            mask.to_local(),
            source.to_local(),
        )
        local_kwargs = {}
        cache_values = [
            input_tensor.layout,
            mask.layout,
            source.layout,
        ]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
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

        Rules:
            1. Inputs must not have Partial status.
            2. All inputs (input, mask, source) must be fully Replicated — no dimension may be sharded.
            3. Output layout follows the input layout.

        Args:
            cache_values (list): [input_layout, mask_layout, source_layout]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If any input tensor is sharded.
        """
        input_layout = cache_values[0]
        mask_layout = cache_values[1]
        source_layout = cache_values[2]

        layouts = [input_layout, mask_layout, source_layout]
        if not self._allow_partial_inputs:
            self._check_partial_inputs(layouts)

        # Check strict replication for all involved distributed tensors
        for i, layout in enumerate(layouts):
            if layout is None:
                continue

            # Use alias_tensor_map for sharding checks — it handles StridedShard tuple mappings.
            # "None" means Replicated; anything else indicates sharding.
            for dim_alias in layout.alias_tensor_map:
                if dim_alias != "None":
                    raise ValueError(
                        f"For {self.op_name}, input {i} is sharded (Layout: {layout}). "
                        f"masked_scatter currently only supports fully Replicated (Unsharded) tensors "
                        f"due to sequential dependency on source elements."
                    )

        # Output layout follows input layout (which we verified is Replicated/None)
        return ((input_layout,), None)
