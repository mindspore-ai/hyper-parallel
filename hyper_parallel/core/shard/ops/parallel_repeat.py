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
Distributed implementation for Repeat operator.
"""

from typing import Tuple

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp


def _normalize_repeat_args(x, *sizes):
    return (x,) + sizes, {}


class RepeatDistributedOp(DistributedOp):
    """Distributed implementation for torch.Tensor.repeat."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for torch.Tensor.repeat.

        Args:
            args (tuple): Input tensor followed by repeat sizes.
            kwargs (dict): Keyword arguments.

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, _ = _normalize_repeat_args(*args, **kwargs)
        input_tensor = args[0]
        sizes = args[1:]

        local_args = (input_tensor.to_local(),) + sizes
        local_kwargs = {}
        cache_values = [input_tensor.layout, sizes]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for torch.Tensor.repeat.

        PyTorch semantics:
          - Repeats this tensor along the specified dimensions.
          - If the number of repeat dimensions is larger than the tensor dimensions,
            the tensor is implicitly unsqueezed at the front.
          - The number of repeat dimensions cannot be smaller than the tensor dimensions.
          - Dimensions being repeated (>1 or 0) MUST be unsharded.

        Rules:
            1. Input must not have Partial status.
            2. Repeat sizes must be provided.
            3. New prepended dimensions are unsharded.
            4. Existing dimensions with repeat size 1 preserve the input sharding.
            5. Existing dimensions with repeat size other than 1 must not be sharded.

        Args:
            cache_values (list): [input_layout, repeat_sizes].

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If input layout is missing, input has Partial status, repeat sizes
                are missing, or a sharded dimension is repeated.
        """
        if not cache_values or cache_values[0] is None:
            raise ValueError(
                f"For {self.op_name}, repeat requires a valid input tensor layout."
            )

        input_layout = cache_values[0]
        self._check_partial_inputs([input_layout])

        in_tensor_map = input_layout.tensor_map
        in_alias_map = input_layout.alias_tensor_map
        input_ndim = len(in_tensor_map)

        if len(cache_values) < 2 or not cache_values[1]:
            raise ValueError(
                f"For {self.op_name}, repeat requires repeat sizes in cache_values."
            )

        # Robustly handle sizes unpacking (e.g., if args are packed as a single tuple)
        repeat_sizes = cache_values[1]
        if len(repeat_sizes) == 1 and isinstance(repeat_sizes[0], (tuple, list)):
            flat_args = repeat_sizes[0]
        else:
            flat_args = repeat_sizes

        # Normalize repeat sizes to tuple of ints
        repeats = []
        for arg in flat_args:
            if not isinstance(arg, int):
                arg = int(arg)
            repeats.append(arg)
        repeats = tuple(repeats)
        output_ndim = len(repeats)

        num_new_dims = output_ndim - input_ndim
        output_map = []

        # Rule 1: New prepended dimensions are always unsharded
        for _ in range(num_new_dims):
            output_map.append("None")

        # Rule 2: Process existing dimensions
        for i in range(input_ndim):
            repeat_idx = num_new_dims + i
            repeat_times = repeats[repeat_idx]

            if repeat_times == 1:
                # If the dimension is not repeated, keep the original sharding
                output_map.append(in_alias_map[i])
            else:
                # If the dimension is repeated (or zeroed), it cannot be currently sharded
                mapping = in_alias_map[i]
                if mapping != "None":
                    raise ValueError(
                        f"For {self.op_name}, Cannot repeat dimension {i} which is sharded. "
                        f"Please redistribute (unshard) the tensor along this dimension first."
                    )
                # Repeated dimension remains unsharded in output
                output_map.append("None")

        # Construct output layout mapping
        mesh_shape = input_layout.mesh_shape
        alias_name = input_layout.alias_name
        rank_list = input_layout.rank_list

        # Instantiate new layout
        output_layout = Layout(
            mesh_shape=mesh_shape,
            alias_name=alias_name,
            rank_list=rank_list
        )
        output_layout = output_layout(*output_map)

        return ((output_layout,), None)
