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
Distributed implementation for Flatten operator.
"""
from typing import Tuple

from hyper_parallel.core.shard.ops.parallel_reshape import ReshapeDistributedOp


def _normalize_flatten_args(x, start_dim=0, end_dim=-1):
    return (x, start_dim, end_dim), {}


class FlattenDistributedOp(ReshapeDistributedOp):
    """Distributed implementation for torch.flatten."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for Flatten operator.

        Args:
            args (tuple): Input tensor followed by optional start_dim and end_dim.
            kwargs (dict): Optional keyword arguments.

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, _ = _normalize_flatten_args(*args, **kwargs)
        input_tensor, start_dim, end_dim = args[0], args[1], args[2]
        local_args = (input_tensor.to_local(), start_dim, end_dim)
        local_kwargs = {}
        cache_values = [input_tensor.layout, start_dim, end_dim, tuple(input_tensor.shape)]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for Flatten operator.

        Rules:
            1. Partial input is allowed and preserved by reshape layout inference.
            2. input_shape must be provided and match the input rank.
            3. start_dim and end_dim must be integers within the valid input rank.
            4. If start_dim >= end_dim after normalization, the output layout is the
               input layout.
            5. Otherwise, dimensions from start_dim through end_dim are merged using
               reshape-compatible sharding rules.

        Args:
            cache_values (list): [input_layout, start_dim, end_dim, input_shape].

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If cache_values are invalid, dimensions are out of range,
                or the flatten would change sharded slices incompatibly.
        """
        input_layout, start_dim, end_dim, input_shape = (
            cache_values[0], cache_values[1], cache_values[2], cache_values[3]
        )
        if input_layout is None:
            raise ValueError(
                f"For {self.op_name}, flatten requires a valid input tensor layout."
            )
        if not isinstance(input_shape, (list, tuple)):
            raise ValueError(
                f"For {self.op_name}, input_shape should be list or tuple, "
                f"but got {type(input_shape)}."
            )
        if len(input_shape) != len(input_layout.tensor_map):
            raise ValueError(
                f"For {self.op_name}, input shape rank should match layout rank, "
                f"but got {len(input_shape)} and {len(input_layout.tensor_map)}."
            )
        if not isinstance(start_dim, int) or not isinstance(end_dim, int):
            raise ValueError(
                f"For {self.op_name}, start_dim and end_dim should be int, "
                f"but got {type(start_dim)} and {type(end_dim)}."
            )

        ndim = len(input_shape)

        if ndim == 0:
            out_layout = input_layout.__class__.from_device_mesh(input_layout.mesh)
            out_layout.set_placements(input_layout.placements)
            out_layout.placement_to_tensor_map(1)
            return ((out_layout,), None)

        if start_dim < 0:
            start_dim += ndim
        if end_dim < 0:
            end_dim += ndim

        if start_dim < 0 or start_dim >= ndim or end_dim < 0 or end_dim >= ndim:
            raise ValueError(
                f"For {self.op_name}, dimension out of range "
                f"(start_dim={start_dim}, end_dim={end_dim}, ndim={ndim})."
            )

        if start_dim >= end_dim:
            return ((input_layout,), None)

        flattened_size = 1
        for i in range(start_dim, end_dim + 1):
            flattened_size *= input_shape[i]
        dst_shape = list(input_shape[:start_dim]) + [flattened_size] + list(input_shape[end_dim + 1:])

        out_layout, _ = self._infer_reshape_layout(input_layout, dst_shape, input_shape)
        return ((out_layout,), None)
