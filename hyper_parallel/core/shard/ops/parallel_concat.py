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
Distributed implementation for Concat operator.
"""

from typing import Tuple

from .parallel_ops import DistributedOp


# pylint: disable=unused-argument
def _normalize_concat_args(tensors, dim=0, **kwargs):
    """
    Normalize arguments for Concat operator.
    """
    return (tensors, dim), {}


class ConcatDistributedOp(DistributedOp):
    """Distributed implementation for Concat."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for Concat operator.

        Args:
            args (tuple): Input arguments, first element is the input tensor sequence.
            kwargs (dict): Keyword arguments, may contain dim.

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, _ = _normalize_concat_args(*args, **kwargs)
        tensors = args[0]
        dim = args[1]

        local_tensors = tuple(t.to_local() if hasattr(t, "to_local") else t for t in tensors)
        layouts = [getattr(t, "layout", None) for t in tensors]

        local_args = (local_tensors, dim)
        local_kwargs = {}
        cache_values = layouts + [dim]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:  # pylint: disable=W0221
        """
        Infer output layouts for Concat operator.

        Rules:
            1. Inputs must not have Partial status.
            2. At least one input must be a DTensor.
            3. All input DTensors must have the same layout.
            4. dim must be an integer within the valid range [-ndim, ndim-1].
            5. The concatenation dimension must not be sharded.
            6. Output layout is identical to the input layout.

        Args:
            cache_values (list): [input_layout, ..., dim] where non-DTensor inputs
                use None as their layout sentinel.

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If inputs are invalid, layouts mismatch, dim is out of range,
                or the concatenation dimension is sharded.
        """
        layouts = cache_values[:-1]
        dim = cache_values[-1]
        valid_layouts = [layout for layout in layouts if layout is not None]

        if not valid_layouts:
            raise ValueError(f"For {self.op_name}, cat requires at least one input DTensor.")

        self._check_partial_inputs(valid_layouts)

        base_layout = valid_layouts[0]

        for layout in valid_layouts:
            if layout != base_layout:
                raise ValueError(
                    f"For {self.op_name}, All input tensors must have the same layout. "
                    f"Expected layout: {base_layout}, Mismatched layout: {layout}"
                )

        if not isinstance(dim, int):
            raise ValueError(
                f"For {self.op_name}, dimension should be int, but got {type(dim)}"
            )

        ndim = len(base_layout.alias_tensor_map)
        if dim < -ndim or dim >= ndim:
            raise ValueError(
                f"For {self.op_name}, dimension out of range "
                f"(expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
            )

        actual_dim = dim if dim >= 0 else dim + ndim

        mapping = base_layout.alias_tensor_map[actual_dim]
        if mapping != "None":
            raise ValueError(
                f"For {self.op_name}, Concatenation along a sharded dimension "
                f"(dim={dim}, normalized_dim={actual_dim}) is not supported."
            )

        return ((base_layout,), None)
