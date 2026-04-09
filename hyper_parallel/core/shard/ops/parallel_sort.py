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
Distributed implementation for Sort operator.
"""

from .parallel_ops import DistributedOp


def _normalize_sort_args(x, dim=-1, descending=False, stable=None):
    return (x,), {'dim': dim, 'descending': descending, 'stable': stable}


class SortDistributedOp(DistributedOp):
    """Distributed implementation for Sort operator."""

    def preprocess(self, args, kwargs):
        args, kwargs = _normalize_sort_args(*args, **kwargs)
        input_tensor = args[0]
        dim = kwargs['dim']
        descending = kwargs['descending']
        stable = kwargs['stable']
        local_args = (input_tensor.to_local(),)
        local_kwargs = {'dim': dim, 'descending': descending, 'stable': stable}
        cache_values = [input_tensor.layout, dim]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values):
        layout = cache_values[0]
        dim = cache_values[1]

        if not isinstance(dim, int):
            raise TypeError(f"For 'sort', dimension must be int, but got {type(dim)}")

        # Get tensor map to check sharding status
        in_tensor_map = layout.tensor_map
        ndim = len(in_tensor_map)

        # Handle negative dimension index
        if dim < -ndim or dim >= ndim:
            raise ValueError(f"Dimension out of range (expected to be in range of [{-ndim}, {ndim-1}], but got {dim})")

        if dim < 0:
            dim += ndim

        # Check if the sorting dimension is sharded
        # In tensor_map, -1 means Replicate (not sharded). Any other value implies sharding.
        mapping = in_tensor_map[dim]
        is_sharded = False

        if isinstance(mapping, (list, tuple)):
            # If mapped to multiple mesh axes, check if any is not -1
            if any(m != -1 for m in mapping):
                is_sharded = True
        elif mapping != -1:
            is_sharded = True

        if is_sharded:
            raise ValueError(
                f"For 'sort', sorting along a sharded dimension (dim {dim} mapped to {mapping}) is not supported. "
                f"Please redistribute the tensor to Replicate status on this dimension before sorting."
            )

        return ((layout, layout), None)
