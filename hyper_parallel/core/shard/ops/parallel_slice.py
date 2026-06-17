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
Distributed implementation for Slice operator.
"""
# pylint: disable=E0402
from typing import Callable, Optional, Tuple

from .parallel_ops import DistributedOp


def _normalize_slice_args(x, begin, end):
    return (x, begin, end), {}


class SliceDistributedOp(DistributedOp):
    """Distributed implementation for Slice operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for Slice operator.

        Args:
            args (tuple): Input arguments containing x, begin and end.
            kwargs (dict): Keyword arguments.

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, _ = _normalize_slice_args(*args, **kwargs)
        input_tensor, begin, end = args
        local_args = (input_tensor.to_local(), begin, end)
        local_kwargs = {}
        cache_values = [input_tensor.layout, begin, end, input_tensor.shape]
        return local_args, local_kwargs, cache_values

    def _is_shard_dim(self, layout):
        """return the shard num in each dim"""
        shard_dim = []
        for axis_name in layout.alias_tensor_map:
            if axis_name == "None":
                shard_dim.append(1)
                continue
            if isinstance(axis_name, (tuple, list)):
                shard_num = 1
                for axis in axis_name:
                    if axis != "None":
                        shard_num *= layout.mesh.get_device_num_along_axis(axis)
                shard_dim.append(shard_num)
                continue
            shard_dim.append(layout.mesh.get_device_num_along_axis(axis_name))
        return shard_dim

    def _check_layout(self, layout, begin, end, shape):
        """check whether layout is valid"""
        if len(layout) != 1:
            raise ValueError(f"Layout must be a tuple of length 1, but got {len(layout)}")
        layout = layout[0]
        shard_dim = self._is_shard_dim(layout)
        for i, _ in enumerate(begin):
            if (shard_dim[i] != 1 and end[i] - begin[i] != shape[i]) and shape[i] != -1:
                raise ValueError(
                    f"Slice: When a dimension({i}) is not fully fetched, the dimension can not be split now, "
                    f"the begin is {begin}, the end is {end}, the shape is {shape}, layout is {layout.to_dict()}")
        return shard_dim

    def infer_layout(self, cache_values: list) -> Tuple[tuple, tuple]:
        """
        Infer output layout for Slice operator.

        Rules:
            1. Input must not have Partial status.
            2. begin, end and global_shape must have the same rank as input layout.
            3. Any sharded dimension must be fully fetched.
            4. Output layout is identical to the input layout.

        Args:
            cache_values (list): [input_layout, begin, end, global_shape].

        Returns:
            tuple: ((output_layout,), (new_begin, new_end)).

        Raises:
            ValueError: If input has Partial status, arguments rank mismatch, or a sharded
                dimension is not fully fetched.
        """
        layout, begin, end, global_shape = cache_values
        self._check_partial_inputs([layout])

        if len(begin) != len(end) or len(begin) != len(global_shape):
            raise ValueError(
                f"For {self.op_name}, begin, end and global_shape must have the same length, "
                f"but got begin: {len(begin)}, end: {len(end)}, global_shape: {len(global_shape)}"
            )
        if len(begin) != len(layout.alias_tensor_map):
            raise ValueError(
                f"For {self.op_name}, slice arguments rank must match input layout rank, "
                f"but got args rank: {len(begin)} and layout rank: {len(layout.alias_tensor_map)}"
            )

        shard_dim = self._check_layout((layout,), begin, end, global_shape)
        new_begin = tuple(begin[i] // shard_dim[i] for i in range(len(begin)))
        new_end = tuple(end[i] // shard_dim[i] for i in range(len(end)))
        return ((layout,), (new_begin, new_end))

    def get_expand_impl(self, func: Optional[Callable], infer_result: tuple,
                        cache_values: list) -> Optional[Callable]:
        """
        Return a custom Slice implementation when local begin/end need adjustment.

        Args:
            func: Original operator callable.
            infer_result (tuple): ((output_layout,), (new_begin, new_end)) from infer_layout.
            cache_values (list): [input_layout, begin, end, global_shape].

        Returns:
            callable | None: expand_impl closure when local slice bounds differ, else None.
        """
        if func is None:
            return None

        begin = cache_values[1]
        end = cache_values[2]
        new_begin, new_end = infer_result[1]
        if begin == new_begin and end == new_end:
            return None

        def expand_impl(input_tensor: object, *_unused_args: object) -> object:
            """Call Slice with local slice bounds."""
            return func(input_tensor, new_begin, new_end)

        return expand_impl
