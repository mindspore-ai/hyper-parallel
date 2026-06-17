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
Distributed implementation for RepeatInterleave operator.
"""
import copy
from typing import Tuple

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp


def _normalize_repeat_interleave_args(x, repeats, dim=None, *, output_size=None):
    """Normalize positional and keyword arguments into a canonical form.

    Args:
        x: Input tensor.
        repeats: Number of repetitions (int or Tensor).
        dim: Dimension along which to repeat values. None means flatten first.
        output_size: Total output size (keyword-only in torch).

    Returns:
        tuple: (positional_args_tuple, kwargs_dict)
    """
    kwargs = {}
    if output_size is not None:
        kwargs['output_size'] = output_size
    return (x, repeats, dim), kwargs


class RepeatInterleaveDistributedOp(DistributedOp):
    """Distributed implementation for torch.repeat_interleave.

    Sharding constraints:
      - When dim is specified: the repeat dimension must be replicated.
      - When dim is None (flatten mode): only the first dimension may be sharded.

    Output layout:
      - When dim is specified: same as input layout.
      - When dim is None: 1-D layout preserving sharding on dim 0.
    """

    @staticmethod
    def _validate_input_layouts(input_layout, dim, op_name: str) -> None:
        """Validate sharding constraints for repeat_interleave.

        Args:
            input_layout: Layout of the input tensor.
            dim: The repeat dimension, or None for flatten mode.
            op_name: Operator name used in error messages.

        Raises:
            ValueError: If dim is out of range or the repeat dimension is sharded.
        """
        in_tensor_map = input_layout.alias_tensor_map
        ndim = len(in_tensor_map)

        if dim is not None:
            actual_dim = dim if dim >= 0 else ndim + dim
            if not 0 <= actual_dim < ndim:
                raise ValueError(
                    f"For {op_name}, dimension should be in [0, {ndim}), but got {dim}."
                )
            mapping = in_tensor_map[actual_dim]
            if isinstance(mapping, (list, tuple)):
                is_sharded = any(axis != "None" for axis in mapping)
            else:
                is_sharded = mapping != "None"
            if is_sharded:
                raise ValueError(
                    f"For {op_name}, the repeat dimension should be replicated, "
                    f"but got dim={dim} mapped to {mapping}."
                )

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """Preprocess arguments for RepeatInterleave operator.

        Extracts local tensors from DTensor inputs and builds cache_values
        containing only the information needed for layout inference.

        Args:
            args: Positional arguments (input, repeats, dim).
            kwargs: Keyword arguments (output_size).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_repeat_interleave_args(*args, **kwargs)
        input_tensor = args[0]
        repeats = args[1]
        dim = args[2]
        output_size = kwargs.get('output_size', None)

        local_input = input_tensor.to_local()

        # repeats can be int or Tensor; handle DTensor defensively
        if hasattr(repeats, 'to_local'):
            local_repeats = repeats.to_local()
        else:
            local_repeats = repeats

        local_args = (local_input, local_repeats, dim)
        local_kwargs = {}
        if output_size is not None:
            local_kwargs['output_size'] = output_size

        cache_values = [input_tensor.layout, dim]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """Infer output layout for RepeatInterleave operator.

        Rules:
            1. Input must not have Partial status.
            2. When dim is specified: the repeat dimension must be replicated.
               Output layout is a deep copy of the input layout.
            3. When dim is None (flatten mode): only dim 0 may be sharded.
               Output is a 1-D layout preserving the mesh axis on dim 0.

        Args:
            cache_values: [input_layout, dim] where dim is the repeat dimension
                or None for flatten mode.

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If input has Partial status, dim is out of range,
                the repeat dimension is sharded, or flatten is attempted on
                a tensor sharded on a non-first dimension.
        """
        input_layout = cache_values[0]
        dim = cache_values[1]

        self._check_partial_inputs([input_layout])
        self._validate_input_layouts(input_layout, dim, self.op_name)

        in_tensor_map = input_layout.alias_tensor_map

        if dim is None:
            # Flatten mode: output is 1-D.
            sharded_dims = []
            for i, shard in enumerate(in_tensor_map):
                if isinstance(shard, (list, tuple)):
                    is_sharded = any(axis != "None" for axis in shard)
                else:
                    is_sharded = shard != "None"
                if is_sharded:
                    sharded_dims.append(i)
            if not sharded_dims:
                output_tensor_map = ("None",)
            elif sharded_dims == [0]:
                output_tensor_map = (in_tensor_map[0],)
            else:
                raise ValueError(
                    f"For {self.op_name}, sharded dims in flatten mode should be [] or [0], "
                    f"but got {sharded_dims}."
                )

            output_layout = Layout(
                mesh_shape=input_layout.mesh_shape,
                alias_name=input_layout.alias_name,
                rank_list=input_layout.rank_list
            )
            return (output_layout(*output_tensor_map),), None

        # dim specified: output layout = deep copy of input layout
        return (copy.deepcopy(input_layout),), None
