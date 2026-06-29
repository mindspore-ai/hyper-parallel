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
Distributed implementation for Expand operator.
"""

from typing import Tuple

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp


def _normalize_expand_args(input_tensor, *sizes):
    return (input_tensor, *sizes), {}


def _normalize_expand_as_args(input_tensor, target_tensor):
    return (input_tensor, target_tensor), {}


class ExpandDistributedOp(DistributedOp):
    """Distributed implementation for torch.Tensor.expand."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for Expand operator.

        Args:
            args (tuple): Input arguments (input_tensor, *sizes).
            kwargs (dict): Keyword arguments (none for expand).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_expand_args(*args, **kwargs)
        input_tensor = args[0]
        sizes = tuple(args[1:])
        local_args = (input_tensor.to_local(), *sizes)
        cache_values = [input_tensor.layout, input_tensor.shape, sizes]
        return local_args, {}, cache_values

    @staticmethod
    def _validate_input_layouts(
        cache_values: list,
        op_name: str,
    ) -> Tuple[Layout, tuple, tuple, int]:
        """Validate all inputs for expand layout inference.

        Performs type checks, shape validation, sizes validation,
        and dimension compatibility checks.

        Rules:
            1. input_shape and sizes must be tuples of positive ints.
            2. Cannot reduce dimensions (output_ndim < input_ndim).
            3. -1 cannot be used for new (prepended) dimensions.

        Args:
            cache_values: [input_layout, input_shape, sizes]
            op_name: Operator name for error messages.

        Returns:
            tuple: (input_layout, input_shape, sizes, num_new_dims)

        Raises:
            ValueError: If any validation rule is violated.
        """
        if not cache_values:
            raise ValueError(
                f"For {op_name}, cache_values should contain input layout, "
                f"but got empty cache_values."
            )
        input_layout = cache_values[0]

        if not isinstance(input_layout, Layout):
            raise ValueError(
                f"For {op_name}, input layout should be a Layout, "
                f"but got {type(input_layout)}."
            )

        input_shape = cache_values[1] if len(cache_values) > 1 else None
        if not isinstance(input_shape, tuple):
            raise ValueError(
                f"For {op_name}, input_shape should be a tuple, "
                f"but got {type(input_shape)}."
            )

        sizes = cache_values[2] if len(cache_values) > 2 else None
        if sizes is None or len(sizes) < 1:
            raise ValueError(
                f"For {op_name}, sizes should be a non-empty tuple of ints, "
                f"but got {sizes}."
            )
        for i, sz in enumerate(sizes):
            if not isinstance(sz, int):
                raise ValueError(
                    f"For {op_name}, elements in sizes should be int, "
                    f"but got {type(sz)} at position {i}."
                )

        in_alias_map = input_layout.alias_tensor_map
        input_ndim = len(in_alias_map)
        output_ndim = len(sizes)
        num_new_dims = output_ndim - input_ndim

        if len(input_shape) != input_ndim:
            raise ValueError(
                f"For {op_name}, input_shape length ({len(input_shape)}) "
                f"must match input_ndim ({input_ndim})."
            )

        if num_new_dims < 0:
            raise ValueError(
                f"For {op_name}, cannot reduce dimensions with expand, "
                f"input has {input_ndim} dims but requested {output_ndim} dims."
            )

        # New dimensions cannot use -1
        for i in range(num_new_dims):
            if sizes[i] == -1:
                raise ValueError(
                    f"For {op_name}, cannot use -1 for new dimension at position {i}, "
                    f"sizes should be positive integers."
                )

        return input_layout, input_shape, sizes, num_new_dims

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for torch.Tensor.expand.

        Rules:
            1. Input must not have Partial status.
            2. input_shape and sizes must be tuples of positive ints.
            3. Cannot reduce dimensions (output_ndim < input_ndim).
            4. -1 cannot be used for new (prepended) dimensions.
            5. Same-size dimension: preserve original sharding.
            6. True broadcast (1 → N): the input dimension must be replicated.
            7. New dimensions are replicated.

        Args:
            cache_values (list): [input_layout, input_shape, sizes]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If any rule above is violated.
        """
        if not self._allow_partial_inputs and cache_values:
            self._check_partial_inputs([cache_values[0]])

        input_layout, input_shape, sizes, num_new_dims = self._validate_input_layouts(
            cache_values, self.op_name
        )

        in_alias_map = input_layout.alias_tensor_map
        input_ndim = len(in_alias_map)

        output_map = []

        # New dimensions: always replicated
        for _ in range(num_new_dims):
            output_map.append("None")

        # Process existing dimensions
        for i in range(input_ndim):
            output_dim_idx = num_new_dims + i
            requested_size = sizes[output_dim_idx]
            input_size = input_shape[i]

            if requested_size in (-1, input_size):
                # Dimension unchanged — preserve original sharding
                output_map.append(in_alias_map[i])
            elif input_size == 1 and requested_size > 1:
                # True broadcast: 1 → N — must be replicated
                if in_alias_map[i] != "None":
                    raise ValueError(
                        f"For {self.op_name}, cannot expand dimension {i} "
                        f"which is sharded "
                        f"(size {input_size} → {requested_size}), "
                        f"got mapping {in_alias_map[i]}."
                    )
                output_map.append("None")
            else:
                raise ValueError(
                    f"For {self.op_name}, cannot expand dimension {i} "
                    f"from size {input_size} to {requested_size}."
                )

        output_layout = Layout(
            mesh_shape=input_layout.mesh_shape,
            alias_name=input_layout.alias_name,
            rank_list=input_layout.rank_list
        )
        output_layout = output_layout(*output_map)
        return ((output_layout,), None)


class ExpandAsDistributedOp(DistributedOp):
    """Distributed implementation for torch.Tensor.expand_as."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for ExpandAs operator.

        Args:
            args (tuple): Input arguments (input_tensor, target_tensor).
            kwargs (dict): Keyword arguments (none for expand_as).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_expand_as_args(*args, **kwargs)
        input_tensor = args[0]
        target_tensor = args[1]
        local_args = (input_tensor.to_local(), target_tensor.to_local())
        cache_values = [input_tensor.layout, input_tensor.shape, target_tensor.shape]
        return local_args, {}, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for expand_as.

        Rules:
            1. Input must not have Partial status.
            2. target_shape must have at least as many dims as input_global_shape.
            3. Matching-size dimensions preserve input sharding.
            4. Singleton (size 1) dimensions being expanded to >1 must be unsharded in input.
            5. Non-singleton dimensions must match exactly.

        Args:
            cache_values (list): [input_layout, input_global_shape, target_shape]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If any rule above is violated.
        """
        if not cache_values:
            raise ValueError(
                f"For {self.op_name}, cache_values should contain input layout, "
                f"but got empty cache_values."
            )
        input_layout = cache_values[0]
        if not self._allow_partial_inputs:
            self._check_partial_inputs([input_layout])

        in_alias_map = input_layout.alias_tensor_map
        input_ndim = len(in_alias_map)

        input_global_shape = cache_values[1]
        target_shape = cache_values[2]

        if not isinstance(target_shape, tuple):
            raise ValueError(
                f"For {self.op_name}, target_shape should be tuple, "
                f"but got {type(target_shape)}."
            )
        if not isinstance(input_global_shape, tuple):
            raise ValueError(
                f"For {self.op_name}, input_global_shape should be tuple, "
                f"but got {type(input_global_shape)}."
            )

        target_ndim = len(target_shape)

        if target_ndim < input_ndim:
            raise ValueError(
                f"For {self.op_name}, target shape {target_shape} (ndim={target_ndim}) "
                f"cannot be smaller than input shape {input_global_shape} (ndim={input_ndim})."
            )

        # Align dimensions: right-align input to target shape
        num_leading_implicit = target_ndim - input_ndim
        aligned_input_shape = (1,) * num_leading_implicit + input_global_shape
        aligned_tensor_map = ("None",) * num_leading_implicit + in_alias_map

        # Validate expansion rules and build output tensor_map
        output_tensor_map = []
        for i, (in_size, tgt_size, shard_spec) in enumerate(
            zip(aligned_input_shape, target_shape, aligned_tensor_map)
        ):
            if in_size == tgt_size:
                # Dimension unchanged - preserve sharding pattern
                output_tensor_map.append(shard_spec)
            elif in_size == 1 and tgt_size > 1:
                # Dimension is expanded (broadcast) - must be unsharded
                if shard_spec != "None":
                    raise ValueError(
                        f"For {self.op_name}, cannot expand sharded dimension {i} "
                        f"which is going to broadcast (global size 1 -> {tgt_size}), "
                        f"got mapping {shard_spec}."
                    )
                output_tensor_map.append("None")
            else:
                raise ValueError(
                    f"For {self.op_name}, cannot expand dimension {i} from size "
                    f"{in_size} to {tgt_size}."
                )

        output_layout = Layout(
            mesh_shape=input_layout.mesh_shape,
            alias_name=input_layout.alias_name,
            rank_list=input_layout.rank_list
        )
        output_layout = output_layout(*output_tensor_map)
        return ((output_layout,), None)
    