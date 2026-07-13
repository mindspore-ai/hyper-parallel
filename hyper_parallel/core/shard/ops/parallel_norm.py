# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
Distributed implementation for Norm operators (RmsNorm, layer_norm).
"""

from typing import Tuple

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp


def _normalize_rmsnorm_args(x, gamma, epsilon=1e-6):
    """Normalize RmsNorm args to positional form.

    MindSpore Primitive RmsNorm receives (x, gamma, epsilon) as positional arguments.
    """
    return (x, gamma, epsilon), {}


def _normalize_layernorm_args(input_tensor, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Normalize layer_norm args to positional form.

    torch.nn.functional.layer_norm(input_tensor, normalized_shape, weight=None, bias=None, eps=1e-5)
    has no keyword-only parameters, so everything stays positional.
    """
    return (input_tensor, normalized_shape, weight, bias, eps), {}


class NormDistributedOp(DistributedOp):
    """Distributed implementation for RmsNorm operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for RmsNorm operator.

        Args:
            args (tuple): Positional arguments (x, gamma) where both are DTensors.
            kwargs (dict): Keyword arguments (none expected).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_rmsnorm_args(*args, **kwargs)
        x, gamma, epsilon = args
        local_args = (x.to_local(), gamma.to_local(), epsilon)
        local_kwargs = {}
        cache_values = [x.layout, gamma.layout]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layouts for RmsNorm operator.

        Rules:
            1. Inputs must not have Partial status.
            2. x and gamma must share the same mesh_shape.
            3. Dimensions being normalized (the last len(gamma_tensor_map) dims of x)
               must not be sharded.
            4. The sharding of the normalized dimensions of x must match gamma's sharding.
            5. Output layout keeps sharding on non-normalized dims and replicates
               on normalized dims.

        Args:
            cache_values (list): [x_layout, gamma_layout]

        Returns:
            tuple: ((x_layout, out_layout), None)

        Raises:
            ValueError: If any rule above is violated.
        """
        if len(cache_values) < 2:
            raise ValueError(
                f"For {self.op_name}, cache_values size {len(cache_values)} is less than 2."
            )
        x_layout = cache_values[0]
        gamma_layout = cache_values[1]
        # Check partial inputs
        if not self._allow_partial_inputs:
            self._check_partial_inputs([x_layout, gamma_layout])
        x_mesh_shape = x_layout.mesh_shape
        gamma_mesh_shape = gamma_layout.mesh_shape
        if x_mesh_shape != gamma_mesh_shape:
            raise ValueError(f"{self.op_name} inputs must have same mesh_shape")
        x_alias_map = x_layout.alias_tensor_map
        gamma_alias_map = gamma_layout.alias_tensor_map
        if len(gamma_alias_map) > len(x_alias_map):
            raise ValueError(
                f"For {self.op_name}, gamma ndim {len(gamma_alias_map)} cannot exceed "
                f"input ndim {len(x_alias_map)}."
            )
        begin_norm_axis = len(x_alias_map) - len(gamma_alias_map)
        for alias_entry in x_alias_map[begin_norm_axis:]:
            entries = alias_entry if isinstance(alias_entry, tuple) else (alias_entry,)
            for name in entries:
                if name == "None":
                    continue
                axis_idx = x_layout.alias_name.index(name)
                if x_mesh_shape[axis_idx] > 1:
                    raise ValueError(f"{self.op_name} is disabled to support the splitting after "
                                     f"begin_norm_axis {begin_norm_axis} for input 0.")
        if x_alias_map[begin_norm_axis:] != gamma_alias_map:
            raise ValueError(f"For {self.op_name}, input sharding from begin_norm_axis "
                             f"{begin_norm_axis}, {x_alias_map[begin_norm_axis:]}, should equal "
                             f"gamma sharding {gamma_alias_map}.")
        output_layout = Layout(
            mesh_shape=x_layout.mesh_shape,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list
        )
        output_map = x_alias_map[:begin_norm_axis] + ("None",) * len(gamma_alias_map)
        out_layout = output_layout(*output_map)
        return ((x_layout, out_layout), None)


class LayerNormDistributedOp(DistributedOp):
    """Distributed implementation for torch.nn.functional.layer_norm."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for layer_norm operator.

        Args:
            args (tuple): Positional arguments (input, normalized_shape, weight, bias, eps).
            kwargs (dict): Keyword arguments (none expected for this functional API).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_layernorm_args(*args, **kwargs)
        input_tensor, normalized_shape, weight, bias, eps = args

        # Normalize normalized_shape: int → (int,), list → tuple
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        elif isinstance(normalized_shape, list):
            normalized_shape = tuple(normalized_shape)

        local_args = [
            input_tensor.to_local(),
            normalized_shape,
            weight.to_local() if weight is not None and hasattr(weight, 'to_local') else weight,
            bias.to_local() if bias is not None and hasattr(bias, 'to_local') else bias,
            eps,
        ]
        local_kwargs = {}

        cache_values = [input_tensor.layout, normalized_shape]
        return tuple(local_args), local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for layer_norm operator.

        Rules:
            1. Input must not have Partial status.
            2. normalized_shape must be int, list, or tuple.
            3. normalized_shape dimensions must be ≤ input ndim.
            4. All dimensions in normalized_shape must be unsharded.
            5. Output layout is identical to input layout.

        Args:
            cache_values (list): [input_layout, normalized_shape]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If any rule above is violated.
        """
        input_layout = cache_values[0]
        if input_layout is None:
            raise ValueError(f"{self.op_name} requires a valid input tensor layout.")
        normalized_shape = cache_values[1]
        # Check partial inputs
        if not self._allow_partial_inputs:
            self._check_partial_inputs([input_layout])

        if normalized_shape is None:
            raise ValueError(f"{self.op_name} requires normalized_shape.")

        if not isinstance(normalized_shape, tuple):
            raise ValueError(f"normalized_shape must be int, list, or tuple, got {type(normalized_shape)}")

        in_alias_map = input_layout.alias_tensor_map
        input_ndim = len(in_alias_map)
        norm_ndim = len(normalized_shape)

        if norm_ndim > input_ndim:
            raise ValueError(
                f"normalized_shape {normalized_shape} (dims={norm_ndim}) is larger than input ndim={input_ndim}."
            )

        # The last `norm_ndim` dimensions are going to be normalized
        dims_to_normalize = list(range(input_ndim - norm_ndim, input_ndim))

        # All normalized dims must be unsharded
        for dim in dims_to_normalize:
            alias_entry = in_alias_map[dim]
            entries = alias_entry if isinstance(alias_entry, tuple) else (alias_entry,)
            for name in entries:
                if name == "None":
                    continue
                raise ValueError(
                    f"Operation {self.op_name}: Cannot perform sharding on normalized dimension {dim}, "
                    f"but found sharding assignment: {in_alias_map[dim]}"
                )

        output_layout = Layout(
            mesh_shape=input_layout.mesh_shape,
            alias_name=input_layout.alias_name,
            rank_list=input_layout.rank_list
        )
        output_layout = output_layout(*in_alias_map)
        return ((output_layout,), None)
