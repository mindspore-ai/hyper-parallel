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
Distributed implementation for torch index_add and index_add_ operators.
"""

from copy import deepcopy
from typing import Tuple

from .parallel_ops import DistributedOp


def _normalize_index_add_args(input_tensor, dim, index, source, *, alpha=1, out=None):
    """Normalize IndexAdd arguments to positional args and keyword args."""
    kwargs = {"alpha": alpha}
    if out is not None:
        kwargs["out"] = out
    return (input_tensor, dim, index, source), kwargs


class IndexAddDistributedOp(DistributedOp):
    """Distributed implementation for torch index_add and index_add_ operators."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for IndexAdd operator.

        Args:
            args (tuple): Input arguments (input_tensor, dim, index, source).
            kwargs (dict): Keyword arguments. ``alpha`` and ``out`` are supported.

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, kwargs = _normalize_index_add_args(*args, **kwargs)
        input_tensor, dim, index, source = args

        local_args = (
            input_tensor.to_local() if hasattr(input_tensor, "_layout") else input_tensor,
            dim,
            index.to_local() if hasattr(index, "_layout") else index,
            source.to_local() if hasattr(source, "_layout") else source,
        )
        local_kwargs = dict(kwargs)
        out = local_kwargs.get("out")
        if out is not None and hasattr(out, "_layout"):
            local_kwargs["out"] = out.to_local()

        cache_values = [
            input_tensor.layout if hasattr(input_tensor, "_layout") else None,
            index.layout if hasattr(index, "_layout") else None,
            source.layout if hasattr(source, "_layout") else None,
            out.layout if out is not None and hasattr(out, "_layout") else None,
            out is not None,
            dim,
        ]
        return local_args, local_kwargs, cache_values

    @staticmethod
    def _is_layout(layout) -> bool:
        """Return whether the object behaves like a HyperParallel layout."""
        return layout is not None and hasattr(layout, "tensor_map")

    @staticmethod
    def _validate_layout_objects(op_name, input_layout, source_layout, out_layout, out_provided) -> None:
        """Validate that all tensor arguments use DTensor layouts."""
        if not IndexAddDistributedOp._is_layout(input_layout):
            raise ValueError(f"For {op_name}, input layout should not be None.")
        if not IndexAddDistributedOp._is_layout(source_layout):
            raise ValueError(f"For {op_name}, source must be a DTensor when input is a DTensor.")
        if out_layout is not None and not IndexAddDistributedOp._is_layout(out_layout):
            raise ValueError(f"For {op_name}, out must be a DTensor when input is a DTensor.")
        if out_provided and out_layout is None:
            raise ValueError(f"For {op_name}, out must be a DTensor when input is a DTensor.")

    @staticmethod
    def _normalize_dim(op_name, dim, ndim: int) -> int:
        """Return a non-negative dimension index."""
        if not isinstance(dim, int):
            raise ValueError(f"For {op_name}, dim should be an integer, but got {type(dim).__name__}.")

        original_dim = dim
        if dim < 0:
            dim += ndim
        if dim < 0 or dim >= ndim:
            raise ValueError(f"For {op_name}, dim {original_dim} is out of bounds for tensor with {ndim} dims.")
        return dim

    @staticmethod
    def _validate_mesh_shape(op_name, input_layout, index_layout, source_layout, out_layout) -> None:
        """Validate that all layouts are defined on the same mesh."""
        if input_layout.mesh_shape != source_layout.mesh_shape:
            raise ValueError(f"For {op_name}, input and source must use the same mesh shape.")
        if index_layout is not None and input_layout.mesh_shape != index_layout.mesh_shape:
            raise ValueError(f"For {op_name}, input and index must use the same mesh shape.")
        if out_layout is not None and input_layout.mesh_shape != out_layout.mesh_shape:
            raise ValueError(f"For {op_name}, out layout should use the same mesh shape as input.")

    @staticmethod
    def _validate_index_layout(op_name, index_map) -> None:
        """Validate index_add vector-index layout rules."""
        if len(index_map) != 1:
            raise ValueError(f"For {op_name}, index must be a 1-D DTensor, but got rank={len(index_map)}.")

        if index_map[0] != "None":
            raise ValueError(
                f"For {op_name}, sharded index is not supported, "
                f"but got index sharded on '{index_map[0]}'."
            )

    @staticmethod
    def _validate_source_rank(op_name, source_map, ndim: int) -> None:
        """Validate source rank against input rank."""
        if len(source_map) != ndim:
            raise ValueError(
                f"For {op_name}, input and source must have the same number of dimensions, "
                f"but got input rank={ndim}, source rank={len(source_map)}."
            )

    @staticmethod
    def _validate_target_dim_maps(op_name, input_map, source_map, dim: int) -> None:
        """Validate that the index_add target dimension is replicated."""
        if input_map[dim] != "None":
            raise ValueError(
                f"For {op_name}, index_add along sharded input dimension {dim} is not supported, "
                f"but got dim {dim} sharded on '{input_map[dim]}'."
            )

        if source_map[dim] != "None":
            raise ValueError(
                f"For {op_name}, source target dimension {dim} should be replicated, "
                f"but got source sharded on '{source_map[dim]}'."
            )

    @staticmethod
    def _validate_non_target_maps(op_name, input_map, source_map, dim: int) -> None:
        """Validate that non-target sharding is preserved."""
        for axis, (input_axis_map, source_axis_map) in enumerate(zip(input_map, source_map)):
            if axis == dim:
                continue
            if input_axis_map != source_axis_map:
                raise ValueError(
                    f"For {op_name}, input and source should use the same sharding on non-target axes, "
                    f"but got mismatch at axis {axis}: input='{input_axis_map}', source='{source_axis_map}'"
                )

    @staticmethod
    def _validate_out_layout(op_name, out_layout, input_map) -> None:
        """Validate optional out DTensor layout."""
        if out_layout is not None and out_layout.alias_tensor_map != input_map:
            raise ValueError(
                f"For {op_name}, out layout should match input layout, "
                f"but got out={out_layout.alias_tensor_map}, input={input_map}."
            )

    @staticmethod
    def _validate_input_layouts(op_name, input_layout, index_layout, source_layout, out_layout, out_provided, dim):
        """Validate layouts and return normalized dim."""
        IndexAddDistributedOp._validate_layout_objects(op_name, input_layout, source_layout, out_layout, out_provided)
        IndexAddDistributedOp._validate_mesh_shape(op_name, input_layout, index_layout, source_layout, out_layout)

        input_map = input_layout.alias_tensor_map
        source_map = source_layout.alias_tensor_map
        dim = IndexAddDistributedOp._normalize_dim(op_name, dim, len(input_map))

        IndexAddDistributedOp._validate_source_rank(op_name, source_map, len(input_map))
        if index_layout is not None:
            IndexAddDistributedOp._validate_index_layout(op_name, index_layout.alias_tensor_map)
        IndexAddDistributedOp._validate_target_dim_maps(op_name, input_map, source_map, dim)
        IndexAddDistributedOp._validate_non_target_maps(op_name, input_map, source_map, dim)
        IndexAddDistributedOp._validate_out_layout(op_name, out_layout, input_map)
        return dim

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for IndexAdd operator.

        Rules:
            1. Input and source must be DTensors.
            2. None of the input layouts may have Partial status.
            3. index may be a plain Tensor or a replicated 1-D DTensor.
            4. The index_add target dimension must be replicated on input and source.
            5. Input and source must have the same sharding on non-target axes.
            6. Optional out must be a DTensor whose layout matches the inferred output layout.
            7. Output layout is identical to input layout.

        Args:
            cache_values (list): [input_layout, index_layout, source_layout, out_layout, out_provided, dim]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If any validation rule above is violated.
        """
        input_layout, index_layout, source_layout, out_layout, out_provided, dim = cache_values

        if not self._allow_partial_inputs:
            self._check_partial_inputs([input_layout, index_layout, source_layout])
            if self._is_layout(out_layout):
                self._check_partial_inputs([out_layout])

        self._validate_input_layouts(
            self.op_name,
            input_layout,
            index_layout,
            source_layout,
            out_layout,
            out_provided,
            dim,
        )

        return ((deepcopy(input_layout),), None)
