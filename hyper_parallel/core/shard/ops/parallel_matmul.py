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
Distributed implementation for MatMul operator.
"""

from typing import Callable, Optional, Tuple

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp


def _propagate_partial_from_inputs(out_layout, x_layout, w_layout):
    """
    Propagate Partial status from input layouts to the output layout for matmul-like operations.

    For matmul ``y = x @ w``, the output should inherit Partial state from its inputs in addition
    to any Partial state induced by the contracting dimension being sharded.

    **Semantic rules for input Partial propagation:**

    +-------------------+----------------------------+----------------------------------------------+
    | x input           | w / weight                 | output behavior                              |
    +-------------------+----------------------------+----------------------------------------------+
    | **Partial(d)**    | **Replicate** (contracting) | **Propagate Partial(d)**.                    |
    |                   |                            | Distributive law:                             |
    |                   |                            | ``(x0 + x1) @ w = x0 @ w + x1 @ w``.        |
    |                   |                            | Output carries Partial(d).                   |
    +-------------------+----------------------------+----------------------------------------------+
    | Replicate         | **Partial(d)**             | **Propagate Partial(d)**. Symmetric to above.|
    +-------------------+----------------------------+----------------------------------------------+
    | **Partial(d1)**   | **Partial(d2)**, d1 != d2  | **Propagate both**.                          |
    |                   |                            | Each partial axis is independent;            |
    |                   |                            | ``(sum over d2)`` applied to x is legal.      |
    +-------------------+----------------------------+----------------------------------------------+
    | **Partial(d)**    | **Partial(d)** same axis   | **Error**. Cross-terms ``x0 @ w1`` and       |
    |                   | same/different ops         | ``x1 @ w0`` cannot be computed locally.      |
    +-------------------+----------------------------+----------------------------------------------+
    | **Partial(d)**    | **Shard(d)** on the same   | **Error** naturally raised by                |
    |                   | device axis in the output  | ``Layout.set_partial_by_dev_axis``:          |
    |                   | dimension map              | "Partial dim must be replicate."             |
    +-------------------+----------------------------+----------------------------------------------+

    Args:
        out_layout (Layout): The partially-built output layout whose ``alias_tensor_map``
            has already been set (via ``Layout.__call__``).
        x_layout (Layout): Layout of the first input tensor (activations).
        w_layout (Layout): Layout of the second input tensor (weight / matrix).

    Raises:
        ValueError: If both ``x_layout`` and ``w_layout`` have Partial on the same device
            axis with different reduce operations (e.g. one is 'sum' and the other 'avg').
    """
    if x_layout is None or w_layout is None:
        return

    # Propagate x's partial status to output
    for dev_idx, op in enumerate(x_layout.partial):
        if op is not None:
            out_layout.set_partial_by_dev_axis(
                x_layout.alias_name[dev_idx], op
            )

    # Propagate w's partial status to output, checking for conflicts with x's partial
    for dev_idx, op in enumerate(w_layout.partial):
        if op is not None:
            axis_alias = w_layout.alias_name[dev_idx]
            existing = out_layout.get_partial_by_dev_id(axis_alias)
            if existing is not None and existing != op:
                raise ValueError(
                    f"Cannot propagate Partial from both input layouts: "
                    f"x has Partial({existing}) on axis '{axis_alias}' while "
                    f"w has Partial({op}) on the same axis. "
                    f"Partial on the same axis with different reduce ops for both inputs is invalid."
                )
            out_layout.set_partial_by_dev_axis(axis_alias, op)


class MatMulExtDistributedOp(DistributedOp):
    """Distributed implementation for MatMul operator."""
    def infer_layout(self, layouts: tuple, extra_args: Optional[tuple] = None) -> tuple:
        """
        Infer output layout for MatMul operator.

        MatMul: output = x @ w

        Rules:
        1. Batch dimensions should have same layout
        2. Contracting dimensions should have same layout
        3. Output dimensions inherit layouts from non-contracting dimensions
        4. Input Partial status is propagated to the output

        Args:
            x_layout (Layout): Layout of input x
            w_layout (Layout): Layout of input w

        Returns:
            tuple: Layout for output tensor
        """
        if len(layouts) != 2:
            raise ValueError(f"MatMul layout length is not 2, but {len(layouts)}")
        x_layout = layouts[0]
        w_layout = layouts[1]
        if not x_layout or not w_layout:
            raise ValueError(f"x_layout : {x_layout}, w_layout : {w_layout}")
        x_mesh_shape = x_layout.mesh_shape
        w_mesh_shape = w_layout.mesh_shape
        if x_mesh_shape != w_mesh_shape:
            raise ValueError("MatMul inputs must have same mesh_shape")

        x_map = x_layout.alias_tensor_map
        w_map = w_layout.alias_tensor_map
        contract_dim = len(x_map) - 1
        w_contract_dim = len(w_map) - 2
        if x_map[contract_dim] != w_map[w_contract_dim]:
            raise ValueError(f"Contracting dimensions must have same layout. "
                             f"Got {x_map[contract_dim]} and {w_map[w_contract_dim]}")

        output_dim = len(w_map) - 1
        output_map = x_map[:-1] + (w_map[output_dim],)

        output_layout = Layout(
            mesh_shape=x_layout.mesh_shape,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list
        )
        out_layout = output_layout(*output_map)

        # Propagate Partial from inputs (e.g., x already has Partial from a prior matmul)
        _propagate_partial_from_inputs(out_layout, x_layout, w_layout)

        # Set partial status from contracting dimension sharding
        if x_map[contract_dim] != "None":
            if isinstance(x_map[contract_dim], tuple):
                for axis in x_map[contract_dim]:
                    out_layout.set_partial_by_dev_axis(axis, 'sum')
            else:
                out_layout.set_partial_by_dev_axis(x_map[contract_dim], 'sum')

        return out_layout


class MatMulDistributedOp(DistributedOp):
    """Distributed implementation for MatMul operator."""
    def infer_layout(self, layouts: tuple, extra_args: Optional[tuple] = None) -> tuple:
        """
        Infer output layout for MatMul operator.

        MatMul: output = x @ w, with possible transpose

        Args:
            layouts (tuple): Layouts of input tensors (x_layout, w_layout)
            extra_args (tuple): Additional arguments (transpose_a, transpose_b)

        Returns:
            Layout: Layout for output tensor
        """
        if len(layouts) < 2:
            raise ValueError("MatMul requires at least two input layouts")

        x_layout, w_layout = layouts[:2]

        if len(extra_args) != 2:
            raise ValueError("MatMul requires two transpose input")
        transpose_a, transpose_b = extra_args[0], extra_args[1]

        x_dict = x_layout.to_dict()
        w_dict = w_layout.to_dict()

        if x_dict["mesh_shape"] != w_dict["mesh_shape"]:
            raise ValueError("MatMul inputs must have same mesh_shape")

        x_map = x_layout.alias_tensor_map
        w_map = w_layout.alias_tensor_map

        # Determine contracting dimensions based on transpose flags
        if transpose_a:
            x_input_dim = len(x_map) - 1
            x_contract_dim = len(x_map) - 2  # Second to last dimension
        else:
            x_input_dim = len(x_map) - 2
            x_contract_dim = len(x_map) - 1  # Last dimension

        if transpose_b:
            w_output_dim = len(w_map) - 2
            w_contract_dim = len(w_map) - 1  # Last dimension
        else:
            w_output_dim = len(w_map) - 1
            w_contract_dim = len(w_map) - 2  # Second to last dimension

        # Validate contracting dimensions
        if x_map[x_contract_dim] != w_map[w_contract_dim]:
            raise ValueError(f"Contracting dimensions must have same layout. "
                             f"Got {x_map[x_contract_dim]} and {w_map[w_contract_dim]}")

        # Create output layout
        output_layout = Layout(
            mesh_shape=x_layout.mesh_shape,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list
        )
        output_map = list(x_map[:-2]) + [x_map[x_input_dim]] + [w_map[w_output_dim]]
        output_layout = output_layout(*output_map)

        # Propagate Partial from inputs (e.g., x already has Partial from a prior matmul)
        _propagate_partial_from_inputs(output_layout, x_layout, w_layout)

        # Set partial status
        if x_map[x_contract_dim] != "None":
            if isinstance(x_map[x_contract_dim], tuple):
                for axis in x_map[x_contract_dim]:
                    output_layout.set_partial_by_dev_axis(axis, 'sum')
            else:
                output_layout.set_partial_by_dev_axis(x_map[x_contract_dim], 'sum')

        return output_layout


class BaseBatchMatMulDistributedOp(DistributedOp):
    """Base class for BatchMatMul distributed implementations."""

    def _merge_batch_entry(self, x_dims, w_dims):
        """
        Merge two batch tensor_map entries with broadcasting:
        - none vs X -> X
        - X vs none -> X
        - X vs X (exact same after normalization) -> X
        - otherwise -> conflict
        """
        if self._is_none_entry(x_dims) and self._is_none_entry(w_dims):
            return "None"
        if self._is_none_entry(x_dims):
            return w_dims
        if self._is_none_entry(w_dims):
            return x_dims
        if x_dims == w_dims:
            return x_dims
        raise ValueError(f"Incompatible batch sharding between inputs: {x_dims} vs {w_dims}")

    def _is_none_entry(self, entry):
        """An entry is 'none' (no sharding) if it is 'None' or tuple of all 'None'."""
        if isinstance(entry, tuple):
            return all(i == "None" for i in entry)
        return entry == "None"

    def _merge_batches(self, x_map, w_map):
        """Right-align and merge batch dims from x_map and w_map."""
        x_batch = list(x_map[:-2])
        w_batch = list(w_map[:-2])
        max_b = max(len(x_batch), len(w_batch))
        x_batch = ["None"] * (max_b - len(x_batch)) + x_batch
        w_batch = ["None"] * (max_b - len(w_batch)) + w_batch
        merged_batch = []
        for xb, wb in zip(x_batch, w_batch):
            merged_batch.append(self._merge_batch_entry(xb, wb))
        return merged_batch

    def _build_output_layout(self, x_layout, w_layout, merged_batch, x_n, w_p, x_contract):
        """Construct output layout from merged dims and set partial status if needed."""
        output_map = tuple(merged_batch) + (x_n, w_p)

        output_layout = Layout(
            mesh_shape=x_layout.mesh_shape,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list
        )
        output_layout = output_layout(*output_map)

        # Propagate Partial from inputs
        _propagate_partial_from_inputs(output_layout, x_layout, w_layout)

        # Set partial status
        if x_contract != "None":
            if isinstance(x_contract, tuple):
                for axis in x_contract:
                    output_layout.set_partial_by_dev_axis(axis, 'sum')
            else:
                output_layout.set_partial_by_dev_axis(x_contract, 'sum')

        return output_layout


class BatchMatMulExtDistributedOp(BaseBatchMatMulDistributedOp):
    """Distributed implementation for BatchMatMulExt operator."""

    def infer_layout(self, layouts: tuple, extra_args: Optional[tuple] = None) -> tuple:
        """
        Infer output layout for BatchMatMulExt operator. Inputs shape are x=[b, n, m] and w=[b, m, p].

        BatchMatMulExt: output = x @ w.

        Rules:
        - Mesh shape must match.
        - Contracting K dims must have identical layout: x[-1] == w[-2].
        - Batch dims are right-aligned broadcast:
            none vs shard -> shard
            shard vs none -> shard
            shard vs shard (different) -> error
        - Output batch dims = merged batch dims
        - Output N inherits x[-2], Output P inherits w[-1]

        Args:
            x_layout (Layout): Layout of input x
            w_layout (Layout): Layout of input w

        Returns:
            tuple: Layout for output tensor

        Examples:
            layout = Layout((2, 2, 2), ("dp", "cp", "mp"))
            x_layout = layout("dp", "cp", "mp")
            w_layout = layout("dp", "mp", "None")
            out_layout = layout("dp", "cp", "None")
        """

        if len(layouts) < 2:
            raise ValueError("BatchMatMul requires at least two input layouts")
        x_layout, w_layout = layouts[:2]

        if x_layout.mesh_shape != w_layout.mesh_shape:
            raise ValueError("BatchMatMul inputs must have same mesh_shape")

        x_map = x_layout.alias_tensor_map
        w_map = w_layout.alias_tensor_map

        # contracting dims
        x_contract = x_map[-1]
        w_contract = w_map[-2]
        if x_contract != w_contract:
            raise ValueError(f"Contracting (M) dim layouts must match, got {x_contract} (x) vs {w_contract} (w)")

        merged_batch = self._merge_batches(x_map, w_map)
        x_n = x_map[-2]
        w_p = w_map[-1]

        return self._build_output_layout(x_layout, w_layout, merged_batch, x_n, w_p, x_contract)


class BatchMatMulDistributedOp(BaseBatchMatMulDistributedOp):
    """Distributed implementation for BatchMatMul operator."""

    def infer_layout(self, layouts: tuple, extra_args: Optional[tuple] = None) -> tuple:
        """
        Infer output layout for BatchMatMul operator. Inputs shape are x=[b, n, m] and w=[b, m, p].

        BatchMatMul: output = x @ w, with possible transpose.

        Rules:
        - Mesh shape must match.
        - Contracting K dims must have identical layout: x[-1] == w[-2].
        - Batch dims are right-aligned broadcast:
            none vs shard -> shard
            shard vs none -> shard
            shard vs shard (different) -> error
        - Output batch dims = merged batch dims
        - Output N inherits x[-2], Output P inherits w[-1]

        Args:
            layouts (tuple): Layouts of input tensors (x_layout, w_layout)
            extra_args (tuple): Additional arguments (transpose_a, transpose_b)

        Returns:
            tuple: Layout for output tensor

        Examples:
            ms.mint.bmm((x_layout, w_layout),(transpose_a=True, transpose_b=False))
            layout = Layout((2, 2, 2), ("dp", "cp", "mp"))
            x_layout = layout("dp", "mp", "cp")
            w_layout = layout("dp", "mp", "None")
            out_layout = layout("dp", "cp", "None")
        """

        if len(layouts) < 2:
            raise ValueError("BatchMatMul requires at least two input layouts")
        if len(extra_args) != 2:
            raise ValueError("BatchMatMul requires two transpose input")

        x_layout, w_layout = layouts[:2]
        transpose_a, transpose_b = extra_args

        if x_layout.mesh_shape != w_layout.mesh_shape:
            raise ValueError("BatchMatMul inputs must have same mesh_shape")

        x_map = x_layout.alias_tensor_map
        w_map = w_layout.alias_tensor_map

        # handle transpose
        if transpose_a:
            x_n = x_map[-1]
            x_contract = x_map[-2]
        else:
            x_n = x_map[-2]
            x_contract = x_map[-1]

        if transpose_b:
            w_contract = w_map[-1]
            w_p = w_map[-2]
        else:
            w_contract = w_map[-2]
            w_p = w_map[-1]

        if x_contract != w_contract:
            raise ValueError(f"Contracting (M) dim layouts must match, got {x_contract} (x) vs {w_contract} (w)")

        merged_batch = self._merge_batches(x_map, w_map)

        return self._build_output_layout(x_layout, w_layout, merged_batch, x_n, w_p, x_contract)


def _normalize_linear_args(x, weight, bias=None):
    return (x, weight, bias), {}


class LinearDistributedOp(DistributedOp):
    """Distributed implementation for Linear operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for Linear operator.

        Args:
            args (tuple): Input arguments containing x and weight tensors.
            kwargs (dict): Keyword arguments, may contain bias.

        Returns:
            tuple: (local_args, local_kwargs, cache_values) where local_args contains
                local tensors for x, weight, and bias; local_kwargs is empty; and
                cache_values contains layouts and None-sentinel for absent bias.
        """
        args, kwargs = _normalize_linear_args(*args, **kwargs)
        x_tensor, w_tensor, bias = args[0], args[1], args[2]
        local_args = (
            x_tensor.to_local(),
            w_tensor.to_local(),
            bias.to_local() if hasattr(bias, '_layout') else bias,
        )
        local_kwargs = {}
        cache_values = [
            x_tensor.layout,
            w_tensor.layout,
            bias.layout if hasattr(bias, '_layout') else None,
        ]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for Linear operator (output = x @ weight.T + bias).

        Rules:
            1. x and weight must share the same mesh_shape.
            2. weight must be 2D [out_features, in_features].
            3. Contracting dimensions (in_features) must have the same layout.
            4. Output batch dimensions inherit from x; output feature dim inherits from weight dim 0.
            5. Partial state is set on the output when the contracting dimension is sharded.

        Args:
            cache_values (list): [x_layout, w_layout, bias_layout] where bias_layout may be None.

        Returns:
            tuple: ((out_layout,), None)

        Raises:
            ValueError: If cache_values length is not 3, layouts are invalid, mesh shapes differ,
                weight is not 2D, contracting dims mismatch, or bias sharding is inconsistent.
        """
        if len(cache_values) != 3:
            raise ValueError(
                f"For {self.op_name}, cache_values length should be 3, but got {len(cache_values)}"
            )
        x_layout = cache_values[0]
        w_layout = cache_values[1]
        bias_layout = cache_values[2]

        if not x_layout or not w_layout:
            raise ValueError(f"x_layout : {x_layout}, w_layout : {w_layout}")

        x_mesh_shape = x_layout.mesh_shape
        w_mesh_shape = w_layout.mesh_shape
        if x_mesh_shape != w_mesh_shape:
            raise ValueError(
                f"For {self.op_name}, x and weight must have the same mesh_shape, "
                f"but got x: {x_mesh_shape} and weight: {w_mesh_shape}"
            )
        if bias_layout and bias_layout.mesh_shape != x_mesh_shape:
            raise ValueError(
                f"For {self.op_name}, bias and x must have the same mesh_shape, "
                f"but got bias: {bias_layout.mesh_shape} and x: {x_mesh_shape}"
            )

        x_map = x_layout.alias_tensor_map
        w_map = w_layout.alias_tensor_map

        if len(w_map) != 2:
            raise ValueError(
                f"For {self.op_name}, weight should be 2D [out_features, in_features], "
                f"but got {len(w_map)}D"
            )

        x_contract_dim = len(x_map) - 1
        w_contract_dim = len(w_map) - 1
        if x_map[x_contract_dim] != w_map[w_contract_dim]:
            raise ValueError(
                f"For {self.op_name}, contracting dimensions must have the same layout, "
                f"but got x: {x_map[x_contract_dim]} and weight: {w_map[w_contract_dim]}"
            )

        output_dim = 0
        output_map = x_map[:-1] + (w_map[output_dim],)
        if bias_layout and bias_layout.alias_tensor_map[0] != w_map[output_dim]:
            raise ValueError(
                f"For {self.op_name}, bias output dim sharding must match weight output dim sharding, "
                f"but got weight: {w_map[output_dim]} and bias: {bias_layout.alias_tensor_map[0]}"
            )

        output_layout = Layout(
            mesh_shape=x_layout.mesh_shape,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list,
        )
        out_layout = output_layout(*output_map)

        # Propagate Partial from inputs (e.g., x already has Partial from a prior matmul)
        _propagate_partial_from_inputs(out_layout, x_layout, w_layout)

        # Set partial status when contracting dimension is sharded
        if x_map[x_contract_dim] != "None":
            if isinstance(x_map[x_contract_dim], tuple):
                for axis in x_map[x_contract_dim]:
                    out_layout.set_partial_by_dev_axis(axis, 'sum')
            else:
                out_layout.set_partial_by_dev_axis(x_map[x_contract_dim], 'sum')

        return ((out_layout,), None)

    def get_expand_impl(self, func: Callable, infer_result: tuple,
                        cache_values: list) -> Optional[Callable]:
        """
        Return a custom expand implementation when bias scaling is needed.

        When the contracting dimension is sharded each rank computes a partial sum
        (x_shard @ w_shard.T + bias).  After AllReduce the bias would accumulate
        scaling_factor times.  The returned closure pre-divides bias by scaling_factor
        to keep the result numerically correct.

        Args:
            func: Original operator callable.
            infer_result (tuple): ((out_layout,), None) from infer_layout.
            cache_values (list): [x_layout, w_layout, bias_layout].

        Returns:
            callable | None: expand_impl closure when scaling is required, else None.
        """
        x_layout = cache_values[0]
        bias_layout = cache_values[2]
        x_map = x_layout.alias_tensor_map
        x_contract_dim = len(x_map) - 1

        # Guard: scaling only needed when contract dim is sharded AND bias is present
        if x_map[x_contract_dim] == "None" or not bias_layout:
            return None

        output_layout = infer_result[0][0]
        scaling_factor = 1
        if isinstance(x_map[x_contract_dim], tuple):
            for axis in x_map[x_contract_dim]:
                scaling_factor *= output_layout.mesh.get_device_num_along_axis(axis)
        else:
            scaling_factor *= output_layout.mesh.get_device_num_along_axis(x_map[x_contract_dim])

        def expand_impl(x: object, w: object, bias: object) -> object:
            """Pre-scale bias to counteract the AllReduce accumulation over shards.

            Args:
                x (object): Local input activation tensor.
                w (object): Local weight tensor.
                bias (object): Local bias tensor to be pre-scaled.

            Returns:
                object: Result of the linear operation with pre-scaled bias.
            """
            return func(x, w, bias / scaling_factor)

        return expand_impl
