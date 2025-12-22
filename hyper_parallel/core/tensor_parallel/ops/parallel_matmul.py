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
Distributed implementation for MatMul operator.
"""

from hyper_parallel.core.layout import Layout
from .parallel_ops import DistributedOp


class MatMulExtDistributedOp(DistributedOp):
    """Distributed implementation for MatMul operator."""
    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for MatMul operator.

        MatMul: output = x @ w

        Rules:
        1. Batch dimensions should have same layout
        2. Contracting dimensions should have same layout
        3. Output dimensions inherit layouts from non-contracting dimensions

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
        x_device_matrix = x_layout.device_matrix
        w_device_matrix = w_layout.device_matrix
        if x_device_matrix != w_device_matrix:
            raise ValueError("MatMul inputs must have same device_matrix")

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
            device_matrix=x_layout.device_matrix,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list
        )
        out_layout = output_layout(*output_map)

        # Set partial status
        if x_map[contract_dim] != "None":
            if isinstance(x_map[contract_dim], tuple):
                for axis in x_map[contract_dim]:
                    out_layout.set_partial_by_dev_axis(axis, 'sum')
            else:
                out_layout.set_partial_by_dev_axis(x_map[contract_dim], 'sum')

        return out_layout


class MatMulDistributedOp(DistributedOp):
    """Distributed implementation for MatMul operator."""
    def infer_layout(self, input_layouts, extra_args):
        """
        Infer output layout for MatMul operator.

        MatMul: output = x @ w, with possible transpose

        Args:
            input_layouts (tuple): Layouts of input tensors (x_layout, w_layout)
            extra_args (tuple): Additional arguments (transpose_a, transpose_b)

        Returns:
            Layout: Layout for output tensor
        """
        if len(input_layouts) < 2:
            raise ValueError("MatMul requires at least two input layouts")

        x_layout, w_layout = input_layouts[:2]

        if len(extra_args) != 2:
            raise ValueError("MatMul requires two transpose input")
        transpose_a, transpose_b = extra_args[0], extra_args[1]

        x_dict = x_layout.to_dict()
        w_dict = w_layout.to_dict()

        if x_dict["device_matrix"] != w_dict["device_matrix"]:
            raise ValueError("MatMul inputs must have same device_matrix")

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
            device_matrix=x_layout.device_matrix,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list
        )
        output_map = list(x_map[:-2]) + [x_map[x_input_dim]] + [w_map[w_output_dim]]
        output_layout = output_layout(*output_map)

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

    def _build_output_layout(self, x_layout, merged_batch, x_n, w_p, x_contract):
        """Construct output layout from merged dims and set partial status if needed."""
        output_map = tuple(merged_batch) + (x_n, w_p)

        output_layout = Layout(
            device_matrix=x_layout.device_matrix,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list
        )
        output_layout = output_layout(*output_map)

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

    def infer_layout(self, input_layouts, extra_args=None):
        """
        Infer output layout for BatchMatMulExt operator. Inputs shape are x=[b, n, m] and w=[b, m, p].

        BatchMatMulExt: output = x @ w.

        Rules:
        - Device matrix must match.
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

        if len(input_layouts) < 2:
            raise ValueError("BatchMatMul requires at least two input layouts")
        x_layout, w_layout = input_layouts[:2]

        if x_layout.device_matrix != w_layout.device_matrix:
            raise ValueError("BatchMatMul inputs must have same device_matrix")

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

        return self._build_output_layout(x_layout, merged_batch, x_n, w_p, x_contract)


class BatchMatMulDistributedOp(BaseBatchMatMulDistributedOp):
    """Distributed implementation for BatchMatMul operator."""

    def infer_layout(self, input_layouts, extra_args):
        """
        Infer output layout for BatchMatMul operator. Inputs shape are x=[b, n, m] and w=[b, m, p].

        BatchMatMul: output = x @ w, with possible transpose.

        Rules:
        - Device matrix must match.
        - Contracting K dims must have identical layout: x[-1] == w[-2].
        - Batch dims are right-aligned broadcast:
            none vs shard -> shard
            shard vs none -> shard
            shard vs shard (different) -> error
        - Output batch dims = merged batch dims
        - Output N inherits x[-2], Output P inherits w[-1]

        Args:
            input_layouts (tuple): Layouts of input tensors (x_layout, w_layout)
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

        if len(input_layouts) < 2:
            raise ValueError("BatchMatMul requires at least two input layouts")
        if len(extra_args) != 2:
            raise ValueError("BatchMatMul requires two transpose input")

        x_layout, w_layout = input_layouts[:2]
        transpose_a, transpose_b = extra_args

        if x_layout.device_matrix != w_layout.device_matrix:
            raise ValueError("BatchMatMul inputs must have same device_matrix")

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

        return self._build_output_layout(x_layout, merged_batch, x_n, w_p, x_contract)

class LinearDistributedOp(DistributedOp):
    """Distributed implementation for Linear operator."""
    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for MatMul operator.

        Linear: output = x @ w

        Rules:
        1. Batch dimensions should have same layout
        2. Contracting dimensions should have same layout
        3. Output dimensions inherit layouts from non-contracting dimensions

        Args:
            x_layout (Layout): Layout of input x
            w_layout (Layout): Layout of input w

        Returns:
            tuple: Layout for output tensor
        """
        if len(layouts) != 3:
            raise ValueError(f"Linear layout length is not 3, but {len(layouts)}")
        x_layout = layouts[0]
        w_layout = layouts[1]
        bias_layout = layouts[2]
        if not x_layout or not w_layout:
            raise ValueError(f"x_layout : {x_layout}, w_layout : {w_layout}")
        x_device_matrix = x_layout.device_matrix
        w_device_matrix = w_layout.device_matrix
        if x_device_matrix != w_device_matrix:
            raise ValueError("Linear inputs must have same device_matrix")
        if bias_layout and bias_layout.device_matrix != x_device_matrix:
            raise ValueError("Linear bias and x must have same device_matrix")
        x_map = x_layout.alias_tensor_map
        w_map = w_layout.alias_tensor_map
        contract_dim = len(x_map) - 1
        w_contract_dim = len(w_map) - 1
        if x_map[contract_dim] != w_map[w_contract_dim]:
            raise ValueError(f"Contracting dimensions must have same layout. "
                             f"Got {x_map[contract_dim]} and {w_map[w_contract_dim]}")

        output_dim = 0
        output_map = x_map[:-1] + (w_map[output_dim],)
        if bias_layout and bias_layout.alias_tensor_map[0] != w_map[output_dim]:
            raise ValueError(f"Output dimensions must have same sharding. "
                             f"Got weight output dim sharding size: {w_map[output_dim]}"
                             f" and bias output dim sharding size : {bias_layout.alias_tensor_map[0]}")
        output_layout = Layout(
            device_matrix=x_layout.device_matrix,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list
        )
        out_layout = output_layout(*output_map)

        if x_map[contract_dim] != "None" and bias_layout:
            raise ValueError("Linear/Dense ops cannot shard reduce_dim with bias.")

        return out_layout
