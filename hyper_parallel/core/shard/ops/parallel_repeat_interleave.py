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
from hyper_parallel.core.layout import Layout
from .parallel_ops import DistributedOp

class RepeatInterleaveDistributedOp(DistributedOp):
    """Distributed implementation for torch.repeat_interleave."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for RepeatInterleave operator.

        RepeatInterleave: output = repeat_interleave(input, repeats, dim)
        
        Rules:
        1. dim = None if not specified.
        2. The dimension `dim` MUST be unsharded to ensure global repeat_interleave correctness.
        3. Output layout usually same as input, but shape changes.

        Args:
            layouts (tuple): Layouts of inputs. Expected:
                layouts[0] (Layout): Input tensor layout (required).
            extra_args (tuple, optional): Contains repeats and dim. Expected:
                extra_args[0] (int or Tensor): Number of repeats or Tensor
                extra_args[1] (int, optional): Dimension to repeat. Defaults to -1.

        Returns:
            tuple: Layouts for values.
        """
        if not layouts or layouts[0] is None:
            raise ValueError("repeat_interleave requires a valid input tensor layout.")

        input_layout = layouts[0]
        in_tensor_map = input_layout.tensor_map
        dim = None # The dimension along which to repeat values. By default, use the flattened input array, and return a flat output array.
        if len(extra_args) >= 2 and extra_args[1] is not None:
            dim = extra_args[1]
        if dim is None:
            sharded_dims = [i for i, shard in enumerate(in_tensor_map) if shard != -1]
            if not sharded_dims: # not shard
                output_tensor_map = [-1]
            # Only can shard on the first dimension.
            elif sharded_dims == [0] and in_tensor_map[0] != -1: 
                output_tensor_map = [in_tensor_map[0]]
            else:
                # Other dims must NOT be sharded.
                raise ValueError(
                    f"Operation {self.op_name}: Cannot flatten tensor when dim=None."
                )
            def idx_to_alias(idx, aliases):
                if idx == -1:
                    return "None"
                return aliases[len(aliases) - idx - 1]
            output_map = tuple(idx_to_alias(idx, input_layout.alias_name) for idx in output_tensor_map)

            output_layout = Layout(
                mesh_shape=input_layout.mesh_shape,
                alias_name=input_layout.alias_name,
                rank_list=input_layout.rank_list
            )

            return output_layout(*output_map)
        input_dim = len(in_tensor_map)
        if dim < 0:
            dim = input_dim + dim
        # Check if dimension is within valid range
        if not 0 <= dim < input_dim:
            raise ValueError(f"Dimension out of range (expected to be in [0, {input_dim}), but got {dim}).")
        # The chosen dim must NOT be sharded
        if in_tensor_map[dim] != -1:
            raise ValueError(
                f"Operation {self.op_name}: Cannot perform sharding on params along the chosen dim"
            )
        
        # Output layout same as input layout (shape change does not affect sharding pattern)
        return input_layout