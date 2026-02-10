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
Distributed implementation for Multinomial operator.
"""

from hyper_parallel.core.layout import Layout
from .parallel_ops import DistributedOp


class MultinomialDistributedOp(DistributedOp):
    """Distributed implementation for Multinomial operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for Multinomial operator.

        Args:
            layouts (tuple): Layouts of input tensor.
            extra_args (tuple): Arguments for the operator (num_samples, replacement, generator).
                                Note: logic assumes num_samples affects shape but not the sharding map pattern.

        Returns:
            Layout: Layout for output tensor.
        """
        layout = layouts[0]
        in_tensor_map = layout.alias_tensor_map
        ndim = len(in_tensor_map)

        # PyTorch multinomial supports 1D or 2D tensors
        if ndim not in (1, 2):
            raise ValueError(f"For 'multinomial', input dimension must be 1 or 2, but got {ndim}.")

        # The last dimension (probability dimension) must NOT be sharded.
        # Multinomial sampling requires the full probability distribution (cumulative sum, normalization)
        # to be present on the device. If it's sharded, we cannot sample correctly without communication.
        # In alias_tensor_map, "None" indicates the dimension is NOT sharded.
        prob_dim_map = in_tensor_map[-1]

        # Check if the last dimension is sharded (i.e., not "None")
        if prob_dim_map != "None":
            raise ValueError(
                f"For 'multinomial', the last dimension (probability category dimension) "
                f"must not be sharded (Replicated). Got layout map: {in_tensor_map}. "
                f"Please redistribute the tensor to replicate the last dimension before calling multinomial."
            )

        out_tensor_map = ()
        if ndim == 1:
            # Input: (C,) -> Probability distribution
            # Output: (num_samples,) -> Sampled indices
            # Since the input distribution C is replicated (checked above), the samples generated
            # from it have no spatial correlation to shards, so the output is also Replicated ("None").
            out_tensor_map = ("None",)
        else:
            # Input: (N, C) -> Batch of distributions
            # Output: (N, num_samples) -> Batch of sampled indices
            #
            # Rule:
            # 1. Preserve sharding on the Batch dimension (dim 0). If input is data parallel, output is too.
            # 2. The new dimension (num_samples) is created locally and is not sharded ("None").
            batch_dim_map = in_tensor_map[0]
            out_tensor_map = (batch_dim_map, "None")

        output_layout = Layout(
            mesh_shape=layout.mesh_shape,
            alias_name=layout.alias_name,
            rank_list=layout.rank_list
        )

        # Create output layout using the inferred tensor map aliases
        return output_layout(*out_tensor_map)
