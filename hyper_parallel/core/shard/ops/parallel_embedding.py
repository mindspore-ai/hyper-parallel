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
Distributed implementation for the embedding operator.
This module handles Tensor Parallelism for Embedding layers (Column/Row Parallel).
"""

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp

class EmbeddingDistributedOp(DistributedOp):
    """
    Distributed implementation for embedding operators.
    Supports Column Parallelism (CP) and Row Parallelism (RP).
    """

    def infer_layout(self, layouts, extra_args=None):
        """
        Infers the output layout (sharding state) based on input and weight layouts.
        """
        if not layouts or len(layouts) < 2:
            raise ValueError(
                f"Operation {self.op_name}: requires both input and weight layouts."
            )

        input_layout = layouts[0]
        weight_layout = layouts[1]
        weight_tensor_map = weight_layout.tensor_map

        # weight_tensor_map: [vocab_size_dim, embed_dim_dim]
        w_shard_vocab_axis = weight_tensor_map[0]
        w_shard_embed_axis = weight_tensor_map[1]

        # Output shape is [*input_shape, embed_dim]
        output_tensor_map = list(input_layout.tensor_map)
        output_tensor_map.append(w_shard_embed_axis)

        output_layout = Layout(
            mesh_shape=input_layout.mesh_shape,
            alias_name=input_layout.alias_name,
            rank_list=input_layout.rank_list
        )
        output_layout.set_tensor_map(tuple(output_tensor_map))

        # If vocab is sharded (Row Parallelism), output is in Partial Sum state
        if w_shard_vocab_axis != -1:
            # pylint: disable=protected-access
            output_layout._partial = list(input_layout.partial)
            vocab_axis_name = input_layout.alias_name[len(input_layout.alias_name) - 1 - w_shard_vocab_axis]
            output_layout.set_partial_by_dev_axis(vocab_axis_name, 'sum')
        # pylint: disable=protected-access
        output_layout._alias_tensor_map = output_layout._build_readable_tensor_map()
        # pylint: disable=protected-access
        output_layout.tensor_map_to_placement()
        output_layout.update_compact_str()

        return output_layout

    def _parse_params(self, args, kwargs):
        """
        Extracts padding_idx, max_norm, and scale_grad from args or kwargs.
        F.embedding signature: (input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq...)
        """
        padding_idx = args[2] if len(args) > 2 else kwargs.get('padding_idx', None)
        max_norm = args[3] if len(args) > 3 else kwargs.get('max_norm', None)
        scale_grad = args[5] if len(args) > 5 else kwargs.get('scale_grad_by_freq', False)
        return padding_idx, max_norm, scale_grad

    def _handle_rp_input(self, input_tensor, weight_tensor, weight_layout, w_shard_vocab_axis,
                         new_args, kwargs, is_args_pad, padding_idx):
        """
        Processes Row Parallelism input: shifts indices, handles padding, and generates masks.
        """
        mesh = weight_layout.mesh
        mesh_dim_idx = len(mesh.mesh_shape) - 1 - w_shard_vocab_axis
        vocab_coord = mesh.get_local_rank(mesh_dim_idx)

        vocab_size_per_partition = weight_tensor.shape[0]
        vocab_start_index = int(vocab_coord * vocab_size_per_partition)
        vocab_end_index = int(vocab_start_index + vocab_size_per_partition)

        # Map global padding_idx to local rank range
        if padding_idx is not None:
            if vocab_start_index <= padding_idx < vocab_end_index:
                mapped_padding_idx = int(padding_idx - vocab_start_index)
                if is_args_pad:
                    new_args[2] = mapped_padding_idx
                else:
                    kwargs['padding_idx'] = mapped_padding_idx
            else:
                if is_args_pad:
                    new_args[2] = None
                else:
                    kwargs.pop('padding_idx', None)

        # Calculate out-of-bounds mask
        mask = (input_tensor >= vocab_start_index) & (input_tensor < vocab_end_index)

        # Cross-platform cast to matching int dtype
        mask_int = mask.to(input_tensor.dtype) if hasattr(mask, "to") else mask.astype(input_tensor.dtype)

        # Shift global indices to local range using native scalar broadcast
        local_input = input_tensor - vocab_start_index

        # Zero out invalid indices mathematically instead of using .where() or clamp().
        # This prevents NPU out-of-bounds memory access during the embedding lookup
        # while keeping the code perfectly backend-neutral.
        local_input = local_input * mask_int

        return local_input, mask_int

    def get_expand_impl(self, func, infer_result, layouts, extra_args=None):
        """
        Returns the execution implementation wrapper.
        Helper functions are used to keep Cyclomatic Complexity (CCN) low.
        """
        weight_layout = layouts[1]
        w_shard_vocab_axis = weight_layout.tensor_map[0]
        w_shard_embed_axis = weight_layout.tensor_map[1]

        # Use native implementation if no weight sharding is applied
        if w_shard_vocab_axis == -1 and w_shard_embed_axis == -1:
            return None

        def distributed_embedding_impl(*args, **kwargs):
            input_tensor, weight_tensor = args[0], args[1]
            new_args, new_kwargs = list(args), kwargs.copy()

            # 1. Parameter extraction and validation
            padding_idx, max_norm, scale_grad = self._parse_params(args, kwargs)

            # Check for max_norm with specific error messages for CP and RP
            if max_norm is not None:
                if w_shard_embed_axis != -1:
                    raise ValueError("Column-Parallel Embedding does not support `max_norm` parameter.")
                if w_shard_vocab_axis != -1:
                    raise ValueError("Row-Parallel Embedding does not support `max_norm` parameter.")

            # Check for scale_grad_by_freq with RP
            if scale_grad and w_shard_vocab_axis != -1:
                raise ValueError("Row-Parallel Embedding does not support `scale_grad_by_freq=True`.")

            # 2. Row Parallel Processing
            input_mask_int = None
            if w_shard_vocab_axis != -1:
                is_args_pad = len(args) > 2
                mapped_input, input_mask_int = self._handle_rp_input(
                    input_tensor, weight_tensor, weight_layout, w_shard_vocab_axis,
                    new_args, new_kwargs, is_args_pad, padding_idx
                )
                new_args[0] = mapped_input

            # 3. Native Operator Execution
            output = func(*new_args, **new_kwargs)

            # 4. Erase invalid partial embeddings (Row-Parallel only)
            if w_shard_vocab_axis != -1 and input_mask_int is not None:
                expanded_mask = input_mask_int[..., None].to(output.dtype)
                output = output * expanded_mask

            return output

        return distributed_embedding_impl
