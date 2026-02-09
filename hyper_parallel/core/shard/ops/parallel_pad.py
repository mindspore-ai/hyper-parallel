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
Distributed implementation for Pad operator.
"""

from .parallel_ops import DistributedOp


class PadDistributedOp(DistributedOp):
    """Distributed implementation for Pad operator."""

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for Pad operator.

        The Pad operator expands the tensor size. In a distributed setting,
        padding on a sharded dimension is generally not supported without
        explicit redistribution because it disrupts the uniform slicing logic.

        Args:
            layouts (tuple): Layouts of input tensor.
            extra_args (tuple): Arguments for the operator.
                                extra_args[0] should be the 'pad' tuple.

        Returns:
            Layout: Layout for output tensor (same as input layout).

        Raises:
            ValueError: If padding is attempted on a sharded dimension.
        """
        input_layout = layouts[0]
        tensor_map = input_layout.alias_tensor_map
        ndim = len(tensor_map)

        # Parse extra_args
        # Pytorch style pad: (input, pad, mode, value) -> inputs are stripped by dispatcher
        # extra_args received by dispatcher: (pad, mode, value) (mode and value are optional/defaulted if not passed?)
        # Based on OpDispatcher logic, extra_args contains arguments without layout.
        if not extra_args or not isinstance(extra_args[0], (tuple, list)):
            raise ValueError(f"For '{self.op_name}', expected pad tuple as the first element in extra_args, "
                             f"but got {extra_args}")

        pad = extra_args[0]
        pad_len = len(pad)

        if pad_len % 2 != 0:
            raise ValueError(f"Pad tuple length must be even, but got {pad_len}")

        # Pytorch pad tuple format: (last_dim_left, last_dim_right, 2nd_last_left, 2nd_last_right, ...)
        # We need to check if any dimension being padded is currently sharded.
        num_padded_dims = pad_len // 2
        if num_padded_dims > ndim:
            raise ValueError(f"Padding {num_padded_dims} dimensions but tensor only has {ndim} dimensions.")

        for i in range(num_padded_dims):
            # Calculate the dimension index in the tensor (from 0 to ndim-1)
            # pad index 0,1 -> last dimension (ndim - 1)
            # pad index 2,3 -> second to last dimension (ndim - 2)
            dim_index = ndim - 1 - i

            pad_left = pad[2 * i]
            pad_right = pad[2 * i + 1]

            # If padding is applied on this dimension
            if pad_left != 0 or pad_right != 0:
                axis_alias = tensor_map[dim_index]
                is_sharded = False

                # Check if the axis alias indicates sharding (i.e., not "None")
                if isinstance(axis_alias, (tuple, list)):
                    for sub_alias in axis_alias:
                        if sub_alias != "None":
                            is_sharded = True
                            break
                elif axis_alias != "None":
                    is_sharded = True

                if is_sharded:
                    raise ValueError(
                        f"Distributed Pad operator does not support padding on a sharded dimension. "
                        f"Dimension {dim_index} (alias: {axis_alias}) is sharded. "
                        f"Please redistribute the tensor to Replicate status on this dimension before padding."
                    )

        # If no sharded dimension is padded, the output layout is identical to the input layout.
        # The local tensor shape changes, but the mapping from device mesh to tensor dimensions remains valid.
        return input_layout

    # Note: get_expand_impl is not overridden because we default to returning None.
    # OpDispatcher will use the original function (e.g., torch.nn.functional.pad) on the local tensor.
    # Since we ensured the padded dimensions are Replicated, local padding is mathematically correct.
