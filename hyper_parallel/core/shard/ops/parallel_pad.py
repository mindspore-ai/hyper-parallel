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

from typing import Tuple

from .parallel_ops import DistributedOp


def _normalize_pad_args(x, pad, mode='constant', value=None):
    return (x, pad, mode, value), {}


class PadDistributedOp(DistributedOp):
    """Distributed implementation for Pad operator."""

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for Pad operator.

        Args:
            args (tuple): Input tensor followed by pad arguments.
            kwargs (dict): Keyword arguments for pad.

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        args, _ = _normalize_pad_args(*args, **kwargs)
        input_tensor = args[0]
        pad, mode, value = args[1], args[2], args[3]

        local_args = (input_tensor.to_local(), pad, mode, value)
        local_kwargs = {}
        cache_values = [input_tensor.layout, pad]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:  # pylint: disable=W0221
        """
        Infer output layout for Pad operator.

        Rules:
            1. Input must not have Partial status.
            2. pad must be a tuple or list with even length.
            3. The number of padded dimensions must not exceed input ndim.
            4. Any dimension with non-zero padding must not be sharded.
            5. Output layout is identical to the input layout.

        Args:
            cache_values (list): [input_layout, pad].

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If input has Partial status, pad is invalid, or padding is
                attempted on a sharded dimension.
        """
        if len(cache_values) != 2:
            raise ValueError(
                f"For {self.op_name}, cache_values length should be 2, but got {len(cache_values)}"
            )

        input_layout, pad = cache_values[0], cache_values[1]
        if input_layout is None:
            raise ValueError(f"For {self.op_name}, pad requires a valid input tensor layout.")

        self._check_partial_inputs([input_layout])

        tensor_map = input_layout.alias_tensor_map
        ndim = len(tensor_map)

        if not isinstance(pad, (tuple, list)):
            raise ValueError(
                f"For {self.op_name}, expected pad tuple or list, but got {type(pad)}"
            )

        pad_len = len(pad)

        if pad_len % 2 != 0:
            raise ValueError(f"For {self.op_name}, Pad tuple length must be even, but got {pad_len}")

        # Pytorch pad tuple format: (last_dim_left, last_dim_right, 2nd_last_left, 2nd_last_right, ...)
        # We need to check if any dimension being padded is currently sharded.
        num_padded_dims = pad_len // 2
        if num_padded_dims > ndim:
            raise ValueError(
                f"For {self.op_name}, Padding {num_padded_dims} dimensions but tensor only has {ndim} dimensions."
            )

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
                is_sharded = (any(alias != "None" for alias in axis_alias)
                              if isinstance(axis_alias, (tuple, list))
                              else axis_alias != "None")

                if is_sharded:
                    raise ValueError(
                        f"For {self.op_name}, Distributed Pad operator does not support padding "
                        f"on a sharded dimension. "
                        f"Dimension {dim_index} (alias: {axis_alias}) is sharded. "
                        f"Please redistribute the tensor to Replicate status on this dimension before padding."
                    )

        # If no sharded dimension is padded, the output layout is identical to the input layout.
        # The local tensor shape changes, but the mapping from device mesh to tensor dimensions remains valid.
        return ((input_layout,), None)

    # Note: get_expand_impl is not overridden because we default to returning None.
    # OpDispatcher will use the original function (e.g., torch.nn.functional.pad) on the local tensor.
    # Since we ensured the padded dimensions are Replicated, local padding is mathematically correct.
