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
Distributed implementation for HistcExt operator.
"""

from typing import Tuple

from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.platform import get_platform
from .parallel_ops import DistributedOp

platform = get_platform()


def _normalize_histc_args(x, bins=100, min_val=0, max_val=0):
    return (x, bins, min_val, max_val), {}


class HistcExtDistributedOp(DistributedOp):
    """
    Distributed implementation for HistcExt operator.

    HistcExt computes the histogram of a tensor. In distributed setting:
    - Each device computes a local histogram
    - Local histograms are aggregated using AllReduce(SUM)
    - Output is always replicated (1D tensor with shape (bins,))
    """

    def __init__(self, op_name: str = "HistcExt") -> None:
        """Initialize HistcExtDistributedOp."""
        super().__init__(op_name)

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """
        Preprocess arguments for HistcExt operator.

        Args:
            args (tuple): Input arguments, first element is the input tensor.
            kwargs (dict): Keyword arguments (bins, min, max).

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        # Map external API parameter names (min, max) to internal names to avoid
        # shadowing Python builtins.
        if "min" in kwargs:
            kwargs["min_val"] = kwargs.pop("min")
        if "max" in kwargs:
            kwargs["max_val"] = kwargs.pop("max")
        args, kwargs = _normalize_histc_args(*args, **kwargs)
        x, bins, min_val, max_val = args
        local_args = (x.to_local(), bins, min_val, max_val)
        local_kwargs = {}
        cache_values = [x.layout, bins, min_val, max_val]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """
        Infer output layout for HistcExt operator.

        Rules:
            1. Input layout must not be None.
            2. bins must be a positive integer.
            3. min and max must be numbers with min <= max.
            4. Output is always a 1D replicated tensor of shape (bins,).
            5. When input is sharded, output carries Partial(sum) on sharded device axes.

        Args:
            cache_values (list): [input_layout, bins, min, max]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If any rule above is violated.
        """
        x_layout = cache_values[0]
        bins = cache_values[1]
        min_val = cache_values[2]
        max_val = cache_values[3]

        if not self._allow_partial_inputs:
            self._check_partial_inputs([x_layout])

        if x_layout is None or x_layout.mesh_shape is None:
            raise ValueError(
                f"For {self.op_name}, input layout should not be None, "
                f"but got {x_layout}"
            )

        if not isinstance(bins, int):
            raise ValueError(
                f"For {self.op_name}, bins should be an integer, "
                f"but got {type(bins).__name__}"
            )
        if bins <= 0:
            raise ValueError(
                f"For {self.op_name}, bins should be a positive integer, "
                f"but got {bins}"
            )
        if not isinstance(min_val, (int, float)):
            raise ValueError(
                f"For {self.op_name}, min should be a number, "
                f"but got {type(min_val).__name__}"
            )
        if not isinstance(max_val, (int, float)):
            raise ValueError(
                f"For {self.op_name}, max should be a number, "
                f"but got {type(max_val).__name__}"
            )
        if min_val > max_val:
            raise ValueError(
                f"For {self.op_name}, min should be less than or equal to max, "
                f"but got min={min_val}, max={max_val}"
            )

        output_layout = Layout(
            mesh_shape=x_layout.mesh_shape,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list,
        )
        out_layout = output_layout("None",)

        has_sharding = any(
            alias is not None and alias != "None"
            for alias in x_layout.alias_tensor_map
        )

        if has_sharding:
            for alias, tensor_map_val in zip(x_layout.alias_name, x_layout.alias_tensor_map):
                if tensor_map_val is not None and tensor_map_val != "None":
                    out_layout.set_partial_by_dev_axis(alias, "sum")

        return ((out_layout,), None)
