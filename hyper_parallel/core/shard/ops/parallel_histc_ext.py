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

from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.platform import get_platform
from .parallel_ops import DistributedOp

platform = get_platform()


class HistcExtDistributedOp(DistributedOp):
    """
    Distributed implementation for HistcExt operator.

    HistcExt computes the histogram of a tensor. In distributed setting:
    - Each device computes a local histogram
    - Local histograms are aggregated using AllReduce(SUM)
    - Output is always replicated (1D tensor with shape (bins,))
    """

    def __init__(self, op_name="HistcExt"):
        super().__init__(op_name)

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for HistcExt operator.

        Args:
            layouts (tuple): Layouts of input tensor.
            extra_args (tuple): (bins, min, max) parameters.

        Returns:
            Layout: Layout for output histogram tensor.
        """
        if not layouts or len(layouts) < 1:
            raise ValueError(
                f"{self.__class__.__name__} requires at least one input layout, "
                f"got {len(layouts) if layouts else 0}"
            )
        x_layout = layouts[0]
        if x_layout is None or x_layout.mesh_shape is None:
            raise ValueError("Input layout cannot be None.")

        bins = extra_args[0] if len(extra_args) > 0 else 100
        min_val = extra_args[1] if len(extra_args) > 1 else 0
        max_val = extra_args[2] if len(extra_args) > 2 else 0

        if not isinstance(bins, int):
            raise ValueError(f"bins must be an integer, got {type(bins)}")
        if bins <= 0:
            raise ValueError(f"bins must be a positive integer, got {bins}")
        if not isinstance(min_val, (int, float)):
            raise ValueError(f"min must be a number, got {type(min_val)}")
        if not isinstance(max_val, (int, float)):
            raise ValueError(f"max must be a number, got {type(max_val)}")
        if min_val > max_val:
            raise ValueError(f"min must be less than or equal to max, got min={min_val}, max={max_val}")

        output_layout = Layout(
            mesh_shape=x_layout.mesh_shape,
            alias_name=x_layout.alias_name,
            rank_list=x_layout.rank_list
        )
        output_layout.set_tensor_map((-1,))  # Output is 1D histogram with shape (bins,)

        has_sharding = any(
            alias is not None and alias != "None"
            for alias in x_layout.alias_tensor_map
        )

        if has_sharding:
            for alias, tensor_map_val in zip(x_layout.alias_name, x_layout.alias_tensor_map):
                if tensor_map_val is not None and tensor_map_val != "None":
                    output_layout.set_partial_by_dev_axis(alias, "sum")
        # pylint: disable=protected-access
        output_layout._alias_tensor_map = output_layout._build_readable_tensor_map()
        # pylint: disable=protected-access
        output_layout.tensor_map_to_placement()
        output_layout.update_compact_str()

        return output_layout
