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
Distributed implementation for Slice operator.
"""
# pylint: disable=E0402
from .parallel_ops import DistributedOp


class SliceDistributedOp(DistributedOp):
    """Distributed implementation for MatMul operator."""

    def _is_shard_dim(self, layout):
        """return the shard num in each dim"""
        shard_dim = []
        dev_mat = layout.mesh_shape
        tensor_map = layout.tensor_map
        for dev_idx in tensor_map:
            if isinstance(dev_idx, (tuple, list)):
                shard_num = 1
                for idx in dev_idx:
                    if idx != -1 and dev_mat[len(dev_mat) - idx - 1] != 1:
                        shard_num *= dev_mat[len(dev_mat) - idx - 1]
                shard_dim.append(shard_num)
            else:
                if dev_idx != -1 and dev_mat[len(dev_mat) - dev_idx - 1] != 1:
                    shard_dim.append(dev_mat[len(dev_mat) - dev_idx - 1])
                else:
                    shard_dim.append(1)
        return shard_dim

    def _check_layout(self, layout, begin, end, shape):
        """check whether layout is valid"""
        if len(layout) != 1:
            raise ValueError(f"Layout must be a tuple of length 1, but got {len(layout)}")
        layout = layout[0]
        shard_dim = self._is_shard_dim(layout)
        for i, _ in enumerate(begin):
            if (shard_dim[i] != 1 and end[i] - begin[i] != shape[i]) and shape[i] != -1:
                raise ValueError(
                    f"Slice: When a dimension({i}) is not fully fetched, the dimension can not be split now, "
                    f"the begin is {begin}, the end is {end}, the shape is {shape}, layout is {layout.to_dict()}")
        return shard_dim

    def infer_layout(self, layouts, extra_args):
        """
        Infer output layout for slice operator. The shard dim must be fully fetched.

        Args:
            layouts (Layout): Layout of input x
            extra_args: (begin, end, global shape)

        Returns:
            layout (Layout): the out layout
            new_begin (tuple): begin after modification
            new_end (tuple): end after modification
        """
        begin = extra_args[0]
        end = extra_args[1]
        global_shape = extra_args[2]
        shard_dim = self._check_layout(layouts, begin, end, global_shape)
        new_begin = tuple(begin[i] // shard_dim[i] for i in range(len(begin)))
        new_end = tuple(end[i] // shard_dim[i] for i in range(len(end)))
        return layouts[0], new_begin, new_end
