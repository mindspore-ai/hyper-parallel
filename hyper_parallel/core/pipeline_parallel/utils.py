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
"""pipeline parallel utils"""


class BatchDimSpec:
    """
    Specify the batch dimension of a Tensor.

    Args:
        batch_dim (int): batch dimension.
    """
    __slots__ = ("batch_dim",)

    def __init__(self, batch_dim):
        if not isinstance(batch_dim, int):
            raise TypeError(f"batch_dim must be int, but got type {type(batch_dim)}.")
        self.batch_dim = batch_dim

    def __repr__(self):
        return f"BatchDimSpec({self.batch_dim})"

    def __str__(self):
        return f"BatchDim(dim={self.batch_dim})"

    @staticmethod
    def from_tuple(batch_dims):
        if not isinstance(batch_dims, tuple):
            raise TypeError(f"batch_dims must be tuple, but got type {type(batch_dims)}.")
        return tuple(BatchDimSpec(dim) for dim in batch_dims)

    @staticmethod
    def from_dict(batch_dims):
        if not isinstance(batch_dims, dict):
            raise TypeError(f"batch_dims must be dict, but got type {type(batch_dims)}.")
        return {k: BatchDimSpec(v) for k, v in batch_dims.items()}


class _RecvInfo:
    """
    Used for construct forward Receive operation and backward Send operation.
    """

    def __init__(self, global_rank, buffer=None):
        self._global_rank = global_rank
        self._buffer = buffer

    @property
    def global_rank(self):
        return self._global_rank

    @property
    def buffer(self):
        return self._buffer

    @buffer.setter
    def buffer(self, val):
        self._buffer = val
