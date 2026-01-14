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
"""Placement types for tensor placement on DeviceMesh."""
from typing import Optional


class Placement:
    """
    Abstract base class representing the strategy for placing a tensor on a DeviceMesh.
    Acts as a superclass for specific strategies like Shard, Replicate, and Partial.
    """
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def is_shard(self, dim: Optional[int] = None) -> bool:
        """
        Determine if this placement instance represents a Sharding strategy.
        
        Args:
            dim (int, optional): The dimension to check against.
        """
        # pylint: disable=W0613
        return False

    def is_replicate(self) -> bool:
        """
        Determine if this placement instance represents a Replication strategy.
        """
        return False

    def is_partial(self, reduce_op: Optional[str] = None) -> bool:
        """
        Determine if this placement instance represents a Partial strategy.

        Args:
            reduce_op (str, optional): The reduction operation to check against.
        """
        # pylint: disable=W0613
        return False


class Shard(Placement):
    """
    Placement strategy indicating that the tensor is split along a specific dimension.
    """
    def __init__(self, dim: int):
        super().__init__()
        self._dim = dim

    @property
    def dim(self) -> int:
        """Get the dimension along which the tensor is sharded."""
        return self._dim

    def is_shard(self, dim: Optional[int] = None) -> bool:
        if dim is None:
            return True
        return self._dim == dim

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Shard) and self.dim == other.dim

    def __hash__(self) -> int:
        return self._dim

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name}(dim={self.dim})"

    def __str__(self) -> str:
        return f"S({self._dim})"


class Replicate(Placement):
    """
    Placement strategy indicating that the tensor is fully replicated across devices.
    """
    def is_replicate(self) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return 0

    def __str__(self) -> str:
        return "R"


class Partial(Placement):
    """
    Placement strategy indicating that the tensor exists in a partial state,
    requiring a reduction operation to synchronize.
    """
    def __init__(self, reduce_op: str = "sum"):
        super().__init__()
        self._reduce_op = reduce_op

    @property
    def reduce_op(self) -> str:
        """Get the reduction operation type."""
        return self._reduce_op

    def is_partial(self, reduce_op: Optional[str] = None) -> bool:
        if reduce_op is None:
            return True
        return self._reduce_op == reduce_op

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Partial):
            return False
        return self._reduce_op == other.reduce_op

    def __hash__(self) -> int:
        return hash((self._reduce_op, 'Partial'))

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name}(reduce_op='{self._reduce_op}')"

    def __str__(self) -> str:
        return "P"
