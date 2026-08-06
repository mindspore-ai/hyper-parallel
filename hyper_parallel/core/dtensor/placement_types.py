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

    def is_ragged_shard(self) -> bool:
        """Return whether this placement represents non-uniform contiguous sharding."""
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
        return type(self) is type(other) and self.dim == other.dim

    def __hash__(self) -> int:
        return self._dim

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name}(dim={self.dim})"

    def __str__(self) -> str:
        return f"S({self._dim})"


class StridedShard(Shard):
    """
    Placement strategy indicating that the tensor is sharded on a dimension whose
    right-side mesh dimensions have already split the same tensor dimension.
    """

    def __init__(self, dim: int, split_factor: int):
        super().__init__(dim)
        if split_factor < 1:
            raise ValueError(f"split_factor must be positive, but got {split_factor}")
        self._split_factor = split_factor

    @property
    def split_factor(self) -> int:
        """Get the split factor contributed by right-side shard dimensions."""
        return self._split_factor

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, StridedShard)
            and self.dim == other.dim
            and self.split_factor == other.split_factor
        )

    def __hash__(self) -> int:
        return hash((self._dim, self._split_factor, "StridedShard"))

    def __repr__(self) -> str:
        return f"StridedShard(dim={self.dim}, split_factor={self.split_factor})"

    def __str__(self) -> str:
        return f"SS({self._dim}, {self._split_factor})"


class RaggedShard(Placement):
    """Placement for non-uniform sharding of a contiguous tensor prefix.

    Args:
        dims: Non-empty prefix dimensions covered by the placement.
        local_units: Relative non-negative allocation for ranks on one mesh axis.

    Raises:
        ValueError: If either tuple violates the phase-one RaggedShard contract.
    """

    def __init__(self, dims: tuple[int, ...], local_units: tuple[int, ...]) -> None:
        """Initialize one validated phase-one RaggedShard placement."""
        super().__init__()
        if not isinstance(dims, tuple) or not dims:
            raise ValueError(f"RaggedShard dims must be a non-empty tuple, got {dims!r}")
        if any(not isinstance(dim, int) or isinstance(dim, bool) for dim in dims):
            raise ValueError(f"RaggedShard dims must contain only integers, got dims={dims!r}")
        if dims != tuple(range(len(dims))):
            raise ValueError(
                f"RaggedShard dims must be prefix dims in phase 1, got dims={dims!r}"
            )
        if not isinstance(local_units, tuple) or not local_units:
            raise ValueError(
                f"RaggedShard local_units must be a non-empty tuple, got {local_units!r}"
            )
        if any(not isinstance(unit, int) or isinstance(unit, bool) for unit in local_units):
            raise ValueError(
                f"RaggedShard local_units must contain only integers, got local_units={local_units!r}"
            )
        if any(unit < 0 for unit in local_units):
            raise ValueError(
                f"RaggedShard local_units must be non-negative, got local_units={local_units!r}"
            )
        if sum(local_units) <= 0:
            raise ValueError(
                f"RaggedShard local_units must have a positive sum, got local_units={local_units!r}"
            )
        self._dims = dims
        self._local_units = local_units

    @property
    def dims(self) -> tuple[int, ...]:
        """Return the prefix dimensions covered by this placement."""
        return self._dims

    @property
    def local_units(self) -> tuple[int, ...]:
        """Return the relative allocation for ranks on the ragged mesh axis."""
        return self._local_units

    def is_ragged_shard(self) -> bool:
        """Return true for RaggedShard placements."""
        return True

    def __eq__(self, other: object) -> bool:
        """Compare RaggedShard values by concrete type, dims, and units."""
        return (
            type(self) is type(other)
            and self.dims == other.dims
            and self.local_units == other.local_units
        )

    def __hash__(self) -> int:
        """Return a hash that isolates different dims and local units."""
        return hash((self._dims, self._local_units, "RaggedShard"))

    def __repr__(self) -> str:
        """Return an unambiguous constructor-style representation."""
        return f"RaggedShard(dims={self.dims!r}, local_units={self.local_units!r})"

    def __str__(self) -> str:
        """Return a compact human-readable representation."""
        return f"RS({self.dims}, {self.local_units})"


class Replicate(Placement):
    """
    Placement strategy indicating that the tensor is fully replicated across devices.
    """
    def is_replicate(self) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return -1

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
