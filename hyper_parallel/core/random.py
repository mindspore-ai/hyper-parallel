# Copyright (c) Meta Platforms, Inc. and affiliates
"""RNG state management for distributed tensor operations.

Provides utilities for tracking and synchronizing random number generator states
across multiple devices in distributed training scenarios.
"""
import contextlib
from logging import getLogger
import typing
from typing import Optional
import functools
import operator

from hyper_parallel.core.placement_types import Shard
from hyper_parallel.platform import get_platform

platform = get_platform()
DTensorBase = platform.DTensorBase
Tensor = platform.tensor

logger = getLogger(__name__)

__all__ = [
    "is_rng_supported_mesh",
    "OffsetBasedRNGTracker",
]

_rng_tracker: Optional["_RNGStateTracker"] = None


def is_rng_supported_mesh() -> bool:
    """Check if the device mesh supports DTensor random operations.

    Currently, DTensor random operations are only supported on CUDA and CUDA-like
    devices. Users should call this function before using DTensor random APIs to
    verify compatibility.

    Returns:
        bool: ``True`` if the device mesh supports DTensor random operations,
            ``False`` otherwise.
    """
    device_handle = platform.get_device_handle()
    if device_handle and hasattr(device_handle, "set_rng_state"):
        return True
    return False


class _PhiloxState:
    """
    Convenience accessor for interpreting the packed bits of (seed: uint64, offset: uint64) in the philox state,
    which for some reason is actually exposed as a size-16 uint8 tensor.

    The state is always moved to .cpu since it is necessary for it to be on CPU before applying it back to a generator.
    """

    def __init__(self, state: Tensor):
        self._state = state.to("cpu")

    @property
    def state(self):
        return self._state

    @property
    def offset(self) -> int:
        return int(self._state[8:].view(dtype=platform.tensor_dtype.int64).item())

    @offset.setter
    def offset(self, offset: int) -> None:
        offset_tensor = Tensor([offset], dtype=platform.tensor_dtype.uint64).view(
            platform.tensor_dtype.uint8
        ) # device?
        self._state[8:] = offset_tensor

    @property
    def seed(self) -> int:
        return int(self._state[:8].view(dtype=platform.tensor_dtype.uint64).item())

    @seed.setter
    def seed(self, seed: int) -> None:
        seed_tensor = Tensor([seed], dtype=platform.tensor_dtype.uint64).view(
            platform.tensor_dtype.uint8
        )# device
        self._state[:8] = seed_tensor


class _RNGStateTracker:
    """
    Tracks and manages RNG states for DTensor random operations.

    Maintains a mapping from operation tags to RNG state tensors (ByteTensor),
    providing standardized interfaces for state access and modification.

    The core method `_distribute_region` establishes the proper RNG context
    when DTensor executes random operators across distributed devices.
    """

    def __init__(self, device):
        self._device = device
        self._device_handle = platform.get_device_handle()
        if not self._device_handle:
            raise RuntimeError(
                f"{self.__class__.__name__} instantiation requires the presence of "
            )
        self._use_distribute_region = True

    @property
    def distribute_region_enabled(self) -> bool:
        return self._use_distribute_region

    @distribute_region_enabled.setter
    def distribute_region_enabled(self, value) -> None:
        self._use_distribute_region = value

    def _distribute_region(
        self, device_mesh, placements, global_shape, generator = None
    ):
        pass

    def _manual_seed(self, parallel_seed: int) -> None:
        pass


class OffsetBasedRNGTracker(_RNGStateTracker):
    """
    This subclass of ``_RNGStateTracker`` defines the default policy of how RNG states
    should be shared and synchronized among all ranks to respect the semantics of DTensor
    random operators.
    """

    def __init__(
        self,
        run_state_sync: bool = True,
    ):
        super().__init__(_resolve_device())
        rng_state = self._get_device_state()
        if run_state_sync:
            # synchronize RNG state using rank 0's current one
            platform.broadcast(rng_state, 0)
            my_rng_state = self._get_device_state()
            if not all(my_rng_state == rng_state):
                logger.warning(
                    "DTensor is synchronizing RNG states of every rank with the state from rank 0. "
                    "This behavior is deprecated. "
                    "Please call `manual_seed()` on every rank that participates in SPMD DTensor Operations with "
                    "the same seed. If using Pipeline Parallelism, each pipelining state would use a different seed, "
                    "but all ranks belonging to one pipeline stage would use the same seed."
                )
            self._set_device_state(rng_state)

    def _get_device_state(self):
        rng_state = self._device_handle.get_rng_state().to(self._device)
        return rng_state

    def _set_device_state(self, state: Tensor):
        # It seems that the underlying generator wants a cpu tensor but the dtensor code expects `_get_device_state`
        # to convert to a 'device' tensor, probably because we may use it with our backend comms for sync/debug
        # for now, we just convert back to cpu here to make sure it always works.
        self._device_handle.set_rng_state(state.to("cpu"))

    @contextlib.contextmanager
    def _distribute_region(
        self, device_mesh, placements, global_shape, generator = None
    ):

        # regular (non-LocalTensor) mode
        if generator is not None:
            # This is a little hacky, but for any user-passed generator, we store its state under a unique key,
            # not because we need to keep a copy of it but because its the easiest way to make it work with the
            # existing set/get APIs. We also ensure we remove it from rng_states after each _distribute_region.
            state = _PhiloxState(generator.get_state())
        else:
            state = _PhiloxState(self._get_device_state())

        if self.distribute_region_enabled:
            old_offset = state.offset
            self._set_pre_op_offset(state, device_mesh, placements, global_shape)
            with fork_rng(
                devices=[self._device], device_type=platform.device_type()
            ):
                self._device_handle.set_rng_state(state.state)
                try:
                    yield  # execute the region code
                finally:
                    # update offset to synchronize among ranks
                    self._set_post_op_offset(state, device_mesh, old_offset)

        else:
            yield

        if generator is not None:
            # ensure we (a) propagate the state advancement back to the user's RNG so its visible and impacts any future
            # usage of that RNG (dtensor or non-dtensor), (b) drop it from our own cache so that if the user updates
            # the seed value in their rng and uses it with DTensor again, we always use the latest value
            generator.set_state(state.state)
        else:
            self._set_device_state(state.state)

    def _set_pre_op_offset(self, state: _PhiloxState, device_mesh, placements, global_shape) -> None:
        """Set the starting random number generator (RNG) offset for the local shard
        on the current process before operation execution.The offset value begins from
        the current accumulated position and increments by the local shard size until
        covering the total elements of the global distributed tensor. Multiple processes
        holding replicas of the same shard will share identical starting offset values.

        Args:
            state (`Tensor`): The generator state to modify
            device_mesh (DeviceMesh): The device mesh describing the device topology.
            placements (Sequence[Placement]): The placement strategy for each mesh dimension.
            Each element should be a Placement object (Shard, Replicate, Partial, etc.).
            global_shape: input global shape

        Returns:
            None

        .. warning::
            The current implementation does not consider memory layout contiguity.

        Example:
            take a DTensor of shape [8, 16] as an example. Assume that the DTensor
            is placed on a device mesh with placements ([Shard(1), Replicate(), Shard(0)]),
            and the mesh is:
                [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
            ``mesh.get_coordinate()`` provides the coordinate of the current rank
            in the mesh. For example, the coordinate of rank 5 is (1, 0, 1).

            Another concept to introduce besides rank coordinate is shard coordinate.
            Each rank holds a local shard of the DTensor. In the example, the DTensor
            is partitioned into 4 [4, 8] shards. The first shard has 2 replicas and
            rank 0 (coord (0, 0, 0)) and rank 2 (coord (0, 1, 0)) have 1 replica each.
            That being said, the local shard on rank 0 and rank 2 correspond to the same
            shard of the DTensor. To denote each DTensor shard, we use a shard coordinate
            (in the example, it will be a tuple (i, j) where shard (i, j) has the slice
            DTensor[4 * i : 4 * (i + 1), 8 * j : 8 * (j + 1)], 0 <= i < 2, 0 <= j < 2).

            Once we have rank coordinate and shard coordinate, we can calculate on each rank
            what shard of the DTensor the rank holds, with the help of dim_map. The dim_map
            of the above DTensor is [2, 0] so the shard coordinate of a rank with rank coord
            (x, y, z) is simply (z, x) by taking(rank_coord[dim_map[0]],rank_coord[dim_map[1]]).
            Following this calculation,
            rank 0 and rank 2 holds the shard of coord (0, 0);
            rank 1 and rank 3 holds the shard of coord (0, 1);
            rank 4 and rank 6 holds the shard of coord (1, 0);
            rank 5 and rank 7 holds the shard of coord (1, 1);

            The last value to calculate before obtaining the starting offset is the shard linear index.
            The starting offset for each rank will be its shard_linear_index * local_tensor_numel.
        """
        mesh = device_mesh
        mesh_coordinate = mesh.get_coordinate()

        # Compute shard index and total number of shards on each tensor dim
        shard_idx_by_dim, total_num_shards_by_dim = _calc_shard_info(
            mesh_coordinate, device_mesh, placements
        )

        # compute shard linear index
        shard_linear_idx = self._calc_shard_linear_idx(
            shard_idx_by_dim, total_num_shards_by_dim
        )

        # compute starting offset using the first shard's size
        local_size_on_rank_0 = _calc_first_shard_size(device_mesh, placements, global_shape)

        local_size = functools.reduce(operator.mul, local_size_on_rank_0, 1)

        # get current RNG offset
        current_offset = state.offset

        offset_incr = (shard_linear_idx * local_size + 3) // 4 * 4
        state.offset = current_offset + offset_incr

    def _set_post_op_offset(
        self, state: _PhiloxState, device_mesh, old_offset: int
    ) -> None:
        """Sets the RNG to a synchronized state after running the local random op.
        Restores the random number generator to a globally consistent state following
        local shard execution. Each process must advance its offset by the total element
        count of the distributed tensor, measured from the offset value recorded before
        the operation began.

        Args:
            state (`Tensor`): The generator state to modify.
            device_mesh (DeviceMesh): The device mesh describing the device topology.

        Returns:
            None
        """
        dtensor_shape = device_mesh.mesh_shape

        numel = functools.reduce(operator.mul, dtensor_shape, 1)
        numel = (numel + 3) // 4 * 4
        state.offset = old_offset + numel

    def _calc_shard_linear_idx(
        self, shard_coord: list[int], shard_size: list[int]
    ) -> int:
        return _calc_shard_linear_idx(shard_coord, shard_size)


def _calc_first_shard_size(device_mesh, placements, global_shape) -> list[int]:
    """Calculate the size of the first shard on rank 0.

        Args:
            device_mesh: The device mesh describing the device topology.
            placements: Sequence of Placement objects (Shard, Replicate, etc.).
            global_shape: input global shape

        Returns:
            list[int]: Shape of rank 0's local shard.
        """
    local_size_on_rank_0 = list(global_shape)
    for idx, placement in enumerate(placements):
        if isinstance(placement, Shard):
            mesh_dim_size = device_mesh.size(idx)
            shard_dim = placement.dim
            local_size_on_rank_0[shard_dim], _ = local_shard_size_and_offset(
                device_mesh.mesh_shape[shard_dim],
                mesh_dim_size,
                0,
            )
    return local_size_on_rank_0


def _calc_shard_info(
    mesh_coordinate, device_mesh, placements
):
    """Calculate shard information for a specific rank."""
    mesh_size = device_mesh.mesh_shape
    # note: dim_map does not allow double sharding which is the FSDP(fully_shard)+TP
    # case. Replace the custom logic with dim_map once we support it.
    dim_map = [-1] * device_mesh.ndim
    for i, placement in enumerate(placements):
        if isinstance(placement, Shard):
            shard_dim = placement.dim
            if dim_map[shard_dim] == -1:
                dim_map[shard_dim] = [i]
            else:
                mesh_dim_list = dim_map[shard_dim]
                assert isinstance(mesh_dim_list, list)
                mesh_dim_list.append(i)

    # Compute shard coordinate:
    # The coordinate on each tensor dim is a tuple (idx, range)
    # If a DTensor is partitioned on its dim i into n shards, and the current rank
    # holds the j-th, then its shard coordinate will be (idx=j, range=n) on dim i
    assert mesh_coordinate is not None
    shard_idx_by_dim = []
    total_num_shards_by_dim = []  # total number of shards on each tensor dim
    for mesh_dim in dim_map:
        shard_idx = 0
        total_num_shards = 1
        # the tensor dim is sharded on more than 1 mesh dim
        if isinstance(mesh_dim, list):
            rank_coord = [mesh_coordinate[d] for d in mesh_dim]
            num_shards = [mesh_size[d] for d in mesh_dim]
            # compute the shard idx and total number of shards
            for idx, size in zip(rank_coord, num_shards):
                shard_idx = shard_idx * size + idx
                total_num_shards *= size

        shard_idx_by_dim.append(shard_idx)
        total_num_shards_by_dim.append(total_num_shards)
    return shard_idx_by_dim, total_num_shards_by_dim


def _calc_shard_linear_idx(shard_coord: list[int], shard_size: list[int]) -> int:
    # compute shard linear index
    shard_linear_idx = 0
    shard_coord_stride = 1
    for idx, size in zip(reversed(shard_coord), reversed(shard_size)):
        shard_linear_idx += idx * shard_coord_stride
        shard_coord_stride *= size

    return shard_linear_idx


def _resolve_device():
    device_handle = platform.get_device_handle()
    device_idx = platform.get_rank() % platform.device_count(device_handle)

    def get_device(device_idx):
        return platform.device(device_idx)

    return get_device(device_idx)

def local_shard_size_and_offset(
    curr_local_size: int,
    num_chunks: int,
    rank,
):
    """
    Given the size of the current local tensor (which may already be sharded on some dimensions),
    computes the new local shard size and offset given the desired number of chunks
    (num_chunks is generally equal to the size of the current sharding dim).

    Note: new local shard offset is relative to the current sharded tensor, not the global tensor.
    See `_utils.compute_local_shape_and_global_offset` for computing global offset.

    Returns (new local shard size, offset)

    """
    # Compute the chunk size inline
    if curr_local_size % num_chunks == 0:
        full_chunk_size = curr_local_size // num_chunks
        shard_starting_idx = full_chunk_size * rank
        return full_chunk_size, shard_starting_idx

    # uneven sharding case
    full_chunk_size = (curr_local_size + num_chunks - 1) // num_chunks
    shard_starting_idx = full_chunk_size * rank

    if curr_local_size < shard_starting_idx:
        return 0, typing.cast(int, curr_local_size)
    local_shard_size = (
            min(curr_local_size, shard_starting_idx + full_chunk_size)
            - shard_starting_idx
        )
    return local_shard_size, shard_starting_idx


_fork_rng_warned_already = False

@contextlib.contextmanager
def fork_rng(
    devices=None,
    enabled=True,
    device_type="npu",
):
    """
    Forks the RNG, so that when you return, the RNG is reset
    to the state that it was previously in.

    Args:
        devices (iterable of Device IDs): devices for which to fork
            the RNG. CPU RNG state is always forked. By default, :meth:`fork_rng` operates
            on all devices, but will emit a warning if your machine has a lot
            of devices, since this function will run very slowly in that case.
            If you explicitly specify devices, this warning will be suppressed
        enabled (bool): if ``False``, the RNG is not forked.  This is a convenience
            argument for easily disabling the context manager without having
            to delete it and unindent your Python code under it.
        device_type (str): device type str, default is `npu`. As for supported device,
            see details in :ref:`accelerator<accelerators>`
    """

    device_mod = platform.get_device_handle()
    if device_mod is None:
        raise RuntimeError(
            f"{platform} has no module of `{device_type}`, you should register "
        )
    global _fork_rng_warned_already

    if not enabled:
        yield
        return

    if devices is None:
        num_devices = platform.device_count(device_mod)
        if num_devices > 1 and not _fork_rng_warned_already:
            _fork_rng_warned_already = True
        devices = list(range(num_devices))
    else:
        # Protect against user passing us a generator; we need to traverse this
        # multiple times but a generator will be exhausted upon first traversal
        devices = list(devices)

    cpu_rng_state = platform.get_rng_state()
    device_rng_states = [platform.get_rng_state(device, device_mod) for device in devices]

    try:
        yield
    finally:
        platform.set_rng_state(cpu_rng_state)
        for device, device_rng_state in zip(devices, device_rng_states):
            platform.set_rng_state(device_rng_state, device, device_mod)
