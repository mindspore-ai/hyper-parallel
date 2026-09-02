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
"""DCP load through the read-and-broadcast pipeline; run via parallel_run or torchrun.

``execute_read`` does not read a plan through and then broadcast it. It walks the shards in
the one order every rank works out for itself, starts each broadcast the moment its shard is
in place instead of waiting for it to land, keeps only so many going at once, and reads a
batch ahead on a thread of its own. None of that is meant to change what a load produces,
which is why it needs cases that would notice if it did: more shards than broadcasts allowed
in flight, groups that interleave with each other and with shards nobody shares, and state
that is not a tensor at all.

Every buffer is poisoned with a rank-specific sentinel first, and every shard carries a value
only it should hold, so a shard that never arrived, that arrived out of some other broadcast,
or that was written into a copy of the state dict entry rather than the entry itself, fails
the comparison instead of passing on a zero-filled buffer.
"""
# pylint: disable=C0413
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import shutil
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hyper_parallel import DTensor
from hyper_parallel.core.distributed_checkpoint import load, save
from hyper_parallel.core.distributed_checkpoint.util import _MAX_BROADCASTS_IN_FLIGHT
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.platform import get_platform
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device

_WORLD_SIZE = 4
_MESH_SHAPE = (2, 2)
_MESH_DIM_NAMES = ("dp", "tp")
_LOCAL_SHAPE = (4, 4)

# Global rank layout of the (dp=2, tp=2) mesh:
#   rank0=(dp0,tp0)  rank1=(dp0,tp1)  rank2=(dp1,tp0)  rank3=(dp1,tp1)
# Replicating both dims puts every rank in one group; sharding tp splits the dp pairs into
# (0, 2) and (1, 3); sharding both leaves every rank a shard nobody else wants.
_PLACEMENTS = {
    "everyone": [Replicate(), Replicate()],
    "column": [Replicate(), Shard(1)],
    "private": [Shard(0), Shard(1)],
}

# Which of a parameter's shards a rank holds, which is what decides the value it should end
# up with: one shard for everyone, one per tp column, one per rank.
_SHARD_OF = {
    "everyone": lambda rank: 0,
    "column": lambda rank: rank % 2,
    "private": lambda rank: rank,
}

# Enough shared shards that the pipeline runs out of room and has to wait on a broadcast
# already going before it can start another, several times over. A load that fits inside the
# window never reaches that code at all.
_ROUNDS_OF_IN_FLIGHT = 3
_SHARED_PARAMS = _ROUNDS_OF_IN_FLIGHT * _MAX_BROADCASTS_IN_FLIGHT

# State that is not a tensor: saved as pickled bytes, read whole rather than sliced, and never
# broadcast, so a load has to carry it alongside everything the pipeline is doing.
_SCALARS = {
    "epoch": 12,
    "learning_rate": 0.0003,
    "step": 4500,
    "tags": ["warmup", "cosine"],
    "limits": {"grad_clip": 1.0, "patience": 3},
}
# The same shape holding wrong values: a load resolves nested state by flattened name, so
# a placeholder that dropped a list entry or renamed a key would ask the checkpoint for
# something it never saved, and fail there rather than on what the load produced.
_WRONG_SCALARS = {
    "epoch": -1,
    "learning_rate": -1.0,
    "step": -1,
    "tags": ["wrong", "wrong"],
    "limits": {"grad_clip": -1.0, "patience": -1},
}


def _setup(seed: int) -> tuple[Any, int]:
    """Initialize the backend and return the platform plus this rank."""
    init_backend(_DEVICE_TYPE)
    torch.manual_seed(seed)
    platform = get_platform()
    world_size = platform.get_world_size()
    assert world_size == _WORLD_SIZE, f"expect world_size={_WORLD_SIZE}, got {world_size}"
    return platform, platform.get_rank()


def _fresh_checkpoint_dir(platform: Any, rank: int, name: str) -> Path:
    """Return an empty checkpoint directory, agreed on by every rank."""
    checkpoint_path = Path(f"./{name}")
    if rank == 0 and checkpoint_path.exists():
        shutil.rmtree(checkpoint_path)
    platform.barrier()
    return checkpoint_path


def _shared_specs(count: int) -> list[tuple[str, str]]:
    """``count`` parameters every rank holds a copy of, so each one is a broadcast."""
    return [(f"shared{index:02d}", "everyone") for index in range(count)]


def _interleaved_specs(rounds: int) -> list[tuple[str, str]]:
    """
    Parameters whose names put the groups in each other's way.

    The pipeline walks shards in name order, so naming them this way makes a rank alternate
    between the group of all four and the group of its tp column, with a shard nobody shares
    dropped in between. Ranks that disagreed on that order would deadlock against each other
    rather than load anything.
    """
    specs: list[tuple[str, str]] = []
    for index in range(rounds):
        specs.append((f"round{index:02d}_a_everyone", "everyone"))
        specs.append((f"round{index:02d}_b_column", "column"))
        specs.append((f"round{index:02d}_c_private", "private"))
    return specs


def _local(index: int, kind: str, rank: int) -> torch.Tensor:
    """The shard this rank holds of parameter ``index``, filled with a value only it carries."""
    return to_device(
        torch.full(_LOCAL_SHAPE, index + 1.0 + _SHARD_OF[kind](rank) / 8.0), _DEVICE_TYPE
    )


def _poisoned(rank: int) -> torch.Tensor:
    """A buffer filled with a rank-specific sentinel no saved value can match."""
    return to_device(torch.full(_LOCAL_SHAPE, -100.0 - rank), _DEVICE_TYPE)


def _build_state(
        specs: list[tuple[str, str]], device_mesh: Any, rank: int, poisoned: bool
) -> dict:
    """The state dict for these parameters, holding either their values or the sentinel."""
    state: dict[str, Any] = {}
    for index, (name, kind) in enumerate(specs):
        local = _poisoned(rank) if poisoned else _local(index, kind, rank)
        state[name] = DTensor.from_local(local, device_mesh, _PLACEMENTS[kind])
    return state


def _assert_arrived(loaded: dict, expected: dict, rank: int, scenario: str) -> None:
    """Every entry must hold what was saved, whether this rank read it or was sent it."""
    for name, want in expected.items():
        got = loaded[name]
        if not isinstance(want, torch.Tensor):
            assert got == want, (
                f"[{scenario}] rank{rank} has {name}={got!r} after load, wanted {want!r}"
            )
            continue
        got_local = got.to_local() if isinstance(got, DTensor) else got
        assert np.allclose(
            got_local.cpu().detach().numpy(), want.cpu().detach().numpy(), rtol=1e-5, atol=1e-5
        ), (
            f"[{scenario}] rank{rank} does not hold {name} after load: its broadcast never "
            f"arrived, another one landed in its place, or it wrote into a copy of the entry"
        )


def _run_pipeline_load(
        scenario: str,
        checkpoint_name: str,
        specs: list[tuple[str, str]],
        seed: int,
        scalars: bool = False,
        loads: int = 1,
) -> None:
    """Save these parameters and load them back with the broadcast pipeline turned on."""
    platform, rank = _setup(seed)
    device_mesh = init_device_mesh(
        device_type=_DEVICE_TYPE, mesh_shape=_MESH_SHAPE, mesh_dim_names=_MESH_DIM_NAMES
    )
    checkpoint_path = _fresh_checkpoint_dir(platform, rank, checkpoint_name)

    saved = _build_state(specs, device_mesh, rank, poisoned=False)
    expected: dict[str, Any] = {
        name: value.to_local().clone() for name, value in saved.items()
    }
    if scalars:
        saved.update(_SCALARS)
        expected.update(_SCALARS)
    save(saved, checkpoint_id=checkpoint_path, use_collectives=True)
    platform.barrier()

    threads_before = threading.active_count()
    for attempt in range(loads):
        target = _build_state(specs, device_mesh, rank, poisoned=True)
        if scalars:
            target.update(_WRONG_SCALARS)
        load(
            target,
            checkpoint_id=checkpoint_path,
            use_collectives=True,
            broadcast_replicated_tensors=True,
        )
        _assert_arrived(target, expected, rank, f"{scenario}, load {attempt + 1}")

    assert threading.active_count() <= threads_before, (
        f"[{scenario}] rank{rank} is running {threading.active_count()} threads after "
        f"{loads} load(s), up from {threads_before}: a load is leaking a thread"
    )
    platform.barrier()
    if rank == 0:
        shutil.rmtree(checkpoint_path, ignore_errors=True)


def test_dcp_load_pipeline_more_shards_than_broadcasts_in_flight() -> None:
    """
    Feature: read and broadcast pipelined past the in-flight limit.
    Description: A state dict of several times as many replicated parameters as the pipeline
        allows broadcasts in flight, loaded into sentinel-filled buffers with broadcasting on.
        Each rank reads its share and receives the rest, so both sides run out of room.
    Expectation: Run success, every rank holds every parameter. Going past the limit is what
        makes the pipeline wait on the oldest broadcast to start the next, and what leaves
        several going at once while later shards are read; a load that stays under it never
        reaches either.
    """
    _run_pipeline_load(
        scenario="more shards than the in-flight limit",
        checkpoint_name="test_dcp_pipeline_in_flight",
        specs=_shared_specs(_SHARED_PARAMS),
        seed=17,
    )


def test_dcp_load_pipeline_interleaves_groups_and_private_shards() -> None:
    """
    Feature: pipelining across groups that interleave, with unshared shards mixed in.
    Description: Parameters named so that the order the pipeline walks them alternates
        between the group of all four ranks, the group of a rank's tp column, and a shard
        nobody else holds.
    Expectation: Run success, every rank holds its own shard of each. Every rank works the
        order out for itself from the gathered plans rather than agreeing on it, so a rank
        that ordered two groups differently from another would hang both of them here; the
        unshared shards check that leaving them to the end still loads them.
    """
    _run_pipeline_load(
        scenario="groups interleaved with private shards",
        checkpoint_name="test_dcp_pipeline_interleaved",
        specs=_interleaved_specs(_MAX_BROADCASTS_IN_FLIGHT + 2),
        seed=23,
    )


def test_dcp_load_pipeline_carries_non_tensor_state() -> None:
    """
    Feature: pickled state through a pipelined load.
    Description: A load mixing replicated and column-sharded parameters with entries that are
        not tensors at all - counters, a float, a list, a dict - started from wrong values.
    Expectation: Run success, the tensors and the pickled entries all come back. Bytes are
        read whole rather than sliced and are never broadcast, so they go down a different
        path through the read than every shard around them, and they are unpacked where the
        state dict is written rather than where the file is read.
    """
    _run_pipeline_load(
        scenario="non-tensor state alongside shards",
        checkpoint_name="test_dcp_pipeline_non_tensor",
        specs=_interleaved_specs(4),
        seed=29,
        scalars=True,
    )


def test_dcp_load_pipeline_repeats_cleanly() -> None:
    """
    Feature: a pipelined load run more than once in a process.
    Description: The same checkpoint loaded three times over into freshly poisoned buffers,
        with the thread count taken before and after.
    Expectation: Run success every time, and no more threads at the end than at the start.
        A load carries state that only lives as long as it does - the files it holds open,
        the broadcasts still going, the thread reading ahead - and a training job that loads
        more than once is the thing that would show any of it being left behind.

        The thread count catches a read thread that is still running or blocked when its load
        returns. It cannot catch a load that simply never waits for one, because a read thread
        that got through its work ends on its own; the case that pins the waiting is
        ``test_a_read_that_fails_on_its_own_thread_fails_the_load`` in the unit tests, where
        the read fails and the thread has to be seen out rather than left holding a batch.
    """
    _run_pipeline_load(
        scenario="repeated loads",
        checkpoint_name="test_dcp_pipeline_repeated",
        specs=_shared_specs(_MAX_BROADCASTS_IN_FLIGHT + 4),
        seed=31,
        loads=3,
    )
