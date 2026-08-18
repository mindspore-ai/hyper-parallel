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
"""DCP load with ``broadcast_from_minimum_rank``; run via parallel_run or torchrun.

Only the minimum rank of each same-shard group reads its shard from storage; every other
rank of the group receives it through a broadcast. Two paths reach that broadcast, and each
case below runs on both: the caller either pre-builds the communication groups, or the load
all-gathers the missing rank tuples and creates the groups on demand.

The load buffers are poisoned with a rank-specific sentinel before every load, so a shard
that is never broadcast - or one broadcast into a copy of the state dict entry rather than
into the entry itself - keeps the sentinel and fails the comparison, instead of passing
silently on a zero-filled buffer.
"""
# pylint: disable=C0413
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from hyper_parallel import DTensor
from hyper_parallel.core.distributed_checkpoint import load, save
from hyper_parallel.core.distributed_checkpoint.metadata import CHUNK_INFO, ChunkInfo, ChunkStorageMetadata
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.platform import get_platform
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device

_WORLD_SIZE = 4
_MESH_SHAPE = (2, 2)
_MESH_DIM_NAMES = ("dp", "tp")

# Global rank layout of the (dp=2, tp=2) mesh:
#   rank0=(dp0,tp0)  rank1=(dp0,tp1)  rank2=(dp1,tp0)  rank3=(dp1,tp1)
# so which ranks hold the same shard follows from the placements.
_PARAM_CONFIGS = (
    # Replicated on both mesh dims: a single group holding every rank, rank 0 reads.
    {"name": "replicated", "placements": [Replicate(), Replicate()], "local_shape": (6, 6)},
    # Sharded on tp only: the two dp ranks of a tp column hold the same shard,
    # so the groups are (0, 2) and (1, 3) with rank 0 resp. rank 1 reading.
    {"name": "tp_sharded", "placements": [Replicate(), Shard(1)], "local_shape": (8, 4)},
    # Sharded on both mesh dims: every shard is unique, nothing to broadcast.
    {"name": "fully_sharded", "placements": [Shard(0), Shard(1)], "local_shape": (6, 4)},
)

# Every same-shard group the configs above produce, plus the all-rank group the plain tensor
# needs; see _build_groups for which of them a given rank pre-builds.
_BROADCAST_GROUP_RANKS = ((0, 1, 2, 3), (0, 2), (1, 3))

# A plain (non-DTensor) tensor that an integration marks with CHUNK_INFO: replicated on
# every rank, so the same-shard group is the whole world and rank 0 is the one that reads.
_PLAIN_TENSOR_NAME = "plain_replicated"
_PLAIN_TENSOR_SHAPE = (8, 6)


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


def _poisoned(shape: tuple, rank: int) -> torch.Tensor:
    """Return a buffer filled with a rank-specific sentinel no saved value can match."""
    return to_device(torch.full(shape, -100.0 - rank), _DEVICE_TYPE)


def _mark_replicated_plain_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Annotate a plain tensor the way an integration that does not use DTensor does."""
    setattr(
        tensor,
        CHUNK_INFO,
        ChunkInfo(
            chunk=ChunkStorageMetadata(offsets=(0, 0), sizes=_PLAIN_TENSOR_SHAPE),
            global_shape=_PLAIN_TENSOR_SHAPE,
            replica_rank_list=tuple(range(_WORLD_SIZE)),
        ),
    )
    return tensor


def _build_save_state(device_mesh: Any) -> tuple[dict, dict]:
    """Build the state dict to save, keeping a copy of every local shard."""
    state_dict: dict[str, Any] = {}
    expected: dict[str, Any] = {}
    for config in _PARAM_CONFIGS:
        local = to_device(torch.randn(*config["local_shape"]), _DEVICE_TYPE)
        dtensor = DTensor.from_local(local, device_mesh, config["placements"])
        state_dict[config["name"]] = dtensor
        expected[config["name"]] = dtensor.to_local().clone()

    plain = to_device(torch.randn(*_PLAIN_TENSOR_SHAPE), _DEVICE_TYPE)
    state_dict[_PLAIN_TENSOR_NAME] = plain
    expected[_PLAIN_TENSOR_NAME] = plain.clone()
    return state_dict, expected


def _build_poisoned_load_state(device_mesh: Any, rank: int) -> dict:
    """Build the load target with every buffer set to a rank-specific sentinel."""
    load_state: dict[str, Any] = {}
    for config in _PARAM_CONFIGS:
        local = _poisoned(config["local_shape"], rank)
        load_state[config["name"]] = DTensor.from_local(local, device_mesh, config["placements"])
    load_state[_PLAIN_TENSOR_NAME] = _mark_replicated_plain_tensor(_poisoned(_PLAIN_TENSOR_SHAPE, rank))
    return load_state


def _assert_shards_arrived(load_state: dict, expected: dict, rank: int, scenario: str) -> None:
    """Every rank must hold its own shard, whether it read it or was sent it."""
    for name, want in expected.items():
        got = load_state[name]
        got_local = got.to_local() if isinstance(got, DTensor) else got
        assert np.allclose(
            got_local.cpu().detach().numpy(),
            want.cpu().detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
        ), (
            f"[{scenario}] rank{rank} does not hold {name} after load: the broadcast was "
            f"skipped, or it wrote into a copy instead of the state dict entry"
        )


def _build_groups(platform: Any, rank: int, prebuild_groups: bool) -> Optional[dict]:
    """Pre-build the same-shard groups this rank belongs to, or leave them to the load.

    ``create_group`` returns a handle only to members of the group, so each rank asks for its
    own groups. The creation is still collective: a rank tuple is a template that expands to
    the whole partition, so every rank issues the same underlying ``new_group`` sequence.
    """
    if not prebuild_groups:
        return None
    return {ranks: platform.create_group(ranks) for ranks in _BROADCAST_GROUP_RANKS if rank in ranks}


def _run_broadcast_load(scenario: str, checkpoint_name: str, seed: int, prebuild_groups: bool) -> None:
    """Save a (2, 2)-mesh state dict and load it back with broadcasting enabled."""
    platform, rank = _setup(seed)
    device_mesh = init_device_mesh(
        device_type=_DEVICE_TYPE, mesh_shape=_MESH_SHAPE, mesh_dim_names=_MESH_DIM_NAMES
    )
    checkpoint_path = _fresh_checkpoint_dir(platform, rank, checkpoint_name)

    state_dict, expected = _build_save_state(device_mesh)
    save(state_dict, checkpoint_id=checkpoint_path, use_collectives=True)
    platform.barrier()

    broadcast_groups = _build_groups(platform, rank, prebuild_groups)
    load_state = _build_poisoned_load_state(device_mesh, rank)
    load(
        load_state,
        checkpoint_id=checkpoint_path,
        use_collectives=True,
        broadcast_from_minimum_rank=True,
        broadcast_groups=broadcast_groups,
    )

    _assert_shards_arrived(load_state, expected, rank, scenario)
    platform.barrier()
    if rank == 0:
        shutil.rmtree(checkpoint_path, ignore_errors=True)


def _run_plain_tensor_broadcast_load(scenario: str, checkpoint_name: str, prebuild_groups: bool) -> None:
    """Save and load a state dict holding nothing but a CHUNK_INFO-marked plain tensor."""
    platform, rank = _setup(13)
    checkpoint_path = _fresh_checkpoint_dir(platform, rank, checkpoint_name)

    saved = to_device(torch.randn(*_PLAIN_TENSOR_SHAPE), _DEVICE_TYPE)
    save({_PLAIN_TENSOR_NAME: saved}, checkpoint_id=checkpoint_path, use_collectives=True)
    platform.barrier()

    broadcast_groups = _build_groups(platform, rank, prebuild_groups)
    buffer = _mark_replicated_plain_tensor(_poisoned(_PLAIN_TENSOR_SHAPE, rank))
    load(
        {_PLAIN_TENSOR_NAME: buffer},
        checkpoint_id=checkpoint_path,
        use_collectives=True,
        broadcast_from_minimum_rank=True,
        broadcast_groups=broadcast_groups,
    )

    _assert_shards_arrived({_PLAIN_TENSOR_NAME: buffer}, {_PLAIN_TENSOR_NAME: saved}, rank, scenario)
    platform.barrier()
    if rank == 0:
        shutil.rmtree(checkpoint_path, ignore_errors=True)


def test_dcp_load_broadcast_from_minimum_rank() -> None:
    """
    Feature: ``load(..., broadcast_from_minimum_rank=True)`` without pre-built groups.
    Description: Save a (2, 2)-mesh state dict mixing replicated, tp-sharded and fully sharded
        DTensors with a CHUNK_INFO-marked plain tensor, then load it into sentinel-filled
        buffers while only the minimum rank of each same-shard group reads from storage; the
        missing groups are all-gathered and created during the load.
    Expectation: Run success, every rank holds its own shard after the load.
    """
    _run_broadcast_load(
        scenario="groups created on demand",
        checkpoint_name="test_dcp_broadcast_from_minimum_rank",
        seed=7,
        prebuild_groups=False,
    )


def test_dcp_load_broadcast_with_prebuilt_groups() -> None:
    """
    Feature: ``load(..., broadcast_from_minimum_rank=True, broadcast_groups=...)``.
    Description: Same save/load as the on-demand case, but the caller pre-builds every
        same-shard communication group and hands them to ``load``.
    Expectation: Run success, every rank holds its own shard after the load.
    """
    _run_broadcast_load(
        scenario="caller supplied groups",
        checkpoint_name="test_dcp_broadcast_prebuilt_groups",
        seed=11,
        prebuild_groups=True,
    )


def test_dcp_load_broadcast_plain_tensor_with_chunk_info() -> None:
    """
    Feature: broadcast path for a state dict holding no DTensor at all.
    Description: Integrations that do not use DTensor mark a replicated plain tensor with
        ``ChunkInfo(replica_rank_list=...)``; with broadcasting on, only the minimum rank of
        that rank list reads the tensor and the others receive it. Run against both group
        paths, since each one broadcasts on its own.
    Expectation: Run success, the ranks that did not read still hold the saved tensor.
    """
    _run_plain_tensor_broadcast_load(
        scenario="plain tensor, groups created on demand",
        checkpoint_name="test_dcp_broadcast_plain_tensor_on_demand",
        prebuild_groups=False,
    )
    _run_plain_tensor_broadcast_load(
        scenario="plain tensor, caller supplied groups",
        checkpoint_name="test_dcp_broadcast_plain_tensor_prebuilt",
        prebuild_groups=True,
    )
