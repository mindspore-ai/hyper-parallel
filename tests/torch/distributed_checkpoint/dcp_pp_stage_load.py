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
"""DCP load with broadcasting under pipeline parallelism; run via parallel_run or torchrun.

Every other broadcast case here loads a state dict that every rank holds the same keys of.
Pipeline parallelism breaks that: a stage holds the layers assigned to it and nothing else,
so ``embed`` exists on the first stage only, ``head`` on the last, and two stages have almost
no key in common. Everything the load coordinates has to survive that - the plans are
gathered across the world but hold different parameters, the replica groups come out per
stage, and a rank walks a shard order most of which belongs to someone else.

Two shapes run over the same eight ranks. ``pp=2 x dp=2 x tp=2`` gives each stage four ranks
and a two-dimensional sub-mesh, so a stage parameter can be replicated over the stage, split
down a tp column, or split every way. ``pp=4 x dp=2`` gives each stage two ranks, which is
what makes a parameter tied between the first and the last stage land on ranks 0, 1, 6 and 7
- a replica group that reaches across the pipeline and skips the middle of it.

Buffers are poisoned with a rank-specific sentinel before each load and every parameter
carries a value only its shard should hold, so a shard that never arrived, or one that
arrived out of another stage's broadcast, fails the comparison rather than passing on a
buffer that happened to be zero.
"""
# pylint: disable=C0413
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import shutil
import threading
from pathlib import Path
from typing import Any, NamedTuple, Union

import numpy as np
import torch

from hyper_parallel import DTensor
from hyper_parallel.core.distributed_checkpoint import load, save
from hyper_parallel.core.distributed_checkpoint.util import _MAX_BROADCASTS_IN_FLIGHT
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.platform import get_platform
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device

_WORLD_SIZE = 8
_LOCAL_SHAPE = (4, 4)


class _Shape(NamedTuple):
    """One way of laying the eight ranks out as a pipeline, and what a stage can shard over.

    ``placements`` is keyed by the kind of parameter and gives its placements on a stage's
    own sub-mesh, so every group they produce - bar a tied parameter's - sits inside a stage.
    """

    mesh_shape: tuple
    mesh_dim_names: tuple
    stage_dims: tuple
    placements: dict

    @property
    def stages(self) -> int:
        """How many pipeline stages this lays the world out as."""
        return self.mesh_shape[0]

    @property
    def ranks_per_stage(self) -> int:
        """How many ranks a stage holds."""
        return _WORLD_SIZE // self.stages

    @property
    def kinds(self) -> tuple:
        """The kinds of parameter a stage of this shape can hold."""
        return tuple(self.placements)


# Stage 0 = ranks 0-3, stage 1 = ranks 4-7; inside a stage +0=(dp0,tp0) +1=(dp0,tp1)
# +2=(dp1,tp0) +3=(dp1,tp1). Replicating both dims groups the stage's four ranks, sharding tp
# splits them into the tp columns (0, 2) and (1, 3) resp. (4, 6) and (5, 7), and sharding
# both leaves every rank a shard nobody else wants.
_PP2 = _Shape(
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("pp", "dp", "tp"),
    stage_dims=("dp", "tp"),
    placements={
        "stage": [Replicate(), Replicate()],
        "column": [Replicate(), Shard(1)],
        "private": [Shard(0), Shard(1)],
    },
)

# Stage 0 = ranks 0-1, stage 1 = 2-3, stage 2 = 4-5, stage 3 = 6-7. A stage's sub-mesh is its
# dp pair, so a stage parameter groups two ranks - and one tied between the first stage and
# the last groups (0, 1, 6, 7), which is the shape this file is here for.
_PP4 = _Shape(
    mesh_shape=(4, 2),
    mesh_dim_names=("pp", "dp"),
    stage_dims=("dp",),
    placements={
        "stage": [Replicate()],
        "private": [Shard(0)],
    },
)

# Which shard of a parameter a rank holds, from its position inside its stage. This decides
# the value it should end up with, and - for a tied parameter, which every rank of the stages
# holding it replicates - it is 0 everywhere, so those stages agree on what was saved.
_SHARD_OF = {
    "stage": lambda local_rank: 0,
    "column": lambda local_rank: local_rank % 2,
    "private": lambda local_rank: local_rank,
}

# Which stages hold a parameter: None for every one of them, an int for a single stage, or a
# tuple for a parameter tied between the ones named.
_EVERY_STAGE = None

# Enough replicated parameters per stage that a stage runs out of room and has to wait on a
# broadcast already going before it can start another, twice over.
_LAYERS_PER_STAGE = 2 * _MAX_BROADCASTS_IN_FLIGHT

# Batching gathers every shard under the threshold into one broadcast; at 0 each goes on its
# own. Both are run over the same parameters so a stage split is covered either way.
_BATCH_SETTINGS = (0, 6 * 1024 * 1024)


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


def _layer_specs(shape: _Shape) -> list[tuple]:
    """A model split across the stages the way pipeline parallelism splits one.

    The stages take contiguous runs of layers, and the pieces that live at the ends of a
    model - the embedding on the first stage, the head and final norm on the last - belong to
    one stage only. Sorting these names puts each stage's shards in a run of its own, so a
    stage works through every broadcast it has before the next stage reaches its first.
    """
    specs: list[tuple] = [("embed.weight", "stage", 0)]
    for index in range(_LAYERS_PER_STAGE * shape.stages):
        stage = index // _LAYERS_PER_STAGE
        for kind in shape.kinds:
            specs.append((f"layers.{index:02d}.{kind}", kind, stage))
    specs.append(("norm.weight", "stage", shape.stages - 1))
    specs.append(("head.weight", shape.kinds[-1], shape.stages - 1))
    return specs


def _interleaved_specs(shape: _Shape, count: int) -> list[tuple]:
    """Parameters named so the shard order alternates between the stages, shard by shard.

    A rank walks the whole global order and acts only on the shards it holds, so naming the
    parameters this way leaves it stepping over other stages' shards between every two of its
    own. Nothing lines the stages up against each other, and this is what would show it if
    something did.
    """
    kinds = shape.kinds
    return [
        (f"p{index:03d}.{kinds[index % len(kinds)]}",
         kinds[index % len(kinds)],
         index % shape.stages)
        for index in range(count)
    ]


def _lopsided_specs(shape: _Shape) -> list[tuple]:
    """One stage with replicated parameters only, the rest with nothing to broadcast at all.

    A rank enters group creation on a condition every rank agrees on rather than on whether
    it has a shard to broadcast, and this is the shape that tells the two apart: every stage
    but the first shards its parameters on all of its mesh dims, so no rank of them shares a
    shard with anyone and none has a broadcast of its own to make. They still have to take
    part in creating the groups the first stage broadcasts through.
    """
    specs: list[tuple] = [
        (f"front.{index:02d}.weight", "stage", 0) for index in range(_LAYERS_PER_STAGE)
    ]
    specs.extend(
        (f"back.{index:02d}.weight", "private", 1 + index % (shape.stages - 1))
        for index in range(_LAYERS_PER_STAGE)
    )
    return specs


def _tied_specs(shape: _Shape, tied_stages: Union[tuple, None]) -> list[tuple]:
    """Parameters the named stages share, alongside parameters each stage holds on its own.

    Tying an embedding to the output projection puts the same name on the first and the last
    stage. The shards match, so they form one replica group spanning those stages - the only
    group in any of these cases that reaches across the pipeline - and one rank of it reads
    for all the others. On four stages that group skips the middle two, which is a rank list
    no partition of the world contains.
    """
    specs: list[tuple] = [("tied.embedding", "stage", tied_stages)]
    for index in range(_LAYERS_PER_STAGE):
        specs.append((f"layers.{index:02d}.first", shape.kinds[0], index % shape.stages))
        specs.append((f"layers.{index:02d}.second", shape.kinds[-1], index % shape.stages))
    specs.append(("tied.head", "stage", tied_stages))
    return specs


def _held_by(stage: Any, pp_rank: int) -> bool:
    """Whether a rank on ``pp_rank`` holds this parameter.

    ``None`` means every stage does, a tuple means the ones it names, an int means that one.
    """
    if stage is None:
        return True
    if isinstance(stage, tuple):
        return pp_rank in stage
    return stage == pp_rank


def _local(index: int, kind: str, local_rank: int) -> torch.Tensor:
    """The shard this rank holds of parameter ``index``, filled with a value only it carries.

    The index runs over every parameter of the model rather than over the ones this stage
    holds, so a shard that arrived out of another stage's broadcast carries a value from
    somewhere else in the model and does not match.
    """
    return to_device(
        torch.full(_LOCAL_SHAPE, index + 1.0 + _SHARD_OF[kind](local_rank) / 8.0), _DEVICE_TYPE
    )


def _poisoned(rank: int) -> torch.Tensor:
    """A buffer filled with a rank-specific sentinel no saved value can match."""
    return to_device(torch.full(_LOCAL_SHAPE, -100.0 - rank), _DEVICE_TYPE)


def _build_state(
        specs: list[tuple], shape: _Shape, stage_mesh: Any, rank: int, pp_rank: int,
        poisoned: bool
) -> dict:
    """This stage's share of the parameters, holding either their values or the sentinel."""
    local_rank = rank % shape.ranks_per_stage
    state: dict[str, Any] = {}
    for index, (name, kind, stage) in enumerate(specs):
        if not _held_by(stage, pp_rank):
            continue
        local = _poisoned(rank) if poisoned else _local(index, kind, local_rank)
        state[name] = DTensor.from_local(local, stage_mesh, shape.placements[kind])
    return state


def _assert_stages_differ(specs: list[tuple], state: dict, pp_rank: int, rank: int) -> None:
    """The stages must really hold different parameters, or the case proves nothing.

    Every other broadcast case loads a state dict whose keys every rank shares. If a change
    to the fixture ever made this one do the same, the cases below would still pass and would
    no longer be about pipeline parallelism at all, so check the shape rather than trust it.
    """
    mine = {name for name, kind, stage in specs if _held_by(stage, pp_rank)}
    theirs = {name for name, kind, stage in specs if not _held_by(stage, pp_rank)}
    assert set(state) == mine, (
        f"rank{rank} was built a state dict of {len(state)} entries, "
        f"expected its stage's {len(mine)}"
    )
    assert theirs, (
        f"rank{rank} on stage {pp_rank} holds every parameter in the model, "
        f"so this is not a pipeline split"
    )


def _assert_arrived(loaded: dict, expected: dict, rank: int, scenario: str) -> None:
    """Every entry must hold what was saved, whether this rank read it or was sent it."""
    assert set(loaded) == set(expected), (
        f"[{scenario}] rank{rank} holds {len(loaded)} entries after load, "
        f"saved {len(expected)}"
    )
    for name, want in expected.items():
        got = loaded[name]
        got_local = got.to_local() if isinstance(got, DTensor) else got
        assert np.allclose(
            got_local.cpu().detach().numpy(), want.cpu().detach().numpy(), rtol=1e-5, atol=1e-5
        ), (
            f"[{scenario}] rank{rank} does not hold {name} after load: its broadcast never "
            f"arrived, another stage's landed in its place, or it wrote into a copy of the "
            f"entry rather than the entry itself"
        )


def _run_pp_load(
        scenario: str, checkpoint_name: str, shape: _Shape, specs: list[tuple], seed: int
) -> None:
    """Save this pipeline-split model and load it back through the broadcast pipeline.

    The load runs once per batching setting, into buffers poisoned again each time, so the
    stage split is covered both with the small shards gathered into one broadcast and with
    each going on its own.
    """
    platform, rank = _setup(seed)
    mesh = init_device_mesh(
        device_type=_DEVICE_TYPE, mesh_shape=shape.mesh_shape,
        mesh_dim_names=shape.mesh_dim_names,
    )
    stage_mesh = mesh[shape.stage_dims]
    pp_rank = mesh.get_local_rank("pp")
    checkpoint_path = _fresh_checkpoint_dir(platform, rank, checkpoint_name)

    saved = _build_state(specs, shape, stage_mesh, rank, pp_rank, poisoned=False)
    _assert_stages_differ(specs, saved, pp_rank, rank)
    expected = {name: value.to_local().clone() for name, value in saved.items()}
    save(saved, checkpoint_id=checkpoint_path, use_collectives=True)
    platform.barrier()

    threads_before = threading.active_count()
    for batch_bytes in _BATCH_SETTINGS:
        target = _build_state(specs, shape, stage_mesh, rank, pp_rank, poisoned=True)
        load(
            target,
            checkpoint_id=checkpoint_path,
            use_collectives=True,
            broadcast_replicated_tensors=True,
            broadcast_batch_bytes=batch_bytes,
        )
        where = f"{scenario}, batch_bytes={batch_bytes}"
        _assert_arrived(target, expected, rank, where)

    assert threading.active_count() <= threads_before, (
        f"[{scenario}] rank{rank} is running {threading.active_count()} threads after "
        f"{len(_BATCH_SETTINGS)} load(s), up from {threads_before}: a load is leaking a "
        f"thread"
    )
    platform.barrier()
    if rank == 0:
        shutil.rmtree(checkpoint_path, ignore_errors=True)


def test_dcp_load_pp_stages_hold_different_parameters() -> None:
    """
    Feature: broadcast load where each pipeline stage holds its own layers.
    Description: A model split across two stages the way pipeline parallelism splits one -
        contiguous runs of layers, the embedding on the first stage, the head and final norm
        on the last - each stage sharding its layers over its own (dp=2, tp=2) sub-mesh, and
        each holding more replicated parameters than the pipeline allows broadcasts in
        flight. Loaded into sentinel-filled buffers, with and without small-shard batching.
    Expectation: Run success, every rank holds its own stage's parameters and nothing else.
        The plans are gathered over the world but carry different parameters per stage, so
        this is where a load that assumed every rank plans the same shards would come apart;
        the replica groups land inside a stage, so the two stages broadcast through disjoint
        groups and neither waits on the other.
    """
    _run_pp_load(
        scenario="stages hold different layers",
        checkpoint_name="test_dcp_pp_layers",
        shape=_PP2,
        specs=_layer_specs(_PP2),
        seed=41,
    )


def test_dcp_load_pp_stage_without_replicated_shards() -> None:
    """
    Feature: broadcast load where one stage has nothing to broadcast.
    Description: The first stage's parameters are replicated across its four ranks; the
        second stage's are sharded on both its mesh dims, so no rank of it shares a shard
        with anyone. Loaded with broadcasting on.
    Expectation: Run success. Creating a communication group is collective over the world, so
        the second stage's ranks have to reach group creation even though they have no shard
        to send and end up in none of the groups being made. A load that entered that step
        only when it had something to broadcast would hang the first stage here - and this is
        not a contrived shape, it is what a stage of embeddings followed by a stage of
        fully-sharded layers looks like.
    """
    _run_pp_load(
        scenario="one stage with no replicated shard",
        checkpoint_name="test_dcp_pp_lopsided",
        shape=_PP2,
        specs=_lopsided_specs(_PP2),
        seed=43,
    )


def test_dcp_load_pp_stages_interleave_in_the_shard_order() -> None:
    """
    Feature: broadcast load where the stages alternate through the shard order.
    Description: Parameters named so that sorting them alternates between the stages shard by
        shard, cycling through a stage-replicated parameter, a tp column and a shard nobody
        shares. Every rank steps over one of the other stage's shards between every two of
        its own.
    Expectation: Run success, every rank holds its stage's shards. Each rank works the order
        out for itself from the gathered plans rather than agreeing on one, and acts only on
        the shards it holds; a rank that took the shards it skips as something to wait for -
        or that ordered its own groups by where it met them rather than by the global order -
        would hang both stages against each other here.
    """
    _run_pp_load(
        scenario="stages interleaved in the shard order",
        checkpoint_name="test_dcp_pp_interleaved",
        shape=_PP2,
        specs=_interleaved_specs(_PP2, 6 * _MAX_BROADCASTS_IN_FLIGHT),
        seed=47,
    )


def test_dcp_load_pp_tied_parameter_spans_both_stages() -> None:
    """
    Feature: broadcast load with a parameter both stages hold.
    Description: Two parameters saved under the same name on the first and the last stage, as
        a tied embedding is, replicated across each stage's sub-mesh, mixed with parameters
        each stage holds on its own. On two stages that is every rank in the world.
    Expectation: Run success, both stages hold the tied parameter and their own layers. Its
        shards match across the pipeline, so it forms one replica group of all eight ranks
        and one rank reads it for everybody. Being the whole world, that group is the one the
        job already runs on rather than one the load has to raise, so this is also where a
        load that rebuilt it - or worse, destroyed it afterwards - would show.
    """
    _run_pp_load(
        scenario="a parameter tied across both stages",
        checkpoint_name="test_dcp_pp_tied",
        shape=_PP2,
        specs=_tied_specs(_PP2, _EVERY_STAGE),
        seed=53,
    )


def test_dcp_load_pp4_stages_hold_different_parameters() -> None:
    """
    Feature: broadcast load across four pipeline stages.
    Description: The same layer split over ``pp=4 x dp=2``, so a stage is two ranks and its
        replicated parameters group that pair. Four stages rather than two, each with its own
        run of layers and its own broadcasts.
    Expectation: Run success, every rank holds its own stage's parameters. Four disjoint
        groups broadcast at once instead of two, and a stage sees three others' shards in the
        order it walks rather than one, so a load that let what another stage is doing bear
        on its own has more ways to show it here.
    """
    _run_pp_load(
        scenario="four stages hold different layers",
        checkpoint_name="test_dcp_pp4_layers",
        shape=_PP4,
        specs=_layer_specs(_PP4),
        seed=59,
    )


def test_dcp_load_pp4_tied_parameter_skips_the_middle_stages() -> None:
    """
    Feature: broadcast load with a parameter tied between two stages that are not neighbours.
    Description: Over ``pp=4 x dp=2``, a parameter saved under the same name on the first and
        the last stage, so its replica group is ranks 0, 1, 6 and 7 - the two middle stages
        hold nothing of it. Mixed with parameters each of the four stages holds on its own.
    Expectation: Run success, the first and last stage hold the tied parameter and every
        stage holds its own layers. That rank list is not one that any partition of the world
        contains: dividing eight ranks into fours gives (0, 1, 2, 3) and (4, 5, 6, 7), and
        into pairs gives neighbours, so a load that built its groups by expanding the rank
        list as a template would refuse this one outright. Tying an embedding to the output
        projection of a four-stage pipeline is what produces it.
    """
    _run_pp_load(
        scenario="a parameter tied across the ends of a four-stage pipeline",
        checkpoint_name="test_dcp_pp4_tied",
        shape=_PP4,
        specs=_tied_specs(_PP4, (0, 3)),
        seed=61,
    )
