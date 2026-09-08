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
"""DCP async_save + load API tests (fully_shard + multi-parameter model)."""
# pylint: disable=C0413
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
import torch_npu  # pylint: disable=W0611

from hyper_parallel import init_device_mesh
from hyper_parallel.core.distributed_checkpoint import async_save, load as dcp_load
from hyper_parallel.core.distributed_checkpoint.metadata import Metadata
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from tests.torch.utils import init_dist

_IN = 24
_HIDDEN = 48
_BLOCKS = 3
_OUT = 16

# Written into the live tensors right after async_save returns, to prove the
# checkpoint holds the staged copy rather than whatever the model holds later.
_MUTATION = 1000.0

_MP = MixedPrecisionPolicy(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    output_dtype=torch.float32,
    cast_forward_inputs=True,
)


class FsdpAsyncDemoNet(nn.Module):
    """
    Multi-parameter MLP-style stack for async DCP: stem, repeated block (Linear + LayerNorm + GELU), output head.
    """

    def __init__(self, in_features: int, hidden: int, num_blocks: int, out_features: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_features, hidden)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
            )
            for _ in range(num_blocks)
        )
        self.out_proj = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.in_proj(x))
        for block in self.blocks:
            x = x + block(x)
        return self.out_proj(x)


def _build_sharded_demo_model(world_size: int) -> FsdpAsyncDemoNet:
    """Apply ``fully_shard`` on a 1-D mesh (ZeRO-3 style) over ``world_size`` ranks."""
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(world_size,),
        mesh_dim_names=("dp",),
    )
    model = FsdpAsyncDemoNet(_IN, _HIDDEN, _BLOCKS, _OUT).npu()
    for block in model.blocks:
        fully_shard(
            block,
            mesh=mesh,
            reshard_after_forward=True,
            mp_policy=_MP,
        )
    fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=True,
        mp_policy=_MP,
    )
    return model


def _local_of(value: Any) -> torch.Tensor:
    """Return the rank-local tensor behind a state dict entry."""
    return value.to_local() if isinstance(value, DTensor) else value


def _snapshot(state_dict: dict) -> dict:
    """Copy every local shard, to compare against after the live tensors were mutated."""
    return {key: _local_of(value).clone() for key, value in state_dict.items()}


def _mutate_in_place(state_dict: dict) -> None:
    """Overwrite the live tensors the way the next training step would.

    ``async_save`` stages synchronously, so once it returns the caller owns its tensors again;
    what reaches the checkpoint must be the staged copy, not whatever the tensors hold later.
    """
    with torch.no_grad():
        for value in state_dict.values():
            _local_of(value).add_(_MUTATION)


def _fresh_ckpt_dir(rank: int, tag: str) -> Path:
    """Return an empty checkpoint directory, agreed on by every rank.

    The ranks that coordinate through the storage poll for each other's files, so the name has
    to be identical on every rank - hence MASTER_PORT, which torchrun gives them all, and not
    the pid, which differs per rank.
    """
    ckpt_dir = Path(f"tmp_async_dcp_{tag}_{os.environ.get('MASTER_PORT', '0')}")
    if rank == 0 and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    dist.barrier()
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    return ckpt_dir


def _run_async_save_load(tag: str, use_collectives: bool, use_gloo: bool) -> None:
    """Async-save a fully_shard model, mutate it, then load the checkpoint back and compare.

    Args:
        tag (str): Suffix for the checkpoint directory of this scenario.
        use_collectives (bool): Whether the persist child coordinates with the other ranks.
        use_gloo (bool): Whether that coordination runs on a gloo group the child reinitialises
            (the alternative is exchanging plans and write results through the storage).
    """
    init_dist()
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 4, f"expect world_size=4, got {world}"

    model = _build_sharded_demo_model(world)
    save_data = {"model_state_dict": model.state_dict()}
    expected = _snapshot(save_data["model_state_dict"])
    ckpt_dir = _fresh_ckpt_dir(rank, tag)

    callback_calls = []
    resp = async_save(
        save_data,
        checkpoint_id=ckpt_dir,
        use_collectives=use_collectives,
        use_gloo=use_gloo,
        callback=lambda: callback_calls.append(1),
    )
    # Staging already finished, so the training loop is free to write into its tensors again.
    _mutate_in_place(save_data["model_state_dict"])

    persist_meta = resp.get_result()
    assert isinstance(persist_meta, Metadata), f"[{tag}] async_save did not return Metadata"
    assert callback_calls == [1], f"[{tag}] callback ran {len(callback_calls)} times, expected once"
    assert resp.persist_completion.done(), f"[{tag}] persist future is not done after get_result"
    dist.barrier()

    model_for_load = _build_sharded_demo_model(world)
    load_data = {"model_state_dict": model_for_load.state_dict()}
    dcp_load(load_data, checkpoint_id=ckpt_dir, use_collectives=use_collectives)
    dist.barrier()

    for key, want in expected.items():
        got = _local_of(load_data["model_state_dict"][key])
        assert torch.allclose(got, want), (
            f"[{tag}] rank{rank} {key} mismatch after load: the checkpoint holds the mutated "
            f"tensors instead of the staged copy, or the persist step lost data"
        )

    dist.barrier()
    if rank == 0:
        shutil.rmtree(ckpt_dir, ignore_errors=True)


def test_dcp_async_save_load():
    """
    Feature: ``async_save`` without cross-rank coordination (``use_collectives=False``).
    Description: Async-save a ``fully_shard`` model whose tensors are overwritten as soon as
        the call returns, then load the checkpoint into a fresh model.
    Expectation: Run success, the checkpoint holds the staged values and the callback ran once.
    """
    _run_async_save_load("no_comm", use_collectives=False, use_gloo=False)


def test_dcp_async_save_load_with_storage_comm():
    """
    Feature: ``async_save`` coordinating through the storage (``use_collectives=True``).
    Description: Same scenario, but the persist child exchanges local plans and write results
        with the other ranks as files instead of over a process group.
    Expectation: Run success, the checkpoint holds the staged values and the callback ran once.
    """
    _run_async_save_load("storage_comm", use_collectives=True, use_gloo=False)


def test_dcp_async_save_load_with_gloo_comm():
    """
    Feature: ``async_save`` coordinating over gloo (``use_gloo=True``).
    Description: Same scenario, but the persist child reinitialises a gloo process group from
        MASTER_ADDR / MASTER_PORT and runs the plan collectives on it.
    Expectation: Run success, the checkpoint holds the staged values and the callback ran once.
    """
    _run_async_save_load("gloo_comm", use_collectives=True, use_gloo=True)


def test_dcp_async_save_twice_reuses_the_plan_cache():
    """
    Feature: back-to-back ``async_save`` calls.
    Description: Persist the same model twice; the child hands its plan cache back to the
        parent through the result queue, so the second save starts from a warm cache.
    Expectation: Run success, both saves return Metadata and the second checkpoint loads back.
    """
    _run_async_save_load("twice_first", use_collectives=True, use_gloo=False)
    _run_async_save_load("twice_second", use_collectives=True, use_gloo=False)
