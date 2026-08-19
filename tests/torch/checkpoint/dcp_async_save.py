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


def _sd_values_close(a: torch.Tensor | DTensor, b: torch.Tensor | DTensor) -> None:
    if isinstance(a, DTensor) and isinstance(b, DTensor):
        assert torch.allclose(a.to_local(), b.to_local()), "DTensor local shard mismatch after load"
    else:
        assert torch.allclose(a, b), "tensor mismatch after load"


def test_dcp_async_save_load():
    """
    Feature: ``async_save`` + ``load`` with ``fully_shard`` (1-D mesh) and a multi-parameter model.

    Staging runs in the training process; persistence and load use rank-local metadata.
    """
    init_dist()
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 4, f"expect world_size=4, got {world}"

    model = _build_sharded_demo_model(world)
    save_data = {"model_state_dict": model.state_dict()}

    pid = os.getpid()
    ckpt_dir = Path(f"tmp_async_dcp_save_and_load_{pid}")
    if rank == 0 and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    dist.barrier()
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    resp = async_save(save_data, checkpoint_id=ckpt_dir, use_collectives=False)
    persist_meta = resp.persist_completion.result()
    assert isinstance(persist_meta, Metadata)
    dist.barrier()

    model_for_load = _build_sharded_demo_model(world)
    load_data = {"model_state_dict": model_for_load.state_dict()}
    dcp_load(load_data, checkpoint_id=ckpt_dir, use_collectives=False)
    dist.barrier()

    for key in save_data["model_state_dict"]:
        _sd_values_close(save_data["model_state_dict"][key], load_data["model_state_dict"][key])

    dist.barrier()
    if rank == 0:
        shutil.rmtree(ckpt_dir)
