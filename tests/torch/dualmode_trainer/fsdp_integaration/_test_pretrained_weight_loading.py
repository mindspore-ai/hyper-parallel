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
"""Distributed NPU worker for finalized-layout pretrained weight loading."""

import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

import hyper_parallel.auto_models._transformers.checkpoint_loader as checkpoint_loader
from hyper_parallel import DTensor, Shard, distribute_tensor, init_device_mesh
from tests.torch.utils import init_dist


class _ShardedModel(nn.Module):
    """Expose both DTensor and explicit _sharding_spec targets."""

    def __init__(self, dtensor: DTensor) -> None:
        super().__init__()
        self.dt_weight = nn.Parameter(dtensor)
        self.spec_weight = nn.Parameter(torch.zeros_like(dtensor.to_local()))
        self.spec_weight._sharding_spec = dtensor.layout  # pylint: disable=W0212

    @staticmethod
    def save_pretrained(save_directory: Path, *, state_dict: dict, **kwargs) -> None:
        """Minimal Transformers save contract used by the manager integration test."""
        del kwargs
        save_directory.mkdir(parents=True, exist_ok=True)
        save_file(state_dict, str(save_directory / "model.safetensors"))


def test_pretrained_weights_follow_final_layout_npu() -> None:
    """
    Feature: Final-layout checkpoint loading on two NPU ranks.
    Description: Load one logical tensor into DTensor and _sharding_spec targets.
    Expectation: Parameter identity is preserved and each rank receives its row shard.
    """
    rank, device_id = init_dist()
    mesh = init_device_mesh(
        "npu",
        (dist.get_world_size(),),
        mesh_dim_names=("tp",),
    )
    full_tensor = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    sharded_tensor = distribute_tensor(
        full_tensor.npu(device_id),
        mesh,
        [Shard(0)],
    )
    model = _ShardedModel(sharded_tensor)
    dt_parameter = model.dt_weight
    spec_parameter = model.spec_weight

    checkpoint_dir = Path("/tmp") / (
        f"hyper_parallel_pretrained_load_{os.environ['MASTER_PORT']}"
    )
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "dt_weight": full_tensor,
                "spec_weight": full_tensor + 10,
            },
            str(checkpoint_dir / "model.safetensors"),
        )
    dist.barrier()
    checkpoint_loader.get_model_conversion_mapping = lambda *_, **__: []

    checkpoint_loader.load_pretrained_weights(model, str(checkpoint_dir))

    expected = full_tensor.chunk(dist.get_world_size(), dim=0)[rank].npu(device_id)
    assert model.dt_weight is dt_parameter
    assert model.spec_weight is spec_parameter
    assert torch.equal(model.dt_weight.to_local(), expected)
    assert torch.equal(model.spec_weight, expected + 10)

    export_dir = checkpoint_dir / "export"
    manager = checkpoint_loader.CheckpointManager(model)
    wrote_checkpoint = manager.save_pretrained(export_dir)
    assert wrote_checkpoint == (rank == 0)
    dist.barrier()
    if rank == 0:
        with safe_open(
            str(export_dir / "model.safetensors"),
            framework="pt",
            device="cpu",
        ) as exported:
            assert torch.equal(exported.get_tensor("dt_weight"), full_tensor)
            assert torch.equal(exported.get_tensor("spec_weight"), full_tensor + 10)
