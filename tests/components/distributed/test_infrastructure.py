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
"""Tests for distributed topology derivation from Trainer configuration."""

from types import SimpleNamespace

import pytest

import hyper_models.components.distributed.infrastructure as infrastructure
from hyper_models._transformers.infrastructure import instantiate_infrastructure
from hyper_models.components.distributed.config import FSDP2Config
from hyper_models.components.distributed.fsdp2 import FSDP2Manager
from hyper_models.trainer.config import AcceleratorConfig


class _FakeDeviceMesh:
    def __init__(self, mesh_dim_names):
        self.mesh_dim_names = mesh_dim_names

    @staticmethod
    def get_local_rank(_dim_name: str) -> int:
        return 0


def _config(*, dp_shard_size: int, **accelerator_kwargs):
    return SimpleNamespace(
        accelerator=AcceleratorConfig(**accelerator_kwargs),
        fsdp_config=FSDP2Config(dp_shard_size=dp_shard_size),
    )


def _patch_distributed(monkeypatch, *, world_size: int, mesh_dim_names):
    monkeypatch.setattr(infrastructure.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        infrastructure.dist,
        "get_world_size",
        lambda: world_size,
    )
    monkeypatch.setattr(
        infrastructure,
        "_build_device_mesh_from_accelerator",
        lambda _accel, _shard, _replicate, _world: (
            _FakeDeviceMesh(mesh_dim_names),
            mesh_dim_names,
        ),
    )


def test_create_distributed_setup_derives_fsdp_dp_size(monkeypatch) -> None:
    _patch_distributed(
        monkeypatch,
        world_size=8,
        mesh_dim_names=("dp_shard", "tp"),
    )

    setup = infrastructure.create_distributed_setup_from_config(
        _config(dp_shard_size=4, tp_size=2)
    )

    assert setup.mesh_context.dp_size == 4
    assert setup.mesh_context.dp_replicate_size == 1
    assert setup.strategy_config.dp_shard_size == 4


def test_create_distributed_setup_derives_hsdp_replicate_size(monkeypatch) -> None:
    _patch_distributed(
        monkeypatch,
        world_size=16,
        mesh_dim_names=("dp_replicate", "dp_shard", "tp"),
    )

    setup = infrastructure.create_distributed_setup_from_config(
        _config(dp_shard_size=4, tp_size=2)
    )

    assert setup.mesh_context.dp_size == 8
    assert setup.mesh_context.dp_replicate_size == 2


def test_ep_size_uses_the_existing_dense_rank_domain(monkeypatch) -> None:
    _patch_distributed(
        monkeypatch,
        world_size=8,
        mesh_dim_names=("dp_replicate", "dp_shard", "tp"),
    )

    setup = infrastructure.create_distributed_setup_from_config(
        _config(dp_shard_size=2, tp_size=2, ep_size=4)
    )

    assert setup.mesh_context.dp_size == 4
    assert setup.mesh_context.dp_replicate_size == 2
    assert "ep" not in setup.mesh_context.device_mesh.mesh_dim_names


def test_instantiate_infrastructure_uses_distributed_strategy_config() -> None:
    strategy_config = FSDP2Config(dp_shard_size=2)
    setup = infrastructure.DistributedSetup(
        mesh_context=infrastructure.MeshContext(),
        strategy_config=strategy_config,
    )

    _, fsdp_manager, _ = instantiate_infrastructure(
        distributed_setup=setup,
    )

    assert isinstance(fsdp_manager, FSDP2Manager)
    assert fsdp_manager.config is strategy_config


def test_create_distributed_setup_rejects_non_dp_size_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(infrastructure.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(infrastructure.dist, "get_world_size", lambda: 10)

    with pytest.raises(ValueError, match="non-DP size"):
        infrastructure.create_distributed_setup_from_config(
            _config(dp_shard_size=1, tp_size=4)
        )


def test_create_distributed_setup_rejects_fsdp_shard_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(infrastructure.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(infrastructure.dist, "get_world_size", lambda: 12)

    with pytest.raises(ValueError, match="FSDP shard size"):
        infrastructure.create_distributed_setup_from_config(
            _config(dp_shard_size=4, tp_size=2)
        )
