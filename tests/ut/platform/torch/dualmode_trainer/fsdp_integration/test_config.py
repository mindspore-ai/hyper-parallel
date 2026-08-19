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
"""Tests for top-level fsdp_config resolution and setup propagation."""

import pytest

from hyper_models.components.distributed.config import FSDP2Config
import hyper_models.components.distributed.infrastructure as infrastructure_module
from hyper_models.components.distributed.infrastructure import (
    _build_device_mesh_from_accelerator,
    create_distributed_setup_from_config,
)
from hyper_models.config.resolver import resolve_component
from hyper_models.trainer.config import AcceleratorConfig
from tests.common.mark_utils import arg_mark


class _FakeWorldMesh:
    """Minimal named mesh used by setup propagation tests."""

    def __init__(
        self,
        mesh_shape: tuple[int, ...] = (2, 1, 2),
        mesh_dim_names: tuple[str, ...] = ("dp", "cp", "tp"),
    ) -> None:
        """Store the topology recorded by the fake mesh builder."""
        self.mesh_shape = mesh_shape
        self.mesh_dim_names = mesh_dim_names

    @staticmethod
    def get_local_rank(dim_name: str) -> int:
        """Return rank zero for every fake mesh dimension."""
        del dim_name
        return 0


def test_fsdp_config_resolves_as_top_level_dataclass_without_target() -> None:
    """Resolve the upstream top-level YAML mapping into FSDP2Config."""
    accelerator_config = resolve_component(
        {
            "dp_shard_size": 4,
            "edp_shard_size": 2,
            "reshard_after_forward": False,
            "forward_prefetch_depth": 2,
            "backward_prefetch_depth": 3,
        },
        expected_type=FSDP2Config,
        path="$.fsdp_config",
    )

    assert isinstance(accelerator_config, FSDP2Config)
    assert accelerator_config.dp_shard_size == 4
    assert accelerator_config.edp_shard_size == 2
    assert accelerator_config.forward_prefetch_depth == 2
    assert accelerator_config.backward_prefetch_depth == 3


def test_setup_reuses_resolved_fsdp_config(monkeypatch) -> None:
    """Propagate the YAML-created config instead of constructing a default."""
    fsdp_config = FSDP2Config(dp_shard_size=1)
    accelerator_config = AcceleratorConfig(tp_size=2)
    trainer_config = type(
        "TrainerConfig",
        (),
        {"accelerator": accelerator_config, "fsdp_config": fsdp_config},
    )()

    monkeypatch.setattr(infrastructure_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(infrastructure_module.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(
        infrastructure_module,
        "_build_device_mesh_from_accelerator",
        lambda accelerator, dp_shard_size, dp_replicate_size, world_size, edp_shard_size: (
            infrastructure_module.MeshContext(
                device_mesh=_FakeWorldMesh(),
                dp_size=world_size // accelerator.tp_size,
                dp_replicate_size=dp_replicate_size,
                dp_shard_size=dp_shard_size,
                edp_shard_size=edp_shard_size,
                tp_size=accelerator.tp_size,
            ),
            ("dp", "cp", "tp"),
        ),
    )

    distributed_setup = create_distributed_setup_from_config(trainer_config)

    assert distributed_setup.strategy_config is fsdp_config
    assert distributed_setup.mesh_context.dp_replicate_size == 2
    assert distributed_setup.mesh_context.dp_shard_size == 1
    assert distributed_setup.mesh_context.edp_shard_size == 1


def test_mesh_builder_keeps_size_one_shard_axis_for_hsdp(monkeypatch) -> None:
    """Keep the explicit FSDP shard axis when its degree equals one."""
    accelerator_config = AcceleratorConfig(
        tp_size=2,
    )
    mesh_arguments = []

    def _init_device_mesh(
        *,
        device_type: str,
        mesh_shape: tuple[int, ...],
        mesh_dim_names: tuple[str, ...],
    ) -> _FakeWorldMesh:
        """Record mesh construction arguments."""
        mesh_arguments.append(
            {
                "device_type": device_type,
                "mesh_shape": mesh_shape,
                "mesh_dim_names": mesh_dim_names,
            }
        )
        return _FakeWorldMesh(mesh_shape, mesh_dim_names)

    monkeypatch.setattr(infrastructure_module, "init_device_mesh", _init_device_mesh)
    monkeypatch.setattr(infrastructure_module, "get_device_type", lambda: "cpu")

    _, mesh_dim_names = _build_device_mesh_from_accelerator(
        accelerator_config,
        dp_shard_size=1,
        dp_replicate_size=2,
        world_size=4,
    )

    assert [arguments["mesh_shape"] for arguments in mesh_arguments] == [
        (2, 1, 2),
        (2, 1, 2),
    ]
    assert mesh_dim_names == ("dp", "cp", "tp")


def test_mesh_builder_reshapes_dp_cp_domain_for_fsdp(monkeypatch) -> None:
    """Build basic (2,2,2) and FSDP (1,4,2) meshes from eight ranks."""
    accelerator_config = AcceleratorConfig(tp_size=2, cp_size=2)
    mesh_arguments = []

    def _init_device_mesh(
        *,
        device_type: str,
        mesh_shape: tuple[int, ...],
        mesh_dim_names: tuple[str, ...],
    ) -> _FakeWorldMesh:
        """Record basic and dense FSDP mesh construction."""
        mesh_arguments.append((device_type, mesh_shape, mesh_dim_names))
        return _FakeWorldMesh(mesh_shape, mesh_dim_names)

    monkeypatch.setattr(infrastructure_module, "init_device_mesh", _init_device_mesh)
    monkeypatch.setattr(infrastructure_module, "get_device_type", lambda: "cpu")

    mesh_context, _ = _build_device_mesh_from_accelerator(
        accelerator_config,
        dp_shard_size=4,
        dp_replicate_size=1,
        world_size=8,
    )

    assert mesh_arguments == [
        ("cpu", (2, 2, 2), ("dp", "cp", "tp")),
        ("cpu", (1, 4, 2), ("fsdp_replicate", "fsdp_shard", "tp")),
    ]
    assert mesh_context.dp_size == 2
    assert mesh_context.dp_replicate_size == 1


def test_mesh_builder_derives_expert_fsdp_mesh_from_edp_shard_size(monkeypatch) -> None:
    """Build (edp_replicate=2, edp_shard=2, ep=2) over eight ranks."""
    accelerator_config = AcceleratorConfig(tp_size=2, ep_size=2)
    mesh_arguments = []

    def _init_device_mesh(
        *,
        device_type: str,
        mesh_shape: tuple[int, ...],
        mesh_dim_names: tuple[str, ...],
    ) -> _FakeWorldMesh:
        """Record basic, dense FSDP, and expert FSDP mesh construction."""
        mesh_arguments.append((device_type, mesh_shape, mesh_dim_names))
        return _FakeWorldMesh(mesh_shape, mesh_dim_names)

    monkeypatch.setattr(infrastructure_module, "init_device_mesh", _init_device_mesh)
    monkeypatch.setattr(infrastructure_module, "get_device_type", lambda: "cpu")

    mesh_context, _ = _build_device_mesh_from_accelerator(
        accelerator_config,
        dp_shard_size=2,
        dp_replicate_size=2,
        world_size=8,
        edp_shard_size=2,
    )

    assert mesh_arguments == [
        ("cpu", (4, 1, 2), ("dp", "cp", "tp")),
        ("cpu", (2, 2, 2), ("fsdp_replicate", "fsdp_shard", "tp")),
        ("cpu", (2, 2, 2), ("edp_replicate", "edp_shard", "ep")),
    ]
    assert mesh_context.edp_shard_size == 2


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_mesh_builder_keeps_extended_ep_distinct_from_tp(monkeypatch) -> None:
    """
    Feature: Independent TP and TP-extend-EP mesh construction.
    Description: Build an eight-rank topology with tp_size=2 and ep_size=4.
    Expectation: The device mesh keeps TP size two and the expert mesh exposes EP size four.
    """
    accelerator_config = AcceleratorConfig(tp_size=2, ep_size=4)
    mesh_arguments = []

    def _init_device_mesh(
        *,
        device_type: str,
        mesh_shape: tuple[int, ...],
        mesh_dim_names: tuple[str, ...],
    ) -> _FakeWorldMesh:
        """Record the independent device, dense-FSDP, and expert-FSDP domains."""
        mesh_arguments.append((device_type, mesh_shape, mesh_dim_names))
        return _FakeWorldMesh(mesh_shape, mesh_dim_names)

    monkeypatch.setattr(infrastructure_module, "init_device_mesh", _init_device_mesh)
    monkeypatch.setattr(infrastructure_module, "get_device_type", lambda: "cpu")

    mesh_context, _ = _build_device_mesh_from_accelerator(
        accelerator_config,
        dp_shard_size=2,
        dp_replicate_size=2,
        world_size=8,
        edp_shard_size=2,
    )

    assert mesh_arguments == [
        ("cpu", (4, 1, 2), ("dp", "cp", "tp")),
        ("cpu", (2, 2, 2), ("fsdp_replicate", "fsdp_shard", "tp")),
        ("cpu", (2, 4), ("edp_shard", "ep")),
    ]
    assert mesh_context.device_mesh.mesh_shape[-1] == 2
    assert mesh_context.fsdp_moe_mesh.mesh_shape[-1] == 4


def test_fsdp_config_rejects_invalid_edp_shard_size() -> None:
    """Reject an expert FSDP shard size smaller than one."""
    with pytest.raises(
        ValueError,
        match="edp_shard_size must be greater than or equal to 1",
    ):
        FSDP2Config(edp_shard_size=0)
