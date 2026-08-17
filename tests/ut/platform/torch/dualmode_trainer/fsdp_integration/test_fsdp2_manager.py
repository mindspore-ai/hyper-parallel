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
"""Tests for dual-mode FSDP2Manager wrapping and metadata distribution."""

# These unit tests intentionally validate private orchestration helpers.
# pylint: disable=protected-access

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import Mock

import pytest
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel import Replicate, Shard
import hyper_models.components.distributed.fsdp2 as fsdp2_module
from hyper_models.components.distributed.config import FSDP2Config
from hyper_models.components.distributed.fsdp2 import FSDP2Manager
from hyper_models.components.distributed.infrastructure import MeshContext


@dataclass(frozen=True)
class _FakeSourceShardMetaInfo:
    """Mirror the core metadata contract supplied by the FSDP core PR."""

    mesh: object
    placements: tuple
    origin_is_dtensor: bool = False


class _FakeMesh:
    """Record DeviceMesh slicing and flattening without distributed setup."""

    def __init__(self, mesh_dim_names: tuple[str, ...]) -> None:
        """Initialize a fake mesh with named dimensions."""
        self.mesh_dim_names = tuple(mesh_dim_names)

    def __getitem__(self, mesh_dim_names: str | tuple[str, ...]) -> _FakeMesh:
        """Return a fake named submesh."""
        if isinstance(mesh_dim_names, str):
            mesh_dim_names = (mesh_dim_names,)
        return _FakeMesh(mesh_dim_names)

    def flatten(self, mesh_dim_name: str) -> _FakeMesh:
        """Return a fake flattened mesh."""
        return _FakeMesh((mesh_dim_name,))


class _FakeDeviceMesh:
    """Provide the concatenate operation used by FSDP2Manager."""

    @staticmethod
    def concatenate(meshes: list[_FakeMesh]) -> _FakeMesh:
        """Concatenate fake mesh dimensions in input order."""
        return _FakeMesh(
            tuple(
                dim_name
                for mesh in meshes
                for dim_name in mesh.mesh_dim_names
            )
        )


class _FakeBlock(nn.Module):  # pylint: disable=abstract-method
    """Small transformer block for wrap-policy tests."""

    def __init__(self) -> None:
        """Initialize one projection parameter."""
        super().__init__()
        self.gradient_checkpointing = False
        self.proj = nn.Linear(4, 4, bias=False)


class _FakeBackbone(nn.Module):  # pylint: disable=abstract-method
    """HF-like container marked for gradient checkpointing."""

    def __init__(self, num_layers: int = 3) -> None:
        """Initialize an HF-like transformer block container."""
        super().__init__()
        self.gradient_checkpointing = False
        self.embed_tokens = nn.Embedding(8, 4)
        self.layers = nn.ModuleList(_FakeBlock() for _ in range(num_layers))


class _FakeCausalLM(nn.Module):  # pylint: disable=abstract-method
    """HF-like causal LM with root-owned embedding and output parameters."""

    def __init__(self, num_layers: int = 3, tied: bool = False) -> None:
        """Initialize an HF-like causal language model."""
        super().__init__()
        self.model = _FakeBackbone(num_layers)
        self.lm_head = nn.Linear(4, 8, bias=False)
        if tied:
            self.lm_head.weight = self.model.embed_tokens.weight


@pytest.fixture(name="core_metadata")
def _core_metadata(monkeypatch: pytest.MonkeyPatch) -> type[_FakeSourceShardMetaInfo]:
    """Install the SourceShardMetaInfo supplied by the dependent core change."""
    monkeypatch.setattr(
        fsdp2_module.fully_shard_utils,
        "SourceShardMetaInfo",
        _FakeSourceShardMetaInfo,
        raising=False,
    )
    return _FakeSourceShardMetaInfo


def _make_manager(**config_changes) -> FSDP2Manager:
    """Build a manager with the fake dense topology used by these tests."""
    config = FSDP2Config(
        dp_shard_size=2,
        enable_fsdp2_prefetch=True,
        **config_changes,
    )
    mesh_context = MeshContext(
        device_mesh=_FakeMesh(("dp_replicate", "dp_shard", "cp", "tp")),
        dp_replicate_size=2,
        dp_shard_size=2,
        tp_size=2,
        cp_size=2,
    )
    return FSDP2Manager(config, mesh_context)


def test_actual_mesh_flattens_shard_and_cp_then_keeps_replicate(monkeypatch) -> None:
    """Build a 2-D HSDP mesh with replicate then flattened shard axes."""
    monkeypatch.setattr(fsdp2_module, "DeviceMesh", _FakeDeviceMesh)
    manager = _make_manager()

    fsdp_actual_mesh = manager._build_fsdp_actual_mesh()

    assert fsdp_actual_mesh.mesh_dim_names == ("dp_replicate", "fsdp_shard")


def test_actual_expert_mesh_uses_edp_replicate_and_shard_axes() -> None:
    """Select the EDP axes from the prebuilt expert mesh for HSDP wrapping."""
    manager = _make_manager()
    manager.mesh_context.fsdp_moe_mesh = _FakeMesh(
        ("edp_replicate", "edp_shard", "ep")
    )

    fsdp_actual_mesh = manager._build_fsdp_actual_mesh(expert=True)

    assert fsdp_actual_mesh.mesh_dim_names == ("edp_replicate", "edp_shard")


def test_parallelize_rejects_tied_layout_conflict(
    core_metadata,
) -> None:
    """Resolve FQNs internally and reject conflicting tied-alias layouts."""
    assert core_metadata is _FakeSourceShardMetaInfo
    manager = _make_manager()
    model = _FakeCausalLM(tied=True)
    tp_mesh = manager.mesh_context.device_mesh["tp"]

    with pytest.raises(ValueError, match="conflicting source layouts"):
        manager.parallelize(
            model,
            {
                "model.embed_tokens.weight": ((Shard(0),), tp_mesh),
                "lm_head.weight": ((Replicate(),), tp_mesh),
            },
        )


def test_tied_parameter_is_owned_by_root() -> None:
    """Keep a parameter aliased outside transformer blocks on the root state."""
    manager = _make_manager()
    model = _FakeCausalLM(tied=True)
    wrap_modules = manager._find_wrap_modules(model)

    owner_by_parameter = manager._resolve_parameter_owners(model, wrap_modules)

    assert owner_by_parameter[model.model.embed_tokens.weight] is model


def test_parallelize_distributes_metadata_and_configures_prefetch(
    monkeypatch,
    core_metadata,
) -> None:
    """Wrap child blocks then root with owned metadata and configured depths."""
    assert core_metadata is _FakeSourceShardMetaInfo
    monkeypatch.setattr(fsdp2_module, "DeviceMesh", _FakeDeviceMesh)
    manager = _make_manager(
        reshard_after_forward=True,
        forward_prefetch_depth=2,
        backward_prefetch_depth=2,
    )
    model = _FakeCausalLM(num_layers=3)
    tp_mesh = manager.mesh_context.device_mesh["tp"]
    source_shard_info = {"model.layers.0.proj.weight": ((Shard(0),), tp_mesh)}
    fully_shard_calls = []

    def _fake_fully_shard(module: nn.Module, **kwargs: object) -> nn.Module:
        """Record one fully_shard call and install prefetch spies."""
        module.set_modules_to_forward_prefetch = Mock()
        module.set_modules_to_backward_prefetch = Mock()
        fully_shard_calls.append((module, kwargs))
        return module

    monkeypatch.setattr(fsdp2_module, "fully_shard", _fake_fully_shard)

    result = manager.parallelize(
        model,
        source_shard_info,
        compile_hooks_enabled=True,
    )

    assert result is model
    assert [call[0] for call in fully_shard_calls] == [
        *model.model.layers,
        model,
    ]
    for layer, kwargs in fully_shard_calls[:-1]:
        assert kwargs["reshard_after_forward"] is True
        assert kwargs["compile_hooks_enabled"] is True
        assert set(kwargs["source_shard_infos"]) == set(layer.parameters())
    assert fully_shard_calls[-1][1]["reshard_after_forward"] is False
    assert fully_shard_calls[-1][1]["compile_hooks_enabled"] is True
    assert set(fully_shard_calls[-1][1]["source_shard_infos"]) == {
        model.model.embed_tokens.weight,
        model.lm_head.weight,
    }

    first_layer, second_layer, third_layer = model.model.layers
    first_layer.set_modules_to_forward_prefetch.assert_called_once_with(
        [second_layer, third_layer]
    )
    second_layer.set_modules_to_forward_prefetch.assert_called_once_with(
        [third_layer]
    )
    third_layer.set_modules_to_backward_prefetch.assert_called_once_with(
        [second_layer, first_layer]
    )

    first_layer_metadata = fully_shard_calls[0][1]["source_shard_infos"]
    assert first_layer_metadata[first_layer.proj.weight].placements == (Shard(0),)
    assert first_layer_metadata[first_layer.proj.weight].origin_is_dtensor is False
    second_layer_metadata = fully_shard_calls[1][1]["source_shard_infos"]
    assert second_layer_metadata[second_layer.proj.weight].placements == (Replicate(),)


def test_parallelize_resolves_replicate_parameter_fqns(
    monkeypatch,
    core_metadata,
) -> None:
    """Resolve configured replicate FQNs inside the root parallelize call."""
    assert core_metadata is _FakeSourceShardMetaInfo
    monkeypatch.setattr(fsdp2_module, "DeviceMesh", _FakeDeviceMesh)
    manager = _make_manager(replicate_params=["model.embed_tokens.weight"])
    model = _FakeCausalLM(num_layers=1)
    fully_shard_calls = []

    def _fake_fully_shard(module: nn.Module, **kwargs: object) -> nn.Module:
        """Record each fully_shard invocation."""
        module.set_modules_to_forward_prefetch = Mock()
        module.set_modules_to_backward_prefetch = Mock()
        fully_shard_calls.append((module, kwargs))
        return module

    monkeypatch.setattr(fsdp2_module, "fully_shard", _fake_fully_shard)

    manager.parallelize(
        model,
        {"model.layers.0.proj.weight": ((Shard(0),), manager.mesh_context.device_mesh["tp"])},
    )

    assert fully_shard_calls[-1][0] is model
    assert fully_shard_calls[-1][1]["replicate_params"] == {
        model.model.embed_tokens.weight,
    }


def test_parallelize_rejects_non_bool_compile_hook_flag() -> None:
    """Reject ambiguous values before any FSDP units are created."""
    manager = _make_manager()

    with pytest.raises(ValueError, match="compile_hooks_enabled must be a bool"):
        manager.parallelize(_FakeCausalLM(), compile_hooks_enabled=1)
