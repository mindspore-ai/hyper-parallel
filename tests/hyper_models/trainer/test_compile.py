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
"""Unit tests for Trainer decoder-layer compilation."""

import os
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# Platform selection must happen before importing HyperParallel modules.
# pylint: disable=wrong-import-position,protected-access
import pytest
import torch
from torch import nn

from hyper_models._transformers import infrastructure
from hyper_models.components.distributed.compile import (
    _install_dynamo_mapping_get_polyfill,
    apply_compile,
    get_compile_layers,
    resolve_compile_kwargs,
)
from hyper_models.config.manager import parse_training_args
from hyper_models.trainer.config import CompileConfig


def _config_model_target() -> None:
    """Provide a deferred model target for config parsing tests."""


def _config_optimizer_target() -> None:
    """Provide a deferred optimizer target for config parsing tests."""


class _TinyDecoder(nn.Module):
    """Small model exposing the same layer contract as a Llama causal LM."""

    def __init__(self) -> None:
        """Build two repeated decoder layers."""
        super().__init__()
        self.config = SimpleNamespace(model_type="llama")
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(4, 4), nn.ReLU()) for _ in range(2)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the input through each decoder layer."""
        for layer in self.model.layers:
            inputs = layer(inputs)
        return inputs


class _FakeFSDP2Manager:
    """Record the compile-aware hook mode used during FSDP parallelization."""

    def __init__(self) -> None:
        """Initialize an unset compile mode and an observation list."""
        self.compile_flags = []

    def parallelize(
        self,
        model: nn.Module,
        *,
        compile_hooks_enabled: bool = False,
        **_kwargs: Any,
    ) -> nn.Module:
        """Return the model unchanged after recording the configured mode."""
        self.compile_flags.append(compile_hooks_enabled)
        return model


def test_compile_config_rejects_invalid_values() -> None:
    """Reject invalid compile field values at config construction time."""
    with pytest.raises(TypeError, match="compile.enabled"):
        CompileConfig(enabled=1)
    with pytest.raises(ValueError, match="options"):
        CompileConfig(mode="reduce-overhead", options={"triton.cudagraphs": True})
    with pytest.raises(ValueError, match="cache_size_limit"):
        CompileConfig(dynamo_cache_size_limit=0)


def test_compile_config_uses_existing_cli_override_path(tmp_path: Path) -> None:
    """Parse nested compile overrides through the existing Trainer config path."""
    config_file = tmp_path / "train.yaml"
    config_file.write_text(
        f"""
model:
  _target_: {__name__}._config_model_target
optimizer:
  _target_: {__name__}._config_optimizer_target
""".lstrip(),
        encoding="utf-8",
    )

    config = parse_training_args(
        [
            str(config_file),
            "--compile.enabled=true",
            "--compile.backend=aot_eager",
        ]
    )

    assert config.compile.enabled
    assert config.compile.backend == "aot_eager"


def test_compile_kwargs_use_options_without_mode() -> None:
    """Do not pass mutually exclusive mode and backend options together."""
    config = CompileConfig(enabled=True, options={"triton.cudagraphs": False})
    kwargs = resolve_compile_kwargs(config)
    assert kwargs["options"] == {"triton.cudagraphs": False}
    assert "mode" not in kwargs


def test_mapping_get_polyfill_is_idempotent_and_keeps_eager_semantics() -> None:
    """Dynamo sees a traceable Mapping.get while eager behavior remains unchanged."""
    _install_dynamo_mapping_get_polyfill()
    _install_dynamo_mapping_get_polyfill()

    mapping = {"present": 3}
    assert Mapping.get(mapping, "present", 0) == 3
    assert Mapping.get(mapping, "missing", 7) == 7

    def lookup(value: torch.Tensor) -> torch.Tensor:
        """Exercise the substituted Mapping.get during Dynamo tracing."""
        return value + Mapping.get(mapping, "present", 0)

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(lookup, backend="eager")
    torch.testing.assert_close(compiled(torch.ones(2)), torch.full((2,), 4.0))
    assert not torch._dynamo.utils.counters["graph_break"]


def test_compile_layers_are_stable_and_compile_in_place() -> None:
    """Compile decoder layers without replacing modules or state-dict keys."""
    model = _TinyDecoder()
    original_layers = [layer for _, layer in get_compile_layers(model)]
    original_keys = list(model.state_dict())

    apply_compile(model, CompileConfig(enabled=True, backend="aot_eager"))

    assert [layer for _, layer in get_compile_layers(model)] == original_layers
    assert list(model.state_dict()) == original_keys
    output = model(torch.randn(2, 4, requires_grad=True)).sum()
    output.backward()
    assert torch.isfinite(output)


@pytest.mark.parametrize(
    ("validate_placement", "expected_compile", "expected_fsdp_calls"),
    [(True, False, 0), (False, True, 1)],
)
def test_model_infrastructure_compiles_only_for_execution(
    monkeypatch: pytest.MonkeyPatch,
    validate_placement: bool,
    expected_compile: bool,
    expected_fsdp_calls: int,
) -> None:
    """Compile only the execution model and keep validation free of FSDP/compile calls."""
    model = _TinyDecoder()
    fsdp2_manager = _FakeFSDP2Manager()
    compile_calls = []

    def fake_apply_compile(
        compiled_model: nn.Module,
        config: CompileConfig,
    ) -> nn.Module:
        """Record requests to compile the production model."""
        del config
        compile_calls.append(compiled_model)
        return compiled_model

    monkeypatch.setattr(infrastructure, "FSDP2Manager", _FakeFSDP2Manager)
    monkeypatch.setattr(infrastructure, "apply_compile", fake_apply_compile)

    result = infrastructure.apply_model_infrastructure(
        model,
        fsdp2_manager=fsdp2_manager,
        compile_config=CompileConfig(enabled=True, backend="aot_eager"),
        validate_placement=validate_placement,
    )

    assert result is model
    assert fsdp2_manager.compile_flags == [expected_compile] * expected_fsdp_calls
    assert len(compile_calls) == int(expected_compile)
