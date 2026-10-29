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
"""Model-class ownership tests for generated artifacts."""

from types import ModuleType, SimpleNamespace

from hyper_parallel.codegen import loader
from hyper_parallel.codegen.meta import COVERED_DEFAULTS


class _GeneratedModel:
    """Stand-in for a class copied into generated source."""


class _NativeModel:
    """Stand-in for the registered custom model class."""


def _module() -> ModuleType:
    module = ModuleType("generated_modeling")
    module.DemoForCausalLM = _GeneratedModel
    return module


def test_runtime_owned_artifact_uses_registered_native_class(monkeypatch) -> None:
    """Runtime-owned custom models must preserve native Python identity."""
    monkeypatch.setattr(
        "hyper_parallel.models.registry._resolve_custom_model_cls",
        lambda architecture: _NativeModel if architecture == "DemoForCausalLM" else None,
    )
    config = SimpleNamespace(architectures=["DemoForCausalLM"])
    meta = SimpleNamespace(
        covered={**COVERED_DEFAULTS, "model_class": False},
        model_class="DemoForCausalLM",
    )

    assert loader._resolve_runtime_model_class(_module(), config, meta) is _NativeModel


def test_legacy_artifact_keeps_generated_class(monkeypatch) -> None:
    """Absent ownership metadata retains legacy inline behavior."""
    monkeypatch.setattr(
        "hyper_parallel.models.registry._resolve_custom_model_cls",
        lambda _architecture: _NativeModel,
    )
    config = SimpleNamespace(architectures=["DemoForCausalLM"])
    meta = SimpleNamespace(covered={"module_overrides": True}, model_class="DemoForCausalLM")

    assert loader._resolve_runtime_model_class(_module(), config, meta) is _GeneratedModel


def test_runtime_owned_hf_model_keeps_generated_class(monkeypatch) -> None:
    """HF-only models have no native custom registration to select."""
    monkeypatch.setattr(
        "hyper_parallel.models.registry._resolve_custom_model_cls",
        lambda _architecture: None,
    )
    config = SimpleNamespace(architectures=["DemoForCausalLM"])
    meta = SimpleNamespace(
        covered={**COVERED_DEFAULTS, "model_class": False},
        model_class="DemoForCausalLM",
    )

    assert loader._resolve_runtime_model_class(_module(), config, meta) is _GeneratedModel
