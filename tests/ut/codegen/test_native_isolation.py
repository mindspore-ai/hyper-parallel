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
"""Native-build isolation contracts for the internal codegen observation seam."""

import inspect
from dataclasses import fields
from types import SimpleNamespace

import pytest

from hyper_parallel.models._transformers import auto_model, model_builder
from hyper_parallel.trainer.config.resolver import ConfigResolutionError, resolve_config
from hyper_parallel.trainer.config.trainer import TrainerConfig


def test_public_auto_model_api_does_not_expose_codegen_session():
    """Concrete collection state must not cross the public AutoModel API."""
    for method in (
        auto_model._BaseHyperAutoModelClass.from_config,
        auto_model._BaseHyperAutoModelClass.from_pretrained,
    ):
        parameters = inspect.signature(method).parameters
        assert "codegen" not in parameters
        assert "codegen_artifact_dir" not in parameters
        assert "codegen_session" not in parameters
        assert "codegen_config" not in parameters


def test_trainer_config_removes_codegen_switch_with_migration_error():
    """Legacy YAML must fail with an actionable migration message."""
    assert "codegen" not in {field.name for field in fields(TrainerConfig)}
    with pytest.raises(ConfigResolutionError, match="codegen CLI"):
        resolve_config({"codegen": True})


def test_native_from_config_does_not_create_observer_or_touch_rng(monkeypatch):
    """The default native path must remain free of codegen lifecycle work."""
    captured = {}
    config = SimpleNamespace(model_type="toy", architectures=["ToyModel"])
    distributed_setup = SimpleNamespace(mesh_context=object())

    monkeypatch.setattr(
        auto_model.torch,
        "get_rng_state",
        lambda: (_ for _ in ()).throw(AssertionError("native build saved RNG state")),
    )
    monkeypatch.setattr(
        auto_model,
        "instantiate_infrastructure",
        lambda **_kwargs: ("planner", "fsdp"),
    )
    monkeypatch.setattr(auto_model, "_current_device", lambda: None)
    monkeypatch.setattr(auto_model, "get_is_hf_model", lambda *_args, **_kwargs: True)

    def _build_model(_cls, *_args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        auto_model._BaseHyperAutoModelClass,
        "_build_model",
        classmethod(_build_model),
    )

    auto_model._BaseHyperAutoModelClass.from_config(
        config,
        distributed_setup=distributed_setup,
    )

    assert "build_observer" not in captured
    assert "publish_codegen" not in captured
    assert "observation" not in captured


def test_native_model_build_has_no_codegen_dependency():
    """Native AutoModel and model-builder sources remain codegen-agnostic."""
    source = inspect.getsource(auto_model) + inspect.getsource(model_builder)
    assert "codegen" not in source.lower()
    assert "observer" not in source.lower()
    assert "observation" not in source.lower()
