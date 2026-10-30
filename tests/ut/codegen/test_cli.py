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
"""Codegen CLI and build-only orchestration contracts."""

from types import SimpleNamespace
from typing import Any

import pytest

from hyper_parallel.codegen import build, cli
from hyper_parallel.models.build_observer import current_model_build_observation


def test_generate_uses_native_build_and_typed_overrides(monkeypatch):
    """Generate consumes TrainerConfig but delegates only to build-only orchestration."""
    captured = {}
    config = object()
    layout = SimpleNamespace(meta_path="meta.json", artifact_dir="generated")

    def _load_training_config(yaml_path, overrides=()):
        captured["yaml_path"] = yaml_path
        captured["overrides"] = list(overrides)
        return config

    def _generate(received_config, *, artifact_dir=None):
        captured["config"] = received_config
        captured["artifact_dir"] = artifact_dir
        return layout

    monkeypatch.setattr(cli.config_parser, "load_training_config", _load_training_config)
    monkeypatch.setattr(cli, "generate_from_native_build", _generate)
    monkeypatch.setattr(cli, "load_codegen_meta", lambda _path: SimpleNamespace(signature="abc"))

    result = cli.main([
        "generate",
        "--config",
        "train.yaml",
        "--output",
        "bundle",
        "--accelerator.tp_size=2",
    ])

    assert result == 0
    assert captured == {
        "yaml_path": "train.yaml",
        "overrides": ["--accelerator.tp_size=2"],
        "config": config,
        "artifact_dir": "bundle",
    }


def test_generate_rejects_torchrun_environment(monkeypatch):
    """Multiple CLI ranks fail before parsing config or touching devices."""
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr(
        cli.config_parser,
        "load_training_config",
        lambda *_args, **_kwargs: pytest.fail("config was loaded under torchrun"),
    )

    result = cli.main(["generate", "--config", "train.yaml"])

    assert result == 1


def test_native_build_orchestration_builds_only_model(monkeypatch):
    """Build-only generation never constructs training-owned components."""
    events = []
    layout = object()

    class ModelTarget:
        @staticmethod
        def build(**kwargs: Any) -> object:
            """Capture build-only arguments and emulate artifact publication."""
            observation = current_model_build_observation()
            events.append(("model", kwargs, observation))
            observation.artifact_layout = layout
            return object()

    config = SimpleNamespace(
        training=SimpleNamespace(backend="gloo", seed=7, enable_full_determinism=False),
        model=ModelTarget(),
        peft=None,
        activation_checkpoint=SimpleNamespace(
            mode="none",
            selection=None,
            swap_inputs=False,
            allow_cache_entry_mutation=False,
        ),
        activation_swap="none",
        compile=None,
        model_init_dtype=None,
    )
    setup = object()
    monkeypatch.setattr(build, "create_offline_distributed_setup", lambda _config: setup)
    monkeypatch.setattr(
        build,
        "normalize_distributed_setup_overrides",
        lambda received_setup, received_config: events.append(("normalize", received_setup, received_config)),
    )
    result = build.generate_from_native_build(config, artifact_dir="bundle")

    assert result is layout
    assert [event[0] for event in events] == ["normalize", "model"]
    observation = events[-1][2]
    assert observation.artifact_dir == "bundle"
    assert observation.observation_only is True
    assert current_model_build_observation() is None


def test_native_build_requires_published_artifact(monkeypatch):
    """A completed model build cannot be reported as successful without a bundle."""
    config = SimpleNamespace(
        training=SimpleNamespace(backend="gloo", seed=7, enable_full_determinism=False),
        model=SimpleNamespace(build=lambda **_kwargs: object()),
        peft=None,
        activation_checkpoint=SimpleNamespace(mode="none", selection=None),
        activation_swap="none",
        compile=None,
        model_init_dtype=None,
    )
    monkeypatch.setattr(build, "create_offline_distributed_setup", lambda _config: object())
    monkeypatch.setattr(build, "normalize_distributed_setup_overrides", lambda *_args: None)

    with pytest.raises(RuntimeError, match="without publishing"):
        build.generate_from_native_build(config)
