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

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from hyper_parallel.codegen import build, cli
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0", card_mark="onecard",
          essential_mark="essential")
def test_generate_uses_native_build_and_typed_overrides(monkeypatch):
    """Feature: CLI native-build delegation.
    Description: Generate with typed overrides and a mocked native build.
    Expectation: The resolved config and overrides reach generation unchanged.
    """
    captured = {}
    config = object()
    layout = SimpleNamespace(artifact_dir="generated")

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


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0", card_mark="onecard",
          essential_mark="essential")
def test_generate_rejects_torchrun_environment(monkeypatch):
    """Feature: Single-process CLI enforcement.
    Description: Run generation with a multi-rank environment.
    Expectation: Generation fails before parsing config or touching devices.
    """
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr(
        cli.config_parser,
        "load_training_config",
        lambda *_args, **_kwargs: pytest.fail("config was loaded under torchrun"),
    )

    result = cli.main(["generate", "--config", "train.yaml"])

    assert result == 1


@pytest.mark.parametrize("removed_command", ["train", "check"])
@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0", card_mark="onecard",
          essential_mark="essential")
def test_cli_rejects_removed_commands(removed_command):
    """Feature: CLI command surface.
    Description: Invoke commands removed from the public CLI.
    Expectation: Argument parsing rejects every removed command.
    """
    with pytest.raises(SystemExit) as captured:
        cli.main([removed_command])

    assert captured.value.code == 2


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0", card_mark="onecard",
          essential_mark="essential")
def test_verbose_cli_logs_warning_with_traceback(monkeypatch):
    """Feature: Verbose codegen failure logging.
    Description: Raise a generation error while verbose logging is enabled.
    Expectation: The CLI logs a warning with traceback details and returns failure.
    """
    captured = {}

    def _warning(message, error, *, exc_info):
        captured["message"] = message
        captured["error"] = error
        captured["exc_info"] = exc_info

    monkeypatch.setattr(cli.logger, "warning", _warning)
    monkeypatch.setattr(
        cli,
        "cmd_generate",
        lambda _args: (_ for _ in ()).throw(RuntimeError("broken model build")),
    )

    result = cli.main([
        "--verbose",
        "generate",
        "--config",
        "train.yaml",
    ])

    assert result == 1, f"Expected verbose CLI failure rc=1, got rc={result}"
    assert captured["message"] == "codegen: %s", (
        f"Expected exception log format 'codegen: %s', "
        f"got message={captured['message']!r}"
    )
    assert str(captured["error"]) == "broken model build", (
        f"Expected original exception in verbose log, "
        f"got error={captured['error']!r}"
    )
    assert captured["exc_info"] is True


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0", card_mark="onecard",
          essential_mark="essential")
def test_cli_logs_failure_as_warning(monkeypatch):
    """Feature: Codegen failure log level.
    Description: Raise a generation error with default logging.
    Expectation: The CLI logs a warning without traceback and returns failure.
    """
    captured = {}

    def _warning(message, error, *, exc_info):
        captured["message"] = message
        captured["error"] = error
        captured["exc_info"] = exc_info

    monkeypatch.setattr(cli.logger, "warning", _warning)
    monkeypatch.setattr(
        cli,
        "cmd_generate",
        lambda _args: (_ for _ in ()).throw(RuntimeError("broken model build")),
    )

    result = cli.main([
        "generate",
        "--config",
        "train.yaml",
    ])

    assert result == 1
    assert captured["message"] == "codegen: %s"
    assert str(captured["error"]) == "broken model build"
    assert captured["exc_info"] is False


@pytest.mark.parametrize(
    "activation_checkpoint",
    [None, SimpleNamespace(mode="off"), SimpleNamespace(mode="hybrid", full=object(), selective=object())],
)
@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0", card_mark="onecard",
          essential_mark="essential")
def test_native_build_orchestration_builds_only_model(monkeypatch, activation_checkpoint):
    """Feature: Build-only codegen orchestration.
    Description: Run generation with mocked model construction and planner setup.
    Expectation: Only the model path is built and its artifact is returned.
    """
    events = []
    layout = object()

    class ModelTarget:
        @staticmethod
        def build(**kwargs: Any) -> object:
            """Capture build-only arguments."""
            events.append(("model", kwargs))
            return object()

    @contextmanager
    def _model_overlay(_session, received_artifact_dir, *, defer_publish):
        """Provide the artifact normally populated by the overlay."""
        assert received_artifact_dir == "bundle"
        assert defer_publish is True
        yield SimpleNamespace(artifact_layout=layout)

    config = SimpleNamespace(
        training=SimpleNamespace(backend="gloo", seed=7, enable_full_determinism=False),
        model=ModelTarget(),
        peft=None,
        activation_checkpoint=activation_checkpoint,
        compile=None,
        model_init_dtype=None,
    )
    setup = object()
    monkeypatch.setattr(build, "create_offline_distributed_setup", lambda _config: setup)
    monkeypatch.setattr(build, "use_codegen_model_build_overlay", _model_overlay)
    monkeypatch.setattr(
        build,
        "normalize_distributed_setup_overrides",
        lambda received_setup, received_config: events.append(("normalize", received_setup, received_config)),
    )
    result = build.generate_from_native_build(config, artifact_dir="bundle")

    assert result is layout
    assert [event[0] for event in events] == ["normalize", "model"]
    model_kwargs = events[-1][1]
    assert model_kwargs["activation_checkpoint"] is activation_checkpoint
    assert "activation_checkpoint_selection" not in model_kwargs
    assert "swap_inputs" not in model_kwargs
    assert "allow_cache_entry_mutation" not in model_kwargs
    assert "activation_swap" not in model_kwargs


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0", card_mark="onecard",
          essential_mark="essential")
def test_native_build_requires_published_artifact(monkeypatch):
    """Feature: Codegen publication contract.
    Description: Complete model construction without publishing an artifact.
    Expectation: Generation reports the missing bundle as a failure.
    """
    config = SimpleNamespace(
        training=SimpleNamespace(backend="gloo", seed=7, enable_full_determinism=False),
        model=SimpleNamespace(build=lambda **_kwargs: object()),
        peft=None,
        activation_checkpoint=None,
        compile=None,
        model_init_dtype=None,
    )
    monkeypatch.setattr(build, "create_offline_distributed_setup", lambda _config: object())
    monkeypatch.setattr(build, "normalize_distributed_setup_overrides", lambda *_args: None)

    with pytest.raises(RuntimeError, match="without publishing"):
        build.generate_from_native_build(config)


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0", card_mark="onecard",
          essential_mark="essential")
def test_native_build_reports_isolated_publication_failure(monkeypatch):
    """Feature: Codegen publication diagnostics.
    Description: Finish an isolated build with a retained publication failure.
    Expectation: Generation surfaces the recorded stage and reason.
    """
    config = SimpleNamespace(
        training=SimpleNamespace(backend="gloo", seed=7, enable_full_determinism=False),
        model=SimpleNamespace(build=lambda **_kwargs: object()),
        peft=None,
        activation_checkpoint=None,
        compile=None,
        model_init_dtype=None,
    )

    @contextmanager
    def _failed_overlay(_session, _artifact_dir, *, defer_publish):
        assert defer_publish is True
        yield SimpleNamespace(
            artifact_layout=None,
            snapshot=SimpleNamespace(
                failure=SimpleNamespace(
                    stage="generation",
                    reason="recorded patch target was not found",
                ),
            ),
        )

    monkeypatch.setattr(build, "create_offline_distributed_setup", lambda _config: object())
    monkeypatch.setattr(build, "normalize_distributed_setup_overrides", lambda *_args: None)
    monkeypatch.setattr(build, "use_codegen_model_build_overlay", _failed_overlay)

    with pytest.raises(
        RuntimeError,
        match="codegen generation failed: recorded patch target was not found",
    ):
        build.generate_from_native_build(config)
