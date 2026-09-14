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
"""Core contract tests for codegen artifacts, metadata, and coordination."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyper_parallel.codegen import manager
from hyper_parallel.codegen.artifact import (
    artifact_exists,
    clean_temp_artifacts,
    resolve_default_artifact_layout,
    resolve_artifact_layout,
    write_bundle_atomic,
)
from hyper_parallel.codegen.check.preflight import verify_output_hashes
from hyper_parallel.codegen.hash import canonical_json, signature_from_spec
from hyper_parallel.codegen.meta import CodegenMeta, load_codegen_meta
from hyper_parallel.codegen.modeling_backend import (
    ModelingBackend,
    resolve_modeling_backend,
)
from hyper_parallel.codegen.plan.freeze import (
    parse_placement,
    placement_to_string,
)
from hyper_parallel.codegen.source import resolver
from hyper_parallel.codegen.spec import project
from hyper_parallel.models._transformers import model_builder
from hyper_parallel.trainer.config import manager as config_manager


@dataclass
class _ModelTarget:
    """Minimal immutable-like model target for config preparation tests."""

    kwargs: dict[str, object] = field(default_factory=dict)

    def replace(self, **changes: object) -> "_ModelTarget":
        """Return a target carrying the supplied configured arguments."""
        return _ModelTarget({**self.kwargs, **changes})


@dataclass
class _TrainingConfig:
    """Minimal dataclass accepted by ``dataclasses.replace``."""

    model: _ModelTarget = field(default_factory=_ModelTarget)
    codegen: bool = False
    modeling_backend: str | None = None


def _meta(signature: str = "sig-current") -> CodegenMeta:
    return CodegenMeta(
        codegen_version="3",
        signature=signature,
        yaml_path="train.yaml",
        yaml_sha256="yaml-sha",
        source={"sha256": "source-sha"},
        parallel_dims={"tp": 2},
    )


def _bundle_files(layout: object) -> dict[str, str]:
    return {
        os.path.basename(layout.modeling_path): "VALUE = 1\n",
        os.path.basename(layout.diff_path): "--- source\n+++ generated\n",
        "__init__.py": "",
    }


def test_artifact_layout_uses_example_generated_dir_and_safe_model_name(
    tmp_path: Path,
) -> None:
    yaml_path = tmp_path / "recipes" / "train.demo.yaml"
    layout = resolve_artifact_layout(str(yaml_path), "Qwen3-MoE")

    assert Path(layout.artifact_dir) == yaml_path.parent / "generated"
    assert Path(layout.modeling_path).name == "modeling_Qwen3_MoE_gen_npu.py"
    assert Path(layout.diff_path).name == "modeling_Qwen3_MoE_gen_npu.py.diff"
    assert Path(layout.meta_path).name == "codegen_meta.json"


def test_artifact_layout_is_shared_by_yamls_in_one_example_dir(
    tmp_path: Path,
) -> None:
    first = resolve_artifact_layout(str(tmp_path / "first.yaml"), "demo")
    second = resolve_artifact_layout(str(tmp_path / "second.yaml"), "demo")

    assert first.artifact_dir == second.artifact_dir
    assert Path(first.artifact_dir) == tmp_path / "generated"


def test_default_artifact_layout_uses_current_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    layout = resolve_default_artifact_layout(model_name="Qwen3-MoE")

    assert layout.yaml_path == ""
    assert Path(layout.artifact_dir) == tmp_path / "generated"
    assert Path(layout.modeling_path).name == "modeling_Qwen3_MoE_gen_npu.py"


def test_load_training_config_does_not_prepare_codegen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _TrainingConfig(codegen=True, modeling_backend="gen")
    yaml_path = tmp_path / "train.yaml"
    monkeypatch.setattr(config_manager, "_load_training_config", lambda *_args: config)

    loaded = config_manager.load_training_config(yaml_path)

    assert loaded is config
    assert loaded._yaml_path == str(yaml_path.resolve())  # pylint: disable=protected-access


def test_training_args_only_parse_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _TrainingConfig(codegen=True, modeling_backend="gen")
    yaml_path = tmp_path / "train.yaml"
    monkeypatch.setattr(config_manager, "_load_training_config", lambda *_args: config)

    assert config_manager.parse_training_args([str(yaml_path)]) is config


def test_codegen_layout_prefers_yaml_path_and_hf_model_type(
    tmp_path: Path,
) -> None:
    yaml_path = tmp_path / "examples" / "training_demo" / "train_codegen.yaml"
    config = _TrainingConfig(codegen=True, modeling_backend="gen")
    config._yaml_path = str(yaml_path)  # pylint: disable=protected-access
    hf_config = SimpleNamespace(model_type="qwen3_moe")

    layout = manager._resolve_layout(  # pylint: disable=protected-access
        config,
        yaml_path=None,
        artifact_dir=None,
        hf_config=hf_config,
    )

    assert Path(layout.artifact_dir) == yaml_path.parent / "generated"
    assert Path(layout.modeling_path).name == "modeling_Qwen3_Moe_gen_npu.py"


@pytest.mark.parametrize(("rank", "expected"), [("0", True), ("1", False)])
def test_rank_detection_uses_torchrun_environment_before_process_group(
    monkeypatch: pytest.MonkeyPatch,
    rank: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("RANK", rank)

    assert manager._is_rank0() is expected


def test_atomic_bundle_hashes_staged_outputs_and_writes_meta_last(
    tmp_path: Path,
) -> None:
    layout = resolve_artifact_layout(str(tmp_path / "train.yaml"), "demo")
    meta = _meta()

    write_bundle_atomic(layout, _bundle_files(layout), meta)

    loaded = load_codegen_meta(layout.meta_path)
    assert loaded is not None
    assert artifact_exists(layout)
    assert set(loaded.outputs) == {"modeling", "diff", "init"}
    assert "meta" not in loaded.outputs
    verify_output_hashes(loaded, layout)

    raw = json.loads(Path(layout.meta_path).read_text(encoding="utf-8"))
    assert raw["signature"] == "sig-current"


def test_atomic_bundle_hashes_remote_siblings(tmp_path: Path) -> None:
    """Record and verify every source file consumed by a remote bundle."""
    layout = resolve_artifact_layout(str(tmp_path / "train.yaml"), "demo")
    files = _bundle_files(layout)
    files["configuration_demo.py"] = "VALUE = 1\n"
    meta = _meta()
    meta.remote_siblings = ["configuration_demo.py"]

    write_bundle_atomic(layout, files, meta)

    loaded = load_codegen_meta(layout.meta_path)
    assert loaded is not None
    assert "configuration_demo.py" in loaded.outputs
    verify_output_hashes(loaded, layout)

    sibling = Path(layout.artifact_dir) / "configuration_demo.py"
    sibling.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="configuration_demo.py hash drift"):
        verify_output_hashes(loaded, layout)


def test_partial_bundle_update_preserves_complete_published_bundle(
    tmp_path: Path,
) -> None:
    layout = resolve_artifact_layout(str(tmp_path / "train.yaml"), "demo")
    write_bundle_atomic(layout, _bundle_files(layout), _meta())
    original_meta = Path(layout.meta_path).read_bytes()
    original_modeling = Path(layout.modeling_path).read_bytes()

    write_bundle_atomic(layout, {"source_helper.py": "HELPER = 1\n"})

    assert Path(layout.meta_path).read_bytes() == original_meta
    assert Path(layout.modeling_path).read_bytes() == original_modeling
    assert (Path(layout.artifact_dir) / "source_helper.py").read_text(
        encoding="utf-8"
    ) == "HELPER = 1\n"


@pytest.mark.parametrize(
    "name",
    ["../escape.py", "subdir/escape.py", "subdir\\escape.py", "", ".", ".."],
)
def test_bundle_rejects_paths_that_escape_staging_directory(
    tmp_path: Path,
    name: str,
) -> None:
    layout = resolve_artifact_layout(str(tmp_path / "train.yaml"), "demo")

    with pytest.raises(ValueError, match="plain names"):
        write_bundle_atomic(layout, {name: "bad\n"}, _meta())

    with pytest.raises(ValueError, match="plain names"):
        write_bundle_atomic(
            layout, {str(tmp_path / "absolute_escape.py"): "bad\n"}, _meta()
        )


def test_complete_bundle_replacement_drops_obsolete_files(tmp_path: Path) -> None:
    """A full publication must contain exactly the newly emitted files."""
    layout = resolve_artifact_layout(str(tmp_path / "train.yaml"), "demo")
    initial = _bundle_files(layout)
    initial["obsolete_helper.py"] = "OLD = True\n"
    write_bundle_atomic(layout, initial, _meta("sig-old"))

    write_bundle_atomic(layout, _bundle_files(layout), _meta("sig-current"))

    assert not (Path(layout.artifact_dir) / "obsolete_helper.py").exists()


def test_output_hash_verification_detects_post_publish_mutation(tmp_path: Path) -> None:
    layout = resolve_artifact_layout(str(tmp_path / "train.yaml"), "demo")
    write_bundle_atomic(layout, _bundle_files(layout), _meta())
    loaded = load_codegen_meta(layout.meta_path)
    assert loaded is not None

    Path(layout.modeling_path).write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="modeling hash drift"):
        verify_output_hashes(loaded, layout)


def test_temp_artifact_cleanup_only_removes_codegen_siblings(tmp_path: Path) -> None:
    layout = resolve_artifact_layout(str(tmp_path / "train.yaml"), "demo")
    parent = Path(layout.artifact_dir).parent
    (parent / ".codegen_tmp_stale").mkdir(parents=True)
    (parent / ".codegen_old_stale").mkdir()
    keep = parent / "keep"
    keep.mkdir()

    clean_temp_artifacts(layout)

    assert not (parent / ".codegen_tmp_stale").exists()
    assert not (parent / ".codegen_old_stale").exists()
    assert keep.is_dir()


def test_signature_is_stable_for_key_order_and_sensitive_to_values() -> None:
    left = {"parallel": {"tp": 2, "cp": 1}, "source": {"sha256": "abc"}}
    reordered = {"source": {"sha256": "abc"}, "parallel": {"cp": 1, "tp": 2}}
    changed = {"source": {"sha256": "abc"}, "parallel": {"cp": 1, "tp": 4}}

    assert canonical_json(left) == canonical_json(reordered)
    assert signature_from_spec(left) == signature_from_spec(reordered)
    assert signature_from_spec(left) != signature_from_spec(changed)


def test_codegen_implementation_digest_tracks_python_sources(tmp_path: Path) -> None:
    """Invalidate a signature input when implementation Python changes."""
    package = tmp_path / "codegen"
    package.mkdir()
    (package / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "notes.txt").write_text("ignored\n", encoding="utf-8")
    initial = manager._codegen_implementation_digest(str(package))

    (package / "notes.txt").write_text("still ignored\n", encoding="utf-8")
    assert manager._codegen_implementation_digest(str(package)) == initial

    (package / "runtime.py").write_bytes(b"VALUE = 1\r\n")
    assert manager._codegen_implementation_digest(str(package)) == initial

    (package / "runtime.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert manager._codegen_implementation_digest(str(package)) != initial


def test_projected_signature_payload_includes_codegen_implementation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Make implementation identity part of the canonical signature input."""

    class Projected:
        """Minimal projected spec used to isolate implementation identity."""

        @staticmethod
        def to_dict() -> dict[str, object]:
            """Return a canonical payload without a resolvable source."""
            return {"source": {}, "target": {}}

    monkeypatch.setattr(
        project,
        "project_codegen_spec",
        lambda *_args, **_kwargs: Projected(),
    )
    monkeypatch.setattr(
        resolver, "resolve_model_source", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        manager, "_codegen_implementation_digest", lambda: "implementation-sha"
    )
    config = SimpleNamespace(codegen=False, modeling_backend="hf")
    yaml_path = str(tmp_path / "train.yaml")
    layout = resolve_artifact_layout(yaml_path, "demo")

    payload = manager._project_spec(config, layout)

    assert payload["codegen_implementation"] == {
        "version": manager.CODEGEN_VERSION,
        "sha256": "implementation-sha",
    }


@pytest.mark.parametrize("text", ["R", "S(1)", "S(-1)", "P(sum)", "SS(0,2)"])
def test_placement_strings_round_trip(text: str) -> None:
    assert placement_to_string(parse_placement(text)) == text


@pytest.mark.parametrize("text", ["", "Shard(0)", "S(x)", "SS(0)", "P()"])
def test_malformed_placement_strings_fail_fast(text: str) -> None:
    with pytest.raises(ValueError):
        parse_placement(text)


def test_codegen_switch_defaults_to_generated_backend() -> None:
    config = SimpleNamespace(architectures=["UnknownModel"])
    assert resolve_modeling_backend(config, codegen_enabled=True) is ModelingBackend.GEN


def test_force_hf_overrides_codegen_and_emits_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = SimpleNamespace(architectures=["UnknownModel"])

    with caplog.at_level(logging.WARNING):
        resolved = resolve_modeling_backend(config, force_hf=True, codegen_enabled=True)

    assert resolved is ModelingBackend.HF
    assert "generated modeling file will not be used" in caplog.text


def test_explicit_backend_precedes_codegen_switch() -> None:
    config = SimpleNamespace(architectures=["UnknownModel"])
    resolved = resolve_modeling_backend(
        config, modeling_backend="hf", codegen_enabled=True
    )
    assert resolved is ModelingBackend.HF


def test_unknown_backend_fails_at_configuration_boundary() -> None:
    config = SimpleNamespace(architectures=["UnknownModel"])
    with pytest.raises(ValueError, match="unknown modeling_backend"):
        resolve_modeling_backend(config, modeling_backend="invalid")


def test_generated_backend_model_keeps_hf_sharding_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = object()
    from hyper_parallel.codegen import loader

    monkeypatch.setattr(loader, "init_generated_model", lambda *_args, **_kwargs: marker)

    is_custom_model, model = model_builder._init_model(  # pylint: disable=protected-access
        object,
        "demo-model",
        SimpleNamespace(architectures=["Qwen3MoeForCausalLM"]),
        "sdpa",
        "bfloat16",
        True,
        backend=ModelingBackend.GEN,
        codegen_artifact_dir="/tmp/generated",
    )

    assert is_custom_model is False
    assert model is marker


def test_generated_sharding_skips_pure_data_parallel_mesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published: list[tuple[str, object]] = []
    from hyper_parallel.codegen import runtime

    monkeypatch.setattr(
        runtime,
        "publish_codegen_mesh_context",
        lambda artifact_dir, mesh: published.append((artifact_dir, mesh)),
    )
    mesh = SimpleNamespace(tp_size=1, cp_size=1, ep_size=1)

    handled, source_shard_info = model_builder._parallelize_from_generated(  # pylint: disable=protected-access
        object(),
        mesh,
        "/tmp/generated",
    )

    assert handled is False
    assert source_shard_info is None
    assert published == [("/tmp/generated", mesh)]


def test_wait_for_rank0_ignores_stale_signature(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = resolve_artifact_layout(str(tmp_path / "train.yaml"), "demo")
    observed = iter((_meta("sig-old"), _meta("sig-current")))
    calls: list[str] = []

    def fake_load(path: str) -> CodegenMeta:
        """Return stale metadata once, followed by the current bundle."""
        calls.append(path)
        return next(observed)

    monkeypatch.setattr(manager, "load_codegen_meta", fake_load)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    manager.wait_for_rank0_artifact(layout, "sig-current", timeout_s=1.0)
    assert calls == [layout.meta_path, layout.meta_path]


def test_wait_for_rank0_times_out_when_only_stale_meta_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    layout = resolve_artifact_layout(str(tmp_path / "train.yaml"), "demo")
    ticks = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(manager, "load_codegen_meta", lambda _path: _meta("sig-old"))
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    with pytest.raises(TimeoutError, match="timed out waiting for rank0 artifact"):
        manager.wait_for_rank0_artifact(layout, "sig-current", timeout_s=1.0)
