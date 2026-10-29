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
"""Model-anchored artifact layout: build, family resolution, priority, guard."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyper_parallel.codegen.artifact import (
    ArtifactLayout,
    artifact_failure_path,
    artifact_generation_lock,
    clear_artifact_failure,
    ensure_artifact_dir,
    publish_artifact_failure,
    resolve_model_artifact_layout,
)
from hyper_parallel.codegen import manager

MODELS_ROOT = Path(manager.__file__).resolve().parents[1] / "models"


class TestBuildModel:
    """ArtifactLayout.build_model_anchored anchors the bundle inside the model dir."""

    def test_layout_paths(self):
        layout = ArtifactLayout.build_model_anchored("qwen3_moe", model_name="Qwen3_Moe")
        expected = os.path.join(str(MODELS_ROOT), "qwen3_moe", "generated")
        assert layout.artifact_dir == expected
        assert layout.modeling_path == os.path.join(
            expected, "modeling_Qwen3_Moe_gen_npu.py"
        )
        assert layout.diff_path == layout.modeling_path + ".diff"
        assert layout.meta_path == os.path.join(expected, "codegen_meta.json")
        assert layout.init_path == os.path.join(expected, "__init__.py")

    def test_yaml_path_recorded_as_provenance(self, tmp_path):
        yaml_path = str(tmp_path / "train_codegen.yaml")
        layout = ArtifactLayout.build_model_anchored("qwen3_moe", yaml_path=yaml_path)
        assert layout.yaml_path == os.path.abspath(yaml_path)

    def test_yaml_path_empty_when_unanchored(self):
        layout = ArtifactLayout.build_model_anchored("qwen3_moe")
        assert layout.yaml_path == ""

    def test_missing_model_dir_warns_and_anchors_per_model_type(self, caplog):
        """A model type without a models/ directory still gets its own
        bundle home — one directory per model type, no aliasing — and the
        user is warned about the missing directory."""
        with caplog.at_level("WARNING", logger="hyper_parallel.codegen.artifact"):
            layout = ArtifactLayout.build_model_anchored("some_new_model")
        assert layout.artifact_dir == os.path.join(
            str(MODELS_ROOT), "some_new_model", "generated"
        )
        assert "some_new_model" in caplog.text
        assert "hyper_parallel/models" in caplog.text


class TestResolveModelArtifactLayout:
    """Strict 1:1 model_type → bundle directory, no aliasing."""

    def test_known_model_type(self):
        layout = resolve_model_artifact_layout("qwen3_moe")
        assert layout.artifact_dir == os.path.join(
            str(MODELS_ROOT), "qwen3_moe", "generated"
        )

    def test_one_dir_per_model_type_no_alias(self, caplog):
        """deepseek_v2 and deepseek_v3 may share adapter code, but their
        bundles never merge: each model type owns its directory."""
        with caplog.at_level("WARNING", logger="hyper_parallel.codegen.artifact"):
            layout_v2 = resolve_model_artifact_layout("deepseek_v2")
        layout_v3 = resolve_model_artifact_layout("deepseek_v3")
        assert layout_v2.artifact_dir.endswith(
            os.path.join("models", "deepseek_v2", "generated")
        )
        assert layout_v3.artifact_dir.endswith(
            os.path.join("models", "deepseek_v3", "generated")
        )
        assert layout_v2.artifact_dir != layout_v3.artifact_dir

    def test_unknown_model_type_gets_its_own_dir(self):
        layout = resolve_model_artifact_layout("no_such_model_type")
        assert layout.artifact_dir == os.path.join(
            str(MODELS_ROOT), "no_such_model_type", "generated"
        )


class TestResolveLayoutPriority:
    """manager._resolve_layout: explicit > YAML override > model > YAML > cwd."""

    @staticmethod
    def _config(**model_kwargs):
        return SimpleNamespace(model=SimpleNamespace(**model_kwargs))

    def test_explicit_artifact_dir_wins(self, tmp_path):
        config = self._config(codegen_artifact_dir=str(tmp_path / "override"))
        layout = manager._resolve_layout(
            config, None, str(tmp_path / "explicit"), hf_config=_hf("qwen3_moe")
        )
        assert layout.artifact_dir == str(tmp_path / "explicit")

    def test_yaml_override_beats_model_anchored(self, tmp_path):
        config = self._config(codegen_artifact_dir=str(tmp_path / "override"))
        layout = manager._resolve_layout(config, None, None, hf_config=_hf("qwen3_moe"))
        assert layout.artifact_dir == str(tmp_path / "override")

    def test_model_anchored_beats_yaml_anchored(self, tmp_path):
        config = self._config()
        layout = manager._resolve_layout(
            config, str(tmp_path / "train.yaml"), None, hf_config=_hf("qwen3_moe")
        )
        assert layout.artifact_dir == os.path.join(
            str(MODELS_ROOT), "qwen3_moe", "generated"
        )
        # The YAML stays recorded as provenance.
        assert layout.yaml_path == os.path.abspath(str(tmp_path / "train.yaml"))

    def test_deepseek_v4_uses_v41_family_directory(self):
        """The upstream V4 config identity resolves to the V4.1 adapter home."""
        layout = manager._resolve_layout(
            self._config(), None, None, hf_config=_hf("deepseek_v4")
        )
        assert layout.artifact_dir == os.path.join(
            str(MODELS_ROOT), "deepseek_v41", "generated"
        )

    def test_unknown_model_type_still_model_anchored(self, tmp_path, caplog):
        """A model type without a models/ directory warns and keeps its own
        model-anchored home instead of silently drifting to <yaml_dir>."""
        config = self._config()
        yaml_path = str(tmp_path / "train.yaml")
        with caplog.at_level("WARNING", logger="hyper_parallel.codegen.artifact"):
            layout = manager._resolve_layout(
                config, yaml_path, None, hf_config=_hf("no_such_model_type")
            )
        assert layout.artifact_dir == os.path.join(
            str(MODELS_ROOT), "no_such_model_type", "generated"
        )
        assert "no_such_model_type" in caplog.text

    def test_no_model_type_uses_yaml_anchored(self, tmp_path):
        config = self._config()
        yaml_path = str(tmp_path / "train.yaml")
        layout = manager._resolve_layout(config, yaml_path, None, hf_config=None)
        assert layout.artifact_dir == os.path.join(str(tmp_path), "generated")

    def test_from_config_path_anchors_like_the_trainer(self, monkeypatch, tmp_path):
        """CLI chain on a ``from_config`` YAML (``config_path`` spelling)
        resolves the model type and anchors to the same family directory the
        trainer chain uses."""
        from hyper_parallel.models._transformers import config_resolver

        monkeypatch.setattr(
            config_resolver,
            "get_hf_config",
            lambda *_args, **_kwargs: SimpleNamespace(model_type="qwen3_moe"),
        )
        config = self._config(config_path=str(tmp_path / "model-config"))
        layout = manager._resolve_layout(config, None, None)
        assert layout.artifact_dir == os.path.join(
            str(MODELS_ROOT), "qwen3_moe", "generated"
        )


class TestEnsureArtifactDirGuard:
    """A read-only install (site-packages) gets a readable error, not a crash."""

    def test_unwritable_directory_raises_helpful_error(self, tmp_path, monkeypatch):
        layout = ArtifactLayout.build_default(str(tmp_path / "generated"))

        def _raise_unwritable(self, parents=False, exist_ok=False):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(Path, "mkdir", _raise_unwritable)
        with pytest.raises(RuntimeError, match="pip install -e ."):
            ensure_artifact_dir(layout)


class TestArtifactCoordination:
    """Owner publication coordination leaves no stale process-local state."""

    def test_failure_marker_round_trip(self, tmp_path):
        """Failure markers atomically expose and clear the owner's reason."""
        layout = ArtifactLayout.build_default(str(tmp_path / "generated"))

        publish_artifact_failure(layout, "abc123", "render failed")

        marker = Path(artifact_failure_path(layout, "abc123"))
        marker_text = marker.read_text(encoding="utf-8")
        assert marker_text == "render failed", (
            f"failure marker mismatch: expected=render failed, got={marker_text}"
        )
        clear_artifact_failure(layout, "abc123")
        assert not marker.exists(), f"failure marker was not removed: path={marker}"

    def test_generation_lock_rejects_concurrent_owner_and_cleans_up(self, tmp_path):
        """Only one owner publishes and releasing it removes the lock."""
        layout = ArtifactLayout.build_default(str(tmp_path / "generated"))
        Path(layout.artifact_dir).parent.mkdir(parents=True, exist_ok=True)

        with artifact_generation_lock(layout):
            with pytest.raises(TimeoutError, match="artifact lock"):
                with artifact_generation_lock(layout, timeout_s=0):
                    pass

        with artifact_generation_lock(layout, timeout_s=0):
            pass


def _hf(model_type):
    return SimpleNamespace(model_type=model_type)
