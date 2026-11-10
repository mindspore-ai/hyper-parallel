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
"""Model-anchored artifact layouts and publication locking."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hyper_parallel.codegen import artifact, manager
from hyper_parallel.codegen.artifact import (
    ArtifactLayout,
    artifact_generation_lock,
    ensure_artifact_dir,
    resolve_model_artifact_layout,
)

MODELS_ROOT = Path(manager.__file__).resolve().parents[1] / "models"


class TestBuildModel:
    """Model-anchored bundles live beside their model adapter."""

    def test_layout_paths(self):
        layout = ArtifactLayout.build_model_anchored("qwen3_moe", model_name="Qwen3_Moe")
        expected = os.path.join(str(MODELS_ROOT), "qwen3_moe", "generated")
        assert layout.artifact_dir == expected
        assert layout.modeling_path == os.path.join(
            expected, "modeling_Qwen3_Moe_gen_npu.py"
        )
        assert layout.diff_path == layout.modeling_path + ".diff"
        assert layout.init_path == os.path.join(expected, "__init__.py")

    def test_missing_model_dir_warns_and_anchors_per_model_type(self, caplog):
        with caplog.at_level("WARNING", logger="hyper_parallel.codegen.artifact"):
            layout = ArtifactLayout.build_model_anchored("some_new_model")
        assert layout.artifact_dir == os.path.join(
            str(MODELS_ROOT), "some_new_model", "generated"
        )
        assert "some_new_model" in caplog.text
        assert "hyper_parallel/models" in caplog.text


class TestResolveModelArtifactLayout:
    """Each model type owns one bundle directory."""

    def test_known_model_type(self):
        layout = resolve_model_artifact_layout("qwen3_moe")
        assert layout.artifact_dir == os.path.join(
            str(MODELS_ROOT), "qwen3_moe", "generated"
        )

    def test_one_dir_per_model_type_no_alias(self, caplog):
        with caplog.at_level("WARNING", logger="hyper_parallel.codegen.artifact"):
            layout_v2 = resolve_model_artifact_layout("deepseek_v2")
        layout_v3 = resolve_model_artifact_layout("deepseek_v3")
        assert layout_v2.artifact_dir != layout_v3.artifact_dir

    def test_unknown_model_type_gets_its_own_dir(self):
        layout = resolve_model_artifact_layout("no_such_model_type")
        assert layout.artifact_dir == os.path.join(
            str(MODELS_ROOT), "no_such_model_type", "generated"
        )


class TestEnsureArtifactDirGuard:
    """A read-only install gets a readable error."""

    def test_unwritable_directory_raises_helpful_error(self, tmp_path, monkeypatch):
        layout = ArtifactLayout.build_default(str(tmp_path / "generated"))

        def _raise_unwritable(self, parents=False, exist_ok=False):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(Path, "mkdir", _raise_unwritable)
        with pytest.raises(RuntimeError, match="pip install -e ."):
            ensure_artifact_dir(layout)


class TestArtifactPublication:
    """Independent single-process publishers serialize bundle writes."""

    @staticmethod
    def _layout_and_lock(tmp_path):
        layout = ArtifactLayout.build_default(str(tmp_path / "generated"))
        lock_path = tmp_path / ".generated.codegen.lock"
        return layout, lock_path

    def test_generation_lock_records_pid_and_cleans_up(self, tmp_path):
        layout, lock_path = self._layout_and_lock(tmp_path)

        with artifact_generation_lock(layout):
            assert lock_path.read_text(encoding="ascii") == str(os.getpid())
            with pytest.raises(TimeoutError, match="artifact lock"):
                with artifact_generation_lock(layout, timeout_s=0):
                    pass

        assert not lock_path.exists()

    def test_generation_lock_recovers_dead_owner(self, tmp_path, monkeypatch):
        layout, lock_path = self._layout_and_lock(tmp_path)
        lock_path.write_text("43210", encoding="ascii")
        monkeypatch.setattr(artifact, "_pid_is_alive", lambda pid: pid != 43210)

        with artifact_generation_lock(layout, timeout_s=0):
            assert lock_path.read_text(encoding="ascii") == str(os.getpid())

        assert not lock_path.exists()

    def test_generation_lock_preserves_live_owner(self, tmp_path, monkeypatch):
        layout, lock_path = self._layout_and_lock(tmp_path)
        lock_path.write_text("43210", encoding="ascii")
        monkeypatch.setattr(artifact, "_pid_is_alive", lambda _pid: True)

        with pytest.raises(TimeoutError, match="artifact lock"):
            with artifact_generation_lock(layout, timeout_s=0):
                pass

        assert lock_path.read_text(encoding="ascii") == "43210"

    def test_recent_incomplete_lock_is_not_reclaimed(self, tmp_path):
        layout, lock_path = self._layout_and_lock(tmp_path)
        lock_path.write_text("", encoding="ascii")

        with pytest.raises(TimeoutError, match="artifact lock"):
            with artifact_generation_lock(layout, timeout_s=0):
                pass

        assert lock_path.exists()
