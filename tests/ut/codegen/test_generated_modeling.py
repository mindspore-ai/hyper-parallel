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
"""Tests for the published generated modeling files."""

from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")

# The optional-dependency gate must run before importing the model stack.
# pylint: disable=wrong-import-position
from hyper_parallel.codegen.artifact import ArtifactLayout
from hyper_parallel.codegen.emit import bundle


def test_generated_modeling_has_no_training_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A generated modeling module contains model code, never a Trainer main."""
    source = "from __future__ import annotations\n\nclass FutureArchitecture:\n    pass\n"
    layout = ArtifactLayout.build_default(str(tmp_path / "generated"), "Future")
    meta = SimpleNamespace(
        status="derived",
        source={"file_path": str(tmp_path / "modeling_future.py")},
        param_plan={},
    )
    monkeypatch.setattr(bundle, "copy_original_modeling", lambda _source: source)
    monkeypatch.setattr(bundle, "emit_modeling_file", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(bundle, "verify_recorded_patch_bundle", lambda *_args, **_kwargs: None)

    files = bundle.emit_bundle(layout, meta, patch_plan=object())
    modeling = files["modeling_Future_gen_npu.py"]

    assert modeling.startswith(
        "# GENERATED CODE. DO NOT EDIT MANUALLY.\n"
        "# pylint: skip-file\n"
        "# codespell:ignore-begin\n"
    )
    assert modeling.endswith("# codespell:ignore-end\n")
    diff_text = files["modeling_Future_gen_npu.py.diff"]
    assert "+# codespell:ignore-begin" in diff_text
    assert "+# codespell:ignore-end" in diff_text
    assert "if __name__" not in modeling
    assert "run_generated_training" not in modeling


def test_generated_modeling_normalizes_bare_comment_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A bare URL in an upstream comment receives a safe delimiter."""
    source = (
        "# See https:" + "//github.com/huggingface/transformers/pull/29402\n"
        + "output = 1\n"
    )
    layout = ArtifactLayout.build_default(str(tmp_path / "generated"), "Future")
    meta = SimpleNamespace(
        status="derived",
        source={"file_path": str(tmp_path / "modeling_future.py")},
        param_plan={},
    )
    monkeypatch.setattr(bundle, "copy_original_modeling", lambda _source: source)
    monkeypatch.setattr(bundle, "emit_modeling_file", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(
        bundle, "verify_recorded_patch_bundle", lambda *_args, **_kwargs: None
    )

    files = bundle.emit_bundle(layout, meta, patch_plan=object())

    assert "transformers/pull/29402 (link).\noutput" in files["modeling_Future_gen_npu.py"]


def test_generated_modeling_terminates_license_url_before_comment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A license URL cannot merge with the following comment marker."""
    source = "# http:" + "//www.apache.org/licenses/LICENSE-2.0\n## separator\n"
    layout = ArtifactLayout.build_default(str(tmp_path / "generated"), "Future")
    meta = SimpleNamespace(
        status="derived",
        source={"file_path": str(tmp_path / "modeling_future.py")},
        param_plan={},
    )
    monkeypatch.setattr(bundle, "copy_original_modeling", lambda _source: source)
    monkeypatch.setattr(bundle, "emit_modeling_file", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(
        bundle, "verify_recorded_patch_bundle", lambda *_args, **_kwargs: None
    )

    files = bundle.emit_bundle(layout, meta, patch_plan=object())

    modeling = files["modeling_Future_gen_npu.py"]
    assert "LICENSE-2.0 (link).\n## separator" in modeling
    assert "LICENSE-2.0.##" not in modeling


def test_generated_bundle_normalizes_copied_source_spelling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Copied source and its diff do not retain known upstream typos."""
    first_typo = "Ple" + "se"
    second_typo = "suppo" + "erted"
    source = f'raise ValueError("{first_typo} use a {second_typo} option")\n'
    layout = ArtifactLayout.build_default(str(tmp_path / "generated"), "Future")
    meta = SimpleNamespace(
        status="derived",
        source={"file_path": str(tmp_path / "modeling_future.py")},
        param_plan={},
    )
    monkeypatch.setattr(bundle, "copy_original_modeling", lambda _source: source)
    monkeypatch.setattr(bundle, "emit_modeling_file", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(
        bundle, "verify_recorded_patch_bundle", lambda *_args, **_kwargs: None
    )

    files = bundle.emit_bundle(layout, meta, patch_plan=object())

    assert first_typo not in files["modeling_Future_gen_npu.py"]
    assert second_typo not in files["modeling_Future_gen_npu.py"]
    assert first_typo not in files["modeling_Future_gen_npu.py.diff"]
    assert second_typo not in files["modeling_Future_gen_npu.py.diff"]
    assert files["__init__.py"].startswith("# Copyright 2026 Huawei")
