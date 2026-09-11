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
"""Generated-module loading and remote-source compatibility tests."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from hyper_parallel.codegen import loader
from hyper_parallel.codegen.source.compat import sanitize_source_compat


def _write_modeling(bundle: Path, text: str) -> Path:
    bundle.mkdir(parents=True, exist_ok=True)
    path = bundle / "modeling_demo_gen_npu.py"
    path.write_text(text, encoding="utf-8")
    return path


def test_find_generated_modeling_file_requires_exactly_one_candidate(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="artifact dir"):
        loader.find_generated_modeling_file(str(tmp_path / "missing"))

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with pytest.raises(FileNotFoundError, match="bundle is incomplete"):
        loader.find_generated_modeling_file(str(bundle))

    _write_modeling(bundle, "VALUE = 1\n")
    (bundle / "modeling_stale_gen_npu.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="2 generated modeling files"):
        loader.find_generated_modeling_file(str(bundle))


def test_generated_import_is_cached_for_same_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_modeling(bundle, "VALUE = object()\n")

    first = loader.import_generated_module(str(bundle))
    second = loader.import_generated_module(str(bundle))

    assert first is second
    assert first.VALUE is second.VALUE


def test_same_bundle_name_in_different_paths_does_not_collide(tmp_path: Path) -> None:
    left = tmp_path / "left" / "bundle"
    right = tmp_path / "right" / "bundle"
    _write_modeling(left, "VALUE = 'left'\n")
    _write_modeling(right, "VALUE = 'right'\n")

    left_module = loader.import_generated_module(str(left))
    right_module = loader.import_generated_module(str(right))

    assert left_module is not right_module
    assert left_module.VALUE == "left"
    assert right_module.VALUE == "right"
    assert left_module.__name__ != right_module.__name__


def test_generated_import_resolves_remote_sibling_modules(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_modeling(
        bundle, "from .configuration_demo import VALUE\nRESULT = VALUE + 1\n"
    )
    (bundle / "configuration_demo.py").write_text("VALUE = 41\n", encoding="utf-8")

    module = loader.import_generated_module(str(bundle))

    assert module.RESULT == 42
    sibling_name = module.__package__ + ".configuration_demo"
    assert sibling_name in sys.modules


def test_failed_generated_import_does_not_cache_partial_module(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    modeling_path = _write_modeling(
        bundle, "raise RuntimeError('broken generated source')\n"
    )
    module_name = loader._module_name_for(str(bundle), modeling_path.stem)

    with pytest.raises(RuntimeError, match="broken generated source"):
        loader.import_generated_module(str(bundle))

    assert module_name not in sys.modules


def test_source_compat_is_byte_preserving_without_removed_symbol() -> None:
    source = "from transformers.utils import is_torch_available\nVALUE = 1\n"
    assert sanitize_source_compat(source) == source


def test_source_compat_replaces_removed_single_import_with_callable(
    tmp_path: Path,
) -> None:
    source = (
        "from transformers.utils.import_utils import is_torch_fx_available\n"
        "RESULT = is_torch_fx_available()\n"
    )

    transformed = sanitize_source_compat(source)
    bundle = tmp_path / "compat-bundle"
    _write_modeling(bundle, transformed)
    module = loader.import_generated_module(str(bundle))

    assert module.RESULT is True
    assert (
        "from transformers.utils.import_utils import is_torch_fx_available"
        not in transformed
    )


def test_source_compat_preserves_other_names_in_mixed_import() -> None:
    source = (
        "from transformers.utils.import_utils import (\n"
        "    is_torch_fx_available,\n"
        "    is_torch_available,\n"
        ")\n"
        "if is_torch_fx_available():\n"
        "    VALUE = 1\n"
    )

    transformed = sanitize_source_compat(source)
    tree = ast.parse(transformed)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert "is_torch_fx_available" not in imported
    assert "is_torch_available" in imported
    assert "if is_torch_fx_available():" in transformed


def test_source_compat_preserves_aliases_and_ignores_unrelated_modules() -> None:
    source = (
        "from transformers.utils import is_torch_fx_available, is_torch_available as torch_ok\n"
        "from project.compat import is_torch_fx_available as project_fx\n"
    )

    transformed = sanitize_source_compat(source)

    assert (
        "from transformers.utils import is_torch_available as torch_ok" in transformed
    )
    assert (
        "from project.compat import is_torch_fx_available as project_fx" in transformed
    )
