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
"""Unit tests for indexed helper discovery and source-mode compilation."""

from __future__ import annotations

import builtins
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from typing import Any
import unittest
from unittest.mock import Mock, patch

from hyper_parallel.data import indexed
from tests.common.mark_utils import arg_mark


_CPP_MODULE_NAME = "hyper_parallel.data.indexed._indexed_helpers_cpp"
_HELPER_SOURCE = Path(indexed.__file__).with_name("indexed_helpers.py")


class _MissingThenAvailableImporter:
    """Raise for the first helper import, then return a fake built module."""

    def __init__(self, cpp_module: ModuleType) -> None:
        """Initialize the importer with the module returned after compilation."""
        self.cpp_module = cpp_module
        self.calls = 0
        self._original_import = builtins.__import__

    def __call__(
        self,
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        """Delegate ordinary imports and simulate one missing native helper."""
        if name == _CPP_MODULE_NAME:
            self.calls += 1
            if self.calls == 1:
                raise ImportError("indexed helper is not built")
            return self.cpp_module
        return self._original_import(name, globals_, locals_, fromlist, level)


def _make_cpp_module():
    """Create a fake native helper module with the required exports."""
    cpp_module = ModuleType(_CPP_MODULE_NAME)
    cpp_module.build_blending_indices = Mock(name="build_blending_indices")
    cpp_module.build_sample_index_int32 = Mock(name="build_sample_index_int32")
    cpp_module.build_sample_index_int64 = Mock(name="build_sample_index_int64")
    return cpp_module


def _execute_helper_module(importer):
    """Execute indexed_helpers.py under an isolated module name."""
    spec = importlib.util.spec_from_file_location("_indexed_helpers_under_test", _HELPER_SOURCE)
    module = importlib.util.module_from_spec(spec)
    with patch("builtins.__import__", side_effect=importer):
        spec.loader.exec_module(module)
    return module


class TestIndexedPackagePath(unittest.TestCase):
    """Verify source builds expose their staged native helper directory."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_extend_native_path_appends_existing_payload_once(self):
        """Verify source payload discovery does not duplicate package paths.

        Feature: Indexed helper source delivery.
        Description: Create a repository-shaped temporary payload and extend the package path twice.
        Expectation: The staged native directory is appended exactly once.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory)
            package_file = repository / "hyper_parallel/data/indexed/__init__.py"
            package_file.parent.mkdir(parents=True)
            package_file.touch()
            (repository / "setup.py").touch()
            native = repository / "build/native/payload/hyper_parallel/data/indexed"
            native.mkdir(parents=True)
            package_path = []

            indexed._extend_native_path(str(package_file), package_path)
            indexed._extend_native_path(str(package_file), package_path)

        self.assertEqual(package_path, [str(native)])


class TestIndexedHelperCompilation(unittest.TestCase):
    """Verify rank-local source compilation without invoking a compiler."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rank_zero_builds_and_loads_helper(self):
        """Verify rank zero builds and atomically publishes the native helper.

        Feature: Indexed helper JIT compilation.
        Description: Simulate a missing extension on local rank zero with compiler calls mocked.
        Expectation: Make writes a temporary library that is atomically published and imported.
        """
        cpp_module = _make_cpp_module()
        importer = _MissingThenAvailableImporter(cpp_module)
        expected_library = _HELPER_SOURCE.parent / "_indexed_helpers_cpp.test.so"

        with patch.dict(os.environ, {"LOCAL_RANK": "0"}), \
                patch("sysconfig.get_config_var", return_value=".test.so"), \
                patch("subprocess.run") as mock_run, \
                patch("os.replace") as mock_replace, \
                patch("time.monotonic", side_effect=[10.0, 12.5]), \
                patch("importlib.invalidate_caches") as mock_invalidate:
            module = _execute_helper_module(importer)

        mock_run.assert_called_once_with(
            [
                "make",
                "-B",
                "-C",
                str(_HELPER_SOURCE.parent / "csrc"),
                f"PYTHON={sys.executable}",
                f"OUTPUT={expected_library}.tmp",
            ],
            check=True,
        )
        mock_replace.assert_called_once_with(f"{expected_library}.tmp", expected_library)
        mock_invalidate.assert_called_once_with()
        self.assertEqual(importer.calls, 2)
        self.assertIs(getattr(module, "build_sample_index_int32"), getattr(cpp_module, "build_sample_index_int32"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_nonzero_rank_waits_for_rank_zero_output(self):
        """Verify nonzero ranks wait for the rank-zero helper output.

        Feature: Indexed helper JIT coordination.
        Description: Simulate the shared library appearing after one polling interval.
        Expectation: The nonzero rank waits without compiling and then imports the helper.
        """
        cpp_module = _make_cpp_module()
        importer = _MissingThenAvailableImporter(cpp_module)

        with patch.dict(os.environ, {"LOCAL_RANK": "1"}), \
                patch("sysconfig.get_config_var", return_value=".test.so"), \
                patch("pathlib.Path.is_file", side_effect=[False, True]), \
                patch("time.monotonic", side_effect=[10.0, 11.0]), \
                patch("time.sleep") as mock_sleep, \
                patch("subprocess.run") as mock_run, \
                patch("importlib.invalidate_caches") as mock_invalidate:
            module = _execute_helper_module(importer)

        mock_sleep.assert_called_once_with(0.1)
        mock_run.assert_not_called()
        mock_invalidate.assert_called_once_with()
        self.assertEqual(importer.calls, 2)
        self.assertIs(getattr(module, "build_sample_index_int64"), getattr(cpp_module, "build_sample_index_int64"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_nonzero_rank_times_out_waiting_for_helper(self):
        """Verify nonzero-rank waiting has a bounded failure.

        Feature: Indexed helper JIT timeout.
        Description: Simulate rank zero never publishing the shared helper library.
        Expectation: The waiting rank raises the original import failure after 300 seconds.
        """
        importer = _MissingThenAvailableImporter(_make_cpp_module())

        with patch.dict(os.environ, {"LOCAL_RANK": "1"}), \
                patch("sysconfig.get_config_var", return_value=".test.so"), \
                patch("pathlib.Path.is_file", return_value=False), \
                patch("time.monotonic", side_effect=[10.0, 311.0]), \
                patch("time.sleep") as mock_sleep, \
                patch("importlib.invalidate_caches") as mock_invalidate:
            with self.assertRaisesRegex(ImportError, "Timed out waiting for rank 0"):
                _execute_helper_module(importer)

        mock_sleep.assert_not_called()
        mock_invalidate.assert_not_called()
        self.assertEqual(importer.calls, 1)


if __name__ == "__main__":
    unittest.main()
