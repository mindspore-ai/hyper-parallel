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
"""Unit tests for custom_ops module loading logic."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

_MODULE_NAME = "hyper_parallel_custom_ops_ms"
_PKG = "hyper_parallel.platform.mindspore.custom_ops.custom_op_impl"


def _force_reimport():
    """Remove cached module to allow clean re-import in each test."""
    for key in list(sys.modules):
        if key == _PKG or key.startswith(_PKG + "."):
            del sys.modules[key]
    sys.modules.pop(_MODULE_NAME, None)


class TestCustomOpsLoading(unittest.TestCase):
    """Tests for custom_ops module loading logic."""

    def setUp(self):
        _force_reimport()
        self._mock_ms = MagicMock()
        self._patcher = patch.dict(sys.modules, {"mindspore": self._mock_ms})
        self._patcher.start()
        super().setUp()

    def tearDown(self):
        self._patcher.stop()
        _force_reimport()
        super().tearDown()

    def test_imports_prebuilt_so_when_available(self):
        """Pre-built .so module is imported when it exists in sys.modules."""
        mock_mod = MagicMock()
        with patch.dict(sys.modules, {_MODULE_NAME: mock_mod}):
            from hyper_parallel.platform.mindspore.custom_ops import custom_op_impl

        self.assertIs(custom_op_impl._custom_ops, mock_mod)

    def test_build_lib_is_searched_without_sys_path_mutation(self):
        """The prebuilt directory is explicit and does not leak into global import state."""
        mock_mod = MagicMock()
        original_sys_path = list(sys.path)
        with patch.dict(sys.modules, {_MODULE_NAME: mock_mod}):
            from hyper_parallel.platform.mindspore.custom_ops import custom_op_impl

        expected_suffix = os.path.join("custom_ops", "lib")
        self.assertTrue(any(expected_suffix in path for path in custom_op_impl._extension_search_paths()))
        self.assertEqual(sys.path, original_sys_path)

    def test_binary_loader_error_is_wrapped_with_native_guidance(self):
        """An incompatible prebuilt extension reports a stable native load error."""
        mock_mod = MagicMock()
        with patch.dict(sys.modules, {_MODULE_NAME: mock_mod}):
            from hyper_parallel.platform.mindspore.custom_ops import custom_op_impl

        sys.modules.pop(_MODULE_NAME, None)
        with tempfile.TemporaryDirectory() as temporary_directory:
            library = Path(temporary_directory) / f"{_MODULE_NAME}.so"
            library.touch()
            spec = MagicMock()
            spec.loader.exec_module.side_effect = OSError("incompatible binary")
            with patch.object(
                custom_op_impl,
                "_extension_search_paths",
                return_value=[temporary_directory],
            ), patch.object(
                custom_op_impl.machinery,
                "EXTENSION_SUFFIXES",
                (".so",),
            ), patch.object(
                custom_op_impl.util,
                "spec_from_file_location",
                return_value=spec,
            ), patch.object(
                custom_op_impl.util,
                "module_from_spec",
                return_value=MagicMock(),
            ):
                with self.assertRaisesRegex(ImportError, "incompatible binary"):
                    custom_op_impl._load_prebuilt_extension()

        self.assertNotIn(_MODULE_NAME, sys.modules)

    def test_binary_loader_does_not_remove_concurrently_registered_module(self):
        """A failed candidate only removes the module object that it registered."""
        mock_mod = MagicMock()
        with patch.dict(sys.modules, {_MODULE_NAME: mock_mod}):
            from hyper_parallel.platform.mindspore.custom_ops import custom_op_impl

        sys.modules.pop(_MODULE_NAME, None)
        replacement = MagicMock()
        with tempfile.TemporaryDirectory() as temporary_directory:
            library = Path(temporary_directory) / f"{_MODULE_NAME}.so"
            library.touch()
            spec = MagicMock()

            def fail_after_replacement(module):
                del module
                sys.modules[_MODULE_NAME] = replacement
                raise OSError("incompatible binary")

            spec.loader.exec_module.side_effect = fail_after_replacement
            with patch.object(
                custom_op_impl, "_extension_search_paths", return_value=[temporary_directory]
            ), patch.object(
                custom_op_impl.machinery, "EXTENSION_SUFFIXES", (".so",)
            ), patch.object(
                custom_op_impl.util, "spec_from_file_location", return_value=spec
            ), patch.object(
                custom_op_impl.util, "module_from_spec", return_value=MagicMock()
            ):
                with self.assertRaisesRegex(ImportError, "incompatible binary"):
                    custom_op_impl._load_prebuilt_extension()

        self.assertIs(sys.modules.pop(_MODULE_NAME), replacement)


if __name__ == "__main__":
    unittest.main()
