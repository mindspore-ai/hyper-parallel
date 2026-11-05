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
import sys
import unittest
from types import SimpleNamespace
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

    def test_build_lib_added_to_sys_path(self):
        """build/lib is added to sys.path before attempting import."""
        from hyper_parallel.platform.mindspore.custom_ops import custom_op_impl

        expected_suffix = os.path.join("custom_ops", "build", "lib")
        build_lib_paths = [p for p in sys.path if expected_suffix in p]
        self.assertTrue(build_lib_paths, f"build/lib not found in sys.path: {sys.path[:5]}")

    def test_missing_optional_kda_does_not_rebuild(self) -> None:
        """An older usable extension need not contain optional KDA bridges."""
        mock_mod = SimpleNamespace(npu_situ_glu=object(), npu_lightning_indexer_enhance=object())
        with patch.dict(os.environ, {"HP_CUSTOM_OPS_NO_AUTO_REBUILD": "0"}), \
                patch.dict(sys.modules, {_MODULE_NAME: mock_mod}):
            custom_op_impl = __import__(_PKG, fromlist=["_custom_ops"])
        self.assertIs(custom_op_impl._custom_ops, mock_mod)
        self._mock_ms.ops.CustomOpBuilder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
