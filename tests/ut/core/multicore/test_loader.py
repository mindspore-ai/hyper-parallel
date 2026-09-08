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
"""Unit tests for unified multicore payload lookup and OPP diagnostics."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hyper_parallel.core.multicore import _loader
from hyper_parallel.core.multicore._loader import NativeComponentUnavailableError


class TestMulticoreNative(unittest.TestCase):
    """Verify component-owned OPP environment and payload lookup."""

    def setUp(self) -> None:
        """Create a local native root with one vendor and the Torch adapter directory."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.native_root = self.root / "hyper_parallel" / "core" / "multicore" / "lib"
        self.vendor_root = (
            self.native_root / "vendors" / "hyper_parallel_multicore_nn"
        )
        library = self.vendor_root / "op_api" / "lib" / "libcust_opapi.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"vendor")
        adapter = (
            self.native_root / "framework" / "torch"
            / "libhyper_parallel_mega_moe_torch.so"
        )
        adapter.parent.mkdir(parents=True)
        adapter.write_bytes(b"adapter")

    def test_component_paths_accept_sourced_environment_without_modifying_it(self):
        """Lookup accepts the sourced vendor paths without changing the process environment."""
        op_api_root = self.vendor_root / "op_api" / "lib"
        environment = {
            "ASCEND_CUSTOM_OPP_PATH": os.pathsep.join(["preexisting", str(self.vendor_root)]),
            "LD_LIBRARY_PATH": os.pathsep.join(["existing-lib", str(op_api_root)]),
        }
        with patch.dict(os.environ, environment, clear=True), patch.object(
            _loader, "_component_root", return_value=self.native_root
        ), patch.object(_loader, "sys", SimpleNamespace(modules={})):
            vendor_root, adapter = _loader.get_multicore_paths()
            opp_value = os.environ["ASCEND_CUSTOM_OPP_PATH"]
            library_value = os.environ["LD_LIBRARY_PATH"]

        self.assertEqual(vendor_root, self.vendor_root.resolve())
        self.assertEqual(adapter.name, "libhyper_parallel_mega_moe_torch.so")
        self.assertEqual(opp_value, environment["ASCEND_CUSTOM_OPP_PATH"])
        self.assertEqual(library_value, environment["LD_LIBRARY_PATH"])

    def test_missing_opp_environment_requires_set_env(self):
        """The loader requires explicit environment activation before a framework is loaded."""
        with patch.dict(os.environ, {}, clear=True), patch.object(
            _loader, "sys", SimpleNamespace(modules={})
        ), patch.object(
            _loader, "_component_root", return_value=self.native_root
        ), self.assertRaisesRegex(
            NativeComponentUnavailableError,
            f"HP-NATIVE-OPP-NOT-ACTIVATED.*source {self.native_root / 'set_env.bash'}",
        ):
            _loader.require_multicore_environment()

    def test_missing_opp_environment_after_framework_import_is_too_late(self):
        """The loader rejects activation after framework initialization."""
        with patch.dict(os.environ, {}, clear=True), patch.object(
            _loader, "sys", SimpleNamespace(modules={"torch": object()})
        ), patch.object(
            _loader, "_component_root", return_value=self.native_root
        ), self.assertRaisesRegex(
            NativeComponentUnavailableError,
            f"HP-NATIVE-OPP-ACTIVATION-TOO-LATE.*source {self.native_root / 'set_env.bash'}",
        ):
            _loader.require_multicore_environment()
