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

import importlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hyper_parallel.core.multicore import _loader
from hyper_parallel.core.multicore._loader import NativeComponentUnavailableError


class TestMulticoreNative(unittest.TestCase):
    """Verify component-owned OPP environment and payload lookup."""

    def setUp(self) -> None:
        """Create a local native root with one vendor and both adapter directories."""
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
            self.native_root / "framework" / "mindspore"
            / "hyper_parallel_mega_moe_ms.so"
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
            vendor_root, adapter = _loader.get_multicore_paths("mindspore")
            opp_value = os.environ["ASCEND_CUSTOM_OPP_PATH"]
            library_value = os.environ["LD_LIBRARY_PATH"]

        self.assertEqual(vendor_root, self.vendor_root.resolve())
        self.assertEqual(adapter.name, "hyper_parallel_mega_moe_ms.so")
        self.assertEqual(opp_value, environment["ASCEND_CUSTOM_OPP_PATH"])
        self.assertEqual(library_value, environment["LD_LIBRARY_PATH"])

    def test_missing_opp_environment_requires_set_env(self):
        """The loader requires explicit environment activation before a framework is loaded."""
        with patch.dict(os.environ, {}, clear=True), patch.object(
            _loader, "sys", SimpleNamespace(modules={})
        ), patch.object(
            _loader, "_component_root", return_value=self.native_root
        ):
            with self.assertRaisesRegex(
                NativeComponentUnavailableError,
                f"HP-NATIVE-OPP-NOT-ACTIVATED.*source {self.native_root / 'set_env.bash'}",
            ):
                _loader.require_multicore_environment()

    def test_missing_opp_environment_after_framework_import_is_too_late(self):
        """The loader rejects activation after framework initialization."""
        with patch.dict(os.environ, {}, clear=True), patch.object(
            _loader, "sys", SimpleNamespace(modules={"mindspore": object()})
        ), patch.object(
            _loader, "_component_root", return_value=self.native_root
        ):
            with self.assertRaisesRegex(
                NativeComponentUnavailableError,
                f"HP-NATIVE-OPP-ACTIVATION-TOO-LATE.*source {self.native_root / 'set_env.bash'}",
            ):
                _loader.require_multicore_environment()

    def test_adapter_load_error_has_stable_native_diagnostic(self):
        """An ABI loader failure is wrapped with component and recovery context."""
        loader = SimpleNamespace(exec_module=Mock(side_effect=OSError("bad ABI")))
        spec = SimpleNamespace(loader=loader)
        module = object()
        with patch.object(
            _loader.importlib.util, "spec_from_file_location", return_value=spec
        ), patch.object(
            _loader.importlib.util, "module_from_spec", return_value=module
        ):
            with self.assertRaisesRegex(
                NativeComponentUnavailableError,
                "HP-NATIVE-FRAMEWORK-ADAPTER-LOAD-FAILED.*bad ABI",
            ):
                _loader.load_cpython_extension("hp_bad_adapter", Path("bad.so"))

        self.assertNotIn("hp_bad_adapter", _loader.sys.modules)

    def test_adapter_create_error_has_stable_native_diagnostic(self):
        """A dlopen failure during module creation is wrapped with recovery context."""
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=Mock()))
        with patch.object(
            _loader.importlib.util, "spec_from_file_location", return_value=spec
        ), patch.object(
            _loader.importlib.util, "module_from_spec", side_effect=ImportError("file too short")
        ):
            with self.assertRaisesRegex(
                NativeComponentUnavailableError,
                "HP-NATIVE-FRAMEWORK-ADAPTER-LOAD-FAILED.*file too short",
            ):
                _loader.load_cpython_extension("hp_broken_adapter", Path("broken.so"))

        self.assertNotIn("hp_broken_adapter", _loader.sys.modules)

    def test_st_activation_falls_back_from_stale_wheel_to_source_payload(self):
        """A stale wheel locator cannot block a valid PYTHONPATH payload activation."""
        stale_script = self.root / "stale-wheel-set-env.bash"
        stale_script.write_text("echo stale wheel >&2\nreturn 1\n", encoding="utf-8")
        source_script = self.root / "source-set-env.bash"
        source_script.write_text(
            f"export ASCEND_CUSTOM_OPP_PATH='{self.vendor_root}'\n"
            f"export LD_LIBRARY_PATH='{self.vendor_root / 'op_api/lib'}'\n",
            encoding="utf-8",
        )
        cann_environment = {
            "ASCEND_HOME_PATH": "/cann",
            "ASCEND_OPP_PATH": "/cann/opp",
            "ASCEND_AICPU_PATH": "/cann/aicpu",
            "RANK_TABLE_FILE": "/tmp/stale-16-card-rank-table.json",
            "RANK_ID": "7",
            "RANK_SIZE": "16",
        }
        with patch.dict(os.environ, cann_environment, clear=True):
            helper_path = Path(__file__).resolve().parents[2] / "common" / "multicore_test_env.py"
            helper_spec = importlib.util.spec_from_file_location("hp_multicore_test_env", helper_path)
            self.assertIsNotNone(helper_spec)
            self.assertIsNotNone(helper_spec.loader)
            test_env = importlib.util.module_from_spec(helper_spec)
            helper_spec.loader.exec_module(test_env)
            with patch.object(
                test_env,
                "multicore_activation_scripts",
                return_value=[stale_script, source_script],
            ):
                test_env.prepare_multicore_test_environment()

            self.assertEqual(
                os.environ["ASCEND_CUSTOM_OPP_PATH"],
                str(self.vendor_root),
                f"Expected source payload vendor={self.vendor_root}, "
                f"got={os.environ.get('ASCEND_CUSTOM_OPP_PATH')}",
            )
            with test_env.without_inherited_rank_environment():
                for variable in test_env.INHERITED_RANK_VARIABLES:
                    self.assertNotIn(
                        variable,
                        os.environ,
                        f"The explicit msrun case inherited stale rank metadata: variable={variable}",
                    )
            for variable in test_env.INHERITED_RANK_VARIABLES:
                self.assertEqual(
                    os.environ[variable],
                    cann_environment[variable],
                    f"The ST helper did not restore parent rank metadata: variable={variable}",
                )
