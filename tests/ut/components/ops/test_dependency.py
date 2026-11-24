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
"""Optional dependency failure classification and required-symbol contracts."""

import unittest
from types import ModuleType
from unittest.mock import call, patch

from hyper_parallel.components.ops import _dependency
from hyper_parallel.components.ops.registry import ImplementationUnavailable


class TestOptionalDependencies(unittest.TestCase):
    """Missing optional packages permit fallback; broken installed packages do not."""

    def test_package_is_returned_without_wrapping(self) -> None:
        """Successful imports preserve the package object used by adapters."""
        package = ModuleType("extension")
        with patch.object(_dependency, "import_module", return_value=package) as loader:
            self.assertIs(_dependency.require_package("extension"), package)
        loader.assert_called_once_with("extension")

    def test_missing_package_or_parent_is_unavailable(self) -> None:
        """A missing parent of a dotted optional package is also a fallback condition."""
        for missing in ("extension", "extension.kernels"):
            error = ModuleNotFoundError(name=missing)
            with self.subTest(missing=missing), patch.object(_dependency, "import_module", side_effect=error):
                with self.assertRaisesRegex(
                    ImplementationUnavailable, "missing optional package extension.kernels",
                ) as caught:
                    _dependency.require_package("extension.kernels")
                self.assertIs(caught.exception.__cause__, error)

    def test_broken_package_errors_propagate(self) -> None:
        """An installed extension's internal dependency or ABI error must remain visible."""
        errors = (ModuleNotFoundError(name="internal_dependency"), ModuleNotFoundError(), ImportError("ABI mismatch"))
        for error in errors:
            with self.subTest(error=error), patch.object(_dependency, "import_module", side_effect=error):
                with self.assertRaises(type(error)) as caught:
                    _dependency.require_package("extension")
                self.assertIs(caught.exception, error)

    def test_required_symbol_must_be_callable(self) -> None:
        """Missing symbols and noncallable placeholders cannot pass the dependency check."""
        package = ModuleType("extension")
        package.kernel = None
        for name in ("missing", "kernel"):
            with self.subTest(symbol=name), self.assertRaisesRegex(
                ImplementationUnavailable, f"missing operator {name}",
            ):
                _dependency.require_symbol(package, name)
        package.kernel = lambda: None
        self.assertIsNone(_dependency.require_symbol(package, "kernel"))

    def test_cann_checks_all_symbols_before_returning_adapter(self) -> None:
        """An unavailable second symbol must reject the entire multi-kernel adapter."""
        package = ModuleType("extension")
        package.forward = lambda: None

        def kernel(value):
            """Keep the adapter identity observable after dependency checks."""
            return value

        with patch.object(_dependency, "require_package", return_value=package) as loader:
            with self.assertRaisesRegex(ImplementationUnavailable, "missing operator backward"):
                _dependency.load_cann(kernel, "forward", "backward")
            package.backward = lambda: None
            self.assertIs(_dependency.load_cann(kernel, "forward", "backward"), kernel)
        self.assertEqual(loader.call_args_list, [call("cann_ops_transformer")] * 2)
