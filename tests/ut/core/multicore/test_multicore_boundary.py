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

"""Verify component exports and absence of framework dispatch."""

import ast
import sys
import unittest
from pathlib import Path

import hyper_parallel
from hyper_parallel.core import multicore
from tests.common.mark_utils import arg_mark


class TestMulticoreBoundary(unittest.TestCase):
    """Check the component contract without initializing a native runtime."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_business_symbols_are_exported_only_by_multicore(self):
        """Verify the Multicore public export boundary.

        Feature: Multicore public exports.
        Description: Inspect the component and HyperParallel root package symbols.
        Expectation: Business modules and profiler are exported only by Multicore.
        """
        self.assertCountEqual(multicore.__all__, [
            "HyperMegaMhc", "MegaMhc", "MhcState", "MegaMhcOutput", "GateOutput", "GateBackend",
            "MegaGate", "MegaMoeExperts", "MegaKernelDiagnostics",
            "prepare_mega_kernel_diagnostics", "validate_mega_kernel_diagnostic_contract",
        ])
        self.assertTrue(callable(multicore.prepare_mega_kernel_diagnostics))
        for name in (
            "HyperMegaMhc",
            "MegaMhc",
            "MhcState",
            "MegaMhcOutput",
            "GateOutput",
            "GateBackend",
            "MegaGate",
            "MegaMoeExperts",
            "MulticoreModule",
            "mega_moe",
            "mega_moe_grad",
        ):
            self.assertNotIn(name, hyper_parallel.__all__)
            self.assertFalse(hasattr(hyper_parallel, name))
        self.assertNotIn("hyper_parallel_shmem_torch", sys.modules)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_runtime_sources_import_no_platform_layer(self):
        """Verify production sources stay Torch-only.

        Feature: Multicore runtime convergence.
        Description: Parse imports in the converged production Python tree.
        Expectation: Runtime sources import no HyperParallel Platform layer.
        """
        root = Path(multicore.__file__).parent
        for path in root.rglob("*.py"):
            # Benchmark examples import common MoE as an independent reference.
            if any(part in {"lib", "examples"} for part in path.relative_to(root).parts):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                for module in modules:
                    with self.subTest(path=path, module=module):
                        self.assertFalse(module.startswith("hyper_parallel.platform"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_diagnostics_dependency_boundaries(self):
        """Feature: MegaKernel diagnostics.
        Description: Keep common code tool-neutral and both tools independent of business adapters.
        Expectation: Diagnostics dependency boundaries.
        """
        root = Path(multicore.__file__).parent / "diagnostics"
        for path in root.rglob("*.py"):
            relative = path.relative_to(root)
            forbidden = ["hyper_parallel.core.multicore.modules"]
            if relative.parts[0] in ("common", "api.py", "__init__.py"):
                forbidden.extend(("diagnostics.profiler", "diagnostics.dump"))
            elif relative.parts[0] == "dump":
                forbidden.append("diagnostics.profiler")
            elif relative.parts[0] == "profiler":
                forbidden.append("diagnostics.dump")
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                modules = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                           else [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                for module in modules:
                    with self.subTest(path=path, module=module):
                        self.assertFalse(any(item in module for item in forbidden))
