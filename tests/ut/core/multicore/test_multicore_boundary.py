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
from pathlib import Path
import unittest

import hyper_parallel
from hyper_parallel.core import multicore
from hyper_parallel.core.multicore.modules.mega_moe.module import MegaMoeExperts
from hyper_parallel.core.multicore.shmem import _bindings
from hyper_parallel.platform.platform import Platform
from tests.common.mark_utils import arg_mark


class TestMulticoreBoundary(unittest.TestCase):
    """Check the component contract without initializing a native runtime."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_business_symbols_are_exported_only_by_multicore(self):
        """Verify the Multicore public export boundary.

        Feature: Multicore public exports.
        Description: Inspect the component and HyperParallel root package symbols.
        Expectation: MegaMoeExperts is exported only by the explicit Multicore module.
        """
        self.assertEqual(MegaMoeExperts.__name__, "MegaMoeExperts")
        self.assertIs(multicore.MegaMoeExperts, MegaMoeExperts)
        self.assertEqual(multicore.__all__, ["MegaMoeExperts"])
        for name in ("MegaMoeExperts", "MulticoreModule", "mega_moe", "mega_moe_grad"):
            self.assertNotIn(name, hyper_parallel.__all__)
            self.assertFalse(hasattr(hyper_parallel, name))
        self.assertFalse(hasattr(multicore, "__getattr__"))
        self.assertIsNone(_bindings._manager)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_platform_has_no_component_hooks(self):
        """Verify removal of shared Platform hooks.

        Feature: Multicore Platform independence.
        Description: Inspect the shared Platform class for former component hooks.
        Expectation: Neither Multicore nor SHMEM can be requested from Platform.
        """
        self.assertFalse(hasattr(Platform, "get_symmetric_memory_handler"))
        self.assertFalse(hasattr(Platform, "get_multicore_handler"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_runtime_sources_do_not_import_platform_or_mindspore(self):
        """Verify production sources stay Torch-only and Platform-free.

        Feature: Multicore runtime convergence.
        Description: Parse imports in the converged production Python tree.
        Expectation: Runtime sources import neither MindSpore nor HyperParallel Platform.
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
                        self.assertFalse(module.startswith(("mindspore", "hyper_parallel.platform")))
