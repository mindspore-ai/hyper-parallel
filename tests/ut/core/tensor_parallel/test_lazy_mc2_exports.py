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
# pylint: disable=C0413,C0415,protected-access
"""Tests that MC2 torch-only symbols are lazy-exported from package __init__."""
import os
import subprocess
import sys
import textwrap
import unittest

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import hyper_parallel as hp
import hyper_parallel.core.tensor_parallel as tp


_MC2_NAMES = ("MC2Linear", "MC2ColwiseParallel", "MC2RowwiseParallel")


def _run_isolated(script: str) -> subprocess.CompletedProcess:
    """Run ``script`` in a fresh interpreter so ``sys.modules`` is not polluted."""
    env = os.environ.copy()
    env["HYPER_PARALLEL_PLATFORM"] = "torch"
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


class TestLazyMC2Exports(unittest.TestCase):
    """MC2 symbols stay off the eager import path until accessed."""

    def test_lazy_names_are_mapped_in_both_packages(self):
        """
        Feature: MC2 lazy export map
        Description: inspect ``_LAZY_EXPORTS`` and ``__all__`` on both packages
        Expectation: MC2Linear/MC2ColwiseParallel/MC2RowwiseParallel are mapped
        """
        for name in _MC2_NAMES:
            self.assertIn(name, tp._LAZY_EXPORTS)
            self.assertIn(name, hp._LAZY_EXPORTS)
            self.assertIn(name, tp.__all__)
            self.assertIn(name, hp.__all__)

    def test_unknown_attribute_still_raises(self):
        """
        Feature: MC2 ``__getattr__`` unknown name
        Description: access names that are not public exports
        Expectation: AttributeError
        """
        with self.assertRaises(AttributeError):
            getattr(tp, "NotATensorParallelSymbol")
        with self.assertRaises(AttributeError):
            getattr(hp, "NotAHyperParallelSymbol")

    def test_getattr_resolves_and_caches_mc2_symbols(self):
        """
        Feature: MC2 lazy getattr identity
        Description: import MC2 symbols from package, tensor_parallel, and implementing modules
        Expectation: all import paths resolve to the same class objects
        """
        from hyper_parallel import MC2ColwiseParallel, MC2Linear, MC2RowwiseParallel
        from hyper_parallel.core.tensor_parallel import (
            MC2ColwiseParallel as TpMC2ColwiseParallel,
            MC2Linear as TpMC2Linear,
            MC2RowwiseParallel as TpMC2RowwiseParallel,
        )
        from hyper_parallel.core.tensor_parallel.mc2 import MC2Linear as DirectMC2Linear
        from hyper_parallel.core.tensor_parallel.mc2_style import (
            MC2ColwiseParallel as DirectMC2ColwiseParallel,
            MC2RowwiseParallel as DirectMC2RowwiseParallel,
        )

        self.assertIs(MC2Linear, DirectMC2Linear)
        self.assertIs(TpMC2Linear, DirectMC2Linear)
        self.assertIs(tp.MC2Linear, DirectMC2Linear)
        self.assertIs(hp.MC2Linear, DirectMC2Linear)
        self.assertIs(MC2ColwiseParallel, DirectMC2ColwiseParallel)
        self.assertIs(TpMC2ColwiseParallel, DirectMC2ColwiseParallel)
        self.assertIs(MC2RowwiseParallel, DirectMC2RowwiseParallel)
        self.assertIs(TpMC2RowwiseParallel, DirectMC2RowwiseParallel)

    def test_importing_packages_does_not_load_mc2_modules(self):
        """
        Feature: package import without MC2 modules
        Description: import hyper_parallel and tensor_parallel in an isolated process
        Expectation: mc2/mc2_style are not loaded; ColwiseParallel remains available
        """
        script = textwrap.dedent(
            """
            import sys
            import hyper_parallel
            import hyper_parallel.core.tensor_parallel as tp
            loaded = [name for name in sys.modules if name in (
                "hyper_parallel.core.tensor_parallel.mc2",
                "hyper_parallel.core.tensor_parallel.mc2_style",
            )]
            assert not loaded, loaded
            assert tp.ColwiseParallel is not None
            assert hyper_parallel.ColwiseParallel is not None
            """
        )
        result = _run_isolated(script)
        self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)

    def test_accessing_mc2_symbol_loads_mc2_modules(self):
        """
        Feature: MC2Linear lazy load
        Description: import MC2Linear from hyper_parallel in an isolated process
        Expectation: mc2 and mc2_style modules are loaded
        """
        script = textwrap.dedent(
            """
            import sys
            import hyper_parallel
            from hyper_parallel import MC2ColwiseParallel, MC2Linear
            assert MC2Linear is not None
            assert MC2ColwiseParallel is not None
            assert "hyper_parallel.core.tensor_parallel.mc2" in sys.modules
            assert "hyper_parallel.core.tensor_parallel.mc2_style" in sys.modules
            """
        )
        result = _run_isolated(script)
        self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
