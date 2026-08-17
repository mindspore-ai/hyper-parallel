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
"""Tests that torch-only optimizer symbols are lazy-exported from package __init__."""
import os
import subprocess
import sys
import textwrap
import unittest

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import hyper_parallel.core.optimizer as opt


_LAZY_NAMES = ("AdamW", "Muon", "ChainedOptimizer", "detect_dtensor_backend")
_TORCH_ONLY_MODULES = (
    "hyper_parallel.core.optimizer.adamw",
    "hyper_parallel.core.optimizer.muon",
    "hyper_parallel.core.optimizer.optimizer",
    "hyper_parallel.core.optimizer.dtensor_compat",
    "hyper_parallel.core.optimizer.utils",
    "hyper_parallel.core.optimizer.lr_scheduler",
    "hyper_parallel.core.optimizer.sharding_category",
    "hyper_parallel.core.optimizer.muon_shard",
)


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


class TestLazyOptimizerExports(unittest.TestCase):
    """Torch-only optimizer symbols stay off the eager import path until accessed."""

    def test_lazy_names_are_mapped(self):
        """
        Feature: optimizer lazy export map
        Description: inspect ``_LAZY_EXPORTS`` for torch-only names
        Expectation: AdamW/Muon/ChainedOptimizer/detect_dtensor_backend are mapped
        """
        for name in _LAZY_NAMES:
            self.assertIn(name, opt._LAZY_EXPORTS)

    def test_swap_api_stays_eager(self):
        """
        Feature: SwapOptimizer eager export
        Description: inspect public swap symbols on the optimizer package
        Expectation: they are in ``__all__`` and not lazy-loaded
        """
        for name in ("SwapOptimizer", "SwapOptimizerConfig", "swap_optimizer"):
            self.assertIn(name, opt.__all__)
            self.assertNotIn(name, opt._LAZY_EXPORTS)
            self.assertIsNotNone(getattr(opt, name))

    def test_unknown_attribute_still_raises(self):
        """
        Feature: optimizer ``__getattr__`` unknown name
        Description: access a name that is not a public export
        Expectation: AttributeError
        """
        with self.assertRaises(AttributeError):
            getattr(opt, "NotAnOptimizerSymbol")

    def test_getattr_resolves_and_caches_torch_symbols(self):
        """
        Feature: optimizer lazy getattr identity
        Description: import AdamW/Muon/ChainedOptimizer from package and submodule
        Expectation: package re-exports are the same objects as the implementing modules
        """
        from hyper_parallel.core.optimizer import AdamW, ChainedOptimizer, Muon
        from hyper_parallel.core.optimizer.adamw import AdamW as DirectAdamW
        from hyper_parallel.core.optimizer.muon import Muon as DirectMuon
        from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer as DirectChained

        self.assertIs(AdamW, DirectAdamW)
        self.assertIs(opt.AdamW, DirectAdamW)
        self.assertIs(Muon, DirectMuon)
        self.assertIs(ChainedOptimizer, DirectChained)

    def test_importing_package_does_not_load_torch_optimizer_modules(self):
        """
        Feature: optimizer package import without torch-only modules
        Description: import SwapOptimizer and factory APIs in an isolated process
        Expectation: adamw/muon/utils modules are not loaded
        """
        script = textwrap.dedent(
            f"""
            import sys
            from hyper_parallel.core.optimizer import (
                SwapOptimizer,
                SwapOptimizerConfig,
                get_hyper_optimizer,
                get_hyper_lr_scheduler,
                swap_optimizer,
            )
            loaded = [name for name in {_TORCH_ONLY_MODULES!r} if name in sys.modules]
            assert not loaded, loaded
            assert SwapOptimizer is not None
            assert SwapOptimizerConfig is not None
            assert swap_optimizer is not None
            assert get_hyper_optimizer is not None
            assert get_hyper_lr_scheduler is not None
            """
        )
        result = _run_isolated(script)
        self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)

    def test_accessing_adamw_loads_torch_optimizer_modules(self):
        """
        Feature: optimizer AdamW lazy load
        Description: import AdamW from the optimizer package in an isolated process
        Expectation: ``adamw`` module is loaded
        """
        script = textwrap.dedent(
            """
            import sys
            from hyper_parallel.core.optimizer import AdamW
            assert AdamW is not None
            assert "hyper_parallel.core.optimizer.adamw" in sys.modules
            """
        )
        result = _run_isolated(script)
        self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
