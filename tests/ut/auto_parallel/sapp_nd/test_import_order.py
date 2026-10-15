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
"""Every ND module imports as the first of its package.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/test_import_order.py -v
"""
import subprocess
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Run in a fresh interpreter. Before each module it drops every auto_parallel
# module already loaded, so each one is imported first, as a script or a test
# needing only it imports it, and an import cycle cannot hide behind the order
# in which the suite happened to load the package.
_PROBE = r"""
import importlib
import pkgutil
import sys

PREFIX = "hyper_parallel.auto_parallel"
names = [PREFIX + "._layer_census"]
for root in (PREFIX + ".sapp_nd", PREFIX + ".config_adapter"):
    names.append(root)
    names += [m.name for m in pkgutil.walk_packages(importlib.import_module(root).__path__, root + ".")]

cycles = []
for name in sorted(names):
    for loaded in [m for m in sys.modules if m == PREFIX or m.startswith(PREFIX + ".")]:
        del sys.modules[loaded]
    try:
        importlib.import_module(name)
    except ImportError as exc:
        if "partially initialized module" in str(exc):
            cycles.append(name + ": " + str(exc).splitlines()[0])
print(f"{len(names)} modules imported first")
print("\n".join(cycles))
sys.exit(1 if cycles else 0)
"""


class TestImportOrder(unittest.TestCase):
    """No ND module needs another to have been imported before it."""

    def test_every_module_imports_first(self) -> None:
        """
        Feature: SAPP-ND package imports.
        Description: Import each module of sapp_nd, config_adapter and the layer census as the first of its package.
        Expectation: None fails on a partially initialized module, the mark of an import cycle.
        """
        result = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=300,
            check=False,
        )
        self.assertEqual(
            result.returncode, 0,
            f"modules that cannot be imported first: returncode={result.returncode}\n"
            f"{result.stdout}{result.stderr[-2000:]}",
        )


if __name__ == "__main__":
    unittest.main()
