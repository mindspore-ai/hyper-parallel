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
"""Verify source-defined operator interfaces are packaged without code generation."""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OPS_DIR = ROOT / "hyper_parallel/components/ops"


class TestOperatorPackaging(unittest.TestCase):
    """Preserve source interfaces and the typing marker in metadata and Python builds."""

    def test_build_packages_source_interfaces_without_generating_stubs(self) -> None:
        """Packaging works with no generator script and leaves source declarations untouched."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("setup.py", "README.md", "requirements.txt"):
                shutil.copyfile(ROOT / name, root / name)
            target = root / "hyper_parallel/components/ops"
            shutil.copytree(OPS_DIR, target, ignore=shutil.ignore_patterns("__pycache__"))
            for package in (target.parent, target.parent.parent):
                (package / "__init__.py").touch()
            source = target / "__init__.py"
            before = source.read_bytes(), source.stat().st_mtime_ns
            metadata = subprocess.run(
                [sys.executable, "setup.py", "egg_info"], cwd=root,
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(metadata.returncode, 0, metadata.stdout + metadata.stderr)
            output = root / "build/lib"
            built = subprocess.run(
                [sys.executable, "setup.py", "build_py", "--build-lib", str(output)],
                cwd=root, capture_output=True, text=True, check=False,
            )
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            built_ops = output / "hyper_parallel/components/ops"
            self.assertEqual((built_ops / "__init__.py").read_bytes(), source.read_bytes())
            self.assertEqual((built_ops / "impl/swiglu.py").read_bytes(), (target / "impl/swiglu.py").read_bytes())
            self.assertTrue((built_ops / "py.typed").is_file())
            self.assertFalse(list(built_ops.glob("*.pyi")))
            self.assertEqual((source.read_bytes(), source.stat().st_mtime_ns), before)
