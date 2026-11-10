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
"""The admission guard's parser passes its embedded positive controls.

The self-test encodes the field incident that motivated the script: npu-smi's
fixed-width HBM column drops the space before the slash once the used value
reaches five digits, so a parser requiring that space reads a fully occupied
node as idle. A guard validated only against idle nodes fails open; this test
keeps the busy fixture a hard requirement.
"""

import shutil
import subprocess
import unittest
from pathlib import Path

from tests.common.mark_utils import arg_mark

_SCRIPT = (Path(__file__).resolve().parents[3]
           / ".agent" / "skills" / "hf-model-integration" / "scripts"
           / "node_guard.sh")


class TestNodeGuardSelfTest(unittest.TestCase):
    """Run the shell self-test (idle parse, fixed-width busy parse, fail-closed)."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_self_test_passes(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash not available on this platform")
        done = subprocess.run([bash, str(_SCRIPT), "self-test"],
                              capture_output=True, text=True, check=False)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("SELF-TEST PASS", done.stdout)
