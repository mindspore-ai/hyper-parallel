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
"""Check the native-Torch import rule through the actual repository lint plugin."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.common.mark_utils import arg_mark


class TestNativeTorchLint(unittest.TestCase):
    """The removed backend layer must not prevent valid native Torch code."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_torch_imports_pass_and_removed_framework_is_rejected(self) -> None:
        """
        Feature: Torch-only library architecture.
        Description: Run the custom lint plugin over library-scoped import statements.
        Expectation: Native Torch imports pass while MindSpore imports produce the dedicated violation.
        """
        executable = shutil.which("pylint")
        self.assertIsNotNone(executable, "The repository lint dependency must be installed")
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "hyper_parallel" / "probe.py"
            target.parent.mkdir()
            for statement, rejected in (("import torch", False), ("from torch import nn", False),
                                        ("import mindspore", True), ("from mindspore import ops", True)):
                with self.subTest(statement=statement):
                    target.write_text(statement + "\n", encoding="utf-8")
                    result = subprocess.run(
                        [executable, "--disable=all", "--enable=forbidden-backend-import", str(target)],
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual("forbidden-backend-import" in result.stdout, rejected, result.stdout+result.stderr)
