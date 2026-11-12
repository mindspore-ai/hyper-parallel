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
"""Public module and RL examples share a namespace in either import order."""

from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TestExampleNamespaces(unittest.TestCase):
    """Use fresh interpreters outside the checkout to expose import-order collisions."""

    def test_public_examples_coexist_with_rl_examples(self) -> None:
        """Import both example trees before and after adding the independent RL root."""
        rl_root = Path(find_spec("hyper_parallel").origin).parent / "rl"
        for family_first in (True, False):
            program = f"""
import sys
if {family_first!r}:
    from examples.torch.module_family import DecoderConfig, FamilyDecoder
sys.path.insert(0, {str(rl_root)!r})
from examples import gsm8k
from examples.torch.module_family import DecoderConfig, FamilyDecoder
import torch
model = FamilyDecoder(DecoderConfig())
assert model(torch.ones(1, 2, 8)).shape == (1, 2, 8)
assert gsm8k.__file__
"""
            with self.subTest(family_first=family_first), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    [sys.executable, "-c", program], cwd=directory,
                    capture_output=True, text=True, check=False, timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
