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
"""Verify real MindSpore APIs reach their RaggedShard whitelist names."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import pytest

pytest.importorskip("mindspore")


class TestMindSporeRaggedElementwise(unittest.TestCase):
    """Run MindSpore DTensor dispatch in an isolated backend process."""

    def test_public_interfaces_dispatch_real_primitive_names(self) -> None:
        """All requested interfaces must run without synthetic operator names."""
        case_file = Path(__file__).with_name("ragged_elementwise_case.py")
        env = os.environ.copy()
        env["HYPER_PARALLEL_PLATFORM"] = "mindspore"
        records = []
        for selection in ("regular", "clone", "gelu_ext"):
            result = subprocess.run(
                [sys.executable, str(case_file), selection],
                cwd=Path(__file__).parents[5],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(
                result.returncode, 0,
                msg=f"selection={selection}\nstdout={result.stdout}\nstderr={result.stderr}",
            )
            record_line = next(
                (line for line in result.stdout.splitlines() if line.startswith("RAGGED_OP_RECORDS=")),
                None,
            )
            self.assertIsNotNone(
                record_line,
                msg=f"selection={selection}: missing records in stdout={result.stdout!r}",
            )
            records.extend(json.loads(record_line.partition("=")[2]))
        self.assertEqual(len(records), 30, msg=f"expected=30 interfaces, got={len(records)}")
