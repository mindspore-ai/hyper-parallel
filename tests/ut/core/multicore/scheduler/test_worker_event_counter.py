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
"""Compile production event publication and polling against CPU API models."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import hyper_parallel


class TestWorkerEventCounter(unittest.TestCase):
    """Keep Cube/Vector event publication and time units valid without NPU hardware."""

    def test_production_event_publication_and_polling(self) -> None:
        """Check concurrent increments and actual refresh thresholds on both device generations."""
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("g++ is required for the native event-writer test")
        source = Path(hyper_parallel.__file__).resolve().parent / "core/multicore/ops/runtime/worker_kernel.h"
        begin = "  __aicore__ inline void AtomicAddForAllEventCounters("
        end = "  __aicore__ inline void ExecuteTaskFast("
        source_text = source.read_text(encoding="utf-8")
        method = begin + source_text.split(begin, 1)[1].split(end, 1)[0]
        constants = source_text.split("#if __NPU_ARCH__ == 3510", 1)[1].split("// The selected SwiGLU", 1)[0]
        constants = "#if __NPU_ARCH__ == 3510" + constants
        wait_begin = "  __aicore__ inline void WaitForDependency("
        wait_end = "  __aicore__ inline void TriggerEvent("
        wait_method = wait_begin + source_text.split(wait_begin, 1)[1].split(wait_end, 1)[0]
        with tempfile.TemporaryDirectory(prefix="hp-event-counter-") as directory:
            root = Path(directory)
            (root / "event_counter_method.inc").write_text(method, encoding="utf-8")
            (root / "event_poll_constants.inc").write_text(constants, encoding="utf-8")
            (root / "event_poll_method.inc").write_text(wait_method, encoding="utf-8")
            for architecture in (2201, 2203, 3510):
                for core in ("cube", "vector"):
                    with self.subTest(architecture=architecture, core=core):
                        binary = root / f"event-{architecture}-{core}"
                        defines = [f"-D__NPU_ARCH__={architecture}"]
                        if core == "cube":
                            defines.append("-D__DAV_CUBE__=1")
                        compiled = subprocess.run([
                            compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread",
                            *defines, "-I", directory, str(Path(__file__).with_name("event_counter_probe.cpp")),
                            "-o", str(binary),
                        ], capture_output=True, text=True, check=False)
                        self.assertEqual(compiled.returncode, 0,
                                         f"expected=0 actual={compiled.returncode} stderr={compiled.stderr}")
                        result = subprocess.run([str(binary)], capture_output=True, text=True,
                                                check=False, timeout=10)
                        self.assertEqual(result.returncode, 0,
                                         f"expected=0 actual={result.returncode} stderr={result.stderr}")
                        self.assertIn("EVENT_COUNTER_PASS", result.stdout)
                        self.assertIn("EVENT_POLL_PASS", result.stdout)
