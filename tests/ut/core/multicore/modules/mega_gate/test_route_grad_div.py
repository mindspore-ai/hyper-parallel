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
"""Observe the production gradient Div precision dispatch without requiring an NPU."""

from pathlib import Path
import subprocess
import tempfile
import unittest

import hyper_parallel


_ROOT = Path(hyper_parallel.__file__).resolve().parent.parent
_MODEL = r'''
#include <cstdint>
#include <iostream>
int observed_mode = -1;
int observed_count = -1;
template<typename T> void Div(T* output, T* left, T* right, int count) {
    observed_mode = 0;
    observed_count = count;
    for (int index = 0; index < count; ++index) { output[index] = left[index] / right[index]; }
}
template<typename T> void Mul(T* output, T* left, T* right, int count) {
    observed_mode = 2;
    observed_count = count;
    for (int index = 0; index < count; ++index) { output[index] = left[index] * right[index]; }
}
#if __NPU_ARCH__ == 3510
enum class DivAlgo { PRECISION_0ULP_FTZ_FALSE = 1 };
struct DivConfig { DivAlgo algo; };
template<typename T, const DivConfig& config> void Div(T* output, T* left, T* right, int count) {
    Div(output, left, right, count);
    observed_mode = static_cast<int>(config.algo);
}
#endif
int main() {
    for (bool multiply : {false, true}) {
        for (int element_count : {1, 6, 18, 21}) {
            float left_value[24], right_value[24], destination[24];
            for (int index = 0; index < 24; ++index) {
                left_value[index] = (index - 12) * 0.125f;
                right_value[index] = (index + 1) * 0.25f;
                destination[index] = -7.0f;
            }
HP_BINARY_CALL
            const int expected_mode = multiply ? 2 : (__NPU_ARCH__ == 3510 ? 1 : 0);
            std::cout << "arch=" << __NPU_ARCH__ << " mode=" << observed_mode
                      << " expected=" << expected_mode << " count=" << observed_count << '\n';
            if (observed_mode != expected_mode || observed_count != element_count) { return 1; }
            for (int index = 0; index < 24; ++index) {
                float expected = -7.0f;
                if (index < element_count) {
                    expected = multiply ? left_value[index] * right_value[index] : left_value[index] / right_value[index];
                }
                if (destination[index] != expected) { return 1; }
            }
        }
    }
    return 0;
}
'''


class TestRouteGradDiv(unittest.TestCase):
    """Preserve A2/A3 and Mul dispatch while selecting the public A5 precision mode."""

    def test_platform_precision_selection_and_bounds(self) -> None:
        """Compile actual RunBinary dispatch; CPU arithmetic is not a CANN accuracy model."""
        header = _ROOT / "hyper_parallel/core/multicore/ops/hyper_mega_gate_grad/op_kernel/route_grad_pipeline.h"
        source = header.read_text(encoding="utf-8")
        method = source[source.index("void RunBinary("):source.index("void RunAdd(")]
        dispatch = method[method.index("if (multiply)"):method.index("output_queue.EnQue(destination)")]
        model = _MODEL.replace("HP_BINARY_CALL", dispatch)
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "gate-grad-div"
            for architecture in (2201, 2203, 3510):
                with self.subTest(architecture=architecture):
                    result = subprocess.run(
                        ["c++", "-std=c++17", f"-D__NPU_ARCH__={architecture}", "-x", "c++", "-",
                         "-o", str(executable)], input=model, capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 0, f"expected compile=0 actual={result.stderr}")
                    result = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
                    self.assertEqual(result.returncode, 0,
                                     f"expected precision mode/count and intact tail actual={result.stdout}")
