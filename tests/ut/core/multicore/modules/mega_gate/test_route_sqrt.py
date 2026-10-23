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
"""Compile the actual Gate Sqrt dispatch with CPU spies for the public SDK overloads."""

from pathlib import Path
import subprocess
import tempfile
import unittest

import hyper_parallel


_ROOT = Path(hyper_parallel.__file__).resolve().parent.parent
_MODEL = r'''
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
int observed_mode = -1;
int observed_count = 0;
int default_calls = 0;
int precise_calls = 0;
bool precise_domain_valid = true;
template<typename T> void Sqrt(T* output, T* input, int count) {
    observed_mode = 0;
    observed_count = count;
    ++default_calls;
    for (int index = 0; index < count; ++index) {
        output[index] = std::nextafter(std::sqrt(input[index]), std::numeric_limits<float>::infinity());
    }
}
#if __NPU_ARCH__ == 3510
enum class SqrtAlgo { PRECISION_0ULP_FTZ_FALSE = 1 };
struct SqrtConfig { SqrtAlgo algo; };
template<typename T> using LocalTensor = T*;
struct MaskBuffer {
    uint8_t data[24];
    template<typename T> T* Get() { return reinterpret_cast<T*>(data); }
} precision_mask_buffer;
enum class CMPMODE { LE };
enum class SELMODE { VSEL_TENSOR_TENSOR_MODE };
enum { PIPE_V };
template<int> void PipeBarrier() {}
void CompareScalar(uint8_t* output, float* input, float limit, CMPMODE, int count) {
    for (int index = 0; index < count; ++index) { output[index] = input[index] <= limit; }
}
void Mins(float* output, float* input, float limit, int count) {
    for (int index = 0; index < count; ++index) { output[index] = std::fmin(input[index], limit); }
}
void Select(float* output, uint8_t* mask, float* left, float* right, SELMODE, int count) {
    for (int index = 0; index < count; ++index) { output[index] = mask[index] ? left[index] : right[index]; }
}
template<typename T, const SqrtConfig& config> void Sqrt(T* output, T* input, int count) {
    observed_mode = static_cast<int>(config.algo);
    ++precise_calls;
    for (int index = 0; index < count; ++index) {
        precise_domain_valid &= input[index] >= 0 && input[index] <= 8.507058665963222e37f;
        output[index] = std::sqrt(input[index]);
    }
}
#endif
int main() {
    for (int rows : {1, 3}) {
        constexpr int expert_align_ = 8;
        float input[24], original[24], output[24];
        for (int index = 0; index < 24; ++index) {
            input[index] = index == 0 ? 0.0f : index * 0.125f;
            output[index] = -7.0f;
        }
        input[1] = std::numeric_limits<float>::denorm_min();
        input[2] = std::numeric_limits<float>::max() / 4;
        input[3] = std::numeric_limits<float>::max();
        input[4] = std::numeric_limits<float>::infinity();
        input[5] = std::numeric_limits<float>::quiet_NaN();
        for (int index = 0; index < 24; ++index) { original[index] = input[index]; }
        default_calls = 0;
        precise_calls = 0;
HP_SQRT_CALL
        const int expected_mode = __NPU_ARCH__ == 3510 ? 1 : 0;
        std::cout << "arch=" << __NPU_ARCH__ << " mode=" << observed_mode
                  << " expected=" << expected_mode << " count=" << observed_count << '\n';
        if (observed_mode != expected_mode || observed_count != rows * expert_align_) { return 1; }
        if (default_calls != 1 || precise_calls != expected_mode || !precise_domain_valid) { return 1; }
        for (int index = 0; index < 24; ++index) {
            float expected = -7.0f;
            if (index < observed_count) {
                expected = std::sqrt(original[index]);
                if (expected_mode == 0 || !(original[index] <= 8.507058665963222e37f)) {
                    expected = std::nextafter(expected, std::numeric_limits<float>::infinity());
                }
            }
            if (output[index] != expected && !(std::isnan(output[index]) && std::isnan(expected))) { return 1; }
        }
    }
    return 0;
}
'''


def _sqrt_call(source: str) -> str:
    """Extract the production call and architecture guard, without rewriting the selection."""
    method = source[source.index("void SqrtScore()"):source.index("void AddBias()")]
    start = method.index("LocalTensor<float> output = output_queue.AllocTensor<float>();")
    start = method.index("\n", start)
    return method[start:method.index("output_queue.EnQue(output)", start)]


class TestRouteSqrt(unittest.TestCase):
    """A5 selects 0 ULP within its domain; A2/A3 retain their original SDK overload."""

    def test_platform_precision_selection_and_count(self) -> None:
        """Observe overload/config and tail bounds; this is not a CANN numerical emulator."""
        header = _ROOT / "hyper_parallel/core/multicore/ops/hyper_mega_gate/op_kernel/route_pipeline.h"
        source = _MODEL.replace("HP_SQRT_CALL", _sqrt_call(header.read_text(encoding="utf-8")))
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "gate-sqrt"
            for architecture in (2201, 2203, 3510):
                with self.subTest(architecture=architecture):
                    result = subprocess.run(
                        ["c++", "-std=c++17", f"-D__NPU_ARCH__={architecture}", "-x", "c++", "-",
                         "-o", str(executable)], input=source, capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 0, f"expected compile=0 actual={result.stderr}")
                    result = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
                    self.assertEqual(result.returncode, 0,
                                     f"expected platform mode/count and intact tail actual={result.stdout}")
