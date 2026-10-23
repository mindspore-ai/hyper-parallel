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
"""Execute the real RMSNorm cache-write block with poisoned CPU destination storage."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import hyper_parallel


_ROOT = Path(hyper_parallel.__file__).resolve().parent.parent
_MODEL = r'''
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#define __aicore__
constexpr unsigned NUM_PER_BLK_FP32 = 8;
bool atomic = false;
template<typename T> struct LocalTensor { T* data; };
struct GlobalTensor {
    float* data;
    GlobalTensor operator[](unsigned offset) { return {data + offset}; }
};
struct Queue {
    float values[64];
    template<typename T> LocalTensor<T> DeQue() { return {values}; }
    template<typename T> void FreeTensor(LocalTensor<T>) {}
};
template<typename T> void SetAtomicAdd() { atomic = true; }
void SetAtomicNone() { atomic = false; }
void DataCopy(GlobalTensor destination, LocalTensor<float> source, unsigned count) {
    for (unsigned index = 0; index < count; ++index) {
        if (atomic) { destination.data[index] += source.data[index]; }
        else { destination.data[index] = source.data[index]; }
    }
}
template<typename T> void DataCopyCustom(GlobalTensor destination, LocalTensor<T> source, unsigned count) {
    DataCopy(destination, source, count);
}
struct DataCopyExtParams { unsigned count, bytes, src_stride, dst_stride, reserved; };
void DataCopyPad(GlobalTensor destination, LocalTensor<float> source, DataCopyExtParams params) {
    DataCopy(destination, source, params.bytes / sizeof(float));
}
struct Kernel {
    GlobalTensor rstdGm;
    Queue outQueueRstd;
    unsigned row_factor = 32;
HP_CACHE_WRITE
};
int main() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (float initial : {17.0f, nan}) {
        for (unsigned count : {1U, 7U, 8U, 32U}) {
            float destination[96];
            std::fill_n(destination, 96, initial);
            Kernel kernel{{destination}, {}};
            std::fill_n(kernel.outQueueRstd.values, 64, 0.5f);
            kernel.CopyOutRstd(1, count);
            bool correct = true;
            for (unsigned index = 0; index < 96; ++index) {
                float expected = index >= 32 && index < 32 + count ? 0.5f : initial;
                correct &= std::isnan(expected) ? std::isnan(destination[index]) : destination[index] == expected;
            }
            std::cout << "count=" << count << " poisoned=" << std::isnan(initial)
                      << " cache_overwritten=" << correct << '\n';
            if (!correct || atomic) { return 1; }
        }
    }
    return 0;
}
'''


def _cache_write_block(adapter: str) -> str:
    """Read the resulting CopyOutRstd method from the complete production adapter hunk."""
    section = next(hunk for hunk in adapter.split("\n@@")[1:] if "void CopyOutRstd(" in hunk)
    return "\n".join(line[1:] for line in section.splitlines()[1:] if line.startswith((" ", "+")))


class TestRmsNormCache(unittest.TestCase):
    """Cache rows must be assigned, not accumulated into uninitialized memory."""

    def test_cache_write_overwrites_poison_without_touching_neighbor_rows(self) -> None:
        """Compile the actual write method for A2/A3 and A5, including unaligned tails."""
        component = _ROOT / "hyper_parallel/core/multicore"
        adapter = component / "ops/rms_norm_worker_fusion.patch"
        lock = json.loads((component / "_build/dependencies.lock.json").read_text(encoding="utf-8"))
        entry = next(item for item in lock["components"]["multicore"]["ops_nn"]["patches"]
                     if item["path"].endswith(adapter.name))
        actual_hash = hashlib.sha256(adapter.read_bytes()).hexdigest()
        self.assertEqual(actual_hash, entry["sha256"], f"expected={entry['sha256']} actual={actual_hash}")
        source = _MODEL.replace("HP_CACHE_WRITE", _cache_write_block(adapter.read_text(encoding="utf-8")))
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "rms-cache-write"
            for core, architecture in ((220, 2201), (220, 2203), (310, 3510)):
                with self.subTest(architecture=architecture):
                    result = subprocess.run(
                        ["c++", "-std=c++17", f"-D__CCE_AICORE__={core}", f"-D__NPU_ARCH__={architecture}",
                         "-x", "c++", "-", "-o", str(executable)], input=source,
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 0, f"expected compile=0 actual={result.stderr}")
                    result = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
                    self.assertEqual(result.returncode, 0,
                                     f"expected exact cache overwrite and intact neighbors actual={result.stdout}")
