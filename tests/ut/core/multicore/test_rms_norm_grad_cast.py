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
"""Compile real RMSNormGrad cast blocks with a hardware-independent output model."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import hyper_parallel


_ROOT = Path(hyper_parallel.__file__).resolve().parent.parent
_MODEL = r'''
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>
using std::is_same;
struct half {};
struct bfloat16_t {};
enum class RoundMode { CAST_NONE, CAST_RINT };
constexpr int PIPE_V = 0;
template<int> void PipeBarrier() {}
template<typename T> struct LocalTensor {
    unsigned char* bytes;
    template<typename U> LocalTensor<U> ReinterpretCast() { return {bytes}; }
};
int cast_count = 0;
template<typename T> void Cast(LocalTensor<T> destination, LocalTensor<float> source,
                              RoundMode mode, unsigned count) {
    if (mode != RoundMode::CAST_RINT) { throw "expected BF16 RINT conversion"; }
    ++cast_count;
    for (unsigned index = 0; index < count; ++index) {
        uint32_t bits;
        std::memcpy(&bits, source.bytes + index * sizeof(float), sizeof(bits));
        uint16_t rounded = (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16;
        std::memcpy(destination.bytes + index * sizeof(rounded), &rounded, sizeof(rounded));
    }
}
int main() {
    using T_DY = bfloat16_t;
    uint32_t values[4] = {0x3f807fc1U, 0x3f807fc1U, 0x3f807fc1U, 0x3f807fc1U};
    LocalTensor<float> dxLocal{reinterpret_cast<unsigned char*>(values)};
    constexpr unsigned colValAlign_ = 4;
    constexpr unsigned elementNum = 4;
HP_CAST_BLOCK
    uint16_t first;
    std::memcpy(&first, values, sizeof(first));
    std::cout << "casts=" << cast_count << " first_word=" << std::hex << first << '\n';
    constexpr uint16_t expected = HP_EXPECT_CAST ? 0x3f80U : 0x7fc1U;
    return cast_count == HP_EXPECT_CAST && first == expected ? 0 : 1;
}
'''


def _cast_probe_source(block: str) -> str:
    """Insert an actual production cast block, not a copy of its architecture condition."""
    return _MODEL.replace("HP_CAST_BLOCK", block)


def _adapter_cast_blocks(adapter: str) -> list[str]:
    """Read the resulting sides of the two complete output-conversion hunks."""
    blocks = []
    for hunk in adapter.split("\n@@")[1:]:
        if "else if constexpr (is_same<T_DY, bfloat16_t>::value)" not in hunk:
            continue
        body = "\n".join(line[1:] for line in hunk.splitlines()[1:] if line.startswith((" ", "+")))
        begin = body.index("        if constexpr (is_same<T_DY, half>::value)")
        end = body.index("#endif", begin) + len("#endif")
        blocks.append(body[begin:end])
    return blocks


class TestRmsNormGradCast(unittest.TestCase):
    """Reject omission of BF16 packing before native CopyOut interprets the buffer."""

    def test_both_output_casts_cover_the_selected_device_architecture(self) -> None:
        """Run the actual Compute/ComputeSmallD conversion blocks on CPU storage."""
        component = _ROOT / "hyper_parallel/core/multicore"
        adapter = component / "ops/rms_norm_grad_worker_fusion.patch"
        lock = json.loads((component / "_build/dependencies.lock.json").read_text(encoding="utf-8"))
        entry = next(item for item in lock["components"]["multicore"]["ops_nn"]["patches"]
                     if item["path"].endswith(adapter.name))
        actual_hash = hashlib.sha256(adapter.read_bytes()).hexdigest()
        self.assertEqual(actual_hash, entry["sha256"], f"expected={entry['sha256']} actual={actual_hash}")
        blocks = _adapter_cast_blocks(adapter.read_text(encoding="utf-8"))
        self.assertEqual(len(blocks), 2, f"expected=2 Compute output blocks actual={len(blocks)}")
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "rms-grad-cast"
            for index, block in enumerate(blocks):
                for core, architecture, expected in ((220, 2201, 1), (220, 2203, 1), (310, 3510, 1),
                                                     (310, 3113, 0), (310, 9999, 0)):
                    with self.subTest(block=index, architecture=architecture):
                        result = subprocess.run(
                            ["c++", "-std=c++17", f"-D__CCE_AICORE__={core}", f"-D__NPU_ARCH__={architecture}",
                             f"-DHP_EXPECT_CAST={expected}", "-x", "c++", "-", "-o", str(executable)],
                            input=_cast_probe_source(block), capture_output=True, text=True, check=False,
                        )
                        self.assertEqual(result.returncode, 0, f"expected compile=0 actual={result.stderr}")
                        result = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
                        self.assertEqual(result.returncode, 0,
                                         f"expected casts={expected} and packed output actual={result.stdout}")
