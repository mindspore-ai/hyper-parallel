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
"""Run native scatter-index selection with CPU models of CANN's two index contracts."""

from pathlib import Path
import subprocess
import tempfile
import unittest

import hyper_parallel


_ROOT = Path(hyper_parallel.__file__).resolve().parent.parent
_MODEL = r'''
#include <cstdint>
#include <iostream>
#include <vector>
#define L0_DFX(...)
#define CHECK_RET(condition, result) do { if (!(condition)) { return result; } } while (false)
namespace op { enum class DataType { DT_INT64 }; }
struct aclTensor {
    int rows, columns;
    std::vector<int64_t> values;
    op::DataType GetDataType() const { return op::DataType::DT_INT64; }
};
struct aclOpExecutor {};
namespace op {
enum class SocVersion { ASCEND910B, ASCEND910_93, ASCEND950 };
struct Platform { SocVersion GetSocVersion() const { return static_cast<SocVersion>(HP_SOC); } };
Platform GetCurrentPlatformInfo() { return {}; }
}
int linear_calls = 0;
bool metadata_valid = true;
const aclTensor* EnsureKernelTensorMetadata(const aclTensor* tensor, aclOpExecutor*) {
    return metadata_valid ? tensor : nullptr;
}
namespace l0op {
aclTensor linear_indices{};
const aclTensor* LinearIndex(const aclTensor* indices, const aclTensor* output, int axis, bool combine,
                            aclOpExecutor*) {
    ++linear_calls;
    if (axis != 1 || combine) { return nullptr; }
    linear_indices = *indices;
    for (int row = 0; row < indices->rows; ++row) {
        for (int column = 0; column < indices->columns; ++column) {
            linear_indices.values[row * indices->columns + column] += row * output->columns;
        }
    }
    return &linear_indices;
}
HP_SELECT_INDICES
}
int main() {
    for (int rows : {1, 17, 257}) {
        constexpr int experts = 33, top_k = 7;
        aclTensor indices{rows, top_k, {}}, output{rows, experts, {}};
        for (int row = 0; row < rows; ++row) {
            for (int column = 0; column < top_k; ++column) {
                indices.values.push_back((row * 3 + column / 2) % experts);
            }
        }
        aclOpExecutor executor;
        linear_calls = 0;
        const aclTensor* selected = l0op::HP_SCATTER_FUNCTION(&indices, &output, &executor);
        if (!selected) { return 1; }
        std::vector<int64_t> actual(rows * experts, 0), expected(rows * experts, 0);
        for (int row = 0; row < rows; ++row) {
            for (int column = 0; column < top_k; ++column) {
                int element = row * top_k + column;
                int offset = selected->values[element] + (HP_SOC == 2 ? row * experts : 0);
                if (offset < 0 || offset >= rows * experts) { return 1; }
                actual[offset] += column + 1;
                expected[row * experts + indices.values[element]] += column + 1;
            }
        }
        bool passed = actual == expected && linear_calls == (HP_SOC == 2 ? 0 : 1);
        std::cout << "rows=" << rows << " linear_calls=" << linear_calls << " parity=" << passed << '\n';
        if (!passed) { return 1; }
        metadata_valid = false;
        if (l0op::HP_SCATTER_FUNCTION(&indices, &output, &executor) != nullptr) { return 1; }
        metadata_valid = true;
        if (l0op::HP_SCATTER_FUNCTION(nullptr, &output, &executor) != nullptr) { return 1; }
    }
    return 0;
}
'''


def _scatter_function(source: str, name: str = "HyperMegaGateScatterIndices") -> str:
    """Extract the actual host function without substituting its platform decision."""
    start = source.index(f"const aclTensor *{name}(")
    return source[start:source.index("\n}", start) + 2]


class TestScatterIndices(unittest.TestCase):
    """Preserve linear offsets on A2/A3 and axis-local INT64 indices on A5."""

    def test_native_selection_matches_platform_scatter_semantics(self) -> None:
        """Check repeated indices, tail rows, nulls and invalid metadata without an NPU."""
        source = _ROOT / "hyper_parallel/core/multicore/ops/hyper_mega_gate_grad/op_host/op_api"
        function = _scatter_function((source / "hyper_mega_gate_route_grad_post.cpp").read_text(encoding="utf-8"))
        model = _MODEL.replace("HP_SELECT_INDICES", function).replace(
            "HP_SCATTER_FUNCTION", "HyperMegaGateScatterIndices",
        )
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "scatter-indices"
            for soc, name in enumerate(("ascend910b", "ascend910_93", "ascend950")):
                with self.subTest(soc=name):
                    result = subprocess.run(
                        ["c++", "-std=c++17", f"-DHP_SOC={soc}", "-x", "c++", "-", "-o", str(executable)],
                        input=model, capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 0, f"expected compile=0 actual={result.stderr}")
                    result = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
                    self.assertEqual(result.returncode, 0,
                                     f"expected scatter parity and platform contract actual={result.stdout}")
