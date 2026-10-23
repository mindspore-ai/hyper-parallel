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
"""Run the real RMSNormGrad initialization branches against shared-workspace sentinels."""

from pathlib import Path
import subprocess
import tempfile
import unittest

import hyper_parallel


_ROOT = Path(hyper_parallel.__file__).resolve().parent.parent
_MODEL = r'''
#include <algorithm>
#include <cstdint>
#include <iostream>
#define __gm__
constexpr unsigned ALIGN_32 = 8;
unsigned GetBlockNum() { return 32; }
unsigned GetBlockIdx() { return 0; }
int synchronizations = 0;
template<typename... Args> void SyncAll(Args...) { ++synchronizations; }
template<typename T> struct GlobalTensor {
    T* data = nullptr;
    void SetGlobalBuffer(T* address, unsigned = 0) { data = address; }
};
template<typename T> struct LocalTensor {};
struct Buffer { template<typename T> LocalTensor<T> Get() { return {}; } };
struct Pipe { void InitBuffer(Buffer&, unsigned) {} };
template<typename T> void InitGmZero(GlobalTensor<T>& tensor, Buffer&, unsigned count, unsigned offset) {
    std::fill_n(tensor.data + offset, count, T{});
}
struct Kernel {
    bool taskMode_ = HP_TASK_MODE;
    unsigned isDeterministic_ = 0, blockIdx_ = 0, colVal_ = 512, colValAlign_ = 512;
    GlobalTensor<int32_t> syncTmpSpaceGm_;
    GlobalTensor<float> workspaceGm_, workspaceGmOri_, dgammaGm_;
    Pipe pipe;
    Buffer outZeroTmpBuf_, syncTmpBuf_;
    void InitTmpBuffer() {}
    void InitWorkspace(unsigned char*) {}
    void Initialize(unsigned char* usrWorkspace) {
HP_INITIALIZATION
    }
};
int main() {
    int32_t workspace[1024];
    std::fill_n(workspace, 1024, 17);
    float gamma[512];
    std::fill_n(gamma, 512, 19.0f);
    Kernel kernel;
    kernel.dgammaGm_.SetGlobalBuffer(gamma);
    kernel.Initialize(reinterpret_cast<unsigned char*>(workspace));
    const bool unchanged = std::all_of(workspace, workspace + 1024, [](int32_t x) { return x == 17; });
    const bool gamma_unchanged = std::all_of(gamma, gamma + 512, [](float x) { return x == 19.0f; });
    std::cout << "workspace_preserved=" << unchanged << " gamma_preserved=" << gamma_unchanged
              << " synchronizations=" << synchronizations << '\n';
    if (HP_TASK_MODE) { return unchanged && gamma_unchanged && synchronizations == 0 ? 0 : 1; }
#if __CCE_AICORE__ == 220
    return unchanged && gamma_unchanged && synchronizations == 1 ? 0 : 1;
#else
    return !unchanged && !gamma_unchanged && synchronizations == 1 ? 0 : 1;
#endif
}
'''


def _initialization_branch(adapter: str) -> str:
    """Extract both complete architecture branches from the resulting adapter context."""
    hunks = adapter.split("\n@@")
    body = "\n".join(line[1:] for hunk in hunks[1:] for line in hunk.splitlines()[1:]
                     if line.startswith((" ", "+")))
    start = body.index("#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)")
    end = body.index("#endif", start) + len("#endif")
    return body[start:end]


class TestRmsNormGradTask(unittest.TestCase):
    """Task-local RMSNormGrad must never initialize the enclosing graph's workspace."""

    def test_task_preserves_shared_workspace_and_standalone_keeps_initialization(self) -> None:
        """Compile actual initialization branches with CPU models of their GM side effects."""
        adapter = _ROOT / "hyper_parallel/core/multicore/ops/rms_norm_grad_worker_fusion.patch"
        source = _MODEL.replace("HP_INITIALIZATION", _initialization_branch(adapter.read_text(encoding="utf-8")))
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "rms-grad-task"
            for core in (220, 310):
                for task_mode in (0, 1):
                    with self.subTest(core=core, task_mode=task_mode):
                        result = subprocess.run(
                            ["c++", "-std=c++17", f"-D__CCE_AICORE__={core}", f"-DHP_TASK_MODE={task_mode}",
                             "-x", "c++", "-", "-o", str(executable)], input=source,
                            capture_output=True, text=True, check=False,
                        )
                        self.assertEqual(result.returncode, 0, f"expected compile=0 actual={result.stderr}")
                        result = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
                        self.assertEqual(result.returncode, 0,
                                         f"expected task-only workspace preservation actual={result.stdout}")
