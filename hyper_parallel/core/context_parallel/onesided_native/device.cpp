// Copyright 2026 Huawei Technologies Co., Ltd
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "kernel_operator.h"
#include "shmem.h"
using namespace AscendC;
inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg { 0 };
namespace state_pull {
__global__ __aicore__ void signal_kernel(GM_ADDR flag, int value, int peer) {
  TPipe pipe;
  aclshmemx_signal_op(reinterpret_cast<__gm__ int32_t *>(flag), value, ACLSHMEM_SIGNAL_SET, peer);
}
__global__ __aicore__ void wait_kernel(GM_ADDR flag, int value) {
  aclshmem_signal_wait_until(reinterpret_cast<__gm__ int32_t *>(flag), ACLSHMEM_CMP_GE, value);
}
void signal(void *stream, void *flag, int value, int peer) {
  signal_kernel<<<1, nullptr, stream>>>((GM_ADDR)flag, value, peer);
}
void wait(void *stream, void *flag, int value) { wait_kernel<<<1, nullptr, stream>>>((GM_ADDR)flag, value); }
}  // namespace state_pull
