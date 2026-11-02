// Copyright 2026 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

#ifndef HYPER_PARALLEL_ROUTED_FORWARD_COMMON_H_
#define HYPER_PARALLEL_ROUTED_FORWARD_COMMON_H_
#include "kernel_operator.h"
using namespace AscendC;
inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg {0};

__aicore__ inline void FlushCell(__gm__ int32_t *address) {
  GlobalTensor<int32_t> cell;
  cell.SetGlobalBuffer(address, 16);
  __asm__ __volatile__("" ::: "memory");
  DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(cell);
  __asm__ __volatile__("" ::: "memory");
}

__aicore__ inline int32_t ReadCell(__gm__ int32_t *address) {
  FlushCell(address);
  return *reinterpret_cast<__gm__ volatile int32_t *>(address);
}

__aicore__ inline void WriteCell(__gm__ int32_t *address, int32_t value) {
  PipeBarrier<PIPE_ALL>();
  *reinterpret_cast<__gm__ volatile int32_t *>(address) = value;
  FlushCell(address);
}

__aicore__ inline void CheckTask(const __gm__ int64_t *task, int32_t rows, int32_t experts,
                                int32_t tile_rows) {
  if (task[0] < 0 || task[0] >= experts || task[1] < 0 || task[1] >= rows || task[2] < 1 ||
      task[2] > tile_rows || task[2] > static_cast<int64_t>(rows) - task[1]) {
    Trap();
  }
}
#endif
