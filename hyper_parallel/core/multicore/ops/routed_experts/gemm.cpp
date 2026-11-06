// Copyright 2026 Huawei Technologies Co., Ltd
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "acl/acl.h"
#include "transport.h"
#include "progress.h"

using namespace AscendC;
inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg {0};

namespace {

constexpr int32_t kReadyCellElements = 16;

inline bool InvalidGemmPolling(void *input_ready, void *input_expected, void *workspace,
                                int32_t producers, int32_t active_experts, int64_t watchdog_cycles) {
  return input_ready != nullptr &&
    (input_expected == nullptr || workspace == nullptr || reinterpret_cast<uintptr_t>(workspace) % 64 != 0 ||
     producers < 1 || watchdog_cycles < 1 ||
     reinterpret_cast<uintptr_t>(input_ready) % 64 != 0 ||
     reinterpret_cast<uintptr_t>(input_expected) % sizeof(int32_t) != 0 ||
     static_cast<uint64_t>(producers) > INT64_MAX / 64 / static_cast<uint64_t>(active_experts));
}

__aicore__ inline void FlushReadyCell(__gm__ int32_t *address) {
  GlobalTensor<int32_t> cell;
  cell.SetGlobalBuffer(address, kReadyCellElements);
  // Scalar polling and publication must reach GM across independent kernel streams.
  __asm__ __volatile__("" ::: "memory");
  DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(cell);
  __asm__ __volatile__("" ::: "memory");
}

__aicore__ inline bool ExpertReady(GM_ADDR ready, GM_ADDR expected, int32_t producers, int32_t experts,
                                  int32_t expert) {
  if (ready == nullptr) {
    return true;
  }
  const auto *counts = reinterpret_cast<__gm__ int32_t *>(expected);
  auto *cells = reinterpret_cast<__gm__ int32_t *>(ready);
  for (int32_t producer = 0; producer < producers; ++producer) {
    const int64_t index = static_cast<int64_t>(producer) * experts + expert;
    int32_t count = counts[index];
    if (count < 0) {
      Trap();
    }
    if (count == 0) {
      continue;
    }
    auto *cell = cells + index * kReadyCellElements;
    FlushReadyCell(cell);
    if (cell[routed::kProgressMode] == 1) {
      count = cell[routed::kProgressTaskCount];
    }
    const int32_t value = *reinterpret_cast<__gm__ volatile int32_t *>(cell);
    if (value < 0) {
      Trap();
    }
    if (value < count) {
      return false;
    }
  }
  PipeBarrier<PIPE_ALL>();
  return true;
}

template <bool TransposeB, typename MatrixMultiply>
__aicore__ inline void ComputeTask(MatrixMultiply &mm, GM_ADDR input, GM_ADDR output,
                                   const __gm__ int64_t *pointers, const __gm__ int64_t *task,
                                   int32_t n, int32_t k) {
  GlobalTensor<bfloat16_t> x, weight, result;
  x.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(input) + task[1] * k);
  weight.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pointers[task[0]]) +
                          (TransposeB ? task[3] * k : task[3]));
  result.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output) + task[1] * n + task[3]);
  result.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  mm.SetOrgShape(task[2], n, k);
  mm.SetSingleShape(task[2], task[4], k);
  mm.SetTensorA(x);
  mm.SetTensorB(weight, TransposeB);
  mm.IterateAll(result);
}

template <bool TransposeB, typename MatrixMultiply>
__aicore__ inline void RunReadyTasks(MatrixMultiply &mm, GM_ADDR input, GM_ADDR output,
                                    const __gm__ int64_t *pointers, const __gm__ int64_t *descriptors,
                                    GM_ADDR workspace, int32_t task_count, int32_t n, int32_t k,
                                    GM_ADDR input_ready, GM_ADDR input_expected, GM_ADDR output_ready,
                                    int32_t producers, int32_t active_experts, uint64_t watchdog_cycles) {
  // Each core retains its original unique tiles. Only the choice of the next
  // ready expert changes; no shared atomic queue or partial-output reduction is needed.
  // Scalar caches write whole lines: separate writers must never share one.
  const int64_t range_stride = ((static_cast<int64_t>(active_experts) * 2 + 7) / 8) * 8;
  auto *ranges = reinterpret_cast<__gm__ int64_t *>(workspace) + GetBlockIdx() * range_stride;
  for (int32_t expert = 0; expert < active_experts; ++expert) {
    ranges[expert * 2] = -1;
    ranges[expert * 2 + 1] = 0;
  }
  int32_t remaining = 0;
  int64_t previous_expert = -1;
  for (int32_t index = GetBlockIdx(); index < task_count; index += GetBlockNum()) {
    const int64_t expert = descriptors[static_cast<int64_t>(index) * 5];
    if (expert < previous_expert || expert < 0 || expert >= active_experts) {
      Trap();
    }
    if (expert != previous_expert) {
      ranges[expert * 2] = index;
      ++remaining;
      previous_expert = expert;
    }
    ranges[expert * 2 + 1] = static_cast<int64_t>(index) + GetBlockNum();
  }
  auto last_progress = GetSystemCycle();
  while (remaining > 0) {
    bool advanced = false;
    for (int32_t expert = 0; expert < active_experts; ++expert) {
      const int64_t first = ranges[expert * 2];
      if (first < 0 || !ExpertReady(input_ready, input_expected, producers, active_experts, expert)) {
        continue;
      }
      const int64_t end = ranges[expert * 2 + 1];
      routed::TaskProgressPublisher publisher;
      publisher.Init(output_ready, active_experts, expert);
      for (int64_t index = first; index < end; index += GetBlockNum()) {
        ComputeTask<TransposeB>(mm, input, output, pointers, descriptors + index * 5, n, k);
        publisher.Publish(index);
      }
      publisher.Finish();
      ranges[expert * 2] = -1;
      --remaining;
      advanced = true;
    }
    if (advanced) {
      last_progress = GetSystemCycle();
    } else if (GetSystemCycle() - last_progress >= watchdog_cycles) {
      Trap();
    }
  }
}

template <bool TransposeB>
__global__ __aicore__ void GemmKernel(GM_ADDR input, GM_ADDR weights, GM_ADDR tasks, GM_ADDR output,
                                     GM_ADDR config, GM_ADDR workspace, int32_t task_count, int32_t n, int32_t k,
                                     GM_ADDR input_ready, GM_ADDR input_expected, GM_ADDR output_ready,
                                     int32_t producers, int32_t active_experts, uint64_t watchdog_cycles) {
  using A = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
  using B = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, TransposeB>;
  using C = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
  TPipe pipe;
  matmul::Matmul<A, B, C, C, CFG_MDL> mm;
  TCubeTiling tiling;
  auto *local = reinterpret_cast<int32_t *>(&tiling);
  auto *global = reinterpret_cast<__gm__ int32_t *>(config);
  for (int32_t index = 0; index < sizeof(TCubeTiling) / sizeof(int32_t); ++index) {
    local[index] = global[index];
  }
  REGIST_MATMUL_OBJ(&pipe, workspace, mm, &tiling);
  const auto *pointers = reinterpret_cast<__gm__ int64_t *>(weights);
  const auto *descriptors = reinterpret_cast<__gm__ int64_t *>(tasks);
  if (input_ready != nullptr) {
    RunReadyTasks<TransposeB>(mm, input, output, pointers, descriptors, workspace, task_count, n, k,
                             input_ready, input_expected, output_ready, producers, active_experts, watchdog_cycles);
    mm.End();
    return;
  }
  const bool readiness_enabled = input_ready != nullptr || output_ready != nullptr;
  int32_t current_expert = -1;
  routed::TaskProgressPublisher publisher;
  for (int32_t index = GetBlockIdx(); index < task_count; index += GetBlockNum()) {
    // [weight reference, first input row, rows, first output column, columns]
    const auto *task = descriptors + static_cast<int64_t>(index) * 5;
    if (readiness_enabled) {
      if (task[0] < 0 || task[0] >= active_experts || task[0] < current_expert) {
        Trap();
      }
      const int32_t expert = static_cast<int32_t>(task[0]);
      if (expert != current_expert) {
        publisher.Finish();
        publisher.Init(output_ready, active_experts, expert);
        current_expert = expert;
      }
    }
    ComputeTask<TransposeB>(mm, input, output, pointers, task, n, k);
    publisher.Publish(index);
  }
  publisher.Finish();
  mm.End();
}

}  // namespace

extern "C" int RoutedExpertsGemm(void *stream, void *input, void *weights, void *tasks, void *output,
                                 void *config, void *workspace, int32_t task_count, int32_t n, int32_t k,
                                 bool transpose_b, int32_t cores, void *input_ready, void *input_expected,
                                 void *output_ready, int32_t producers, int32_t active_experts,
                                 int64_t watchdog_cycles) {
  if (task_count < 0 || cores < 1 || cores > 24 || n < 1 || k < 1 ||
      ((input_ready != nullptr || output_ready != nullptr) && active_experts < 1) ||
      (output_ready != nullptr && reinterpret_cast<uintptr_t>(output_ready) % 64 != 0)) {
    return -1;
  }
  if (InvalidGemmPolling(input_ready, input_expected, workspace, producers, active_experts, watchdog_cycles)) {
    return -1;
  }
  const auto previous_error = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
  if (previous_error != ACL_SUCCESS) {
    return static_cast<int>(previous_error);
  }
  if (transpose_b) {
    GemmKernel<true><<<cores, nullptr, stream>>>(static_cast<uint8_t *>(input), static_cast<uint8_t *>(weights),
      static_cast<uint8_t *>(tasks), static_cast<uint8_t *>(output), static_cast<uint8_t *>(config),
      static_cast<uint8_t *>(workspace), task_count, n, k, static_cast<uint8_t *>(input_ready),
      static_cast<uint8_t *>(input_expected), static_cast<uint8_t *>(output_ready), producers, active_experts,
      static_cast<uint64_t>(watchdog_cycles));
  } else {
    GemmKernel<false><<<cores, nullptr, stream>>>(static_cast<uint8_t *>(input), static_cast<uint8_t *>(weights),
      static_cast<uint8_t *>(tasks), static_cast<uint8_t *>(output), static_cast<uint8_t *>(config),
      static_cast<uint8_t *>(workspace), task_count, n, k, static_cast<uint8_t *>(input_ready),
      static_cast<uint8_t *>(input_expected), static_cast<uint8_t *>(output_ready), producers, active_experts,
      static_cast<uint64_t>(watchdog_cycles));
  }
  return static_cast<int>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
}
