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

#include "forward_common.h"
#include "forward.h"
#include "lib/matmul_intf.h"
#include "acl/acl.h"
#include <initializer_list>

namespace {

inline bool InvalidCubeGeometry(int32_t rows, int32_t experts, int32_t hidden, int32_t inner,
                                int32_t task_count, int32_t tile_rows, int32_t cores) {
  return rows < 0 || rows > (1 << 20) || experts < 0 || experts > 128 || hidden != 2048 || inner != 768 ||
    task_count < 0 || task_count > (1 << 20) || tile_rows != 128 || cores < 0 || cores > 24;
}

inline bool InvalidCubePolling(int64_t watchdog, int32_t producers, void *input_ready, void *input_expected) {
  return watchdog <= 0 || producers < 0 || producers > 48 ||
    ((input_ready == nullptr) != (input_expected == nullptr));
}

inline bool InvalidCubeStorage(std::initializer_list<const void *> operands, void *mailboxes,
                               void *input_ready, void *input_expected, void *output_ready) {
  for (const void *pointer : operands) {
    if (pointer == nullptr || reinterpret_cast<uintptr_t>(pointer) % 32) {
      return true;
    }
  }
  return reinterpret_cast<uintptr_t>(mailboxes) % 64 ||
    (input_ready != nullptr && reinterpret_cast<uintptr_t>(input_ready) % 64) ||
    (input_expected != nullptr && reinterpret_cast<uintptr_t>(input_expected) % 32) ||
    (output_ready != nullptr && reinterpret_cast<uintptr_t>(output_ready) % 64);
}

template <bool TransposeB, typename MatrixMultiply>
__aicore__ inline void RunRole(MatrixMultiply &mm, __gm__ bfloat16_t *input, __gm__ bfloat16_t *weight,
                              __gm__ bfloat16_t *output, int32_t rows, int32_t n, int32_t k) {
  GlobalTensor<bfloat16_t> left, right, result;
  left.SetGlobalBuffer(input);
  for (int32_t column = 0; column < n; column += 512) {
    right.SetGlobalBuffer(weight + (TransposeB ? static_cast<int64_t>(column) * k : column));
    result.SetGlobalBuffer(output + column);
    result.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    const int32_t columns = n - column < 512 ? n - column : 512;
    mm.SetOrgShape(rows, n, k);
    mm.SetSingleShape(rows, columns, k);
    mm.SetTensorA(left);
    mm.SetTensorB(right, TransposeB);
    mm.IterateAll(result);
  }
  PipeBarrier<PIPE_ALL>();
}

__aicore__ inline bool ExpertReady(GM_ADDR ready, GM_ADDR expected, int32_t producers,
                                  int32_t experts, int32_t expert, int32_t task,
                                  int32_t expected_columns) {
  if (ready == nullptr) {
    return true;
  }
  auto *cells = reinterpret_cast<__gm__ int32_t *>(ready);
  const auto *counts = reinterpret_cast<__gm__ int32_t *>(expected);
  for (int32_t producer = 0; producer < producers; ++producer) {
    const int64_t index = static_cast<int64_t>(producer) * experts + expert;
    const int32_t expected_count = counts[static_cast<int64_t>(producer) * expected_columns +
                                          (expected_columns == experts ? expert : task)];
    if (expected_count < 0) {
      Trap();
    }
    if (expected_count == 0) {
      continue;
    }
    const int32_t actual = ReadCell(cells + index * 16);
    if (actual < 0) {
      Trap();
    }
    if (actual < expected_count) {
      return false;
    }
  }
  PipeBarrier<PIPE_ALL>();
  return true;
}

template <bool TransposeB, typename MatrixMultiply>
__aicore__ inline void ComputeChunk(MatrixMultiply &mm, GM_ADDR x, GM_ADDR gate, GM_ADDR down,
                                    GM_ADDR tasks, GM_ADDR z, GM_ADDR activation, GM_ADDR y,
                                    GM_ADDR mailboxes, int32_t index, int32_t generation, int32_t rows, int32_t experts,
                                    int32_t hidden, int32_t inner, int32_t tile_rows, int64_t watchdog) {
  const auto *task = reinterpret_cast<__gm__ int64_t *>(tasks) + static_cast<int64_t>(index) * 3;
  CheckTask(task, rows, experts, tile_rows);
  const int32_t core = GetBlockIdx();
  auto *cells = reinterpret_cast<__gm__ int32_t *>(mailboxes);
  auto *local_activation = reinterpret_cast<__gm__ bfloat16_t *>(activation) +
    static_cast<int64_t>(core) * tile_rows * inner;
  const auto gate_pointer = reinterpret_cast<__gm__ int64_t *>(gate)[task[0]];
  const auto down_pointer = reinterpret_cast<__gm__ int64_t *>(down)[task[0]];
  if (gate_pointer <= 0 || down_pointer <= 0 || gate_pointer % 32 || down_pointer % 32) {
    Trap();
  }
  RunRole<TransposeB>(mm,
    reinterpret_cast<__gm__ bfloat16_t *>(x) + task[1] * hidden,
    reinterpret_cast<__gm__ bfloat16_t *>(gate_pointer),
    reinterpret_cast<__gm__ bfloat16_t *>(z) + task[1] * inner * 2, task[2], inner * 2, hidden);
  *reinterpret_cast<__gm__ volatile int32_t *>(cells + core * 16 + 1) = index;
  WriteCell(cells + core * 16, generation);
  const auto started = GetSystemCycle();
  while (ReadCell(cells + (GetBlockNum() + core) * 16) != generation) {
    if (GetSystemCycle() - started >= static_cast<uint64_t>(watchdog)) {
      Trap();
    }
  }
  PipeBarrier<PIPE_ALL>();
  RunRole<TransposeB>(mm, local_activation,
    reinterpret_cast<__gm__ bfloat16_t *>(down_pointer),
    reinterpret_cast<__gm__ bfloat16_t *>(y) + task[1] * hidden, task[2], hidden, inner);
}

template <bool TransposeB>
__global__ __aicore__ void CubeKernel(GM_ADDR x, GM_ADDR gate, GM_ADDR down, GM_ADDR tasks, GM_ADDR z,
                                      GM_ADDR activation, GM_ADDR y, GM_ADDR config, GM_ADDR mailboxes,
                                      int32_t rows, int32_t experts, int32_t hidden, int32_t inner,
                                      int32_t task_count, int32_t tile_rows, int64_t watchdog,
                                      GM_ADDR input_ready, GM_ADDR input_expected, GM_ADDR output_ready,
                                      int32_t producers, int32_t expected_columns) {
  using A = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
  using B = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, TransposeB>;
  using C = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
  TPipe pipe;
  matmul::Matmul<A, B, C, C, CFG_MDL> mm;
  TCubeTiling tiling;
  auto *target = reinterpret_cast<int32_t *>(&tiling);
  const auto *source = reinterpret_cast<__gm__ int32_t *>(config);
  for (uint32_t index = 0; index < sizeof(TCubeTiling) / sizeof(int32_t); ++index) {
    target[index] = source[index];
  }
  REGIST_MATMUL_OBJ(&pipe, nullptr, mm, &tiling);
  const int32_t core = GetBlockIdx();
  const auto *descriptors = reinterpret_cast<__gm__ int64_t *>(tasks);
  int32_t first[128];
  int32_t end[128];
  for (int32_t expert = 0; expert < experts; ++expert) {
    first[expert] = -1;
    end[expert] = 0;
  }
  int32_t remaining = 0;
  int32_t previous_expert = -1;
  for (int32_t index = core; index < task_count; index += GetBlockNum()) {
    const auto *task = descriptors + static_cast<int64_t>(index) * 3;
    CheckTask(task, rows, experts, tile_rows);
    const int32_t expert = task[0];
    if (expert < previous_expert) {
      Trap();
    }
    if (expert != previous_expert) {
      first[expert] = index;
      ++remaining;
      previous_expert = expert;
    }
    end[expert] = index + GetBlockNum();
  }
  int32_t generation = 0;
  const bool tile_ready = expected_columns == task_count && task_count > experts;
  auto last_progress = GetSystemCycle();
  while (remaining > 0) {
    bool advanced = false;
    for (int32_t expert = 0; expert < experts; ++expert) {
      if (first[expert] < 0) {
        continue;
      }
      if (!tile_ready && !ExpertReady(input_ready, input_expected, producers, experts, expert,
                                      first[expert], expected_columns)) {
        continue;
      }
      while (first[expert] < end[expert] &&
             (!tile_ready || ExpertReady(input_ready, input_expected, producers, experts, expert,
                                          first[expert], expected_columns))) {
        ComputeChunk<TransposeB>(mm, x, gate, down, tasks, z, activation, y, mailboxes, first[expert],
                                ++generation, rows, experts, hidden, inner, tile_rows, watchdog);
        first[expert] += GetBlockNum();
        advanced = true;
      }
      if (first[expert] < end[expert]) {
        continue;
      }
      if (output_ready != nullptr) {
        WriteCell(reinterpret_cast<__gm__ int32_t *>(output_ready) +
                  (static_cast<int64_t>(core) * experts + expert) * 16, 1);
      }
      first[expert] = -1;
      --remaining;
    }
    if (advanced) {
      last_progress = GetSystemCycle();
    } else if (GetSystemCycle() - last_progress >= static_cast<uint64_t>(watchdog)) {
      Trap();
    }
  }
  mm.End();
  WriteCell(reinterpret_cast<__gm__ int32_t *>(mailboxes) + core * 16, -1);
}

}  // namespace

extern "C" int RoutedExpertsForwardCube(void *stream, void *x, void *gate, void *down, void *tasks, void *z,
                              void *activation, void *y, void *config, void *mailboxes, int32_t rows,
                              int32_t experts, int32_t hidden, int32_t inner, int32_t task_count,
                              int32_t tile_rows, bool transpose, int32_t cores, int64_t watchdog,
                              void *input_ready, void *input_expected, void *output_ready, int32_t producers,
                              int32_t expected_columns) {
  if (InvalidCubeGeometry(rows, experts, hidden, inner, task_count, tile_rows, cores) ||
      InvalidCubePolling(watchdog, producers, input_ready, input_expected) ||
      (input_ready != nullptr && expected_columns != experts && expected_columns != task_count)) {
    return -1;
  }
  if (rows == 0) {
    return experts == 0 && task_count == 0 && cores == 0 ? 0 : -1;
  }
  if (experts == 0 || cores == 0 || task_count < cores || (input_ready != nullptr && producers == 0)) {
    return -1;
  }
  if (InvalidCubeStorage({x, gate, down, tasks, z, activation, y, config, mailboxes}, mailboxes,
                         input_ready, input_expected, output_ready)) {
    return -1;
  }
  const auto previous_error = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
  if (previous_error != ACL_SUCCESS) {
    return static_cast<int>(previous_error);
  }
  if (transpose) {
    CubeKernel<true><<<cores, nullptr, stream>>>(static_cast<uint8_t *>(x), static_cast<uint8_t *>(gate),
      static_cast<uint8_t *>(down), static_cast<uint8_t *>(tasks), static_cast<uint8_t *>(z),
      static_cast<uint8_t *>(activation), static_cast<uint8_t *>(y), static_cast<uint8_t *>(config),
      static_cast<uint8_t *>(mailboxes), rows, experts, hidden, inner, task_count, tile_rows, watchdog,
      static_cast<uint8_t *>(input_ready), static_cast<uint8_t *>(input_expected),
      static_cast<uint8_t *>(output_ready), producers, expected_columns);
  } else {
    CubeKernel<false><<<cores, nullptr, stream>>>(static_cast<uint8_t *>(x), static_cast<uint8_t *>(gate),
      static_cast<uint8_t *>(down), static_cast<uint8_t *>(tasks), static_cast<uint8_t *>(z),
      static_cast<uint8_t *>(activation), static_cast<uint8_t *>(y), static_cast<uint8_t *>(config),
      static_cast<uint8_t *>(mailboxes), rows, experts, hidden, inner, task_count, tile_rows, watchdog,
      static_cast<uint8_t *>(input_ready), static_cast<uint8_t *>(input_expected),
      static_cast<uint8_t *>(output_ready), producers, expected_columns);
  }
  return static_cast<int>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
}
