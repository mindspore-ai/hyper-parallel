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
#include "acl/acl.h"
#include <initializer_list>

namespace {

inline bool InvalidVectorGeometry(int32_t rows, int32_t experts, int32_t inner, int32_t task_count,
                                  int32_t tile_rows, int32_t cores, int32_t vector_cores, int64_t watchdog) {
  return rows < 0 || rows > (1 << 20) || experts < 0 || experts > 128 || inner != 768 ||
    task_count < 0 || task_count > (1 << 20) || tile_rows != 128 || cores < 0 || cores > 24 ||
    vector_cores < 0 || vector_cores > 8 || vector_cores > cores || watchdog <= 0;
}

inline bool InvalidVectorStorage(std::initializer_list<const void *> operands, void *mailboxes) {
  for (const void *pointer : operands) {
    if (pointer == nullptr || reinterpret_cast<uintptr_t>(pointer) % 32) {
      return true;
    }
  }
  return reinterpret_cast<uintptr_t>(mailboxes) % 64;
}

class ActivationWorker {
 public:
  __aicore__ inline void Init(GM_ADDR z, GM_ADDR activation, int32_t inner) {
    inner_ = inner;
    source_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(z));
    destination_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(activation));
    destination_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    pipe_.InitBuffer(gate_buffer_, kElements * sizeof(bfloat16_t));
    pipe_.InitBuffer(value_buffer_, kElements * sizeof(bfloat16_t));
    pipe_.InitBuffer(result_buffer_, kElements * sizeof(bfloat16_t));
    pipe_.InitBuffer(source_float_buffer_, kElements * sizeof(float));
    pipe_.InitBuffer(result_float_buffer_, kElements * sizeof(float));
  }

  __aicore__ inline void Compute(int64_t source_row, int32_t rows, int64_t destination_row) {
    auto gate = gate_buffer_.Get<bfloat16_t>();
    auto value = value_buffer_.Get<bfloat16_t>();
    auto result = result_buffer_.Get<bfloat16_t>();
    auto source_float = source_float_buffer_.Get<float>();
    auto result_float = result_float_buffer_.Get<float>();
    const DataCopyPadExtParams<bfloat16_t> padding{false, 0, 0, static_cast<bfloat16_t>(0.0F)};
    for (int32_t column = 0; column < inner_; column += kColumns) {
      const uint32_t columns = inner_ - column < kColumns ? inner_ - column : kColumns;
      for (int32_t row = 0; row < rows; row += kRows) {
        const uint16_t count_rows = rows - row < kRows ? rows - row : kRows;
        const uint32_t elements = columns * count_rows;
        const DataCopyExtParams input_copy{count_rows, columns * 2,
          static_cast<uint32_t>(inner_ * 2 - columns) * 2, 0, 0};
        const DataCopyExtParams output_copy{count_rows, columns * 2, 0,
          static_cast<uint32_t>(inner_ - columns) * 2, 0};
        const int64_t offset = (source_row + row) * inner_ * 2 + column;
        DataCopyPad(gate, source_[offset], input_copy, padding);
        DataCopyPad(value, source_[offset + inner_], input_copy, padding);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(source_float, gate, RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        Muls(result_float, source_float, -1.0F, elements);
        PipeBarrier<PIPE_V>();
        Exp(result_float, result_float, elements);
        PipeBarrier<PIPE_V>();
        Adds(result_float, result_float, 1.0F, elements);
        PipeBarrier<PIPE_V>();
        Div(result_float, source_float, result_float, elements);
        PipeBarrier<PIPE_V>();
        Cast(source_float, value, RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        Mul(result_float, result_float, source_float, elements);
        PipeBarrier<PIPE_V>();
        Cast(result, result_float, RoundMode::CAST_RINT, elements);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(destination_[(destination_row + row) * inner_ + column], result, output_copy);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
      }
    }
    PipeBarrier<PIPE_ALL>();
  }

 private:
  static constexpr uint32_t kRows = 8;
  static constexpr uint32_t kColumns = 768;
  static constexpr uint32_t kElements = kRows * kColumns;
  TPipe pipe_;
  TBuf<TPosition::VECIN> gate_buffer_, value_buffer_;
  TBuf<TPosition::VECOUT> result_buffer_;
  TBuf<TPosition::VECCALC> source_float_buffer_, result_float_buffer_;
  GlobalTensor<bfloat16_t> source_, destination_;
  int32_t inner_ = 0;
};

__global__ __aicore__ void VectorKernel(GM_ADDR z, GM_ADDR activation, GM_ADDR tasks, GM_ADDR mailboxes,
                                        int32_t rows, int32_t experts, int32_t inner, int32_t task_count,
                                        int32_t tile_rows, int32_t cores, int64_t watchdog) {
  ActivationWorker worker;
  worker.Init(z, activation, inner);
  const auto *descriptors = reinterpret_cast<__gm__ int64_t *>(tasks);
  auto *cells = reinterpret_cast<__gm__ int32_t *>(mailboxes);
  int32_t completed[24] = {};
  int32_t remaining = 0;
  for (int32_t core = GetBlockIdx(); core < cores; core += GetBlockNum()) {
    ++remaining;
  }
  auto last_progress = GetSystemCycle();
  while (remaining > 0) {
    bool advanced = false;
    for (int32_t core = GetBlockIdx(); core < cores; core += GetBlockNum()) {
      if (completed[core] < 0) {
        continue;
      }
      const int32_t request = ReadCell(cells + core * 16);
      if (request == -1) {
        completed[core] = -1;
        --remaining;
        advanced = true;
      } else if (request > completed[core]) {
        const int32_t index = *reinterpret_cast<__gm__ volatile int32_t *>(cells + core * 16 + 1);
        if (index < 0 || index >= task_count || index % cores != core) {
          Trap();
        }
        const auto *task = descriptors + static_cast<int64_t>(index) * 3;
        CheckTask(task, rows, experts, tile_rows);
        PipeBarrier<PIPE_ALL>();
        worker.Compute(task[1], task[2], static_cast<int64_t>(core) * tile_rows);
        WriteCell(cells + (cores + core) * 16, request);
        completed[core] = request;
        advanced = true;
      } else if (request < 0) {
        Trap();
      }
    }
    if (advanced) {
      last_progress = GetSystemCycle();
    } else if (GetSystemCycle() - last_progress >= static_cast<uint64_t>(watchdog)) {
      Trap();
    }
  }
}

}  // namespace

extern "C" int RoutedExpertsForwardVector(void *stream, void *z, void *activation, void *tasks, void *mailboxes,
                                int32_t rows, int32_t experts, int32_t inner, int32_t task_count,
                                int32_t tile_rows, int32_t cores, int32_t vector_cores, int64_t watchdog) {
  if (InvalidVectorGeometry(rows, experts, inner, task_count, tile_rows, cores, vector_cores, watchdog)) {
    return -1;
  }
  if (rows == 0) {
    return experts == 0 && task_count == 0 && cores == 0 && vector_cores == 0 ? 0 : -1;
  }
  if (experts == 0 || cores == 0 || vector_cores == 0 || task_count < cores) {
    return -1;
  }
  if (InvalidVectorStorage({z, activation, tasks, mailboxes}, mailboxes)) {
    return -1;
  }
  const auto previous_error = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
  if (previous_error != ACL_SUCCESS) {
    return static_cast<int>(previous_error);
  }
  VectorKernel<<<vector_cores, nullptr, stream>>>(static_cast<uint8_t *>(z), static_cast<uint8_t *>(activation),
    static_cast<uint8_t *>(tasks), static_cast<uint8_t *>(mailboxes), rows, experts, inner, task_count,
    tile_rows, cores, watchdog);
  return static_cast<int>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
}
