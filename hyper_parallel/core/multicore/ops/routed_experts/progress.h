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

#ifndef HYPER_PARALLEL_ROUTED_PROGRESS_H_
#define HYPER_PARALLEL_ROUTED_PROGRESS_H_

#include "kernel_operator.h"

namespace routed {

// The mutable count shares its existing cache line with immutable task geometry.
// Python producer_progress() initializes these words before producer admission.
constexpr int32_t kProgressMode = 1;
constexpr int32_t kProgressFirstRow = 2;
constexpr int32_t kProgressTileRows = 3;
constexpr int32_t kProgressColumns = 4;
constexpr int32_t kProgressFirstTask = 5;
constexpr int32_t kProgressTaskCount = 6;
constexpr int32_t kProgressFirstGlobal = 7;
constexpr int32_t kProgressProducers = 8;
constexpr int32_t kProgressRows = 9;

class TaskProgressPublisher {
 public:
  __aicore__ inline void Init(GM_ADDR ready, int32_t experts, int32_t expert) {
    cell_ = ready == nullptr ? nullptr : reinterpret_cast<__gm__ int32_t *>(ready) +
      (static_cast<int64_t>(AscendC::GetBlockIdx()) * experts + expert) * 16;
    enabled_ = false;
    if (cell_ == nullptr) {
      return;
    }
    Flush();
    enabled_ = cell_[kProgressMode] == 1;
    if (enabled_) {
      first_ = cell_[kProgressFirstGlobal];
      total_ = cell_[kProgressTaskCount];
      producers_ = cell_[kProgressProducers];
      if (first_ < 0 || total_ < 1 || producers_ != AscendC::GetBlockNum()) {
        AscendC::Trap();
      }
    }
  }

  __aicore__ inline void Publish(int32_t task) {
    if (!enabled_) {
      return;
    }
    if (task < first_ || (task - first_) % producers_ != 0) {
      AscendC::Trap();
    }
    const int32_t completed = (task - first_) / producers_ + 1;
    if (completed > total_) {
      AscendC::Trap();
    }
    Store(completed);
  }

  __aicore__ inline void Finish() {
    if (cell_ != nullptr && !enabled_) {
      Store(1);
    }
  }

 private:
  __aicore__ inline void Flush() {
    AscendC::GlobalTensor<int32_t> line;
    line.SetGlobalBuffer(cell_, 16);
    __asm__ __volatile__("" ::: "memory");
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                     AscendC::DcciDst::CACHELINE_OUT>(line);
    __asm__ __volatile__("" ::: "memory");
  }

  __aicore__ inline void Store(int32_t value) {
    AscendC::PipeBarrier<PIPE_ALL>();
    *reinterpret_cast<__gm__ volatile int32_t *>(cell_) = value;
    Flush();
  }

  __gm__ int32_t *cell_ = nullptr;
  int32_t first_ = 0, total_ = 0, producers_ = 0;
  bool enabled_ = false;
};

// The same immutable geometry can be read directly from GM or from a UB snapshot.
template <typename Pointer>
__aicore__ inline int32_t RequiredTaskProgress(Pointer cell, int64_t first_row, int64_t last_row) {
  const int32_t origin = cell[kProgressFirstRow];
  const int32_t rows = cell[kProgressTileRows];
  const int32_t columns = cell[kProgressColumns];
  const int32_t stride = cell[kProgressProducers];
  const int32_t first = cell[kProgressFirstTask];
  const int32_t total = cell[kProgressTaskCount];
  if (rows < 1 || columns < 1 || stride < 1 || first < 0 || total < 1 || first_row < origin || last_row < first_row ||
      last_row >= static_cast<int64_t>(origin) + cell[kProgressRows]) {
    AscendC::Trap();
  }
  const int64_t first_task = (first_row - origin) / rows * columns;
  const int64_t last_task = (last_row - origin) / rows * columns + columns - 1;
  if (last_task < first) {
    return 0;
  }
  const int64_t begin = first_task <= first ? 0 : (first_task - first + stride - 1) / stride;
  const int64_t end = (last_task - first) / stride + 1;
  const int32_t needed = end < total ? static_cast<int32_t>(end) : total;
  return begin < needed ? needed : 0;
}

}  // namespace routed
#endif
