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
#include "acl/acl.h"
#include "combine.h"

using namespace AscendC;

inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg {0};

namespace {

class CombineWorker {
 public:
  __aicore__ inline void Init(GM_ADDR values, GM_ADDR inverse_order, GM_ADDR scores, GM_ADDR output,
                              int64_t tokens, int64_t selections, int32_t hidden, int32_t top_k) {
    values_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(values), selections * hidden);
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output), tokens * hidden);
    inverse_ = reinterpret_cast<__gm__ int64_t *>(inverse_order);
    scores_ = reinterpret_cast<__gm__ uint16_t *>(scores);
    tokens_ = tokens;
    selections_ = selections;
    hidden_ = hidden;
    top_k_ = top_k;
    tiles_ = (hidden + kTileColumns - 1) / kTileColumns;
    groups_ = (static_cast<int64_t>(top_k) + kGroupSlots - 1) / kGroupSlots;
    pipe_.InitBuffer(input_buffer_, kBanks * kGroupSlots * kTileColumns * sizeof(bfloat16_t));
    pipe_.InitBuffer(output_buffer_, kBanks * kTileColumns * sizeof(bfloat16_t));
    pipe_.InitBuffer(value_buffer_, kTileColumns * sizeof(float));
    pipe_.InitBuffer(sum_buffer_, kTileColumns * sizeof(float));
    input_ = input_buffer_.Get<bfloat16_t>();
    rounded_ = output_buffer_.Get<bfloat16_t>();
    value_ = value_buffer_.Get<float>();
    sum_ = sum_buffer_.Get<float>();
  }

  __aicore__ inline void Run() {
    const int64_t tasks = tokens_ * tiles_;
    if (GetBlockIdx() >= tasks) {
      return;
    }
    const int64_t local_tasks = (tasks - 1 - GetBlockIdx()) / GetBlockNum() + 1;
    const int64_t stages = local_tasks * groups_;
    Prefetch(0, 0);
    for (int64_t stage = 0; stage < stages; ++stage) {
      const int32_t bank = stage % kBanks;
      WaitFlag<HardEvent::MTE2_V>(bank);
      if (stage + 1 < stages) {
        Prefetch(stage + 1, (stage + 1) % kBanks);
      }
      // Keep this bank intact until MTE3 has consumed its prior output.
      if (output_pending_[bank]) {
        WaitFlag<HardEvent::MTE3_V>(bank);
        output_pending_[bank] = false;
      }
      Compute(stage, bank);
      SetFlag<HardEvent::V_MTE2>(bank);
      input_used_[bank] = true;
    }
    for (int32_t bank = 0; bank < kBanks; ++bank) {
      if (input_used_[bank]) {
        WaitFlag<HardEvent::V_MTE2>(bank);
      }
      if (output_pending_[bank]) {
        WaitFlag<HardEvent::MTE3_V>(bank);
      }
    }
  }

 private:
  static constexpr int32_t kTileColumns = 2048;
  static constexpr int32_t kGroupSlots = 8;
  static constexpr int32_t kBanks = 2;

  __aicore__ inline int64_t Task(int64_t stage) {
    return GetBlockIdx() + (stage / groups_) * GetBlockNum();
  }

  __aicore__ inline uint32_t Count(int64_t task) {
    const int32_t remaining = hidden_ - (task % tiles_) * kTileColumns;
    return remaining < kTileColumns ? remaining : kTileColumns;
  }

  __aicore__ inline void Prefetch(int64_t stage, int32_t bank) {
    // MTE2 may reuse this bank only after its previous vector reader completes.
    if (input_used_[bank]) {
      WaitFlag<HardEvent::V_MTE2>(bank);
      input_used_[bank] = false;
    }
    const int64_t task = Task(stage);
    const int64_t token = task / tiles_;
    const int32_t column = (task % tiles_) * kTileColumns;
    const int64_t first_slot = (stage % groups_) * kGroupSlots;
    const uint32_t count = Count(task);
    const DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
    const DataCopyPadExtParams<bfloat16_t> padding{false, 0, 0, static_cast<bfloat16_t>(0.0F)};
    for (int32_t slot = 0; slot < kGroupSlots && first_slot + slot < top_k_; ++slot) {
      const int64_t row = inverse_[token * top_k_ + first_slot + slot];
      if (row < 0 || row >= selections_) {
        Trap();
        return;
      }
      DataCopyPad(input_[(bank * kGroupSlots + slot) * kTileColumns],
                  values_[row * hidden_ + column], copy, padding);
    }
    SetFlag<HardEvent::MTE2_V>(bank);
  }

  __aicore__ inline void Compute(int64_t stage, int32_t bank) {
    const int64_t task = Task(stage);
    const int64_t token = task / tiles_;
    const int64_t first_slot = (stage % groups_) * kGroupSlots;
    const uint32_t count = Count(task);
    auto rounded = rounded_[bank * kTileColumns];
    if (first_slot == 0) {
      Duplicate(sum_, 0.0F, count);
      PipeBarrier<PIPE_V>();
    }
    for (int32_t slot = 0; slot < kGroupSlots && first_slot + slot < top_k_; ++slot) {
      Cast(value_, input_[(bank * kGroupSlots + slot) * kTileColumns], RoundMode::CAST_NONE, count);
      PipeBarrier<PIPE_V>();
      // Preserve BF16 product rounding before ascending-slot FP32 additions.
      if (scores_ != nullptr) {
        const int64_t selection = token * top_k_ + first_slot + slot;
        const uint32_t score_bits = static_cast<uint32_t>(scores_[selection]) << 16;
        const float score = __builtin_bit_cast(float, score_bits);
        Muls(value_, value_, score, count);
        PipeBarrier<PIPE_V>();
        Cast(rounded, value_, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        Cast(value_, rounded, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
      }
      Add(sum_, sum_, value_, count);
      PipeBarrier<PIPE_V>();
    }
    if (stage % groups_ + 1 == groups_) {
      Cast(rounded, sum_, RoundMode::CAST_RINT, count);
      SetFlag<HardEvent::V_MTE3>(bank);
      WaitFlag<HardEvent::V_MTE3>(bank);
      const DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
      DataCopyPad(output_[token * hidden_ + (task % tiles_) * kTileColumns], rounded, copy);
      SetFlag<HardEvent::MTE3_V>(bank);
      output_pending_[bank] = true;
    }
  }

  TPipe pipe_;
  TBuf<TPosition::VECIN> input_buffer_;
  TBuf<TPosition::VECOUT> output_buffer_;
  TBuf<TPosition::VECCALC> value_buffer_, sum_buffer_;
  LocalTensor<bfloat16_t> input_, rounded_;
  LocalTensor<float> value_, sum_;
  GlobalTensor<bfloat16_t> values_, output_;
  __gm__ int64_t *inverse_ = nullptr;
  __gm__ uint16_t *scores_ = nullptr;
  int64_t tokens_ = 0, selections_ = 0, groups_ = 0;
  int32_t hidden_ = 0, top_k_ = 0, tiles_ = 0;
  bool input_used_[kBanks] = {};
  bool output_pending_[kBanks] = {};
};

__global__ __aicore__ void CombineKernel(GM_ADDR values, GM_ADDR inverse_order, GM_ADDR scores, GM_ADDR output,
                                        int64_t tokens, int64_t selections, int32_t hidden, int32_t top_k) {
  CombineWorker worker;
  worker.Init(values, inverse_order, scores, output, tokens, selections, hidden, top_k);
  worker.Run();
}

}  // namespace

extern "C" int RoutedExpertsCombine(void *stream, void *values, void *inverse_order, void *scores,
                                    void *output, int64_t tokens, int64_t selections, int32_t hidden,
                                    int32_t top_k, int32_t cores) {
  if (tokens == 0) {
    return 0;
  }
  const auto previous_error = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
  if (previous_error != ACL_SUCCESS) {
    return static_cast<int>(previous_error);
  }
  CombineKernel<<<cores, nullptr, stream>>>(static_cast<uint8_t *>(values), static_cast<uint8_t *>(inverse_order),
    static_cast<uint8_t *>(scores), static_cast<uint8_t *>(output), tokens, selections, hidden, top_k);
  return static_cast<int>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
}
