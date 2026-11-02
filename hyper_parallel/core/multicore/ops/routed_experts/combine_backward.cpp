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
#include "combine_backward.h"

using namespace AscendC;

inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg {0};

namespace {

class CombineBackwardWorker {
 public:
  __aicore__ inline void Init(GM_ADDR dy, GM_ADDR values, GM_ADDR order, GM_ADDR scores, GM_ADDR dv, GM_ADDR ds,
                              int64_t tokens, int64_t selections, int32_t hidden, int32_t top_k) {
    dy_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dy), tokens * hidden);
    values_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(values), selections * hidden);
    dv_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dv), selections * hidden);
    ds_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ds), selections);
    order_ = reinterpret_cast<__gm__ int64_t *>(order);
    scores_ = reinterpret_cast<__gm__ uint16_t *>(scores);
    selections_ = selections;
    hidden_ = hidden;
    top_k_ = top_k;
    tiles_ = (hidden + kTileColumns - 1) / kTileColumns;
    pipe_.InitBuffer(dy_input_buffer_, kBanks * kGroupRows * kTileColumns * sizeof(bfloat16_t));
    pipe_.InitBuffer(value_input_buffer_, kBanks * kGroupRows * kTileColumns * sizeof(bfloat16_t));
    pipe_.InitBuffer(output_buffer_, kBanks * kGroupRows * kTileColumns * sizeof(bfloat16_t));
    pipe_.InitBuffer(score_output_buffer_, kBanks * kGroupRows * kBlockBytes);
    pipe_.InitBuffer(dy_float_buffer_, kTileColumns * sizeof(float));
    pipe_.InitBuffer(product_buffer_, kTileColumns * sizeof(float));
    pipe_.InitBuffer(reduce_scratch_buffer_, kTileColumns * sizeof(float));
    pipe_.InitBuffer(tile_sum_buffer_, kBlockBytes);
    pipe_.InitBuffer(row_sum_buffer_, kGroupRows * kBlockBytes);
    dy_input_ = dy_input_buffer_.Get<bfloat16_t>();
    value_input_ = value_input_buffer_.Get<bfloat16_t>();
    rounded_ = output_buffer_.Get<bfloat16_t>();
    score_output_ = score_output_buffer_.Get<bfloat16_t>();
    dy_float_ = dy_float_buffer_.Get<float>();
    product_ = product_buffer_.Get<float>();
    reduce_scratch_ = reduce_scratch_buffer_.Get<float>();
    tile_sum_ = tile_sum_buffer_.Get<float>();
    row_sum_ = row_sum_buffer_.Get<float>();
  }

  __aicore__ inline void Run() {
    if (GetBlockIdx() >= selections_) {
      return;
    }
    const int64_t rows = (selections_ - 1 - GetBlockIdx()) / GetBlockNum() + 1;
    const int64_t stages = ((rows + kGroupRows - 1) / kGroupRows) * tiles_;
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
      Store(stage, bank);
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
  static constexpr int32_t kGroupRows = 4;
  static constexpr uint32_t kBlockBytes = 32;
  static constexpr int32_t kBanks = 2;

  __aicore__ inline int64_t Row(int64_t stage, int32_t index) {
    return GetBlockIdx() + ((stage / tiles_) * kGroupRows + index) * GetBlockNum();
  }

  __aicore__ inline uint32_t Count(int64_t stage) {
    const int32_t remaining = hidden_ - (stage % tiles_) * kTileColumns;
    return remaining < kTileColumns ? remaining : kTileColumns;
  }

  __aicore__ inline void Prefetch(int64_t stage, int32_t bank) {
    // MTE2 may reuse this bank only after its previous vector reader completes.
    if (input_used_[bank]) {
      WaitFlag<HardEvent::V_MTE2>(bank);
      input_used_[bank] = false;
    }
    const int32_t column = (stage % tiles_) * kTileColumns;
    const DataCopyExtParams copy{1, Count(stage) * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
    const DataCopyPadExtParams<bfloat16_t> padding{false, 0, 0, static_cast<bfloat16_t>(0.0F)};
    for (int32_t index = 0; index < kGroupRows; ++index) {
      const int64_t row = Row(stage, index);
      if (row >= selections_) {
        break;
      }
      const int64_t selection = order_[row];
      if (selection < 0 || selection >= selections_) {
        Trap();
        return;
      }
      selection_[bank][index] = selection;
      const int32_t offset = (bank * kGroupRows + index) * kTileColumns;
      DataCopyPad(dy_input_[offset], dy_[(selection / top_k_) * hidden_ + column], copy, padding);
      DataCopyPad(value_input_[offset], values_[row * hidden_ + column], copy, padding);
    }
    SetFlag<HardEvent::MTE2_V>(bank);
  }

  __aicore__ inline void Compute(int64_t stage, int32_t bank) {
    const uint32_t count = Count(stage);
    for (int32_t index = 0; index < kGroupRows; ++index) {
      if (Row(stage, index) >= selections_) {
        break;
      }
      auto row_sum = row_sum_[index * (kBlockBytes / sizeof(float))];
      if (stage % tiles_ == 0) {
        Duplicate(row_sum, 0.0F, kBlockBytes / sizeof(float));
        PipeBarrier<PIPE_V>();
      }
      const int32_t offset = (bank * kGroupRows + index) * kTileColumns;
      auto rounded = rounded_[offset];
      Cast(dy_float_, dy_input_[offset], RoundMode::CAST_NONE, count);
      Cast(product_, value_input_[offset], RoundMode::CAST_NONE, count);
      PipeBarrier<PIPE_V>();
      Mul(product_, product_, dy_float_, count);
      PipeBarrier<PIPE_V>();
      Cast(rounded, product_, RoundMode::CAST_RINT, count);
      PipeBarrier<PIPE_V>();
      Cast(product_, rounded, RoundMode::CAST_NONE, count);
      PipeBarrier<PIPE_V>();
      // Grouping independent rows preserves each original tile reduction and
      // the ascending-column FP32 additions within that row.
      ReduceSum(tile_sum_, product_, reduce_scratch_, count);
      PipeBarrier<PIPE_V>();
      Add(row_sum, row_sum, tile_sum_, 1);
      PipeBarrier<PIPE_V>();
      const uint32_t score_bits = static_cast<uint32_t>(scores_[selection_[bank][index]]) << 16;
      const float score = __builtin_bit_cast(float, score_bits);
      Muls(product_, dy_float_, score, count);
      PipeBarrier<PIPE_V>();
      Cast(rounded, product_, RoundMode::CAST_RINT, count);
      PipeBarrier<PIPE_V>();
      if (stage % tiles_ + 1 == tiles_) {
        Cast(score_output_[(bank * kGroupRows + index) * (kBlockBytes / sizeof(bfloat16_t))],
              row_sum, RoundMode::CAST_RINT, 1);
        PipeBarrier<PIPE_V>();
      }
    }
  }

  __aicore__ inline void Store(int64_t stage, int32_t bank) {
    SetFlag<HardEvent::V_MTE3>(bank);
    WaitFlag<HardEvent::V_MTE3>(bank);
    const int32_t column = (stage % tiles_) * kTileColumns;
    const DataCopyExtParams copy{1, Count(stage) * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
    const DataCopyExtParams scalar_copy{1, sizeof(bfloat16_t), 0, 0, 0};
    for (int32_t index = 0; index < kGroupRows; ++index) {
      const int64_t row = Row(stage, index);
      if (row >= selections_) {
        break;
      }
      DataCopyPad(dv_[row * hidden_ + column], rounded_[(bank * kGroupRows + index) * kTileColumns], copy);
      if (stage % tiles_ + 1 == tiles_) {
        DataCopyPad(ds_[selection_[bank][index]],
          score_output_[(bank * kGroupRows + index) * (kBlockBytes / sizeof(bfloat16_t))], scalar_copy);
      }
    }
    SetFlag<HardEvent::MTE3_V>(bank);
    output_pending_[bank] = true;
  }

  TPipe pipe_;
  TBuf<TPosition::VECIN> dy_input_buffer_, value_input_buffer_;
  TBuf<TPosition::VECOUT> output_buffer_, score_output_buffer_;
  TBuf<TPosition::VECCALC> dy_float_buffer_, product_buffer_, reduce_scratch_buffer_;
  TBuf<TPosition::VECCALC> tile_sum_buffer_, row_sum_buffer_;
  LocalTensor<bfloat16_t> dy_input_, value_input_, rounded_, score_output_;
  LocalTensor<float> dy_float_, product_, reduce_scratch_, tile_sum_, row_sum_;
  GlobalTensor<bfloat16_t> dy_, values_, dv_, ds_;
  __gm__ int64_t *order_ = nullptr;
  __gm__ uint16_t *scores_ = nullptr;
  int64_t selections_ = 0;
  int32_t hidden_ = 0, top_k_ = 0, tiles_ = 0;
  int64_t selection_[kBanks][kGroupRows] = {};
  bool input_used_[kBanks] = {};
  bool output_pending_[kBanks] = {};
};

__global__ __aicore__ void CombineBackwardKernel(GM_ADDR dy, GM_ADDR values, GM_ADDR order, GM_ADDR scores,
                                                GM_ADDR dv, GM_ADDR ds, int64_t tokens, int64_t selections,
                                                int32_t hidden, int32_t top_k) {
  CombineBackwardWorker worker;
  worker.Init(dy, values, order, scores, dv, ds, tokens, selections, hidden, top_k);
  worker.Run();
}

}  // namespace

extern "C" int RoutedExpertsCombineBackward(void *stream, void *dy, void *values, void *order, void *scores,
                                            void *dv, void *ds, int64_t tokens, int64_t selections,
                                            int32_t hidden, int32_t top_k, int32_t cores) {
  if (tokens == 0) {
    return 0;
  }
  const auto previous_error = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
  if (previous_error != ACL_SUCCESS) {
    return static_cast<int>(previous_error);
  }
  CombineBackwardKernel<<<cores, nullptr, stream>>>(static_cast<uint8_t *>(dy), static_cast<uint8_t *>(values),
    static_cast<uint8_t *>(order), static_cast<uint8_t *>(scores), static_cast<uint8_t *>(dv),
    static_cast<uint8_t *>(ds), tokens, selections, hidden, top_k);
  return static_cast<int>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
}
