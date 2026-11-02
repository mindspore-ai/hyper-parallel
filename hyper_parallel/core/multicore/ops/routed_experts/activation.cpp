// Copyright 2026 Huawei Technologies Co., Ltd
//
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
#include "activation.h"
#include <cmath>

using namespace AscendC;

inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg {0};

namespace {

class ActivationWorker {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR gradient, GM_ADDR output,
                              int64_t rows, int32_t inner, float limit) {
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(input), rows * inner * 2);
    if (gradient != nullptr) {
      gradient_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(gradient), rows * inner);
    }
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output),
                            rows * inner * (gradient == nullptr ? 1 : 2));
    rows_ = rows;
    inner_ = inner;
    tiles_ = (inner + kColumns - 1) / kColumns;
    limit_ = limit;
    backward_ = gradient != nullptr;
    pipe_.InitBuffer(gate_buffer_, kColumns * sizeof(bfloat16_t));
    pipe_.InitBuffer(up_buffer_, kColumns * sizeof(bfloat16_t));
    pipe_.InitBuffer(gradient_buffer_, kColumns * sizeof(bfloat16_t));
    pipe_.InitBuffer(result_buffer_, kColumns * sizeof(bfloat16_t));
    pipe_.InitBuffer(gate_float_buffer_, kColumns * sizeof(float));
    pipe_.InitBuffer(up_float_buffer_, kColumns * sizeof(float));
    pipe_.InitBuffer(derivative_buffer_, kColumns * sizeof(float));
    pipe_.InitBuffer(sigmoid_buffer_, kColumns * sizeof(float));
    pipe_.InitBuffer(work_buffer_, kColumns * sizeof(float));
    pipe_.InitBuffer(mask_buffer_, kColumns * sizeof(uint8_t));
  }

  __aicore__ inline void Run() {
    const int64_t tasks = rows_ * tiles_;
    for (int64_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
      const int64_t row = task / tiles_;
      const int32_t column = static_cast<int32_t>(task % tiles_) * kColumns;
      const uint32_t count = inner_ - column < kColumns ? inner_ - column : kColumns;
      Load(row, column, count);
      if (backward_) {
        Backward(row, column, count);
      } else {
        Forward(row, column, count);
      }
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
  }

 private:
  static constexpr int32_t kColumns = 256;

  __aicore__ inline void Load(int64_t row, int32_t column, uint32_t count) {
    const DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
    const DataCopyPadExtParams<bfloat16_t> padding{false, 0, 0, static_cast<bfloat16_t>(0.0F)};
    DataCopyPad(gate_buffer_.Get<bfloat16_t>(), input_[row * inner_ * 2 + column], copy, padding);
    DataCopyPad(up_buffer_.Get<bfloat16_t>(), input_[row * inner_ * 2 + inner_ + column], copy, padding);
    if (backward_) {
      DataCopyPad(gradient_buffer_.Get<bfloat16_t>(), gradient_[row * inner_ + column], copy, padding);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(gate_float_buffer_.Get<float>(), gate_buffer_.Get<bfloat16_t>(), RoundMode::CAST_NONE, count);
    Cast(up_float_buffer_.Get<float>(), up_buffer_.Get<bfloat16_t>(), RoundMode::CAST_NONE, count);
    if (backward_) {
      Cast(derivative_buffer_.Get<float>(), gradient_buffer_.Get<bfloat16_t>(), RoundMode::CAST_NONE, count);
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Sigmoid(uint32_t count) {
    auto gate = gate_float_buffer_.Get<float>();
    auto sigmoid = sigmoid_buffer_.Get<float>();
    Mins(gate, gate, limit_, count);
    PipeBarrier<PIPE_V>();
    Muls(sigmoid, gate, -1.0F, count);
    PipeBarrier<PIPE_V>();
    Exp(sigmoid, sigmoid, count);
    PipeBarrier<PIPE_V>();
    Adds(sigmoid, sigmoid, 1.0F, count);
    PipeBarrier<PIPE_V>();
    Duplicate(work_buffer_.Get<float>(), 1.0F, count);
    PipeBarrier<PIPE_V>();
    Div(sigmoid, work_buffer_.Get<float>(), sigmoid, count);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Store(int64_t row, int32_t column, uint32_t count,
                               bool up_branch) {
    auto result = result_buffer_.Get<bfloat16_t>();
    Cast(result, work_buffer_.Get<float>(), RoundMode::CAST_RINT, count);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    const DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
    const int64_t offset = row * inner_ * (backward_ ? 2 : 1) + column + (up_branch ? inner_ : 0);
    DataCopyPad(output_[offset], result, copy);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  }

  __aicore__ inline void Forward(int64_t row, int32_t column, uint32_t count) {
    Sigmoid(count);
    auto gate = gate_float_buffer_.Get<float>();
    auto up = up_float_buffer_.Get<float>();
    auto result = work_buffer_.Get<float>();
    Mins(up, up, limit_, count);
    PipeBarrier<PIPE_V>();
    Maxs(up, up, -limit_, count);
    PipeBarrier<PIPE_V>();
    Mul(result, gate, sigmoid_buffer_.Get<float>(), count);
    PipeBarrier<PIPE_V>();
    Mul(result, result, up, count);
    PipeBarrier<PIPE_V>();
    Store(row, column, count, false);
  }

  __aicore__ inline void Backward(int64_t row, int32_t column, uint32_t count) {
    Sigmoid(count);
    auto gate = gate_float_buffer_.Get<float>();
    auto up = up_float_buffer_.Get<float>();
    auto derivative = derivative_buffer_.Get<float>();
    auto sigmoid = sigmoid_buffer_.Get<float>();
    auto result = work_buffer_.Get<float>();
    auto mask = mask_buffer_.Get<uint8_t>();
    const uint32_t mask_count = (count + 63) / 64 * 64;
    // dUp uses the clamped gate. The raw up projection remains in its BF16
    // input buffer until both bound masks have been applied.
    Mul(result, gate, sigmoid, count);
    PipeBarrier<PIPE_V>();
    Mul(result, result, derivative, count);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask, up, limit_, CMPMODE::LE, mask_count);
    PipeBarrier<PIPE_V>();
    Select(result, mask, result, 0.0F, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask, up, -limit_, CMPMODE::GE, mask_count);
    PipeBarrier<PIPE_V>();
    Select(result, mask, result, 0.0F, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    PipeBarrier<PIPE_V>();
    Store(row, column, count, true);

    // dGate uses the clipped up projection and zeroes values beyond the
    // gate's upper bound. Recast the original BF16 gate for that mask.
    Mins(up, up, limit_, count);
    PipeBarrier<PIPE_V>();
    Maxs(up, up, -limit_, count);
    PipeBarrier<PIPE_V>();
    Muls(result, sigmoid, -1.0F, count);
    PipeBarrier<PIPE_V>();
    Adds(result, result, 1.0F, count);
    PipeBarrier<PIPE_V>();
    Mul(result, result, gate, count);
    PipeBarrier<PIPE_V>();
    Adds(result, result, 1.0F, count);
    PipeBarrier<PIPE_V>();
    Mul(result, result, sigmoid, count);
    PipeBarrier<PIPE_V>();
    Mul(result, result, up, count);
    PipeBarrier<PIPE_V>();
    Mul(result, result, derivative, count);
    PipeBarrier<PIPE_V>();
    Cast(gate, gate_buffer_.Get<bfloat16_t>(), RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask, gate, limit_, CMPMODE::LE, mask_count);
    PipeBarrier<PIPE_V>();
    Select(result, mask, result, 0.0F, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    PipeBarrier<PIPE_V>();
    Store(row, column, count, false);
  }

  TPipe pipe_;
  TBuf<TPosition::VECIN> gate_buffer_, up_buffer_, gradient_buffer_;
  TBuf<TPosition::VECOUT> result_buffer_;
  TBuf<TPosition::VECCALC> gate_float_buffer_, up_float_buffer_, derivative_buffer_;
  TBuf<TPosition::VECCALC> sigmoid_buffer_, work_buffer_, mask_buffer_;
  GlobalTensor<bfloat16_t> input_, gradient_, output_;
  int64_t rows_ = 0;
  int32_t inner_ = 0, tiles_ = 0;
  float limit_ = 0;
  bool backward_ = false;
};

__global__ __aicore__ void ActivationKernel(GM_ADDR input, GM_ADDR gradient, GM_ADDR output,
                                           int64_t rows, int32_t inner, float limit) {
  ActivationWorker worker;
  worker.Init(input, gradient, output, rows, inner, limit);
  worker.Run();
}

bool InvalidStorage(void *input, void *gradient, void *output) {
  return input == nullptr || output == nullptr ||
    reinterpret_cast<uintptr_t>(input) % 32 || reinterpret_cast<uintptr_t>(output) % 32 ||
    (gradient != nullptr && reinterpret_cast<uintptr_t>(gradient) % 32);
}

}  // namespace

extern "C" int RoutedExpertsClippedSwiGlu(void *stream, void *input, void *gradient, void *output,
                                           int64_t rows, int32_t inner, float limit, int32_t cores) {
  if (rows < 0 || rows > (1 << 20) || inner <= 0 || inner > 32768 || inner % 16 ||
      !std::isfinite(limit) || !(limit > 0.0F) || cores <= 0 || cores > 48) {
    return -1;
  }
  if (rows == 0) {
    return 0;
  }
  if (InvalidStorage(input, gradient, output)) {
    return -1;
  }
  const auto previous_error = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
  if (previous_error != ACL_SUCCESS) {
    return static_cast<int>(previous_error);
  }
  ActivationKernel<<<cores, nullptr, stream>>>(static_cast<uint8_t *>(input), static_cast<uint8_t *>(gradient),
                                              static_cast<uint8_t *>(output), rows, inner, limit);
  return static_cast<int>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
}
