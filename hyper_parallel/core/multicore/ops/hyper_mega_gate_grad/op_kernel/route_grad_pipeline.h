/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifndef HYPER_MEGA_GATE_ROUTE_GRAD_PIPELINE_H
#define HYPER_MEGA_GATE_ROUTE_GRAD_PIPELINE_H

#include "adv_api/pad/broadcast.h"
#include "adv_api/reduce/reduce.h"
#include "kernel_operator.h"

namespace HyperMegaGate {

using namespace AscendC;  // NOLINT(build/namespaces)

class TokenRowPipeline {
 protected:
  __aicore__ inline void InitRows(uint32_t worker_id, int64_t token_count, uint32_t rows_per_worker,
                                  uint32_t batch_rows) {
    first_row_ = static_cast<uint64_t>(worker_id) * rows_per_worker;
    const uint64_t remaining = static_cast<uint64_t>(token_count) - first_row_;
    row_count_ = static_cast<uint32_t>(remaining < rows_per_worker ? remaining : rows_per_worker);
    batch_rows_ = batch_rows;
  }

  __aicore__ inline uint32_t BatchRowCount(uint32_t offset) const {
    const uint32_t remaining = row_count_ - offset;
    return remaining < batch_rows_ ? remaining : batch_rows_;
  }

  template <typename T>
  __aicore__ inline void CopyInRows(LocalTensor<T> destination, __gm__ T *source, uint64_t row, uint32_t rows,
                                    uint32_t columns, uint32_t alignment) const {
    GlobalTensor<T> source_tensor;
    source_tensor.SetGlobalBuffer(source);
    DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(columns * sizeof(T)), 0,
                             static_cast<uint32_t>(alignment * sizeof(T) / 32 - (columns * sizeof(T) + 31) / 32), 0};
    DataCopyPadExtParams<T> padding{true, 0, static_cast<uint8_t>(alignment - columns), static_cast<T>(0)};
    DataCopyPad(destination, source_tensor[row * columns], params, padding);
  }

  template <typename T>
  __aicore__ inline void CopyOutRows(__gm__ T *destination, uint64_t row, uint32_t rows, uint32_t columns,
                                     uint32_t alignment, LocalTensor<T> source) const {
    GlobalTensor<T> destination_tensor;
    destination_tensor.SetGlobalBuffer(destination);
    DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(columns * sizeof(T)),
                             static_cast<uint32_t>(alignment * sizeof(T) / 32 - (columns * sizeof(T) + 31) / 32), 0, 0};
    DataCopyPad(destination_tensor[row * columns], source, params);
  }

  __aicore__ inline void CopyInScalars(LocalTensor<float> destination, __gm__ float *source, uint64_t row,
                                       uint32_t rows) const {
    GlobalTensor<float> source_tensor;
    source_tensor.SetGlobalBuffer(source);
    DataCopyExtParams params{1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padding{false, 0, 0, 0.0F};
    DataCopyPad(destination, source_tensor[row], params, padding);
  }

  __aicore__ inline void CopyOutScalars(__gm__ float *destination, uint64_t row, uint32_t rows,
                                        LocalTensor<float> source) const {
    GlobalTensor<float> destination_tensor;
    destination_tensor.SetGlobalBuffer(destination);
    DataCopyExtParams params{1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0};
    DataCopyPad(destination_tensor[row], source, params);
  }

  __aicore__ inline void CopyInElements(LocalTensor<float> destination, __gm__ float *source, uint64_t element_offset,
                                        uint32_t count) const {
    GlobalTensor<float> source_tensor;
    source_tensor.SetGlobalBuffer(source);
    DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    const uint8_t tail_padding = static_cast<uint8_t>((8 - count % 8) % 8);
    DataCopyPadExtParams<float> padding{true, 0, tail_padding, 0.0F};
    DataCopyPad(destination, source_tensor[element_offset], params, padding);
  }

  __aicore__ inline void CopyOutElements(__gm__ float *destination, uint64_t element_offset, uint32_t count,
                                         LocalTensor<float> source) const {
    GlobalTensor<float> destination_tensor;
    destination_tensor.SetGlobalBuffer(destination);
    DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    DataCopyPad(destination_tensor[element_offset], source, params);
  }

  __aicore__ inline void FinishStage(TPipe &pipe) const {
    const event_t gm_write_complete = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(gm_write_complete);
    WaitFlag<HardEvent::MTE3_S>(gm_write_complete);
    pipe.Destroy();
  }

  uint64_t first_row_ = 0;
  uint32_t row_count_ = 0;
  uint32_t batch_rows_ = 1;
};

/** Build the inputs consumed by the standalone ScatterElementsV2 kernel. */
class RouteGradPipeline : private TokenRowPipeline {
 public:
  __aicore__ inline void Init(uint32_t worker_id, __gm__ float *selected_scores, __gm__ float *denominator,
                              __gm__ float *grad_weights, __gm__ float *selected_score_grad,
                              __gm__ float *zero_score_grad, GM_ADDR workspace,
                              const HyperMegaGateRouteGradTilingData &tiling) {
    selected_scores_ = selected_scores;
    denominator_ = denominator;
    grad_weights_ = grad_weights;
    selected_score_grad_ = selected_score_grad;
    zero_score_grad_ = zero_score_grad;
    auto *base = reinterpret_cast<__gm__ uint8_t *>(workspace);
    k0_ = reinterpret_cast<__gm__ float *>(base + tiling.kSlot0Offset);
    k1_ = reinterpret_cast<__gm__ float *>(base + tiling.kSlot1Offset);
    k2_ = reinterpret_cast<__gm__ float *>(base + tiling.kSlot2Offset);
    k3_ = reinterpret_cast<__gm__ float *>(base + tiling.kSlot3Offset);
    k4_ = reinterpret_cast<__gm__ float *>(base + tiling.kSlot4Offset);
    cross_term_ = reinterpret_cast<__gm__ float *>(base + tiling.crossTermOffset);
    direct_term_ = reinterpret_cast<__gm__ float *>(base + tiling.directTermOffset);
    broadcast_row_sum_ = reinterpret_cast<__gm__ float *>(base + tiling.broadcastRowSumOffset);
    row_sum_ = reinterpret_cast<__gm__ float *>(base + tiling.rowSumOffset);
    expert_count_ = tiling.expertCount;
    top_k_ = tiling.topK;
    expert_align_ = tiling.expertAlign;
    k_align_ = tiling.kAlign;
    shared_tmp_bytes_ = tiling.sharedTmpBytes;
    scaling_ = tiling.routedScalingFactor;
    InitRows(worker_id, tiling.tokenCount, tiling.rowsPerWorker, tiling.batchRows);
  }

  __aicore__ inline void MulsScaleGrad() {
    if (top_k_ == 1) {
      RunUnaryMuls(grad_weights_, selected_score_grad_, scaling_);
      return;
    }
    RunUnaryMuls(grad_weights_, k0_, scaling_);
  }

  __aicore__ inline void BroadcastDenominator() { RunBroadcast(denominator_, k1_); }
  __aicore__ inline void NegScaledGrad() { RunUnaryNeg(k0_, k2_); }
  __aicore__ inline void DivSelected() { RunBinary(selected_scores_, k1_, k3_, false); }
  // Match div_tensor_other_backward: -grad * ((selected / denominator) / denominator).
  // Dividing -grad instead changes FP32 rounding before BF16 gradient accumulation.
  __aicore__ inline void DivSelectedRatio() { RunBinary(k3_, k1_, k4_, false); }
  __aicore__ inline void MulCross() { RunBinary(k2_, k4_, cross_term_, true); }
  __aicore__ inline void DivDirect() { RunBinary(k0_, k1_, direct_term_, false); }

  __aicore__ inline void ReduceCrossTerm() {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input_queue;
    TQue<QuePosition::VECOUT, 1> output_queue;
    TBuf<TPosition::VECCALC> shared_buffer;
    const uint32_t row_alignment = 8;
    pipe.InitBuffer(input_queue, 1, batch_rows_ * k_align_ * sizeof(float));
    pipe.InitBuffer(output_queue, 1, batch_rows_ * row_alignment * sizeof(float));
    pipe.InitBuffer(shared_buffer, shared_tmp_bytes_);
    for (uint32_t offset = 0; offset < row_count_; offset += batch_rows_) {
      const uint64_t row = first_row_ + offset;
      const uint32_t rows = BatchRowCount(offset);
      LocalTensor<float> input = input_queue.AllocTensor<float>();
      CopyInRows(input, cross_term_, row, rows, top_k_, k_align_);
      input_queue.EnQue(input);
      input = input_queue.DeQue<float>();
      LocalTensor<float> output = output_queue.AllocTensor<float>();
      LocalTensor<uint8_t> shared = shared_buffer.Get<uint8_t>();
      uint32_t shape[2] = {rows, k_align_};
      AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(output, input, shared, shape, true);
      output_queue.EnQue(output);
      output = output_queue.DeQue<float>();
      // The AR reduction result is packed as one contiguous value per row.
      // Keep that layout when writing [T, 1] to GM; row-aligned CopyOutRows
      // would incorrectly read UB positions 0, 8, 16, ... .
      CopyOutScalars(row_sum_, row, rows, output);
      input_queue.FreeTensor(input);
      output_queue.FreeTensor(output);
    }
    FinishStage(pipe);
  }

  __aicore__ inline void BroadcastRowSum() { RunBroadcast(row_sum_, broadcast_row_sum_); }

  __aicore__ inline void AddSelectedGrad() { RunAdd(direct_term_, broadcast_row_sum_, selected_score_grad_); }

  __aicore__ inline void ZerosLike() {
    TPipe pipe;
    TQue<QuePosition::VECOUT, 1> output_queue;
    pipe.InitBuffer(output_queue, 1, batch_rows_ * expert_align_ * sizeof(float));
    for (uint32_t offset = 0; offset < row_count_; offset += batch_rows_) {
      const uint64_t row = first_row_ + offset;
      const uint32_t rows = BatchRowCount(offset);
      LocalTensor<float> output = output_queue.AllocTensor<float>();
      Duplicate(output, 0.0F, rows * expert_align_);
      output_queue.EnQue(output);
      output = output_queue.DeQue<float>();
      CopyOutRows(zero_score_grad_, row, rows, expert_count_, expert_align_, output);
      output_queue.FreeTensor(output);
    }
    FinishStage(pipe);
  }

 private:
  __aicore__ inline void RunUnaryMuls(__gm__ float *input, __gm__ float *output, float scalar) {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input_queue;
    TQue<QuePosition::VECOUT, 1> output_queue;
    const uint32_t count = batch_rows_ * k_align_;
    pipe.InitBuffer(input_queue, 1, count * sizeof(float));
    pipe.InitBuffer(output_queue, 1, count * sizeof(float));
    for (uint32_t offset = 0; offset < row_count_; offset += batch_rows_) {
      const uint64_t row = first_row_ + offset;
      const uint32_t rows = BatchRowCount(offset);
      LocalTensor<float> source = input_queue.AllocTensor<float>();
      CopyInElements(source, input, row * top_k_, rows * top_k_);
      input_queue.EnQue(source);
      source = input_queue.DeQue<float>();
      LocalTensor<float> destination = output_queue.AllocTensor<float>();
      Muls(destination, source, scalar, rows * top_k_);
      output_queue.EnQue(destination);
      destination = output_queue.DeQue<float>();
      CopyOutElements(output, row * top_k_, rows * top_k_, destination);
      input_queue.FreeTensor(source);
      output_queue.FreeTensor(destination);
    }
    FinishStage(pipe);
  }

  __aicore__ inline void RunUnaryNeg(__gm__ float *input, __gm__ float *output) {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input_queue;
    TQue<QuePosition::VECOUT, 1> output_queue;
    const uint32_t count = batch_rows_ * k_align_;
    pipe.InitBuffer(input_queue, 1, count * sizeof(float));
    pipe.InitBuffer(output_queue, 1, count * sizeof(float));
    for (uint32_t offset = 0; offset < row_count_; offset += batch_rows_) {
      const uint64_t row = first_row_ + offset;
      const uint32_t rows = BatchRowCount(offset);
      LocalTensor<float> source = input_queue.AllocTensor<float>();
      CopyInElements(source, input, row * top_k_, rows * top_k_);
      input_queue.EnQue(source);
      source = input_queue.DeQue<float>();
      LocalTensor<float> destination = output_queue.AllocTensor<float>();
      AscendC::Muls(destination, source, -1.0F, rows * top_k_);
      output_queue.EnQue(destination);
      destination = output_queue.DeQue<float>();
      CopyOutElements(output, row * top_k_, rows * top_k_, destination);
      input_queue.FreeTensor(source);
      output_queue.FreeTensor(destination);
    }
    FinishStage(pipe);
  }

  __aicore__ inline void RunBroadcast(__gm__ float *input, __gm__ float *output) {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input_queue;
    TQue<QuePosition::VECOUT, 1> output_queue;
    TBuf<TPosition::VECCALC> shared_buffer;
    const uint32_t row_alignment = 8;
    pipe.InitBuffer(input_queue, 1, batch_rows_ * row_alignment * sizeof(float));
    pipe.InitBuffer(output_queue, 1, batch_rows_ * k_align_ * sizeof(float));
    pipe.InitBuffer(shared_buffer, shared_tmp_bytes_);
    for (uint32_t offset = 0; offset < row_count_; offset += batch_rows_) {
      const uint64_t row = first_row_ + offset;
      const uint32_t rows = BatchRowCount(offset);
      LocalTensor<float> source = input_queue.AllocTensor<float>();
      CopyInScalars(source, input, row, rows);
      input_queue.EnQue(source);
      source = input_queue.DeQue<float>();
      LocalTensor<float> destination = output_queue.AllocTensor<float>();
      LocalTensor<uint8_t> shared = shared_buffer.Get<uint8_t>();
      uint32_t source_shape[2] = {rows, 1};
      uint32_t destination_shape[2] = {rows, k_align_};
      AscendC::Broadcast<float, 2, 1>(destination, source, destination_shape, source_shape, shared);
      output_queue.EnQue(destination);
      destination = output_queue.DeQue<float>();
      CopyOutRows(output, row, rows, top_k_, k_align_, destination);
      input_queue.FreeTensor(source);
      output_queue.FreeTensor(destination);
    }
    FinishStage(pipe);
  }

  __aicore__ inline void RunBinary(__gm__ float *left, __gm__ float *right, __gm__ float *output, bool multiply) {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> left_queue;
    TQue<QuePosition::VECIN, 1> right_queue;
    TQue<QuePosition::VECOUT, 1> output_queue;
    const uint32_t count = batch_rows_ * k_align_;
    pipe.InitBuffer(left_queue, 1, count * sizeof(float));
    pipe.InitBuffer(right_queue, 1, count * sizeof(float));
    pipe.InitBuffer(output_queue, 1, count * sizeof(float));
    for (uint32_t offset = 0; offset < row_count_; offset += batch_rows_) {
      const uint64_t row = first_row_ + offset;
      const uint32_t rows = BatchRowCount(offset);
      LocalTensor<float> left_value = left_queue.AllocTensor<float>();
      LocalTensor<float> right_value = right_queue.AllocTensor<float>();
      // Pointwise stages need no row padding; GM remains packed [T,K] for later row-wise stages.
      CopyInElements(left_value, left, row * top_k_, rows * top_k_);
      CopyInElements(right_value, right, row * top_k_, rows * top_k_);
      left_queue.EnQue(left_value);
      right_queue.EnQue(right_value);
      left_value = left_queue.DeQue<float>();
      right_value = right_queue.DeQue<float>();
      LocalTensor<float> destination = output_queue.AllocTensor<float>();
      const uint32_t element_count = rows * top_k_;
      if (multiply) {
        Mul(destination, left_value, right_value, element_count);
      } else {
        Div(destination, left_value, right_value, element_count);
      }
      output_queue.EnQue(destination);
      destination = output_queue.DeQue<float>();
      CopyOutElements(output, row * top_k_, rows * top_k_, destination);
      left_queue.FreeTensor(left_value);
      right_queue.FreeTensor(right_value);
      output_queue.FreeTensor(destination);
    }
    FinishStage(pipe);
  }

  __aicore__ inline void RunAdd(__gm__ float *left, __gm__ float *right, __gm__ float *output) {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> left_queue;
    TQue<QuePosition::VECIN, 1> right_queue;
    TQue<QuePosition::VECOUT, 1> output_queue;
    const uint32_t count = batch_rows_ * k_align_;
    pipe.InitBuffer(left_queue, 1, count * sizeof(float));
    pipe.InitBuffer(right_queue, 1, count * sizeof(float));
    pipe.InitBuffer(output_queue, 1, count * sizeof(float));
    for (uint32_t offset = 0; offset < row_count_; offset += batch_rows_) {
      const uint64_t row = first_row_ + offset;
      const uint32_t rows = BatchRowCount(offset);
      LocalTensor<float> left_value = left_queue.AllocTensor<float>();
      LocalTensor<float> right_value = right_queue.AllocTensor<float>();
      CopyInElements(left_value, left, row * top_k_, rows * top_k_);
      CopyInElements(right_value, right, row * top_k_, rows * top_k_);
      left_queue.EnQue(left_value);
      right_queue.EnQue(right_value);
      left_value = left_queue.DeQue<float>();
      right_value = right_queue.DeQue<float>();
      LocalTensor<float> destination = output_queue.AllocTensor<float>();
      const uint32_t element_count = rows * top_k_;
      Add(destination, left_value, right_value, element_count);
      output_queue.EnQue(destination);
      destination = output_queue.DeQue<float>();
      CopyOutElements(output, row * top_k_, rows * top_k_, destination);
      left_queue.FreeTensor(left_value);
      right_queue.FreeTensor(right_value);
      output_queue.FreeTensor(destination);
    }
    FinishStage(pipe);
  }

  __gm__ float *selected_scores_ = nullptr;
  __gm__ float *denominator_ = nullptr;
  __gm__ float *grad_weights_ = nullptr;
  __gm__ float *selected_score_grad_ = nullptr;
  __gm__ float *zero_score_grad_ = nullptr;
  __gm__ float *k0_ = nullptr;
  __gm__ float *k1_ = nullptr;
  __gm__ float *k2_ = nullptr;
  __gm__ float *k3_ = nullptr;
  __gm__ float *k4_ = nullptr;
  __gm__ float *cross_term_ = nullptr;
  __gm__ float *direct_term_ = nullptr;
  __gm__ float *broadcast_row_sum_ = nullptr;
  __gm__ float *row_sum_ = nullptr;
  uint32_t expert_count_ = 0;
  uint32_t top_k_ = 0;
  uint32_t expert_align_ = 0;
  uint32_t k_align_ = 0;
  uint32_t shared_tmp_bytes_ = 0;
  float scaling_ = 1.0F;
};

}  // namespace HyperMegaGate

#endif  // HYPER_MEGA_GATE_ROUTE_GRAD_PIPELINE_H
