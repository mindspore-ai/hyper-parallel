/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifndef HYPER_MEGA_GATE_ROUTE_PIPELINE_H
#define HYPER_MEGA_GATE_ROUTE_PIPELINE_H

#include "kernel_operator.h"

namespace HyperMegaGate {

using namespace AscendC;  // NOLINT(build/namespaces)

/**
 * @brief Execute one row shard of the ten-stage sqrt-softplus routing pipeline.
 *
 * The caller owns a fixed contiguous row range. Every stage reads and writes
 * only that range, so workers can execute the same descriptor stream without
 * cross-core events. Intermediate tensors are packed in GM and reuse the
 * workspace offsets supplied by the operator tiling data. UB queues own the
 * in-flight copy lifetime within a stage.
 */
class RoutePipeline {
 public:
  __aicore__ inline void Init(__gm__ float *logits, __gm__ float *text_bias, __gm__ float *vision_bias,
                              __gm__ bool *image_mask, bool use_vision_bias, __gm__ float *routing_weights,
                              __gm__ int64_t *expert_indices, __gm__ float *route_scores, __gm__ float *selected_scores,
                              __gm__ float *normalization_denominator, GM_ADDR user_workspace, int64_t expert_count,
                              int64_t top_k, float routed_scaling_factor, uint32_t batch_rows, uint32_t expert_align,
                              uint32_t k_align, uint32_t shared_tmp_bytes, uint64_t score_a_offset,
                              uint64_t topk_values_offset, uint64_t indices_i32_offset, uint64_t row_sum_offset,
                              const TopkTiling *topk_tiling, uint64_t first_row, uint32_t row_count) {
    logits_ = logits;
    text_bias_ = text_bias;
    vision_bias_ = vision_bias;
    image_mask_ = image_mask;
    use_vision_bias_ = use_vision_bias;
    routing_weights_ = routing_weights;
    expert_indices_ = expert_indices;
    route_scores_ = route_scores;
    selected_scores_ = selected_scores;
    normalization_denominator_ = normalization_denominator;
    score_a_ = reinterpret_cast<__gm__ float *>(user_workspace + score_a_offset);
    topk_values_ = reinterpret_cast<__gm__ float *>(user_workspace + topk_values_offset);
    indices_i32_ = reinterpret_cast<__gm__ int32_t *>(user_workspace + indices_i32_offset);
    row_sum_ = reinterpret_cast<__gm__ float *>(user_workspace + row_sum_offset);
    expert_count_ = expert_count;
    top_k_ = top_k;
    routed_scaling_factor_ = routed_scaling_factor;
    batch_rows_ = batch_rows;
    expert_align_ = expert_align;
    k_align_ = k_align;
    shared_tmp_bytes_ = shared_tmp_bytes;
    topk_tiling_ = topk_tiling;
    first_row_ = first_row;
    row_count_ = row_count;
  }

  __aicore__ inline void Softplus() {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input_queue;
    TQue<QuePosition::VECOUT, 1> output_queue;
    TBuf<TPosition::VECCALC> exp_buffer;
    TBuf<TPosition::VECCALC> temp_buffer;
    TBuf<TPosition::VECCALC> high_mask_buffer;
    TBuf<TPosition::VECCALC> log1p_mask_buffer;
    const uint32_t score_count = batch_rows_ * expert_align_;
    const uint32_t mask_bytes = ((score_count + 255) / 256) * 32;
    pipe.InitBuffer(input_queue, 1, score_count * sizeof(float));
    pipe.InitBuffer(output_queue, 1, score_count * sizeof(float));
    pipe.InitBuffer(exp_buffer, score_count * sizeof(float));
    pipe.InitBuffer(temp_buffer, score_count * sizeof(float));
    pipe.InitBuffer(high_mask_buffer, mask_bytes);
    pipe.InitBuffer(log1p_mask_buffer, mask_bytes);

    for (uint32_t offset = 0; offset < row_count_; offset += batch_rows_) {
      const uint64_t row = first_row_ + offset;
      const uint32_t rows = BatchRowCount(offset);
      LocalTensor<float> input = input_queue.AllocTensor<float>();
      CopyInRows(input, logits_, row, rows, expert_count_, expert_align_);
      input_queue.EnQue(input);
      input = input_queue.DeQue<float>();
      LocalTensor<float> output = output_queue.AllocTensor<float>();
      LocalTensor<float> exp_value = exp_buffer.Get<float>();
      LocalTensor<float> temp = temp_buffer.Get<float>();
      LocalTensor<uint8_t> high_mask = high_mask_buffer.Get<uint8_t>();
      LocalTensor<uint8_t> log1p_mask = log1p_mask_buffer.Get<uint8_t>();
      const uint32_t count = rows * expert_align_;

      CompareScalar(high_mask, input, 20.0f, CMPMODE::GT, count);
      PipeBarrier<PIPE_V>();
      Mins(exp_value, input, 20.0f, count);
      PipeBarrier<PIPE_V>();
      Exp(exp_value, exp_value, count);
      PipeBarrier<PIPE_V>();
      Adds(temp, exp_value, 1.0f, count);
      PipeBarrier<PIPE_V>();
      CompareScalar(log1p_mask, temp, 1.0f, CMPMODE::EQ, count);
      PipeBarrier<PIPE_V>();
      Adds(temp, temp, -1.0f, count);
      PipeBarrier<PIPE_V>();
      Select(output, high_mask, input, temp, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
      PipeBarrier<PIPE_V>();
      Adds(temp, temp, 1.0f, count);
      PipeBarrier<PIPE_V>();
      Log(temp, temp, count);
      PipeBarrier<PIPE_V>();
      // Match the compensated log1p order: log(u) * (exp(x) / (u - 1)).
      // Dividing log(u) first can change rounding at BF16 gradient midpoints.
      Div(output, exp_value, output, count);
      PipeBarrier<PIPE_V>();
      Mul(temp, temp, output, count);
      PipeBarrier<PIPE_V>();
      Select(temp, log1p_mask, exp_value, temp, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
      PipeBarrier<PIPE_V>();
      Select(output, high_mask, input, temp, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
      output_queue.EnQue(output);
      output = output_queue.DeQue<float>();
      CopyOutRows(score_a_, row, rows, expert_count_, expert_align_, output);
      input_queue.FreeTensor(input);
      output_queue.FreeTensor(output);
    }
    FinishStage(pipe);
  }

  __aicore__ inline void SqrtScore() {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input_queue;
    TQue<QuePosition::VECOUT, 1> output_queue;
    const uint32_t count = batch_rows_ * expert_align_;
    pipe.InitBuffer(input_queue, 1, count * sizeof(float));
    pipe.InitBuffer(output_queue, 1, count * sizeof(float));
    for (uint32_t offset = 0; offset < row_count_; offset += batch_rows_) {
      const uint64_t row = first_row_ + offset;
      const uint32_t rows = BatchRowCount(offset);
      LocalTensor<float> input = input_queue.AllocTensor<float>();
      CopyInRows(input, score_a_, row, rows, expert_count_, expert_align_);
      input_queue.EnQue(input);
      input = input_queue.DeQue<float>();
      LocalTensor<float> output = output_queue.AllocTensor<float>();
      Sqrt(output, input, rows * expert_align_);
      output_queue.EnQue(output);
      output = output_queue.DeQue<float>();
      CopyOutRows(route_scores_, row, rows, expert_count_, expert_align_, output);
      input_queue.FreeTensor(input);
      output_queue.FreeTensor(output);
    }
    FinishStage(pipe);
  }

  __aicore__ inline void AddBias() {
    if (use_vision_bias_) {
      AddVisionBias();
      return;
    }
    AddTextBias();
  }

  __aicore__ inline void AddTextBias() {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input_queue;
    TQue<QuePosition::VECOUT, 1> output_queue;
    TBuf<TPosition::VECCALC> bias_buffer;
    const uint32_t count = batch_rows_ * expert_align_;
    pipe.InitBuffer(input_queue, 1, count * sizeof(float));
    pipe.InitBuffer(output_queue, 1, count * sizeof(float));
    pipe.InitBuffer(bias_buffer, expert_align_ * sizeof(float));
    LocalTensor<float> local_bias = bias_buffer.Get<float>();
    CopyInVector(local_bias, text_bias_, expert_count_);
    event_t bias_ready = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(bias_ready);
    WaitFlag<HardEvent::MTE2_V>(bias_ready);

    for (uint32_t offset = 0; offset < row_count_; offset += batch_rows_) {
      const uint64_t row = first_row_ + offset;
      const uint32_t rows = BatchRowCount(offset);
      LocalTensor<float> input = input_queue.AllocTensor<float>();
      CopyInRows(input, route_scores_, row, rows, expert_count_, expert_align_);
      input_queue.EnQue(input);
      input = input_queue.DeQue<float>();
      LocalTensor<float> output = output_queue.AllocTensor<float>();
      for (uint32_t local_row = 0; local_row < rows; ++local_row) {
        Add(output[local_row * expert_align_], input[local_row * expert_align_], local_bias,
            static_cast<uint32_t>(expert_count_));
      }
      output_queue.EnQue(output);
      output = output_queue.DeQue<float>();
      CopyOutRows(score_a_, row, rows, expert_count_, expert_align_, output);
      input_queue.FreeTensor(input);
      output_queue.FreeTensor(output);
    }
    FinishStage(pipe);
  }

  __aicore__ inline void AddVisionBias() {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input_queue;
    TQue<QuePosition::VECIN, 1> text_bias_queue;
    TQue<QuePosition::VECIN, 1> vision_bias_queue;
    TQue<QuePosition::VECIN, 1> mask_queue;
    TQue<QuePosition::VECOUT, 1> output_queue;
    const uint32_t count = batch_rows_ * expert_align_;
    const uint32_t mask_bytes = ((batch_rows_ + 31) / 32) * 32;
    pipe.InitBuffer(input_queue, 1, count * sizeof(float));
    pipe.InitBuffer(text_bias_queue, 1, expert_align_ * sizeof(float));
    pipe.InitBuffer(vision_bias_queue, 1, expert_align_ * sizeof(float));
    pipe.InitBuffer(mask_queue, 1, mask_bytes);
    pipe.InitBuffer(output_queue, 1, count * sizeof(float));

    LocalTensor<float> local_text_bias = text_bias_queue.AllocTensor<float>();
    CopyInVector(local_text_bias, text_bias_, expert_count_);
    text_bias_queue.EnQue(local_text_bias);
    local_text_bias = text_bias_queue.DeQue<float>();
    LocalTensor<float> local_vision_bias = vision_bias_queue.AllocTensor<float>();
    CopyInVector(local_vision_bias, vision_bias_, expert_count_);
    vision_bias_queue.EnQue(local_vision_bias);
    local_vision_bias = vision_bias_queue.DeQue<float>();

    for (uint32_t offset = 0; offset < row_count_; offset += batch_rows_) {
      const uint64_t row = first_row_ + offset;
      const uint32_t rows = BatchRowCount(offset);
      LocalTensor<float> input = input_queue.AllocTensor<float>();
      CopyInRows(input, route_scores_, row, rows, expert_count_, expert_align_);
      input_queue.EnQue(input);
      input = input_queue.DeQue<float>();
      LocalTensor<bool> mask = mask_queue.AllocTensor<bool>();
      CopyInVector(mask, image_mask_ + row, rows);
      mask_queue.EnQue(mask);
      mask = mask_queue.DeQue<bool>();
      LocalTensor<float> output = output_queue.AllocTensor<float>();
      for (uint32_t local_row = 0; local_row < rows; ++local_row) {
        if (mask.GetValue(local_row) != 0) {
          Add(output[local_row * expert_align_], input[local_row * expert_align_], local_vision_bias,
              static_cast<uint32_t>(expert_count_));
        } else {
          Add(output[local_row * expert_align_], input[local_row * expert_align_], local_text_bias,
              static_cast<uint32_t>(expert_count_));
        }
      }
      output_queue.EnQue(output);
      output = output_queue.DeQue<float>();
      CopyOutRows(score_a_, row, rows, expert_count_, expert_align_, output);
      input_queue.FreeTensor(input);
      mask_queue.FreeTensor(mask);
      output_queue.FreeTensor(output);
    }
    text_bias_queue.FreeTensor(local_text_bias);
    vision_bias_queue.FreeTensor(local_vision_bias);
    FinishStage(pipe);
  }

  __aicore__ inline void TopKScore() {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input_queue;
    TQue<QuePosition::VECOUT, 1> value_queue;
    TQue<QuePosition::VECOUT, 1> index_queue;
    TBuf<TPosition::VECCALC> one_dim_buffer;
    pipe.InitBuffer(input_queue, 1, batch_rows_ * expert_align_ * sizeof(float));
    pipe.InitBuffer(one_dim_buffer, expert_align_ * sizeof(int32_t));
    pipe.InitBuffer(value_queue, 1, batch_rows_ * k_align_ * sizeof(float));
    pipe.InitBuffer(index_queue, 1, batch_rows_ * k_align_ * sizeof(int32_t));
    LocalTensor<int32_t> one_dim = one_dim_buffer.Get<int32_t>();
    ArithProgression(one_dim, 0, 1, expert_align_);
    PipeBarrier<PIPE_V>();
    event_t padding_ready = static_cast<event_t>(0);
    union {
      uint32_t bits;
      float value;
    } negative_infinity{0xFF800000};
    if (expert_count_ != expert_align_) {
      padding_ready = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE2));
    }

    for (uint32_t offset = 0; offset < row_count_; offset += batch_rows_) {
      const uint64_t row = first_row_ + offset;
      const uint32_t rows = BatchRowCount(offset);
      LocalTensor<float> input = input_queue.AllocTensor<float>();
      if (expert_count_ != expert_align_) {
        Duplicate(input, negative_infinity.value, rows * expert_align_);
        SetFlag<HardEvent::V_MTE2>(padding_ready);
        WaitFlag<HardEvent::V_MTE2>(padding_ready);
        // DMA overwrites the last partial 32-byte block, including prefilled lanes.
        // Explicit padding keeps it at -inf; the remaining 64-lane stride stays prefilled.
        GlobalTensor<float> source;
        source.SetGlobalBuffer(score_a_);
        const uint32_t columns = static_cast<uint32_t>(expert_count_);
        const uint32_t block_columns = (columns + 7) / 8 * 8;
        DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(columns * sizeof(float)), 0,
                                 (expert_align_ - block_columns) / 8, 0};
        DataCopyPadExtParams<float> padding{true, 0, static_cast<uint8_t>(block_columns - columns),
                                            negative_infinity.value};
        DataCopyPad(input, source[row * columns], params, padding);
      } else {
        CopyInRows(input, score_a_, row, rows, expert_count_, expert_align_);
      }
      input_queue.EnQue(input);
      input = input_queue.DeQue<float>();
      LocalTensor<float> values = value_queue.AllocTensor<float>();
      LocalTensor<int32_t> indices = index_queue.AllocTensor<int32_t>();
      LocalTensor<bool> finished;
      TopKInfo info;
      info.outter = rows;
      info.inner = expert_align_;
      info.n = static_cast<uint32_t>(expert_count_);
      TopK<float, true, false, true, TopKMode::TOPK_NORMAL>(values, indices, input, one_dim, finished,
                                                            static_cast<uint32_t>(top_k_), *topk_tiling_, info, true);
      value_queue.EnQue(values);
      index_queue.EnQue(indices);
      values = value_queue.DeQue<float>();
      indices = index_queue.DeQue<int32_t>();
      CopyOutRows(topk_values_, row, rows, top_k_, k_align_, values);
      CopyOutRows(indices_i32_, row, rows, top_k_, k_align_, indices);
      input_queue.FreeTensor(input);
      value_queue.FreeTensor(values);
      index_queue.FreeTensor(indices);
    }
    FinishStage(pipe);
  }

  __aicore__ inline void GatherScore() {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> score_queue;
    TQue<QuePosition::VECIN, 1> index_queue;
    TQue<QuePosition::VECOUT, 1> gather_queue;
    TBuf<TPosition::VECCALC> row_index_buffer;
    TBuf<TPosition::VECCALC> row_base_buffer;
    TBuf<TPosition::VECCALC> offset_fp32_buffer;
    TBuf<TPosition::VECCALC> offset_buffer;
    TBuf<TPosition::VECCALC> mask_row_buffer;
    TBuf<TPosition::VECCALC> mask_buffer;
    TBuf<TPosition::VECCALC> shared_tmp_buffer;
    pipe.InitBuffer(score_queue, 1, batch_rows_ * expert_align_ * sizeof(float));
    pipe.InitBuffer(index_queue, 1, batch_rows_ * k_align_ * sizeof(int32_t));
    pipe.InitBuffer(row_index_buffer, ((batch_rows_ + 7) / 8) * 8 * sizeof(float));
    pipe.InitBuffer(row_base_buffer, batch_rows_ * k_align_ * sizeof(float));
    pipe.InitBuffer(offset_fp32_buffer, batch_rows_ * k_align_ * sizeof(float));
    pipe.InitBuffer(offset_buffer, batch_rows_ * k_align_ * sizeof(int32_t));
    pipe.InitBuffer(gather_queue, 1, batch_rows_ * k_align_ * sizeof(float));
    pipe.InitBuffer(mask_row_buffer, k_align_ * sizeof(float));
    pipe.InitBuffer(mask_buffer, batch_rows_ * k_align_ * sizeof(float));
    pipe.InitBuffer(shared_tmp_buffer, shared_tmp_bytes_);
    LocalTensor<float> mask_row = mask_row_buffer.Get<float>();
    Duplicate(mask_row, 0.0f, k_align_);
    PipeBarrier<PIPE_V>();
    Duplicate(mask_row, 1.0f, static_cast<uint32_t>(top_k_));
    PipeBarrier<PIPE_V>();

    for (uint32_t offset = 0; offset < row_count_; offset += batch_rows_) {
      const uint64_t row = first_row_ + offset;
      const uint32_t rows = BatchRowCount(offset);
      LocalTensor<float> scores = score_queue.AllocTensor<float>();
      CopyInRows(scores, route_scores_, row, rows, expert_count_, expert_align_);
      score_queue.EnQue(scores);
      scores = score_queue.DeQue<float>();
      LocalTensor<int32_t> indices = index_queue.AllocTensor<int32_t>();
      CopyInRows(indices, indices_i32_, row, rows, top_k_, k_align_);
      index_queue.EnQue(indices);
      indices = index_queue.DeQue<int32_t>();
      LocalTensor<int32_t> row_index_i32 = row_index_buffer.Get<int32_t>();
      ArithProgression(row_index_i32, static_cast<int32_t>(0), static_cast<int32_t>(expert_align_ * sizeof(float)),
                       rows);
      PipeBarrier<PIPE_V>();
      LocalTensor<float> row_index = offset_fp32_buffer.Get<float>();
      Cast(row_index, row_index_i32, RoundMode::CAST_NONE, rows);
      PipeBarrier<PIPE_V>();
      LocalTensor<float> row_base = row_base_buffer.Get<float>();
      LocalTensor<uint8_t> shared_tmp = shared_tmp_buffer.Get<uint8_t>();
      uint32_t base_dst[2] = {rows, k_align_};
      uint32_t base_src[2] = {rows, 1};
      BroadCast<float, 2, 1>(row_base, row_index, base_dst, base_src, shared_tmp);
      PipeBarrier<PIPE_V>();

      LocalTensor<float> offset_fp32 = offset_fp32_buffer.Get<float>();
      Cast(offset_fp32, indices, RoundMode::CAST_NONE, rows * k_align_);
      PipeBarrier<PIPE_V>();
      if (k_align_ != static_cast<uint32_t>(top_k_)) {
        LocalTensor<float> mask = mask_buffer.Get<float>();
        uint32_t mask_dst[2] = {rows, k_align_};
        uint32_t mask_src[2] = {1, k_align_};
        BroadCast<float, 2, 0>(mask, mask_row, mask_dst, mask_src, shared_tmp);
        PipeBarrier<PIPE_V>();
        Mul(offset_fp32, offset_fp32, mask, rows * k_align_);
        PipeBarrier<PIPE_V>();
      }
      Muls(offset_fp32, offset_fp32, 4.0f, rows * k_align_);
      PipeBarrier<PIPE_V>();
      Add(offset_fp32, offset_fp32, row_base, rows * k_align_);
      PipeBarrier<PIPE_V>();
      LocalTensor<int32_t> offsets = offset_buffer.Get<int32_t>();
      Cast(offsets, offset_fp32, RoundMode::CAST_RINT, rows * k_align_);
      PipeBarrier<PIPE_V>();
      LocalTensor<float> gathered = gather_queue.AllocTensor<float>();
      Gather(gathered, scores, offsets.ReinterpretCast<uint32_t>(), 0, rows * k_align_);
      gather_queue.EnQue(gathered);
      gathered = gather_queue.DeQue<float>();
      CopyOutRows(selected_scores_, row, rows, top_k_, k_align_, gathered);
      score_queue.FreeTensor(scores);
      index_queue.FreeTensor(indices);
      gather_queue.FreeTensor(gathered);
    }
    FinishStage(pipe);
  }

  __aicore__ inline void ReduceSelected() {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input_queue;
    TQue<QuePosition::VECOUT, 1> sum_queue;
    TBuf<TPosition::VECCALC> shared_tmp_buffer;
    pipe.InitBuffer(input_queue, 1, batch_rows_ * k_align_ * sizeof(float));
    pipe.InitBuffer(sum_queue, 1, ((batch_rows_ + 7) / 8) * 8 * sizeof(float));
    pipe.InitBuffer(shared_tmp_buffer, shared_tmp_bytes_);
    for (uint32_t offset = 0; offset < row_count_; offset += batch_rows_) {
      const uint64_t row = first_row_ + offset;
      const uint32_t rows = BatchRowCount(offset);
      LocalTensor<float> input = input_queue.AllocTensor<float>();
      CopyInRows(input, selected_scores_, row, rows, top_k_, k_align_);
      input_queue.EnQue(input);
      input = input_queue.DeQue<float>();
      LocalTensor<float> sums = sum_queue.AllocTensor<float>();
      LocalTensor<uint8_t> shared_tmp = shared_tmp_buffer.Get<uint8_t>();
      SumParams params{rows, k_align_, static_cast<uint32_t>(top_k_)};
      Sum(sums, input, shared_tmp, params);
      sum_queue.EnQue(sums);
      sums = sum_queue.DeQue<float>();
      CopyOutVector(row_sum_, row, rows, sums);
      input_queue.FreeTensor(input);
      sum_queue.FreeTensor(sums);
    }
    FinishStage(pipe);
  }

  __aicore__ inline void AddEpsilon() {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input_queue;
    TQue<QuePosition::VECOUT, 1> output_queue;
    const uint32_t row_align = 8;
    pipe.InitBuffer(input_queue, 1, batch_rows_ * row_align * sizeof(float));
    pipe.InitBuffer(output_queue, 1, batch_rows_ * row_align * sizeof(float));
    for (uint32_t offset = 0; offset < row_count_; offset += batch_rows_) {
      const uint64_t row = first_row_ + offset;
      const uint32_t rows = BatchRowCount(offset);
      LocalTensor<float> input = input_queue.AllocTensor<float>();
      CopyInVector(input, row_sum_ + row, rows);
      input_queue.EnQue(input);
      input = input_queue.DeQue<float>();
      LocalTensor<float> output = output_queue.AllocTensor<float>();
      Adds(output, input, 1.0e-20f, rows);
      output_queue.EnQue(output);
      output = output_queue.DeQue<float>();
      CopyOutVector(normalization_denominator_, row, rows, output);
      input_queue.FreeTensor(input);
      output_queue.FreeTensor(output);
    }
    FinishStage(pipe);
  }

  __aicore__ inline void DivideSelected() {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> weight_queue;
    TQue<QuePosition::VECIN, 1> denominator_queue;
    TQue<QuePosition::VECOUT, 1> output_queue;
    TBuf<TPosition::VECCALC> broadcast_buffer;
    TBuf<TPosition::VECCALC> shared_tmp_buffer;
    pipe.InitBuffer(weight_queue, 1, batch_rows_ * k_align_ * sizeof(float));
    pipe.InitBuffer(denominator_queue, 1, ((batch_rows_ + 7) / 8) * 8 * sizeof(float));
    pipe.InitBuffer(output_queue, 1, batch_rows_ * k_align_ * sizeof(float));
    pipe.InitBuffer(broadcast_buffer, batch_rows_ * k_align_ * sizeof(float));
    pipe.InitBuffer(shared_tmp_buffer, shared_tmp_bytes_);
    for (uint32_t offset = 0; offset < row_count_; offset += batch_rows_) {
      const uint64_t row = first_row_ + offset;
      const uint32_t rows = BatchRowCount(offset);
      LocalTensor<float> weights = weight_queue.AllocTensor<float>();
      CopyInRows(weights, selected_scores_, row, rows, top_k_, k_align_);
      weight_queue.EnQue(weights);
      weights = weight_queue.DeQue<float>();
      LocalTensor<float> denominators = denominator_queue.AllocTensor<float>();
      CopyInVector(denominators, normalization_denominator_ + row, rows);
      denominator_queue.EnQue(denominators);
      denominators = denominator_queue.DeQue<float>();
      LocalTensor<float> output = output_queue.AllocTensor<float>();
      if (top_k_ == 1) {
        DataCopy(output, weights, rows * k_align_);
      } else {
        LocalTensor<float> broadcast = broadcast_buffer.Get<float>();
        LocalTensor<uint8_t> shared_tmp = shared_tmp_buffer.Get<uint8_t>();
        uint32_t dst_shape[2] = {rows, k_align_};
        uint32_t src_shape[2] = {rows, 1};
        BroadCast<float, 2, 1>(broadcast, denominators, dst_shape, src_shape, shared_tmp);
        PipeBarrier<PIPE_V>();
        Div(output, weights, broadcast, rows * k_align_);
      }
      output_queue.EnQue(output);
      output = output_queue.DeQue<float>();
      CopyOutRows(topk_values_, row, rows, top_k_, k_align_, output);
      weight_queue.FreeTensor(weights);
      denominator_queue.FreeTensor(denominators);
      output_queue.FreeTensor(output);
    }
    FinishStage(pipe);
  }

  __aicore__ inline void ScaleSelected() {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input_queue;
    TQue<QuePosition::VECOUT, 1> output_queue;
    pipe.InitBuffer(input_queue, 1, batch_rows_ * k_align_ * sizeof(float));
    pipe.InitBuffer(output_queue, 1, batch_rows_ * k_align_ * sizeof(float));
    for (uint32_t offset = 0; offset < row_count_; offset += batch_rows_) {
      const uint64_t row = first_row_ + offset;
      const uint32_t rows = BatchRowCount(offset);
      LocalTensor<float> input = input_queue.AllocTensor<float>();
      CopyInRows(input, topk_values_, row, rows, top_k_, k_align_);
      input_queue.EnQue(input);
      input = input_queue.DeQue<float>();
      LocalTensor<float> output = output_queue.AllocTensor<float>();
      Muls(output, input, routed_scaling_factor_, rows * k_align_);
      output_queue.EnQue(output);
      output = output_queue.DeQue<float>();
      CopyOutRows(routing_weights_, row, rows, top_k_, k_align_, output);
      input_queue.FreeTensor(input);
      output_queue.FreeTensor(output);
    }
    FinishStage(pipe);
  }

  __aicore__ inline void CastIndices() {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input_queue;
    TQue<QuePosition::VECOUT, 1> output_queue;
    pipe.InitBuffer(input_queue, 1, batch_rows_ * k_align_ * sizeof(int32_t));
    pipe.InitBuffer(output_queue, 1, batch_rows_ * k_align_ * sizeof(int64_t));
    for (uint32_t offset = 0; offset < row_count_; offset += batch_rows_) {
      const uint64_t row = first_row_ + offset;
      const uint32_t rows = BatchRowCount(offset);
      LocalTensor<int32_t> input = input_queue.AllocTensor<int32_t>();
      CopyInRows(input, indices_i32_, row, rows, top_k_, k_align_);
      input_queue.EnQue(input);
      input = input_queue.DeQue<int32_t>();
      LocalTensor<int64_t> output = output_queue.AllocTensor<int64_t>();
      Cast(output, input, RoundMode::CAST_NONE, rows * k_align_);
      output_queue.EnQue(output);
      output = output_queue.DeQue<int64_t>();
      CopyOutRows(expert_indices_, row, rows, top_k_, k_align_, output);
      input_queue.FreeTensor(input);
      output_queue.FreeTensor(output);
    }
    FinishStage(pipe);
  }

 private:
  __aicore__ inline uint32_t BatchRowCount(uint32_t offset) const {
    const uint32_t remaining = row_count_ - offset;
    return remaining < batch_rows_ ? remaining : batch_rows_;
  }

  template <typename T>
  __aicore__ inline void CopyInRows(LocalTensor<T> destination, __gm__ T *source, uint64_t row, uint32_t rows,
                                    int64_t columns, uint32_t alignment) {
    GlobalTensor<T> source_tensor;
    source_tensor.SetGlobalBuffer(source);
    DataCopyExtParams params{
      static_cast<uint16_t>(rows), static_cast<uint32_t>(columns * sizeof(T)), 0,
      static_cast<uint32_t>(alignment * sizeof(T) / 32 - (static_cast<uint32_t>(columns) * sizeof(T) + 31) / 32), 0};
    DataCopyPadExtParams<T> padding{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(destination, source_tensor[row * columns], params, padding);
  }

  template <typename T>
  __aicore__ inline void CopyOutRows(__gm__ T *destination, uint64_t row, uint32_t rows, int64_t columns,
                                     uint32_t alignment, LocalTensor<T> source) {
    GlobalTensor<T> destination_tensor;
    destination_tensor.SetGlobalBuffer(destination);
    DataCopyExtParams params{
      static_cast<uint16_t>(rows), static_cast<uint32_t>(columns * sizeof(T)),
      static_cast<uint32_t>(alignment * sizeof(T) / 32 - (static_cast<uint32_t>(columns) * sizeof(T) + 31) / 32), 0, 0};
    DataCopyPad(destination_tensor[row * columns], source, params);
  }

  template <typename T>
  __aicore__ inline void CopyInVector(LocalTensor<T> destination, __gm__ T *source, uint32_t count) {
    GlobalTensor<T> source_tensor;
    source_tensor.SetGlobalBuffer(source);
    DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padding{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(destination, source_tensor, params, padding);
  }

  template <typename T>
  __aicore__ inline void CopyOutVector(__gm__ T *destination, uint64_t offset, uint32_t count, LocalTensor<T> source) {
    GlobalTensor<T> destination_tensor;
    destination_tensor.SetGlobalBuffer(destination);
    DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPad(destination_tensor[offset], source, params);
  }

  __aicore__ inline void FinishStage(TPipe &pipe) {
    const event_t gm_write_complete = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(gm_write_complete);
    WaitFlag<HardEvent::MTE3_S>(gm_write_complete);
    pipe.Destroy();
  }

  __gm__ float *logits_ = nullptr;
  __gm__ float *text_bias_ = nullptr;
  __gm__ float *vision_bias_ = nullptr;
  __gm__ bool *image_mask_ = nullptr;
  bool use_vision_bias_ = false;
  __gm__ float *routing_weights_ = nullptr;
  __gm__ int64_t *expert_indices_ = nullptr;
  __gm__ float *route_scores_ = nullptr;
  __gm__ float *selected_scores_ = nullptr;
  __gm__ float *normalization_denominator_ = nullptr;
  __gm__ float *score_a_ = nullptr;
  __gm__ float *topk_values_ = nullptr;
  __gm__ int32_t *indices_i32_ = nullptr;
  __gm__ float *row_sum_ = nullptr;
  int64_t expert_count_ = 0;
  int64_t top_k_ = 0;
  float routed_scaling_factor_ = 1.0f;
  uint32_t batch_rows_ = 1;
  uint32_t expert_align_ = 0;
  uint32_t k_align_ = 0;
  uint32_t shared_tmp_bytes_ = 0;
  const TopkTiling *topk_tiling_ = nullptr;
  uint64_t first_row_ = 0;
  uint32_t row_count_ = 0;
};

}  // namespace HyperMegaGate

#endif  // HYPER_MEGA_GATE_ROUTE_PIPELINE_H
