/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "hyper_mega_gate_route_tiling.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {

using namespace AscendC;  // NOLINT(build/namespaces) TopKMode lives in AscendC

constexpr uint32_t kAivWorkerSlotCapacity = 48;
constexpr uint32_t kRouteStageCount = 10;
constexpr uint32_t kExpertAlignment = 64;
constexpr uint32_t kSelectedExpertAlignment = 8;
constexpr uint32_t kMaximumBatchRows = 256;
constexpr uint64_t kWorkspaceAlignment = 512;

uint32_t CeilDiv(uint32_t value, uint32_t factor) { return value / factor + (value % factor != 0); }

uint64_t AlignUp(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }

ge::graphStatus TilingFunc(gert::TilingContext *context) {
  OP_CHECK_NULL_WITH_CONTEXT(context, context);
  const auto *logits_shape = context->GetInputShape(0);
  const auto *text_bias_shape = context->GetInputShape(1);
  const auto *vision_bias_shape = context->GetInputShape(2);
  const auto *image_mask_shape = context->GetInputShape(3);
  const auto *runtime_shape = context->GetInputShape(4);
  const auto *profile_shape = context->GetInputShape(5);
  OP_CHECK_NULL_WITH_CONTEXT(context, logits_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, text_bias_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, vision_bias_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, image_mask_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, runtime_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, profile_shape);
  const auto &logits = logits_shape->GetStorageShape();
  const auto &text_bias = text_bias_shape->GetStorageShape();
  const auto &vision_bias = vision_bias_shape->GetStorageShape();
  const auto &image_mask = image_mask_shape->GetStorageShape();
  const auto &runtime = runtime_shape->GetStorageShape();
  const auto &profile = profile_shape->GetStorageShape();
  auto *attrs = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const int64_t *top_k = attrs->GetAttrPointer<int64_t>(0);
  const float *scaling = attrs->GetAttrPointer<float>(1);
  const bool *use_vision_bias = attrs->GetAttrPointer<bool>(2);
  OP_CHECK_NULL_WITH_CONTEXT(context, top_k);
  OP_CHECK_NULL_WITH_CONTEXT(context, scaling);
  OP_CHECK_NULL_WITH_CONTEXT(context, use_vision_bias);
  const int64_t expected_mask_elements = *use_vision_bias ? logits.GetDim(0) : 1;
  if (logits.GetDimNum() != 2 || text_bias.GetDimNum() != 1 || vision_bias.GetDimNum() != 1 ||
      image_mask.GetDimNum() != 1 || logits.GetDim(0) <= 0 || logits.GetDim(0) > std::numeric_limits<uint32_t>::max() ||
      logits.GetDim(1) <= 0 || logits.GetDim(1) > std::numeric_limits<uint32_t>::max() ||
      text_bias.GetDim(0) != logits.GetDim(1) || vision_bias.GetDim(0) != logits.GetDim(1) ||
      image_mask.GetDim(0) != expected_mask_elements || runtime.GetDimNum() != 1 || runtime.GetDim(0) <= 0 ||
      profile.GetDimNum() != 1 || profile.GetDim(0) <= 0 || *top_k <= 0 || *top_k > logits.GetDim(1) ||
      !std::isfinite(*scaling)) {
    const auto &logits_origin = logits_shape->GetOriginShape();
    const auto &text_bias_origin = text_bias_shape->GetOriginShape();
    OP_LOGE(context->GetNodeName(),
            "Invalid HyperMegaGateRoute shape or attributes: logits storage=[%ld,%ld] rank=%zu, "
            "origin=[%ld,%ld] rank=%zu, text_bias storage=[%ld] rank=%zu, origin=[%ld] rank=%zu, "
            "vision_bias=[%ld], image_mask=[%ld], use_vision_bias=%d, top_k=%ld, scaling=%f.",
            logits.GetDimNum() > 0 ? logits.GetDim(0) : -1, logits.GetDimNum() > 1 ? logits.GetDim(1) : -1,
            logits.GetDimNum(), logits_origin.GetDimNum() > 0 ? logits_origin.GetDim(0) : -1,
            logits_origin.GetDimNum() > 1 ? logits_origin.GetDim(1) : -1, logits_origin.GetDimNum(),
            text_bias.GetDimNum() > 0 ? text_bias.GetDim(0) : -1, text_bias.GetDimNum(),
            text_bias_origin.GetDimNum() > 0 ? text_bias_origin.GetDim(0) : -1, text_bias_origin.GetDimNum(),
            vision_bias.GetDimNum() > 0 ? vision_bias.GetDim(0) : -1,
            image_mask.GetDimNum() > 0 ? image_mask.GetDim(0) : -1, static_cast<int>(*use_vision_bias), *top_k,
            *scaling);
    return ge::GRAPH_FAILED;
  }

  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  const uint32_t vector_cores = platform.GetCoreNumAiv();
  if (vector_cores == 0) {
    OP_LOGE(context->GetNodeName(), "HyperMegaGateRoute requires AIV cores.");
    return ge::GRAPH_FAILED;
  }
  const uint32_t tokens = static_cast<uint32_t>(logits.GetDim(0));
  const uint32_t worker_capacity = std::min({tokens, vector_cores, kAivWorkerSlotCapacity});
  const uint32_t rows_per_worker = CeilDiv(tokens, worker_capacity);
  const uint32_t active_workers = CeilDiv(tokens, rows_per_worker);
  const uint32_t e_align = CeilDiv(static_cast<uint32_t>(logits.GetDim(1)), kExpertAlignment) * kExpertAlignment;
  const uint32_t k_align = CeilDiv(static_cast<uint32_t>(*top_k), kSelectedExpertAlignment) * kSelectedExpertAlignment;
  uint64_t ub_size = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

  uint32_t batch = std::min(rows_per_worker, kMaximumBatchRows);
  uint32_t shared_tmp = 0;
  uint32_t topk_tmp = 0;
  while (batch > 0) {
    uint32_t sum_tmp_max = 0;
    uint32_t sum_tmp_min = 0;
    if (*top_k > 1) {
      GetSumMaxMinTmpSize(static_cast<uint32_t>(*top_k), sizeof(float), false, sum_tmp_max, sum_tmp_min);
    }
    uint32_t topk_tmp_min = 0;
    (void)GetTopKMaxMinTmpSize(platform, e_align, batch, false, true, TopKMode::TOPK_NORMAL, true, sizeof(float),
                               topk_tmp, topk_tmp_min);
    uint32_t bcast_tmp_max = 0;
    uint32_t bias_bcast_tmp = 0;
    uint32_t row_bcast_tmp = 0;
    uint32_t mask_bcast_tmp = 0;
    (void)GetBroadCastMaxMinTmpSize(platform, ge::Shape({1, static_cast<int64_t>(e_align)}),
                                    ge::Shape({batch, static_cast<int64_t>(e_align)}), sizeof(float), true,
                                    bcast_tmp_max, bias_bcast_tmp);
    (void)GetBroadCastMaxMinTmpSize(platform, ge::Shape({batch, 1}), ge::Shape({batch, k_align}), sizeof(float), false,
                                    bcast_tmp_max, row_bcast_tmp);
    (void)GetBroadCastMaxMinTmpSize(platform, ge::Shape({1, k_align}), ge::Shape({batch, k_align}), sizeof(float),
                                    false, bcast_tmp_max, mask_bcast_tmp);
    shared_tmp = std::max({sum_tmp_max, bias_bcast_tmp, row_bcast_tmp, mask_bcast_tmp});

    // Mirror every InitBuffer in RouteBatchKernel, including double-buffered
    // queues and both FP32 Softplus predicate masks. The library temporary is
    // the larger of the broadcast/sum scratch and TopK scratch.
    const uint64_t score_queue = 2ULL * batch * e_align * sizeof(float);
    const uint64_t weight_queue = 2ULL * batch * k_align * sizeof(float);
    const uint64_t score_buffers = 2ULL * batch * e_align * sizeof(float);
    const uint64_t k_buffers = static_cast<uint64_t>(batch) * k_align * (8 * sizeof(float) + sizeof(int64_t));
    const uint64_t fixed = e_align * sizeof(int32_t) + CeilDiv(batch, 8) * 8 * sizeof(float) + k_align * sizeof(float);
    const uint64_t predicate = 2ULL * CeilDiv(batch * e_align, 256) * 32ULL;
    const uint64_t vision_buffers =
      *use_vision_bias ? 2ULL * e_align * sizeof(float) + AlignUp(batch * sizeof(bool), 32) : 0;
    const uint64_t total = score_queue + weight_queue + score_buffers + k_buffers + fixed + predicate + shared_tmp +
                           topk_tmp + vision_buffers;
    if (total <= ub_size / 2) {
      break;
    }
    if (batch == 1) {
      OP_LOGE(context->GetNodeName(), "Route buffers exceed available UB.");
      return ge::GRAPH_FAILED;
    }
    batch = std::max(1u, batch / 2);
  }

  HyperMegaGateRouteTilingData tiling;
  tiling.set_tokenCount(logits.GetDim(0));
  tiling.set_expertCount(logits.GetDim(1));
  tiling.set_topK(*top_k);
  tiling.set_routedScalingFactor(*scaling);
  tiling.set_activeAivWorkerCount(active_workers);
  tiling.set_rowsPerWorker(rows_per_worker);
  tiling.set_aivWorkerSlotCapacity(kAivWorkerSlotCapacity);
  tiling.set_batchRows(batch);
  tiling.set_expertAlign(e_align);
  tiling.set_kAlign(k_align);
  tiling.set_sharedTmpBytes(shared_tmp);
  tiling.set_stageCount(kRouteStageCount);
  tiling.set_useVisionBias(*use_vision_bias ? 1U : 0U);
  tiling.set_runtimeConfigBytes(static_cast<uint64_t>(runtime.GetDim(0)));
  tiling.set_profileBufferBytes(static_cast<uint64_t>(profile.GetDim(0)));
  uint64_t route_workspace = 0;
  const uint64_t score_bytes = static_cast<uint64_t>(tokens) * logits.GetDim(1) * sizeof(float);
  const uint64_t selected_float_bytes = static_cast<uint64_t>(tokens) * *top_k * sizeof(float);
  const uint64_t selected_i32_bytes = static_cast<uint64_t>(tokens) * *top_k * sizeof(int32_t);
  const uint64_t row_bytes = static_cast<uint64_t>(tokens) * sizeof(float);
  tiling.set_scoreAOffset(route_workspace);
  route_workspace = AlignUp(route_workspace + score_bytes, kWorkspaceAlignment);
  tiling.set_topkValuesOffset(route_workspace);
  route_workspace = AlignUp(route_workspace + selected_float_bytes, kWorkspaceAlignment);
  tiling.set_indicesI32Offset(route_workspace);
  route_workspace = AlignUp(route_workspace + selected_i32_bytes, kWorkspaceAlignment);
  tiling.set_rowSumOffset(route_workspace);
  route_workspace = AlignUp(route_workspace + row_bytes, kWorkspaceAlignment);
  tiling.set_routeWorkspaceBytes(route_workspace);
  TopKTilingFunc(platform, e_align, batch, static_cast<uint32_t>(*top_k), sizeof(float), true, TopKMode::TOPK_NORMAL,
                 true, tiling.topkTiling);
  OP_LOGD(context->GetNodeName(),
          "HyperMegaGate Route tiling: tokens=%u experts=%ld availableAivWorkers=%u activeAivWorkers=%u "
          "rowsPerWorker=%u batchRows=%u useVisionBias=%d",
          tokens, logits.GetDim(1), vector_cores, active_workers, rows_per_worker, batch,
          static_cast<int>(*use_vision_bias));
  auto *raw_tiling = context->GetRawTilingData();
  OP_CHECK_NULL_WITH_CONTEXT(context, raw_tiling);
  tiling.SaveToBuffer(raw_tiling->GetData(), raw_tiling->GetCapacity());
  raw_tiling->SetDataSize(tiling.GetDataSize());
  context->SetBlockDim(active_workers);
  context->SetTilingKey(0);
  size_t *workspace_sizes = context->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, workspace_sizes);
  workspace_sizes[0] = platform.GetLibApiWorkSpaceSize() + route_workspace;
  return ge::GRAPH_SUCCESS;
}

struct HyperMegaGateRouteCompileInfo {};

ge::graphStatus TilingPrepare(gert::TilingParseContext *context) {
  return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

}  // namespace

IMPL_OP_OPTILING(HyperMegaGateRoute).Tiling(TilingFunc).TilingParse<HyperMegaGateRouteCompileInfo>(TilingPrepare);

}  // namespace optiling
