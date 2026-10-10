/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "hyper_mega_gate_route_grad_tiling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {

constexpr uint32_t kAivWorkerSlotCapacity = 48;
constexpr uint32_t kGreaterOneStageCount = 11;
constexpr uint32_t kOneStageCount = 2;
constexpr uint32_t kFp32Alignment = 8;
constexpr uint64_t kWorkspaceAlignment = 512;
constexpr uint64_t kUbReserveBytes = 1024;

uint32_t CeilDiv(uint32_t value, uint32_t divisor) { return value / divisor + (value % divisor != 0); }

uint64_t AlignUp(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }

uint64_t ElementCount(const gert::Shape &shape) {
  if (shape.GetDimNum() == 0) {
    return 0;
  }
  uint64_t count = 1;
  for (size_t index = 0; index < shape.GetDimNum(); ++index) {
    const int64_t extent = shape.GetDim(index);
    if (extent <= 0 || count > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(extent)) {
      return 0;
    }
    count *= static_cast<uint64_t>(extent);
  }
  return count;
}

bool IsMatrix(const gert::Shape &shape, int64_t rows, int64_t columns) {
  return shape.GetDimNum() == 2 && shape.GetDim(0) == rows && shape.GetDim(1) == columns;
}

bool ReserveWorkspace(uint64_t bytes, uint64_t &total, uint64_t &offset) {
  offset = total;
  if (bytes > std::numeric_limits<uint64_t>::max() - total) {
    return false;
  }
  total = AlignUp(total + bytes, kWorkspaceAlignment);
  return true;
}

struct WorkerLayout {
  uint32_t available_workers;
  uint32_t active_workers;
  uint32_t rows_per_worker;
  uint64_t ub_bytes;
  uint64_t library_workspace;
};

bool GetWorkerLayout(gert::TilingContext *context, uint32_t tokens, WorkerLayout &layout) {
  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  layout.available_workers = platform.GetCoreNumAiv();
  if (layout.available_workers == 0) {
    return false;
  }
  const uint32_t worker_capacity = std::min({tokens, layout.available_workers, kAivWorkerSlotCapacity});
  layout.rows_per_worker = CeilDiv(tokens, worker_capacity);
  layout.active_workers = CeilDiv(tokens, layout.rows_per_worker);
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, layout.ub_bytes);
  layout.library_workspace = platform.GetLibApiWorkSpaceSize();
  return layout.ub_bytes > kUbReserveBytes;
}

bool QuerySharedTmpBytes(gert::TilingContext *context, uint32_t rows, uint32_t k_align, uint32_t &shared_tmp_bytes) {
  const AscendC::TensorShape reduce_shape(std::vector<int64_t>{rows, k_align});
  uint32_t reduce_max = 0;
  uint32_t reduce_min = 0;
  AscendC::GetReduceSumMaxMinTmpSize(reduce_shape, AscendC::TensorDataType::DT_FLOAT, AscendC::ReducePattern::AR, true,
                                     false, reduce_max, reduce_min);
  if (reduce_max < reduce_min) {
    return false;
  }

  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  const AscendC::TensorShape broadcast_source(std::vector<int64_t>{rows, 1});
  const AscendC::TensorShape broadcast_destination(std::vector<int64_t>{rows, k_align});
  uint32_t broadcast_max = 0;
  uint32_t broadcast_min = 0;
  AscendC::GetBroadCastMaxMinTmpSize(platform, broadcast_source, broadcast_destination, sizeof(float), false,
                                     broadcast_max, broadcast_min);
  if (broadcast_max < broadcast_min) {
    return false;
  }
  shared_tmp_bytes = static_cast<uint32_t>(std::max<uint64_t>(32, AlignUp(std::max(reduce_min, broadcast_min), 32)));
  return true;
}

uint64_t RequiredUbBytes(uint32_t rows, uint32_t experts_align, uint32_t k_align, uint32_t shared_tmp_bytes,
                         bool normalized) {
  const uint64_t k_binary = 3ULL * rows * k_align * sizeof(float);
  const uint64_t zero = static_cast<uint64_t>(rows) * experts_align * sizeof(float);
  if (!normalized) {
    return kUbReserveBytes + std::max(k_binary, zero);
  }
  const uint64_t row_buffer = static_cast<uint64_t>(rows) * kFp32Alignment * sizeof(float);
  const uint64_t broadcast = row_buffer + static_cast<uint64_t>(rows) * k_align * sizeof(float) + shared_tmp_bytes;
  const uint64_t reduce = static_cast<uint64_t>(rows) * k_align * sizeof(float) + row_buffer + shared_tmp_bytes;
  return kUbReserveBytes + std::max({k_binary, zero, broadcast, reduce});
}

bool SelectBatchRows(gert::TilingContext *context, const WorkerLayout &workers, uint32_t experts_align, uint32_t k,
                     uint32_t k_align, uint32_t &batch_rows, uint32_t &shared_tmp_bytes) {
  const bool normalized = k > 1;
  for (uint32_t candidate = workers.rows_per_worker; candidate > 0; --candidate) {
    uint32_t candidate_tmp = 32;
    if (normalized && !QuerySharedTmpBytes(context, candidate, k_align, candidate_tmp)) {
      return false;
    }
    if (RequiredUbBytes(candidate, experts_align, k_align, candidate_tmp, normalized) <= workers.ub_bytes) {
      batch_rows = candidate;
      shared_tmp_bytes = candidate_tmp;
      return true;
    }
  }
  return false;
}

ge::graphStatus TilingRouteGrad(gert::TilingContext *context) {
  OP_CHECK_NULL_WITH_CONTEXT(context, context);
  const auto *selected_shape = context->GetInputShape(0);
  const auto *denominator_shape = context->GetInputShape(1);
  const auto *grad_shape = context->GetInputShape(2);
  const auto *route_scores_shape = context->GetInputShape(3);
  const auto *indices_shape = context->GetInputShape(4);
  const auto *runtime_shape = context->GetInputShape(5);
  const auto *profile_shape = context->GetInputShape(6);
  OP_CHECK_NULL_WITH_CONTEXT(context, selected_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, denominator_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, grad_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, route_scores_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, indices_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, runtime_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, profile_shape);
  const auto *attrs = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const int64_t *top_k = attrs->GetAttrPointer<int64_t>(0);
  const float *scaling = attrs->GetAttrPointer<float>(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, top_k);
  OP_CHECK_NULL_WITH_CONTEXT(context, scaling);

  const auto &selected = selected_shape->GetStorageShape();
  const auto &denominator = denominator_shape->GetStorageShape();
  const auto &grad = grad_shape->GetStorageShape();
  const auto &scores = route_scores_shape->GetStorageShape();
  const auto &indices = indices_shape->GetStorageShape();
  if (selected.GetDimNum() != 2 || selected.GetDim(0) <= 0 || selected.GetDim(1) != *top_k || *top_k <= 0 ||
      scores.GetDimNum() != 2 || scores.GetDim(0) != selected.GetDim(0) || scores.GetDim(1) < *top_k ||
      !std::isfinite(*scaling) || !IsMatrix(denominator, selected.GetDim(0), 1) ||
      !IsMatrix(grad, selected.GetDim(0), *top_k) || !IsMatrix(indices, selected.GetDim(0), *top_k)) {
    OP_LOGE(context->GetNodeName(),
            "Invalid HyperMegaGateRouteGrad matrix metadata: selected storage/origin ranks=%zu/%zu, "
            "scores storage/origin ranks=%zu/%zu, top_k=%ld, scaling=%f.",
            selected_shape->GetStorageShape().GetDimNum(), selected_shape->GetOriginShape().GetDimNum(),
            route_scores_shape->GetStorageShape().GetDimNum(), route_scores_shape->GetOriginShape().GetDimNum(), *top_k,
            *scaling);
    return ge::GRAPH_FAILED;
  }
  if (selected.GetDim(0) > std::numeric_limits<uint32_t>::max() ||
      scores.GetDim(1) > std::numeric_limits<uint32_t>::max() || *top_k > std::numeric_limits<uint32_t>::max()) {
    return ge::GRAPH_FAILED;
  }

  const uint32_t tokens = static_cast<uint32_t>(selected.GetDim(0));
  const uint32_t experts = static_cast<uint32_t>(scores.GetDim(1));
  const uint32_t k = static_cast<uint32_t>(*top_k);
  const uint32_t expert_align = CeilDiv(experts, kFp32Alignment) * kFp32Alignment;
  const uint32_t k_align = CeilDiv(k, kFp32Alignment) * kFp32Alignment;
  WorkerLayout workers{};
  if (!GetWorkerLayout(context, tokens, workers)) {
    return ge::GRAPH_FAILED;
  }
  uint32_t batch_rows = 0;
  uint32_t shared_tmp_bytes = 0;
  if (!SelectBatchRows(context, workers, expert_align, k, k_align, batch_rows, shared_tmp_bytes)) {
    OP_LOGE(context->GetNodeName(), "No legal HyperMegaGateRouteGrad UB layout.");
    return ge::GRAPH_FAILED;
  }

  uint64_t workspace_bytes = 0;
  uint64_t offsets[9]{};
  if (k > 1) {
    const uint64_t k_tensor_bytes = static_cast<uint64_t>(tokens) * k * sizeof(float);
    for (uint32_t index = 0; index < 8; ++index) {
      if (!ReserveWorkspace(k_tensor_bytes, workspace_bytes, offsets[index])) {
        return ge::GRAPH_FAILED;
      }
    }
    if (!ReserveWorkspace(static_cast<uint64_t>(tokens) * sizeof(float), workspace_bytes, offsets[8])) {
      return ge::GRAPH_FAILED;
    }
  }

  HyperMegaGateRouteGradTilingData tiling;
  tiling.set_tokenCount(tokens);
  tiling.set_expertCount(experts);
  tiling.set_topK(k);
  tiling.set_routedScalingFactor(*scaling);
  tiling.set_stageCount(k == 1 ? kOneStageCount : kGreaterOneStageCount);
  tiling.set_activeAivWorkerCount(workers.active_workers);
  tiling.set_rowsPerWorker(workers.rows_per_worker);
  tiling.set_aivWorkerSlotCapacity(kAivWorkerSlotCapacity);
  tiling.set_batchRows(batch_rows);
  tiling.set_expertAlign(expert_align);
  tiling.set_kAlign(k_align);
  tiling.set_sharedTmpBytes(shared_tmp_bytes);
  tiling.set_runtimeConfigBytes(ElementCount(runtime_shape->GetStorageShape()));
  tiling.set_profileBufferBytes(ElementCount(profile_shape->GetStorageShape()));
  tiling.set_kSlot0Offset(offsets[0]);
  tiling.set_kSlot1Offset(offsets[1]);
  tiling.set_kSlot2Offset(offsets[2]);
  tiling.set_kSlot3Offset(offsets[3]);
  tiling.set_kSlot4Offset(offsets[4]);
  tiling.set_crossTermOffset(offsets[5]);
  tiling.set_directTermOffset(offsets[6]);
  tiling.set_broadcastRowSumOffset(offsets[7]);
  tiling.set_rowSumOffset(offsets[8]);
  auto *raw_tiling = context->GetRawTilingData();
  if (raw_tiling == nullptr) {
    return ge::GRAPH_FAILED;
  }
  tiling.SaveToBuffer(raw_tiling->GetData(), raw_tiling->GetCapacity());
  raw_tiling->SetDataSize(tiling.GetDataSize());
  context->SetBlockDim(workers.active_workers);
  context->SetTilingKey(0);
  size_t *workspace_sizes = context->GetWorkspaceSizes(1);
  if (workspace_sizes == nullptr || workspace_bytes > std::numeric_limits<size_t>::max() - workers.library_workspace) {
    return ge::GRAPH_FAILED;
  }
  workspace_sizes[0] = workers.library_workspace + workspace_bytes;
  OP_LOGD(context->GetNodeName(),
          "RouteGrad tokens=%u experts=%u topK=%u availableAiv=%u activeAiv=%u rows=%u batch=%u tmp=%u stages=%u",
          tokens, experts, k, workers.available_workers, workers.active_workers, workers.rows_per_worker, batch_rows,
          shared_tmp_bytes, k == 1 ? kOneStageCount : kGreaterOneStageCount);
  return ge::GRAPH_SUCCESS;
}

struct RouteGradCompileInfo {};

ge::graphStatus TilingPrepare(gert::TilingParseContext *context) {
  return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

}  // namespace

IMPL_OP_OPTILING(HyperMegaGateRouteGrad).Tiling(TilingRouteGrad).TilingParse<RouteGradCompileInfo>(TilingPrepare);

}  // namespace optiling
