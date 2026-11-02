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

#ifndef ASCENDC_CUBE_ONLY
#error "Routed expert gradients require ASCENDC_CUBE_ONLY"
#endif
#ifdef SPLIT_CORE_VEC
#error "Routed expert gradients must be compiled for cube cores"
#endif

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "acl/acl.h"
#include "gradient.h"

using namespace AscendC;
namespace routed = hyper_parallel::multicore::routed_experts;

namespace {

__aicore__ inline void Require(bool valid) {
  if (!valid) {
    Trap();
  }
}

__aicore__ inline void CheckHeader(const __gm__ routed::GradientConfigHeader *header,
                                  int32_t features, int32_t cols) {
  Require(header->magic == routed::kGradientConfigMagic && header->features == features && header->cols == cols &&
          header->tiling_bytes == sizeof(TCubeTiling) && header->max_tokens > 0 &&
          header->max_tokens <= routed::kGradientMaxTokens && header->tile_features > 0 &&
          header->tile_features <= routed::kGradientMaxTile && header->tile_cols > 0 &&
          header->tile_cols <= routed::kGradientMaxTile &&
          header->tile_features % routed::kGradientAlignment == 0 &&
          header->tile_cols % routed::kGradientAlignment == 0);
}

__aicore__ inline void CheckTask(const __gm__ int64_t *task, const __gm__ int64_t *boundaries,
                                const __gm__ routed::GradientConfigHeader *header, int64_t index,
                                int64_t input_rows, int32_t features, int32_t cols, int32_t experts,
                                int64_t feature_tiles, int64_t col_tiles) {
  const int64_t expert = index / (feature_tiles * col_tiles);
  const int64_t first_feature = (index / col_tiles % feature_tiles) * header->tile_features;
  const int64_t first_col = index % col_tiles * header->tile_cols;
  const int64_t rows =
    features - first_feature < header->tile_features ? features - first_feature : header->tile_features;
  const int64_t columns = cols - first_col < header->tile_cols ? cols - first_col : header->tile_cols;
  Require(expert >= 0 && expert < experts && task[0] == expert && task[1] == first_feature && task[2] == rows &&
          task[3] == first_col && task[4] == columns);
  const int64_t begin = boundaries[expert];
  const int64_t end = boundaries[expert + 1];
  Require(begin >= 0 && begin < end && end <= input_rows && end - begin <= header->max_tokens &&
          task[5] == begin && task[6] == end - begin);
}

__global__ __aicore__ void GradientKernel(GM_ADDR input, GM_ADDR grad_output, GM_ADDR boundaries, GM_ADDR tasks,
                                         GM_ADDR output, GM_ADDR config, int64_t input_rows, int32_t features,
                                         int32_t cols, int32_t experts, int64_t task_count) {
  const auto *header = reinterpret_cast<const __gm__ routed::GradientConfigHeader *>(config);
  CheckHeader(header, features, cols);
  const int64_t feature_tiles = (features + header->tile_features - 1) / header->tile_features;
  const int64_t col_tiles = (cols + header->tile_cols - 1) / header->tile_cols;
  Require(task_count == static_cast<int64_t>(experts) * feature_tiles * col_tiles);
  const auto *ends = reinterpret_cast<const __gm__ int64_t *>(boundaries);
  Require(ends[0] == 0 && ends[experts] == input_rows);
  if (GetBlockIdx() >= task_count) {
    return;
  }
  using A = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
  using B = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
  using C = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
  TPipe pipe;
  matmul::Matmul<A, B, C, C, CFG_MDL> mm;
  TCubeTiling tiling;
  auto *local = reinterpret_cast<int32_t *>(&tiling);
  const auto *global = reinterpret_cast<const __gm__ int32_t *>(config + sizeof(routed::GradientConfigHeader));
  for (int32_t index = 0; index < sizeof(TCubeTiling) / sizeof(int32_t); ++index) {
    local[index] = global[index];
  }
  REGIST_MATMUL_OBJ(&pipe, nullptr, mm, &tiling);
  const auto *descriptors = reinterpret_cast<const __gm__ int64_t *>(tasks);
  for (int64_t index = GetBlockIdx(); index < task_count; index += GetBlockNum()) {
    const auto *task = descriptors + index * routed::kGradientTaskFields;
    CheckTask(task, ends, header, index, input_rows, features, cols, experts, feature_tiles, col_tiles);
    GlobalTensor<bfloat16_t> x, dy;
    GlobalTensor<float> dw;
    x.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(input) + task[5] * features + task[1]);
    dy.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(grad_output) + task[5] * cols + task[3]);
    dw.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output) + (task[0] * features + task[1]) * cols + task[3]);
    // Transposed A retains the full feature stride; every output tile owns the entire token reduction.
    mm.SetOrgShape(features, cols, task[6]);
    mm.SetSingleShape(task[2], task[4], task[6]);
    mm.SetTensorA(x, true);
    mm.SetTensorB(dy, false);
    mm.IterateAll(dw, 0);
  }
  mm.End();
}

}  // namespace

extern "C" int RoutedExpertsGradient(void *stream, void *input, void *grad_output, void *boundaries, void *tasks,
                                     void *output, void *config, int64_t input_rows, int32_t features, int32_t cols,
                                     int32_t experts, int64_t task_count, int32_t cores) {
  const auto previous_error = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
  if (previous_error != ACL_SUCCESS) {
    return static_cast<int>(previous_error);
  }
  if (task_count == 0) {
    return 0;
  }
  const int32_t blocks = task_count < cores ? static_cast<int32_t>(task_count) : cores;
  GradientKernel<<<blocks, nullptr, stream>>>(static_cast<uint8_t *>(input), static_cast<uint8_t *>(grad_output),
    static_cast<uint8_t *>(boundaries), static_cast<uint8_t *>(tasks), static_cast<uint8_t *>(output),
    static_cast<uint8_t *>(config), input_rows, features, cols, experts, task_count);
  return static_cast<int>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
}
