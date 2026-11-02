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

#include <cstring>

#include "gradient.h"
#include "tiling/matmul/matmul_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace routed = hyper_parallel::multicore::routed_experts;

namespace {

inline bool InvalidGradientGeometry(int32_t features, int32_t cols, int32_t max_tokens,
                                    int32_t tile_features, int32_t tile_cols) {
  return features < 1 || features > routed::kGradientMaxWidth || cols < 1 || cols > routed::kGradientMaxWidth ||
    features % routed::kGradientAlignment != 0 || cols % routed::kGradientAlignment != 0 ||
    max_tokens < 1 || max_tokens > routed::kGradientMaxTokens || tile_features < 1 ||
    tile_features > routed::kGradientMaxTile || tile_cols < 1 || tile_cols > routed::kGradientMaxTile ||
    tile_features % routed::kGradientAlignment != 0 || tile_cols % routed::kGradientAlignment != 0;
}

inline int SetGradientBuilderInputs(matmul_tiling::MatmulApiTiling &builder, int32_t features, int32_t cols,
                                      int32_t max_tokens, int32_t tile_features, int32_t tile_cols) {
  const auto tile_m = tile_features < features ? tile_features : features;
  const auto tile_n = tile_cols < cols ? tile_cols : cols;
  return builder.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_BFLOAT16, true) != 0 ||
    builder.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_BFLOAT16, false) != 0 ||
    builder.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_FLOAT) != 0 ||
    builder.SetShape(tile_m, tile_n, max_tokens) != 0 ||
    builder.SetOrgShape(features, cols, max_tokens) != 0 || builder.SetBias(false) != 0 ||
    builder.SetBufferSpace(-1, -1, -1) != 0;
}

}  // namespace

extern "C" uint64_t RoutedExpertsGradientConfigBytes() {
  return sizeof(routed::GradientConfigHeader) + sizeof(AscendC::tiling::TCubeTiling);
}

extern "C" int RoutedExpertsGradientConfig(void *output, uint64_t capacity, int32_t features, int32_t cols,
                                           int32_t max_tokens, int32_t tile_features, int32_t tile_cols,
                                           uint64_t *written, uint64_t *workspace_bytes) {
  if (output == nullptr || written == nullptr || workspace_bytes == nullptr ||
      capacity < RoutedExpertsGradientConfigBytes() ||
      InvalidGradientGeometry(features, cols, max_tokens, tile_features, tile_cols)) {
    return -1;
  }
  *written = 0;
  *workspace_bytes = 0;
  auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  if (platform == nullptr) {
    return -2;
  }
  matmul_tiling::MatmulApiTiling builder(*platform);
  builder.SetMatmulConfigParams(1);
  if (SetGradientBuilderInputs(builder, features, cols, max_tokens, tile_features, tile_cols)) {
    return -3;
  }
  AscendC::tiling::TCubeTiling tiling{};
  if (builder.GetTiling(tiling) != 0) {
    return -4;
  }
  const routed::GradientConfigHeader header{routed::kGradientConfigMagic, features, cols, max_tokens,
                                             tile_features, tile_cols, sizeof(tiling)};
  std::memcpy(output, &header, sizeof(header));
  std::memcpy(static_cast<uint8_t *>(output) + sizeof(header), &tiling, sizeof(tiling));
  *written = RoutedExpertsGradientConfigBytes();
  return 0;
}
