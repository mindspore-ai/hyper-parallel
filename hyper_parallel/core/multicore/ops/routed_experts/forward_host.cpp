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

#include <cstring>
#include "forward.h"
#include "tiling/matmul/matmul_tiling.h"
#include "tiling/platform/platform_ascendc.h"
extern "C" uint64_t RoutedExpertsForwardConfigBytes() {
  return sizeof(AscendC::tiling::TCubeTiling);
}
extern "C" int RoutedExpertsForwardConfig(void *output, uint64_t capacity, int32_t tile_rows, int32_t hidden,
                                bool transpose) {
  if (output == nullptr || capacity != RoutedExpertsForwardConfigBytes() || tile_rows != 128 || hidden != 2048) {
    return -1;
  }
  auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  if (platform == nullptr) {
    return -2;
  }
  matmul_tiling::MatmulApiTiling builder(*platform);
  builder.SetMatmulConfigParams(1);
  builder.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_BFLOAT16, false);
  builder.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_BFLOAT16, transpose);
  builder.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_BFLOAT16);
  builder.SetShape(tile_rows, 512, hidden);
  builder.SetOrgShape(tile_rows, 512, hidden);
  builder.SetBias(false);
  builder.SetBufferSpace(-1, -1, -1);
  if (builder.SetFixSplit(128, 256, 64) != 0) {
    return -3;
  }
  builder.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTM);
  AscendC::tiling::TCubeTiling tiling;
  if (builder.GetTiling(tiling) != 0) {
    return -4;
  }
  tiling.stepM = 1;
  tiling.stepN = 1;
  tiling.stepKa = 4;
  tiling.stepKb = 4;
  tiling.depthA1 = 8;
  tiling.depthB1 = 8;
  tiling.dbL0C = 1;
  tiling.shareL1Size = 393216;
  tiling.shareL0CSize = 131072;
  std::memcpy(output, &tiling, sizeof(tiling));
  return 0;
}
