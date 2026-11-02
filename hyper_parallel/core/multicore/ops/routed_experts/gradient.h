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
#pragma once

#include <cstdint>

namespace hyper_parallel::multicore::routed_experts {

inline constexpr uint64_t kGradientConfigMagic = 0x4850524752414431ULL;
inline constexpr int32_t kGradientMaxWidth = 32768;
inline constexpr int32_t kGradientMaxTokens = 1 << 20;
inline constexpr int32_t kGradientMaxTile = 1024;
inline constexpr int32_t kGradientAlignment = 16;
inline constexpr int32_t kGradientTaskFields = 7;

// A versioned prefix permits device validation without reading the config back to the host.
struct GradientConfigHeader {
  uint64_t magic;
  int32_t features;
  int32_t cols;
  int32_t max_tokens;
  int32_t tile_features;
  int32_t tile_cols;
  uint32_t tiling_bytes;
};
static_assert(sizeof(GradientConfigHeader) == 32);

}  // namespace hyper_parallel::multicore::routed_experts

/** Return the exact config extent shared by the host tiler and device kernel. */
extern "C" uint64_t RoutedExpertsGradientConfigBytes();

/** Build a cube-only BF16-transpose-A/FP32-output tiling with zero GM workspace. */
extern "C" int RoutedExpertsGradientConfig(void *output, uint64_t capacity, int32_t features, int32_t cols,
                                           int32_t max_tokens, int32_t tile_features, int32_t tile_cols,
                                           uint64_t *written, uint64_t *workspace_bytes);

/**
 * Compute compact expert gradients into FP32 [experts, features, cols].
 *
 * Boundaries are int64 [experts + 1] covering the packed input rows. Each seven-field
 * task is [expert, first_feature, features, first_col, cols, first_token, tokens].
 * Expert is the compact active-expert index. Tasks must cover the canonical grid in
 * expert/feature/column order, with each expert's entire token span in every task.
 * The device checks this contract before each write; no atomic accumulation is used.
 */
extern "C" int RoutedExpertsGradient(void *stream, void *input, void *grad_output, void *boundaries, void *tasks,
                                     void *output, void *config, int64_t input_rows, int32_t features, int32_t cols,
                                     int32_t experts, int64_t task_count, int32_t cores);
