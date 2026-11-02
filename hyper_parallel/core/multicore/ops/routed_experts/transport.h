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

extern "C" int RoutedExpertsTransport(
  void *stream, void *input, void *output, void *mapping, void *send_edges, void *receive_edges,
  void *peer_offsets, void *ring, void *signals, void *progress, int32_t rank, int32_t ranks,
  int32_t lanes, int32_t block_rows, int32_t row_bytes, bool gather, uint64_t idle_cycles,
  int64_t input_rows, int64_t output_rows, int64_t mapping_rows, int64_t send_count, int64_t receive_count,
  void *input_ready, void *input_expected, void *output_ready, void *expert_slots,
  int32_t producers, int32_t active_experts, int32_t slot_count,
  void *headers, void *dedup_state, void *group_metadata, bool reduce, bool publish_each_block,
  float reduce_dense_ratio, float forward_compact_max_unique_ratio);

extern "C" uint64_t RoutedExpertsGemmConfigBytes();
extern "C" int RoutedExpertsGemmConfig(void *output, uint64_t capacity, int32_t m, int32_t n, int32_t k,
                                       bool transpose_b, uint64_t *written, uint64_t *workspace_bytes);
extern "C" int RoutedExpertsGemm(void *stream, void *input, void *weights, void *tasks, void *output,
                                 void *config, void *workspace, int32_t task_count, int32_t n, int32_t k,
                                 bool transpose_b, int32_t cores, void *input_ready, void *input_expected,
                                 void *output_ready, int32_t producers, int32_t active_experts,
                                 int64_t watchdog_cycles);
