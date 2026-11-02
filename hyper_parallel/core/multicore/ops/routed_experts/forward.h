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

#ifndef HYPER_PARALLEL_ROUTED_FORWARD_H_
#define HYPER_PARALLEL_ROUTED_FORWARD_H_
#include <cstdint>
extern "C" int RoutedExpertsForwardCube(void *stream, void *x, void *gate, void *down, void *tasks, void *z,
                              void *activation, void *y, void *config, void *mailboxes, int32_t rows,
                              int32_t experts, int32_t hidden, int32_t inner, int32_t task_count,
                              int32_t tile_rows, bool transpose, int32_t cores, int64_t watchdog,
                              void *input_ready, void *input_expected, void *output_ready, int32_t producers,
                              int32_t expected_columns);
extern "C" int RoutedExpertsForwardVector(void *stream, void *z, void *activation, void *tasks, void *mailboxes,
                                int32_t rows, int32_t experts, int32_t inner, int32_t task_count,
                                int32_t tile_rows, int32_t cores, int32_t vector_cores, int64_t watchdog);
extern "C" uint64_t RoutedExpertsForwardConfigBytes();
extern "C" int RoutedExpertsForwardConfig(void *output, uint64_t capacity, int32_t tile_rows, int32_t hidden,
                                bool transpose);
#endif
