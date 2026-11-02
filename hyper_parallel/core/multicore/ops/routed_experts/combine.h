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

// The caller validates contiguous BF16 payloads and INT64 inverse indices.
// A null scores address selects unweighted dx accumulation. Every output tile
// has one writer, and top-k slots are accumulated in ascending order.
extern "C" int RoutedExpertsCombine(void *stream, void *values, void *inverse_order, void *scores,
                                    void *output, int64_t tokens, int64_t selections, int32_t hidden,
                                    int32_t top_k, int32_t cores);
