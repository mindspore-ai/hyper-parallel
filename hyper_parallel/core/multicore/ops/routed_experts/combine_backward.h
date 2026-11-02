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

// Payloads are contiguous BF16: dy [N,H], unweighted values/dv [N*K,H],
// scores/ds [N,K]. INT64 order [N*K] maps each value row to token*K+slot.
// The host plan guarantees a bijection; the kernel checks each index range.
// dscore rounds each dy*value product to BF16 before its FP32 reduction.
extern "C" int RoutedExpertsCombineBackward(void *stream, void *dy, void *values, void *order, void *scores,
                                            void *dv, void *ds, int64_t tokens, int64_t selections,
                                            int32_t hidden, int32_t top_k, int32_t cores);
