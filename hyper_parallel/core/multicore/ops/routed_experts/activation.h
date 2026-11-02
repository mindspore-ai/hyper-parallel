// Copyright 2026 Huawei Technologies Co., Ltd
//
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

// BF16 packed gate/up [rows, 2 * inner] and output [rows, inner] in forward,
// [rows, 2 * inner] in backward. Positive limit applies FP32 gate upper and
// up symmetric clamps before SiLU; backward masks values outside each clamp.
extern "C" int RoutedExpertsClippedSwiGlu(void *stream, void *input, void *gradient, void *output,
                                           int64_t rows, int32_t inner, float limit, int32_t cores);
