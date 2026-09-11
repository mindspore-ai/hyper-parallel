/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <vector>

namespace hyper_parallel::multicore::shmem::tests::fake_device {

enum class CallKind : uint8_t {
  Put,
  Get,
  Signal,
  Wait,
  Barrier,
};

struct Call {
  CallKind kind;
  void *first_address;
  void *second_address;
  uint32_t bytes;
  int32_t target_pe;
  int32_t operation;
  int32_t value;
};

void Reset();
const std::vector<Call> &Calls();
void SetObservedSignal(int32_t value);

}  // namespace hyper_parallel::multicore::shmem::tests::fake_device
