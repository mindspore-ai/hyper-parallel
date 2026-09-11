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
#include <string>

#include "cann/host.h"

namespace hyper_parallel::multicore::shmem::tests::fake_host {

struct State {
  runtime::Status initialize_status;
  runtime::Status finalize_status;
  runtime::Status device_identity_status;
  runtime::Status current_device_status;
  runtime::Status allocation_status;
  runtime::Status aligned_allocation_status;
  DeviceIdentity device_identity{0, DeviceModel::kAscend910B, "Ascend910B1"};
  int32_t current_device_index{0};
  uintptr_t next_allocation_base{0x10000U};
  uint64_t last_allocate_bytes{0};
  uint64_t last_alignment_bytes{0};
  uintptr_t last_freed_address{0};
  runtime::StreamView last_barrier_stream{};
  uint32_t initialize_calls{0};
  uint32_t finalize_calls{0};
  uint32_t current_device_calls{0};
  uint32_t allocate_calls{0};
  uint32_t aligned_allocate_calls{0};
  uint32_t free_calls{0};
  uint32_t barrier_calls{0};
};

State &Get();
void Reset();

runtime::Status Error(runtime::ErrorCode error_code, std::string message, int32_t cann_error_code = 0);

}  // namespace hyper_parallel::multicore::shmem::tests::fake_host
