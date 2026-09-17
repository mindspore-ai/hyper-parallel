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

namespace hyper_parallel::multicore::shmem {

/** @brief Framework-independent NPU models recognized by the symmetric-memory component. */
enum class DeviceModel : uint8_t {
  kAscend910B,
  kAscend910C,
  kAscend950,
};

/** @brief Current ACL device and its owned SoC identity observed during Runtime initialization. */
struct DeviceIdentity {
  int32_t device_index;
  DeviceModel model;
  std::string soc_name;
};

}  // namespace hyper_parallel::multicore::shmem
