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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "shmem.h"

namespace hyper_parallel::multicore::shmem::tests::fake_acl_cann {

struct State {
  int32_t tls_result{ACLSHMEM_SUCCESS};
  int32_t init_result{ACLSHMEM_SUCCESS};
  int32_t finalize_result{ACLSHMEM_SUCCESS};
  aclError get_device_result{ACL_SUCCESS};
  int32_t current_device_index{0};
  std::string soc_name{"Ascend910B1"};
  void *malloc_result{reinterpret_cast<void *>(0x10000U)};
  void *aligned_malloc_result{reinterpret_cast<void *>(0x20000U)};
  std::vector<std::string> calls;
  int32_t init_mode{0};
  aclshmemx_init_attr_t init_attributes{};
  std::size_t malloc_bytes{0};
  std::size_t aligned_bytes{0};
  std::size_t alignment_bytes{0};
  void *freed_address{nullptr};
  void *first_address{nullptr};
  void *second_address{nullptr};
  std::size_t transfer_bytes{0};
  int32_t target_pe{0};
  int32_t signal_operation{0};
  int32_t signal_comparison{0};
  int32_t signal_value{0};
  aclrtStream stream{nullptr};
};

State &Get();
void Reset();

}  // namespace hyper_parallel::multicore::shmem::tests::fake_acl_cann
