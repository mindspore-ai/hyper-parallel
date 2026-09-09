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
#include <string_view>

#include "runtime/types.h"

namespace hyper_parallel::multicore::shmem::runtime {

inline constexpr std::string_view kHeapSizeEnv = "HYPER_PARALLEL_SHMEM_HEAP_SIZE";
inline constexpr std::string_view kTimeoutEnv = "HYPER_PARALLEL_SHMEM_TIMEOUT_SEC";
inline constexpr std::string_view kDataEngineEnv = "HYPER_PARALLEL_SHMEM_DATA_ENGINE";
inline constexpr std::string_view kBootstrapEndpointEnv = "HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT";

inline constexpr uint64_t kDefaultHeapSizeBytes = 1073741824ULL;
inline constexpr uint32_t kDefaultTimeoutSeconds = 120U;
inline constexpr std::string_view kDefaultDataEngine = "mte";
inline constexpr std::string_view kDefaultBootstrapEndpoint = "tcp://127.0.0.1:8662";
inline constexpr std::size_t kMaxBootstrapEndpointBytes = 63U;

/** @brief Data engines accepted by the first Runtime release. */
enum class DataEngine : uint8_t {
  Mte,
};

/** @brief Validated, framework-independent Runtime configuration. */
struct Config {
  uint64_t heap_size_bytes;
  uint32_t timeout_seconds;
  DataEngine data_engine;
  std::string bootstrap_endpoint_base;
};

/**
 * @brief Load and validate the four HYPER_PARALLEL_SHMEM_* Runtime variables.
 *
 * @return A normalized Config, or ErrorCode::InvalidConfig for an invalid environment value.
 */
Result<Config> LoadConfigFromEnvironment();

}  // namespace hyper_parallel::multicore::shmem::runtime
