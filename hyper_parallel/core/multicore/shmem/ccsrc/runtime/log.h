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

#include "runtime/dfx.h"

namespace hyper_parallel::multicore::shmem::runtime::log {

inline constexpr const char *kLogLevelEnv = "HYPER_PARALLEL_SHMEM_LOG_LEVEL";

/** @brief Native Runtime screen-log verbosity threshold configured independently in each process. */
enum class Level : uint8_t {
  Debug = 0,
  Info = 1,
  Error = 2,
};

/** @brief Return the process log level parsed once from HYPER_PARALLEL_SHMEM_LOG_LEVEL: 0=Debug, 1=Info, 2=Error; unset or invalid falls back to Error. */
Level ActiveLevel() noexcept;

/**
 * @brief Write one best-effort Native Runtime line when level is enabled.
 *
 * @param level Severity required by this line.
 * @param root_rank CANN Root WORLD rank, or -1 before a trustworthy rank is available.
 * @param file Source filename captured at the call site.
 * @param line Source line captured at the call site.
 * @param format printf-compatible format followed by scalar arguments.
 */
#if defined(__GNUC__)
__attribute__((format(printf, 5, 6)))
#endif
void Line(Level level, int32_t root_rank, const char *file, int line, const char *format, ...) noexcept;

/** @brief Write one Runtime-owned failure without changing the supplied Status. */
void Failure(const char *file, int line, DfxOperation operation, DfxPhase phase, int32_t root_rank,
             const Status &error) noexcept;

}  // namespace hyper_parallel::multicore::shmem::runtime::log

#define HP_SM_LOG_INFO(root_rank, ...)                                                                              \
  do {                                                                                                              \
    if (static_cast<uint8_t>(::hyper_parallel::multicore::shmem::runtime::log::ActiveLevel()) <=                    \
        static_cast<uint8_t>(::hyper_parallel::multicore::shmem::runtime::log::Level::Info)) {                      \
      ::hyper_parallel::multicore::shmem::runtime::log::Line(                                                       \
        ::hyper_parallel::multicore::shmem::runtime::log::Level::Info, root_rank, __FILE__, __LINE__, __VA_ARGS__); \
    }                                                                                                               \
  } while (false)

#define HP_SM_LOG_DEBUG(root_rank, ...)                                                                              \
  do {                                                                                                               \
    if (static_cast<uint8_t>(::hyper_parallel::multicore::shmem::runtime::log::ActiveLevel()) <=                     \
        static_cast<uint8_t>(::hyper_parallel::multicore::shmem::runtime::log::Level::Debug)) {                      \
      ::hyper_parallel::multicore::shmem::runtime::log::Line(                                                        \
        ::hyper_parallel::multicore::shmem::runtime::log::Level::Debug, root_rank, __FILE__, __LINE__, __VA_ARGS__); \
    }                                                                                                                \
  } while (false)
