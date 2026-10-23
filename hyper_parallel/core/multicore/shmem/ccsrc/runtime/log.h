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
#include <string_view>

#include "runtime/dfx.h"

namespace hyper_parallel::multicore::shmem::runtime::log {

/** @brief HP Runtime levels use the values accepted by the CANN SHMEM logger. */
enum class Level : int32_t {
  Debug = 0,
  Info = 1,
  Error = 3,
};

/** @brief Return whether one HP Runtime level is enabled by the process SHMEM_LOG_LEVEL configuration. */
bool is_enabled(Level level) noexcept;

/**
 * @brief Emit one best-effort Runtime line through the built-in CANN SHMEM log sink.
 *
 * @param level Severity required by this line.
 * @param root_rank CANN Root WORLD rank, or a negative value to omit the rank field when none is available.
 * @param file Source filename captured at the call site.
 * @param line Source line captured at the call site.
 * @param operation User-facing operation name.
 * @param stage User-facing stage within the operation.
 * @param format printf-compatible format followed by scalar arguments.
 */
#if defined(__GNUC__)
__attribute__((format(printf, 7, 8)))
#endif
void Line(Level level, int32_t root_rank, const char *file, int line, std::string_view operation,
          std::string_view stage, const char *format, ...) noexcept;

/** @brief Write one Runtime-owned failure without changing the supplied Status. */
void Failure(const char *file, int line, const DfxFailure &failure) noexcept;

}  // namespace hyper_parallel::multicore::shmem::runtime::log

#define HP_SM_LOG_INFO(root_rank, operation, stage, ...)                                                         \
  do {                                                                                                           \
    if (::hyper_parallel::multicore::shmem::runtime::log::is_enabled(                                            \
          ::hyper_parallel::multicore::shmem::runtime::log::Level::Info)) {                                      \
      ::hyper_parallel::multicore::shmem::runtime::log::Line(                                                    \
        ::hyper_parallel::multicore::shmem::runtime::log::Level::Info, root_rank, __FILE__, __LINE__, operation, \
        stage, __VA_ARGS__);                                                                                     \
    }                                                                                                            \
  } while (false)

#define HP_SM_LOG_DEBUG(root_rank, operation, stage, ...)                                                         \
  do {                                                                                                            \
    if (::hyper_parallel::multicore::shmem::runtime::log::is_enabled(                                             \
          ::hyper_parallel::multicore::shmem::runtime::log::Level::Debug)) {                                      \
      ::hyper_parallel::multicore::shmem::runtime::log::Line(                                                     \
        ::hyper_parallel::multicore::shmem::runtime::log::Level::Debug, root_rank, __FILE__, __LINE__, operation, \
        stage, __VA_ARGS__);                                                                                      \
    }                                                                                                             \
  } while (false)
