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

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string_view>

namespace hyper_parallel::multicore::shmem::cann::log {

constexpr std::size_t kErrorDetailCapacity = 1024U;

/**
 * @brief Capture the first CANN ERROR emitted during one synchronous Adapter call.
 *
 * The capture owns the process-wide synchronous call window from construction through the first finish(). A nested
 * capture on the same thread fails before acquiring the call-window lock. Normal CANN logs are emitted through
 * the built-in SHMEM log sink while capture is active. The returned detail is best-effort because CANN background logs
 * do not carry a call identifier. At most the first 1024 characters of the first ERROR are retained; line breaks and
 * double quotes are sanitized for single-line output.
 */
class ErrorCapture {
 public:
  ErrorCapture();
  ~ErrorCapture() noexcept;

  ErrorCapture(const ErrorCapture &) = delete;
  ErrorCapture &operator=(const ErrorCapture &) = delete;

  /**
   * @brief Stop capturing and return the sanitized first ERROR, or an empty string when none was emitted.
   *
   * Repeated calls return the result saved by the first call without changing process-wide capture state. The returned
   * view remains valid until this ErrorCapture is destroyed.
   */
  std::string_view finish() noexcept;

 private:
  std::unique_lock<std::mutex> call_lock_;
  std::array<char, kErrorDetailCapacity + 1U> detail_{};
  std::size_t detail_size_{0U};
  bool finished_{false};
};

/**
 * @brief Install the process-wide CANN SHMEM logger used for error capture and built-in-sink relay.
 *
 * CANN external logger registration is first-wins and exposes no ownership query, so successful return does not
 * guarantee this Adapter owns the logger. The callback captures the first ERROR in an active synchronous call and
 * forwards every original CANN line to the CANN SHMEM built-in sink. Error detail capture is therefore best-effort.
 *
 * @return The original ACLSHMEM status code from aclshmemx_set_extern_logger().
 */
int32_t configure() noexcept;

/**
 * @brief Emit one message through the built-in CANN SHMEM log sink.
 *
 * The HP Adapter accesses the private hpshmem_log symbol only through this function. The first call initializes the
 * native SHMEM log level so the sink consistently honors SHMEM_LOG_LEVEL.
 */
int32_t emit(int32_t level, const char *message) noexcept;

}  // namespace hyper_parallel::multicore::shmem::cann::log
