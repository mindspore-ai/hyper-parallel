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
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace hyper_parallel::multicore::shmem::runtime {

/** @brief Lifecycle states of the process-wide symmetric-memory Runtime. */
enum class State : uint8_t {
  Uninitialized,
  Ready,
  Finalized,
};

/** @brief Runtime error categories shared by the C++ core and framework bindings. */
enum class ErrorCode : uint8_t {
  Ok,
  InvalidState,
  InvalidConfig,
  InvalidArgument,
  UnsupportedCapability,
  DoubleFree,
  CannError,
  Timeout,
};

/**
 * @brief Status for operations without a value and the error branch of Result.
 *
 * A successful status uses ErrorCode::Ok, no CANN error code, and an empty message. cann_error_code is absent when
 * the failing API does not expose an original CANN return code.
 */
struct Status {
  ErrorCode error_code{ErrorCode::Ok};
  std::optional<int32_t> cann_error_code;
  std::string message;
};

/**
 * @brief Own either one successful value or one non-OK Status.
 *
 * @tparam T Successful value type.
 */
template <typename T>
class Result {
 public:
  /** @brief Construct a successful result that owns value. */
  static Result Success(T value) { return Result(std::move(value)); }

  /**
   * @brief Construct a failed result that owns error.
   *
   * @throws std::invalid_argument If error.error_code is ErrorCode::Ok.
   */
  static Result Failure(Status error) {
    if (error.error_code == ErrorCode::Ok) {
      throw std::invalid_argument("Result failure requires a non-OK status");
    }
    return Result(std::move(error));
  }

  /** @brief Return true when this result stores a successful value. */
  bool ok() const noexcept { return std::holds_alternative<T>(storage_); }

  /** @brief Return the successful value; throws std::bad_variant_access when ok() is false. */
  const T &value() const { return std::get<T>(storage_); }

  /** @brief Return the failure status; throws std::bad_variant_access when ok() is true. */
  const Status &error() const { return std::get<Status>(storage_); }

 private:
  explicit Result(T value) : storage_(std::move(value)) {}
  explicit Result(Status error) : storage_(std::move(error)) {}

  std::variant<T, Status> storage_;
};

/**
 * @brief Framework-independent Root WORLD identity supplied during Runtime initialization.
 *
 * root_rank is a CANN Root WORLD PE coordinate.
 */
struct RootWorldInfo {
  int32_t root_rank;
  int32_t root_size;
};

/**
 * @brief Non-owning view of a framework current NPU Stream.
 *
 * native_handle remains opaque outside the CANN Host Surface. device_index identifies the Stream's NPU.
 */
struct StreamView {
  uintptr_t native_handle;
  int32_t device_index;
};

}  // namespace hyper_parallel::multicore::shmem::runtime
