/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cann/log.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <stdexcept>

#include "host/utils/hpshmem_log.h"
#include "shmem.h"

namespace hyper_parallel::multicore::shmem::cann::log {
namespace {

constexpr int32_t kErrorLogLevel = 3;

int32_t configure_log_level() noexcept {
  static const int32_t status = aclshmemx_set_log_level(kErrorLogLevel);
  return status;
}

struct CaptureState {
  std::mutex mutex;
  bool active{false};
  std::array<char, kErrorDetailCapacity + 1U> detail{};
  std::size_t detail_size{0U};
};

CaptureState &state() {
  static CaptureState state;
  return state;
}

std::mutex &call_mutex() {
  static std::mutex mutex;
  return mutex;
}

bool &capture_active_on_current_thread() noexcept {
  static thread_local bool active = false;
  return active;
}

char sanitize_character(char character) noexcept {
  if (character == '\r' || character == '\n') {
    return ' ';
  }
  return character == '"' ? '\'' : character;
}

void forward(int32_t level, const char *message) noexcept {
  if (message == nullptr) {
    return;
  }
  if (level >= kErrorLogLevel) {
    try {
      auto &capture_state = state();
      std::lock_guard<std::mutex> lock(capture_state.mutex);
      if (capture_state.active && capture_state.detail_size == 0U) {
        capture_state.detail_size = std::min(std::strlen(message), kErrorDetailCapacity);
        std::copy_n(message, capture_state.detail_size, capture_state.detail.data());
        capture_state.detail[capture_state.detail_size] = '\0';
      }
    } catch (...) {
      // Diagnostics must never change the outcome of the underlying CANN call.
    }
  }
  static_cast<void>(emit(level, message));
}

}  // namespace

ErrorCapture::ErrorCapture() {
  if (capture_active_on_current_thread()) {
    throw std::logic_error("nested CANN error capture on the same thread is unsupported");
  }
  call_lock_ = std::unique_lock<std::mutex>(call_mutex());
  auto &capture_state = state();
  std::lock_guard<std::mutex> lock(capture_state.mutex);
  capture_state.active = true;
  capture_state.detail_size = 0U;
  capture_state.detail[0] = '\0';
  capture_active_on_current_thread() = true;
}

ErrorCapture::~ErrorCapture() noexcept {
  if (!finished_) {
    static_cast<void>(finish());
  }
}

std::string_view ErrorCapture::finish() noexcept {
  if (finished_) {
    return std::string_view(detail_.data(), detail_size_);
  }
  try {
    auto &capture_state = state();
    std::lock_guard<std::mutex> lock(capture_state.mutex);
    capture_state.active = false;
    detail_size_ = capture_state.detail_size;
    for (std::size_t index = 0U; index < detail_size_; ++index) {
      detail_[index] = sanitize_character(capture_state.detail[index]);
    }
    detail_[detail_size_] = '\0';
  } catch (...) {
    detail_size_ = 0U;
    detail_[0] = '\0';
  }
  capture_active_on_current_thread() = false;
  try {
    if (call_lock_.owns_lock()) {
      call_lock_.unlock();
    }
  } catch (...) {
    detail_size_ = 0U;
    detail_[0] = '\0';
  }
  finished_ = true;
  return std::string_view(detail_.data(), detail_size_);
}

int32_t configure() noexcept {
  const int32_t status = configure_log_level();
  return status == ACLSHMEM_SUCCESS ? aclshmemx_set_extern_logger(forward) : status;
}

int32_t emit(int32_t level, const char *message) noexcept {
  const int32_t status = configure_log_level();
  return status == ACLSHMEM_SUCCESS ? hpshmem_log(level, message) : status;
}

}  // namespace hyper_parallel::multicore::shmem::cann::log
