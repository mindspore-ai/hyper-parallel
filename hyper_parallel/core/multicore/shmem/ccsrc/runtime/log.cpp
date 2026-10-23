/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/log.h"

#include <charconv>
#include <climits>
#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <algorithm>
#include <string_view>

#include "cann/log.h"

namespace hyper_parallel::multicore::shmem::runtime::log {
namespace {

constexpr std::size_t kLogLineCapacity = 2048U;
constexpr std::string_view kTruncationMarker = "...";
constexpr int32_t kDefaultLogLevel = 3;

int32_t configured_log_level() noexcept {
  static const int32_t configured_level = []() noexcept {
    const char *value = std::getenv("SHMEM_LOG_LEVEL");
    if (value == nullptr) {
      return kDefaultLogLevel;
    }
    if (std::strcmp(value, "DEBUG") == 0) {
      return 0;
    }
    if (std::strcmp(value, "INFO") == 0) {
      return 1;
    }
    if (std::strcmp(value, "WARN") == 0) {
      return 2;
    }
    if (std::strcmp(value, "ERROR") == 0) {
      return 3;
    }
    if (std::strcmp(value, "FATAL") == 0) {
      return 4;
    }
    return kDefaultLogLevel;
  }();
  return configured_level;
}

class FixedBufferWriter {
 public:
  FixedBufferWriter(char *buffer, std::size_t capacity) noexcept : buffer_(buffer), capacity_(capacity) {}

  void Append(std::string_view value) noexcept {
    if (capacity_ == 0U) {
      truncated_ = truncated_ || !value.empty();
      return;
    }
    const std::size_t available = capacity_ - 1U - length_;
    const std::size_t copied = std::min(value.size(), available);
    std::copy_n(value.data(), copied, buffer_ + length_);
    length_ += copied;
    buffer_[length_] = '\0';
    truncated_ = truncated_ || copied != value.size();
  }

  template <typename Integer>
  void AppendInteger(Integer value) noexcept {
    char text[32]{};
    const auto result = std::to_chars(text, text + sizeof(text), value);
    if (result.ec != std::errc{}) {
      truncated_ = true;
      return;
    }
    Append(std::string_view(text, static_cast<std::size_t>(result.ptr - text)));
  }

  std::size_t length() const noexcept { return length_; }

  bool truncated() const noexcept { return truncated_; }

 private:
  char *buffer_;
  std::size_t capacity_;
  std::size_t length_ = 0U;
  bool truncated_ = false;
};

struct LineContext {
  Level level;
  int32_t root_rank;
  const char *file;
  int line_number;
  std::string_view operation;
  std::string_view stage;
};

const char *Basename(const char *file) noexcept {
  if (file == nullptr) {
    return "unknown";
  }
  const char *slash = std::strrchr(file, '/');
  const char *backslash = std::strrchr(file, '\\');
  const char *separator = slash;
  if (separator == nullptr || (backslash != nullptr && backslash > separator)) {
    separator = backslash;
  }
  return separator == nullptr ? file : separator + 1;
}

const char *OperationName(DfxOperation operation) noexcept {
  switch (operation) {
    case DfxOperation::Initialize:
      return "Initialize";
    case DfxOperation::Allocate:
      return "Allocate";
    case DfxOperation::Free:
      return "Free";
    case DfxOperation::Barrier:
      return "Barrier";
    case DfxOperation::Shutdown:
      return "Shutdown";
  }
  return "Unknown";
}

const char *PhaseName(DfxPhase phase) noexcept {
  switch (phase) {
    case DfxPhase::Validation:
      return "Validation";
    case DfxPhase::CannCall:
      return "CannCall";
  }
  return "Unknown";
}

void finish_message(char (&message)[kLogLineCapacity], std::size_t length, bool truncated) noexcept {
  if (truncated && kTruncationMarker.size() + 1U < kLogLineCapacity) {
    length = kLogLineCapacity - kTruncationMarker.size() - 1U;
    std::copy_n(kTruncationMarker.data(), kTruncationMarker.size(), message + length);
    length += kTruncationMarker.size();
  }
  length = std::min(length, kLogLineCapacity - 1U);
  for (std::size_t index = 0; index < length; ++index) {
    if (message[index] == '\n' || message[index] == '\r') {
      message[index] = ' ';
    }
  }
  message[length] = '\0';
}

void WriteLine(const LineContext &context, const char *format, std::va_list arguments) noexcept {
  char output[kLogLineCapacity]{};
  FixedBufferWriter writer(output, sizeof(output));
  writer.Append("[HP_SHMEM ");
  writer.Append(context.file == nullptr ? "unknown" : context.file);
  if (context.line_number > 0) {
    writer.Append(":");
    writer.AppendInteger(context.line_number);
  }
  writer.Append("]");
  if (context.root_rank >= 0) {
    writer.Append(" [rank ");
    writer.AppendInteger(context.root_rank);
    writer.Append("]");
  }
  writer.Append(" [");
  writer.Append(context.operation);
  writer.Append("|");
  writer.Append(context.stage);
  writer.Append("] ");

  const std::size_t prefix_length = writer.length();
  const std::size_t remaining = sizeof(output) - prefix_length;
  const int message_size = std::vsnprintf(output + prefix_length, remaining, format, arguments);
  if (message_size < 0) {
    return;
  }

  const bool truncated = writer.truncated() || static_cast<std::size_t>(message_size) >= remaining;
  const std::size_t message_length = std::min(static_cast<std::size_t>(message_size), remaining - 1U);
  finish_message(output, std::min(prefix_length + message_length, sizeof(output) - 1U), truncated);
  static_cast<void>(cann::log::emit(static_cast<int32_t>(context.level), output));
}

}  // namespace

bool is_enabled(Level level) noexcept { return static_cast<int32_t>(level) >= configured_log_level(); }

void Line(Level level, int32_t root_rank, const char *file, int line, std::string_view operation,
          std::string_view stage, const char *format, ...) noexcept {
  if (!is_enabled(level)) {
    return;
  }
  std::va_list arguments;
  va_start(arguments, format);
  WriteLine(LineContext{level, root_rank, Basename(file), line, operation, stage}, format, arguments);
  va_end(arguments);
}

void Failure(const char *file, int line, const DfxFailure &failure) noexcept {
  const Status &error = failure.status;
  const std::string_view message =
    error.log_message.has_value() ? std::string_view(*error.log_message) : std::string_view(error.message);
  const int message_length = static_cast<int>(std::min<std::size_t>(message.size(), INT_MAX));
  Line(Level::Error, failure.root_rank, file, line, OperationName(failure.operation), PhaseName(failure.phase), "%.*s",
       message_length, message.data());
}

}  // namespace hyper_parallel::multicore::shmem::runtime::log
