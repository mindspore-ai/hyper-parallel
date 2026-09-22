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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <algorithm>
#include <chrono>
#include <string_view>

namespace hyper_parallel::multicore::shmem::runtime::log {
namespace {

constexpr std::size_t kLogLineCapacity = 2048U;
constexpr std::size_t kTimestampCapacity = 32U;
constexpr int64_t kMillisecondsPerSecond = 1000;
constexpr std::string_view kTruncationMarker = "...";
constexpr std::string_view kFallbackTimestamp = "0000-00-00-00:00:00.000";

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

const char *LevelName(Level level) noexcept {
  switch (level) {
    case Level::Error:
      return "ERROR";
    case Level::Info:
      return "INFO";
    case Level::Debug:
      return "DEBUG";
  }
  return "ERROR";
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

const char *ErrorName(ErrorCode error_code) noexcept {
  switch (error_code) {
    case ErrorCode::Ok:
      return "OK";
    case ErrorCode::InvalidState:
      return "INVALID_STATE";
    case ErrorCode::InvalidConfig:
      return "INVALID_CONFIG";
    case ErrorCode::InvalidArgument:
      return "INVALID_ARGUMENT";
    case ErrorCode::UnsupportedCapability:
      return "UNSUPPORTED_CAPABILITY";
    case ErrorCode::DoubleFree:
      return "DOUBLE_FREE";
    case ErrorCode::CannError:
      return "CANN_ERROR";
    case ErrorCode::Timeout:
      return "TIMEOUT";
  }
  return "UNKNOWN";
}

void FormatTimestamp(char (&timestamp)[kTimestampCapacity]) noexcept {
  const auto now = std::chrono::system_clock::now();
  const std::time_t current_time = std::chrono::system_clock::to_time_t(now);
  std::tm local_time{};
#if defined(_WIN32)
  const bool converted = localtime_s(&local_time, &current_time) == 0;
#else
  const bool converted = localtime_r(&current_time, &local_time) != nullptr;
#endif
  if (!converted || std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d-%H:%M:%S", &local_time) == 0) {
    std::copy_n(kFallbackTimestamp.data(), kFallbackTimestamp.size(), timestamp);
    timestamp[kFallbackTimestamp.size()] = '\0';
    return;
  }

  const auto milliseconds =
    std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % kMillisecondsPerSecond;
  const std::size_t length = std::strlen(timestamp);
  timestamp[length] = '.';
  timestamp[length + 1U] = static_cast<char>('0' + milliseconds / 100);
  timestamp[length + 2U] = static_cast<char>('0' + milliseconds / 10 % 10);
  timestamp[length + 3U] = static_cast<char>('0' + milliseconds % 10);
  timestamp[length + 4U] = '\0';
}

void FinishLine(char (&line)[kLogLineCapacity], std::size_t length, bool truncated) noexcept {
  if (truncated && kTruncationMarker.size() + 2U < kLogLineCapacity) {
    length = kLogLineCapacity - kTruncationMarker.size() - 2U;
    std::copy_n(kTruncationMarker.data(), kTruncationMarker.size(), line + length);
    length += kTruncationMarker.size();
  }
  length = std::min(length, kLogLineCapacity - 2U);
  for (std::size_t index = 0; index < length; ++index) {
    if (line[index] == '\n' || line[index] == '\r') {
      line[index] = ' ';
    }
  }
  line[length++] = '\n';
  line[length] = '\0';
  static_cast<void>(std::fwrite(line, 1, length, stderr));
  static_cast<void>(std::fflush(stderr));
}

void WriteLine(const LineContext &context, const char *format, std::va_list arguments) noexcept {
  char timestamp[kTimestampCapacity]{};
  FormatTimestamp(timestamp);

  char output[kLogLineCapacity]{};
  FixedBufferWriter writer(output, sizeof(output));
  writer.Append("[HP-SHMEM][rank ");
  writer.AppendInteger(context.root_rank);
  writer.Append("][");
  writer.Append(LevelName(context.level));
  writer.Append("] ");
  writer.Append(timestamp);
  writer.Append(" [");
  writer.Append(context.file == nullptr ? "unknown" : context.file);
  writer.Append(":");
  writer.AppendInteger(context.line_number);
  writer.Append("] ");

  const std::size_t prefix_length = writer.length();
  const std::size_t remaining = sizeof(output) - prefix_length;
  const int message_size = std::vsnprintf(output + prefix_length, remaining, format, arguments);
  if (message_size < 0) {
    return;
  }

  const bool truncated = writer.truncated() || static_cast<std::size_t>(message_size) >= remaining;
  const std::size_t message_length = std::min(static_cast<std::size_t>(message_size), remaining - 1U);
  FinishLine(output, std::min(prefix_length + message_length, sizeof(output) - 1U), truncated);
}

Level ParseActiveLevel() noexcept {
  const char *value = std::getenv(kLogLevelEnv);
  if (value == nullptr || value[0] == '\0') {
    return Level::Error;
  }
  if (std::strcmp(value, "0") == 0) {
    return Level::Debug;
  }
  if (std::strcmp(value, "1") == 0) {
    return Level::Info;
  }
  if (std::strcmp(value, "2") == 0) {
    return Level::Error;
  }

  char timestamp[kTimestampCapacity]{};
  FormatTimestamp(timestamp);
  char output[kLogLineCapacity]{};
  FixedBufferWriter writer(output, sizeof(output));
  writer.Append("[HP-SHMEM][rank -1][ERROR] ");
  writer.Append(timestamp);
  writer.Append(" [");
  writer.Append(Basename(__FILE__));
  writer.Append(":");
  writer.AppendInteger(__LINE__);
  writer.Append("] invalid ");
  writer.Append(kLogLevelEnv);
  writer.Append("=\"");
  writer.Append(value);
  writer.Append("\", using ERROR");
  FinishLine(output, writer.length(), writer.truncated());
  return Level::Error;
}

}  // namespace

Level ActiveLevel() noexcept {
  static const Level active_level = ParseActiveLevel();
  return active_level;
}

void Line(Level level, int32_t root_rank, const char *file, int line, const char *format, ...) noexcept {
  // ActiveLevel() is the lower threshold: smaller values are more verbose, so a line is emitted only when
  // level >= ActiveLevel() (0=Debug emits everything, 2=Error emits Error only).
  if (static_cast<uint8_t>(level) < static_cast<uint8_t>(ActiveLevel())) {
    return;
  }
  std::va_list arguments;
  va_start(arguments, format);
  WriteLine(LineContext{level, root_rank, Basename(file), line}, format, arguments);
  va_end(arguments);
}

void Failure(const char *file, int line, const DfxFailure &failure) noexcept {
  const Status &error = failure.status;
  const int message_length = static_cast<int>(std::min<std::size_t>(error.message.size(), INT_MAX));
  if (error.cann_error_code.has_value()) {
    Line(Level::Error, failure.root_rank, file, line, "op=%s phase=%s err=%s cann=%d msg=\"%.*s\"",
         OperationName(failure.operation), PhaseName(failure.phase), ErrorName(error.error_code),
         *error.cann_error_code, message_length, error.message.data());
    return;
  }
  Line(Level::Error, failure.root_rank, file, line, "op=%s phase=%s err=%s cann=None msg=\"%.*s\"",
       OperationName(failure.operation), PhaseName(failure.phase),
       ErrorName(error.error_code), message_length, error.message.data());
}

}  // namespace hyper_parallel::multicore::shmem::runtime::log
