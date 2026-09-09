/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/config.h"

#include <charconv>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace hyper_parallel::multicore::shmem::runtime {
namespace {

constexpr std::string_view kTcpScheme = "tcp://";

struct ConfigInput {
  std::optional<std::string> heap_size;
  std::optional<std::string> timeout;
  std::optional<std::string> data_engine;
  std::optional<std::string> bootstrap_endpoint_base;
};

Status InvalidConfig(std::string message) { return Status{ErrorCode::InvalidConfig, std::nullopt, std::move(message)}; }

template <typename T>
Result<T> ParsePositiveUnsigned(std::string_view input, std::string_view field_name) {
  if (input.empty()) {
    return Result<T>::Failure(InvalidConfig(std::string(field_name) + " must not be empty"));
  }

  T value{};
  const char *begin = input.data();
  const char *end = begin + input.size();
  const auto [parsed_end, error] = std::from_chars(begin, end, value, 10);
  if (error != std::errc{} || parsed_end != end || value == 0) {
    return Result<T>::Failure(InvalidConfig(
      std::string(field_name) + " must be a positive unsigned decimal integer, but got '" + std::string(input) + "'"));
  }
  return Result<T>::Success(value);
}

bool ContainsWhitespace(std::string_view value) {
  for (const unsigned char character : value) {
    if (character == ' ' || character == '\t' || character == '\n' || character == '\r' || character == '\f' ||
        character == '\v') {
      return true;
    }
  }
  return false;
}

Result<std::string> ParseEndpoint(std::string_view endpoint) {
  if (endpoint.empty() || ContainsWhitespace(endpoint)) {
    return Result<std::string>::Failure(InvalidConfig(std::string(kBootstrapEndpointEnv) +
                                                      " must be non-empty and contain no whitespace, but got '" +
                                                      std::string(endpoint) + "'"));
  }
  if (endpoint.compare(0, kTcpScheme.size(), kTcpScheme) != 0) {
    return Result<std::string>::Failure(InvalidConfig(
      std::string(kBootstrapEndpointEnv) + " must use the tcp scheme, but got '" + std::string(endpoint) + "'"));
  }
  if (endpoint.size() > kMaxBootstrapEndpointBytes) {
    return Result<std::string>::Failure(InvalidConfig(std::string(kBootstrapEndpointEnv) + " must encode to at most " +
                                                      std::to_string(kMaxBootstrapEndpointBytes) +
                                                      " bytes, but got bytes=" + std::to_string(endpoint.size())));
  }
  return Result<std::string>::Success(std::string(endpoint));
}

std::optional<std::string> ReadEnvironment(std::string_view name) {
  const std::string owned_name(name);
  const char *value = std::getenv(owned_name.c_str());
  if (value == nullptr) {
    return std::nullopt;
  }
  return std::string(value);
}

Result<Config> ParseConfig(const ConfigInput &input) {
  Config config{};

  if (input.heap_size.has_value()) {
    auto heap_size = ParsePositiveUnsigned<uint64_t>(*input.heap_size, kHeapSizeEnv);
    if (!heap_size.ok()) {
      return Result<Config>::Failure(heap_size.error());
    }
    config.heap_size_bytes = heap_size.value();
  } else {
    config.heap_size_bytes = kDefaultHeapSizeBytes;
  }

  if (input.timeout.has_value()) {
    auto timeout = ParsePositiveUnsigned<uint32_t>(*input.timeout, kTimeoutEnv);
    if (!timeout.ok()) {
      return Result<Config>::Failure(timeout.error());
    }
    config.timeout_seconds = timeout.value();
  } else {
    config.timeout_seconds = kDefaultTimeoutSeconds;
  }

  const std::string_view data_engine =
    input.data_engine.has_value() ? std::string_view(*input.data_engine) : kDefaultDataEngine;
  if (data_engine != kDefaultDataEngine) {
    return Result<Config>::Failure(InvalidConfig(std::string(kDataEngineEnv) +
                                                 " only accepts the case-sensitive value 'mte', but got '" +
                                                 std::string(data_engine) + "'"));
  }
  config.data_engine = DataEngine::Mte;

  const std::string_view endpoint = input.bootstrap_endpoint_base.has_value()
                                      ? std::string_view(*input.bootstrap_endpoint_base)
                                      : kDefaultBootstrapEndpoint;
  auto parsed_endpoint = ParseEndpoint(endpoint);
  if (!parsed_endpoint.ok()) {
    return Result<Config>::Failure(parsed_endpoint.error());
  }
  config.bootstrap_endpoint_base = parsed_endpoint.value();

  return Result<Config>::Success(std::move(config));
}

}  // namespace

Result<Config> LoadConfigFromEnvironment() {
  ConfigInput input;
  input.heap_size = ReadEnvironment(kHeapSizeEnv);
  input.timeout = ReadEnvironment(kTimeoutEnv);
  input.data_engine = ReadEnvironment(kDataEngineEnv);
  input.bootstrap_endpoint_base = ReadEnvironment(kBootstrapEndpointEnv);
  return ParseConfig(input);
}

}  // namespace hyper_parallel::multicore::shmem::runtime
