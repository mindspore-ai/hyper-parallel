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

#include <array>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>

#include "test_support.h"

namespace hyper_parallel::multicore::shmem::tests {
namespace {

using runtime::ErrorCode;

constexpr std::string_view kTcpScheme = "tcp://";

void SetEnvironment(std::string_view name, const std::optional<std::string> &value) {
#if defined(_WIN32)
  _putenv_s(std::string(name).c_str(), value.has_value() ? value->c_str() : "");
#else
  if (value.has_value()) {
    setenv(std::string(name).c_str(), value->c_str(), 1);
  } else {
    unsetenv(std::string(name).c_str());
  }
#endif
}

class ConfigEnvironment final {
 public:
  ConfigEnvironment() {
    for (std::size_t index = 0; index < kNames.size(); ++index) {
      const char *value = std::getenv(std::string(kNames[index]).c_str());
      saved_[index] = value == nullptr ? std::nullopt : std::optional<std::string>(value);
      SetEnvironment(kNames[index], std::nullopt);
    }
  }

  ~ConfigEnvironment() {
    for (std::size_t index = 0; index < kNames.size(); ++index) {
      SetEnvironment(kNames[index], saved_[index]);
    }
  }

  void Set(std::string_view name, std::string value) { SetEnvironment(name, std::move(value)); }
  void Unset(std::string_view name) { SetEnvironment(name, std::nullopt); }

 private:
  static constexpr std::array<std::string_view, 4> kNames{runtime::kHeapSizeEnv, runtime::kTimeoutEnv,
                                                          runtime::kDataEngineEnv, runtime::kBootstrapEndpointEnv};
  std::array<std::optional<std::string>, kNames.size()> saved_{};
};

void CheckInvalid(ConfigEnvironment &environment, std::string_view name, std::string value) {
  const std::string context = "Invalid config (environment=" + std::string(name) + ", value='" + value + "')";
  environment.Set(name, std::move(value));
  const auto config = runtime::LoadConfigFromEnvironment();
  Check(!config.ok(), context + " must be rejected");
  CheckEq(config.error().error_code, ErrorCode::InvalidConfig, context + " error code");
  environment.Unset(name);
}

}  // namespace

void test_config_strict_unsigned_decimal() {
  ConfigEnvironment environment;

  environment.Set(runtime::kHeapSizeEnv, "18446744073709551615");
  auto config = runtime::LoadConfigFromEnvironment();
  Check(config.ok(), "The largest uint64 heap size must be accepted");
  CheckEq(config.value().heap_size_bytes, UINT64_MAX, "Largest accepted uint64 heap size");
  environment.Unset(runtime::kHeapSizeEnv);

  environment.Set(runtime::kTimeoutEnv, "4294967295");
  config = runtime::LoadConfigFromEnvironment();
  Check(config.ok(), "The largest uint32 timeout must be accepted");
  CheckEq(config.value().timeout_seconds, UINT32_MAX, "Largest accepted uint32 timeout");
  environment.Unset(runtime::kTimeoutEnv);

  constexpr std::array<std::string_view, 5> kInvalidValues{"0", "-1", "+1", " 1", "1s"};
  for (std::string_view value : kInvalidValues) {
    CheckInvalid(environment, runtime::kHeapSizeEnv, std::string(value));
    CheckInvalid(environment, runtime::kTimeoutEnv, std::string(value));
  }
#if !defined(_WIN32)
  // The Windows CRT removes an environment entry when assigned an empty string, so this case is Linux-only.
  CheckInvalid(environment, runtime::kHeapSizeEnv, "");
  CheckInvalid(environment, runtime::kTimeoutEnv, "");
#endif
  CheckInvalid(environment, runtime::kHeapSizeEnv, "18446744073709551616");
  CheckInvalid(environment, runtime::kTimeoutEnv, "4294967296");
}

void test_config_endpoint_minimal_and_passthrough() {
  ConfigEnvironment environment;
  const std::string original_endpoint = std::string(kTcpScheme) + "LOCALHOST:080";
  environment.Set(runtime::kBootstrapEndpointEnv, original_endpoint);
  auto config = runtime::LoadConfigFromEnvironment();
  Check(config.ok(), "A minimally valid endpoint must be accepted");
  CheckEq(config.value().bootstrap_endpoint_base, original_endpoint,
          "Endpoint validation must preserve the caller's original text");

  const std::string endpoint_at_limit =
    std::string(kTcpScheme) + std::string(runtime::kMaxBootstrapEndpointBytes - kTcpScheme.size(), 'a');
  environment.Set(runtime::kBootstrapEndpointEnv, endpoint_at_limit);
  config = runtime::LoadConfigFromEnvironment();
  Check(config.ok(), "An endpoint at the CANN ABI byte limit must be accepted");
  CheckEq(config.value().bootstrap_endpoint_base.size(), runtime::kMaxBootstrapEndpointBytes,
          "Accepted endpoint length at the CANN ABI limit");

  environment.Set(runtime::kBootstrapEndpointEnv, endpoint_at_limit + "a");
  config = runtime::LoadConfigFromEnvironment();
  Check(!config.ok(), "An endpoint beyond the CANN ABI byte limit must be rejected");
  CheckEq(config.error().error_code, ErrorCode::InvalidConfig, "Oversized endpoint error code");
}

void test_config_data_engine_validation() {
  ConfigEnvironment environment;

  environment.Set(runtime::kDataEngineEnv, "mte");
  const auto config = runtime::LoadConfigFromEnvironment();
  Check(config.ok(), "The explicit mte data engine must be accepted");
  CheckEq(config.value().data_engine, runtime::DataEngine::Mte, "Explicit data engine");
  environment.Unset(runtime::kDataEngineEnv);

  CheckInvalid(environment, runtime::kDataEngineEnv, "MTE");
  CheckInvalid(environment, runtime::kDataEngineEnv, "sdma");
}

}  // namespace hyper_parallel::multicore::shmem::tests
