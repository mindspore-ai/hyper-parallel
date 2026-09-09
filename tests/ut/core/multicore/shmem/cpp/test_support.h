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
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace hyper_parallel::multicore::shmem::tests {

inline void Check(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

namespace detail {

template <typename Value>
std::string FormatValue(const Value &value) {
  std::ostringstream stream;
  if constexpr (std::is_enum_v<Value>) {
    using Underlying = std::underlying_type_t<Value>;
    if constexpr (std::is_signed_v<Underlying>) {
      stream << static_cast<intmax_t>(value);
    } else {
      stream << static_cast<uintmax_t>(value);
    }
  } else {
    stream << value;
  }
  return stream.str();
}

}  // namespace detail

template <typename Actual, typename Expected>
void CheckEq(const Actual &actual, const Expected &expected, std::string_view context) {
  if (!(actual == expected)) {
    throw std::runtime_error(std::string(context) + ": expected=" + detail::FormatValue(expected) +
                             ", actual=" + detail::FormatValue(actual));
  }
}

}  // namespace hyper_parallel::multicore::shmem::tests
