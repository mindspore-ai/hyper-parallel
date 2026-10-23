/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include "cann/log.h"
#include "host/utils/hpshmem_log.h"
#include "runtime/log.h"

namespace {

using CannLogger = void (*)(int, const char *);
using hyper_parallel::multicore::shmem::cann::log::ErrorCapture;
using hyper_parallel::multicore::shmem::cann::log::kErrorDetailCapacity;
using hyper_parallel::multicore::shmem::runtime::log::Level;

CannLogger g_cann_logger = nullptr;
std::atomic<int> g_sink_calls{0};

void check(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_capture_state() {
  check(hyper_parallel::multicore::shmem::cann::log::configure() == 0, "logger configuration failed");
  check(g_cann_logger != nullptr, "logger callback was not registered");

  {
    ErrorCapture capture;
    g_cann_logger(1, "ignored info");
    check(capture.finish().empty(), "INFO unexpectedly populated ERROR detail");
    check(capture.finish().empty(), "repeated empty finish changed the result");
  }

  {
    ErrorCapture capture;
    g_cann_logger(3, "first\nerror\rwith \"quotes\"");
    g_cann_logger(3, "second error");
    const std::string first(capture.finish());
    check(first == "first error with 'quotes'", "first ERROR was not retained and sanitized");
    check(capture.finish() == first, "repeated finish changed the captured detail");
  }

  {
    std::string oversized(kErrorDetailCapacity + 64U, 'x');
    oversized[0] = '\n';
    oversized[1] = '"';
    ErrorCapture capture;
    g_cann_logger(3, oversized.c_str());
    const std::string detail(capture.finish());
    check(detail.size() == kErrorDetailCapacity, "captured detail was not truncated at the fixed capacity");
    check(detail[0] == ' ' && detail[1] == '\'', "truncated detail was not sanitized");
  }

  {
    ErrorCapture capture;
    bool nested_rejected = false;
    try {
      ErrorCapture nested;
    } catch (const std::logic_error &) {
      nested_rejected = true;
    }
    check(nested_rejected, "same-thread nested capture was not rejected");
    static_cast<void>(capture.finish());
  }

  {
    ErrorCapture capture;
    g_cann_logger(3, "discarded by destructor");
  }
  {
    ErrorCapture capture;
    g_cann_logger(3, "after destructor");
    check(capture.finish() == "after destructor", "destructor did not release capture state");
  }

  {
    std::promise<void> owner_ready;
    std::promise<void> release_owner;
    std::promise<void> handoff_started;
    std::promise<void> acquired;
    std::atomic<bool> thread_failed{false};
    auto owner_ready_future = owner_ready.get_future();
    auto release_owner_future = release_owner.get_future();
    auto handoff_started_future = handoff_started.get_future();
    auto acquired_future = acquired.get_future();
    std::thread owner([&]() {
      ErrorCapture first;
      owner_ready.set_value();
      release_owner_future.wait();
      static_cast<void>(first.finish());
    });
    std::thread contender([&]() {
      owner_ready_future.wait();
      handoff_started.set_value();
      try {
        ErrorCapture second;
        acquired.set_value();
        static_cast<void>(second.finish());
      } catch (...) {
        thread_failed = true;
        acquired.set_value();
      }
    });
    handoff_started_future.wait();
    release_owner.set_value();
    owner.join();
    acquired_future.wait();
    contender.join();
    check(!thread_failed, "cross-thread capture handoff did not complete");
  }
}

void test_configured_level(const std::string &configuration) {
  const bool debug_enabled = configuration == "DEBUG";
  const bool info_enabled = debug_enabled || configuration == "INFO";
  const bool error_enabled = configuration != "FATAL";
  int debug_rank_evaluations = 0;
  int debug_argument_evaluations = 0;
  int info_rank_evaluations = 0;
  int info_argument_evaluations = 0;
  g_sink_calls = 0;

  HP_SM_LOG_DEBUG(++debug_rank_evaluations, "Test", "Filter", "%d", ++debug_argument_evaluations);
  HP_SM_LOG_INFO(++info_rank_evaluations, "Test", "Filter", "%d", ++info_argument_evaluations);
  hyper_parallel::multicore::shmem::runtime::log::Line(Level::Error, 0, __FILE__, __LINE__, "Test", "Filter", "error");

  check((debug_rank_evaluations == 1) == debug_enabled, "DEBUG rank expression filtering is incorrect");
  check((debug_argument_evaluations == 1) == debug_enabled, "DEBUG argument filtering is incorrect");
  check((info_rank_evaluations == 1) == info_enabled, "INFO rank expression filtering is incorrect");
  check((info_argument_evaluations == 1) == info_enabled, "INFO argument filtering is incorrect");
  const int expected_calls =
    static_cast<int>(debug_enabled) + static_cast<int>(info_enabled) + static_cast<int>(error_enabled);
  check(g_sink_calls == expected_calls, "enabled levels did not reach the sink exactly once");
}

}  // namespace

extern "C" int32_t aclshmemx_set_log_level(int) { return 0; }

extern "C" int32_t aclshmemx_set_extern_logger(CannLogger logger) {
  if (g_cann_logger == nullptr) {
    g_cann_logger = logger;
  }
  return 0;
}

extern "C" int32_t hpshmem_log(int32_t, const char *message) {
  if (message == nullptr) {
    return -1;
  }
  ++g_sink_calls;
  return 0;
}

int main(int argc, char **argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error("expected one test mode");
    }
    const std::string mode(argv[1]);
    if (mode == "capture") {
      test_capture_state();
    } else {
      test_configured_level(mode);
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
