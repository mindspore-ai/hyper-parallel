/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <exception>
#include <iostream>
#include <string_view>

namespace hyper_parallel::multicore::shmem::tests {

#if defined(HP_SHMEM_CORE_TEST)
void test_make_chunk_plan_chunk_count();
void test_divide_aligned_capacity();
void test_get_chunk_tail_rebalance();
void test_partition_work_remainder_distribution();
void test_registry_match_for_free_requires_complete();
void test_registry_double_free_detected();
void test_registry_resolve_allows_subview();
void test_registry_diagnostics_are_ordered_and_track_high_watermark();
void test_config_strict_unsigned_decimal();
void test_config_endpoint_minimal_and_passthrough();
void test_config_data_engine_validation();
void test_runtime_free_device_guard_phase_classification();
void test_runtime_free_failure_preserves_record();
void test_runtime_initialize_failure_allows_retry();
void test_runtime_shutdown_uninitialized_idempotent();
#elif defined(HP_SHMEM_HOST_ADAPTER_TEST)
void test_host_init_attr_fields_and_order();
void test_host_timeout_and_cann_error_mapping();
void test_host_alloc_nullptr_and_free_passthrough();
void test_host_stream_operations_forward_arguments();
#elif defined(HP_SHMEM_DEVICE_SURFACE_TEST)
void test_put_get_forward_once();
void test_put_signal_orders_data_before_signal();
void test_signal_compare_mapping();
#else
#error "Select exactly one HP_SHMEM_*_TEST target"
#endif

struct TestCase {
  std::string_view name;
  void (*run)();
};

constexpr TestCase kCases[]{
#if defined(HP_SHMEM_CORE_TEST)
  {"test_make_chunk_plan_chunk_count", test_make_chunk_plan_chunk_count},
  {"test_divide_aligned_capacity", test_divide_aligned_capacity},
  {"test_get_chunk_tail_rebalance", test_get_chunk_tail_rebalance},
  {"test_partition_work_remainder_distribution", test_partition_work_remainder_distribution},
  {"test_registry_match_for_free_requires_complete", test_registry_match_for_free_requires_complete},
  {"test_registry_double_free_detected", test_registry_double_free_detected},
  {"test_registry_resolve_allows_subview", test_registry_resolve_allows_subview},
  {"test_registry_diagnostics_are_ordered_and_track_high_watermark",
   test_registry_diagnostics_are_ordered_and_track_high_watermark},
  {"test_config_strict_unsigned_decimal", test_config_strict_unsigned_decimal},
  {"test_config_endpoint_minimal_and_passthrough", test_config_endpoint_minimal_and_passthrough},
  {"test_config_data_engine_validation", test_config_data_engine_validation},
  {"test_runtime_free_device_guard_phase_classification", test_runtime_free_device_guard_phase_classification},
  {"test_runtime_free_failure_preserves_record", test_runtime_free_failure_preserves_record},
  {"test_runtime_initialize_failure_allows_retry", test_runtime_initialize_failure_allows_retry},
  {"test_runtime_shutdown_uninitialized_idempotent", test_runtime_shutdown_uninitialized_idempotent},
#elif defined(HP_SHMEM_HOST_ADAPTER_TEST)
  {"test_host_init_attr_fields_and_order", test_host_init_attr_fields_and_order},
  {"test_host_timeout_and_cann_error_mapping", test_host_timeout_and_cann_error_mapping},
  {"test_host_alloc_nullptr_and_free_passthrough", test_host_alloc_nullptr_and_free_passthrough},
  {"test_host_stream_operations_forward_arguments", test_host_stream_operations_forward_arguments},
#elif defined(HP_SHMEM_DEVICE_SURFACE_TEST)
  {"test_put_get_forward_once", test_put_get_forward_once},
  {"test_put_signal_orders_data_before_signal", test_put_signal_orders_data_before_signal},
  {"test_signal_compare_mapping", test_signal_compare_mapping},
#endif
};

}  // namespace hyper_parallel::multicore::shmem::tests

int main(int argc, char **argv) {
  using hyper_parallel::multicore::shmem::tests::kCases;
  if (argc != 3 || std::string_view(argv[1]) != "--case") {
    std::cerr << "usage: " << argv[0] << " --case <name>\n";
    return 2;
  }

  const std::string_view requested_case(argv[2]);
  for (const auto &test_case : kCases) {
    if (test_case.name != requested_case) {
      continue;
    }
    try {
      test_case.run();
      return 0;
    } catch (const std::exception &error) {
      std::cerr << requested_case << " failed: " << error.what() << '\n';
      return 1;
    }
  }
  std::cerr << "unknown test case: " << requested_case << '\n';
  return 2;
}
