/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fake_shmem_device.h"

#include "shmem.h"

namespace hyper_parallel::multicore::shmem::tests::fake_device {
namespace {

std::vector<Call> calls;
int32_t observed_signal = 0;

}  // namespace

void Reset() {
  calls.clear();
  observed_signal = 0;
}

const std::vector<Call> &Calls() { return calls; }

void SetObservedSignal(int32_t value) { observed_signal = value; }

void Record(Call call) { calls.push_back(call); }

int32_t ObservedSignal() { return observed_signal; }

}  // namespace hyper_parallel::multicore::shmem::tests::fake_device

void aclshmem_putmem(void *remote_dst, void *local_src, uint32_t bytes, int32_t target_pe) {
  using namespace hyper_parallel::multicore::shmem::tests::fake_device;
  Record({CallKind::Put, remote_dst, local_src, bytes, target_pe, 0, 0});
}

void aclshmem_getmem(void *local_dst, void *remote_src, uint32_t bytes, int32_t target_pe) {
  using namespace hyper_parallel::multicore::shmem::tests::fake_device;
  Record({CallKind::Get, local_dst, remote_src, bytes, target_pe, 0, 0});
}

void aclshmemx_signal_op(int32_t *remote_signal, int32_t value, int32_t operation, int32_t target_pe) {
  using namespace hyper_parallel::multicore::shmem::tests::fake_device;
  Record({CallKind::Signal, remote_signal, nullptr, 0U, target_pe, operation, value});
}

int32_t aclshmem_signal_wait_until(int32_t *local_signal, int32_t compare, int32_t value) {
  using namespace hyper_parallel::multicore::shmem::tests::fake_device;
  Record({CallKind::Wait, local_signal, nullptr, 0U, 0, compare, value});
  return ObservedSignal();
}

void aclshmem_barrier(int32_t team) {
  using namespace hyper_parallel::multicore::shmem::tests::fake_device;
  Record({CallKind::Barrier, nullptr, nullptr, 0U, team, 0, 0});
}
