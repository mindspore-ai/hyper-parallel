/**
 * Copyright 2024 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <iostream>

#include "kernel_operator.h"
#include "acl/acl.h"

#include "shmem.h"

using namespace AscendC;
// Define constants for compare operations
#define COMPARE_OP_EQUAL 0
#define COMPARE_OP_GREATER 1
#define COMPARE_OP_LESS 2

namespace ShmemKernel {
template <typename T>
__aicore__ void CopyGmSingleValueToUb(GM_ADDR gm_addr, T *result) {
  __ubuf__ T *ubAddr = (__ubuf__ T *)(32);
  aclshmemi_copy_gm2ub(ubAddr, (__gm__ T *)gm_addr, sizeof(T));
  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  *result = *ubAddr;
}

template <typename T>
class SignalWaitUntilKernel {
 public:
  __aicore__ inline SignalWaitUntilKernel() {}
  __aicore__ inline void Init(GM_ADDR depend_target, GM_ADDR signal, GM_ADDR signal_offset, int64_t compare_op,
                              GM_ADDR compare_value) {
    signal_addr_ = reinterpret_cast<__gm__ int32_t *>(signal);

    compare_op_ = compare_op;
    CopyGmSingleValueToUb<int64_t>(signal_offset, &signal_offset_);
    CopyGmSingleValueToUb<int32_t>(compare_value, &compare_value_);

    depend_target_ = reinterpret_cast<__gm__ T *>(depend_target);
  }

  __aicore__ inline void Process() {
    // Wait until the signal condition is met
    auto signal_addr = signal_addr_ + signal_offset_;
    switch (compare_op_) {
      case COMPARE_OP_EQUAL:
        aclshmem_signal_wait_until(signal_addr, ACLSHMEM_CMP_EQ, compare_value_);
        break;
      case COMPARE_OP_GREATER:
        aclshmem_signal_wait_until(signal_addr, ACLSHMEM_CMP_GT, compare_value_);
        break;
      case COMPARE_OP_LESS:
        aclshmem_signal_wait_until(signal_addr, ACLSHMEM_CMP_LT, compare_value_);
        break;
      default:
        // Invalid compare operation
        return;
    }
  }

  __gm__ T *depend_target_;
  __gm__ int32_t *signal_addr_;
  int64_t signal_offset_ = 0;
  int64_t compare_op_ = 0;
  int32_t compare_value_ = 0;
};

template <typename T>
__global__ __aicore__ void signal_wait_until_kernel(GM_ADDR depend_target, GM_ADDR signal, GM_ADDR signal_offset,
                                                    int64_t compare_op, GM_ADDR compare_value) {
  // Assuming float data type for this example
  SignalWaitUntilKernel<T> op;
  op.Init(depend_target, signal, signal_offset, compare_op, compare_value);
  op.Process();
}

void signal_wait_until(uint32_t block_dim, void *stream, uint64_t elementSize, uint8_t *depend_target, uint8_t *signal,
                       uint8_t *signal_offset, uint8_t *compare_value, int64_t compare_op) {
  if (elementSize == 1) {
    signal_wait_until_kernel<int8_t>
      <<<block_dim, nullptr, stream>>>(depend_target, signal, signal_offset, compare_op, compare_value);
  } else if (elementSize == 2) {
    signal_wait_until_kernel<int16_t>
      <<<block_dim, nullptr, stream>>>(depend_target, signal, signal_offset, compare_op, compare_value);
  } else if (elementSize == 4) {
    signal_wait_until_kernel<int32_t>
      <<<block_dim, nullptr, stream>>>(depend_target, signal, signal_offset, compare_op, compare_value);
  } else if (elementSize == 8) {
    signal_wait_until_kernel<int64_t>
      <<<block_dim, nullptr, stream>>>(depend_target, signal, signal_offset, compare_op, compare_value);
  } else {
    std::cerr << "Unknown dtype with size" << elementSize << std::endl;
  }
}
}  // namespace ShmemKernel
// Additional kernel instances for other data types can be added here
