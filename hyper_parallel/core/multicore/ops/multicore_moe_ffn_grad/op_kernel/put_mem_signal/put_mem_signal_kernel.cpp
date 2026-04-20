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

#include "kernel_operator.h"
#include "shmem.h"

using namespace AscendC;

template <typename T>
__aicore__ inline void CopyGmSingleValueToUb(GM_ADDR gm_addr, T* result) {
  __ubuf__ T *ubAddr = (__ubuf__ T*)(32);
  smem_shm_copy_gm2ub(ubAddr, (__gm__ T *)gm_addr, sizeof(T));
  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  *result = *ubAddr;
}

template <typename T>
class PutMemSignalKernel {
 public:
  __aicore__ inline PutMemSignalKernel() {}
  __aicore__ inline void Init(GM_ADDR target, int64_t target_offset, GM_ADDR src,
                              int64_t src_offset, int64_t size, GM_ADDR signal,
                              int64_t signal_offset, int32_t signal_value, int64_t signal_op,
                              int64_t target_pe, bool non_blocking, GM_ADDR workspace) {
    // Initialize pointers and parameters
    int32_t size_temp = 0;
    target_ = reinterpret_cast<__gm__ T *>(target);
    src_ = reinterpret_cast<__gm__ T *>(src);
    signal_ = reinterpret_cast<__gm__ int32_t *>(signal);
    // param
    target_offset_ = target_offset;
    src_offset_ = src_offset;
    size_ = size;
    signal_offset_ = signal_offset;
    signal_value_ = signal_value;
    signal_op_ = signal_op;
    target_pe_ = target_pe;
    non_blocking_ = non_blocking;

    // UB tiling set
    aiv_idx_ = 0;
    aiv_num_ = 1;

    syncGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace), 256);
    pipe.InitBuffer(localwork,1,sizeof(int32_t)*aiv_num_);
  }

  __aicore__ inline void Process() {
    // Perform data transfer
    auto size_per_core = size_ / aiv_num_;
    auto target_ptr = target_ + target_offset_ + aiv_idx_ * size_per_core;
    auto src_ptr = src_ + src_offset_ + aiv_idx_ * size_per_core;
    if (aiv_idx_ == aiv_num_ - 1) {
      size_per_core = size_ - size_per_core * aiv_idx_;
    }
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    /* CopyUB Config Set */
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint64_t copy_ub_size = device_state->mte_config.ub_size;
    __ubuf__ T *ping_buff = reinterpret_cast<__ubuf__ T *>(copy_ub);
    __ubuf__ T *pong_buff = reinterpret_cast<__ubuf__ T *>(copy_ub+copy_ub_size/2);
    auto ptr = aclshmem_ptr(target_ptr, target_pe_);
    __gm__ T* remote_ptr = reinterpret_cast<__gm__ T*>(ptr);
    uint64_t block_size = copy_ub_size/2/ sizeof(T)* sizeof(T);
    uint64_t remain = (size_per_core * sizeof(T)) % block_size;

    uint64_t repeat_times = (size_per_core * sizeof(T)) / block_size;
    uint64_t repeat_elem = block_size / sizeof(T);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint64_t i = 0; i < repeat_times; i++) {
      AscendC::TEventID EVENT_ID = i & 1 ? EVENT_ID0 : EVENT_ID1;
      __ubuf__ T *buf= i & 1 ? ping_buff : pong_buff;
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
      aclshmemi_copy_gm2ub(buf, src_ptr + i * repeat_elem, block_size);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
      aclshmemi_copy_ub2gm(remote_ptr + i * repeat_elem, buf, block_size);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    }
    if (remain > 0) {
      AscendC::TEventID EVENT_ID = repeat_times & 1 ? EVENT_ID0 : EVENT_ID1;
      __ubuf__ T *buf= repeat_times & 1 ? ping_buff : pong_buff;
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
      aclshmemi_copy_gm2ub(buf, src_ptr + repeat_times * repeat_elem, remain);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
      aclshmemi_copy_ub2gm(remote_ptr + repeat_times * repeat_elem, buf, remain);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

    // Write signal after data transfer
    auto signal_ptr = signal_ + signal_offset_;
    if (non_blocking_) {
      aclshmem_fence();
    } else {
      aclshmem_fence();
    }
    // Sync ensure corresponding tasks are done
    if (aiv_num_ > 1) {
      AscendC::LocalTensor<int32_t> workLocal = localwork.AllocTensor<int32_t>();
      AscendC::SyncAll(syncGlobal,workLocal,aiv_num_);
      localwork.FreeTensor(workLocal);
    }

    if (aiv_idx_ == 0) {
      aclshmemx_signal_op(signal_ptr, signal_value_, signal_op_ == 1 ? ACLSHMEM_SIGNAL_ADD : ACLSHMEM_SIGNAL_SET, target_pe_);
    }
  }

 private:
  __gm__ T *target_;
  __gm__ T *src_;
  __gm__ int32_t *signal_;
  AscendC::TPipe pipe;

  int64_t target_offset_;
  int64_t src_offset_;
  int64_t size_;
  int64_t signal_offset_;
  int32_t signal_value_;
  int64_t signal_op_;
  int64_t target_pe_;
  bool non_blocking_;
  AscendC::GlobalTensor<int32_t> syncGlobal;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> localwork;
  // ub
  uint32_t aiv_idx_;
  uint32_t aiv_num_;
};

extern "C" inline __aicore__ void put_mem_signal_kernel(GM_ADDR target, int64_t target_offset,
                                                              GM_ADDR src, int64_t src_offset,
                                                              int64_t size, GM_ADDR signal,
                                                              int64_t signal_offset, int64_t signal_value, GM_ADDR workspace,
                                                              int64_t signal_op, int64_t target_pe,
                                                              bool non_blocking) {
  PutMemSignalKernel<DTYPE_DY> op;
  op.Init(target, target_offset, src, src_offset, size, signal, signal_offset, signal_value, signal_op, target_pe, non_blocking, workspace);
  op.Process();
}