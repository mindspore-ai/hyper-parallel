/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "data_plane/mte_rma.h"
#include "data_plane/transfer_chunking.h"
#include "kernel_operator.h"

using namespace AscendC;
inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg {
  0
};

namespace ShmemKernel {

using hyper_parallel::multicore::shmem::data_plane::ChunkPolicy;
using hyper_parallel::multicore::shmem::data_plane::divide_aligned_capacity;
using hyper_parallel::multicore::shmem::data_plane::get_chunk;
using hyper_parallel::multicore::shmem::data_plane::make_chunk_plan;
using hyper_parallel::multicore::shmem::data_plane::partition_work;
namespace mte_rma = hyper_parallel::multicore::shmem::data_plane::mte_rma;

class AllGatherKernel {
 public:
  __aicore__ inline void Init(GM_ADDR output, GM_ADDR input, uint64_t local_bytes, int32_t root_rank,
                              int32_t root_size) {
    output_ = reinterpret_cast<__gm__ uint8_t *>(output);
    input_ = reinterpret_cast<__gm__ uint8_t *>(input);
    local_bytes_ = local_bytes;
    root_rank_ = root_rank;
    root_size_ = root_size;
    block_index_ = AscendC::GetBlockIdx();
    block_count_ = AscendC::GetBlockNum();
    pipe_.InitBuffer(copy_buffer_, kPrivateUbBytes);
    ping_tensor_ = copy_buffer_.GetWithOffset<uint8_t>(kChunkCapacityBytes, 0);
    pong_tensor_ = copy_buffer_.GetWithOffset<uint8_t>(kChunkCapacityBytes, kChunkCapacityBytes);
    ping_ = reinterpret_cast<__ubuf__ uint8_t *>(ping_tensor_.GetPhyAddr());
    pong_ = reinterpret_cast<__ubuf__ uint8_t *>(pong_tensor_.GetPhyAddr());
  }

  __aicore__ inline void Process() {
    if (local_bytes_ == 0 || root_size_ <= 0 || block_count_ == 0) {
      return;
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    ProcessWorkItems();
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    mte_rma::quiet();
  }

 private:
  static constexpr uint32_t kPrivateUbBytes = 16 * 1024;
  static constexpr uint32_t kBufferCount = 2;
  static constexpr uint32_t kMinimumMteTransferBytes = 32;
  static constexpr uint32_t kChunkCapacityBytes =
    divide_aligned_capacity(kPrivateUbBytes, kBufferCount, kMinimumMteTransferBytes);
  static constexpr ChunkPolicy kChunkPolicy{kChunkCapacityBytes, kMinimumMteTransferBytes};

  static_assert(kChunkCapacityBytes == 8 * 1024);
  static_assert(kMinimumMteTransferBytes <= kChunkCapacityBytes / 2);

  __aicore__ inline void ProcessWorkItems() {
    const auto plan = make_chunk_plan(local_bytes_, kChunkPolicy);
    const uint64_t total_items = plan.chunk_count * static_cast<uint64_t>(root_size_);
    const auto work = partition_work(total_items, block_count_, block_index_);
    const uint64_t work_end = work.first_item + work.item_count;
    uint64_t work_item = work.first_item;
    uint64_t issued_chunk_count = 0;

    while (work_item < work_end) {
      const int32_t target_pe = static_cast<int32_t>(work_item / plan.chunk_count);
      const uint64_t first_chunk_index = work_item % plan.chunk_count;
      uint64_t target_chunk_count = plan.chunk_count - first_chunk_index;
      const uint64_t remaining_items = work_end - work_item;
      if (target_chunk_count > remaining_items) {
        target_chunk_count = remaining_items;
      }

      const auto first_chunk = get_chunk(plan, first_chunk_index);
      const auto last_chunk = get_chunk(plan, first_chunk_index + target_chunk_count - 1);
      const uint64_t copy_bytes = last_chunk.offset_bytes + last_chunk.size_bytes - first_chunk.offset_bytes;
      __gm__ uint8_t *local_output_slot =
        output_ + static_cast<uint64_t>(root_rank_) * local_bytes_ + first_chunk.offset_bytes;
      CopyRange(local_output_slot, input_ + first_chunk.offset_bytes, copy_bytes, target_pe, issued_chunk_count);
      work_item += target_chunk_count;
    }
  }

  __aicore__ inline void CopyRange(__gm__ uint8_t *local_symmetric_output, __gm__ uint8_t *input, uint64_t bytes,
                                   int32_t target_pe, uint64_t &issued_chunk_count) {
    const auto plan = make_chunk_plan(bytes, kChunkPolicy);
    for (uint64_t chunk_index = 0; chunk_index < plan.chunk_count; ++chunk_index) {
      const auto chunk = get_chunk(plan, chunk_index);
      if ((issued_chunk_count & 1) == 0) {
        CopyChunk(local_symmetric_output + chunk.offset_bytes, input + chunk.offset_bytes, chunk.size_bytes, target_pe,
                  ping_tensor_, ping_, EVENT_ID0);
      } else {
        CopyChunk(local_symmetric_output + chunk.offset_bytes, input + chunk.offset_bytes, chunk.size_bytes, target_pe,
                  pong_tensor_, pong_, EVENT_ID1);
      }
      ++issued_chunk_count;
    }
  }

  __aicore__ inline void CopyChunk(__gm__ uint8_t *local_symmetric_output, __gm__ uint8_t *input, uint32_t bytes,
                                   int32_t target_pe, AscendC::LocalTensor<uint8_t> &buffer_tensor,
                                   __ubuf__ uint8_t *buffer, AscendC::TEventID event_id) {
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
    if (target_pe == root_rank_) {
      CopyLocal(local_symmetric_output, input, buffer_tensor, bytes, event_id);
    } else {
      mte_rma::put_nbi(local_symmetric_output, input, buffer, kChunkCapacityBytes, bytes, target_pe, event_id);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
  }

  __aicore__ inline void CopyLocal(__gm__ uint8_t *output, __gm__ uint8_t *input, AscendC::LocalTensor<uint8_t> &buffer,
                                   uint32_t bytes, AscendC::TEventID event_id) {
    AscendC::GlobalTensor<uint8_t> input_tensor;
    AscendC::GlobalTensor<uint8_t> output_tensor;
    input_tensor.SetGlobalBuffer(input, bytes);
    output_tensor.SetGlobalBuffer(output, bytes);
    AscendC::DataCopyExtParams copy_params{1, bytes, 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint8_t> pad_params{false, 0, 0, 0};
    AscendC::DataCopyPad(buffer, input_tensor, copy_params, pad_params);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
    AscendC::DataCopyPad(output_tensor, buffer, copy_params);
  }

  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECOUT> copy_buffer_;
  AscendC::LocalTensor<uint8_t> ping_tensor_;
  AscendC::LocalTensor<uint8_t> pong_tensor_;
  __gm__ uint8_t *output_ = nullptr;
  __gm__ uint8_t *input_ = nullptr;
  __ubuf__ uint8_t *ping_ = nullptr;
  __ubuf__ uint8_t *pong_ = nullptr;
  uint64_t local_bytes_ = 0;
  int32_t root_rank_ = 0;
  int32_t root_size_ = 0;
  uint32_t block_index_ = 0;
  uint32_t block_count_ = 1;
};

__global__ __aicore__ void all_gather_kernel(GM_ADDR output, GM_ADDR input, uint64_t local_bytes, int32_t root_rank,
                                             int32_t root_size) {
  AllGatherKernel kernel;
  kernel.Init(output, input, local_bytes, root_rank, root_size);
  kernel.Process();
}

void launch_all_gather(uint32_t block_dim, void *stream, uint8_t *output, uint8_t *input, uint64_t local_bytes,
                       int32_t root_rank, int32_t root_size) {
  all_gather_kernel<<<block_dim, nullptr, stream>>>(output, input, local_bytes, root_rank, root_size);
}

}  // namespace ShmemKernel
