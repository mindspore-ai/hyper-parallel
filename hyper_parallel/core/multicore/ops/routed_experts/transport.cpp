// Copyright 2026 Huawei Technologies Co., Ltd
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernel_operator.h"
#include "acl/acl.h"
#include "data_plane/mte_rma.h"
#include "data_plane/sync.h"
#include "transport.h"

using namespace AscendC;
namespace communication = hyper_parallel::multicore::shmem::data_plane;

inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg {0};

namespace {

class TransportWorker {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR mapping, GM_ADDR sends, GM_ADDR receives,
                              GM_ADDR offsets, GM_ADDR ring, GM_ADDR signals, GM_ADDR progress, int32_t rank,
                              int32_t ranks, int32_t lanes, int32_t block_rows, int32_t row_bytes, bool gather,
                              uint64_t idle_cycles, int64_t input_rows, int64_t output_rows, int64_t mapping_rows,
                              int64_t send_count, int64_t receive_count, GM_ADDR input_ready, GM_ADDR input_expected,
                              GM_ADDR output_ready, GM_ADDR expert_slots, int32_t producers,
                              int32_t active_experts, int32_t slot_count, GM_ADDR headers,
                              GM_ADDR dedup_state, GM_ADDR group_metadata, bool reduce,
                              bool publish_each_block, float reduce_dense_ratio,
                              float forward_compact_max_unique_ratio) {
    input_ = input;
    output_ = output;
    mapping_ = reinterpret_cast<__gm__ int64_t *>(mapping);
    sends_ = reinterpret_cast<__gm__ int64_t *>(sends);
    receives_ = reinterpret_cast<__gm__ int64_t *>(receives);
    ring_ = ring;
    headers_ = reinterpret_cast<__gm__ int64_t *>(headers);
    dedup_state_ = reinterpret_cast<__gm__ int64_t *>(dedup_state);
    group_metadata_ = reinterpret_cast<__gm__ int64_t *>(group_metadata);
    reduce_ = reduce;
    publish_each_block_ = publish_each_block;
    reduce_dense_ratio_ = reduce_dense_ratio;
    forward_compact_max_unique_ratio_ = forward_compact_max_unique_ratio;
    signals_ = reinterpret_cast<__gm__ int32_t *>(signals);
    split_workers_ = ranks <= 24;
    peer_ = split_workers_ ? GetBlockIdx() / 2 : GetBlockIdx();
    send_worker_ = !split_workers_ || GetBlockIdx() % 2 == 0;
    auto *config = reinterpret_cast<__gm__ int64_t *>(offsets) + peer_ * 6;
    send_cursor_ = config[0];
    send_end_ = config[1];
    receive_cursor_ = config[2];
    receive_begin_ = receive_cursor_;
    receive_end_ = config[3];
    if (config[4] < 0 || config[4] >= (1LL << 30) || config[5] < 0 || config[5] >= (1LL << 30)) {
      Trap();
    }
    send_sequence_ = static_cast<int32_t>(config[4] + 1);
    receive_sequence_ = static_cast<int32_t>(config[5] + 1);
    if (send_cursor_ < 0 || send_end_ < send_cursor_ || send_end_ > send_count || receive_cursor_ < 0 ||
        receive_end_ < receive_cursor_ || receive_end_ > receive_count || send_sequence_ < 1 ||
        receive_sequence_ < 1) {
      Trap();
    }
    progress_ = reinterpret_cast<__gm__ int64_t *>(progress) + GetBlockIdx() * 8;
    rank_ = rank;
    ranks_ = ranks;
    lanes_ = lanes;
    block_rows_ = block_rows;
    row_bytes_ = row_bytes;
    gather_ = gather;
    idle_cycles_ = idle_cycles;
    input_rows_ = input_rows;
    output_rows_ = output_rows;
    mapping_rows_ = mapping_rows;
    if (dedup_state_ != nullptr) {
      if (!gather || (group_metadata_ == nullptr && output_rows_ > 0) ||
          input_rows_ >= (1LL << 31) || output_rows_ >= (1LL << 31)) {
        Trap();
      }
      const int64_t stride = ((input_rows_ > 0 ? input_rows_ : 1) + 7) / 8 * 8;
      seen_ = dedup_state_ + static_cast<int64_t>(peer_) * stride;
      last_ = dedup_state_ + (static_cast<int64_t>(ranks_) + peer_) * stride;
    }
    if (reduce_) {
      if (gather || (group_metadata_ == nullptr && input_rows_ > 0)) {
        Trap();
      }
      // Forward recorded each row's predecessor before backward starts.
      links_ = group_metadata_ == nullptr ? nullptr : group_metadata_ + 1;
      if (send_cursor_ < send_end_ && peer_ != rank_) {
        const int64_t first = sends_[send_cursor_ * 6 + 3];
        const auto *summary = group_metadata_ + first * kTokenStride;
        const int64_t unique = summary[2];
        const int64_t rows = summary[3];
        if (unique < 1 || rows < unique || rows > input_rows_) {
          Trap();
        }
        dense_return_ = static_cast<float>(unique) >= reduce_dense_ratio_ * static_cast<float>(rows);
      }
    }
    input_ready_ = reinterpret_cast<__gm__ int32_t *>(input_ready);
    input_expected_ = reinterpret_cast<__gm__ int32_t *>(input_expected);
    output_ready_ = reinterpret_cast<__gm__ int32_t *>(output_ready);
    expert_slots_ = reinterpret_cast<__gm__ int64_t *>(expert_slots);
    producers_ = producers;
    active_experts_ = active_experts;
    slot_count_ = slot_count;
    CheckReadiness();
    // Pure compact senders use one bank; adaptive dense dispatch and reverse
    // singleton copies use both banks for contiguous transfer overlap.
    const bool compressed_sender = split_workers_ && send_worker_ && peer_ != rank_ &&
                                   dedup_state_ != nullptr && forward_compact_max_unique_ratio_ >= 1.0F;
    pipe_.InitBuffer(buffer_, (compressed_sender ? 1 : 2) * kScratchBytes);
    scratch_ = buffer_.Get<uint8_t>();
    scratch_ptr_ = reinterpret_cast<__ubuf__ uint8_t *>(scratch_.GetPhyAddr());
    pipe_.InitBuffer(header_buffer_, ((block_rows_ + 8) / 8 * 8) * sizeof(int64_t));
    header_local_ = header_buffer_.Get<int64_t>();
    header_ptr_ = reinterpret_cast<__ubuf__ int64_t *>(header_local_.GetPhyAddr());
    if (reduce_ && peer_ != rank_ && send_worker_ && send_cursor_ < send_end_) {
      pipe_.InitBuffer(reduce_input_buffer_, kReduceElements * sizeof(bfloat16_t));
      pipe_.InitBuffer(reduce_value_buffer_, kReduceElements * sizeof(float));
      pipe_.InitBuffer(reduce_sum_buffer_, kReduceElements * sizeof(float));
      reduce_input_ = reduce_input_buffer_.Get<bfloat16_t>();
      reduce_value_ = reduce_value_buffer_.Get<float>();
      reduce_sum_ = reduce_sum_buffer_.Get<float>();
    }
    // The two per-peer tables fit in UB for the common 4K/8K token extent.
    // Larger extents retain the GM fallback for the predecessor table.
    if (compressed_sender && dedup_state_ != nullptr && input_rows_ <= kLocalTokenCapacity) {
      const uint32_t words = static_cast<uint32_t>(((input_rows_ > 0 ? input_rows_ : 1) + 7) / 8 * 8);
      pipe_.InitBuffer(seen_buffer_, words * sizeof(int32_t));
      seen_local_ = seen_buffer_.Get<int32_t>();
      seen_ptr_ = reinterpret_cast<__ubuf__ int32_t *>(seen_local_.GetPhyAddr());
      Duplicate(seen_local_, static_cast<int32_t>(0), words);
      SetFlag<HardEvent::V_S>(EVENT_ID0);
      WaitFlag<HardEvent::V_S>(EVENT_ID0);
      if (input_rows_ <= kLocalTokenCapacity / 2) {
        pipe_.InitBuffer(last_buffer_, words * sizeof(int32_t));
        last_local_ = last_buffer_.Get<int32_t>();
        last_ptr_ = reinterpret_cast<__ubuf__ int32_t *>(last_local_.GetPhyAddr());
        Duplicate(last_local_, static_cast<int32_t>(0), words);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
      }
    }
  }

  __aicore__ inline void Run() {
    if (peer_ == rank_) {
      if (send_worker_) {
        RunLocal();
      }
      return;
    }
    if (!split_workers_) {
      RunBidirectional();
    } else if (send_worker_) {
      RunSender();
    } else {
      RunReceiver();
    }
  }

 private:
  static constexpr uint32_t kScratchBytes = 65536;
  static constexpr uint32_t kSignalStride = 16;
  static constexpr uint32_t kReduceElements = 4096;
  static constexpr uint32_t kTokenStride = 8;
  static constexpr int64_t kLocalTokenCapacity = 16384;

  __aicore__ inline int64_t Seen(int64_t token) {
    return seen_ptr_ == nullptr ? seen_[token] : seen_ptr_[token];
  }

  __aicore__ inline void SetSeen(int64_t token, int64_t row) {
    if (seen_ptr_ == nullptr) {
      seen_[token] = row;
    } else {
      seen_ptr_[token] = static_cast<int32_t>(row);
    }
  }

  __aicore__ inline int64_t Last(int64_t token) {
    return last_ptr_ == nullptr ? last_[token] : last_ptr_[token];
  }

  __aicore__ inline void SetLast(int64_t token, int64_t row) {
    if (last_ptr_ == nullptr) {
      last_[token] = row;
    } else {
      last_ptr_[token] = static_cast<int32_t>(row);
    }
  }

  __aicore__ inline void CheckRemoteAccess() {
    if (send_cursor_ < send_end_) {
      uint64_t unused_address = 0, unused_capacity = 0;
      if (!hyper_parallel::multicore::shmem::cann::device::query_mte_resources(
            peer_, unused_address, unused_capacity)) {
        progress_[4] = -2;
        Trap();
      }
    }
  }

  __aicore__ inline void RunSender() {
    CheckRemoteAccess();
    auto last_progress = GetSystemCycle();
    while (send_cursor_ < send_end_) {
      if (Ready(1, send_sequence_) && InputReady(sends_ + send_cursor_ * 6)) {
        Send();
        progress_[0] = send_cursor_;
        last_progress = GetSystemCycle();
      } else if (GetSystemCycle() - last_progress > idle_cycles_) {
        progress_[2] = send_sequence_;
        progress_[4] = -3;
        Trap();
      }
    }
  }

  __aicore__ inline void RunReceiver() {
    auto last_progress = GetSystemCycle();
    while (receive_cursor_ < receive_end_) {
      if (Ready(0, receive_sequence_)) {
        Receive();
        progress_[1] = receive_cursor_;
        last_progress = GetSystemCycle();
      } else if (GetSystemCycle() - last_progress > idle_cycles_) {
        progress_[3] = receive_sequence_;
        progress_[4] = -4;
        Trap();
      }
    }
  }

  __aicore__ inline void RunBidirectional() {
    CheckRemoteAccess();
    auto last_progress = GetSystemCycle();
    while (send_cursor_ < send_end_ || receive_cursor_ < receive_end_) {
      bool advanced = false;
      if (receive_cursor_ < receive_end_ && Ready(0, receive_sequence_)) {
        Receive();
        advanced = true;
      }
      if (send_cursor_ < send_end_ && Ready(1, send_sequence_) && InputReady(sends_ + send_cursor_ * 6)) {
        Send();
        advanced = true;
      }
      if (advanced) {
        last_progress = GetSystemCycle();
        progress_[0] = send_cursor_;
        progress_[1] = receive_cursor_;
      } else if (GetSystemCycle() - last_progress > idle_cycles_) {
        progress_[2] = send_sequence_;
        progress_[3] = receive_sequence_;
        Trap();
      }
    }
  }

  __aicore__ inline void CheckReadiness() {
    if (producers_ < 0 || active_experts_ < 0 || slot_count_ < 0) {
      Trap();
    }
    if (active_experts_ == 0 || (input_ready_ == nullptr && output_ready_ == nullptr)) {
      return;
    }
    CheckReadinessStorage();
  }

  __aicore__ inline void CheckReadinessStorage() {
    if (expert_slots_ == nullptr || slot_count_ < active_experts_ ||
        (input_ready_ != nullptr && (input_expected_ == nullptr || producers_ == 0)) ||
        reinterpret_cast<uint64_t>(input_ready_) % (kSignalStride * sizeof(int32_t)) != 0 ||
        reinterpret_cast<uint64_t>(output_ready_) % (kSignalStride * sizeof(int32_t)) != 0) {
      Trap();
    }
    const int64_t expert_stride_bytes = static_cast<int64_t>(active_experts_) * kSignalStride * sizeof(int32_t);
    if (producers_ > INT64_MAX / expert_stride_bytes || ranks_ > INT64_MAX / expert_stride_bytes) {
      Trap();
    }
  }

  __aicore__ inline int32_t ExpertSlot(__gm__ int64_t *edge) {
    if (edge[2] < 0 || edge[2] >= slot_count_) {
      Trap();
    }
    const int64_t slot = expert_slots_[edge[2]];
    if (slot < 0 || slot >= active_experts_) {
      Trap();
    }
    return static_cast<int32_t>(slot);
  }

  __aicore__ inline void RefreshLocalFlag(__gm__ int32_t *address) {
    GlobalTensor<int32_t> cell;
    cell.SetGlobalBuffer(address, kSignalStride);
    __asm__ __volatile__("" : : : "memory");
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(cell);
    __asm__ __volatile__("" : : : "memory");
  }

  __aicore__ inline bool InputReady(__gm__ int64_t *edge) {
    if (input_ready_ == nullptr || active_experts_ == 0) {
      return true;
    }
    const int32_t slot = ExpertSlot(edge);
    if (slot == admitted_input_slot_) {
      return true;
    }
    for (int32_t producer = 0; producer < producers_; ++producer) {
      const int64_t index = static_cast<int64_t>(producer) * active_experts_ + slot;
      const int32_t expected = input_expected_[index];
      if (expected < 0) {
        Trap();
      }
      if (expected == 0) {
        continue;
      }
      auto *cell = input_ready_ + index * kSignalStride;
      RefreshLocalFlag(cell);
      const int32_t observed = *reinterpret_cast<__gm__ volatile int32_t *>(cell);
      if (observed < 0) {
        Trap();
      }
      // A returning probe allows receive progress and watchdog checks while the producer is pending.
      if (observed < expected) {
        return false;
      }
    }
    PipeBarrier<PIPE_ALL>();
    // A phase's readiness counters only advance. Repeated blocks of an already
    // admitted expert do not need to invalidate every producer cache line again.
    admitted_input_slot_ = slot;
    return true;
  }

  __aicore__ inline void CompleteOutput(__gm__ int64_t *edge, bool last_expert_block) {
    // Scalar publication must follow completion of every payload MTE write.
    PipeBarrier<PIPE_ALL>();
    if (output_ready_ == nullptr || active_experts_ == 0) {
      return;
    }
    ++pending_output_blocks_;
    if (!last_expert_block && !publish_each_block_) {
      return;
    }
    const int32_t slot = ExpertSlot(edge);
    auto *cell = output_ready_ + (static_cast<int64_t>(peer_) * active_experts_ + slot) * kSignalStride;
    auto *counter = reinterpret_cast<__gm__ volatile int32_t *>(cell);
    // This peer worker is the only writer of its phase-local cache line.
    RefreshLocalFlag(cell);
    const int32_t previous = *counter;
    if (previous < 0 || pending_output_blocks_ > INT32_MAX - previous) {
      Trap();
    }
    *counter = previous + pending_output_blocks_;
    pending_output_blocks_ = 0;
    RefreshLocalFlag(cell);
    PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void RunLocal() {
    auto last_progress = GetSystemCycle();
    while (send_cursor_ < send_end_) {
      auto *edge = sends_ + send_cursor_ * 6;
      if (InputReady(edge)) {
        CopyLocalEdge(edge);
        ++send_cursor_;
        progress_[0] = send_cursor_;
        last_progress = GetSystemCycle();
      } else if (GetSystemCycle() - last_progress > idle_cycles_) {
        progress_[2] = send_sequence_;
        progress_[3] = receive_sequence_;
        Trap();
      }
    }
  }

  __aicore__ inline __gm__ int32_t *Signal(int32_t kind, int32_t peer, int32_t lane) {
    return signals_ + ((kind * ranks_ + peer) * lanes_ + lane) * kSignalStride;
  }

  __aicore__ inline bool Ready(int32_t kind, int32_t sequence) {
    const int32_t lane = (sequence - 1) % lanes_;
    // A returning probe preserves receive progress and bounded watchdog checks.
    auto *signal = Signal(kind, peer_, lane);
    if (kind == 0 && reduce_ &&
        aclshmem_int32_test(signal, ACLSHMEM_CMP_EQ, -sequence) != 0) {
      return true;
    }
    return aclshmem_int32_test(signal, ACLSHMEM_CMP_EQ, sequence) != 0;
  }

  __aicore__ inline GM_ADDR Ring(int32_t source, int32_t lane) {
    return ring_ + static_cast<int64_t>(source * lanes_ + lane) * block_rows_ * row_bytes_;
  }

  __aicore__ inline GM_ADDR Header(int32_t source, int32_t lane) {
    return reinterpret_cast<GM_ADDR>(headers_ +
      static_cast<int64_t>(source * lanes_ + lane) * ((block_rows_ + 8) / 8 * 8));
  }

  __aicore__ inline void CopyLocal(GM_ADDR target, GM_ADDR source, uint64_t bytes) {
    const int32_t chunks = static_cast<int32_t>((bytes + kScratchBytes - 1) / kScratchBytes);
    for (int32_t chunk = 0; chunk < chunks; ++chunk) {
      const int32_t slot = chunk % 2;
      if (chunk >= 2) {
        WaitFlag<HardEvent::MTE3_MTE2>(slot);
      }
      auto local = scratch_[slot * kScratchBytes];
      const uint64_t offset = static_cast<uint64_t>(chunk) * kScratchBytes;
      const uint32_t size = static_cast<uint32_t>(bytes - offset < kScratchBytes ? bytes - offset : kScratchBytes);
      GlobalTensor<uint8_t> src, dst;
      src.SetGlobalBuffer(source + offset, size);
      dst.SetGlobalBuffer(target + offset, size);
      DataCopyExtParams copy{1, size, 0, 0, 0};
      DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
      DataCopyPad(local, src, copy, pad);
      SetFlag<HardEvent::MTE2_MTE3>(slot);
      WaitFlag<HardEvent::MTE2_MTE3>(slot);
      DataCopyPad(dst, local, copy);
      SetFlag<HardEvent::MTE3_MTE2>(slot);
    }
    for (int32_t slot = 0; slot < (chunks < 2 ? chunks : 2); ++slot) {
      WaitFlag<HardEvent::MTE3_MTE2>(slot);
    }
  }

  __aicore__ inline void CopyLocalEdge(__gm__ int64_t *edge) {
    CheckEdge(edge, true, true);
    CopyRows(output_ + edge[4] * row_bytes_, edge[3], edge[5]);
    CompleteOutput(edge, send_cursor_ + 1 == send_end_ || sends_[(send_cursor_ + 1) * 6 + 2] != edge[2]);
  }

  __aicore__ inline void CopyRows(GM_ADDR target, int64_t first, int64_t rows) {
    if (!gather_) {
      CopyLocal(target, input_ + first * row_bytes_, rows * row_bytes_);
      return;
    }
    const int32_t batch_rows = kScratchBytes / row_bytes_;
    const int32_t chunks = static_cast<int32_t>((rows + batch_rows - 1) / batch_rows);
    for (int32_t chunk = 0; chunk < chunks; ++chunk) {
      const int32_t slot = chunk % 2;
      if (chunk >= 2) {
        WaitFlag<HardEvent::MTE3_MTE2>(slot);
      }
      auto local = scratch_[slot * kScratchBytes];
      const int64_t row = static_cast<int64_t>(chunk) * batch_rows;
      const int32_t count = static_cast<int32_t>(rows - row < batch_rows ? rows - row : batch_rows);
      for (int32_t index = 0; index < count; ++index) {
        GlobalTensor<uint8_t> source;
        source.SetGlobalBuffer(input_ + SourceRow(first + row + index) * row_bytes_, row_bytes_);
        DataCopyExtParams copy{1, static_cast<uint32_t>(row_bytes_), 0, 0, 0};
        DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
        DataCopyPad(local[index * row_bytes_], source, copy, pad);
      }
      SetFlag<HardEvent::MTE2_MTE3>(slot);
      WaitFlag<HardEvent::MTE2_MTE3>(slot);
      GlobalTensor<uint8_t> destination;
      destination.SetGlobalBuffer(target + row * row_bytes_, count * row_bytes_);
      DataCopyExtParams copy{1, static_cast<uint32_t>(count * row_bytes_), 0, 0, 0};
      DataCopyPad(destination, local, copy);
      SetFlag<HardEvent::MTE3_MTE2>(slot);
    }
    for (int32_t slot = 0; slot < (chunks < 2 ? chunks : 2); ++slot) {
      WaitFlag<HardEvent::MTE3_MTE2>(slot);
    }
  }

  __aicore__ inline int64_t PackEntry(int64_t predecessor, int32_t marker) {
    return static_cast<int64_t>((static_cast<uint64_t>(predecessor) << 32) |
                                static_cast<uint32_t>(marker));
  }

  __aicore__ inline int32_t Marker(int64_t entry) {
    return static_cast<int32_t>(static_cast<uint32_t>(entry));
  }

  __aicore__ inline void ReduceTokenRow(int64_t carrier_row, LocalTensor<uint8_t> target) {
    // Forward chose the final row in receive order as carrier. Backward send
    // admission follows that same order, so its earlier members are ready.
    int64_t first = carrier_row + 1;
    if (first < 1 || first > input_rows_) {
      Trap();
    }
    if (links_[(first - 1) * kTokenStride] == 1) {
      GlobalTensor<uint8_t> source;
      source.SetGlobalBuffer(input_ + (first - 1) * row_bytes_, row_bytes_);
      DataCopyExtParams copy{1, static_cast<uint32_t>(row_bytes_), 0, 0, 0};
      DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
      DataCopyPad(target, source, copy, pad);
      return;
    }
    const int32_t hidden = row_bytes_ / sizeof(bfloat16_t);
    for (int32_t column = 0; column < hidden; column += kReduceElements) {
      const uint32_t count = hidden - column < kReduceElements ? hidden - column : kReduceElements;
      Duplicate(reduce_sum_, 0.0F, count);
      PipeBarrier<PIPE_V>();
      int64_t link = first;
      while (link != 0) {
        if (link < 1 || link > input_rows_) {
          Trap();
        }
        GlobalTensor<bfloat16_t> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
          input_ + (link - 1) * row_bytes_ + column * sizeof(bfloat16_t)), count);
        DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, static_cast<bfloat16_t>(0.0F)};
        DataCopyPad(reduce_input_, source, copy, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(reduce_value_, reduce_input_, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Add(reduce_sum_, reduce_sum_, reduce_value_, count);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        const int64_t predecessor = links_[(link - 1) * kTokenStride];
        if (predecessor == 0 || predecessor < -input_rows_ - 1 || predecessor > input_rows_ + 1) {
          Trap();
        }
        link = (predecessor < 0 ? -predecessor : predecessor) - 1;
      }
      Cast(target.ReinterpretCast<bfloat16_t>()[column], reduce_sum_, RoundMode::CAST_RINT, count);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void RecordForwardGroup(int64_t row, int64_t entry) {
    const int64_t previous = static_cast<uint64_t>(entry) >> 32;
    const int32_t marker = Marker(entry);
    auto *metadata = group_metadata_ + row * kTokenStride;
    if (previous > output_rows_ || (previous == 0) != (marker < 0) ||
        (marker >= 0 && marker >= output_rows_)) {
      Trap();
    }
    // Positive link codes are carriers; a later row flips only its old carrier.
    if (previous != 0) {
      auto *prior_code = group_metadata_ + (previous - 1) * kTokenStride + 1;
      if (*prior_code < 1 || *prior_code > output_rows_ + 1 || previous == row + 1) {
        Trap();
      }
      *prior_code = -*prior_code;
    }
    metadata[1] = previous + 1;
  }

  __aicore__ inline void SendCompressed(__gm__ int64_t *edge, int32_t lane, GM_ADDR target) {
    int32_t unique = 0;
    for (int64_t row = 0; row < edge[5]; ++row) {
      const int64_t token = SourceRow(edge[3] + row);
      const int64_t primary = Seen(token);
      int64_t previous = 0;
      int32_t marker;
      if (primary == 0) {
        SetSeen(token, edge[4] + row + 1);
        marker = -unique - 1;
        ++unique;
      } else {
        previous = Last(token);
        if (previous == 0) {
          previous = primary;
        }
        if (previous >= (1LL << 31)) {
          Trap();
        }
        marker = static_cast<int32_t>(primary - 1);
        SetLast(token, edge[4] + row + 1);
      }
      header_ptr_[row + 1] = PackEntry(previous, marker);
    }
    const bool direct = static_cast<float>(unique) >
                        forward_compact_max_unique_ratio_ * static_cast<float>(edge[5]);
    header_ptr_[0] = direct ? -static_cast<int64_t>(unique) - 1 : unique;
    auto *remote_header = reinterpret_cast<__gm__ uint8_t *>(
      hyper_parallel::multicore::shmem::cann::device::remote_ptr(Header(rank_, lane), peer_));
    GlobalTensor<uint8_t> header_target;
    header_target.SetGlobalBuffer(remote_header, (edge[5] + 1) * sizeof(int64_t));
    DataCopyExtParams header_copy{1, static_cast<uint32_t>((edge[5] + 1) * sizeof(int64_t)), 0, 0, 0};
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    DataCopyPad(header_target, header_local_.ReinterpretCast<uint8_t>(), header_copy);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    if (direct) {
      WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
      CopyRows(target, edge[3], edge[5]);
      progress_[5] += edge[5];
      progress_[6] += edge[5];
      progress_[7] += edge[5];
      return;
    }
    const int32_t tile_rows = kScratchBytes / row_bytes_;
    int64_t packed_start = 0;
    for (int64_t begin = 0; begin < edge[5]; begin += tile_rows) {
      const int32_t rows = edge[5] - begin < tile_rows ? edge[5] - begin : tile_rows;
      int32_t packed = 0;
      for (int32_t row = 0; row < rows; ++row) {
        const int64_t entry = header_ptr_[begin + row + 1];
        if (Marker(entry) >= 0) {
          continue;
        }
        const int64_t token = SourceRow(edge[3] + begin + row);
        GlobalTensor<uint8_t> source;
        source.SetGlobalBuffer(input_ + token * row_bytes_, row_bytes_);
        DataCopyExtParams copy{1, static_cast<uint32_t>(row_bytes_), 0, 0, 0};
        DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
        DataCopyPad(scratch_[packed * row_bytes_], source, copy, pad);
        ++packed;
      }
      if (packed > 0) {
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        GlobalTensor<uint8_t> destination;
        destination.SetGlobalBuffer(target + packed_start * row_bytes_, packed * row_bytes_);
        DataCopyExtParams copy{1, static_cast<uint32_t>(packed * row_bytes_), 0, 0, 0};
        DataCopyPad(destination, scratch_, copy);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        packed_start += packed;
      }
    }
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    progress_[5] += unique;
    progress_[6] += edge[5];
    if (unique == edge[5]) {
      progress_[7] += edge[5];
    }
  }

  __aicore__ inline void ReceiveCompressed(__gm__ int64_t *edge, int32_t lane) {
    GlobalTensor<uint8_t> header_source;
    header_source.SetGlobalBuffer(Header(peer_, lane), (edge[5] + 1) * sizeof(int64_t));
    DataCopyExtParams header_copy{1, static_cast<uint32_t>((edge[5] + 1) * sizeof(int64_t)), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
    DataCopyPad(header_local_.ReinterpretCast<uint8_t>(), header_source, header_copy, pad);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    const int64_t encoded = header_ptr_[0];
    if (encoded < -edge[5] - 1 || encoded > edge[5]) {
      Trap();
    }
    const bool direct = encoded < 0;
    const int64_t unique = direct ? -encoded - 1 : encoded;
    peer_unique_ += unique;
    peer_rows_ += edge[5];
    if (direct) {
      for (int64_t row = 0; row < edge[5]; ++row) {
        RecordForwardGroup(edge[4] + row, header_ptr_[row + 1]);
      }
      CopyLocal(output_ + edge[4] * row_bytes_, Ring(peer_, lane), edge[5] * row_bytes_);
      return;
    }
    // Lane 1 records the reverse group link for every remote row.
    if (unique == edge[5]) {
      // This block contains no references to earlier rows. Keep the token
      // ownership map, then move its contiguous payload in full-size chunks.
      for (int64_t row = 0; row < edge[5]; ++row) {
        const int64_t entry = header_ptr_[row + 1];
        if (Marker(entry) != -row - 1) {
          Trap();
        }
        RecordForwardGroup(edge[4] + row, entry);
      }
      CopyLocal(output_ + edge[4] * row_bytes_, Ring(peer_, lane), edge[5] * row_bytes_);
      return;
    }
    const int32_t tile_rows = kScratchBytes / row_bytes_;
    for (int64_t begin = 0; begin < edge[5]; begin += tile_rows) {
      const int32_t rows = edge[5] - begin < tile_rows ? edge[5] - begin : tile_rows;
      for (int32_t row = 0; row < rows;) {
        const int64_t entry = header_ptr_[begin + row + 1];
        int64_t marker = Marker(entry);
        const int64_t target_row = edge[4] + begin + row;
        RecordForwardGroup(target_row, entry);
        GM_ADDR source;
        int32_t run = 1;
        if (marker >= 0 && marker >= edge[4] && marker < edge[4] + edge[5]) {
          marker = Marker(header_ptr_[marker - edge[4] + 1]);
          if (marker >= 0) {
            Trap();
          }
        } else if (marker >= 0) {
          if (marker >= output_rows_) {
            Trap();
          }
          source = output_ + marker * row_bytes_;
        }
        if (marker < 0) {
          const int64_t compact = -marker - 1;
          if (compact >= unique) {
            Trap();
          }
          if (Marker(entry) < 0) {
            while (row + run < rows) {
              const int64_t following = header_ptr_[begin + row + run + 1];
              const int32_t next_marker = Marker(following);
              if (next_marker >= 0 || -static_cast<int64_t>(next_marker) - 1 != compact + run) {
                break;
              }
              RecordForwardGroup(target_row + run, following);
              ++run;
            }
            if (compact + run > unique) {
              Trap();
            }
          }
          source = Ring(peer_, lane) + compact * row_bytes_;
        }
        GlobalTensor<uint8_t> input;
        input.SetGlobalBuffer(source, run * row_bytes_);
        DataCopyExtParams copy{1, static_cast<uint32_t>(run * row_bytes_), 0, 0, 0};
        DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
        DataCopyPad(scratch_[row * row_bytes_], input, copy, pad);
        row += run;
      }
      SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
      GlobalTensor<uint8_t> destination;
      destination.SetGlobalBuffer(output_ + (edge[4] + begin) * row_bytes_, rows * row_bytes_);
      DataCopyExtParams copy{1, static_cast<uint32_t>(rows * row_bytes_), 0, 0, 0};
      DataCopyPad(destination, scratch_, copy);
      SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
      WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
    }
  }

  __aicore__ inline bool SendReduced(__gm__ int64_t *edge, int32_t lane, GM_ADDR target) {
    if (dense_return_) {
      CopyRows(target, edge[3], edge[5]);
      progress_[5] += edge[5];
      progress_[6] += edge[5];
      progress_[7] += edge[5];
      return true;
    }
    int32_t unique = 0;
    bool all_singletons = true;
    for (int64_t row = 0; row < edge[5]; ++row) {
      const int64_t source_row = edge[3] + row;
      const int64_t predecessor = links_[source_row * kTokenStride];
      if (predecessor == 0 || predecessor < -input_rows_ - 1 || predecessor > input_rows_ + 1) {
        Trap();
      }
      all_singletons = all_singletons && predecessor == 1;
      const bool carrier = predecessor > 0;
      header_ptr_[row + 1] = PackEntry(0, carrier ? -++unique : 0);
    }
    header_ptr_[0] = unique;
    if (all_singletons) {
      // Exactly one destination row per original row: no reduction or header
      // is needed. A negative ready sequence selects this identity branch.
      CopyRows(target, edge[3], edge[5]);
      progress_[5] += edge[5];
      progress_[6] += edge[5];
      progress_[7] += edge[5];
      return true;
    }
    auto *remote_header = reinterpret_cast<__gm__ uint8_t *>(
      hyper_parallel::multicore::shmem::cann::device::remote_ptr(Header(rank_, lane), peer_));
    GlobalTensor<uint8_t> header_target;
    header_target.SetGlobalBuffer(remote_header, (edge[5] + 1) * sizeof(int64_t));
    DataCopyExtParams header_copy{1, static_cast<uint32_t>((edge[5] + 1) * sizeof(int64_t)), 0, 0, 0};
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    DataCopyPad(header_target, header_local_.ReinterpretCast<uint8_t>(), header_copy);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    const int32_t tile_rows = kScratchBytes / row_bytes_;
    int64_t packed_start = 0;
    for (int64_t begin = 0; begin < edge[5]; begin += tile_rows) {
      const int32_t rows = edge[5] - begin < tile_rows ? edge[5] - begin : tile_rows;
      int32_t packed = 0;
      for (int32_t row = 0; row < rows;) {
        const int64_t entry = header_ptr_[begin + row + 1];
        if (Marker(entry) >= 0) {
          ++row;
          continue;
        }
        const int64_t source_row = edge[3] + begin + row;
        if (links_[source_row * kTokenStride] == 1) {
          int32_t run = 1;
          while (row + run < rows && Marker(header_ptr_[begin + row + run + 1]) < 0 &&
                 links_[(source_row + run) * kTokenStride] == 1) {
            ++run;
          }
          GlobalTensor<uint8_t> source;
          source.SetGlobalBuffer(input_ + source_row * row_bytes_, run * row_bytes_);
          DataCopyExtParams copy{1, static_cast<uint32_t>(run * row_bytes_), 0, 0, 0};
          DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
          DataCopyPad(scratch_[packed * row_bytes_], source, copy, pad);
          packed += run;
          row += run;
        } else {
          ReduceTokenRow(source_row, scratch_[packed * row_bytes_]);
          ++packed;
          ++row;
        }
      }
      if (packed > 0) {
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        GlobalTensor<uint8_t> destination;
        destination.SetGlobalBuffer(target + packed_start * row_bytes_, packed * row_bytes_);
        DataCopyExtParams copy{1, static_cast<uint32_t>(packed * row_bytes_), 0, 0, 0};
        DataCopyPad(destination, scratch_, copy);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        packed_start += packed;
      }
    }
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    progress_[5] += unique;
    progress_[6] += edge[5];
    if (unique == edge[5]) {
      progress_[7] += edge[5];
    }
    return false;
  }

  __aicore__ inline void ReceiveReduced(__gm__ int64_t *edge, int32_t lane) {
    GlobalTensor<uint8_t> header_source;
    header_source.SetGlobalBuffer(Header(peer_, lane), (edge[5] + 1) * sizeof(int64_t));
    DataCopyExtParams header_copy{1, static_cast<uint32_t>((edge[5] + 1) * sizeof(int64_t)), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
    DataCopyPad(header_local_.ReinterpretCast<uint8_t>(), header_source, header_copy, pad);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    const int64_t unique = header_ptr_[0];
    if (unique < 0 || unique > edge[5]) {
      Trap();
    }
    if (unique == edge[5]) {
      // Every original row is a return carrier, so compact payload order is
      // already the original block order even when a carrier summed replicas.
      for (int64_t row = 0; row < edge[5]; ++row) {
        if (Marker(header_ptr_[row + 1]) != -row - 1) {
          Trap();
        }
      }
      CopyLocal(output_ + edge[4] * row_bytes_, Ring(peer_, lane), edge[5] * row_bytes_);
      return;
    }
    const int32_t tile_rows = kScratchBytes / row_bytes_;
    for (int64_t begin = 0; begin < edge[5]; begin += tile_rows) {
      const int32_t rows = edge[5] - begin < tile_rows ? edge[5] - begin : tile_rows;
      Duplicate(scratch_.ReinterpretCast<bfloat16_t>(), static_cast<bfloat16_t>(0.0F),
                static_cast<uint32_t>(rows * row_bytes_ / sizeof(bfloat16_t)));
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
      for (int32_t row = 0; row < rows;) {
        const int32_t marker = Marker(header_ptr_[begin + row + 1]);
        if (marker < 0) {
          const int64_t compact = -static_cast<int64_t>(marker) - 1;
          if (compact >= unique) {
            Trap();
          }
          int32_t run = 1;
          while (row + run < rows) {
            const int32_t following = Marker(header_ptr_[begin + row + run + 1]);
            if (following >= 0 || -static_cast<int64_t>(following) - 1 != compact + run) {
              break;
            }
            ++run;
          }
          if (compact + run > unique) {
            Trap();
          }
          GlobalTensor<uint8_t> source;
          source.SetGlobalBuffer(Ring(peer_, lane) + compact * row_bytes_, run * row_bytes_);
          DataCopyExtParams copy{1, static_cast<uint32_t>(run * row_bytes_), 0, 0, 0};
          DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
          DataCopyPad(scratch_[row * row_bytes_], source, copy, pad);
          row += run;
        } else if (marker != 0) {
          Trap();
        } else {
          ++row;
        }
      }
      SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
      // An all-duplicate tile has no MTE2 producer; its zero rows come from V.
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      GlobalTensor<uint8_t> destination;
      destination.SetGlobalBuffer(output_ + (edge[4] + begin) * row_bytes_, rows * row_bytes_);
      DataCopyExtParams copy{1, static_cast<uint32_t>(rows * row_bytes_), 0, 0, 0};
      DataCopyPad(destination, scratch_, copy);
      SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
      WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
    }
  }

  __aicore__ inline void Send() {
    auto *edge = sends_ + send_cursor_ * 6;
    CheckEdge(edge, true, false);
    const int32_t lane = (send_sequence_ - 1) % lanes_;
    // MTE reachability was checked once on entry. Gather a UB batch and perform
    // one remote store; contiguous reverse traffic uses a full-size copy.
    auto *target = reinterpret_cast<__gm__ uint8_t *>(
      hyper_parallel::multicore::shmem::cann::device::remote_ptr(Ring(rank_, lane), peer_));
    bool direct = false;
    if (dedup_state_ != nullptr) {
      SendCompressed(edge, lane, target);
    } else if (reduce_) {
      direct = SendReduced(edge, lane, target);
    } else {
      CopyRows(target, edge[3], edge[5]);
    }
    communication::mte_rma::quiet();
    communication::signal_set(Signal(0, rank_, lane), direct ? -send_sequence_ : send_sequence_, peer_);
    ++send_cursor_;
    ++send_sequence_;
  }

  __aicore__ inline void Receive() {
    auto *edge = receives_ + receive_cursor_ * 6;
    CheckEdge(edge, false, true);
    const int32_t lane = (receive_sequence_ - 1) % lanes_;
    if (dedup_state_ != nullptr) {
      ReceiveCompressed(edge, lane);
      if (receive_cursor_ + 1 == receive_end_) {
        auto *summary = group_metadata_ + receives_[receive_begin_ * 6 + 4] * kTokenStride;
        summary[2] = peer_unique_;
        summary[3] = peer_rows_;
      }
    } else if (reduce_ &&
               aclshmem_int32_test(Signal(0, peer_, lane), ACLSHMEM_CMP_EQ, -receive_sequence_) != 0) {
      CopyLocal(output_ + edge[4] * row_bytes_, Ring(peer_, lane), edge[5] * row_bytes_);
    } else if (reduce_) {
      ReceiveReduced(edge, lane);
    } else {
      CopyLocal(output_ + edge[4] * row_bytes_, Ring(peer_, lane), edge[5] * row_bytes_);
    }
    // Credit is a scalar store. Complete the MTE reads before permitting the
    // sender to overwrite this lane; an MTE3-to-MTE2 event alone does not gate S.
    CompleteOutput(edge, receive_cursor_ + 1 == receive_end_ || receives_[(receive_cursor_ + 1) * 6 + 2] != edge[2]);
    communication::signal_set(Signal(1, rank_, lane), receive_sequence_ + lanes_, peer_);
    ++receive_cursor_;
    ++receive_sequence_;
  }

  __aicore__ inline void CheckEdge(__gm__ int64_t *edge, bool sends, bool receives) {
    const int64_t source_extent = gather_ ? mapping_rows_ : input_rows_;
    if (edge[5] < 1 || edge[5] > block_rows_ || edge[3] < 0 || edge[4] < 0 ||
        (sends && (edge[0] != rank_ || edge[1] != peer_ || edge[3] > source_extent - edge[5])) ||
        (receives && (edge[0] != peer_ || edge[1] != rank_ || edge[4] > output_rows_ - edge[5]))) {
      Trap();
    }
  }

  __aicore__ inline int64_t SourceRow(int64_t selection) {
    const int64_t row = gather_ ? mapping_[selection] : selection;
    if (row < 0 || row >= input_rows_) {
      Trap();
    }
    return row;
  }

  TPipe pipe_;
  TBuf<TPosition::VECOUT> buffer_;
  TBuf<TPosition::VECOUT> header_buffer_;
  TBuf<TPosition::VECCALC> seen_buffer_, last_buffer_;
  TBuf<TPosition::VECIN> reduce_input_buffer_;
  TBuf<TPosition::VECCALC> reduce_value_buffer_, reduce_sum_buffer_;
  LocalTensor<uint8_t> scratch_;
  LocalTensor<int64_t> header_local_;
  LocalTensor<int32_t> seen_local_, last_local_;
  __ubuf__ int64_t *header_ptr_ = nullptr;
  __ubuf__ int32_t *seen_ptr_ = nullptr, *last_ptr_ = nullptr;
  LocalTensor<bfloat16_t> reduce_input_;
  LocalTensor<float> reduce_value_, reduce_sum_;
  __ubuf__ uint8_t *scratch_ptr_ = nullptr;
  GM_ADDR input_ = nullptr;
  GM_ADDR output_ = nullptr;
  GM_ADDR ring_ = nullptr;
  __gm__ int64_t *headers_ = nullptr, *dedup_state_ = nullptr, *group_metadata_ = nullptr;
  __gm__ int64_t *links_ = nullptr;
  __gm__ int64_t *seen_ = nullptr, *last_ = nullptr;
  __gm__ int64_t *mapping_ = nullptr, *sends_ = nullptr, *receives_ = nullptr, *progress_ = nullptr;
  __gm__ int32_t *signals_ = nullptr;
  __gm__ int32_t *input_ready_ = nullptr, *input_expected_ = nullptr, *output_ready_ = nullptr;
  __gm__ int64_t *expert_slots_ = nullptr;
  int32_t producers_ = 0, active_experts_ = 0, slot_count_ = 0;
  int32_t admitted_input_slot_ = -1, pending_output_blocks_ = 0;
  int64_t send_cursor_ = 0, send_end_ = 0, receive_cursor_ = 0, receive_end_ = 0, receive_begin_ = 0;
  int64_t peer_unique_ = 0, peer_rows_ = 0;
  int32_t send_sequence_ = 0, receive_sequence_ = 0;
  int32_t rank_ = 0, ranks_ = 0, peer_ = 0, lanes_ = 0, block_rows_ = 0, row_bytes_ = 0;
  bool gather_ = false, reduce_ = false, publish_each_block_ = false, dense_return_ = false;
  float reduce_dense_ratio_ = 1.0F;
  float forward_compact_max_unique_ratio_ = 1.0F;
  bool split_workers_ = false, send_worker_ = false;
  uint64_t idle_cycles_ = 0;
  int64_t input_rows_ = 0, output_rows_ = 0, mapping_rows_ = 0;
};

// Peer workers must enter as one batch: a resident subset can wait for peers
// whose workers are held behind another stream's dependent communication.
__schedmode__(1) __global__ __aicore__ void TransportKernel(
                                          GM_ADDR input, GM_ADDR output, GM_ADDR mapping, GM_ADDR sends,
                                          GM_ADDR receives, GM_ADDR offsets, GM_ADDR ring, GM_ADDR signals,
                                          GM_ADDR progress, int32_t rank, int32_t ranks, int32_t lanes,
                                          int32_t block_rows, int32_t row_bytes, bool gather, uint64_t idle_cycles,
                                          int64_t input_rows, int64_t output_rows, int64_t mapping_rows,
                                          int64_t send_count, int64_t receive_count, GM_ADDR input_ready,
                                          GM_ADDR input_expected, GM_ADDR output_ready, GM_ADDR expert_slots,
                                          int32_t producers, int32_t active_experts, int32_t slot_count,
                                          GM_ADDR headers, GM_ADDR dedup_state, GM_ADDR group_metadata,
                                          bool reduce, bool publish_each_block, float reduce_dense_ratio,
                                          float forward_compact_max_unique_ratio) {
  TransportWorker worker;
  worker.Init(input, output, mapping, sends, receives, offsets, ring, signals, progress,
              rank, ranks, lanes, block_rows, row_bytes, gather, idle_cycles,
              input_rows, output_rows, mapping_rows, send_count, receive_count, input_ready, input_expected,
              output_ready, expert_slots, producers, active_experts, slot_count,
              headers, dedup_state, group_metadata, reduce, publish_each_block, reduce_dense_ratio,
              forward_compact_max_unique_ratio);
  worker.Run();
}

}  // namespace

extern "C" int RoutedExpertsTransport(
  void *stream, void *input, void *output, void *mapping, void *send_edges, void *receive_edges,
  void *peer_offsets, void *ring, void *signals, void *progress, int32_t rank, int32_t ranks,
  int32_t lanes, int32_t block_rows, int32_t row_bytes, bool gather, uint64_t idle_cycles,
  int64_t input_rows, int64_t output_rows, int64_t mapping_rows, int64_t send_count, int64_t receive_count,
  void *input_ready, void *input_expected, void *output_ready, void *expert_slots,
  int32_t producers, int32_t active_experts, int32_t slot_count,
  void *headers, void *dedup_state, void *group_metadata, bool reduce, bool publish_each_block,
  float reduce_dense_ratio, float forward_compact_max_unique_ratio) {
  if (ranks < 1 || ranks > 48) {
    return static_cast<int>(ACL_ERROR_INVALID_PARAM);
  }
  const auto previous_error = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
  if (previous_error != ACL_SUCCESS) {
    return static_cast<int>(previous_error);
  }
  const int32_t workers = ranks <= 24 ? 2 * ranks : ranks;
  TransportKernel<<<workers, nullptr, stream>>>(static_cast<uint8_t *>(input), static_cast<uint8_t *>(output),
    static_cast<uint8_t *>(mapping), static_cast<uint8_t *>(send_edges), static_cast<uint8_t *>(receive_edges),
    static_cast<uint8_t *>(peer_offsets), static_cast<uint8_t *>(ring), static_cast<uint8_t *>(signals),
    static_cast<uint8_t *>(progress), rank, ranks, lanes, block_rows, row_bytes, gather, idle_cycles,
    input_rows, output_rows, mapping_rows, send_count, receive_count, static_cast<uint8_t *>(input_ready),
    static_cast<uint8_t *>(input_expected), static_cast<uint8_t *>(output_ready), static_cast<uint8_t *>(expert_slots),
    producers, active_experts, slot_count, static_cast<uint8_t *>(headers),
    static_cast<uint8_t *>(dedup_state), static_cast<uint8_t *>(group_metadata),
    reduce, publish_each_block, reduce_dense_ratio, forward_compact_max_unique_ratio);
  return static_cast<int>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
}
