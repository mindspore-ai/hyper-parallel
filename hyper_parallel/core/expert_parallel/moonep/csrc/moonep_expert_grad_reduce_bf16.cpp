// SPDX-License-Identifier: Apache-2.0
// Fuse owner-gradient copies, FP32 accumulation, and BF16 output conversion.
#include <cstdint>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "shmem.h"
#include "launch.h"

using namespace AscendC;

namespace {
constexpr int32_t kElements = 8192;
constexpr int32_t kScratch = 64;
constexpr int32_t kReservedUb = 188 * 1024;
constexpr int32_t kIndexOffset = kScratch + kElements * 10;

template <typename T>
__aicore__ inline LocalTensor<T> LocalAt(int32_t offset, TPosition position) {
    LocalTensor<T> local;
    local.address_.logicPos = static_cast<uint8_t>(position);
    local.address_.bufferAddr = offset;
    local.address_.dataLen = kElements * sizeof(T);
    return local;
}

__aicore__ inline void LoadOwner(LocalTensor<bfloat16_t> local,
                               __gm__ bfloat16_t* source, int32_t elements) {
    GlobalTensor<bfloat16_t> global;
    global.SetGlobalBuffer(source);
    DataCopyExtParams copy{1, static_cast<uint32_t>(elements * sizeof(bfloat16_t)), 0, 0, 0};
    DataCopyPadExtParams<bfloat16_t> padding{false, 0, 0, 0};
    DataCopyPad(local, global, copy, padding);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void StoreOwner(__gm__ bfloat16_t* destination,
                                LocalTensor<bfloat16_t> local, int32_t elements) {
    GlobalTensor<bfloat16_t> global;
    global.SetGlobalBuffer(destination);
    DataCopyExtParams copy{1, static_cast<uint32_t>(elements * sizeof(bfloat16_t)), 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(global, local, copy);
    PipeBarrier<PIPE_ALL>();
}
}  // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void MoonEPSharedGradReturnBf16(uint64_t ffts, GM_ADDR source_address,
                             GM_ADDR output_address, GM_ADDR slots_address,
                             GM_ADDR plan_address, int32_t experts,
                             int64_t matrix, int32_t ranks, int32_t slots,
                             int32_t rank) {
    util_set_ffts_config(ffts);
    if ASCEND_IS_AIC {
        return;
    }
    aclshmemx_barrier_all_vec();
    auto* source = reinterpret_cast<__gm__ bfloat16_t*>(source_address);
    auto* output = reinterpret_cast<__gm__ bfloat16_t*>(output_address);
    auto* remote = reinterpret_cast<__gm__ bfloat16_t*>(slots_address);
    auto* plan = reinterpret_cast<__gm__ int32_t*>(plan_address);
    auto* heads = reinterpret_cast<__ubuf__ int32_t*>(kIndexOffset);
    auto* next = heads + (experts + 7) / 8 * 8;
    const int32_t first_expert = rank * experts;
    for (int32_t expert = 0; expert < experts; ++expert) {
        heads[expert] = -1;
    }
    // Build source-ordered lists once per worker, rather than once per tile.
    for (int32_t rb = ranks * slots - 1; rb >= 0; --rb) {
        const int32_t expert = plan[rb] - first_expert;
        if (expert >= 0 && expert < experts) {
            next[rb] = heads[expert];
            heads[expert] = rb;
        }
    }
    const int32_t sub_blocks = static_cast<int32_t>(GetSubBlockNum());
    const int32_t worker = sub_blocks > 0 ? GetBlockIdx() / sub_blocks : 0;
    const int32_t workers = GetBlockNum();
    const int64_t tiles = (matrix + kElements - 1) / kElements;
    auto packed = LocalAt<bfloat16_t>(kScratch, TPosition::VECIN);
    auto packed_output = LocalAt<bfloat16_t>(kScratch, TPosition::VECOUT);
    auto accumulator = LocalAt<float>(kScratch + 2*kElements, TPosition::VECCALC);
    auto contribution = LocalAt<float>(kScratch + 6*kElements, TPosition::VECCALC);
    for (int32_t expert = 0; expert < experts; ++expert) {
        // Owned native gradients can remain in place when no peer contributes.
        // Every worker still participates in both root barriers.
        if (source_address == output_address && heads[expert] < 0) {
            continue;
        }
        for (int64_t tile = worker; tile < tiles; tile += workers) {
            const int64_t begin = tile * kElements;
            const int32_t elements = matrix - begin < kElements ? matrix - begin : kElements;
            const int64_t offset = static_cast<int64_t>(expert) * matrix + begin;
            LoadOwner(packed, source + offset, elements);
            if (heads[expert] >= 0) {
                Cast(accumulator, packed, RoundMode::CAST_NONE, elements);
                PipeBarrier<PIPE_ALL>();
                for (int32_t rb = heads[expert]; rb >= 0; rb = next[rb]) {
                    const int32_t peer = rb / slots;
                    const int32_t slot = rb % slots;
                    const int64_t peer_offset = static_cast<int64_t>(slot) * matrix + begin;
                    aclshmemx_mte_get_nbi(
                        reinterpret_cast<__ubuf__ bfloat16_t*>(kScratch),
                        remote + peer_offset, elements, peer, EVENT_ID0);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    Cast(contribution, packed, RoundMode::CAST_NONE, elements);
                    PipeBarrier<PIPE_V>();
                    Add(accumulator, accumulator, contribution, elements);
                    PipeBarrier<PIPE_ALL>();
                }
                Cast(packed_output, accumulator, RoundMode::CAST_RINT, elements);
                PipeBarrier<PIPE_ALL>();
            }
            StoreOwner(output + offset, packed_output, elements);
        }
    }
    aclshmem_quiet();
    // All readers must complete before a later backward overwrites the slots.
    aclshmemx_barrier_all_vec();
}

extern "C" ASCEND_MOONEP_EXPORT int hp_moonep_grad_return_bf16(
    void* stream, const void* source, void* output, void* slots,
    const int32_t* plan, int32_t experts, int32_t hidden, int32_t projection,
    int32_t ranks, int32_t rank, int32_t blocks) {
    const int32_t index_entries = (experts + 7) / 8 * 8 + ranks * experts;
    if (stream == nullptr || source == nullptr || output == nullptr ||
        slots == nullptr || plan == nullptr || experts <= 0 || hidden <= 0 ||
        projection <= 0 || ranks != aclshmem_n_pes() || rank < 0 ||
        rank >= ranks || blocks <= 0 ||
        kIndexOffset + index_entries * sizeof(int32_t) > kReservedUb) {
        return ACL_ERROR_INVALID_PARAM;
    }
    MoonEPSharedGradReturnBf16<<<blocks, nullptr, stream>>>(
        shmemx_get_ffts_config(),
        reinterpret_cast<uint8_t*>(const_cast<void*>(source)),
        reinterpret_cast<uint8_t*>(output), reinterpret_cast<uint8_t*>(slots),
        reinterpret_cast<uint8_t*>(const_cast<int32_t*>(plan)), experts,
        static_cast<int64_t>(hidden) * projection, ranks, experts, rank);
    return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
}
