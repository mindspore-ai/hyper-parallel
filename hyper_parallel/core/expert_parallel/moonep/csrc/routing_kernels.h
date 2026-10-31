// SPDX-License-Identifier: Apache-2.0
// MoonEP virtual-route counts and token labels.
#pragma once

namespace routing_kernels {
using namespace AscendC;

template <typename T>
__aicore__ inline void Load(LocalTensor<T> local, GM_ADDR address, int64_t count) {
    GlobalTensor<T> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(address));
    DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(local, global, copy, pad);
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);
}

template <typename T>
__aicore__ inline void Store(GM_ADDR address, LocalTensor<T> local, int64_t count) {
    GlobalTensor<T> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(address));
    SetFlag<HardEvent::S_MTE3>(0);
    WaitFlag<HardEvent::S_MTE3>(0);
    DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPad(global, local, copy);
    SetFlag<HardEvent::MTE3_S>(0);
    WaitFlag<HardEvent::MTE3_S>(0);
}

__global__ __aicore__ void Render(GM_ADDR input, GM_ADDR boundaries, GM_ADDR placements,
                                GM_ADDR counts_out, GM_ADDR repeats_out, GM_ADDR labels_out,
                                int32_t p, int32_t e, int32_t rank, int32_t width,
                                int32_t source_group, int32_t bytes) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, bytes);
    auto memory = buffer.Get<int32_t>();
    int32_t pos = 0;
    auto c = memory[pos]; pos += device_aligned(p * e);
    auto cumulative = memory[pos]; pos += device_aligned(p * e);
    auto copies = memory[pos]; pos += device_aligned(e);
    auto lookup = memory[pos]; pos += device_aligned(p * e);
    auto row = memory[pos]; pos += device_aligned(p * width);
    auto repeats = memory[pos]; pos += device_aligned(p * e);
    auto labels = memory[pos];
    Load(c, input, p * e);
    Load(cumulative, boundaries, p * e);
    Load(copies, placements, e);
    const int32_t local = e / p;
    for (int32_t i = 0; i < p * e; ++i) lookup.SetValue(i, -1);
    for (int32_t dest = 0; dest < p; ++dest) {
        for (int32_t j = 0; j < local; ++j) lookup.SetValue(dest * e + dest * local + j, j);
        for (int32_t j = 0; j < width - local; ++j) {
            int32_t expert = copies.GetValue(dest * local + j);
            if (expert >= 0) lookup.SetValue(dest * e + expert, local + j);
        }
    }
    const int32_t begin = GetBlockIdx() * source_group;
    for (int32_t source = begin; source < p && source < begin + source_group; ++source) {
        for (int32_t i = 0; i < p * width; ++i) row.SetValue(i, 0);
        for (int32_t expert = 0; expert < e; ++expert) {
            int32_t start = 0;
            for (int32_t s = 0; s < source; ++s) start += c.GetValue(s * e + expert);
            int32_t end = start + c.GetValue(source * e + expert), before = 0;
            for (int32_t dest = 0; dest < p; ++dest) {
                int32_t after = cumulative.GetValue(expert * p + dest);
                int32_t left = start > before ? start : before;
                int32_t right = end < after ? end : after;
                int32_t n = right > left ? right - left : 0;
                int32_t slot = lookup.GetValue(dest * e + expert);
                if (slot >= 0) row.SetValue(dest * width + slot, n);
                if (source == rank) {
                    repeats.SetValue(expert * p + dest, n);
                    labels.SetValue(expert * p + dest, n ? dest * width + slot : 0);
                }
                before = after;
            }
        }
        Store(counts_out + source * p * width * 4, row, p * width);
        if (source == rank) {
            Store(repeats_out, repeats, p * e);
            Store(labels_out, labels, p * e);
        }
    }
}

std::vector<torch::Tensor> render(torch::Tensor gathered, torch::Tensor cumulative,
                                torch::Tensor copies, int64_t rank, int64_t width) {
    const int64_t p = gathered.size(0), e = gathered.size(1);
    auto counts = torch::empty({p, p * width}, gathered.options());
    auto repeats = torch::empty({e * p}, gathered.options());
    auto labels = torch::empty({e * p}, gathered.options());
    int64_t group = 1;
    while ((group * p * width) % 32 != 0) ++group;
    const int64_t bytes = (5 * aligned(p * e) + aligned(e) + aligned(p * width)) * 4;
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto run = [=]() -> int {
        Render<<<(p + group - 1) / group, nullptr, stream>>>((GM_ADDR)gathered.data_ptr(),
            (GM_ADDR)cumulative.data_ptr(), (GM_ADDR)copies.data_ptr(), (GM_ADDR)counts.data_ptr(),
            (GM_ADDR)repeats.data_ptr(), (GM_ADDR)labels.data_ptr(), p, e, rank, width, group, bytes);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("MoonEPVirtualCounts", run);
    return {counts, repeats, labels};
}

}  // namespace routing_kernels
