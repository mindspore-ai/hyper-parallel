// SPDX-License-Identifier: Apache-2.0
// Bind expert transfers to the same SHMEM library used by MegaMoE.
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "shmem.h"
#include "launch.h"

extern "C" int hp_moonep_grad_return_bf16(
    void* stream, const void* source, void* output, void* slots,
    const int32_t* plan, int32_t experts, int32_t hidden, int32_t projection,
    int32_t ranks, int32_t rank, int32_t blocks);

static void check_runtime(int64_t rank, int64_t ranks) {
    TORCH_CHECK(rank == aclshmem_my_pe(), "expert-transfer rank differs from active MegaMoE SHMEM PE");
    TORCH_CHECK(ranks == aclshmem_n_pes(), "expert-transfer world differs from active MegaMoE SHMEM root");
}

static void prefetch(torch::Tensor home, torch::Tensor slots, torch::Tensor copies,
                     int64_t rank, int64_t ranks, int64_t blocks) {
    check_runtime(rank, ranks);
    TORCH_CHECK(home.is_contiguous() && slots.is_contiguous() && copies.is_contiguous(), "expert transfer requires contiguous tensors");
    TORCH_CHECK(home.scalar_type() == at::kBFloat16 && slots.scalar_type() == at::kBFloat16, "prefetch weights must be BF16");
    TORCH_CHECK(copies.scalar_type() == at::kInt && copies.dim() == 1, "prefetch copies must be INT32 slots");
    TORCH_CHECK(home.dim() == 3 && slots.dim() == 3 && home.size(0) == 2*slots.size(0), "prefetch home/slot geometry changed");
    TORCH_CHECK(home.size(1) == slots.size(1) && home.size(2) == slots.size(2) && copies.numel() == slots.size(0), "prefetch projection shape changed");
    // Match prefetch_weight_kernel's reserved SHMEM region and double-buffer
    // geometry. Keep this guard on the host so unsupported shapes fail before
    // the kernel's collective barriers, rather than silently skipping a copy.
    constexpr int64_t ub_limit_bytes = 188 * 1024;
    constexpr int64_t scratch_offset = 64;
    constexpr int64_t alignment = 32;
    constexpr int64_t buffers = 2;
    const int64_t slot_count = slots.size(0);
    TORCH_CHECK(slot_count > 0 && slot_count <= (ub_limit_bytes - scratch_offset) / (2 * sizeof(int32_t)),
                "MoonEP prefetch UB budget cannot hold the expert copy table");
    const int64_t table_end = scratch_offset + 2 * slot_count * sizeof(int32_t);
    const int64_t data_offset = (table_end + alignment - 1) / alignment * alignment;
    const int64_t remaining_bytes = ub_limit_bytes - data_offset;
    TORCH_CHECK(remaining_bytes > 0, "MoonEP prefetch UB budget has no staging space");
    const int64_t buffer_bytes = remaining_bytes / buffers / alignment * alignment;
    TORCH_CHECK(home.size(1) > 0 && home.size(2) > 0,
                "MoonEP prefetch projection dimensions must be positive");
    TORCH_CHECK(buffer_bytes / home.element_size() / home.size(2) > 0,
                "MoonEP prefetch UB budget cannot hold one BF16 projection row: width=", home.size(2),
                ", buffer_bytes=", buffer_bytes);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    // OpCommand may execute after this binding returns; own the tensor handles.
    auto run = [=]() -> int {
        // The device kernel publishes home writes and completes all peer reads.
        hp_moonep_prefetch_weight_kernel_do(stream, static_cast<uint8_t*>(home.data_ptr()),
            static_cast<uint8_t*>(slots.data_ptr()), static_cast<uint8_t*>(copies.data_ptr()),
            ranks*slots.size(0), home.size(1), home.size(2), slots.size(0), blocks);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("HyperParallelMoonEPPrefetch", run);
}

static void reduce(torch::Tensor source, torch::Tensor home, torch::Tensor slots,
                   torch::Tensor copies, int64_t rank, int64_t blocks) {
    check_runtime(rank, copies.size(0));
    TORCH_CHECK(source.is_contiguous() && home.is_contiguous() && slots.is_contiguous() && copies.is_contiguous(), "gradient transfer requires contiguous tensors");
    TORCH_CHECK(source.scalar_type() == at::kBFloat16 && home.scalar_type() == at::kBFloat16 && slots.scalar_type() == at::kBFloat16, "expert gradients must be BF16; accumulation uses FP32 inside the kernel");
    TORCH_CHECK(copies.scalar_type() == at::kInt && copies.dim() == 2, "gradient copy plan must be INT32 [root, slots]");
    TORCH_CHECK(home.sizes() == source.sizes() && home.sizes() == slots.sizes() && home.dim() == 3 && home.size(0) == copies.size(1), "gradient transfer shape changed");
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto run = [=]() -> int {
        return hp_moonep_grad_return_bf16(stream, source.data_ptr(), home.data_ptr(), slots.data_ptr(),
            copies.data_ptr<int32_t>(), home.size(0), home.size(1), home.size(2),
            copies.size(0), rank, blocks);
    };
    at_npu::native::OpCommand::RunOpApi("HyperParallelMoonEPGradReduce", run);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("prefetch", &prefetch);
    module.def("reduce", &reduce);
    module.def("runtime_state", []() { return std::vector<int>{aclshmem_my_pe(), aclshmem_n_pes()}; });
}
