// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#if defined(__GNUC__)
#define ASCEND_MOONEP_EXPORT __attribute__((visibility("default")))
#else
#define ASCEND_MOONEP_EXPORT
#endif

extern "C" ASCEND_MOONEP_EXPORT void hp_moonep_prefetch_weight_kernel_do(
    void* stream, uint8_t* home, uint8_t* slots, uint8_t* copies,
    int64_t experts, int64_t hidden, int64_t projection, int64_t local_slots,
    int64_t blocks);
