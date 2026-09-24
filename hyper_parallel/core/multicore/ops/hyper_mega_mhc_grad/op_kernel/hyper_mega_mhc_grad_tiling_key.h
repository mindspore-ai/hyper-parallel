/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef HYPER_MEGA_MHC_GRAD_TILING_KEY_H_
#define HYPER_MEGA_MHC_GRAD_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"
#include "mhc_pre_sinkhorn_backward/arch22/mhc_pre_sinkhorn_backward_data_arch22.h"

#define HYPER_MEGA_MHC_GRAD_DEFAULT 1

ASCENDC_TPL_ARGS_DECL(
    HyperMegaMhcGrad,
    ASCENDC_TPL_UINT_DECL(
        MHC_GRAD_MODE,
        ASCENDC_TPL_2_BW,
        ASCENDC_TPL_UI_LIST,
        HYPER_MEGA_MHC_GRAD_DEFAULT));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
        ASCENDC_TPL_UINT_SEL(
            MHC_GRAD_MODE,
            ASCENDC_TPL_UI_LIST,
            HYPER_MEGA_MHC_GRAD_DEFAULT),
        ASCENDC_TPL_TILING_STRUCT_SEL(MhcPreSinkhornBackwardArch22TilingData)));

#endif  // HYPER_MEGA_MHC_GRAD_TILING_KEY_H_
