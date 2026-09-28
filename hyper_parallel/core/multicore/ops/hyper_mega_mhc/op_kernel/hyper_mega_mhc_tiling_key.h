/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

/*! \file hyper_mega_mhc_tiling_key.h */
#ifndef HYPER_MEGA_MHC_TILING_KEY_H
#define HYPER_MEGA_MHC_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// This is a private dispatch key; the native MhcPreSinkhorn tiling data still
// describes the token/M split. A non-zero key avoids the default AIV path.
#define HYPER_MEGA_MHC_M_SPLIT 1

ASCENDC_TPL_ARGS_DECL(HyperMegaMhc, ASCENDC_TPL_UINT_DECL(MHC_SPLIT_MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                                          HYPER_MEGA_MHC_M_SPLIT));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                                     ASCENDC_TPL_UINT_SEL(MHC_SPLIT_MODE, ASCENDC_TPL_UI_LIST,
                                                          HYPER_MEGA_MHC_M_SPLIT)));

#endif  // HYPER_MEGA_MHC_TILING_KEY_H
