/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef HYPER_MEGA_MHC_TILING_UTIL_H_
#define HYPER_MEGA_MHC_TILING_UTIL_H_

#include "register/op_impl_registry.h"

namespace Ops {
namespace Transformer {
namespace OpTiling {

// HyperMegaMhc initially targets Ascend 910B/910_93 (membase).  The upstream
// mHC dispatcher only uses this helper to select its separate Ascend 950 path.
inline bool IsRegbaseSocVersion(const gert::TilingContext *) { return false; }

inline bool IsRegbaseSocVersion(const gert::TilingParseContext *) { return false; }

}  // namespace OpTiling
}  // namespace Transformer
}  // namespace Ops
#endif  // HYPER_MEGA_MHC_TILING_UTIL_H_
