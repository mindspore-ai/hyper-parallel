/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

#include "acl/acl.h"

namespace ShmemKernel {

/**
 * @brief Launch the Root WORLD Direct-Push AllGather on one NPU Stream.
 * @param stream Non-owning current NPU Stream used for asynchronous launch.
 * @param output Symmetric output base whose flattened slots are ordered by CANN Root WORLD PE.
 * @param input Local input base contributed by root_rank.
 * @param local_bytes Input length in Byte for one Root PE.
 * @param root_rank Calling process PE in CANN Root WORLD coordinates.
 * @param root_size Number of PEs in CANN Root WORLD.
 * @return aclrtGetLastError() captured right after the Kernel launch, with the thread error state drained just before
 * the launch so a non-success result is attributable to this launch; ACL_SUCCESS only confirms Kernel enqueue.
 * @note All Root PEs must call with matching parameters. The caller owns entry and exit barriers.
 */
aclError aclshmem_all_gather(aclrtStream stream, void *output, void *input, uint64_t local_bytes, int32_t root_rank,
                             int32_t root_size);

}  // namespace ShmemKernel
