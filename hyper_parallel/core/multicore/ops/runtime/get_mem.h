/**
 * Copyright 2026 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef HYPER_PARALLEL_MULTICORE_GET_MEM_H
#define HYPER_PARALLEL_MULTICORE_GET_MEM_H

#include "kernel_operator.h"
#include "data_plane/rma.h"

template <typename T>
__aicore__ inline void PullToLocal(GM_ADDR destination, GM_ADDR source, int64_t elements, int source_pe) {
  hyper_parallel::multicore::shmem::data_plane::get_pipelined<T>(destination, source, elements, source_pe);
}

#endif  // HYPER_PARALLEL_MULTICORE_GET_MEM_H
