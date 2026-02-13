/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef HYPER_PARALLEL_PLATFORM_MINDSPORE_CUSTOM_PASS_DUPLICATE_ON_MULTI_USERS_H_
#define HYPER_PARALLEL_PLATFORM_MINDSPORE_CUSTOM_PASS_DUPLICATE_ON_MULTI_USERS_H_

#include "ir/func_graph.h"
#include "mindspore/include/custom_pass_api.h"

namespace mindspore::opt {

/**
 * @brief DuplicatePrimOnMultiUsersPass: Implements communication-for-memory tradeoff by duplicating
 * collective communication primitives to avoid materializing large intermediate results.
 *
 * Execution sequence:
 * 1. DistCommAllGatherIntoTensor duplication (state-aware, full chain duplication)
 *    - Handles distributed communication with UMonad state management
 *    - Creates independent buffer + communication + load chains per consumer
 *
 * 2. AllGather duplication (stateless, shallow copy)
 *    - Handles standard AllGather primitives
 *    - Requires prior Depend elimination to expose true consumer structure
 *
 * Return value: Always true (best-effort transformation; failures logged but not fatal)
 */
class DuplicatePrimOnMultiUsersPass : public Pass {
 public:
  DuplicatePrimOnMultiUsersPass() : Pass("DuplicatePrimOnMultiUsersPass") {}
  ~DuplicatePrimOnMultiUsersPass() override = default;

  bool Run(const FuncGraphPtr &func_graph) override;
};
}  // namespace mindspore::opt

#endif  // HYPER_PARALLEL_PLATFORM_MINDSPORE_CUSTOM_PASS_DUPLICATE_ON_MULTI_USERS_H_
