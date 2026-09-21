/*
 * Copyright 2026 Huawei Technologies Co., Ltd.
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
#ifndef HYPER_PARALLEL_MULTICORE_CACHED_OP_API_H_
#define HYPER_PARALLEL_MULTICORE_CACHED_OP_API_H_

#include "op_plugin/include/npu_cpp_extension.h"

namespace hyper_parallel::multicore {

// Construct one static instance per operator; never cache by argument signature alone.
struct CachedOpApi {
  using InitMemory = int (*)(void *, bool);
  using MemoryCallback = void (*)(void *, bool);
  using CacheCallback = void (*)();

  CachedOpApi(const char *api_name, const char *workspace_name) : name(api_name) {
    GetApiFunc(name, workspace_name, execute, workspace);
  }

  void init_context() const {
    if (init_memory != nullptr) {
      init_memory(nullptr, false);
    }
  }

  void release_context() const {
    if (release_memory != nullptr) {
      release_memory(nullptr, false);
    }
  }

  void uninit_context() const {
    if (uninit_memory != nullptr) {
      uninit_memory(nullptr, false);
    }
    if (uninit_cache != nullptr) {
      uninit_cache();
    }
  }

  const char *name;
  void *execute = nullptr;
  void *workspace = nullptr;
  InitMemory init_memory = reinterpret_cast<InitMemory>(GetOpApiFuncAddr("InitHugeMemThreadLocal"));
  MemoryCallback release_memory = reinterpret_cast<MemoryCallback>(GetOpApiFuncAddr("ReleaseHugeMem"));
  MemoryCallback uninit_memory = reinterpret_cast<MemoryCallback>(GetOpApiFuncAddr("UnInitHugeMemThreadLocal"));
  CacheCallback uninit_cache = reinterpret_cast<CacheCallback>(GetOpApiFuncAddr("UnInitPTACacheThreadLocal"));
};

template <typename... Args>
void execute_cached_op(const CachedOpApi &api, Args &...args) {
  // Cache only process-lifetime symbols, using the same resolver as the extension bridge.
  // Workspace, executor, converted parameters and contexts remain invocation/thread-local.
  static const auto task_queue_enable = OpApiGetTaskQueueEnable();
  auto stream = GetAclStream();
  if (task_queue_enable == 2) {
    auto copied_params = CopyTypesV2(args...);
    auto cache_params = GetCacheParams();
    auto acl_call = [api, copied_params, stream, cache_params]() -> int {
      InitExecSubTheadCtx(stream);
      int status = 0;
      if (hit_cache_v2_ext(stream, api.name, api.execute, copied_params, &status, cache_params)) {
        return status;
      }
      SetExecConfigV2(cache_params);
      api.init_context();
      uint64_t workspace_size = 0;
      aclOpExecutor *executor = nullptr;
      auto converted_params = ConvertTypesV2(copied_params, &workspace_size, &executor);
      auto workspace_func = ConvertToOpApiFunc(converted_params, api.workspace);
      auto workspace_status = call(workspace_func, converted_params);
      TORCH_CHECK(workspace_status == 0, api.name, " workspace query failed: ", workspace_status);
      status = ExecuteApiFuncV2(api.execute, stream, workspace_size, executor);
      ReleaseConvertTypes(converted_params);
      api.release_context();
      api.uninit_context();
      return status;
    };
    RunAclCall(api.name, acl_call);
    return;
  }
  if (hit_cache_ext(stream, api.name, api.execute, args...)) {
    return;
  }
  SetExecConfig();
  api.init_context();
  uint64_t workspace_size = 0;
  aclOpExecutor *executor = nullptr;
  auto *workspace_size_addr = &workspace_size;
  auto **executor_addr = &executor;
  auto converted_params = ConvertTypes(args..., workspace_size_addr, executor_addr);
  auto workspace_func = ConvertToOpApiFunc(converted_params, api.workspace);
  auto workspace_status = call(workspace_func, converted_params);
  TORCH_CHECK(workspace_status == 0, api.name, " workspace query failed: ", workspace_status);
  void *workspace_addr = GetWorkSpaceAddr(workspace_size);
  auto acl_call = [api, converted_params, workspace_addr, workspace_size, stream, executor]() -> int {
    InitExecSubTheadCtx(stream);
    auto status = ExecuteApiFunc(api.execute, stream, workspace_addr, workspace_size, executor);
    ReleaseConvertTypes(converted_params);
    api.release_context();
    return status;
  };
  RunAclCall(api.name, acl_call);
  api.uninit_context();
}

}  // namespace hyper_parallel::multicore

#endif  // HYPER_PARALLEL_MULTICORE_CACHED_OP_API_H_
