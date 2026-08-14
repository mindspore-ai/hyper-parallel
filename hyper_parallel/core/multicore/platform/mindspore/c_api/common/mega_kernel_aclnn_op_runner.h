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

#ifndef HYPER_PARALLEL_MULTICORE_MEGA_KERNEL_ACLNN_OP_RUNNER_H_
#define HYPER_PARALLEL_MULTICORE_MEGA_KERNEL_ACLNN_OP_RUNNER_H_

#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "include/kernel/ascend/custom/pyboost_impl/aclnn_op_runner.h"

namespace ms_multicore {

inline bool IsMegaKernelDryRun() {
  const char *value = std::getenv("HP_MEGA_KERNEL_DRY_RUN");
  return value != nullptr && std::string(value) == "1";
}

class MegaKernelAclnnOpRunner final : public ms::pynative::PyboostRunner {
 public:
  using RunnerFunc = std::function<void(mindspore::device::DeviceContext *, size_t)>;
  using PyboostRunner::PyboostRunner;

  void SetLaunchFunc(const RunnerFunc &func) { launch_func_ = func; }
  void SetDryRunFunc(const RunnerFunc &func) { dryrun_func_ = func; }
  void LaunchKernel() override {}

 protected:
  void _Run() override {
    _PrepareStream();
    _PrepareDeviceAddress();
    auto runner = std::static_pointer_cast<MegaKernelAclnnOpRunner>(shared_from_this());
    mindspore::kernel::pyboost::PyBoostUtils::DispatchRun(
      std::make_shared<mindspore::runtime::PyBoostDeviceTask>([runner]() {
        runner->_MallocDeviceAddress();
        if (IsMegaKernelDryRun()) {
          MS_EXCEPTION_IF_NULL(runner->dryrun_func_);
          runner->dryrun_func_(runner->_device_context_, runner->_stream_id_);
          return;
        }
        runner->_MallocWorkspace();
        runner->_DispatchLaunchTask();
      }));
  }

  void _DispatchLaunchTask() override {
    MS_EXCEPTION_IF_NULL(launch_func_);
    launch_func_(_device_context_, _stream_id_);
  }

 private:
  RunnerFunc launch_func_{nullptr};
  RunnerFunc dryrun_func_{nullptr};
};

// Run the real host-side ACLNN planning path so vendor-specific workspace is
// preserved, allocate that workspace through MindSpore, and skip only launch.
#define MEGA_KERNEL_DRYRUN_ACLNN_FUNC(aclnn_api, ...)                                                            \
  [](auto &&...args) {                                                                                           \
    auto args_t = std::make_tuple(ms::pynative::Arg(std::forward<decltype(args)>(args))...);                     \
    return [args_t](auto device_context, auto stream_id) {                                                       \
      std::apply(                                                                                                \
        [&](auto &&...args) {                                                                                    \
          static const std::string aclnn_name = #aclnn_api;                                                      \
          device_context->device_res_manager_->UseStreamResInCurrentThread(stream_id);                           \
          auto result = GEN_EXECUTOR(aclnn_name, args...);                                                       \
          auto workspace_size = std::get<0>(result);                                                             \
          MS_LOG(INFO) << "[MegaKernelDryRun] " << aclnn_name << " workspace bytes: " << workspace_size;         \
          auto release_func = std::get<3>(result);                                                               \
          if (workspace_size > 0) {                                                                              \
            auto workspace =                                                                                     \
              std::make_shared<mindspore::kernel::pyboost::MemBlock>(device_context, workspace_size, stream_id); \
          }                                                                                                      \
          if (release_func != nullptr) {                                                                         \
            release_func();                                                                                      \
          }                                                                                                      \
        },                                                                                                       \
        args_t);                                                                                                 \
    };                                                                                                           \
  }(__VA_ARGS__)

#define SET_MEGA_KERNEL_ACLNN_FUNC(runner, aclnn_api, ...)                          \
  do {                                                                              \
    (runner)->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnn_api, __VA_ARGS__));             \
    (runner)->SetDryRunFunc(MEGA_KERNEL_DRYRUN_ACLNN_FUNC(aclnn_api, __VA_ARGS__)); \
  } while (false)

}  // namespace ms_multicore

#endif  // HYPER_PARALLEL_MULTICORE_MEGA_KERNEL_ACLNN_OP_RUNNER_H_
