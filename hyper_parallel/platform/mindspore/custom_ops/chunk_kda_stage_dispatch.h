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
#ifndef HP_CHUNK_KDA_STAGE_DISPATCH_H_
#define HP_CHUNK_KDA_STAGE_DISPATCH_H_

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "aclnn/acl_meta.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "kernel/ascend/acl_ir/op_api_exec.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "ms_extension/all.h"

namespace custom::kda_stage {
using Metadata = std::optional<std::vector<int64_t>>;
using OptionalTensor = std::optional<ms::Tensor>;

template <typename F>
F Resolve(const std::string &name) {
  auto *symbol = mindspore::device::ascend::GetOpApiFunc(name.c_str());
  if (symbol == nullptr) {
    MS_LOG(EXCEPTION) << name << " is unavailable. Install a matching FLA-NPU OPP and source its environment.";
  }
  return reinterpret_cast<F>(symbol);
}

// Composite FLA stages inspect OriginalShape as well as ViewShape. Use logical
// contiguous ND descriptors, rather than MindSpore's flattened storage shape.
struct Descriptors {
  using CreateTensor = aclTensor *(*)(const int64_t *, uint64_t, aclDataType, const int64_t *, int64_t, aclFormat,
                                      const int64_t *, uint64_t, void *);
  using CreateArray = aclIntArray *(*)(const int64_t *, uint64_t);
  using DestroyTensor = aclnnStatus (*)(const aclTensor *);
  using DestroyArray = aclnnStatus (*)(const aclIntArray *);
  CreateTensor create_tensor = Resolve<CreateTensor>("aclCreateTensor");
  CreateArray create_array = Resolve<CreateArray>("aclCreateIntArray");
  DestroyTensor destroy_tensor = Resolve<DestroyTensor>("aclDestroyTensor");
  DestroyArray destroy_array = Resolve<DestroyArray>("aclDestroyIntArray");
  std::vector<aclTensor *> tensors;
  std::vector<aclIntArray *> arrays;

  ~Descriptors() {
    for (auto *tensor : tensors) destroy_tensor(tensor);
    for (auto *array : arrays) destroy_array(array);
  }

  const aclTensor *Convert(const ms::Tensor &tensor) {
    if (!tensor.is_defined() || tensor.shape() == std::vector<int64_t>{0}) return nullptr;
    const auto &shape = tensor.shape();
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (size_t axis = shape.size(); axis > 0; --axis) {
      strides[axis - 1] = stride;
      stride *= shape[axis - 1];
    }
    const auto &ptr = tensor.tensor();
    const auto &address = ptr->device_address();
    if (address == nullptr || address->GetMutablePtr() == nullptr) {
      MS_LOG(EXCEPTION) << "Chunk KDA stage tensor has no allocated device memory.";
    }
    auto *result =
      create_tensor(shape.data(), shape.size(), mindspore::device::ascend::AclConverter::ConvertType(ptr->data_type()),
                    strides.data(), static_cast<int64_t>(ptr->storage_offset()), ACL_FORMAT_ND, shape.data(),
                    shape.size(), address->GetMutablePtr());
    if (result == nullptr) MS_LOG(EXCEPTION) << "aclCreateTensor failed for Chunk KDA stage.";
    tensors.push_back(result);
    return result;
  }
  const aclTensor *Convert(const OptionalTensor &tensor) { return tensor.has_value() ? Convert(*tensor) : nullptr; }
  const aclIntArray *Convert(const Metadata &values) {
    if (!values.has_value()) return nullptr;
    auto *result = create_array(values->data(), values->size());
    if (result == nullptr) MS_LOG(EXCEPTION) << "aclCreateIntArray failed for Chunk KDA stage.";
    arrays.push_back(result);
    return result;
  }
  const char *Convert(const std::string &value) { return value.c_str(); }
  template <typename T>
  T Convert(const T &value) {
    return value;
  }
};

template <typename T>
void Prepare(const T &) {}
inline void Prepare(const ms::Tensor &tensor) {
  if (tensor.is_defined() && tensor.shape() != std::vector<int64_t>{0}) ms::TensorToDevice(tensor);
}
inline void Prepare(const OptionalTensor &tensor) {
  if (tensor.has_value()) Prepare(*tensor);
}

template <typename T>
void Track(const std::string &, const mindspore::device::DeviceContext *, size_t, const T &) {}
inline void Track(const std::string &name, const mindspore::device::DeviceContext *context, size_t stream,
                  const ms::Tensor &tensor) {
  if (tensor.is_defined() && tensor.shape() != std::vector<int64_t>{0}) {
    mindspore::runtime::DeviceAddressUtils::ProcessCrossStreamAddress(name, context, stream, tensor.tensor());
  }
}
inline void Track(const std::string &name, const mindspore::device::DeviceContext *context, size_t stream,
                  const OptionalTensor &tensor) {
  if (tensor.has_value()) Track(name, context, stream, *tensor);
}

// Resolve at call time: no FLA headers/link dependency and no import-time rebuild
// or failure when an optional stage is absent from the active OPP.
template <typename... Args>
void Dispatch(const std::string &api, const Args &...args) {
  using WorkspaceFn =
    aclnnStatus (*)(decltype(std::declval<Descriptors>().Convert(args))..., uint64_t *, aclOpExecutor **);
  auto get_workspace = Resolve<WorkspaceFn>(api + "GetWorkspaceSize");
  auto launch = Resolve<mindspore::device::ascend::RunApiFunc>(api);
  (Prepare(args), ...);
  auto *context = ms::GetCurrentDeviceContext();
  const auto stream_id = static_cast<size_t>(mindspore::CurrentStream::id());
  auto owned_args = std::make_tuple(args...);
  mindspore::kernel::pyboost::PyBoostUtils::DispatchRun(std::make_shared<mindspore::runtime::PyBoostDeviceTask>([=]() {
    context->device_res_manager_->UseStreamResInCurrentThread(stream_id);
    auto descriptors = std::make_shared<Descriptors>();
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    const auto status = std::apply(
      [&](const auto &...values) { return get_workspace(descriptors->Convert(values)..., &workspace_size, &executor); },
      owned_args);
    if (status != 0) MS_LOG(EXCEPTION) << api << "GetWorkspaceSize failed: " << status;
    mindspore::kernel::pyboost::MemBlockPtr workspace;
    if (workspace_size != 0) {
      workspace = std::make_shared<mindspore::kernel::pyboost::MemBlock>(context, workspace_size, stream_id);
    }
    auto *stream = context->device_res_manager_->GetStream(stream_id);
    // Match PyBoost's two-level pipeline. Launching from the device task
    // overtakes queued launch tasks (e.g. a preceding contiguous copy), so
    // the kernel can read uninitialized input even on the same stream.
    mindspore::runtime::OpExecutor::DispatchLaunchTask([=]() {
      context->device_res_manager_->BindDeviceToCurrentThread(false);
      const auto launch_status = launch(workspace ? workspace->ptr_ : nullptr, workspace_size, executor, stream);
      if (launch_status != 0) MS_LOG(EXCEPTION) << api << " launch failed: " << launch_status;
      // Retain input/output tensors until the kernel is enqueued.
      (void)owned_args;
      (void)descriptors;
    });
    // As in LAUNCH_ACLNN, track addresses only after the device task has
    // prepared them and submitted the launch task.
    if (mindspore::runtime::RuntimeConf::GetInstance()->launch_blocking()) {
      if (!mindspore::device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {
        MS_LOG(EXCEPTION) << "SyncStream failed for " << api;
      }
    } else {
      std::apply([&](const auto &...values) { (Track(api, context, stream_id, values), ...); }, owned_args);
    }
  }));
}
}  // namespace custom::kda_stage
#endif  // HP_CHUNK_KDA_STAGE_DISPATCH_H_
