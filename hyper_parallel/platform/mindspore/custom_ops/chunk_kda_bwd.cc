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

#include <optional>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "aclnn/acl_meta.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "kernel/ascend/acl_ir/op_api_exec.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "ms_extension/all.h"
#include "module.h"

namespace custom {
namespace {

using CreateTensorFn = aclTensor *(*)(const int64_t *, uint64_t, aclDataType, const int64_t *, int64_t, aclFormat,
                                      const int64_t *, uint64_t, void *);
using CreateIntArrayFn = aclIntArray *(*)(const int64_t *, uint64_t);
using DestroyTensorFn = aclnnStatus (*)(const aclTensor *);
using DestroyIntArrayFn = aclnnStatus (*)(const aclIntArray *);
using ChunkKdaBwdGetWorkspaceSizeFn = aclnnStatus (*)(
  const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
  const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
  const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
  const aclIntArray *, const aclIntArray *, double, int64_t, bool, bool, double, bool, bool, bool, const aclTensor *,
  const aclTensor *, const aclTensor *, const char *, bool, bool, const aclTensor *, const aclTensor *,
  const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
  uint64_t *, aclOpExecutor **);

template <typename Function>
Function LoadOpApi(const char *name) {
  auto *symbol = mindspore::device::ascend::GetOpApiFunc(name);
  if (symbol == nullptr) {
    MS_LOG(EXCEPTION) << "Unable to resolve CANN API " << name << ".";
  }
  return reinterpret_cast<Function>(symbol);
}

struct ChunkKdaBwdAclResources {
  CreateIntArrayFn create_int_array = nullptr;
  DestroyTensorFn destroy_tensor = nullptr;
  DestroyIntArrayFn destroy_int_array = nullptr;
  aclOpExecutor *executor = nullptr;
  std::vector<aclTensor *> tensors;
  std::vector<aclIntArray *> arrays;
  bool released = false;

  void Release() {
    if (released) {
      return;
    }
    released = true;
    if (destroy_tensor != nullptr) {
      for (auto *tensor : tensors) {
        if (tensor != nullptr) {
          destroy_tensor(tensor);
        }
      }
    }
    if (destroy_int_array != nullptr) {
      for (auto *array : arrays) {
        if (array != nullptr) {
          destroy_int_array(array);
        }
      }
    }
  }

  ~ChunkKdaBwdAclResources() { Release(); }
};

// FLA's composite backward transposes beta/dO inside ACLNN. The generic MindSpore
// converter flattens storage dimensions, so provide explicit contiguous ND descriptors.
aclTensor *CreateNdTensor(const ms::Tensor &tensor, const char *name, const CreateTensorFn &create_tensor,
                          ChunkKdaBwdAclResources *resources) {
  if (!tensor.is_defined()) {
    return nullptr;
  }
  const auto tensor_ptr = tensor.tensor();
  const auto &shape = tensor.shape();
  std::vector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (size_t axis = shape.size(); axis > 0; --axis) {
    strides[axis - 1] = stride;
    stride *= shape[axis - 1];
  }
  const auto &address = tensor_ptr->device_address();
  if (address == nullptr || address->GetMutablePtr() == nullptr) {
    MS_LOG(EXCEPTION) << "Chunk KDA backward tensor " << name << " with shape " << shape
                      << " has no allocated device memory.";
  }
  auto *descriptor = create_tensor(shape.data(), shape.size(),
                                   mindspore::device::ascend::AclConverter::ConvertType(tensor_ptr->data_type()),
                                   strides.data(), static_cast<int64_t>(tensor_ptr->storage_offset()), ACL_FORMAT_ND,
                                   shape.data(), shape.size(), address->GetMutablePtr());
  if (descriptor == nullptr) {
    MS_LOG(EXCEPTION) << "aclCreateTensor failed for Chunk KDA backward.";
  }
  resources->tensors.push_back(descriptor);
  return descriptor;
}

aclIntArray *CreateIntArray(const std::optional<std::vector<int64_t>> &values, ChunkKdaBwdAclResources *resources) {
  if (!values.has_value() || values->empty()) {
    return nullptr;
  }
  auto *array = resources->create_int_array(values->data(), values->size());
  if (array == nullptr) {
    MS_LOG(EXCEPTION) << "aclCreateIntArray failed for Chunk KDA backward.";
  }
  resources->arrays.push_back(array);
  return array;
}

void DispatchChunkKdaBwdAclnn(const ms::Tensor &q, const ms::Tensor &k, const ms::Tensor &v, const ms::Tensor &beta,
                              const ms::Tensor &gk, const ms::Tensor &aqk, const ms::Tensor &akk, const ms::Tensor &w,
                              const ms::Tensor &qg, const ms::Tensor &kg, const ms::Tensor &v_new, const ms::Tensor &h,
                              const ms::Tensor &d_o, const std::optional<ms::Tensor> &raw_g_opt,
                              const std::optional<ms::Tensor> &a_log_opt, const std::optional<ms::Tensor> &dt_bias_opt,
                              const std::optional<std::vector<int64_t>> &cu_seqlens_opt,
                              const std::optional<std::vector<int64_t>> &chunk_indices_opt, double scale,
                              int64_t chunk_size, bool safe_gate, bool use_gate_in_kernel, double lower_bound,
                              const std::optional<std::string> &layout_opt, const std::optional<ms::Tensor> &q_rstd_opt,
                              const std::optional<ms::Tensor> &k_rstd_opt,
                              const std::optional<ms::Tensor> &beta_eff_opt, bool use_beta_sigmoid_in_kernel,
                              bool allow_neg_eigval, bool use_exp2, const ms::Tensor &dq, const ms::Tensor &dk,
                              const ms::Tensor &dv, const ms::Tensor &db, const ms::Tensor &dg,
                              const std::optional<ms::Tensor> &d_a_opt, const std::optional<ms::Tensor> &d_bias_opt) {
  auto *device_context = ms::GetCurrentDeviceContext();
  const auto stream_id = static_cast<size_t>(mindspore::CurrentStream::id());
  mindspore::kernel::pyboost::PyBoostUtils::DispatchRun(std::make_shared<mindspore::runtime::PyBoostDeviceTask>([=]() {
    device_context->device_res_manager_->UseStreamResInCurrentThread(stream_id);

    auto resources = std::make_shared<ChunkKdaBwdAclResources>();
    const auto create_tensor = LoadOpApi<CreateTensorFn>("aclCreateTensor");
    resources->create_int_array = LoadOpApi<CreateIntArrayFn>("aclCreateIntArray");
    resources->destroy_tensor = LoadOpApi<DestroyTensorFn>("aclDestroyTensor");
    resources->destroy_int_array = LoadOpApi<DestroyIntArrayFn>("aclDestroyIntArray");

    auto tensor = [&](const ms::Tensor &value, const char *name) {
      return CreateNdTensor(value, name, create_tensor, resources.get());
    };
    auto optional_tensor = [&](const std::optional<ms::Tensor> &value, const char *name) {
      return value.has_value() ? tensor(value.value(), name) : nullptr;
    };
    const std::string layout = layout_opt.value_or("BNSD");
    const auto q_acl = tensor(q, "q");
    const auto k_acl = tensor(k, "k");
    const auto v_acl = tensor(v, "v");
    const auto beta_acl = tensor(beta, "beta");
    const auto gk_acl = tensor(gk, "gk");
    const auto aqk_acl = tensor(aqk, "aqk");
    const auto akk_acl = tensor(akk, "akk");
    const auto w_acl = tensor(w, "w");
    const auto qg_acl = tensor(qg, "qg");
    const auto kg_acl = tensor(kg, "kg");
    const auto v_new_acl = tensor(v_new, "v_new");
    const auto h_acl = tensor(h, "h");
    const auto d_o_acl = tensor(d_o, "d_o");
    const auto raw_g_acl = optional_tensor(raw_g_opt, "raw_g");
    const auto a_log_acl = optional_tensor(a_log_opt, "a_log");
    const auto dt_bias_acl = optional_tensor(dt_bias_opt, "dt_bias");
    const auto q_rstd_acl = optional_tensor(q_rstd_opt, "q_rstd");
    const auto k_rstd_acl = optional_tensor(k_rstd_opt, "k_rstd");
    const auto beta_eff_acl = optional_tensor(beta_eff_opt, "beta_eff");
    const auto dq_acl = tensor(dq, "dq");
    const auto dk_acl = tensor(dk, "dk");
    const auto dv_acl = tensor(dv, "dv");
    const auto db_acl = tensor(db, "db");
    const auto dg_acl = tensor(dg, "dg");
    const auto d_a_acl = optional_tensor(d_a_opt, "d_a");
    const auto d_bias_acl = optional_tensor(d_bias_opt, "d_bias");
    const auto cu_acl = CreateIntArray(cu_seqlens_opt, resources.get());
    const auto chunk_indices_acl = CreateIntArray(chunk_indices_opt, resources.get());

    auto get_workspace = LoadOpApi<ChunkKdaBwdGetWorkspaceSizeFn>("aclnnChunkKdaBwdGetWorkspaceSize");
    uint64_t workspace_size = 0;
    auto status = get_workspace(q_acl, k_acl, v_acl, beta_acl, gk_acl, aqk_acl, akk_acl, w_acl, qg_acl, kg_acl,
                                v_new_acl, h_acl, d_o_acl, raw_g_acl, a_log_acl, dt_bias_acl, nullptr, nullptr, cu_acl,
                                chunk_indices_acl, scale, chunk_size, safe_gate, use_gate_in_kernel, lower_bound, true,
                                use_exp2, false, q_rstd_acl, k_rstd_acl, beta_eff_acl, layout.c_str(),
                                use_beta_sigmoid_in_kernel, allow_neg_eigval, dq_acl, dk_acl, dv_acl, db_acl, dg_acl,
                                nullptr, d_a_acl, d_bias_acl, &workspace_size, &resources->executor);
    if (status != 0) {
      MS_LOG(EXCEPTION) << "aclnnChunkKdaBwdGetWorkspaceSize failed with status " << status << ".";
    }

    mindspore::kernel::pyboost::MemBlockPtr workspace;
    void *workspace_address = nullptr;
    if (workspace_size != 0) {
      workspace = std::make_shared<mindspore::kernel::pyboost::MemBlock>(device_context, workspace_size, stream_id);
      workspace_address = workspace->ptr_;
    }
    const std::string api_name = "aclnnChunkKdaBwd";
    auto *stream = device_context->device_res_manager_->GetStream(stream_id);
    const auto api = mindspore::device::ascend::GetOpApiFunc(api_name.c_str());
    if (api == nullptr) {
      MS_LOG(EXCEPTION) << api_name << " is not available in the active CANN OPP.";
    }
    const auto launch = reinterpret_cast<mindspore::device::ascend::RunApiFunc>(api);
    const auto launch_status = launch(workspace_address, workspace_size, resources->executor, stream);
    if (launch_status != 0) {
      MS_LOG(EXCEPTION) << api_name << " launch failed with status " << launch_status << ".";
    }
    resources->Release();
  }));

  if (mindspore::runtime::RuntimeConf::GetInstance()->launch_blocking()) {
    if (!mindspore::device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {
      MS_LOG(EXCEPTION) << "SyncStream failed for op aclnnChunkKdaBwd.";
    }
  } else {
    const auto raw_g_arg = ms::pynative::Arg(raw_g_opt);
    const auto a_log_arg = ms::pynative::Arg(a_log_opt);
    const auto dt_bias_arg = ms::pynative::Arg(dt_bias_opt);
    const auto q_rstd_arg = ms::pynative::Arg(q_rstd_opt);
    const auto k_rstd_arg = ms::pynative::Arg(k_rstd_opt);
    const auto beta_eff_arg = ms::pynative::Arg(beta_eff_opt);
    const auto d_a_arg = ms::pynative::Arg(d_a_opt);
    const auto d_bias_arg = ms::pynative::Arg(d_bias_opt);
    mindspore::runtime::DeviceAddressUtils::ProcessCrossStreamAddress(
      "aclnnChunkKdaBwd", device_context, stream_id, q.tensor(), k.tensor(), v.tensor(), beta.tensor(), gk.tensor(),
      aqk.tensor(), akk.tensor(), w.tensor(), qg.tensor(), kg.tensor(), v_new.tensor(), h.tensor(), d_o.tensor(),
      raw_g_arg, a_log_arg, dt_bias_arg, q_rstd_arg, k_rstd_arg, beta_eff_arg, dq.tensor(), dk.tensor(), dv.tensor(),
      db.tensor(), dg.tensor(), d_a_arg, d_bias_arg);
  }
}

void ValidateShapeAccess(const ms::Tensor &q) {
  const size_t rank = q.shape().size();
  if (rank != 3U && rank != 4U) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_bwd: q rank must be 3 or 4.";
  }
}

}  // namespace

std::vector<ms::Tensor> npu_chunk_kda_bwd(
  const ms::Tensor &q, const ms::Tensor &k, const ms::Tensor &v, const ms::Tensor &beta, const ms::Tensor &gk,
  const ms::Tensor &aqk, const ms::Tensor &akk, const ms::Tensor &w, const ms::Tensor &qg, const ms::Tensor &kg,
  const ms::Tensor &v_new, const ms::Tensor &h, const ms::Tensor &d_o, const std::optional<ms::Tensor> &raw_g_opt,
  const std::optional<ms::Tensor> &a_log_opt, const std::optional<ms::Tensor> &dt_bias_opt,
  const std::optional<std::vector<int64_t>> &cu_seqlens_opt,
  const std::optional<std::vector<int64_t>> &chunk_indices_opt, double scale, int64_t chunk_size, bool safe_gate,
  bool use_gate_in_kernel, double lower_bound, const std::optional<std::string> &layout_opt,
  const std::optional<ms::Tensor> &q_rstd_opt, const std::optional<ms::Tensor> &k_rstd_opt,
  const std::optional<ms::Tensor> &beta_eff_opt, bool use_beta_sigmoid_in_kernel, bool allow_neg_eigval,
  bool use_exp2) {
  const std::string layout = layout_opt.value_or("BNSD");
  ValidateShapeAccess(q);
  const bool varlen = q.shape().size() == 3U;
  // raw_g supplies the dtype and shape of the mandatory dg output before the
  // ACLNN call. All other contract validation belongs to FLA-NPU.
  if (!raw_g_opt.has_value()) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_bwd: raw_g is required to allocate dg.";
  }

  const int64_t value_heads = varlen ? v.shape()[0] : v.shape()[1];
  const int64_t key_dim = q.shape().back();
  ms::Tensor dq(q.data_type(), q.shape());
  ms::Tensor dk(k.data_type(), k.shape());
  ms::Tensor dv(v.data_type(), v.shape());
  ms::Tensor db(beta.data_type(), beta.shape());
  ms::Tensor dg(raw_g_opt->data_type(), raw_g_opt->shape());
  ms::Tensor d_a(ms::TypeId::kNumberTypeFloat32, {value_heads});
  ms::Tensor d_bias(ms::TypeId::kNumberTypeFloat32, {value_heads, key_dim});

  ms::TensorToDevice(q, k, v, beta, gk, aqk, akk, w, qg, kg, v_new, h, d_o, raw_g_opt.value_or(ms::Tensor()),
                     a_log_opt.value_or(ms::Tensor()), dt_bias_opt.value_or(ms::Tensor()),
                     q_rstd_opt.value_or(ms::Tensor()), k_rstd_opt.value_or(ms::Tensor()),
                     beta_eff_opt.value_or(ms::Tensor()));
  ms::TensorAllocate({dq, dk, dv, db, dg, d_a, d_bias});

  const std::optional<ms::Tensor> empty = std::nullopt;
  const std::optional<ms::Tensor> d_a_opt = use_gate_in_kernel ? std::optional<ms::Tensor>(d_a) : empty;
  const std::optional<ms::Tensor> d_bias_opt = dt_bias_opt.has_value() ? std::optional<ms::Tensor>(d_bias) : empty;
  DispatchChunkKdaBwdAclnn(q, k, v, beta, gk, aqk, akk, w, qg, kg, v_new, h, d_o, raw_g_opt, a_log_opt, dt_bias_opt,
                           cu_seqlens_opt, chunk_indices_opt, scale, chunk_size, safe_gate, use_gate_in_kernel,
                           lower_bound, layout, q_rstd_opt, k_rstd_opt, beta_eff_opt, use_beta_sigmoid_in_kernel,
                           allow_neg_eigval, use_exp2, dq, dk, dv, db, dg, d_a_opt, d_bias_opt);
  return {dq, dk, dv, db, dg, d_a, d_bias};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) { m.def("npu_chunk_kda_bwd", PYBOOST_CALLER(7, custom::npu_chunk_kda_bwd)); }

}  // namespace custom
