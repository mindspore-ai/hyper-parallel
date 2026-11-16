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

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ms_extension/all.h"
#include "module.h"

namespace custom {
namespace {
using mindspore::device::ascend::aclTensor;
// FLA tiling reads the storage shape. MindSpore's default TensorPtr converter
// flattens it, so KDA needs logical ND descriptors at this boundary.
struct NdTensor {
  mindspore::tensor::TensorPtr tensor;
};

aclTensor *ConvertType(const NdTensor &value) {
  const auto &tensor = value.tensor;
  const auto &shape = tensor->shape();
  const auto &strides = tensor->stride();
  const auto &address = tensor->device_address();
  static const auto create_tensor = reinterpret_cast<mindspore::device::ascend::_aclCreateTensor>(
    mindspore::device::ascend::GetOpApiFunc("aclCreateTensor"));
  if (create_tensor == nullptr) MS_LOG(EXCEPTION) << "aclCreateTensor is unavailable.";
  MS_EXCEPTION_IF_NULL(address);
  return create_tensor(shape.data(), shape.size(),
                       mindspore::device::ascend::AclConverter::ConvertType(tensor->data_type()),
                       strides.data(), tensor->storage_offset(), ACL_FORMAT_ND,
                       shape.data(), shape.size(), address->GetMutablePtr());
}

aclTensor *ConvertType(const std::optional<NdTensor> &value) {
  return value.has_value() ? ConvertType(*value) : nullptr;
}

// Let the framework retain ownership of descriptors and cache tensor metadata.
std::vector<void *> GetAddr(const NdTensor &value) {
  return mindspore::device::ascend::GetAddr(value.tensor);
}
std::vector<void *> GetAddr(const std::optional<NdTensor> &value) {
  return value.has_value() ? GetAddr(*value) : std::vector<void *>{nullptr};
}
void GatherInfo(const NdTensor &value) {
  mindspore::device::ascend::GatherInfo(value.tensor);
  mindspore::device::ascend::GatherInfo(ACL_FORMAT_ND);
}
void GatherHash(const NdTensor &value) {
  mindspore::device::ascend::GatherHash(value.tensor);
  mindspore::device::ascend::GatherHash(ACL_FORMAT_ND);
}
void RefreshAddr(const NdTensor &value) { mindspore::device::ascend::RefreshAddr(value.tensor); }
void RefreshAddr(const std::optional<NdTensor> &value) {
  if (value.has_value()) RefreshAddr(*value);
}

NdTensor AsNd(const mindspore::tensor::TensorPtr &tensor) { return {tensor}; }
std::optional<NdTensor> AsNd(const std::optional<mindspore::tensor::TensorPtr> &tensor) {
  return tensor.has_value() ? std::optional<NdTensor>(NdTensor{*tensor}) : std::nullopt;
}
template <typename T>
const T &AsNd(const T &value) {
  return value;
}

// Keep framework launch/workspace management and track the original tensors.
#define DISPATCH_KDA_ND(aclnn_api, ...)                                                                          \
  do {                                                                                                         \
    [](auto &&...values) {                                                                                      \
      auto args = std::make_tuple(ms::pynative::Arg(std::forward<decltype(values)>(values))...);                  \
      auto *context = ms::GetCurrentDeviceContext();                                                            \
      auto stream = static_cast<size_t>(mindspore::CurrentStream::id());                                         \
      mindspore::kernel::pyboost::PyBoostUtils::DispatchRun(                                                     \
        std::make_shared<mindspore::runtime::PyBoostDeviceTask>([=]() {                                          \
          std::apply([&](const auto &...a) {                                                                    \
            LAUNCH_ACLNN(aclnn_api, context, stream, AsNd(a)...);                                                \
            if (!mindspore::runtime::RuntimeConf::GetInstance()->launch_blocking()) {                            \
              mindspore::runtime::DeviceAddressUtils::ProcessCrossStreamAddress(                                \
                #aclnn_api, context, stream, a...);                                                             \
            }                                                                                                  \
          }, args);                                                                                            \
        }));                                                                                                   \
    }(__VA_ARGS__);                                                                                            \
  } while (false)
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
  const size_t rank = q.shape().size();
  // Protect host-side shape indexing and allocation; FLA validates the operator contract.
  if ((rank != 3U && rank != 4U) || v.shape().size() != rank) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_bwd: q and v must have the same rank of 3 or 4.";
  }
  if (!raw_g_opt.has_value()) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_bwd: raw_g is required to allocate dg.";
  }
  const int64_t value_heads = rank == 3U ? v.shape()[0] : v.shape()[1];
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

  const std::optional<ms::Tensor> absent = std::nullopt;
  const auto d_a_opt = use_gate_in_kernel ? std::optional<ms::Tensor>(d_a) : absent;
  const auto d_bias_opt = dt_bias_opt.has_value() ? std::optional<ms::Tensor>(d_bias) : absent;
  using IntArrayArg = std::pair<std::optional<std::vector<int64_t>>, bool>;
  DISPATCH_KDA_ND(aclnnChunkKdaBwd, q, k, v, beta, gk, aqk, akk, w, qg, kg, v_new, h, d_o, raw_g_opt,
                    a_log_opt, dt_bias_opt, absent, absent, IntArrayArg(cu_seqlens_opt, true),
                    IntArrayArg(chunk_indices_opt, true), scale, chunk_size, safe_gate, use_gate_in_kernel,
                    lower_bound, true, use_exp2, false, q_rstd_opt, k_rstd_opt, beta_eff_opt, layout,
                    use_beta_sigmoid_in_kernel, allow_neg_eigval, dq, dk, dv, db, dg, absent, d_a_opt, d_bias_opt);
  return {dq, dk, dv, db, dg, d_a, d_bias};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) { m.def("npu_chunk_kda_bwd", PYBOOST_CALLER(7, custom::npu_chunk_kda_bwd)); }

}  // namespace custom
