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
#include <cstddef>
#include <cstdint>
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

using Metadata = std::optional<std::vector<int64_t>>;
using OptionalTensor = std::optional<ms::Tensor>;
using IntArrayArg = std::pair<Metadata, bool>;
using Shape = std::vector<int64_t>;

struct TokenShape {
  bool rank3;
  int64_t batch;
  int64_t heads;
  int64_t tokens;
  int64_t width;

  TokenShape(const ms::Tensor &tensor, const std::string &layout) {
    rank3 = layout == "NTD" || layout == "TND";
    if (layout != "NTD" && layout != "TND" && layout != "BNSD" && layout != "BSND") {
      MS_LOG(EXCEPTION) << "Chunk KDA stage layout must be BNSD, BSND, NTD or TND.";
    }
    const auto &shape = tensor.shape();
    if (shape.size() != (rank3 ? 3U : 4U)) {
      MS_LOG(EXCEPTION) << "Chunk KDA stage tensor rank does not match layout " << layout;
    }
    const bool sequence_major = layout == "BSND" || layout == "TND";
    const size_t base = rank3 ? 0 : 1;
    batch = rank3 ? 1 : shape[0];
    heads = shape[base + (sequence_major ? 1 : 0)];
    tokens = shape[base + (sequence_major ? 0 : 1)];
    width = shape.back();
  }
  Shape Head(int64_t h, int64_t d) const { return rank3 ? Shape{h, tokens, d} : Shape{batch, h, tokens, d}; }
  Shape Scalar(int64_t h) const { return rank3 ? Shape{h, tokens} : Shape{batch, h, tokens}; }
  Shape Output(int64_t h, int64_t d, const std::string &layout) const {
    if (layout == "NTD" || layout == "BNSD") return Head(h, d);
    if (layout == "TND" || layout == "BSND") {
      return rank3 ? Shape{tokens, h, d} : Shape{batch, tokens, h, d};
    }
    MS_LOG(EXCEPTION) << "Invalid Chunk KDA output layout " << layout;
  }
};

int64_t SequenceCount(const Metadata &cu, int64_t batch) {
  if (!cu.has_value()) return batch;
  if (cu->size() < 2) MS_LOG(EXCEPTION) << "cu_seqlens needs at least two boundaries.";
  return static_cast<int64_t>(cu->size()) - 1;
}

int64_t Chunks(const Metadata &cu, int64_t tokens, int64_t chunk_size) {
  if (chunk_size <= 0) MS_LOG(EXCEPTION) << "chunk_size must be positive for output allocation.";
  if (!cu.has_value()) return (tokens + chunk_size - 1) / chunk_size;
  SequenceCount(cu, 1);
  int64_t count = 0;
  for (size_t i = 1; i < cu->size(); ++i) {
    if ((*cu)[i - 1] < 0 || (*cu)[i] < (*cu)[i - 1] || (*cu)[i] > tokens) {
      MS_LOG(EXCEPTION) << "cu_seqlens must be nondecreasing boundaries within the token extent.";
    }
    count += ((*cu)[i] - (*cu)[i - 1] + chunk_size - 1) / chunk_size;
  }
  return count;
}

}  // namespace

std::vector<ms::Tensor> npu_chunk_kda_fwd_prepare(
  const ms::Tensor &q, const ms::Tensor &k, const ms::Tensor &v, const ms::Tensor &g, const ms::Tensor &beta,
  double scale, int64_t chunk_size, const std::string &layout, const OptionalTensor &a_log,
  const OptionalTensor &dt_bias, const Metadata &cu, const Metadata &indices, double epsilon, bool qk_norm, bool gate,
  bool beta_sigmoid, bool neg_eigval, bool safe_gate, double lower_bound, bool use_exp2, const std::string &mode) {
  const TokenShape qs(q, layout), vs(v, layout);
  if (chunk_size <= 0) MS_LOG(EXCEPTION) << "chunk_size must be positive for output allocation.";
  const bool save = mode == "save";
  const bool aux = save || mode == "recompute";
  const bool akk_enabled = mode != "none";
  if (!aux && mode != "none" && mode != "forward") {
    MS_LOG(EXCEPTION) << "backward_mode must be none, forward, recompute or save.";
  }
  const auto fp32 = ms::TypeId::kNumberTypeFloat32;
  const auto dtype = q.data_type();
  const Shape empty{0};
  const Shape qk_aux_shape = aux ? qs.Head(qs.heads, qs.width) : empty;
  const Shape rstd_aux_shape = aux ? qs.Scalar(qs.heads) : empty;
  ms::Tensor gk(fp32, qs.Head(vs.heads, qs.width));
  ms::Tensor aqk(dtype, qs.Head(vs.heads, chunk_size));
  ms::Tensor akk(dtype, akk_enabled ? qs.Head(vs.heads, chunk_size) : empty);
  ms::Tensor w(dtype, qs.Head(vs.heads, qs.width));
  ms::Tensor u(dtype, qs.Head(vs.heads, vs.width));
  ms::Tensor qg(dtype, save ? qs.Head(vs.heads, qs.width) : empty);
  ms::Tensor kg(dtype, qs.Head(vs.heads, qs.width));
  ms::Tensor qg_scaled(dtype, qs.Head(vs.heads, qs.width));
  ms::Tensor q_hat(dtype, qk_aux_shape);
  ms::Tensor k_hat(dtype, qk_aux_shape);
  ms::Tensor q_rstd(fp32, rstd_aux_shape);
  ms::Tensor k_rstd(fp32, rstd_aux_shape);
  ms::Tensor beta_eff(fp32, aux ? qs.Scalar(vs.heads) : empty);

  ms::TensorToDevice(q, k, v, g, beta, a_log, dt_bias);
  ms::TensorAllocate({gk, aqk, akk, w, u, qg, kg, qg_scaled, q_hat, k_hat, q_rstd, k_rstd, beta_eff});
  const OptionalTensor akk_out = akk_enabled ? OptionalTensor(akk) : std::nullopt;
  const OptionalTensor qg_out = save ? OptionalTensor(qg) : std::nullopt;
  const OptionalTensor q_hat_out = aux ? OptionalTensor(q_hat) : std::nullopt;
  const OptionalTensor k_hat_out = aux ? OptionalTensor(k_hat) : std::nullopt;
  const OptionalTensor q_rstd_out = aux ? OptionalTensor(q_rstd) : std::nullopt;
  const OptionalTensor k_rstd_out = aux ? OptionalTensor(k_rstd) : std::nullopt;
  const OptionalTensor beta_eff_out = aux ? OptionalTensor(beta_eff) : std::nullopt;
  DISPATCH_KDA_ND(aclnnChunkKdaFwdPrepare, q, k, v, g, beta, a_log, dt_bias, IntArrayArg(cu, true),
                    IntArrayArg(indices, true), layout, scale, chunk_size, epsilon, qk_norm, gate, beta_sigmoid,
                    neg_eigval, safe_gate, lower_bound, use_exp2, gk, aqk, akk_out, w, u, qg_out, kg, qg_scaled,
                    q_hat_out, k_hat_out, q_rstd_out, k_rstd_out, beta_eff_out);
  return {gk, aqk, akk, w, u, qg, kg, qg_scaled, q_hat, k_hat, q_rstd, k_rstd, beta_eff};
}

std::vector<ms::Tensor> npu_chunk_kda_pre_process_fwd(const ms::Tensor &k, const ms::Tensor &w, const ms::Tensor &u,
                                                      const OptionalTensor &gk, const Metadata &cu,
                                                      int64_t chunk_size) {
  const TokenShape ks(k, "BNSD"), us(u, "BNSD");
  const auto n = SequenceCount(cu, ks.batch);
  ms::Tensor summary(ms::TypeId::kNumberTypeFloat32, {n, us.heads, ks.width, us.width + ks.width});
  ms::TensorToDevice(k, w, u, gk);
  ms::TensorAllocate(summary);
  const OptionalTensor absent;
  DISPATCH_KDA_ND(aclnnPreProcessFwdKernelMerged, k, w, u, absent, gk, absent, absent, IntArrayArg(cu, true),
                    chunk_size, summary);
  return {summary};
}

std::vector<ms::Tensor> npu_chunk_kda_fwd_h(const ms::Tensor &k, const ms::Tensor &w, const ms::Tensor &u,
                                            const OptionalTensor &gk, const OptionalTensor &initial_state,
                                            bool output_final_state, int64_t chunk_size, const Metadata &cu,
                                            const Metadata &indices, bool use_exp2, bool state_v_first) {
  const TokenShape ks(k, "BNSD"), us(u, "BNSD");
  const auto chunks = Chunks(cu, ks.tokens, chunk_size);
  const auto n = SequenceCount(cu, ks.batch);
  const auto first = state_v_first ? us.width : ks.width;
  const auto last = state_v_first ? ks.width : us.width;
  ms::Tensor h(k.data_type(), {ks.batch, chunks, us.heads, first, last});
  ms::Tensor v_new(u.data_type(), u.shape());
  ms::Tensor state(initial_state.has_value() ? initial_state->data_type() : ms::TypeId::kNumberTypeFloat32,
                   output_final_state ? Shape{n, us.heads, first, last} : Shape{0});
  ms::TensorToDevice(k, w, u, gk, initial_state);
  ms::TensorAllocate({h, v_new, state});
  const OptionalTensor absent;
  const OptionalTensor final_state = output_final_state ? OptionalTensor(state) : std::nullopt;
  DISPATCH_KDA_ND(aclnnChunkFwdH, k, w, u, absent, gk, initial_state, output_final_state, chunk_size, true,
                    IntArrayArg(cu, true), IntArrayArg(indices, true), use_exp2, state_v_first, h, v_new, final_state);
  return {h, v_new, state};
}

std::vector<ms::Tensor> npu_chunk_kda_fwd_finalize(const ms::Tensor &qg_scaled, const ms::Tensor &aqk,
                                                   const ms::Tensor &v_new, const ms::Tensor &h,
                                                   const std::string &output_layout, bool state_v_first,
                                                   const Metadata &cu, const Metadata &indices) {
  const bool rank3 = qg_scaled.shape().size() == 3;
  const TokenShape qs(qg_scaled, rank3 ? "NTD" : "BNSD"), vs(v_new, rank3 ? "NTD" : "BNSD");
  if (rank3 != (output_layout == "NTD" || output_layout == "TND")) {
    MS_LOG(EXCEPTION) << "output_layout rank must match the input caches.";
  }
  ms::Tensor out(qg_scaled.data_type(), qs.Output(vs.heads, vs.width, output_layout));
  ms::TensorToDevice(qg_scaled, aqk, v_new, h);
  ms::TensorAllocate(out);
  DISPATCH_KDA_ND(aclnnChunkKdaFwdFinalize, qg_scaled, aqk, v_new, h, IntArrayArg(cu, true),
                    IntArrayArg(indices, true), output_layout, state_v_first, out);
  return {out};
}

std::vector<ms::Tensor> npu_chunk_kda_bwd_prepare(const ms::Tensor &aqk, const ms::Tensor &v_new, const ms::Tensor &d_o,
                                                  const ms::Tensor &h, double scale, const std::string &layout,
                                                  int64_t chunk_size, bool state_v_first, const Metadata &cu,
                                                  const Metadata &indices) {
  const TokenShape as(aqk, layout), vs(v_new, layout);
  if (chunk_size <= 0) MS_LOG(EXCEPTION) << "chunk_size must be positive for output allocation.";
  ms::Tensor d_aqk(ms::TypeId::kNumberTypeFloat32, as.Head(as.heads, chunk_size));
  ms::Tensor dv(v_new.data_type(), vs.Head(vs.heads, vs.width));
  // The key dimension is carried by h, independent of the value dimension.
  const auto &hs = h.shape();
  if (hs.size() != (as.rank3 ? 4U : 5U)) MS_LOG(EXCEPTION) << "h rank must be token rank + 1.";
  const auto key_dim = hs[hs.size() - (state_v_first ? 1 : 2)];
  ms::Tensor dq_raw(ms::TypeId::kNumberTypeFloat32, as.Head(as.heads, key_dim));
  ms::TensorToDevice(aqk, v_new, d_o, h);
  ms::TensorAllocate({d_aqk, dv, dq_raw});
  DISPATCH_KDA_ND(aclnnChunkKdaBwdPrepare, aqk, v_new, d_o, h, IntArrayArg(cu, true), IntArrayArg(indices, true),
                    layout, scale, chunk_size, state_v_first, d_aqk, dv, dq_raw);
  return {d_aqk, dv, dq_raw};
}

std::vector<ms::Tensor> npu_chunk_kda_pre_process_bwd(const ms::Tensor &q, const ms::Tensor &k, const ms::Tensor &w,
                                                      const ms::Tensor &d_o, const ms::Tensor &dv, double scale,
                                                      int64_t chunk_size, const OptionalTensor &gk,
                                                      const Metadata &cu) {
  const TokenShape qs(q, "BNSD"), os(d_o, "BNSD");
  ms::Tensor summary(ms::TypeId::kNumberTypeFloat32, {os.heads, qs.width, os.width + qs.width});
  ms::TensorToDevice(q, k, w, d_o, dv, gk);
  ms::TensorAllocate(summary);
  const OptionalTensor absent;
  DISPATCH_KDA_ND(aclnnPreProcessBwdKernelMerged, q, k, w, d_o, dv, absent, gk, IntArrayArg(cu, true), scale,
                    chunk_size, summary);
  return {summary};
}

std::vector<ms::Tensor> npu_chunk_kda_bwd_dhu(const ms::Tensor &q, const ms::Tensor &k, const ms::Tensor &w,
                                              const ms::Tensor &d_o, const ms::Tensor &dv, double scale,
                                              int64_t chunk_size, const OptionalTensor &gk, const OptionalTensor &h0,
                                              const OptionalTensor &dht, const Metadata &cu, const Metadata &indices,
                                              bool use_exp2, bool state_v_first) {
  const TokenShape qs(q, "BNSD"), vs(dv, "BNSD");
  const auto chunks = Chunks(cu, qs.tokens, chunk_size);
  const auto n = SequenceCount(cu, qs.batch);
  // Unlike h and dh0, FLA's dh is always K-first, even for V-first boundary states.
  ms::Tensor dh(q.data_type(), {qs.batch, chunks, vs.heads, qs.width, vs.width});
  ms::Tensor dh0(q.data_type(), h0.has_value() ? Shape{n, vs.heads, state_v_first ? vs.width : qs.width,
                                                    state_v_first ? qs.width : vs.width} : Shape{0});
  ms::Tensor dv_scan(dv.data_type(), dv.shape());
  ms::TensorToDevice(q, k, w, d_o, dv, gk, h0, dht);
  ms::TensorAllocate({dh, dh0, dv_scan});
  const OptionalTensor absent;
  const OptionalTensor dh0_out = h0.has_value() ? OptionalTensor(dh0) : std::nullopt;
  DISPATCH_KDA_ND(aclnnChunkGatedDeltaRuleBwdDhu, q, k, w, d_o, dv, absent, gk, h0, dht, IntArrayArg(cu, true),
                    IntArrayArg(indices, true), scale, chunk_size, use_exp2, state_v_first, dh, dh0_out, dv_scan);
  return {dh, dh0, dv_scan};
}

std::vector<ms::Tensor> npu_chunk_kda_bwd_finalize(
  const ms::Tensor &q, const ms::Tensor &k, const ms::Tensor &v, const ms::Tensor &gk, const ms::Tensor &raw_g,
  const ms::Tensor &beta, const OptionalTensor &a_log, const OptionalTensor &dt_bias, const ms::Tensor &akk,
  const ms::Tensor &v_new, const ms::Tensor &h, const ms::Tensor &dh, const ms::Tensor &dv_scan,
  const ms::Tensor &d_aqk, const ms::Tensor &dq_raw, double scale, double lower_bound, int64_t chunk_size,
  bool safe_gate, bool gate, bool use_exp2, bool state_v_first, const OptionalTensor &q_rstd,
  const OptionalTensor &k_rstd, const Metadata &cu, const Metadata &indices, bool beta_sigmoid, bool neg_eigval,
  const std::string &output_layout) {
  const bool rank3 = q.shape().size() == 3;
  const std::string input_layout = rank3 ? "NTD" : "BNSD";
  const TokenShape qs(q, input_layout), ks(k, input_layout), vs(v, input_layout);
  if (rank3 != (output_layout == "NTD" || output_layout == "TND")) {
    MS_LOG(EXCEPTION) << "output_layout rank must match the input caches.";
  }
  auto beta_shape = vs.Output(vs.heads, 1, output_layout);
  beta_shape.pop_back();
  ms::Tensor dq(q.data_type(), qs.Output(qs.heads, qs.width, output_layout));
  ms::Tensor dk(k.data_type(), ks.Output(ks.heads, ks.width, output_layout));
  ms::Tensor dv(v.data_type(), vs.Output(vs.heads, vs.width, output_layout));
  ms::Tensor d_beta(beta.data_type(), beta_shape);
  ms::Tensor d_g(raw_g.data_type(), qs.Output(vs.heads, qs.width, output_layout));
  ms::Tensor d_a_log(ms::TypeId::kNumberTypeFloat32, gate && a_log.has_value() ? a_log->shape() : Shape{0});
  ms::Tensor d_dt_bias(ms::TypeId::kNumberTypeFloat32, gate && dt_bias.has_value() ? dt_bias->shape() : Shape{0});
  ms::TensorToDevice(q, k, v, gk, raw_g, beta, a_log, dt_bias, akk, v_new, h, dh, dv_scan, d_aqk, dq_raw, q_rstd,
                     k_rstd);
  ms::TensorAllocate({dq, dk, dv, d_beta, d_g, d_a_log, d_dt_bias});
  const OptionalTensor d_a_log_out = gate && a_log.has_value() ? OptionalTensor(d_a_log) : std::nullopt;
  const OptionalTensor d_dt_bias_out = gate && dt_bias.has_value() ? OptionalTensor(d_dt_bias) : std::nullopt;
  DISPATCH_KDA_ND(aclnnChunkKdaBwdFinalize, q, k, v, gk, raw_g, beta, a_log, dt_bias, akk, v_new, h, dh, dv_scan,
                    d_aqk, dq_raw, q_rstd, k_rstd, IntArrayArg(cu, true), IntArrayArg(indices, true), scale,
                    lower_bound, chunk_size, safe_gate, gate, use_exp2, state_v_first, beta_sigmoid, neg_eigval,
                    output_layout, dq, dk, dv, d_beta, d_g, d_a_log_out, d_dt_bias_out);
  return {dq, dk, dv, d_beta, d_g, d_a_log, d_dt_bias};
}

std::vector<ms::Tensor> npu_chunk_kda_merge_fwd_bwd(const ms::Tensor &state, const ms::Tensor &summaries, int64_t count,
                                                    int64_t rank, bool forward, bool state_v_first) {
  ms::TensorToDevice(state, summaries);
  DISPATCH_KDA_ND(aclnnMergeFwdBwdKernel, state, summaries, count, rank, forward, state_v_first);
  return {state};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("npu_chunk_kda_fwd_prepare", PYBOOST_CALLER(13, custom::npu_chunk_kda_fwd_prepare));
  m.def("npu_chunk_kda_pre_process_fwd", PYBOOST_CALLER(1, custom::npu_chunk_kda_pre_process_fwd));
  m.def("npu_chunk_kda_fwd_h", PYBOOST_CALLER(3, custom::npu_chunk_kda_fwd_h));
  m.def("npu_chunk_kda_fwd_finalize", PYBOOST_CALLER(1, custom::npu_chunk_kda_fwd_finalize));
  m.def("npu_chunk_kda_bwd_prepare", PYBOOST_CALLER(3, custom::npu_chunk_kda_bwd_prepare));
  m.def("npu_chunk_kda_pre_process_bwd", PYBOOST_CALLER(1, custom::npu_chunk_kda_pre_process_bwd));
  m.def("npu_chunk_kda_bwd_dhu", PYBOOST_CALLER(3, custom::npu_chunk_kda_bwd_dhu));
  m.def("npu_chunk_kda_bwd_finalize", PYBOOST_CALLER(7, custom::npu_chunk_kda_bwd_finalize));
  m.def("npu_chunk_kda_merge_fwd_bwd", PYBOOST_CALLER(1, custom::npu_chunk_kda_merge_fwd_bwd));
}
}  // namespace custom
