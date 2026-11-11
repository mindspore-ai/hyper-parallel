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
#include "chunk_kda_stage_dispatch.h"
#include "module.h"

namespace custom {
namespace {
using kda_stage::Dispatch;
using kda_stage::Metadata;
using kda_stage::OptionalTensor;
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

ms::Tensor New(ms::TypeId dtype, const Shape &shape, bool enabled = true) {
  auto tensor = ms::Tensor(dtype, enabled ? shape : Shape{0});
  if (enabled) ms::TensorAllocate(tensor);
  return tensor;
}
OptionalTensor Slot(const ms::Tensor &tensor, bool enabled) { return enabled ? OptionalTensor(tensor) : std::nullopt; }
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
  if (!save && !aux && mode != "none" && mode != "forward") {
    MS_LOG(EXCEPTION) << "backward_mode must be none, forward, recompute or save.";
  }
  auto fp32 = ms::TypeId::kNumberTypeFloat32;
  auto dtype = q.data_type();
  std::vector<ms::Tensor> out{New(fp32, qs.Head(vs.heads, qs.width)),
                              New(dtype, qs.Head(vs.heads, chunk_size)),
                              New(dtype, qs.Head(vs.heads, chunk_size), akk_enabled),
                              New(dtype, qs.Head(vs.heads, qs.width)),
                              New(dtype, qs.Head(vs.heads, vs.width)),
                              New(dtype, qs.Head(vs.heads, qs.width), save),
                              New(dtype, qs.Head(vs.heads, qs.width)),
                              New(dtype, qs.Head(vs.heads, qs.width)),
                              New(dtype, qs.Head(qs.heads, qs.width), aux),
                              New(dtype, qs.Head(qs.heads, qs.width), aux),
                              New(fp32, qs.Scalar(qs.heads), aux),
                              New(fp32, qs.Scalar(qs.heads), aux),
                              New(fp32, qs.Scalar(vs.heads), aux)};
  Dispatch("aclnnChunkKdaFwdPrepare", q, k, v, g, beta, a_log, dt_bias, cu, indices, layout, scale, chunk_size, epsilon,
           qk_norm, gate, beta_sigmoid, neg_eigval, safe_gate, lower_bound, use_exp2, out[0], out[1],
           Slot(out[2], akk_enabled), out[3], out[4], Slot(out[5], save), out[6], out[7], Slot(out[8], aux),
           Slot(out[9], aux), Slot(out[10], aux), Slot(out[11], aux), Slot(out[12], aux));
  return out;
}

std::vector<ms::Tensor> npu_chunk_kda_pre_process_fwd(const ms::Tensor &k, const ms::Tensor &w, const ms::Tensor &u,
                                                      const OptionalTensor &gk, const Metadata &cu,
                                                      int64_t chunk_size) {
  const TokenShape ks(k, "BNSD"), us(u, "BNSD");
  const auto n = SequenceCount(cu, ks.batch);
  auto summary = New(ms::TypeId::kNumberTypeFloat32, {n, us.heads, ks.width, us.width + ks.width});
  const OptionalTensor absent;
  Dispatch("aclnnPreProcessFwdKernelMerged", k, w, u, absent, gk, absent, absent, cu, chunk_size, summary);
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
  auto h = New(k.data_type(), {ks.batch, chunks, us.heads, first, last});
  auto v_new = New(u.data_type(), u.shape());
  auto state = New(initial_state.has_value() ? initial_state->data_type() : ms::TypeId::kNumberTypeFloat32,
                   {n, us.heads, first, last}, output_final_state);
  const OptionalTensor absent;
  Dispatch("aclnnChunkFwdH", k, w, u, absent, gk, initial_state, output_final_state, chunk_size, true, cu, indices,
           use_exp2, state_v_first, h, v_new, Slot(state, output_final_state));
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
  auto out = New(qg_scaled.data_type(), qs.Output(vs.heads, vs.width, output_layout));
  Dispatch("aclnnChunkKdaFwdFinalize", qg_scaled, aqk, v_new, h, cu, indices, output_layout, state_v_first, out);
  return {out};
}

std::vector<ms::Tensor> npu_chunk_kda_bwd_prepare(const ms::Tensor &aqk, const ms::Tensor &v_new, const ms::Tensor &d_o,
                                                  const ms::Tensor &h, double scale, const std::string &layout,
                                                  int64_t chunk_size, bool state_v_first, const Metadata &cu,
                                                  const Metadata &indices) {
  const TokenShape as(aqk, layout), vs(v_new, layout);
  auto da = New(ms::TypeId::kNumberTypeFloat32, as.Head(as.heads, chunk_size));
  auto dv = New(v_new.data_type(), vs.Head(vs.heads, vs.width));
  // The key dimension is carried by h, independent of the value dimension.
  const auto &hs = h.shape();
  if (hs.size() != (as.rank3 ? 4U : 5U)) MS_LOG(EXCEPTION) << "h rank must be token rank + 1.";
  const auto key_dim = hs[hs.size() - (state_v_first ? 1 : 2)];
  auto dq = New(ms::TypeId::kNumberTypeFloat32, as.Head(as.heads, key_dim));
  Dispatch("aclnnChunkKdaBwdPrepare", aqk, v_new, d_o, h, cu, indices, layout, scale, chunk_size, state_v_first, da, dv,
           dq);
  return {da, dv, dq};
}

std::vector<ms::Tensor> npu_chunk_kda_pre_process_bwd(const ms::Tensor &q, const ms::Tensor &k, const ms::Tensor &w,
                                                      const ms::Tensor &d_o, const ms::Tensor &dv, double scale,
                                                      int64_t chunk_size, const OptionalTensor &gk,
                                                      const Metadata &cu) {
  const TokenShape qs(q, "BNSD"), os(d_o, "BNSD");
  auto summary = New(ms::TypeId::kNumberTypeFloat32, {os.heads, qs.width, os.width + qs.width});
  const OptionalTensor absent;
  Dispatch("aclnnPreProcessBwdKernelMerged", q, k, w, d_o, dv, absent, gk, cu, scale, chunk_size, summary);
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
  auto dh = New(q.data_type(), {qs.batch, chunks, vs.heads, qs.width, vs.width});
  auto dh0 = New(q.data_type(), {n, vs.heads, state_v_first ? vs.width : qs.width, state_v_first ? qs.width : vs.width},
                 h0.has_value());
  auto dv_scan = New(dv.data_type(), dv.shape());
  const OptionalTensor absent;
  Dispatch("aclnnChunkGatedDeltaRuleBwdDhu", q, k, w, d_o, dv, absent, gk, h0, dht, cu, indices, scale, chunk_size,
           use_exp2, state_v_first, dh, Slot(dh0, h0.has_value()), dv_scan);
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
  auto dq = New(q.data_type(), qs.Output(qs.heads, qs.width, output_layout));
  auto dk = New(k.data_type(), ks.Output(ks.heads, ks.width, output_layout));
  auto dv = New(v.data_type(), vs.Output(vs.heads, vs.width, output_layout));
  auto db = New(beta.data_type(), beta_shape);
  auto dg = New(raw_g.data_type(), qs.Output(vs.heads, qs.width, output_layout));
  auto da =
    New(ms::TypeId::kNumberTypeFloat32, a_log.has_value() ? a_log->shape() : Shape{0}, gate && a_log.has_value());
  auto dbias =
    New(ms::TypeId::kNumberTypeFloat32, dt_bias.has_value() ? dt_bias->shape() : Shape{0}, gate && dt_bias.has_value());
  Dispatch("aclnnChunkKdaBwdFinalize", q, k, v, gk, raw_g, beta, a_log, dt_bias, akk, v_new, h, dh, dv_scan, d_aqk,
           dq_raw, q_rstd, k_rstd, cu, indices, scale, lower_bound, chunk_size, safe_gate, gate, use_exp2,
           state_v_first, beta_sigmoid, neg_eigval, output_layout, dq, dk, dv, db, dg,
           Slot(da, gate && a_log.has_value()), Slot(dbias, gate && dt_bias.has_value()));
  return {dq, dk, dv, db, dg, da, dbias};
}

std::vector<ms::Tensor> npu_chunk_kda_merge_fwd_bwd(const ms::Tensor &state, const ms::Tensor &summaries, int64_t count,
                                                    int64_t rank, bool forward, bool state_v_first) {
  Dispatch("aclnnMergeFwdBwdKernel", state, summaries, count, rank, forward, state_v_first);
  return {state};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("has_chunk_kda_stage", [](const std::string &api) {
    return mindspore::device::ascend::GetOpApiFunc(api.c_str()) != nullptr &&
           mindspore::device::ascend::GetOpApiFunc((api + "GetWorkspaceSize").c_str()) != nullptr;
  });
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
