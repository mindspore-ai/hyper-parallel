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
#include <string>
#include <vector>
#include "ms_extension/all.h"
#include "module.h"

namespace custom {
namespace {
ms::Tensor EmptyTensor(const std::optional<ms::Tensor> &tensor_opt) { return tensor_opt.value_or(ms::Tensor()); }

ms::Tensor OptionalLikeOrEmpty(const std::optional<ms::Tensor> &tensor_opt, ms::TypeId dtype) {
    if (tensor_opt.has_value()) {
        return ms::Tensor(tensor_opt.value().data_type(), tensor_opt.value().shape());
    }
    return ms::Tensor(dtype, std::vector<int64_t>{0});
}

ms::Tensor OptionalFloatLikeOrEmpty(const std::optional<ms::Tensor> &tensor_opt) {
    if (tensor_opt.has_value()) {
        return ms::Tensor(ms::TypeId::kNumberTypeFloat32, tensor_opt.value().shape());
    }
    return ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{0});
}
}  // namespace

std::vector<ms::Tensor> npu_sparse_flash_mla_grad(const ms::Tensor &query, const ms::Tensor &dout,
    const ms::Tensor &attn_out, const ms::Tensor &softmax_lse, const std::optional<ms::Tensor> &ori_kv_opt,
    const std::optional<ms::Tensor> &cmp_kv_opt, const std::optional<ms::Tensor> &ori_sparse_indices_opt,
    const std::optional<ms::Tensor> &cmp_sparse_indices_opt, const std::optional<ms::Tensor> &cu_seq_lens_q_opt,
    const std::optional<ms::Tensor> &cu_seq_lens_ori_kv_opt,
    const std::optional<ms::Tensor> &cu_seq_lens_cmp_kv_opt, const std::optional<ms::Tensor> &seq_used_q_opt,
    const std::optional<ms::Tensor> &seq_used_ori_kv_opt, const std::optional<ms::Tensor> &seq_used_cmp_kv_opt,
    const std::optional<ms::Tensor> &cmp_residual_kv_opt, const std::optional<ms::Tensor> &ori_topk_length_opt,
    const std::optional<ms::Tensor> &cmp_topk_length_opt, const std::optional<ms::Tensor> &sinks_opt,
    const std::optional<ms::Tensor> &metadata_opt, double scale_value, int64_t cmp_ratio, int64_t ori_mask_mode,
    int64_t cmp_mask_mode, int64_t ori_win_left, int64_t ori_win_right,
    const std::optional<std::string> &layout_q_opt, const std::optional<std::string> &layout_kv_opt) {
    auto d_query = ms::Tensor(query.data_type(), query.shape());
    auto d_ori_kv = OptionalLikeOrEmpty(ori_kv_opt, query.data_type());
    auto d_cmp_kv = OptionalLikeOrEmpty(cmp_kv_opt, query.data_type());
    auto d_sinks = OptionalLikeOrEmpty(sinks_opt, ms::TypeId::kNumberTypeFloat32);
    auto ori_softmax_l1_norm = OptionalFloatLikeOrEmpty(ori_sparse_indices_opt);
    auto cmp_softmax_l1_norm = OptionalFloatLikeOrEmpty(cmp_sparse_indices_opt);

    auto ori_kv = EmptyTensor(ori_kv_opt);
    auto cmp_kv = EmptyTensor(cmp_kv_opt);
    auto ori_sparse_indices = EmptyTensor(ori_sparse_indices_opt);
    auto cmp_sparse_indices = EmptyTensor(cmp_sparse_indices_opt);
    auto cu_seq_lens_q = EmptyTensor(cu_seq_lens_q_opt);
    auto cu_seq_lens_ori_kv = EmptyTensor(cu_seq_lens_ori_kv_opt);
    auto cu_seq_lens_cmp_kv = EmptyTensor(cu_seq_lens_cmp_kv_opt);
    auto seq_used_q = EmptyTensor(seq_used_q_opt);
    auto seq_used_ori_kv = EmptyTensor(seq_used_ori_kv_opt);
    auto seq_used_cmp_kv = EmptyTensor(seq_used_cmp_kv_opt);
    auto cmp_residual_kv = EmptyTensor(cmp_residual_kv_opt);
    auto ori_topk_length = EmptyTensor(ori_topk_length_opt);
    auto cmp_topk_length = EmptyTensor(cmp_topk_length_opt);
    auto sinks = EmptyTensor(sinks_opt);
    auto metadata = EmptyTensor(metadata_opt);
    std::string layout_q = layout_q_opt.value_or("BSND");
    std::string layout_kv = layout_kv_opt.value_or("BSND");

    ms::TensorToDevice(query, dout, attn_out, softmax_lse, ori_kv, cmp_kv, ori_sparse_indices, cmp_sparse_indices,
        cu_seq_lens_q, cu_seq_lens_ori_kv, cu_seq_lens_cmp_kv, seq_used_q, seq_used_ori_kv, seq_used_cmp_kv,
        cmp_residual_kv, ori_topk_length, cmp_topk_length, sinks, metadata);
    ms::TensorAllocate({d_query, d_ori_kv, d_cmp_kv, d_sinks, ori_softmax_l1_norm, cmp_softmax_l1_norm});
    MS_DISPATCH_ACLNN(aclnnSparseFlashMlaGrad, query, dout, attn_out, softmax_lse, ori_kv_opt, cmp_kv_opt,
        ori_sparse_indices_opt, cmp_sparse_indices_opt, cu_seq_lens_q_opt, cu_seq_lens_ori_kv_opt,
        cu_seq_lens_cmp_kv_opt, seq_used_q_opt, seq_used_ori_kv_opt, seq_used_cmp_kv_opt, cmp_residual_kv_opt,
        ori_topk_length_opt, cmp_topk_length_opt, sinks_opt, metadata_opt, scale_value, cmp_ratio, ori_mask_mode,
        cmp_mask_mode, ori_win_left, ori_win_right, layout_q, layout_kv, d_query, d_ori_kv, d_cmp_kv, d_sinks,
        ori_softmax_l1_norm, cmp_softmax_l1_norm);
    return {d_query, d_ori_kv, d_cmp_kv, d_sinks, ori_softmax_l1_norm, cmp_softmax_l1_norm};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
    m.def("npu_sparse_flash_mla_grad", PYBOOST_CALLER(6, custom::npu_sparse_flash_mla_grad));
    m.def("sparse_flash_mla_grad", PYBOOST_CALLER(6, custom::npu_sparse_flash_mla_grad));
}
}  // namespace custom
