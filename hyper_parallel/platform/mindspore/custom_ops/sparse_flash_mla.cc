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

#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include "ms_extension/all.h"
#include "ir/tensor_new.h"
#include "module.h"

namespace custom {
namespace {
constexpr int64_t kMetadataSize = 1024;

ms::Tensor EmptyTensor(const std::optional<ms::Tensor> &tensor_opt) { return tensor_opt.value_or(ms::Tensor()); }

ms::Tensor EmptyFloat32Tensor() {
    static const std::vector<float> empty;
    auto value = mindspore::tensor::from_buffer(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{0},
                                                const_cast<float *>(empty.data()), 0);
    return ms::Tensor(value);
}

std::vector<int64_t> SoftmaxLseShape(const ms::Tensor &query) {
    auto shape = query.shape();
    if (!shape.empty()) {
        shape.back() = 1;
    }
    return shape;
}

int64_t DimOrZero(const std::optional<ms::Tensor> &tensor_opt, size_t axis) {
    if (!tensor_opt.has_value()) {
        return 0;
    }
    auto shape = tensor_opt->shape();
    return axis < shape.size() ? shape[axis] : 0;
}
}  // namespace

std::vector<ms::Tensor> npu_sparse_flash_mla(const ms::Tensor &query,
    const std::optional<ms::Tensor> &ori_kv_opt, const std::optional<ms::Tensor> &cmp_kv_opt,
    const std::optional<ms::Tensor> &ori_sparse_indices_opt,
    const std::optional<ms::Tensor> &cmp_sparse_indices_opt, const std::optional<ms::Tensor> &ori_block_table_opt,
    const std::optional<ms::Tensor> &cmp_block_table_opt, const std::optional<ms::Tensor> &cu_seq_lens_q_opt,
    const std::optional<ms::Tensor> &cu_seq_lens_ori_kv_opt,
    const std::optional<ms::Tensor> &cu_seq_lens_cmp_kv_opt, const std::optional<ms::Tensor> &seq_used_q_opt,
    const std::optional<ms::Tensor> &seq_used_ori_kv_opt, const std::optional<ms::Tensor> &seq_used_cmp_kv_opt,
    const std::optional<ms::Tensor> &cmp_residual_kv_opt, const std::optional<ms::Tensor> &ori_topk_length_opt,
    const std::optional<ms::Tensor> &cmp_topk_length_opt, const std::optional<ms::Tensor> &sinks_opt,
    double softmax_scale, int64_t cmp_ratio, int64_t ori_mask_mode,
    int64_t cmp_mask_mode, int64_t ori_win_left, int64_t ori_win_right,
    const std::optional<std::string> &layout_q_opt, const std::optional<std::string> &layout_kv_opt,
    int64_t topk_value_mode, bool return_softmax_lse) {
    std::string layout_q = layout_q_opt.value_or("BSND");
    std::string layout_kv = layout_kv_opt.value_or("BSND");

    // Metadata scalars are derived from the tensor shapes; the aicpu metadata
    // dispatch runs inline on this op's stream so its output is ordered before
    // the consuming aclnnSparseFlashMla read.
    auto q_shape = query.shape();
    int64_t head_dim = q_shape.empty() ? 0 : q_shape.back();
    int64_t num_heads_q = 0;
    int64_t batch_size = 0;
    int64_t max_seqlen_q = 0;
    int64_t max_seqlen_ori_kv = 0;
    int64_t max_seqlen_cmp_kv = 0;
    if (layout_q == "BSND") {
        batch_size = q_shape.size() > 0 ? q_shape[0] : 0;
        max_seqlen_q = q_shape.size() > 1 ? q_shape[1] : 0;
        num_heads_q = q_shape.size() > 2 ? q_shape[2] : 0;
        max_seqlen_ori_kv = ori_kv_opt.has_value() ? DimOrZero(ori_kv_opt, 1) : DimOrZero(cmp_kv_opt, 1);
        max_seqlen_cmp_kv = DimOrZero(cmp_kv_opt, 1);
    } else {  // TND — seq lengths derived from cu_seqlens inside the kernel; pass 0 hints.
        num_heads_q = q_shape.size() > 1 ? q_shape[1] : 0;
    }
    int64_t num_heads_kv = 1;
    int64_t ori_topk = 0;
    int64_t cmp_topk = 0;
    if (cmp_sparse_indices_opt.has_value()) {
        auto si_shape = cmp_sparse_indices_opt->shape();
        cmp_topk = si_shape.empty() ? 0 : si_shape.back();
    }
    bool has_ori_kv = ori_kv_opt.has_value();
    bool has_cmp_kv = cmp_kv_opt.has_value();

    auto metadata = ms::Tensor(ms::TypeId::kNumberTypeInt32, std::vector<int64_t>{kMetadataSize});
    auto attn_output = ms::Tensor(query.data_type(), query.shape());
    auto softmax_lse_out =
        return_softmax_lse ? ms::Tensor(ms::TypeId::kNumberTypeFloat32, SoftmaxLseShape(query)) : EmptyFloat32Tensor();

    auto ori_kv = EmptyTensor(ori_kv_opt);
    auto cmp_kv = EmptyTensor(cmp_kv_opt);
    auto ori_sparse_indices = EmptyTensor(ori_sparse_indices_opt);
    auto cmp_sparse_indices = EmptyTensor(cmp_sparse_indices_opt);
    auto ori_block_table = EmptyTensor(ori_block_table_opt);
    auto cmp_block_table = EmptyTensor(cmp_block_table_opt);
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

    ms::TensorToDevice(query, ori_kv, cmp_kv, ori_sparse_indices, cmp_sparse_indices, ori_block_table, cmp_block_table,
        cu_seq_lens_q, cu_seq_lens_ori_kv, cu_seq_lens_cmp_kv, seq_used_q, seq_used_ori_kv, seq_used_cmp_kv,
        cmp_residual_kv, ori_topk_length, cmp_topk_length, sinks);
    ms::TensorAllocate({metadata, attn_output, softmax_lse_out});
    MS_DISPATCH_ACLNN(aclnnSparseFlashMlaMetadata, cu_seq_lens_q_opt, cu_seq_lens_ori_kv_opt, cu_seq_lens_cmp_kv_opt,
        seq_used_q_opt, seq_used_ori_kv_opt, seq_used_cmp_kv_opt, cmp_residual_kv_opt, ori_topk_length_opt,
        cmp_topk_length_opt, num_heads_q, num_heads_kv, head_dim, batch_size, max_seqlen_q, max_seqlen_ori_kv,
        max_seqlen_cmp_kv, ori_topk, cmp_topk, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
        layout_q, layout_kv, has_ori_kv, has_cmp_kv, metadata);
    MS_DISPATCH_ACLNN(aclnnSparseFlashMla, query, ori_kv_opt, cmp_kv_opt, ori_sparse_indices_opt,
        cmp_sparse_indices_opt, ori_block_table_opt, cmp_block_table_opt, cu_seq_lens_q_opt, cu_seq_lens_ori_kv_opt,
        cu_seq_lens_cmp_kv_opt, seq_used_q_opt, seq_used_ori_kv_opt, seq_used_cmp_kv_opt, cmp_residual_kv_opt,
        ori_topk_length_opt, cmp_topk_length_opt, sinks_opt, metadata, softmax_scale, cmp_ratio, ori_mask_mode,
        cmp_mask_mode, ori_win_left, ori_win_right, layout_q, layout_kv, topk_value_mode, return_softmax_lse,
        attn_output, softmax_lse_out);
    return {attn_output, softmax_lse_out};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
    m.def("npu_sparse_flash_mla", PYBOOST_CALLER(2, custom::npu_sparse_flash_mla));
    m.def("sparse_flash_mla", PYBOOST_CALLER(2, custom::npu_sparse_flash_mla));
}
}  // namespace custom
