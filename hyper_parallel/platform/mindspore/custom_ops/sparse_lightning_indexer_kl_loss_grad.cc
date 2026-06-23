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
#include <tuple>
#include <vector>
#include "ms_extension/all.h"
#include "module.h"

namespace custom {
namespace {
constexpr int64_t kMetadataSize = 64;

ms::Tensor EmptyTensor(const std::optional<ms::Tensor> &tensor_opt) { return tensor_opt.value_or(ms::Tensor()); }

std::tuple<ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor> GenResultTensors(
    const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &weights, const ms::Tensor &attn_softmax_l1_norm) {
    auto dq = ms::Tensor(query.data_type(), query.shape());
    auto dk = ms::Tensor(key.data_type(), key.shape());
    auto dw = ms::Tensor(ms::TypeId::kNumberTypeFloat32, weights.shape());
    auto softmax_out = ms::Tensor(ms::TypeId::kNumberTypeFloat32, attn_softmax_l1_norm.shape());
    return std::make_tuple(std::move(dq), std::move(dk), std::move(dw), std::move(softmax_out));
}
}  // namespace

std::vector<ms::Tensor> npu_sparse_lightning_indexer_kl_loss_grad(const ms::Tensor &query, const ms::Tensor &key,
    const ms::Tensor &weights, const ms::Tensor &sparse_indices, const ms::Tensor &attn_softmax_l1_norm,
    const std::optional<ms::Tensor> &cu_seqlens_q_opt, const std::optional<ms::Tensor> &cu_seqlens_k_opt,
    const std::optional<ms::Tensor> &seqused_q_opt, const std::optional<ms::Tensor> &seqused_k_opt,
    const std::optional<ms::Tensor> &cmp_residual_k_opt,
    const std::optional<std::string> &layout_q_opt, const std::optional<std::string> &layout_k_opt,
    const std::optional<int64_t> &mask_mode_opt, const std::optional<int64_t> &cmp_ratio_opt) {
    auto [dq, dk, dw, softmax_out] = GenResultTensors(query, key, weights, attn_softmax_l1_norm);

    std::string layout_q = layout_q_opt.value_or("TND");
    std::string layout_k = layout_k_opt.value_or("TND");
    int64_t mask_mode = mask_mode_opt.value_or(3);
    int64_t cmp_ratio = cmp_ratio_opt.value_or(1);

    // Metadata scalars are derived from the tensor shapes; the aicpu metadata
    // dispatch runs inline on this op's stream so its output is ordered before
    // the consuming aclnnSparseLightningIndexerKLLossGrad read.
    auto q_shape = query.shape();
    auto k_shape = key.shape();
    auto si_shape = sparse_indices.shape();
    int64_t head_dim = q_shape.empty() ? 0 : q_shape.back();
    int64_t num_heads_q = 0;
    int64_t num_heads_k = 0;
    int64_t batch_size = 0;
    int64_t max_seqlen_q = 0;
    int64_t max_seqlen_k = 0;
    if (layout_q == "BSND") {
        batch_size = q_shape.size() > 0 ? q_shape[0] : 0;
        max_seqlen_q = q_shape.size() > 1 ? q_shape[1] : 0;
        num_heads_q = q_shape.size() > 2 ? q_shape[2] : 0;
        max_seqlen_k = k_shape.size() > 1 ? k_shape[1] : 0;
        num_heads_k = k_shape.size() > 2 ? k_shape[2] : 0;
    } else {  // TND — seq lengths derived from cu_seqlens inside the kernel; pass 0 hints.
        num_heads_q = q_shape.size() > 1 ? q_shape[1] : 0;
        num_heads_k = k_shape.size() > 1 ? k_shape[1] : 0;
    }
    int64_t topk = si_shape.empty() ? 0 : si_shape.back();

    auto metadata = ms::Tensor(ms::TypeId::kNumberTypeInt32, std::vector<int64_t>{kMetadataSize});

    auto cu_seqlens_q = EmptyTensor(cu_seqlens_q_opt);
    auto cu_seqlens_k = EmptyTensor(cu_seqlens_k_opt);
    auto seqused_q = EmptyTensor(seqused_q_opt);
    auto seqused_k = EmptyTensor(seqused_k_opt);
    auto cmp_residual_k = EmptyTensor(cmp_residual_k_opt);

    ms::TensorToDevice(query, key, weights, sparse_indices, attn_softmax_l1_norm, cu_seqlens_q, cu_seqlens_k,
        seqused_q, seqused_k, cmp_residual_k);
    ms::TensorAllocate({metadata, dq, dk, dw, softmax_out});
    MS_DISPATCH_ACLNN(aclnnSparseLightningIndexerKLLossGradMetadata, cu_seqlens_q_opt, cu_seqlens_k_opt, seqused_q_opt,
        seqused_k_opt, cmp_residual_k_opt, batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_k, head_dim,
        topk, layout_q, layout_k, mask_mode, cmp_ratio, metadata);
    std::optional<ms::Tensor> metadata_opt = metadata;
    MS_DISPATCH_ACLNN(aclnnSparseLightningIndexerKLLossGrad, query, key, weights, sparse_indices,
        attn_softmax_l1_norm, cu_seqlens_q_opt, cu_seqlens_k_opt, seqused_q_opt, seqused_k_opt, cmp_residual_k_opt,
        metadata_opt, layout_q, layout_k, mask_mode, cmp_ratio, dq, dk, dw, softmax_out);
    return {dq, dk, dw, softmax_out};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
    m.def("sparse_lightning_indexer_kl_loss_grad",
        PYBOOST_CALLER(4, custom::npu_sparse_lightning_indexer_kl_loss_grad));
    m.def("npu_sparse_lightning_indexer_kl_loss_grad",
        PYBOOST_CALLER(4, custom::npu_sparse_lightning_indexer_kl_loss_grad));
}
}  // namespace custom
