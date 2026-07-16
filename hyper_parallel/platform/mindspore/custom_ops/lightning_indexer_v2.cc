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
#include <tuple>
#include <vector>
#include "ms_extension/all.h"
#include "module.h"

namespace custom {
namespace {
constexpr size_t kBatchDim = 0;
constexpr size_t kSeqDim = 1;
constexpr size_t kHeadDim = 2;
constexpr size_t kTndHeadDim = 1;

ms::Tensor EmptyTensor(const std::optional<ms::Tensor> &tensor_opt) { return tensor_opt.value_or(ms::Tensor()); }

std::vector<int64_t> OutputShape(const ms::Tensor &query, const ms::Tensor &key, const std::string &layout_query,
    const std::string &layout_key, int64_t topk) {
    const auto &query_shape = query.shape();
    const auto &key_shape = key.shape();
    const size_t key_head_dim = layout_key == "TND" ? kTndHeadDim : kHeadDim;
    if (layout_query == "BSND") {
        return {query_shape[kBatchDim], query_shape[kSeqDim], key_shape[key_head_dim], topk};
    }
    return {query_shape[kBatchDim], key_shape[key_head_dim], topk};
}

std::tuple<ms::Tensor, ms::Tensor> GenResultTensors(const ms::Tensor &query, const ms::Tensor &key,
    const std::string &layout_query, const std::string &layout_key, int64_t topk) {
    auto out_shape = OutputShape(query, key, layout_query, layout_key, topk);
    auto sparse_indices = ms::Tensor(ms::TypeId::kNumberTypeInt32, out_shape);
    // Always allocate the FULL value buffer, even when return_value is false.  The
    // underlying aclnn kernel unconditionally writes inf into unfilled top-k
    // padding slots of sparse_values; a 0-size buffer here lets those writes
    // overflow into the adjacent device allocation (observed corrupting a
    // neighbouring Parameter such as attn_sink -> nan at the first training step).
    // Sizing it fully contains the writes; the caller ignores sparse_values when
    // return_value is false.
    auto sparse_values = ms::Tensor(ms::TypeId::kNumberTypeFloat32, out_shape);
    return std::make_tuple(std::move(sparse_indices), std::move(sparse_values));
}
}  // namespace

std::vector<ms::Tensor> npu_lightning_indexer_v2(const ms::Tensor &query, const ms::Tensor &key,
    const ms::Tensor &weights, int64_t topk, const std::optional<ms::Tensor> &cu_seqlens_q_opt,
    const std::optional<ms::Tensor> &cu_seqlens_k_opt, const std::optional<ms::Tensor> &seqused_q_opt,
    const std::optional<ms::Tensor> &seqused_k_opt, const std::optional<ms::Tensor> &cmp_residual_k_opt,
    const std::optional<ms::Tensor> &block_table_opt, const std::optional<ms::Tensor> &output_idx_offset_opt,
    const std::optional<ms::Tensor> &metadata_opt, const std::optional<int64_t> &max_seqlen_q_opt,
    const std::optional<std::string> &layout_q_opt, const std::optional<std::string> &layout_k_opt,
    const std::optional<int64_t> &mask_mode_opt, const std::optional<int64_t> &cmp_ratio_opt,
    bool return_value) {
    std::string layout_q = layout_q_opt.value_or("BSND");
    std::string layout_k = layout_k_opt.value_or("BSND");
    int64_t max_seqlen_q = max_seqlen_q_opt.value_or(-1);
    int64_t mask_mode = mask_mode_opt.value_or(0);
    int64_t cmp_ratio = cmp_ratio_opt.value_or(1);

    auto [sparse_indices, sparse_values] = GenResultTensors(query, key, layout_q, layout_k, topk);

    auto cu_seqlens_q = EmptyTensor(cu_seqlens_q_opt);
    auto cu_seqlens_k = EmptyTensor(cu_seqlens_k_opt);
    auto seqused_q = EmptyTensor(seqused_q_opt);
    auto seqused_k = EmptyTensor(seqused_k_opt);
    auto cmp_residual_k = EmptyTensor(cmp_residual_k_opt);
    auto block_table = EmptyTensor(block_table_opt);
    auto output_idx_offset = EmptyTensor(output_idx_offset_opt);
    auto metadata = EmptyTensor(metadata_opt);

    ms::TensorToDevice(query, key, weights, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k,
        block_table, output_idx_offset, metadata);
    ms::TensorAllocate({sparse_indices, sparse_values});
    MS_DISPATCH_ACLNN(aclnnLightningIndexerV2, query, key, weights, cu_seqlens_q_opt, cu_seqlens_k_opt, seqused_q_opt,
        seqused_k_opt, cmp_residual_k_opt, block_table_opt, output_idx_offset_opt, metadata_opt, topk, max_seqlen_q,
        layout_q, layout_k, mask_mode, cmp_ratio, return_value, sparse_indices, sparse_values);
    return {sparse_indices, sparse_values};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
    m.def("npu_lightning_indexer_v2", PYBOOST_CALLER(2, custom::npu_lightning_indexer_v2));
}
}  // namespace custom
