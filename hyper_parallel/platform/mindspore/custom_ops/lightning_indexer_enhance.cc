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

/*!
 * \file lightning_indexer_enhance.cc
 * \brief MindSpore binding for the omni-ops ``aclnnLightningIndexerEnhance`` custom op.
 *
 * Drop-in for the MindSpore built-in ``ops.lightning_indexer`` call site: same positional
 * (query, key, weights) + keyword (actual_seq_lengths_*, layout_*, sparse_count,
 * return_value) convention, plus the two attributes the enhanced kernel adds
 * (``sparse_block_size`` / ``sparse_block_mode``). Keeping that convention is required by
 * ``dsa_cp_fold.fold_lightning_indexer``, which calls the op once per folded block as
 * ``func(q_block, k_prefix, w_block, *rest, **kwargs)``.
 */
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "ms_extension/all.h"
#include "module.h"

// A CANN custom package installs its aclnn headers flat in ``op_api/include``, while a
// built-in install puts them under an ``aclnnop`` sub-directory. Accept both; the include
// stays optional so that a host without the package still compiles the rest of the extension
// (this file is only listed in the build when the header is actually present).
#if __has_include("aclnnop/aclnn_lightning_indexer_enhance.h")
#include "aclnnop/aclnn_lightning_indexer_enhance.h"
#elif __has_include("aclnn_lightning_indexer_enhance.h")
#include "aclnn_lightning_indexer_enhance.h"
#endif

namespace custom {
namespace {
constexpr size_t kBatchDim = 0;
constexpr size_t kSeqDim = 1;
constexpr size_t kHeadDim = 2;
constexpr size_t kTndHeadDim = 1;

ms::Tensor EmptyTensor(const std::optional<ms::Tensor> &tensor_opt) { return tensor_opt.value_or(ms::Tensor()); }

std::vector<int64_t> OutputShape(const ms::Tensor &query, const ms::Tensor &key, const std::string &layout_query,
    const std::string &layout_key, int64_t sparse_count, int64_t sparse_block_size) {
    const auto &query_shape = query.shape();
    const auto &key_shape = key.shape();
    const int64_t block = sparse_block_size > 0 ? sparse_block_size : 1;
    const int64_t real_sparse_count = (sparse_count + block - 1) / block;
    const size_t key_head_dim = layout_key == "TND" ? kTndHeadDim : kHeadDim;
    if (layout_query == "BSND") {
        return {query_shape[kBatchDim], query_shape[kSeqDim], key_shape[key_head_dim], real_sparse_count};
    }
    return {query_shape[kBatchDim], key_shape[key_head_dim], real_sparse_count};
}

std::tuple<ms::Tensor, ms::Tensor> GenResultTensors(const ms::Tensor &query, const ms::Tensor &key,
    const std::string &layout_query, const std::string &layout_key, int64_t sparse_count,
    int64_t sparse_block_size) {
    auto out_shape = OutputShape(query, key, layout_query, layout_key, sparse_count, sparse_block_size);
    auto sparse_indices = ms::Tensor(ms::TypeId::kNumberTypeInt32, out_shape);
    // Always allocate the FULL value buffer, even when return_value is false, mirroring the
    // lightning_indexer_v2 wrapper: a 0-size buffer here would let any kernel-side write into
    // unfilled top-k padding slots overflow the adjacent device allocation. The caller ignores
    // sparse_values when return_value is false.
    auto sparse_values = ms::Tensor(query.data_type(), out_shape);
    return std::make_tuple(std::move(sparse_indices), std::move(sparse_values));
}
}  // namespace

std::vector<ms::Tensor> npu_lightning_indexer_enhance(const ms::Tensor &query, const ms::Tensor &key,
    const ms::Tensor &weights, const std::optional<ms::Tensor> &actual_seq_lengths_query_opt,
    const std::optional<ms::Tensor> &actual_seq_lengths_key_opt, const std::optional<ms::Tensor> &block_table_opt,
    const std::optional<std::string> &layout_query_opt, const std::optional<std::string> &layout_key_opt,
    int64_t sparse_count, int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, bool return_value,
    int64_t sparse_block_size, int64_t sparse_block_mode) {
    std::string layout_query = layout_query_opt.value_or("BSND");
    std::string layout_key = layout_key_opt.value_or("BSND");
    int64_t block_size = sparse_block_size > 0 ? sparse_block_size : 1;

    auto [sparse_indices, sparse_values] =
        GenResultTensors(query, key, layout_query, layout_key, sparse_count, block_size);

    auto actual_seq_lengths_query = EmptyTensor(actual_seq_lengths_query_opt);
    auto actual_seq_lengths_key = EmptyTensor(actual_seq_lengths_key_opt);
    auto block_table = EmptyTensor(block_table_opt);

    ms::TensorToDevice(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table);
    ms::TensorAllocate({sparse_indices, sparse_values});
    MS_DISPATCH_ACLNN(aclnnLightningIndexerEnhance, query, key, weights, actual_seq_lengths_query_opt,
        actual_seq_lengths_key_opt, block_table_opt, layout_query, layout_key, sparse_count, sparse_mode, pre_tokens,
        next_tokens, return_value, block_size, sparse_block_mode, sparse_indices, sparse_values);
    return {sparse_indices, sparse_values};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
    m.def("npu_lightning_indexer_enhance", PYBOOST_CALLER(2, custom::npu_lightning_indexer_enhance));
}
}  // namespace custom
