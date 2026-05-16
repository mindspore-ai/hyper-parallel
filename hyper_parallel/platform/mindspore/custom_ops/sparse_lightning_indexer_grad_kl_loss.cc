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

#include <set>
#include <optional>
#include "ms_extension/all.h"
#include "module.h"

namespace custom {
namespace {
ms::Tensor GetTensorOrEmpty(const std::optional<ms::Tensor> &opt_tensor) {
  return opt_tensor.has_value() ? opt_tensor.value() : ms::Tensor();
}

std::vector<int64_t> GetIntArrayOrEmpty(const std::optional<std::vector<int64_t>> &opt_arr) {
  return opt_arr.has_value() ? opt_arr.value() : std::vector<int64_t>();
}

static std::tuple<ShapeVector, ShapeVector, ShapeVector, ShapeVector> InferShape(const ShapeVector &query_index_shape,
                                                                                 const ShapeVector &key_index_shape,
                                                                                 const ShapeVector &weights_shape) {
  ShapeVector loss_shape{1};
  return {query_index_shape, key_index_shape, weights_shape, loss_shape};
}

static std::tuple<ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor> GenResultTensors(const ms::Tensor &query_index,
                                                                                   const ms::Tensor &key_index,
                                                                                   const ms::Tensor &weights) {
  auto [dqi_shape, dki_shape, dw_shape, loss_shape] =
    InferShape(query_index.shape(), key_index.shape(), weights.shape());

  ms::Tensor d_query_index(query_index.data_type(), dqi_shape);
  ms::Tensor d_key_index(key_index.data_type(), dki_shape);
  ms::Tensor d_weights(weights.data_type(), dw_shape);
  ms::Tensor loss(ms::TypeId::kNumberTypeFloat32, loss_shape);
  return {std::move(d_query_index), std::move(d_key_index), std::move(d_weights), std::move(loss)};
}
}  // namespace

class SparseLightningRunner : public ms::pynative::PyboostRunner {
 public:
  SparseLightningRunner(const mindspore::tensor::TensorPtr &query, const mindspore::tensor::TensorPtr &key,
                        const mindspore::tensor::TensorPtr &query_index, const mindspore::tensor::TensorPtr &key_index,
                        const mindspore::tensor::TensorPtr &weights, const mindspore::tensor::TensorPtr &sparse_indices,
                        const mindspore::tensor::TensorPtr &softmax_max,
                        const mindspore::tensor::TensorPtr &softmax_sum,
                        const std::optional<mindspore::tensor::TensorPtr> &query_rope,
                        const std::optional<mindspore::tensor::TensorPtr> &key_rope,
                        std::vector<int64_t> actual_seq_qlen, std::vector<int64_t> actual_seq_klen, double scale_value,
                        std::string layout, int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens,
                        bool deterministic, const mindspore::tensor::TensorPtr &d_query_index,
                        const mindspore::tensor::TensorPtr &d_key_index, const mindspore::tensor::TensorPtr &d_weights,
                        const mindspore::tensor::TensorPtr &loss)
      : PyboostRunner("aclnnSparseLightningIndexerGradKLLoss"),
        query_(query),
        key_(key),
        query_index_(query_index),
        key_index_(key_index),
        weights_(weights),
        sparse_indices_(sparse_indices),
        softmax_max_(softmax_max),
        softmax_sum_(softmax_sum),
        query_rope_(query_rope),
        key_rope_(key_rope),
        actual_seq_qlen_(std::move(actual_seq_qlen)),
        actual_seq_klen_(std::move(actual_seq_klen)),
        scale_value_(scale_value),
        layout_(std::move(layout)),
        sparse_mode_(sparse_mode),
        pre_tokens_(pre_tokens),
        next_tokens_(next_tokens),
        deterministic_(deterministic),
        d_query_index_(d_query_index),
        d_key_index_(d_key_index),
        d_weights_(d_weights),
        loss_(loss) {}

  void LaunchKernel() override {}

 protected:
  void _DispatchLaunchTask() override {
    auto self = std::static_pointer_cast<SparseLightningRunner>(shared_from_this());
    mindspore::runtime::OpExecutor::DispatchLaunchTask([self]() {
      self->_device_context_->device_res_manager_->BindDeviceToCurrentThread(false);
      self->RunAclnnDirect();
      if (mindspore::runtime::RuntimeConf::GetInstance()->launch_blocking()) {
        if (!self->_device_context_->device_res_manager_->SyncAllStreams()) {
          MS_LOG(EXCEPTION) << "SyncStream failed";
        }
      } else {
        self->ProcessCrossStreamAddress();
      }
    });
  }

 private:
  void RunAclnnDirect() {
    namespace asc = mindspore::device::ascend;

    uint64_t workspace_size = 0;
    asc::aclOpExecutor *executor = nullptr;
    uint64_t *ws_addr = &workspace_size;
    asc::aclOpExecutor **exec_addr = &executor;

    auto converted = asc::ConvertTypes(query_, key_, query_index_, key_index_, weights_, sparse_indices_, softmax_max_,
                                       softmax_sum_, query_rope_, key_rope_, actual_seq_qlen_, actual_seq_klen_,
                                       scale_value_, layout_, sparse_mode_, pre_tokens_, next_tokens_, deterministic_,
                                       d_query_index_, d_key_index_, d_weights_, loss_, ws_addr, exec_addr);

    const auto ws_func_ptr = asc::GetOpApiFunc("aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize");
    if (ws_func_ptr == nullptr) {
      asc::ReleaseConvertTypes(converted);
      MS_LOG(EXCEPTION) << "aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize "
                        << "not found in " << asc::GetOpApiLibName();
    }
    auto typed_func = asc::ConvertToOpApiFunc(converted, ws_func_ptr);
    auto ws_ret = asc::call(typed_func, converted);
    if (ws_ret != 0) {
      asc::ReleaseConvertTypes(converted);
      MS_LOG(EXCEPTION) << "aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize failed, "
                        << "ret=" << ws_ret;
    }

    void *ws_ptr = nullptr;
    if (workspace_size > 0) {
      ws_ptr = _device_context_->device_res_manager_->AllocateMemory(workspace_size, _stream_id_);
      if (ws_ptr == nullptr) {
        asc::ReleaseConvertTypes(converted);
        MS_LOG(EXCEPTION) << "Alloc workspace failed, size=" << workspace_size;
      }
    }

    const auto run_func_ptr = asc::GetOpApiFunc("aclnnSparseLightningIndexerGradKLLoss");
    if (run_func_ptr == nullptr) {
      if (ws_ptr) {
        _device_context_->device_res_manager_->FreeMemory(ws_ptr);
      }
      asc::ReleaseConvertTypes(converted);
      MS_LOG(EXCEPTION) << "aclnnSparseLightningIndexerGradKLLoss "
                        << "not found in " << asc::GetOpApiLibName();
    }
    auto run_func = reinterpret_cast<asc::RunApiFunc>(run_func_ptr);
    auto api_ret = run_func(ws_ptr, workspace_size, executor, _stream_);

    if (ws_ptr) {
      _device_context_->device_res_manager_->FreeMemory(ws_ptr);
    }
    asc::ReleaseConvertTypes(converted);

    if (api_ret != 0) {
      MS_LOG(EXCEPTION) << "aclnnSparseLightningIndexerGradKLLoss failed, ret=" << api_ret;
    }
  }

  mindspore::tensor::TensorPtr query_, key_, query_index_, key_index_;
  mindspore::tensor::TensorPtr weights_, sparse_indices_;
  mindspore::tensor::TensorPtr softmax_max_, softmax_sum_;
  std::optional<mindspore::tensor::TensorPtr> query_rope_, key_rope_;
  std::vector<int64_t> actual_seq_qlen_, actual_seq_klen_;
  double scale_value_;
  std::string layout_;
  int64_t sparse_mode_, pre_tokens_, next_tokens_;
  bool deterministic_;
  mindspore::tensor::TensorPtr d_query_index_, d_key_index_;
  mindspore::tensor::TensorPtr d_weights_, loss_;
};

std::vector<ms::Tensor> npu_sparse_lightning_indexer_grad_kl_loss(
  const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &query_index, const ms::Tensor &key_index,
  const ms::Tensor &weights, const ms::Tensor &sparse_indices, const ms::Tensor &softmax_max,
  const ms::Tensor &softmax_sum, double scale_value, const std::optional<ms::Tensor> &query_rope,
  const std::optional<ms::Tensor> &key_rope, const std::optional<std::vector<int64_t>> &actual_seq_qlen,
  const std::optional<std::vector<int64_t>> &actual_seq_klen, const std::optional<std::string> &layout,
  const std::optional<int64_t> &sparse_mode, const std::optional<int64_t> &pre_tokens,
  const std::optional<int64_t> &next_tokens) {
  auto [d_query_index, d_key_index, d_weights, loss] = GenResultTensors(query_index, key_index, weights);

  auto runner = std::make_shared<SparseLightningRunner>(
    query.tensor(), key.tensor(), query_index.tensor(), key_index.tensor(), weights.tensor(), sparse_indices.tensor(),
    softmax_max.tensor(), softmax_sum.tensor(),
    query_rope.has_value() ? std::optional<mindspore::tensor::TensorPtr>(query_rope->tensor()) : std::nullopt,
    key_rope.has_value()
        ? std::optional<mindspore::tensor::TensorPtr>(key_rope->tensor())
        : std::nullopt,
    GetIntArrayOrEmpty(actual_seq_qlen),
    GetIntArrayOrEmpty(actual_seq_klen),
    scale_value,
    layout.value_or("BSND"),
    sparse_mode.value_or(3),
    pre_tokens.value_or(9223372036854775807),
    next_tokens.value_or(9223372036854775807),
    true,
    d_query_index.tensor(),
    d_key_index.tensor(),
    d_weights.tensor(),
    loss.tensor());

  std::vector<ms::Tensor> inputs_vec = {query,
                                        key,
                                        query_index,
                                        key_index,
                                        weights,
                                        sparse_indices,
                                        softmax_max,
                                        softmax_sum,
                                        GetTensorOrEmpty(query_rope),
                                        GetTensorOrEmpty(key_rope)};
  std::vector<ms::Tensor> outputs_vec = {d_query_index, d_key_index, d_weights, loss};

  runner->Run(inputs_vec, outputs_vec);
  return {d_query_index, d_key_index, d_weights, loss};
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("npu_sparse_lightning_indexer_grad_kl_loss",
        PYBOOST_CALLER(4, custom::npu_sparse_lightning_indexer_grad_kl_loss));
}
}  // namespace custom
