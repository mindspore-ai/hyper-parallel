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
ms::Tensor GetTensorOrEmpty(const std::optional<ms::Tensor> &opt_tensor) {
  return opt_tensor.has_value() ? opt_tensor.value() : ms::Tensor();
}

std::vector<int64_t> GetIntArrayOrEmpty(const std::optional<std::vector<int64_t>> &opt_arr) {
  return opt_arr.has_value() ? opt_arr.value() : std::vector<int64_t>();
}

static std::tuple<ms::Tensor, ms::Tensor> GenResultTensors(const ms::Tensor &query_index, const ms::Tensor &key_index,
                                                           const std::string &layout) {
  std::vector<int64_t> out_shape;
  if (layout == "TND") {
    out_shape = {key_index.shape()[1], query_index.shape()[0]};
  } else {
    out_shape = {query_index.shape()[0], key_index.shape()[2], query_index.shape()[1]};
  }
  ms::Tensor softmax_max_out(ms::TypeId::kNumberTypeFloat32, out_shape);
  ms::Tensor softmax_sum_out(ms::TypeId::kNumberTypeFloat32, out_shape);
  return {std::move(softmax_max_out), std::move(softmax_sum_out)};
}
}  // namespace
class DenseLightningIndexerSoftmaxLseRunner : public ms::pynative::PyboostRunner {
 public:
  DenseLightningIndexerSoftmaxLseRunner(const mindspore::tensor::TensorPtr &query_index,
                                        const mindspore::tensor::TensorPtr &key_index,
                                        const mindspore::tensor::TensorPtr &weight,
                                        const std::optional<std::vector<int64_t>> &actual_seq_qlen,
                                        const std::optional<std::vector<int64_t>> &actual_seq_klen,
                                        const std::string &layout, int64_t sparse_mode, int64_t pre_tokens,
                                        int64_t next_tokens, const mindspore::tensor::TensorPtr &softmax_max_out,
                                        const mindspore::tensor::TensorPtr &softmax_sum_out)
      : PyboostRunner("aclnnDenseLightningIndexerSoftmaxLse"),
        query_index_(query_index),
        key_index_(key_index),
        weight_(weight),
        actual_seq_qlen_(actual_seq_qlen),
        actual_seq_klen_(actual_seq_klen),
        layout_(layout),
        sparse_mode_(sparse_mode),
        pre_tokens_(pre_tokens),
        next_tokens_(next_tokens),
        softmax_max_out_(softmax_max_out),
        softmax_sum_out_(softmax_sum_out) {}

  void LaunchKernel() override {}

 protected:
  void _DispatchLaunchTask() override {
    auto self = std::static_pointer_cast<DenseLightningIndexerSoftmaxLseRunner>(shared_from_this());
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

    auto converted =
      asc::ConvertTypes(query_index_, key_index_, weight_, actual_seq_qlen_, actual_seq_klen_, layout_, sparse_mode_,
                        pre_tokens_, next_tokens_, softmax_max_out_, softmax_sum_out_, ws_addr, exec_addr);

    const auto ws_func_ptr = asc::GetOpApiFunc("aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize");
    if (ws_func_ptr == nullptr) {
      asc::ReleaseConvertTypes(converted);
      MS_LOG(EXCEPTION) << "aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize not found";
    }
    auto typed_func = asc::ConvertToOpApiFunc(converted, ws_func_ptr);
    auto ws_ret = asc::call(typed_func, converted);
    if (ws_ret != 0) {
      asc::ReleaseConvertTypes(converted);
      MS_LOG(EXCEPTION) << "GetWorkspaceSize failed, ret=" << ws_ret;
    }

    void *ws_ptr = nullptr;
    if (workspace_size > 0) {
      ws_ptr = _device_context_->device_res_manager_->AllocateMemory(workspace_size, _stream_id_);
      if (ws_ptr == nullptr) {
        asc::ReleaseConvertTypes(converted);
        MS_LOG(EXCEPTION) << "Alloc workspace failed, size=" << workspace_size;
      }
    }

    const auto run_func_ptr = asc::GetOpApiFunc("aclnnDenseLightningIndexerSoftmaxLse");
    if (run_func_ptr == nullptr) {
      if (ws_ptr) _device_context_->device_res_manager_->FreeMemory(ws_ptr);
      asc::ReleaseConvertTypes(converted);
      MS_LOG(EXCEPTION) << "aclnnDenseLightningIndexerSoftmaxLse not found";
    }
    auto run_func = reinterpret_cast<asc::RunApiFunc>(run_func_ptr);
    auto api_ret = run_func(ws_ptr, workspace_size, executor, _stream_);

    if (ws_ptr) _device_context_->device_res_manager_->FreeMemory(ws_ptr);
    asc::ReleaseConvertTypes(converted);

    if (api_ret != 0) {
      MS_LOG(EXCEPTION) << "aclnnDenseLightningIndexerSoftmaxLse failed, ret=" << api_ret;
    }
  }

  mindspore::tensor::TensorPtr query_index_;
  mindspore::tensor::TensorPtr key_index_;
  mindspore::tensor::TensorPtr weight_;
  std::optional<std::vector<int64_t>> actual_seq_qlen_;
  std::optional<std::vector<int64_t>> actual_seq_klen_;
  std::string layout_;
  int64_t sparse_mode_;
  int64_t pre_tokens_;
  int64_t next_tokens_;
  mindspore::tensor::TensorPtr softmax_max_out_;
  mindspore::tensor::TensorPtr softmax_sum_out_;
};

std::vector<ms::Tensor> npu_dense_lightning_indexer_softmax_lse(
  const ms::Tensor &query_index, const ms::Tensor &key_index, const ms::Tensor &weight,
  const std::optional<std::vector<int64_t>> &actual_seq_qlen,
  const std::optional<std::vector<int64_t>> &actual_seq_klen, const std::optional<std::string> &layout,
  const std::optional<int64_t> &sparse_mode, const std::optional<int64_t> &pre_tokens,
  const std::optional<int64_t> &next_tokens) {
  std::string layout_str = layout.value_or("BSND");
  int64_t sparse_mode_val = sparse_mode.value_or(3);
  int64_t pre_tokens_val = pre_tokens.value_or(9223372036854775807LL);
  int64_t next_tokens_val = next_tokens.value_or(9223372036854775807LL);

  auto [out0, out1] = GenResultTensors(query_index, key_index, layout_str);

  auto runner = std::make_shared<DenseLightningIndexerSoftmaxLseRunner>(
    query_index.tensor(), key_index.tensor(), weight.tensor(), actual_seq_qlen, actual_seq_klen, layout_str,
    sparse_mode_val, pre_tokens_val, next_tokens_val, out0.tensor(), out1.tensor());

  std::vector<ms::Tensor> inputs_vec = {query_index, key_index, weight};
  std::vector<ms::Tensor> outputs_vec = {out0, out1};

  runner->Run(inputs_vec, outputs_vec);
  return {out0, out1};
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("npu_dense_lightning_indexer_softmax_lse", PYBOOST_CALLER(2, custom::npu_dense_lightning_indexer_softmax_lse));
}
}  // namespace custom
