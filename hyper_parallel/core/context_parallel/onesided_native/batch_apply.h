// Copyright 2026 Huawei Technologies Co., Ltd
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <memory>
#include <vector>
#include <torch/extension.h>
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_batch_matmul.h"
#include "aclnn/acl_meta.h"

namespace state_pull {
inline void check_acl(int status, const char *operation) { TORCH_CHECK(status == 0, operation, " returned ", status); }

// Fixed shape/dtype, with addresses rebound on every invocation. Each arena
// owns its plans; its single stream serializes workspace use and address updates.
// Keep BMM and Add separate to preserve the existing FP32 rounding order.
class BatchApplyPlan {
 public:
  BatchApplyPlan(const at::Tensor &matrix, const at::Tensor &entry, const at::Tensor &state, const at::Tensor &output,
                 int64_t heads) {
    std::vector<int64_t> dims{heads, state.size(2), state.size(3)};
    left_ = descriptor(matrix.data_ptr(), dims);
    right_ = descriptor(entry.data_ptr(), dims);
    product_ = descriptor(output.data_ptr(), dims);
    dims.insert(dims.begin(), 1);
    add_left_ = descriptor(output.data_ptr(), dims);
    summary_ = descriptor(state.data_ptr(), dims);
    output_ = descriptor(output.data_ptr(), dims);
    int64_t one = 1;
    alpha_.reset(aclCreateScalar(&one, ACL_INT64));
    TORCH_CHECK(alpha_, "aclCreateScalar failed");
    aclOpExecutor *executor = nullptr;
    auto status = aclnnBatchMatMulGetWorkspaceSize(left_.get(), right_.get(), product_.get(), 0, &mm_bytes_, &executor);
    mm_.reset(executor);
    check_acl(status, "BMM plan");
    check_acl(aclSetAclOpExecutorRepeatable(mm_.get()), "repeat BMM");
    executor = nullptr;
    status =
      aclnnAddGetWorkspaceSize(add_left_.get(), summary_.get(), alpha_.get(), output_.get(), &add_bytes_, &executor);
    add_.reset(executor);
    check_acl(status, "Add plan");
    check_acl(aclSetAclOpExecutorRepeatable(add_.get()), "repeat Add");
    workspace_ = at::empty({static_cast<int64_t>(std::max<uint64_t>(1, std::max(mm_bytes_, add_bytes_)))},
                           output.options().dtype(at::kByte));
  }

  void execute(void *matrix, void *entry, void *state, void *out, aclrtStream stream) {
    check_acl(aclSetInputTensorAddr(mm_.get(), 0, left_.get(), matrix), "BMM input0");
    check_acl(aclSetInputTensorAddr(mm_.get(), 1, right_.get(), entry), "BMM input1");
    check_acl(aclSetOutputTensorAddr(mm_.get(), 0, product_.get(), out), "BMM output");
    check_acl(aclnnBatchMatMul(workspace_.data_ptr(), mm_bytes_, mm_.get(), stream), "BMM execute");
    check_acl(aclSetInputTensorAddr(add_.get(), 0, add_left_.get(), out), "Add input0");
    check_acl(aclSetInputTensorAddr(add_.get(), 1, summary_.get(), state), "Add input1");
    check_acl(aclSetOutputTensorAddr(add_.get(), 0, output_.get(), out), "Add output");
    check_acl(aclnnAdd(workspace_.data_ptr(), add_bytes_, add_.get(), stream), "Add execute");
  }

  size_t workspace_bytes() const { return workspace_.nbytes(); }

 private:
  using Tensor = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;
  using Scalar = std::unique_ptr<aclScalar, decltype(&aclDestroyScalar)>;
  using Executor = std::unique_ptr<aclOpExecutor, decltype(&aclDestroyAclOpExecutor)>;

  static Tensor descriptor(void *ptr, const std::vector<int64_t> &dims) {
    std::vector<int64_t> strides(dims.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= dims[i];
    }
    Tensor result(aclCreateTensor(dims.data(), dims.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, dims.data(),
                                  dims.size(), ptr),
                  aclDestroyTensor);
    TORCH_CHECK(result, "aclCreateTensor failed");
    return result;
  }

  Tensor left_{nullptr, aclDestroyTensor}, right_{nullptr, aclDestroyTensor}, product_{nullptr, aclDestroyTensor};
  Tensor add_left_{nullptr, aclDestroyTensor}, summary_{nullptr, aclDestroyTensor}, output_{nullptr, aclDestroyTensor};
  Scalar alpha_{nullptr, aclDestroyScalar};
  Executor mm_{nullptr, aclDestroyAclOpExecutor}, add_{nullptr, aclDestroyAclOpExecutor};
  uint64_t mm_bytes_ = 0, add_bytes_ = 0;
  at::Tensor workspace_;
};
}  // namespace state_pull
