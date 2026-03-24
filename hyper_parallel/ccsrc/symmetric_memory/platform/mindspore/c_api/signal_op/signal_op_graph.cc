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

// =============================================================================
// GRAPH MODE IMPLEMENTATION
// =============================================================================

#include <vector>
#include "framework/module.h"

// 用于静态图下没有返回值的算子InferShape占位输出
static const mindspore::ShapeArray kFakeOutTensorShapes{mindspore::ShapeVector{1}};
// 用于静态图下没有返回值的算子InferType占位输出
static const std::vector<mindspore::TypeId> kFakeOutTensorTypes{mindspore::TypeId::kNumberTypeInt8};
static const mindspore::ShapeVector kFakeOutShapes{1};

namespace ms_custom_ops {
using namespace mindspore;
using namespace mindspore::kernel;
using namespace mindspore::device::ascend;
using namespace mindspore::ops;

class OPS_API CustomSignalOpOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    return kFakeOutTensorShapes;
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    return kFakeOutTensorTypes;
  }

  bool GeneralInferRegistered() const override { return true; }
};

class CustomSignalOp : public KernelMod {
 public:
  CustomSignalOp() = default;
  ~CustomSignalOp() = default;
  using KernelMod::Init;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    MS_LOG(WARNING) << "default init with input size " << inputs.size() << " output size " << outputs.size();
    return true;
  }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs, void *) override {
    MS_LOG(WARNING) << "default launch with input size " << inputs.size() << " output size " << outputs.size();
    return true;
  }

  std::vector<KernelAttr> GetOpSupport() override {
    MS_LOG(EXCEPTION) << "This interface is not support in simu kernel module.";
  }
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(signal_op, ms_custom_ops::CustomSignalOpOpFuncImpl, ms_custom_ops::CustomSignalOp);
