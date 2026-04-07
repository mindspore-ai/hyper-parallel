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

// moe_ffn_bwd has 7 declared returns (following mega_kernel_gmm_grad_op.yaml):
//   target, hidden_dw, y, grad_gate, gate_dx, grad_x, gate_dw
// (permute_out is written in-place but not declared as a return)
static const mindspore::ShapeArray kBwdFakeOutShapes{
    mindspore::ShapeVector{1}, mindspore::ShapeVector{1},
    mindspore::ShapeVector{1}, mindspore::ShapeVector{1},
    mindspore::ShapeVector{1}, mindspore::ShapeVector{1},
    mindspore::ShapeVector{1}};
static const std::vector<mindspore::TypeId> kBwdFakeOutTypes{
    mindspore::TypeId::kNumberTypeInt8, mindspore::TypeId::kNumberTypeInt8,
    mindspore::TypeId::kNumberTypeInt8, mindspore::TypeId::kNumberTypeInt8,
    mindspore::TypeId::kNumberTypeInt8, mindspore::TypeId::kNumberTypeInt8,
    mindspore::TypeId::kNumberTypeInt8};

namespace ms_multicore {
using mindspore::PrimitivePtr;
using mindspore::ShapeArray;
using mindspore::TypeId;
using mindspore::kernel::KernelAttr;
using mindspore::kernel::KernelMod;
using mindspore::kernel::KernelTensor;
using mindspore::ops::InferInfoPtrList;
using mindspore::ops::OpFuncImpl;

class OPS_API CustomMoeFfnBwdOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    return kBwdFakeOutShapes;
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    return kBwdFakeOutTypes;
  }

  bool GeneralInferRegistered() const override { return true; }
};

class CustomMoeFfnBwd : public KernelMod {
 public:
  CustomMoeFfnBwd() = default;
  ~CustomMoeFfnBwd() = default;
  using KernelMod::Init;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    MS_LOG(WARNING) << "CustomMoeFfnBwd graph-mode stub: input=" << inputs.size()
                    << " output=" << outputs.size();
    return true;
  }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs, void *) override {
    MS_LOG(WARNING) << "CustomMoeFfnBwd graph-mode stub launch: input=" << inputs.size()
                    << " output=" << outputs.size();
    return true;
  }

  std::vector<KernelAttr> GetOpSupport() override {
    MS_LOG(EXCEPTION) << "This interface is not supported in stub kernel module.";
  }
};
}  // namespace ms_multicore

REG_GRAPH_MODE_OP(moe_ffn_bwd, ms_multicore::CustomMoeFfnBwdOpFuncImpl, ms_multicore::CustomMoeFfnBwd);
