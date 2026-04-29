/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file multicore_moe_ffn_grad_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class MulticoreMoeFfnGrad : public OpDef {
 public:
  explicit MulticoreMoeFfnGrad(const char *name) : OpDef(name) {
    // 通用输入配置函数
    auto setInput = [this](const char *name, const std::vector<ge::DataType> &dtypes) {
      this->Input(name)
        .ParamType(REQUIRED)
        .DataType(dtypes)
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    };

    // 通用输出配置函数
    auto setOutput = [this](const char *name, const std::vector<ge::DataType> &dtypes) {
      this->Output(name)
        .ParamType(REQUIRED)
        .DataType(dtypes)
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    };

    // FLOAT16 / BF16 输入
    const std::vector<ge::DataType> kFloatDtype = {ge::DT_FLOAT16, ge::DT_BF16};
    setInput("dispatch_target", kFloatDtype);
    setInput("dy", kFloatDtype);
    setInput("hidden", kFloatDtype);
    setInput("hidden_dw", kFloatDtype);
    setInput("weight", kFloatDtype);
    setInput("y", kFloatDtype);
    setInput("gate", kFloatDtype);
    setInput("grad_gate", kFloatDtype);
    setInput("w1", kFloatDtype);
    setInput("gate_dx", kFloatDtype);
    setInput("grad_x", kFloatDtype);
    setInput("permute_out", kFloatDtype);
    setInput("gate_dw", kFloatDtype);

    // INT64 输入
    const std::vector<ge::DataType> kInt64Dtype = {ge::DT_INT64, ge::DT_INT64};
    setInput("dispatch_target_off", kInt64Dtype);
    setInput("dispatch_src_off", kInt64Dtype);
    setInput("combine_target_off", kInt64Dtype);
    setInput("combine_src_off", kInt64Dtype);
    setInput("group_list", kInt64Dtype);

    // INT32 输入
    const std::vector<ge::DataType> kInt32Dtype = {ge::DT_INT32, ge::DT_INT32};
    setInput("dispatch_size", kInt32Dtype);
    setInput("combine_size", kInt32Dtype);

    // UINT8 输入
    const std::vector<ge::DataType> kUint8Dtype = {ge::DT_UINT8, ge::DT_UINT8};
    setInput("act_grad_tiling", kUint8Dtype);
    setInput("gate_grad_tiling", kUint8Dtype);
    setInput("w2_grad_tiling", kUint8Dtype);
    setInput("w1_grad_tiling", kUint8Dtype);
    setInput("swiglu_grad_tiling", kUint8Dtype);
    setInput("gmm_workspace", kUint8Dtype);
    setInput("swiglu_grad_workspace", kUint8Dtype);
    setInput("runtime_config", kUint8Dtype);
    setInput("all_event_counters", kUint8Dtype);

    // 输出
    setOutput("dispatch_target", kFloatDtype);
    setOutput("hidden_dw", kFloatDtype);
    setOutput("y", kFloatDtype);
    setOutput("grad_gate", kFloatDtype);
    setOutput("gate_dx", kFloatDtype);
    setOutput("grad_x", kFloatDtype);
    setOutput("gate_dw", kFloatDtype);

    // 属性
    this->Attr("rank_id").AttrType(OPTIONAL).Int(0);
    this->Attr("ep").AttrType(OPTIONAL).Int(0);
    this->Attr("expert_num").AttrType(OPTIONAL).Int(0);
    this->Attr("hidden_size").AttrType(OPTIONAL).Int(0);
    this->Attr("seq_size").AttrType(OPTIONAL).Int(0);

    // AICore 配置
    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
      .DynamicFormatFlag(true)
      .DynamicRankSupportFlag(true)
      .DynamicShapeSupportFlag(true)
      .NeedCheckSupportFlag(false)
      .PrecisionReduceFlag(true)
      .ExtendCfgInfo("prebuildPattern.value", "Opaque")
      .ExtendCfgInfo("coreType.value", "AiCore")
      .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");

    this->AICore().AddConfig("ascend910b", aicore_config);
    this->AICore().AddConfig("ascend910_93", aicore_config);
  }
};
OP_ADD(MulticoreMoeFfnGrad);
}  // namespace ops
