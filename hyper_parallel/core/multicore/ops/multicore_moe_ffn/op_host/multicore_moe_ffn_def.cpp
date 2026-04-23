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
 * \file multicore_moe_ffn_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class MulticoreMoeFfn : public OpDef {
 public:
  explicit MulticoreMoeFfn(const char *name) : OpDef(name) {
    auto setCommonIOConfig = [this](const char *inputName, const std::vector<ge::DataType> &dtypes,
                                    const std::vector<ge::Format> &formats = {ge::FORMAT_ND, ge::FORMAT_ND}) {
      this->Input(inputName).ParamType(REQUIRED).DataType(dtypes).Format(formats).UnknownShapeFormat(formats);
    };

    // 浮点类型输入（FLOAT16/BF16）
    setCommonIOConfig("dispatch_target", {ge::DT_FLOAT16, ge::DT_BF16});
    setCommonIOConfig("dispatch_src", {ge::DT_FLOAT16, ge::DT_BF16});
    setCommonIOConfig("weight", {ge::DT_FLOAT16, ge::DT_BF16});
    setCommonIOConfig("y", {ge::DT_FLOAT16, ge::DT_BF16});
    setCommonIOConfig("swiglu_out", {ge::DT_FLOAT16, ge::DT_BF16});
    setCommonIOConfig("down_proj_weight", {ge::DT_FLOAT16, ge::DT_BF16});
    setCommonIOConfig("down_proj_y", {ge::DT_FLOAT16, ge::DT_BF16});
    setCommonIOConfig("combine_target", {ge::DT_FLOAT16, ge::DT_BF16});

    // INT64类型输入
    setCommonIOConfig("dispatch_target_off", {ge::DT_INT64, ge::DT_INT64});
    setCommonIOConfig("dispatch_src_off", {ge::DT_INT64, ge::DT_INT64});
    setCommonIOConfig("up_proj_glist", {ge::DT_INT64, ge::DT_INT64});
    setCommonIOConfig("down_proj_glist", {ge::DT_INT64, ge::DT_INT64});
    setCommonIOConfig("combine_target_off", {ge::DT_INT64, ge::DT_INT64});
    setCommonIOConfig("combine_src_off", {ge::DT_INT64, ge::DT_INT64});

    // INT32类型输入
    setCommonIOConfig("dispatch_size", {ge::DT_INT32, ge::DT_INT32});
    setCommonIOConfig("combine_size", {ge::DT_INT32, ge::DT_INT32});

    // 可选UINT8类型输入
    setCommonIOConfig("gmm_workspace", {ge::DT_UINT8, ge::DT_UINT8});
    this->Input("gmm_workspace").ParamType(OPTIONAL);
    setCommonIOConfig("up_proj_tiling", {ge::DT_UINT8, ge::DT_UINT8});
    this->Input("up_proj_tiling").ParamType(OPTIONAL);
    setCommonIOConfig("swiglu_tiling", {ge::DT_UINT8, ge::DT_UINT8});
    this->Input("swiglu_tiling").ParamType(OPTIONAL);
    setCommonIOConfig("down_proj_tiling", {ge::DT_UINT8, ge::DT_UINT8});
    this->Input("down_proj_tiling").ParamType(OPTIONAL);

    // 必选UINT8类型输入
    setCommonIOConfig("runtime_config", {ge::DT_UINT8, ge::DT_UINT8});
    setCommonIOConfig("all_event_counters", {ge::DT_UINT8, ge::DT_UINT8});

    // 输出配置（与对应输入类型一致）
    this->Output("dispatch_target")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("swiglu_out")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("down_proj_y")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("combine_target")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

    // 属性配置
    this->Attr("rank_id").AttrType(OPTIONAL).Int(0);
    this->Attr("ep").AttrType(OPTIONAL).Int(0);
    this->Attr("expert_num").AttrType(OPTIONAL).Int(0);
    this->Attr("hidden_size").AttrType(OPTIONAL).Int(0);
    this->Attr("seq_size").AttrType(OPTIONAL).Int(0);

    // AICore配置
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
OP_ADD(MulticoreMoeFfn);
}  // namespace ops
