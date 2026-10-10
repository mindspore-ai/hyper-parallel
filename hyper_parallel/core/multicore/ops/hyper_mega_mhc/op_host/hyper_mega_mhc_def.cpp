/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

/*! \file hyper_mega_mhc_def.cpp */
#include "register/op_def_registry.h"

namespace ops {
class HyperMegaMhc : public OpDef {
 public:
  explicit HyperMegaMhc(const char *name) : OpDef(name) {
    AddInput("residual", {ge::DT_BF16});
    AddInput("phi", {ge::DT_FLOAT});
    AddInput("alpha", {ge::DT_FLOAT});
    AddInput("bias", {ge::DT_FLOAT});
    AddInput("previous_output", {ge::DT_BF16});
    AddInput("previous_pre_mix", {ge::DT_FLOAT});
    AddInput("previous_post_mix", {ge::DT_FLOAT});
    AddInput("previous_residual_mix", {ge::DT_FLOAT});
    AddInput("norm_weight", {ge::DT_BF16});
    AddInput("runtime_config", {ge::DT_UINT8});
    AddInput("all_event_counters", {ge::DT_UINT8});
    AddInput("profile_buffer", {ge::DT_UINT8});
    AddInput("new_residual_out", {ge::DT_BF16});
    AddInput("next_pre_mix_out", {ge::DT_FLOAT});
    AddInput("next_post_mix_out", {ge::DT_FLOAT});
    AddInput("next_residual_mix_out", {ge::DT_FLOAT});
    AddInput("block_input_out", {ge::DT_BF16});
    AddInput("hc_before_norm_out", {ge::DT_FLOAT});
    AddInput("inv_rms_out", {ge::DT_FLOAT});
    AddInput("sum_out_out", {ge::DT_FLOAT});
    AddInput("norm_out_out", {ge::DT_FLOAT});
    AddInput("mixed_input_out", {ge::DT_BF16});
    AddInput("rms_rstd_out", {ge::DT_FLOAT});

    AddOutput("new_residual", {ge::DT_BF16});
    AddOutput("next_pre_mix", {ge::DT_FLOAT});
    AddOutput("next_post_mix", {ge::DT_FLOAT});
    AddOutput("next_residual_mix", {ge::DT_FLOAT});
    AddOutput("block_input", {ge::DT_BF16});
    AddOutput("hc_before_norm", {ge::DT_FLOAT});
    AddOutput("inv_rms", {ge::DT_FLOAT});
    AddOutput("sum_out", {ge::DT_FLOAT});
    AddOutput("norm_out", {ge::DT_FLOAT});
    AddOutput("mixed_input", {ge::DT_BF16});
    AddOutput("rms_rstd", {ge::DT_FLOAT});

    this->Attr("hc_mult").AttrType(OPTIONAL).Int(4);
    this->Attr("num_iters").AttrType(OPTIONAL).Int(20);
    this->Attr("hc_eps").AttrType(OPTIONAL).Float(1e-6);
    this->Attr("norm_eps").AttrType(OPTIONAL).Float(1e-6);
    this->Attr("need_backward").AttrType(OPTIONAL).Bool(true);

    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true)
      .DynamicFormatFlag(true)
      .DynamicRankSupportFlag(false)
      .DynamicShapeSupportFlag(true)
      .NeedCheckSupportFlag(false)
      .PrecisionReduceFlag(false)
      .ExtendCfgInfo("prebuildPattern.value", "Opaque")
      .ExtendCfgInfo("coreType.value", "AiCore")
      .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
    this->AICore().AddConfig("ascend910b", config);
    this->AICore().AddConfig("ascend910_93", config);
  }

 private:
  void AddInput(const char *name, std::initializer_list<ge::DataType> types) {
    this->Input(name).ParamType(REQUIRED).DataType(types).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  }

  void AddOutput(const char *name, std::initializer_list<ge::DataType> types) {
    this->Output(name).ParamType(REQUIRED).DataType(types).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  }
};
OP_ADD(HyperMegaMhc);
}  // namespace ops
