/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "register/op_def_registry.h"

namespace ops {
class HyperMegaMhcGrad : public OpDef {
 public:
    explicit HyperMegaMhcGrad(const char *name) : OpDef(name)
    {
        AddInput("grad_hin_placeholder", {ge::DT_BF16});
        AddInput("grad_h_post", {ge::DT_FLOAT});
        AddInput("grad_h_res", {ge::DT_FLOAT});
        AddInput("x", {ge::DT_BF16});
        AddInput("phi", {ge::DT_FLOAT});
        AddInput("alpha", {ge::DT_FLOAT});
        AddInput("bias", {ge::DT_FLOAT});
        AddInput("previous_pre", {ge::DT_FLOAT});
        AddInput("hc_before_norm", {ge::DT_FLOAT});
        AddInput("inv_rms", {ge::DT_FLOAT});
        AddInput("sum_out", {ge::DT_FLOAT});
        AddInput("norm_out", {ge::DT_FLOAT});
        AddInput("grad_current_pre", {ge::DT_FLOAT});
        AddInput("mixed_input", {ge::DT_BF16});
        AddInput("rms_rstd", {ge::DT_FLOAT});
        AddInput("norm_weight", {ge::DT_BF16});
        AddInput("direct_grad_x", {ge::DT_BF16});
        AddInput("previous_residual", {ge::DT_BF16});
        AddInput("previous_output", {ge::DT_BF16});
        AddInput("previous_post", {ge::DT_FLOAT});
        AddInput("previous_residual_mix", {ge::DT_FLOAT});
        AddInput("runtime_config", {ge::DT_UINT8});
        AddInput("all_event_counters", {ge::DT_UINT8});
        AddInput("profile_buffer", {ge::DT_UINT8});
        AddOutput("grad_residual", {ge::DT_BF16});
        AddOutput("grad_phi", {ge::DT_FLOAT});
        AddOutput("grad_alpha", {ge::DT_FLOAT});
        AddOutput("grad_bias", {ge::DT_FLOAT});
        AddOutput("grad_previous_output", {ge::DT_BF16});
        AddOutput("grad_previous_pre", {ge::DT_FLOAT});
        AddOutput("grad_previous_post", {ge::DT_FLOAT});
        AddOutput("grad_previous_residual", {ge::DT_FLOAT});
        AddOutput("grad_norm_weight", {ge::DT_FLOAT});
        this->Attr("hc_eps").AttrType(OPTIONAL).Float(1e-6f);

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
    void AddInput(const char *name, std::initializer_list<ge::DataType> types)
    {
        this->Input(name).ParamType(REQUIRED).DataType(types).Format({ge::FORMAT_ND}).UnknownShapeFormat(
            {ge::FORMAT_ND});
    }

    void AddOutput(const char *name, std::initializer_list<ge::DataType> types)
    {
        this->Output(name).ParamType(REQUIRED).DataType(types).Format({ge::FORMAT_ND}).UnknownShapeFormat(
            {ge::FORMAT_ND});
    }
};
OP_ADD(HyperMegaMhcGrad);
}  // namespace ops
