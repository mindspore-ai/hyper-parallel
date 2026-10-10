/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "register/op_def_registry.h"

namespace ops {

class HyperMegaGateRoute : public OpDef {
 public:
  explicit HyperMegaGateRoute(const char *name) : OpDef(name) {
    this->Input("logits").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->Input("text_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->Input("vision_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->Input("image_mask").ParamType(REQUIRED).DataType({ge::DT_BOOL}).Format({ge::FORMAT_ND});
    this->Input("runtime_config").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND});
    this->Input("profile_buffer").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND});
    this->Output("routing_weights").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->Output("expert_indices").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
    this->Output("route_scores").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->Output("selected_scores").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->Output("normalization_denominator").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->Attr("top_k").AttrType(REQUIRED).Int();
    this->Attr("routed_scaling_factor").AttrType(REQUIRED).Float();
    this->Attr("use_vision_bias").AttrType(REQUIRED).Bool();

    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true)
      .DynamicFormatFlag(true)
      .DynamicRankSupportFlag(true)
      .DynamicShapeSupportFlag(true)
      .NeedCheckSupportFlag(false)
      .PrecisionReduceFlag(false)
      .ExtendCfgInfo("prebuildPattern.value", "Opaque")
      .ExtendCfgInfo("coreType.value", "AiCore")
      .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
    this->AICore().AddConfig("ascend910b", config);
    this->AICore().AddConfig("ascend910_93", config);
  }
};

OP_ADD(HyperMegaGateRoute);

}  // namespace ops
