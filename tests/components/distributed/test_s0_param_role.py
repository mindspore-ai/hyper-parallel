# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S0.1 + S1.1: ParamRole 枚举完备性 + ParameterClassifier 默认规则。"""

import torch.nn as nn

from hyper_models.components.distributed.param_role import (
    ParameterClassifier,
    ParamRole,
    _build_default_rules,
)


class TestParamRoleEnum:
    def test_fourteen_roles(self):
        assert len(ParamRole) == 14

    def test_expected_role_names(self):
        expected = {
            "COLWISE", "ROWWISE", "NORM", "EMBED", "LM_HEAD", "MOE_GATE",
            "MOE_EXPERT", "SHARED_EXPERT", "FUSED_QKV", "FUSED_GATE_UP",
            "BIAS", "REPLICATED", "SPECIAL", "SKIP",
        }
        assert {r.name for r in ParamRole} == expected


class _NamedModel(nn.Module):
    """用 ordered dict 注册参数以精确控制 FQN。"""
    def __init__(self, names):
        super().__init__()
        for n in names:
            # 把点分名注册到嵌套模块
            *path, leaf = n.split(".")
            obj = self
            for p in path:
                if not hasattr(obj, p):
                    setattr(obj, p, nn.Module())
                obj = getattr(obj, p)
            obj.register_parameter(leaf, nn.Parameter(__import__("torch").zeros(2)))


class TestParameterClassifier:
    def setup_method(self):
        self.clf = ParameterClassifier()

    def _role(self, name):
        model = _NamedModel([name])
        return self.clf.classify(model)[name]

    def test_each_role_hit(self):
        """默认规则可产生的 13 个角色每个至少 1 个命中用例（SPECIAL/SKIP 在内；
        REPLICATED 仅经 ARCH_OVERRIDES 指派，见 test_s1_mla_deepseek.py）。"""
        cases = {
            "model.embed_tokens.weight": ParamRole.EMBED,
            "lm_head.weight": ParamRole.LM_HEAD,
            "model.layers.0.self_attn.q_proj.weight": ParamRole.COLWISE,
            "model.layers.0.self_attn.o_proj.weight": ParamRole.ROWWISE,
            "model.layers.0.input_layernorm.weight": ParamRole.NORM,
            "model.layers.0.mlp.gate.weight": ParamRole.MOE_GATE,
            "model.layers.0.mlp.experts.w1": ParamRole.MOE_EXPERT,
            "model.layers.0.mlp.shared_experts.w2": ParamRole.SHARED_EXPERT,
            "model.layers.0.attn.fused_qkv.weight": ParamRole.FUSED_QKV,
            "model.layers.0.mlp.gate_up_proj.weight": ParamRole.FUSED_GATE_UP,
            "model.layers.0.self_attn.q_proj.bias": ParamRole.BIAS,
            "model.layers.0.gated_delta.a_log": ParamRole.SPECIAL,
            "model.rotary_emb.inv_freq": ParamRole.SKIP,
        }
        for name, want in cases.items():
            assert self._role(name) == want, name

    def test_ln_rule_no_false_positive(self):
        """ln_ / norm 规则不误伤 linear / kernel。"""
        assert self._role("model.linear.weight") == ParamRole.SKIP
        assert self._role("model.kernel.weight") == ParamRole.SKIP
        assert self._role("model.layers.0.ln_1.weight") == ParamRole.NORM

    def test_shared_experts_not_moe_expert(self):
        """shared_experts 必须先于 experts 命中 SHARED_EXPERT。"""
        assert self._role("m.mlp.shared_experts.w1") == ParamRole.SHARED_EXPERT
        assert self._role("m.mlp.experts.w1") == ParamRole.MOE_EXPERT

    def test_gate_proj_not_moe_gate(self):
        """dense gate_proj 不得误判为 MOE_GATE。"""
        assert self._role("m.mlp.gate_proj.weight") == ParamRole.COLWISE

    def test_expert_gate_proj_is_moe_expert(self):
        """per-expert 的 gate_proj（experts.N.gate_proj）归 MOE_EXPERT。"""
        assert self._role("m.mlp.experts.0.gate_proj.weight") == ParamRole.MOE_EXPERT

    def test_unmatched_returns_skip(self):
        model = _NamedModel(["a.b.c"])
        assert self.clf.classify(model)["a.b.c"] == ParamRole.SKIP

    def test_default_rules_structure(self):
        rules = _build_default_rules()
        assert all(isinstance(pats, list) and isinstance(r, ParamRole)
                   for pats, r in rules)
