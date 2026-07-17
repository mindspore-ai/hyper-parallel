# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S1.14: DeepSeek MLA 架构覆盖（ARCH_OVERRIDES + ParamRole.REPLICATED）。

缺口背景：deepseek_v2/v3 的 MLA 投影（q_a_proj/q_b_proj/kv_a_proj_with_mqa/
kv_b_proj）不含任何默认命名规则子串，未覆盖时全部落 SKIP → self_attn 组只剩
o_proj(ROWWISE)，has_colwise=False → attention 边界推断失败，MLA 参数全部
不分片（静默），CP wrapper 也不注入。

方案 B：q_a/kv_a 下投影全复制（REPLICATED），q_b/kv_b 上投影按 head 维
COLWISE，o_proj ROWWISE contract head 维——与标准 attention 模板同构。
"""

import torch.nn as nn

from hyper_models.components.distributed.param_role import (
    ParameterClassifier,
    ParamRole,
)
from hyper_models.components.distributed.sharding_config import (
    TP,
    resolve_placements,
)
from hyper_models.components.distributed.sharding_planner import (
    ARCH_OVERRIDES,
    ShardingPlanner,
)
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _TinyMlaAttention(nn.Module):
    """DeepSeek MLA 结构（FQN 与 HF DeepseekV2/V3 Attention 一致）。"""

    def __init__(self, hidden=8, rank=4, q_out=8, kv_out=8):
        super().__init__()
        self.q_a_proj = nn.Linear(hidden, rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(rank)
        self.q_b_proj = nn.Linear(rank, q_out, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(hidden, rank, bias=False)
        self.kv_a_layernorm = nn.RMSNorm(rank)
        self.kv_b_proj = nn.Linear(rank, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, hidden, bias=False)


class _TinyMlp(nn.Module):
    def __init__(self, hidden=8, inter=16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)


class _TinyDeepseek(nn.Module):
    """2 层 MLA 小模型：FQN 仿 HF DeepseekV3（model.layers.N.self_attn.*）。"""

    def __init__(self, architectures=("DeepseekV3ForCausalLM",)):
        super().__init__()
        self.config = _Cfg(architectures=list(architectures),
                           model_type="deepseek_v3")
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(32, 8)
        self.model.layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            layer.self_attn = _TinyMlaAttention()
            layer.mlp = _TinyMlp()
            layer.input_layernorm = nn.RMSNorm(8)
            layer.post_attention_layernorm = nn.RMSNorm(8)
            self.model.layers.append(layer)
        self.model.norm = nn.RMSNorm(8)
        self.lm_head = nn.Linear(8, 32, bias=False)


class TestMlaArchOverride:
    def test_arch_overrides_registered_both_spellings(self):
        """architectures 拼写（deepseekv3）与 model_type 拼写（deepseek_v3）
        均注册同一份 MLA 覆盖；v2/v3 同构。"""
        for key in ("deepseekv2", "deepseekv3", "deepseek_v2", "deepseek_v3"):
            assert key in ARCH_OVERRIDES
            roles = [r for _, r in ARCH_OVERRIDES[key]]
            assert ParamRole.REPLICATED in roles
            assert ParamRole.COLWISE in roles

    def test_classifier_mla_roles(self):
        """q_a/kv_a → REPLICATED；q_b/kv_b → COLWISE；o_proj → ROWWISE（默认规则）。"""
        model = _TinyDeepseek()
        clf = ParameterClassifier(arch_overrides=ARCH_OVERRIDES)
        roles = clf.classify(model, "deepseekv3")
        p = "model.layers.0.self_attn."
        assert roles[p + "q_a_proj.weight"] == ParamRole.REPLICATED
        assert roles[p + "kv_a_proj_with_mqa.weight"] == ParamRole.REPLICATED
        assert roles[p + "q_b_proj.weight"] == ParamRole.COLWISE
        assert roles[p + "kv_b_proj.weight"] == ParamRole.COLWISE
        assert roles[p + "o_proj.weight"] == ParamRole.ROWWISE
        assert roles[p + "q_a_layernorm.weight"] == ParamRole.NORM
        assert roles[p + "kv_a_layernorm.weight"] == ParamRole.NORM

    def test_classifier_model_type_spelling(self):
        """model_type 拼写（deepseek_v3）同样命中覆盖。"""
        model = _TinyDeepseek()
        clf = ParameterClassifier(arch_overrides=ARCH_OVERRIDES)
        roles = clf.classify(model, "deepseek_v3")
        assert roles["model.layers.0.self_attn.q_a_proj.weight"] == ParamRole.REPLICATED

    def test_without_override_mla_params_skip(self):
        """回归保护：无覆盖时 MLA 投影落 SKIP（即修复前的静默缺口）。"""
        model = _TinyDeepseek()
        clf = ParameterClassifier()   # 无 arch_overrides
        roles = clf.classify(model, "deepseekv3")
        p = "model.layers.0.self_attn."
        assert roles[p + "q_a_proj.weight"] == ParamRole.SKIP
        assert roles[p + "kv_b_proj.weight"] == ParamRole.SKIP


class TestMlaPlan:
    def test_attention_boundary_and_placements(self, make_mesh):
        """端到端：architectures 检测 → 覆盖生效 → attention 边界生成，
        REPLICATED 全复制 / q_b,kv_b colwise / o_proj rowwise / cp_attn 置位。"""
        mesh = make_mesh((1,), ("tp",))
        plan = ShardingPlanner().plan(
            _TinyDeepseek(), mesh, tp_size=2, sequence_parallel=True)

        spec = plan.modules["model.layers.0.self_attn"]
        names = ("tp",)
        assert tuple(resolve_placements(
            spec.params["q_a_proj.weight"], names)) == (Replicate(),)
        assert tuple(resolve_placements(
            spec.params["kv_a_proj_with_mqa.weight"], names)) == (Replicate(),)
        assert tuple(resolve_placements(
            spec.params["q_b_proj.weight"], names)) == (Shard(0),)
        assert tuple(resolve_placements(
            spec.params["kv_b_proj.weight"], names)) == (Shard(0),)
        assert tuple(resolve_placements(
            spec.params["o_proj.weight"], names)) == (Shard(1),)
        # attention 模板标记：CP 激活时注入 inner attention wrapper
        assert spec._needs_cp_attn is True
        # 其余边界不受覆盖影响
        assert "model.layers.0.mlp" in plan.modules
        assert "model.embed_tokens" in plan.modules
        assert plan.modules["lm_head"]._is_terminal is True

    def test_model_type_fallback_also_hits(self, make_mesh):
        """config.architectures 缺失时回退 model_type='deepseek_v3' 同样命中。"""
        mesh = make_mesh((1,), ("tp",))
        model = _TinyDeepseek(architectures=())
        model.config.architectures = None
        plan = ShardingPlanner().plan(model, mesh, tp_size=2)
        spec = plan.modules["model.layers.0.self_attn"]
        assert tuple(resolve_placements(
            spec.params["q_b_proj.weight"], ("tp",))) == (Shard(0),)

    def test_both_layers_sharded(self, make_mesh):
        """两层 MLA 均生成 attention spec（无遗漏）。"""
        mesh = make_mesh((1,), ("tp",))
        plan = ShardingPlanner().plan(_TinyDeepseek(), mesh, tp_size=2)
        for i in range(2):
            assert f"model.layers.{i}.self_attn" in plan.modules
            spec = plan.modules[f"model.layers.{i}.self_attn"]
            assert len(spec.params) == 5   # q_a/q_b/kv_a/kv_b/o（layernorm 独立边界）
