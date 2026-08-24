# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_s1_plan_overrides.py: 核心套件合并文件。

来源: test_s1_plan_overrides.py, test_s1_injections.py, test_dist_s5_plan_overrides.py
"""

import copy
import logging
import pytest
import torch
import torch.nn as nn
from hyper_models.components.distributed import (
    ShardingPlanner,
    apply_sharding_plan,
)
from hyper_models.components.distributed.sharding_applier import _preflight_compute_injection
from hyper_models.components.distributed.sharding_config import (
    CP,
    ModuleShardingSpec,
    TP,
    resolve_placements,
)
from hyper_models.components.distributed.sharding_planner import ShardingPlanner
from hyper_models.trainer.config import (
    PlanOverride,
    entries_to_plan_overrides,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Replicate,
    Shard,
)
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyLlamaAttention,
    TinyLlamaForCausalLM,
    TinyLlamaMLP,
    TinyLlamaModel,
    TinyRMSNorm,
    _meta_mesh,
    cp_sdpa_hf_injection,
    ep_archetype_injection,
    run_dist,
)


# ==========================================================================
# 来源: test_s1_plan_overrides.py
# S1.13: ShardingPlanner(plan_overrides=...) —— 统一 override 通道（05 §3.6.7 + 统一改造）。
# ==========================================================================

def _attn_override_spec(key="x"):
    """自研多输入 attention 的手写 spec：契约 key 为真实签名参数名。"""
    return ModuleShardingSpec(
        params={
            "q_proj.weight": {TP: Shard(0), CP: Replicate()},
            "k_proj.weight": {TP: Shard(0), CP: Replicate()},
            "v_proj.weight": {TP: Shard(0), CP: Replicate()},
            "o_proj.weight": {TP: Shard(1), CP: Replicate()},
        },
        in_src={key: {TP: Shard(1)}},
        in_dst={key: {TP: Replicate()}},
        out_src={TP: Partial()},   # 标量简写，合并时应归一化为 {"output": ...}
        out_dst={TP: Shard(1)},
    )


def test_override_merge_full_declaration(tiny_llama, make_mesh):
    """merge 语义：用户全量声明时等价于整体替换；_needs_cp_attn 从推导 spec
    继承；标量简写归一化。"""
    mesh = make_mesh((1,), ("tp",))
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.self_attn": _attn_override_spec(key="x"),
    })
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)

    spec = plan.modules["model.layers.0.self_attn"]
    # 用户 spec 生效：契约 key 为 "x"
    assert set(spec.in_src) == {"x"}
    assert set(spec.in_dst) == {"x"}
    assert tuple(resolve_placements(spec.in_src["x"], ("tp",))) == (Shard(1),)
    # 结构标记从推导 spec 继承（attention 模板 → True）
    assert spec._needs_cp_attn is True
    assert spec.region_dispatch is None
    # 标量简写已归一化
    assert set(spec.out_src) == {"output"}
    assert tuple(resolve_placements(spec.out_src["output"], ("tp",))) == (Partial(),)
    # _is_terminal 由 Phase 5 统一标记（非末端）
    assert spec._is_terminal is False
    # 参数分片声明为用户声明值（字段粒度替换）
    assert spec.params["q_proj.weight"][TP] == Shard(0)
    assert spec.params["o_proj.weight"][TP] == Shard(1)

    # 未被覆盖的模块保持模板推导结果
    other = plan.modules["model.layers.1.self_attn"]
    assert set(other.in_src) == {"hidden_states"}


class TestMergeInheritance:
    """merge 模式的空字段继承（统一改造核心语义）。"""

    def test_injection_only_spec_inherits_contracts(self, tiny_llama, make_mesh):
        """注入字段-only 的 override（params/契约全空）→ 继承推导 spec 的
        params 与 I/O 契约，仅写入注入字段。"""
        mesh = make_mesh((1,), ("tp",))
        baseline = ShardingPlanner().plan(tiny_llama, mesh, tp_size=2)
        derived = baseline.modules["model.layers.0.mlp"]

        my_compute = lambda module, x: x  # noqa: E731
        planner = ShardingPlanner(plan_overrides={
            "model.layers.0.mlp": ModuleShardingSpec(local_compute_fn=my_compute, region_dispatch=False),
        })
        plan = planner.plan(tiny_llama, mesh, tp_size=2)
        spec = plan.modules["model.layers.0.mlp"]
        assert spec.local_compute_fn is my_compute
        # params/契约完整继承推导结果
        assert spec.params == derived.params
        assert spec.in_src == derived.in_src
        assert spec.in_dst == derived.in_dst
        assert spec.out_src == derived.out_src
        assert spec.out_dst == derived.out_dst

    def test_partial_contract_override_inherits_rest(self, tiny_llama, make_mesh):
        """只改 in_dst（all-gather 改 identity），其余契约字段继承。"""
        mesh = make_mesh((1,), ("tp",))
        baseline = ShardingPlanner().plan(tiny_llama, mesh, tp_size=2)
        derived = baseline.modules["model.layers.0.mlp"]
        planner = ShardingPlanner(plan_overrides={
            "model.layers.0.mlp": ModuleShardingSpec(
                in_dst={"hidden_states": {TP: Shard(1)}}),
        })
        plan = planner.plan(tiny_llama, mesh, tp_size=2)
        spec = plan.modules["model.layers.0.mlp"]
        assert spec.in_dst["hidden_states"][TP] == Shard(1)   # 用户值
        assert spec.in_src == derived.in_src                  # 继承
        assert spec.params == derived.params                  # 继承

    def test_internal_flags_always_inherit(self, tiny_hf_native_moe):
        """merge 不改写内部标记：D-10 推导的 _ep_size/_ep_stack
        即使用户 spec 是默认值也保留。"""
        from tests.components.distributed.conftest import _meta_mesh
        mesh = _meta_mesh((4, 2), ("dp", "tp"))
        planner = ShardingPlanner(plan_overrides={
            "model.layers.0.mlp": ModuleShardingSpec(
                local_compute_fn=lambda m, x: x, region_dispatch=False),
        })
        plan = planner.plan(tiny_hf_native_moe, mesh, tp_size=2, ep_size=2)
        spec = plan.modules["model.layers.0.mlp"]
        assert spec._ep_size == 2
        assert spec._ep_stack                      # D-09 堆叠元数据保留
        assert spec.region_dispatch is False        # 用户显式声明（注入字段非 None 即胜）

    def test_glob_key_merges_all_hits(self, tiny_llama, make_mesh):
        """glob key：一条 override 覆盖所有命中边界（继承各自契约）。"""
        mesh = make_mesh((1,), ("tp",))
        planner = ShardingPlanner(plan_overrides={
            "*.self_attn": ModuleShardingSpec(inner_target="self", inner_wrapper="sdpa_hf",
                                     region_dispatch=False),
        })
        plan = planner.plan(tiny_llama, mesh, tp_size=2)
        for i in (0, 1):
            spec = plan.modules[f"model.layers.{i}.self_attn"]
            assert spec.inner_wrapper == "sdpa_hf"
            assert spec.params["q_proj.weight"]    # 契约继承
        assert plan.modules["lm_head"].inner_wrapper is None

    def test_exact_wins_per_field_over_glob(self, tiny_llama, make_mesh):
        """exact + glob 同时命中：按条目顺序逐字段合并——exact（后处理）的
        非空字段优先，glob 的其余字段仍然生效。"""
        mesh = make_mesh((1,), ("tp",))
        planner = ShardingPlanner(plan_overrides={
            "*.self_attn": ModuleShardingSpec(inner_target="self", inner_wrapper="sdpa_hf",
                                     region_dispatch=False),
            "model.layers.0.self_attn": ModuleShardingSpec(
                inner_wrapper="sdpa_qkv", inner_target="self", region_dispatch=False),
        })
        plan = planner.plan(tiny_llama, mesh, tp_size=2)
        spec0 = plan.modules["model.layers.0.self_attn"]
        assert spec0.inner_wrapper == "sdpa_qkv"     # exact 覆盖 glob
        assert spec0.inner_target == "self"
        # layers.1 只吃 glob
        assert plan.modules["model.layers.1.self_attn"].inner_wrapper == "sdpa_hf"

    def test_glob_miss_warns(self, tiny_llama, make_mesh, caplog):
        mesh = make_mesh((1,), ("tp",))
        planner = ShardingPlanner(plan_overrides={
            "*.self_atn": ModuleShardingSpec(inner_target="self", inner_wrapper="sdpa_hf",
                                     region_dispatch=False),
        })
        with caplog.at_level(logging.WARNING):
            planner.plan(tiny_llama, mesh, tp_size=2)
        assert "hit no boundary spec" in caplog.text

    def test_injection_only_on_unmatched_fqn_raises(self, tiny_llama, make_mesh):
        """注入字段-only 的 spec 未命中任何推导边界 → fail-fast（merge 继承
        只对已推导边界生效；插入必须完整自声明）。

        注意 fqn 必须真实存在于 named_modules（否则先触发拼写检查）——
        model.layers.0 是真实模块但不是推导边界。"""
        mesh = make_mesh((1,), ("tp",))
        planner = ShardingPlanner(plan_overrides={
            "model.layers.0": ModuleShardingSpec(inner_target="self", inner_wrapper="sdpa_hf",
                                     region_dispatch=False),
        })
        with pytest.raises(ValueError, match="hit no planner-derived boundary"):
            planner.plan(tiny_llama, mesh, tp_size=2)


class TestContractSentinels:
    """契约字段的字符串哨兵（仅 plan_overrides 输入侧，merge 时解析）：

    "auto" = 显式继承（按模板推导，与不写同义，自文档化）；
    "none" = 显式清空；dict = 字段粒度替换（含 {} = 显式空）。
    """

    def test_auto_explicit_inherit(self, tiny_llama, make_mesh):
        """params="auto"：显式声明按模板推导——与缺省空值结果一致。"""
        mesh = make_mesh((1,), ("tp",))
        baseline = ShardingPlanner().plan(tiny_llama, mesh, tp_size=2)
        derived = baseline.modules["model.layers.0.mlp"]
        planner = ShardingPlanner(plan_overrides={
            "model.layers.0.mlp": ModuleShardingSpec(
                params="auto", in_src="auto", in_dst="auto",
                out_src="auto", out_dst="auto"),
        })
        plan = planner.plan(tiny_llama, mesh, tp_size=2)
        spec = plan.modules["model.layers.0.mlp"]
        assert spec.params == derived.params
        assert spec.in_src == derived.in_src
        assert spec.out_src == derived.out_src

    def test_none_clears_params(self, tiny_llama, make_mesh):
        """params="none"：显式清空已推导的参数分片（其余字段继承）。"""
        mesh = make_mesh((1,), ("tp",))
        baseline = ShardingPlanner().plan(tiny_llama, mesh, tp_size=2)
        derived = baseline.modules["model.layers.0.mlp"]
        # allow_uncovered_params：本用例刻意清空 mlp params（F4b 覆盖校验
        # 的逃生舱，仅用于 override 机制单测）
        planner = ShardingPlanner(plan_overrides={
            "model.layers.0.mlp": ModuleShardingSpec(params="none"),
        }, allow_uncovered_params=True)
        plan = planner.plan(tiny_llama, mesh, tp_size=2)
        spec = plan.modules["model.layers.0.mlp"]
        assert spec.params == {}                       # 已清空
        assert spec.in_dst == derived.in_dst           # 其余继承

    def test_bad_sentinel_raises(self, tiny_llama, make_mesh):
        """未知字符串值 → fail-fast 并列出合法哨兵。"""
        mesh = make_mesh((1,), ("tp",))
        planner = ShardingPlanner(plan_overrides={
            "model.layers.0.mlp": ModuleShardingSpec(params="auto2"),
        })
        with pytest.raises(ValueError, match="auto"):
            planner.plan(tiny_llama, mesh, tp_size=2)

    def test_sentinel_rejected_in_insert_mode(self, tiny_llama, make_mesh):
        """insert（未命中推导边界）下哨兵无意义 → fail-fast。"""
        mesh = make_mesh((1,), ("tp",))
        planner = ShardingPlanner(plan_overrides={
            "model.layers.0": ModuleShardingSpec(
                params="auto",
                in_src={"hidden_states": {TP: Shard(1)}},
                in_dst={"hidden_states": {TP: Shard(1)}},
                out_src={"output": {TP: Shard(1)}},
                out_dst={"output": {TP: Shard(1)}},
            ),
        })
        with pytest.raises(ValueError, match="insert"):
            planner.plan(tiny_llama, mesh, tp_size=2)


def test_override_insert_for_missed_module(tiny_llama, make_mesh):
    """插入语义：planner 不生成 spec 且与所有派生边界无祖孙关系的模块
    （此处为顶层旁路 dropout）可插入，并参与链式传播。"""
    mesh = make_mesh((1,), ("tp",))
    # 顶层旁路模块：planner 不覆盖，且与所有派生边界是兄弟（非祖孙）关系
    tiny_llama.model.dropout = nn.Dropout(0.0)
    identity = ModuleShardingSpec(
        params={},
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Shard(1)}},   # identity
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
    )
    planner = ShardingPlanner(plan_overrides={"model.dropout": identity})
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)

    assert "model.dropout" in plan.modules
    # 插入位置（named_modules 顺序）：model.norm → model.dropout → lm_head，
    # 均被下游引用 → 非末端；lm_head 仍为末端
    assert plan.modules["model.norm"]._is_terminal is False
    assert plan.modules["model.dropout"]._is_terminal is False
    assert plan.modules["lm_head"]._is_terminal is True


class TestDeriveFalse:
    """derive=False：关闭模板推导，plan 只含 plan_overrides 显式声明的
    spec（全部 insert 模式）——取代 plan.modules 后处理剪除写法（多模态
    encoder_dp ViT 桥接场景：子树内任何推导的 TP 集合通信都是数学错误）。"""

    def test_derive_false_yields_only_overrides(self, tiny_llama, make_mesh):
        """derive=False：plan.modules == override 键集合（零推导）。"""
        mesh = make_mesh((1,), ("tp",))
        bridge = ModuleShardingSpec(
            params={},
            region_dispatch=False,
            in_src={"input_ids": {TP: Replicate()}},
            in_dst={"input_ids": {TP: Replicate()}},
            out_src={"output": {TP: Replicate()}},
            out_dst={"output": {TP: Replicate()}},
        )
        plan = ShardingPlanner(plan_overrides={"": bridge}, derive=False).plan(
            tiny_llama, mesh, tp_size=2)
        assert set(plan.modules) == {""}
        # 对照：默认 derive=True 时同一模型推导出内层边界
        full = ShardingPlanner(plan_overrides={"": bridge}).plan(
            tiny_llama, mesh, tp_size=2)
        assert "model.layers.0.self_attn" in full.modules
        assert len(full.modules) > 1

    def test_derive_false_insert_requires_self_declaration(
            self, tiny_llama, make_mesh):
        """derive=False 下无推导可继承：注入字段-only 的 spec → insert 模式
        fail-fast（全未声明）。"""
        mesh = make_mesh((1,), ("tp",))
        with pytest.raises(ValueError, match="hit no planner-derived boundary"):
            ShardingPlanner(
                plan_overrides={
                    "model.layers.0.self_attn": ModuleShardingSpec(
                        inner_target="self", inner_wrapper="sdpa_hf",
                        region_dispatch=False)},
                derive=False).plan(tiny_llama, mesh, tp_size=2)

    def test_derive_false_sentinel_rejected_with_reason(
            self, tiny_llama, make_mesh):
        """derive=False 下 'auto'/'none' 哨兵无继承/清空来源 → fail-fast 且
        报错直接点名 derive=False 与正确写法（{} 显式空）。"""
        mesh = make_mesh((1,), ("tp",))
        with pytest.raises(ValueError, match="derive=False"):
            ShardingPlanner(
                plan_overrides={"": ModuleShardingSpec(params="auto")},
                derive=False).plan(tiny_llama, mesh, tp_size=2)

    def test_derive_false_glob_hits_nothing_warns(
            self, tiny_llama, make_mesh, caplog):
        """derive=False：glob 键无推导边界可命中 → 大声 warning（glob 从不
        插入）。"""
        mesh = make_mesh((1,), ("tp",))
        bridge = ModuleShardingSpec(
            params={},
            region_dispatch=False,
            in_src={"input_ids": {TP: Replicate()}},
            in_dst={"input_ids": {TP: Replicate()}},
            out_src={"output": {TP: Replicate()}},
            out_dst={"output": {TP: Replicate()}},
        )
        with caplog.at_level(logging.WARNING):
            plan = ShardingPlanner(
                plan_overrides={
                    "": bridge,
                    "*.mlp": ModuleShardingSpec(region_dispatch=False)},
                derive=False).plan(tiny_llama, mesh, tp_size=2)
        assert "hit no boundary spec" in caplog.text
        assert set(plan.modules) == {""}


class TestNestedOverrideD14:
    """D-14（05 §13）嵌套 spec：祖孙边界放行，仅守两条 plan 期不变式——
    参数唯一归属（双切 fail-fast）与全声明强制化（缺 in_src fail-fast）。"""

    def test_nested_outer_allowed(self, tiny_llama, make_mesh):
        """override model.layers.0 作为外层边界（params={} 仅 I/O 契约），
        内层派生边界保留 → plan 成功，外层 spec 插入且内层不被改写。"""
        mesh = make_mesh((1,), ("tp",))
        block = ModuleShardingSpec(
            params={},
            in_src={"hidden_states": {TP: Shard(1)}},
            in_dst={"hidden_states": {TP: Shard(1)}},
            out_src={TP: Shard(1)},
            out_dst={TP: Shard(1)},
        )
        planner = ShardingPlanner(plan_overrides={"model.layers.0": block})
        plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
        assert "model.layers.0" in plan.modules
        # 内层派生边界照常存在（不被外层接管）
        assert "model.layers.0.self_attn" in plan.modules
        assert plan.modules["model.layers.0.self_attn"].params["q_proj.weight"]

    def test_param_double_declaration_raises(self, tiny_llama, make_mesh):
        """外层 spec.params 声明了内层边界子树的参数 → ValueError（不变式 1：
        参数唯一归属，双切在 production 静默错误）。"""
        mesh = make_mesh((1,), ("tp",))
        block = ModuleShardingSpec(
            # self_attn.q_proj.weight 已被派生 attention 边界声明
            params={"self_attn.q_proj.weight": {TP: Shard(0)}},
            in_src={"hidden_states": {TP: Shard(1)}},
            in_dst={"hidden_states": {TP: Shard(1)}},
            out_src={TP: Shard(1)},
            out_dst={TP: Shard(1)},
        )
        planner = ShardingPlanner(plan_overrides={"model.layers.0": block})
        with pytest.raises(ValueError, match="exactly one boundary") as exc:
            planner.plan(tiny_llama, mesh, tp_size=2)
        assert "self_attn.q_proj.weight" in str(exc.value)
        assert "model.layers.0" in str(exc.value)

    def test_leaf_override_double_declaration_raises(self, tiny_llama, make_mesh):
        """override 派生边界的叶模块（q_proj）——参数与祖先边界冲突 →
        同样的唯一归属报错（嵌套本身合法，双切非法）。"""
        mesh = make_mesh((1,), ("tp",))
        leaf = ModuleShardingSpec(
            params={"weight": {TP: Shard(0)}},
            in_src={"hidden_states": {TP: Shard(1)}},
            in_dst={"hidden_states": {TP: Replicate()}},
            out_src={TP: Partial()},
            out_dst={TP: Shard(1)},
        )
        planner = ShardingPlanner(plan_overrides={
            "model.layers.0.self_attn.q_proj": leaf,
        })
        with pytest.raises(ValueError, match="exactly one boundary"):
            planner.plan(tiny_llama, mesh, tp_size=2)

    def test_missing_in_src_raises(self, tiny_llama, make_mesh):
        """全声明强制化（Scenario 1 填充已废除）：in_dst 非空而 in_src 为空
        → plan 期 ValueError。"""
        mesh = make_mesh((1,), ("tp",))
        block = ModuleShardingSpec(
            params={},
            in_src={},                                    # ← 空：D-14 后不再填充
            in_dst={"hidden_states": {TP: Shard(1)}},
            out_src={TP: Shard(1)},
            out_dst={TP: Shard(1)},
        )
        planner = ShardingPlanner(plan_overrides={"model.layers.0": block})
        with pytest.raises(ValueError, match="in_src"):
            planner.plan(tiny_llama, mesh, tp_size=2)


def test_override_chain_conflict_no_check(tiny_llama, make_mesh, caplog):
    """D-14：相邻契约一致性不再做任何静态/运行期检查——覆盖 spec 与上游
    out_dst 不一致时无 warning 无报错，声明原样保留（各模块只断言自身
    策略传播，端到端正确性由双模式数值对拍兜底）。

    """
    mesh = make_mesh((1,), ("tp",))
    # mlp 声明 in_src=Replicate，但上游 post_attention_layernorm out_dst=Shard(1)
    bad_mlp = ModuleShardingSpec(
        params={
            "gate_proj.weight": {TP: Shard(0)},
            "up_proj.weight": {TP: Shard(0)},
            "down_proj.weight": {TP: Shard(1)},
        },
        in_src={"hidden_states": {TP: Replicate()}},
        in_dst={"hidden_states": {TP: Replicate()}},
        out_src={TP: Partial()},
        out_dst={TP: Shard(1)},
    )
    planner = ShardingPlanner(plan_overrides={"model.layers.0.mlp": bad_mlp})
    with caplog.at_level(logging.WARNING):
        plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    assert "chain contract mismatch" not in caplog.text
    # 声明不被改写，plan 照常生成
    declared = plan.modules["model.layers.0.mlp"].in_src["hidden_states"]
    assert tuple(resolve_placements(declared, ("tp",))) == (Replicate(),)


def test_override_invalid_fqn_raises(tiny_llama, make_mesh):
    """fqn 未命中 named_modules（拼写错误）→ fail-fast ValueError。"""
    mesh = make_mesh((1,), ("tp",))
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.self_atn": _attn_override_spec(),   # 拼写错误
    })
    with pytest.raises(ValueError, match="named_modules"):
        planner.plan(tiny_llama, mesh, tp_size=2)


def test_override_wrong_type_raises(tiny_llama, make_mesh):
    """override 值必须是 ModuleShardingSpec。"""
    mesh = make_mesh((1,), ("tp",))
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.self_attn": {"params": {}},
    })
    with pytest.raises(TypeError, match="ModuleShardingSpec"):
        planner.plan(tiny_llama, mesh, tp_size=2)


def test_override_user_set_region_dispatch(tiny_llama, make_mesh):
    """region_dispatch 公开可配置（05 §3.6.7）：用户对自研模块显式置 True →
    合并后保留（模板未推断该标记时不受覆盖）；模板推断 True 的模块即使用户
    未设置也强制置位（防数值错误）。"""
    mesh = make_mesh((1,), ("tp",))
    # 用户对自研数据相关模块（此处借 mlp 演示）显式置 True
    custom = ModuleShardingSpec(
        params={
            "gate_proj.weight": {TP: Shard(0)},
            "up_proj.weight": {TP: Shard(0)},
            "down_proj.weight": {TP: Shard(1)},
        },
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Shard(1)}},   # identity
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
        region_dispatch=False,
    )
    planner = ShardingPlanner(plan_overrides={"model.layers.0.mlp": custom})
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    # 用户显式 True 保留（mlp 模板 region_dispatch=False，不会被清掉）
    assert plan.modules["model.layers.0.mlp"].region_dispatch is False
    # 未覆盖的 mlp 保持模板 False
    assert plan.modules["model.layers.1.mlp"].region_dispatch is None


def test_override_inner_wrap_fields_not_mutating_flags(tiny_llama, make_mesh):
    """inner-wrap 自定义入口（05 §4.4.2）：声明 inner_target/inner_wrapper
    后 inner-wrap 门控由 applier 解析链派生——**不改写 _needs_cp_attn**
    （声明互不嵌套），字段随深拷贝保留。"""
    mesh = make_mesh((1,), ("tp",))
    base = dict(
        params={
            "gate_proj.weight": {TP: Shard(0)},
            "up_proj.weight": {TP: Shard(0)},
            "down_proj.weight": {TP: Shard(1)},
        },
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Shard(1)}},   # identity
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
    )
    # inner_target 声明 → _needs_cp_attn 保持原值（False），不被隐式置位
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.mlp": ModuleShardingSpec(inner_target="self", **base),
    })
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._needs_cp_attn is False
    assert spec.inner_target == "self"

    # inner_wrapper（str 注册表名/callable）→ 同样不改写，随深拷贝保留
    my_wrapper = lambda target, cp_mesh: None  # noqa: E731
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.mlp": ModuleShardingSpec(inner_target="self", inner_wrapper=my_wrapper,
                                                 **base, region_dispatch=False),
    })
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._needs_cp_attn is False
    assert spec.inner_wrapper is my_wrapper


def test_override_local_compute_fn_gate_derived(tiny_llama, make_mesh):
    """local-region 自定义计算（05 §4.4.3）：声明 local_compute_fn 后骨架
    门控由 applier 解析链派生——**不改写 region_dispatch**（声明互不嵌套），
    callable 随深拷贝保留。"""
    mesh = make_mesh((1,), ("tp",))

    def my_compute(module, hidden_states):
        return module.gate_proj(hidden_states)

    custom = ModuleShardingSpec(
        params={
            "gate_proj.weight": {TP: Shard(0)},
            "up_proj.weight": {TP: Shard(0)},
            "down_proj.weight": {TP: Shard(1)},
        },
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Shard(1)}},   # identity
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
        local_compute_fn=my_compute, region_dispatch=False)
    planner = ShardingPlanner(plan_overrides={"model.layers.0.mlp": custom})
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    spec = plan.modules["model.layers.0.mlp"]
    # 门控派生：region_dispatch 保持声明原值（False），不被隐式改写
    assert spec.region_dispatch is False
    assert spec.local_compute_fn is my_compute


def test_override_input_spec_not_mutated(tiny_llama, make_mesh):
    """plan() 深拷贝用户 spec：归一化/标记/链式填充不污染调用方对象，可重复调用。"""
    mesh = make_mesh((1,), ("tp",))
    user_spec = _attn_override_spec(key="x")
    snapshot = copy.deepcopy(user_spec)
    planner = ShardingPlanner(plan_overrides={"model.layers.0.self_attn": user_spec})

    planner.plan(tiny_llama, mesh, tp_size=2)
    assert user_spec.out_src == snapshot.out_src            # 未被归一化改写
    assert user_spec._needs_cp_attn is False                # 未被模板补齐改写
    assert user_spec._is_terminal is False                  # 未被 Phase 5 改写
    assert user_spec.in_src == snapshot.in_src              # 未被链式填充改写

    # 重复调用结果一致（不累积污染）
    plan2 = planner.plan(tiny_llama, mesh, tp_size=2)
    spec2 = plan2.modules["model.layers.0.self_attn"]
    assert set(spec2.in_src) == {"x"}
    assert spec2._needs_cp_attn is True


def test_override_dp_declaration_raises(tiny_llama, make_mesh):
    """坐标系约定（05 §3）：plan = 单 dp 切片，override 声明 DP placement
    → fail-first ValueError（教学式报错），而非静默丢弃或选择性保留。"""
    from hyper_models.components.distributed.sharding_config import DP

    mesh = make_mesh((1,), ("tp",))
    base = dict(
        params={},
        in_src={"x": {TP: Shard(1)}},
        in_dst={"x": {TP: Shard(1)}},
        out_src={"output": {TP: Shard(1)}},
        out_dst={"output": {TP: Shard(1)}},
    )
    # 各字段逐一覆盖：params / in_src / out_dst（标量简写与嵌套形式都要拦截）
    bad_variants = [
        {**base, "params": {"q_proj.weight": {DP: Shard(0), TP: Replicate()}}},
        {**base, "in_src": {"x": {DP: Shard(0), TP: Shard(1)}}},
        {**base, "out_dst": {DP: Shard(0), TP: Shard(1)}},   # 标量简写
    ]
    for kwargs in bad_variants:
        planner = ShardingPlanner(plan_overrides={
            "model.layers.0.self_attn": ModuleShardingSpec(**kwargs),
        })
        with pytest.raises(ValueError, match="declaring a DP placement is not allowed"):
            planner.plan(tiny_llama, mesh, tp_size=2)


def test_function_module_uncovered_warns(tiny_llama, make_mesh, caplog):
    """DX guard：FunctionModule 无 spec 覆盖 → plan() warning；有 override
    覆盖 → 不告警且 spec 插入（教程 §10.8）。"""
    import torch
    from hyper_models.components.distributed import FunctionModule

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x

        @staticmethod
        def backward(ctx, grad_out):
            return grad_out

    tiny_llama.model.layers[0].helper_fn = FunctionModule(_Fn)
    mesh = make_mesh((1,), ("tp",))

    with caplog.at_level(logging.WARNING):
        ShardingPlanner().plan(tiny_llama, mesh, tp_size=2)
    assert "helper_fn" in caplog.text and "no boundary spec" in caplog.text

    caplog.clear()
    spec = ModuleShardingSpec(
        params={}, region_dispatch=False,
        in_src={"x": {TP: Shard(1)}}, in_dst={"x": {TP: Shard(1)}},
        out_src={"output": {TP: Shard(1)}}, out_dst={"output": {TP: Shard(1)}},
    )
    with caplog.at_level(logging.WARNING):
        plan = ShardingPlanner(plan_overrides={
            "model.layers.0.helper_fn": spec}).plan(tiny_llama, mesh, tp_size=2)
    assert "no boundary spec" not in caplog.text
    assert "model.layers.0.helper_fn" in plan.modules


class TestOverrideAxisValidation:
    """override placement 轴名/值校验（plan 期 fail-fast，2026-08-05）。"""

    def test_unknown_axis_raises(self, tiny_llama, make_mesh):
        """轴名拼写错误（tp2）→ fail-fast（否则被 resolve_placements 静默忽略）。"""
        mesh = make_mesh((1,), ("tp",))
        spec = ModuleShardingSpec(params={"q_proj.weight": {"tp2": Shard(0)}})
        with pytest.raises(ValueError, match="unknown axis"):
            ShardingPlanner(plan_overrides={
                "*.self_attn": spec}).plan(tiny_llama, mesh, tp_size=2)

    def test_ep_virtual_axis_allowed(self, tiny_llama, make_mesh):
        """'ep' 是虚拟轴（TP-extend-EP 专家分片坐标系）→ 不报错。"""
        from hyper_models.components.distributed.sharding_config import EP
        mesh = make_mesh((1,), ("tp",))
        spec = ModuleShardingSpec(params={"q_proj.weight": {EP: Shard(0)}})
        # allow_uncovered_params：部分 params 替换使其余参数未覆盖（F4b
        # 逃生舱，仅用于轴校验单测）
        ShardingPlanner(plan_overrides={
            "*.self_attn": spec}, allow_uncovered_params=True).plan(
            tiny_llama, mesh, tp_size=2)

    def test_non_placement_value_raises(self, tiny_llama, make_mesh):
        """placement 值不是 Placement 对象（如 Python 侧误传字符串）→ fail-fast。"""
        mesh = make_mesh((1,), ("tp",))
        spec = ModuleShardingSpec(params={"q_proj.weight": {TP: "shard(0)"}})
        with pytest.raises(TypeError, match="Placement"):
            ShardingPlanner(plan_overrides={
                "*.self_attn": spec}).plan(tiny_llama, mesh, tp_size=2)

    def test_mesh_axis_accepted(self, tiny_llama, make_mesh):
        """合法轴名（mesh 内）正常工作——回归保护。"""
        mesh = make_mesh((1,), ("tp",))
        spec = ModuleShardingSpec(params={"q_proj.weight": {TP: Shard(0)}})
        plan = ShardingPlanner(plan_overrides={
            "*.self_attn": spec}, allow_uncovered_params=True).plan(
            tiny_llama, mesh, tp_size=2)
        assert plan.modules["model.layers.0.self_attn"].params[
            "q_proj.weight"][TP] == Shard(0)


class TestYamlDslEndToEnd:
    """YAML 形态（字符串 DSL）→ PlanOverride 脱糖 → planner merge 全链路。"""

    def test_yaml_contract_fields_merge(self, tiny_llama, make_mesh):
        """契约字段从 YAML 字符串 DSL 脱糖为 Placement 对象后参与 merge。"""
        from hyper_models.trainer.config import (
            PlanOverride,
            entries_to_plan_overrides,
        )
        mesh = make_mesh((1,), ("tp",))
        overrides = entries_to_plan_overrides([PlanOverride(
            match="*.self_attn",
            params={
                "q_proj.weight": {"tp": "shard(0)"},
                "k_proj.weight": {"tp": "shard(0)"},
                "v_proj.weight": {"tp": "shard(0)"},
                "o_proj.weight": {"tp": "shard(1)"},
            },
            in_dst={"hidden_states": {"tp": "replicate"}},
            out_src={"tp": "partial"},      # 标量简写
        )])
        plan = ShardingPlanner(plan_overrides=overrides).plan(
            tiny_llama, mesh, tp_size=2)
        spec = plan.modules["model.layers.0.self_attn"]
        assert spec.params["q_proj.weight"][TP] == Shard(0)
        assert spec.params["o_proj.weight"][TP] == Shard(1)
        assert spec.in_dst["hidden_states"][TP] == Replicate()
        # 标量简写已归一化为 {"output": ...}
        assert spec.out_src["output"][TP] == Partial()

    def test_yaml_sentinels_merge(self, tiny_llama, make_mesh):
        """哨兵经 YAML 透传：'none' 清空推导的 in_src/in_dst（D-14 镜像约束
        要求两者一致清空）。"""
        from hyper_models.trainer.config import (
            PlanOverride,
            entries_to_plan_overrides,
        )
        mesh = make_mesh((1,), ("tp",))
        overrides = entries_to_plan_overrides([PlanOverride(
            match="*.self_attn", in_src="none", in_dst="none")])
        plan = ShardingPlanner(plan_overrides=overrides).plan(
            tiny_llama, mesh, tp_size=2)
        spec = plan.modules["model.layers.0.self_attn"]
        assert spec.in_src == {}                  # 'none' 清空
        assert spec.in_dst == {}                  # 'none' 清空

    def test_yaml_insert_mode_full_declaration(self, tiny_llama, make_mesh):
        """insert 模式经 YAML：模板未命中的模块完整自声明契约。"""
        from hyper_models.trainer.config import (
            PlanOverride,
            entries_to_plan_overrides,
        )
        tiny_llama.model.layers[0].extra = nn.Linear(4, 4)  # 非边界模块
        mesh = make_mesh((1,), ("tp",))
        overrides = entries_to_plan_overrides([PlanOverride(
            match="model.layers.0.extra",
            params={"weight": {"tp": "replicate"}},
            in_src={"x": {"tp": "shard(1)"}},
            in_dst={"x": {"tp": "shard(1)"}},
            out_src={"output": {"tp": "shard(1)"}},
            out_dst={"output": {"tp": "shard(1)"}},
        )])
        plan = ShardingPlanner(plan_overrides=overrides,
                               allow_uncovered_params=True).plan(
            tiny_llama, mesh, tp_size=2)
        inserted = plan.modules["model.layers.0.extra"]
        assert inserted.params["weight"][TP] == Replicate()

    def test_yaml_insert_with_sentinel_rejected(self, tiny_llama, make_mesh):
        """insert 模式 + 哨兵 → fail-fast（无可继承的推导值）。"""
        from hyper_models.trainer.config import (
            PlanOverride,
            entries_to_plan_overrides,
        )
        tiny_llama.model.layers[0].extra = nn.Linear(4, 4)  # 非边界模块
        mesh = make_mesh((1,), ("tp",))
        overrides = entries_to_plan_overrides([PlanOverride(
            match="model.layers.0.extra",
            params={"weight": {"tp": "replicate"}},
            in_src="auto",
        )])
        with pytest.raises(ValueError, match="insert"):
            ShardingPlanner(plan_overrides=overrides).plan(
                tiny_llama, mesh, tp_size=2)


class TestUnsetVsExplicitEmpty:
    """「不写继承，写了照办」（2026-08-05）：None（未声明）与 {}（显式空）
    语义分离——同一值在输入/输出、merge/insert 下读法唯一。"""

    def test_unset_inherits_derived(self, tiny_llama, make_mesh):
        """缺省 None（不写字段）→ 继承推导（与 'auto' 同义）。"""
        mesh = make_mesh((1,), ("tp",))
        spec_in = ModuleShardingSpec(inner_target="self", inner_wrapper="sdpa_hf",
                                     region_dispatch=False)  # 契约全未声明
        assert spec_in.params is None
        plan = ShardingPlanner(plan_overrides={
            "*.self_attn": spec_in}).plan(tiny_llama, mesh, tp_size=2)
        spec = plan.modules["model.layers.0.self_attn"]
        assert "q_proj.weight" in spec.params        # 推导继承
        assert spec.inner_wrapper == "sdpa_hf"       # 注入字段生效
        # plan 输出规范化：未声明字段落成具体 dict，绝不带 None
        for s in plan.modules.values():
            assert s.params is not None
            assert s.in_src is not None and s.in_dst is not None

    def test_explicit_empty_clears_merge(self, tiny_llama, make_mesh):
        """显式 {} → 清空推导（params={} = 本边界不切参数，纯 I/O 缝合）。"""
        mesh = make_mesh((1,), ("tp",))
        plan = ShardingPlanner(plan_overrides={
            "*.self_attn": ModuleShardingSpec(params={}),
        }, allow_uncovered_params=True).plan(tiny_llama, mesh, tp_size=2)
        assert plan.modules["model.layers.0.self_attn"].params == {}
        # I/O 契约未声明 → 仍继承推导
        assert "hidden_states" in plan.modules[
            "model.layers.0.self_attn"].in_dst

    def test_explicit_empty_insert_allowed(self, tiny_llama, make_mesh):
        """insert 模式：显式 {} 是合法声明（纯 I/O 缝合边界）。"""
        tiny_llama.model.layers[0].extra = nn.Linear(4, 4)
        mesh = make_mesh((1,), ("tp",))
        spec_in = ModuleShardingSpec(
            params={},
            in_src={"x": {TP: Shard(1)}}, in_dst={"x": {TP: Shard(1)}},
            out_src={"output": {TP: Shard(1)}},
            out_dst={"output": {TP: Shard(1)}})
        plan = ShardingPlanner(plan_overrides={
            "model.layers.0.extra": spec_in}, allow_uncovered_params=True
        ).plan(tiny_llama, mesh, tp_size=2)
        assert plan.modules["model.layers.0.extra"].params == {}

    def test_insert_all_unset_fails(self, tiny_llama, make_mesh):
        """insert 模式：全部未声明（全 None）→ fail-fast。"""
        tiny_llama.model.layers[0].extra = nn.Linear(4, 4)
        mesh = make_mesh((1,), ("tp",))
        with pytest.raises(ValueError, match="hit no planner-derived boundary"):
            ShardingPlanner(plan_overrides={
                "model.layers.0.extra": ModuleShardingSpec(),
            }).plan(tiny_llama, mesh, tp_size=2)

    def test_partial_params_replace_warns(self, tiny_llama, make_mesh, caplog):
        """部分覆盖：被丢弃的推导参数 → WARNING 列出（可见性防呆）。"""
        mesh = make_mesh((1,), ("tp",))
        with caplog.at_level(logging.WARNING):
            ShardingPlanner(plan_overrides={
                "*.self_attn": ModuleShardingSpec(
                    params={"q_proj.weight": {TP: Shard(0)}}),
            }, allow_uncovered_params=True).plan(tiny_llama, mesh, tp_size=2)
        assert "strips the derived sharding" in caplog.text
        assert "o_proj.weight" in caplog.text

    def test_full_params_replace_no_warn(self, tiny_llama, make_mesh, caplog):
        """完整覆盖（推导 key 全集）→ 无 WARNING。"""
        mesh = make_mesh((1,), ("tp",))
        full = {k: {TP: Shard(0)} for k in (
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight")}
        with caplog.at_level(logging.WARNING):
            ShardingPlanner(plan_overrides={
                "*.self_attn": ModuleShardingSpec(params=full),
            }).plan(tiny_llama, mesh, tp_size=2)
        assert "strips the derived sharding" not in caplog.text

    def test_yaml_empty_dict_is_explicit_clear(self, tiny_llama, make_mesh):
        """YAML 形态 params: {} → 显式清空（不再被拒绝）。"""
        from hyper_models.trainer.config import (
            PlanOverride,
            entries_to_plan_overrides,
        )
        mesh = make_mesh((1,), ("tp",))
        overrides = entries_to_plan_overrides([
            PlanOverride(match="*.self_attn", params={})])
        plan = ShardingPlanner(plan_overrides=overrides,
                               allow_uncovered_params=True).plan(
            tiny_llama, mesh, tp_size=2)
        assert plan.modules["model.layers.0.self_attn"].params == {}


# ==========================================================================
# 来源: test_s1_injections.py
# S1.14: PlanOverride 脱糖（entries_to_plan_overrides）+ 显式注入 preflight（单进程）。
# ==========================================================================

class TestInjectionDesugar:
    """PlanOverride / entries_to_plan_overrides 脱糖为 plan_overrides dict。"""

    def test_to_override_basic(self):
        entry = PlanOverride(match="*.self_attn", inner_wrapper="sdpa_hf")
        match, spec = entry.to_override()
        assert match == "*.self_attn"
        assert spec.inner_wrapper == "sdpa_hf"
        # 未设置字段保持 spec 默认（merge 语义下视为"未设置"，继承推导值）
        assert spec.local_compute_fn is None
        assert spec.inner_target is None
        assert spec.region_dispatch is None
        assert spec.tp_divide_attrs is None

    def test_tp_divide_attrs_desugar(self):
        _, spec = PlanOverride(
            match="*.self_attn",
            tp_divide_attrs=["hidden_size"],
        ).to_override()
        assert spec.tp_divide_attrs == ["hidden_size"]

    @pytest.mark.parametrize("attrs", ["hidden_size", ["bad.name"], ["x", "x"]])
    def test_tp_divide_attrs_invalid(self, attrs):
        with pytest.raises(ValueError, match="tp_divide_attrs"):
            PlanOverride(
                match="*.self_attn", tp_divide_attrs=attrs,
            ).to_override()

    def test_inner_out_src_desugar(self):
        """inner_out_src 脱糖：哨兵 / 单输出 DSL / 多输出 DSL / 非法值。"""
        _, spec = PlanOverride(
            match="*.self_attn", inner_out_src="first_input").to_override()
        assert spec.inner_out_src == "first_input"

        _, spec = PlanOverride(
            match="m", inner_out_src={"cp": "shard(2)"}).to_override()
        assert spec.inner_out_src["cp"] == Shard(2)

        _, spec = PlanOverride(match="m", inner_out_src={
            "hidden": {"cp": "shard(2)"},
            "aux": {"tp": "partial"},
        }).to_override()
        assert spec.inner_out_src["hidden"]["cp"] == Shard(2)
        assert spec.inner_out_src["aux"]["tp"] == Partial()

        with pytest.raises(ValueError, match="first_input"):
            PlanOverride(match="m", inner_out_src="bogus").to_override()

    def test_to_override_missing_match_raises(self):
        entry = PlanOverride(match="", inner_wrapper="sdpa_hf")
        with pytest.raises(ValueError, match="match"):
            entry.to_override()

    def test_to_plan_overrides_merges_same_match(self):
        """同 match 的多条 entry 逐字段合并（后者非 None 字段优先）。"""
        overrides = entries_to_plan_overrides([
            PlanOverride(match="*.mlp", region_dispatch=False),
            PlanOverride(match="*.mlp", inner_target="self"),
        ])
        assert set(overrides) == {"*.mlp"}
        spec = overrides["*.mlp"]
        assert spec.region_dispatch is False
        assert spec.inner_target == "self"

    def test_tp_divide_attrs_later_entry_replaces(self):
        overrides = entries_to_plan_overrides([
            PlanOverride(
                match="*.self_attn", tp_divide_attrs=["hidden_size"],
            ),
            PlanOverride(match="*.self_attn", tp_divide_attrs=[]),
        ])
        assert overrides["*.self_attn"].tp_divide_attrs == []


    def test_desugared_entry_equivalent_to_handwritten(self, tiny_llama, make_mesh):
        """脱糖结果与手写 {match: spec} 走同一统一通道（glob merge、契约继承）。"""
        mesh = make_mesh((1,), ("tp",))
        plan = ShardingPlanner(
            plan_overrides=entries_to_plan_overrides([
                PlanOverride(match="*.self_attn", inner_target="self",
                     inner_wrapper="sdpa_hf"),
            ])).plan(tiny_llama, mesh, tp_size=2)
        spec = plan.modules["model.layers.0.self_attn"]
        assert spec.inner_wrapper == "sdpa_hf"
        # planner 推导的参数分片/契约未被触碰（merge 继承）
        assert spec.params["q_proj.weight"]
        assert "hidden_states" in spec.in_dst
        # 两个 layer 均命中；未命中的边界不受影响
        assert plan.modules["model.layers.1.self_attn"].inner_wrapper == "sdpa_hf"
        assert plan.modules["lm_head"].inner_wrapper is None

    def test_desugared_glob_miss_warns(self, tiny_llama, make_mesh, caplog):
        mesh = make_mesh((1,), ("tp",))
        with caplog.at_level(logging.WARNING):
            ShardingPlanner(
                plan_overrides=entries_to_plan_overrides([
                    PlanOverride(match="*.self_atn", inner_wrapper="sdpa_hf"),
                ])).plan(tiny_llama, mesh, tp_size=2)
        assert "hit no boundary spec" in caplog.text

    def test_desugared_composes_with_handwritten_overrides(self, tiny_llama, make_mesh):
        """脱糖 dict 与手写 override 合并传入：exact merge 契约 + glob merge 注入。"""
        from hyper_models.components.distributed.sharding_config import (
            TP,
            ModuleShardingSpec,
        )
        from hyper_parallel.core.dtensor.placement_types import (
            Partial,
            Replicate,
            Shard,
        )
        mesh = make_mesh((1,), ("tp",))
        override = ModuleShardingSpec(
            params={
                "q_proj.weight": {TP: Shard(0), },
                "k_proj.weight": {TP: Shard(0)},
                "v_proj.weight": {TP: Shard(0)},
                "o_proj.weight": {TP: Shard(1)},
            },
            in_src={"hidden_states": {TP: Shard(1)}},
            in_dst={"hidden_states": {TP: Replicate()}},
            out_src={TP: Partial()},
            out_dst={TP: Shard(1)},
        )
        planner = ShardingPlanner(plan_overrides={
            "model.layers.0.self_attn": override,
            **entries_to_plan_overrides([
                PlanOverride(match="*.self_attn", inner_target="self",
                     inner_wrapper="sdpa_hf"),
            ]),
        })
        plan = planner.plan(tiny_llama, mesh, tp_size=2)
        spec = plan.modules["model.layers.0.self_attn"]
        # 用户契约生效（merge 字段粒度替换）+ 注入生效
        assert spec.in_dst["hidden_states"][TP] == Replicate()
        assert spec.inner_wrapper == "sdpa_hf"
        # 另一个 layer 是推导 spec + glob merge 注入
        assert plan.modules["model.layers.1.self_attn"].inner_wrapper \
            == "sdpa_hf"


class TestTpLocalAttrPlan:
    """Planner finalizes automatic and explicit TP-local attributes."""

    def test_d17_auto_attrs_require_no_yaml(self, tiny_llama, make_mesh):
        mesh = make_mesh((1,), ("tp",))
        plan = ShardingPlanner().plan(tiny_llama, mesh, tp_size=2)
        attr_plan = plan.modules[
            "model.layers.0.self_attn"
        ]._tp_local_attr_plan
        assert "num_heads" in attr_plan.auto_divide
        assert attr_plan.user_divide == ()

    def test_explicit_width_attr(self, tiny_llama, make_mesh):
        for layer in tiny_llama.model.layers:
            layer.self_attn.hidden_size = 16
        mesh = make_mesh((1,), ("tp",))
        plan = ShardingPlanner(plan_overrides={
            "*.self_attn": ModuleShardingSpec(
                tp_divide_attrs=["hidden_size"],
            ),
        }).plan(tiny_llama, mesh, tp_size=2)
        attr_plan = plan.modules[
            "model.layers.0.self_attn"
        ]._tp_local_attr_plan
        assert attr_plan.user_divide == ("hidden_size",)

    def test_redundant_auto_attr_fails(self, tiny_llama, make_mesh):
        mesh = make_mesh((1,), ("tp",))
        planner = ShardingPlanner(plan_overrides={
            "*.self_attn": ModuleShardingSpec(
                tp_divide_attrs=["num_heads"],
            ),
        })
        with pytest.raises(ValueError, match="D-17"):
            planner.plan(tiny_llama, mesh, tp_size=2)

    def test_missing_user_attr_fails(self, tiny_llama, make_mesh):
        mesh = make_mesh((1,), ("tp",))
        planner = ShardingPlanner(plan_overrides={
            "*.self_attn": ModuleShardingSpec(
                tp_divide_attrs=["missing_width"],
            ),
        })
        with pytest.raises(ValueError, match="plain int"):
            planner.plan(tiny_llama, mesh, tp_size=2)


class TestHfNativeMoeLocalMapCleared:
    def test_per_expert_layout_cleared(self, tiny_hf_native_moe):
        """HF 原生 per-expert 布局（D-10）：region_dispatch 被清除——模块自己的
        forward 非 EP-aware，必须显式注入 local_compute_fn。"""
        mesh = _meta_mesh((4, 2), ("dp", "tp"))
        plan = ShardingPlanner().plan(
            tiny_hf_native_moe, mesh, tp_size=2, ep_size=2)
        spec = plan.modules["model.layers.0.mlp"]
        assert spec._ep_size == 2
        assert spec.region_dispatch is None

    def test_batched_layout_cleared(self, tiny_hf_batched_moe):
        mesh = _meta_mesh((4, 2), ("dp", "tp"))
        plan = ShardingPlanner().plan(
            tiny_hf_batched_moe, mesh, tp_size=2, ep_size=4)
        spec = plan.modules["model.layers.0.mlp"]
        assert spec._ep_size == 4
        assert spec.region_dispatch is None

    def test_custom_naming_kept(self, tiny_moe):
        """自定义命名（w1/w2/w3，模块作者预堆叠）：EP-aware by construction，
        region_dispatch 保留。"""
        mesh = _meta_mesh((4, 2), ("dp", "tp"))
        plan = ShardingPlanner().plan(tiny_moe, mesh, tp_size=2, ep_size=2)
        spec = plan.modules["model.layers.0.mlp"]
        assert spec._ep_size == 2
        assert spec.region_dispatch is False

    def test_ep1_untouched(self, tiny_hf_native_moe, make_mesh):
        """ep=1 不进入 D-10 标记路径：region_dispatch 保持模板值。"""
        mesh = make_mesh((1,), ("tp",))
        plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2)
        spec = plan.modules["model.layers.0.mlp"]
        assert spec._ep_size == 0
        assert spec.region_dispatch is False


class TestExplicitExpertMesh:
    """expert mesh 的框架统一派生（D-10）：apply 时派生一次，参数分片与注入
    compute 共享同一对象（ep_mesh 上下文）；用户不可经工厂 Target 配置
    expert_mesh（保留上下文制度——配置即 fail-fast）。"""

    def test_expert_mesh_config_key_rejected(self, tiny_hf_native_moe):
        """工厂签名已无 expert_mesh 参数——旧式配置（expert_mesh=...）按
        "未声明的配置键"fail-fast 并列出合法形参（迁移信号明确）。"""
        from hyper_models.components.distributed import build_expert_mesh
        from hyper_models.components.distributed.ep_compute import (
            routed_only_ep_compute_fn,
        )
        from hyper_models.components.distributed.sharding_config import (
            ModuleShardingSpec,
        )
        from hyper_models.components.distributed.sharding_applier import (
            _resolve_local_compute_fn,
        )
        from hyper_models.trainer.config import Target
        mesh = _meta_mesh((4, 2), ("dp", "tp"))
        user_mesh = build_expert_mesh(mesh, ep_size=4)
        overrides = {"*.mlp": ModuleShardingSpec(
            local_compute_fn=Target(
                routed_only_ep_compute_fn,
                target_path="hyper_models.components.distributed."
                            "ep_compute.routed_only_ep_compute_fn",
                expert_mesh=user_mesh), region_dispatch=False)}
        plan = ShardingPlanner(plan_overrides=overrides).plan(
            tiny_hf_native_moe, mesh, tp_size=2, ep_size=4)
        spec = plan.modules["model.layers.0.mlp"]
        mlp = tiny_hf_native_moe.model.layers[0].mlp
        with pytest.raises(ValueError, match="expert_mesh"):
            _resolve_local_compute_fn(
                mlp, spec, mesh, plan.mesh_dim_names, expert_mesh=None)

    def test_ep_mesh_context_key_reserved(self, tiny_hf_native_moe):
        """ep_mesh 是框架保留上下文键——用户配置即 fail-fast（框架统一派生，
        保证 a2a 通信域与专家参数分片域是同一对象）。"""
        from hyper_models.components.distributed.ep_compute import (
            routed_only_ep_compute_fn,
        )
        from hyper_models.components.distributed.sharding_config import (
            ModuleShardingSpec,
        )
        from hyper_models.components.distributed.sharding_applier import (
            _resolve_local_compute_fn,
        )
        from hyper_models.trainer.config import Target
        mesh = _meta_mesh((4, 2), ("dp", "tp"))
        overrides = {"*.mlp": ModuleShardingSpec(
            local_compute_fn=Target(
                routed_only_ep_compute_fn,
                target_path="hyper_models.components.distributed."
                            "ep_compute.routed_only_ep_compute_fn",
                ep_mesh="user-mesh"), region_dispatch=False)}
        plan = ShardingPlanner(plan_overrides=overrides).plan(
            tiny_hf_native_moe, mesh, tp_size=2, ep_size=4)
        spec = plan.modules["model.layers.0.mlp"]
        mlp = tiny_hf_native_moe.model.layers[0].mlp
        with pytest.raises(ValueError, match="framework-reserved context keys"):
            _resolve_local_compute_fn(
                mlp, spec, mesh, plan.mesh_dim_names, expert_mesh=None)


class TestPreflightFailFast:
    def test_cp_without_inner_wrapper_raises(self, tiny_llama):
        """cp>1 + attention 边界无 inner_wrapper → apply 前 fail-fast。"""
        mesh = _meta_mesh((2,), ("cp",))
        plan = ShardingPlanner().plan(tiny_llama, mesh, cp_size=2)
        with pytest.raises(ValueError, match="inner_wrapper"):
            apply_sharding_plan(tiny_llama, plan, mesh)

    def test_cp_with_injection_passes_preflight(self, tiny_llama):
        """显式注入后 preflight 放行（apply 后续阶段需要真实 mesh，单测只
        验证 preflight 本身）。"""
        mesh = _meta_mesh((2,), ("cp",))
        planner = ShardingPlanner(plan_overrides=cp_sdpa_hf_injection())
        plan = planner.plan(tiny_llama, mesh, cp_size=2)
        _preflight_compute_injection(plan, mesh)   # 不抛错即通过

    def test_builtin_cp_wrapper_rejects_dispatch_true(self, tiny_llama):
        """Shipped CP wrappers contain collectives and require black-box validation."""
        mesh = _meta_mesh((2,), ("cp",))
        planner = ShardingPlanner(plan_overrides={
            "*.self_attn": ModuleShardingSpec(
                inner_target="self",
                inner_wrapper="sdpa_hf",
                region_dispatch=True,
            )
        })
        plan = planner.plan(tiny_llama, mesh, cp_size=2)
        with pytest.raises(ValueError, match="requires region_dispatch=False"):
            _preflight_compute_injection(plan, mesh)

    def test_child_wrapper_requires_inner_output_contract(self, tiny_llama):
        """A child target cannot inherit the enclosing boundary output layout."""
        mesh = _meta_mesh((2,), ("cp",))
        planner = ShardingPlanner(plan_overrides={
            "*.self_attn": ModuleShardingSpec(
                inner_target="attention_core",
                inner_wrapper="sdpa_hf",
                region_dispatch=False,
            )
        })
        plan = planner.plan(tiny_llama, mesh, cp_size=2)
        with pytest.raises(ValueError, match="inner_out_src.*first_input"):
            _preflight_compute_injection(plan, mesh)

    def test_cp_size1_no_check(self, tiny_llama, make_mesh):
        """cp 轴 size=1（plan 已过滤 cp 维）→ 无 preflight 要求。"""
        mesh = make_mesh((1,), ("tp",))
        plan = ShardingPlanner().plan(tiny_llama, mesh, tp_size=1)
        _preflight_compute_injection(plan, mesh)

    def test_ep_without_compute_fn_raises(self, tiny_hf_native_moe):
        """ep>1（D-10）+ HF 原生 MoE 无 local_compute_fn → fail-fast。"""
        mesh = _meta_mesh((4, 2), ("dp", "tp"))
        plan = ShardingPlanner().plan(
            tiny_hf_native_moe, mesh, tp_size=2, ep_size=2)
        with pytest.raises(ValueError, match="local_compute_fn"):
            apply_sharding_plan(tiny_hf_native_moe, plan, mesh)

    def test_ep_with_injection_passes_preflight(self, tiny_hf_native_moe):
        mesh = _meta_mesh((4, 2), ("dp", "tp"))
        planner = ShardingPlanner(plan_overrides=ep_archetype_injection())
        plan = planner.plan(tiny_hf_native_moe, mesh, tp_size=2, ep_size=2)
        spec = plan.modules["model.layers.0.mlp"]
        assert spec.local_compute_fn is not None   # Target 已 overlay
        _preflight_compute_injection(plan, mesh)

    def test_ep_aware_module_region_dispatch_passes(self, tiny_moe):
        """自研 EP-aware 模块（region_dispatch 保留）→ 无 local_compute_fn
        也放行（模块 forward 自带 a2a）。"""
        mesh = _meta_mesh((4, 2), ("dp", "tp"))
        plan = ShardingPlanner().plan(tiny_moe, mesh, tp_size=2, ep_size=2)
        _preflight_compute_injection(plan, mesh)

    def test_ep_error_message_teaches_yaml(self, tiny_hf_native_moe):
        """报错信息教学化：包含可粘贴的 YAML 片段与默认实现路径。"""
        mesh = _meta_mesh((4, 2), ("dp", "tp"))
        plan = ShardingPlanner().plan(
            tiny_hf_native_moe, mesh, tp_size=2, ep_size=2)
        with pytest.raises(ValueError) as exc:
            apply_sharding_plan(tiny_hf_native_moe, plan, mesh)
        msg = str(exc.value)
        assert "ep_compute.qwen2moe_ep_compute_fn" in msg
        assert "region_dispatch" in msg


# ==========================================================================
# 来源: test_dist_s5_plan_overrides.py
# S5.7（2 进程）: plan_overrides 端到端 —— 自研多输入 attention 双模式等价（05 §3.6.7）。
# ==========================================================================

class MultiInputAttention(TinyLlamaAttention):
    """forward(self, attn_bias, x, ...)：x 在位置 1 且名字非 hidden_states。"""

    def forward(self, attn_bias, x, position_ids=None):
        return TinyLlamaAttention.forward(self, x, position_ids)


class MultiInputDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = TinyRMSNorm(config.hidden_size)
        self.self_attn = MultiInputAttention(config)
        self.post_attention_layernorm = TinyRMSNorm(config.hidden_size)
        self.mlp = TinyLlamaMLP(config)

    def forward(self, hidden_states, position_ids=None):
        hidden_states = hidden_states + self.self_attn(
            None, self.input_layernorm(hidden_states), position_ids)
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states))
        return hidden_states


class MultiInputModel(TinyLlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            MultiInputDecoderLayer(config)
            for _ in range(config.num_hidden_layers))


class MultiInputForCausalLM(TinyLlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = MultiInputModel(config)


def _build():
    torch.manual_seed(1234)
    return MultiInputForCausalLM(TinyConfig()).eval()


def _worker(rank, world_size):
    ref_model = _build()
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref = ref_model(x)

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    for mode in ("production", "validate"):
        model = _build()
        # 先推导一次拿模板填充的 spec，把契约 key 改为真实签名参数名后作为
        # override 回注——用户只需关心 key，placement 沿用模板推导结果
        base_plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
        overrides = {}
        for fqn, spec in base_plan.modules.items():
            if fqn.endswith("self_attn"):
                new_spec = copy.deepcopy(spec)
                for attr in ("in_src", "in_dst"):
                    d = getattr(new_spec, attr)
                    d["x"] = d.pop("hidden_states")
                overrides[fqn] = new_spec
        assert len(overrides) == 2

        planner = ShardingPlanner(plan_overrides=overrides)
        plan = planner.plan(model, mesh, tp_size=world_size)
        assert set(plan.modules["model.layers.0.self_attn"].in_src) == {"x"}

        model, _ = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))
        with torch.no_grad():
            out = model(x)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_plan_overrides_multi_input_e2e_tp2():
    run_dist(2, _worker)
