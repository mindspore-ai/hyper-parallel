# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_s4_ep.py: 核心套件合并文件。

来源: test_s4_local_compute_fn.py, test_s4_moe_gate_compile.py, test_s5_hf_native_moe.py, test_s6_ep_extend.py
"""

# Injection factories intentionally keep the complete runtime callback signature,
# and these tests directly validate private planner metadata.
# pylint: disable=unused-argument,protected-access

import functools
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from hyper_models.components.distributed import ep_compute
from hyper_models.components.distributed.ep_compute import routed_only_ep_compute_fn
from hyper_models.components.distributed.ep_utils import (
    MOE_ROUTER_ADAPTERS,
    bind_local_expert_forward,
    ep_routed_forward,
    _local_swiglu_expert_forward,
    _sigmoid_group_router,
    _softmax_topk_router,
    resolve_swiglu_weights,
    _topk_router_module,
)
from hyper_models.components.distributed.injection import (
    local_compute,
)
from hyper_models.components.distributed.precompiled_boundary import PrecompiledBoundary
from hyper_models.components.distributed.sharding.apply import (
    _StackedExperts,
    _stack_moe_experts,
)
from hyper_models.components.distributed.sharding_applier import (
    _apply_phase_c,
    _expert_mesh_layout,
    _rewrap_local_outputs,
    _resolve_local_compute_fn,
    _wrap_local_region_forward,
)
from hyper_models.components.distributed.sharding_config import (
    CP,
    EP,
    ModuleShardingSpec,
    ShardingPlan,
    TEMPLATES,
    TP,
    _normalize_out_fields,
)
from hyper_models.components.distributed.sharding_planner import ShardingPlanner
from hyper_models.trainer.config import Target
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import (
    Replicate,
    Shard,
)
from tests.components.distributed.conftest import _ensure_pg


# ==========================================================================
# 来源: test_s4_local_compute_fn.py
# S4.6: local_compute_fn 用户自定义 compute_fn + 派生门控（05 §4.4.3，单进程）。
# ==========================================================================

class _TinyMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = _TinyMod()

    def forward(self, x):
        return self.mod(x)


def _identity_spec():
    return _normalize_out_fields(ModuleShardingSpec(
        in_src={"x": {TP: Shard(1)}},
        in_dst={"x": {TP: Shard(1)}},   # identity
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
    ))


class TestRewrapLocalOutputs:
    def test_preserves_list_and_wraps_all_declared_outputs(self, make_mesh, monkeypatch):
        """Every declared Tensor output is wrapped and list remains list."""
        mesh = make_mesh((1,), ("tp",))
        spec = ModuleShardingSpec(
            out_src={
                "hidden": {TP: Replicate()},
                "aux": {TP: Shard(0)},
            },
            out_names=["hidden", "aux", "metadata"],
        )
        calls = []

        def fake_from_local(tensor, device_mesh, placements):
            calls.append((tensor, device_mesh, placements))
            return f"wrapped-{len(calls)}"

        monkeypatch.setattr(
            "hyper_models.components.distributed.sharding_applier.DTensor.from_local",
            fake_from_local,
        )
        outputs = [torch.ones(2), torch.zeros(2), None]
        result = _rewrap_local_outputs(outputs, spec, mesh, ("tp",), "TestModule")

        assert isinstance(result, list)
        assert result == ["wrapped-1", "wrapped-2", None]
        assert len(calls) == 2

    def test_declared_output_index_out_of_range_fails(self, make_mesh):
        """A stale out_names contract fails with boundary context."""
        mesh = make_mesh((1,), ("tp",))
        spec = ModuleShardingSpec(
            out_src={"aux": {TP: Replicate()}},
            out_names=["hidden", "aux"],
        )
        with pytest.raises(ValueError, match="TestModule.*only 1 output"):
            _rewrap_local_outputs(
                (torch.ones(2),), spec, mesh, ("tp",), "TestModule")


class TestResolveLocalComputeFn:
    """Validate local-compute factory resolution and contract checks."""

    def test_user_fn_wins_even_with_ep_size(self, make_mesh):
        """local_compute_fn 是解析链环 1：_ep_size 元数据在时仍直接返回用户
        fn（内置 EP 自动注入链路已删除，_ep_size 只驱动参数分片）。"""
        built = []

        @local_compute
        def my_fn(mesh, tp_mesh, cp_mesh, ep_mesh):
            def compute_fn(module, x):
                return x
            built.append(compute_fn)
            return compute_fn

        spec = _identity_spec()
        spec.local_compute_fn = my_fn
        spec.region_dispatch = False
        spec._ep_size = 2
        fn = _resolve_local_compute_fn(
            _TinyMod(), spec, make_mesh((1,), ("tp",)), ("tp",),
            expert_mesh=None)
        assert isinstance(fn, functools.partial)
        # 工厂形态：apply 期调用一次工厂，partial 绑定的是工厂返回的 compute_fn
        assert len(built) == 1 and fn.func is built[0]

    def test_ep_size_alone_returns_none(self, make_mesh):
        """改造后 _ep_size>0 不再注入任何 compute——无 local_compute_fn 且
        region_dispatch=False → None（apply 侧 preflight 会对此 fail-fast）。"""
        spec = _identity_spec()
        spec._ep_size = 2
        fn = _resolve_local_compute_fn(
            _TinyMod(), spec, make_mesh((1,), ("tp",)), ("tp",),
            expert_mesh=None)
        assert fn is None

    def test_region_dispatch_resolves_to_module_forward(self, make_mesh):
        """region_dispatch 纯门控（无用户 fn）→ 模块自身 forward。"""
        mod = _TinyMod()
        spec = _identity_spec()
        spec.region_dispatch = False
        fn = _resolve_local_compute_fn(
            mod, spec, make_mesh((1,), ("tp",)), ("tp",), expert_mesh=None)
        assert fn == mod.forward  # pylint: disable=comparison-with-callable

    def test_inner_wrapper_does_not_resolve_module_forward(self, make_mesh):
        """inner_wrapper 托管局部计算时，不选择整个模块 forward 作为骨架。"""
        spec = _identity_spec()
        spec.region_dispatch = False
        spec.inner_wrapper = "sdpa_hf"
        fn = _resolve_local_compute_fn(
            _TinyMod(), spec, make_mesh((1,), ("tp",)), ("tp",),
            expert_mesh=None)
        assert fn is None

    def test_no_declaration_returns_none(self, make_mesh):
        """两个来源皆无 → None（模块不走骨架，门控派生为 False）。"""
        fn = _resolve_local_compute_fn(
            _TinyMod(), _identity_spec(), make_mesh((1,), ("tp",)), ("tp",),
            expert_mesh=None)
        assert fn is None

    def test_derived_gate_via_apply_path(self, make_mesh):
        """派生门控端到端：region_dispatch=False + local_compute_fn →
        _apply_phase_c 仍注入骨架并执行 custom fn（门控不读存储的 bool）。"""
        mesh = make_mesh((1,), ("tp",))
        calls = []

        @local_compute
        def my_compute(mesh, tp_mesh, cp_mesh, ep_mesh):
            def compute_fn(module, x):
                calls.append(x)
                return module.lin(x) * 3
            return compute_fn

        model = _TinyModel()
        spec = _identity_spec()
        spec.local_compute_fn = my_compute
        spec.region_dispatch = False   # 注入纪律：显式声明（黑盒托管）
        plan = ShardingPlan(modules={"mod": spec}, mesh_dim_names=("tp",))
        _apply_phase_c(model, plan, mesh, validate_mode=False)

        x = torch.randn(2, 4)
        out = model.mod(x)
        assert len(calls) == 1               # custom fn 被执行 → 骨架已注入
        torch.testing.assert_close(out, model.mod.lin(x) * 3)


class TestTargetLocalComputeFn:
    """local_compute_fn 的 Target 工厂形态（YAML sharding.injections 载体）。"""

    def test_target_factory_built_with_context(self, make_mesh):
        """Target 工厂：apply 时 build（通用上下文 module/mesh/expert_mesh
        按签名过滤），返回的 compute fn 被 partial 绑定 module。"""
        seen = []

        @local_compute
        def my_factory(module, mesh, tp_mesh, cp_mesh, ep_mesh):
            seen.append((module, mesh, tp_mesh, cp_mesh, ep_mesh))

            def compute_fn(mod, x):
                return mod.lin(x) * 2
            return compute_fn

        mesh = make_mesh((1,), ("tp",))
        mod = _TinyMod()
        spec = _identity_spec()
        spec.local_compute_fn = Target(
            my_factory, target_path="tests.my_factory")
        spec.region_dispatch = False
        fn = _resolve_local_compute_fn(
            mod, spec, mesh, ("tp",), expert_mesh=None)
        assert isinstance(fn, functools.partial)
        assert seen and seen[0][0] is mod        # module 上下文已注入
        assert seen[0][2] is mesh["tp"]          # tp_mesh 框架填充
        assert seen[0][3] is None                # 无 cp 轴 → cp_mesh=None
        assert seen[0][4] is None                # 无 EP → ep_mesh=None
        x = torch.randn(2, 4)
        torch.testing.assert_close(fn(x), mod.lin(x) * 2)

    def test_config_keys_pass_through_untouched(self, make_mesh):
        """配置键纯用户所有：框架只填上下文，不做任何自动填充——未配置
        的配置键工厂收到其默认值（None），配置了就原样直达。"""
        seen = []

        @local_compute
        def cfg_factory(mesh, tp_mesh, cp_mesh, ep_mesh, block_size=None):
            seen.append(block_size)

            def compute_fn(mod, x):
                return x
            return compute_fn

        mesh = make_mesh((1,), ("tp",))
        spec = _identity_spec()
        spec.local_compute_fn = Target(
            cfg_factory, target_path="tests.cfg_factory")
        spec.region_dispatch = False
        _resolve_local_compute_fn(
            _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)
        assert seen == [None]                    # 框架不填充配置键

        spec2 = _identity_spec()
        spec2.local_compute_fn = Target(
            cfg_factory, target_path="tests.cfg_factory", block_size=128)
        spec2.region_dispatch = False
        _resolve_local_compute_fn(
            _TinyMod(), spec2, mesh, ("tp",), expert_mesh=None)
        assert seen[-1] == 128                   # 用户显式配置直达

    def test_target_bad_return_raises(self, make_mesh):
        """Target 工厂返回非 callable → TypeError（契约：必须返回 compute fn）。"""
        @local_compute
        def bad_factory(mesh, tp_mesh, cp_mesh, ep_mesh):
            return 42

        spec = _identity_spec()
        spec.local_compute_fn = Target(bad_factory, target_path="tests.bad")
        spec.region_dispatch = False
        with pytest.raises(TypeError, match="local_compute_fn"):
            _resolve_local_compute_fn(
                _TinyMod(), spec, make_mesh((1,), ("tp",)), ("tp",),
                expert_mesh=None)

    def test_target_undecorated_factory_raises(self, make_mesh):
        """注入纪律：Target 指向未装饰的工厂 → fail-fast 提示 @local_compute。"""
        spec = _identity_spec()
        spec.local_compute_fn = Target(lambda: 42, target_path="tests.bad")
        spec.region_dispatch = False
        with pytest.raises(TypeError, match="@local_compute"):
            _resolve_local_compute_fn(
                _TinyMod(), spec, make_mesh((1,), ("tp",)), ("tp",),
                expert_mesh=None)

    def test_plain_callable_undecorated_raises(self, make_mesh):
        """注入纪律：未装饰的可调用对象 → fail-fast 提示 @local_compute。"""
        spec = _identity_spec()
        spec.local_compute_fn = lambda module, x: x
        spec.region_dispatch = False
        with pytest.raises(TypeError, match="@local_compute"):
            _resolve_local_compute_fn(
                _TinyMod(), spec, make_mesh((1,), ("tp",)), ("tp",),
                expert_mesh=None)

    def test_compute_fn_param_mismatch_raises(self, make_mesh):
        """原则 1：compute fn 入参与原 forward 不匹配 → apply 时 fail-fast。"""
        @local_compute
        def bad_compute(mesh, tp_mesh, cp_mesh, ep_mesh):
            def compute_fn(module, hidden):   # 原 forward 形参名是 x
                return hidden
            return compute_fn

        spec = _identity_spec()
        spec.local_compute_fn = bad_compute
        spec.region_dispatch = False
        with pytest.raises(TypeError, match="同名"):
            _resolve_local_compute_fn(
                _TinyMod(), spec, make_mesh((1,), ("tp",)), ("tp",),
                expert_mesh=None)

    def test_target_typo_config_key_raises(self, make_mesh):
        """配置了工厂未声明的键（rounter 拼写错误）→ fail-fast 并列出合法
        形参——配置键按名绑定，拼错不得被静默吞掉。"""
        spec = _identity_spec()
        spec.local_compute_fn = Target(
            routed_only_ep_compute_fn,
            target_path="hyper_models.components.distributed."
                        "ep_compute.routed_only_ep_compute_fn",
            blok_size="oops")                     # 拼写错误：应为 block_size
        spec.region_dispatch = False
        with pytest.raises(ValueError, match="未声明的键"):
            _resolve_local_compute_fn(
                _TinyMod(), spec, make_mesh((1,), ("tp",)), ("tp",),
                expert_mesh=None)

    def test_target_reserved_context_key_raises(self, make_mesh):
        """上下文键是框架保留名：用户在 Target 里配置 mesh → fail-fast
        （mesh 家族只能由框架填充）。"""
        spec = _identity_spec()
        spec.local_compute_fn = Target(
            routed_only_ep_compute_fn,
            target_path="hyper_models.components.distributed."
                        "ep_compute.routed_only_ep_compute_fn",
            mesh="oops")
        spec.region_dispatch = False
        with pytest.raises(ValueError, match="保留"):
            _resolve_local_compute_fn(
                _TinyMod(), spec, make_mesh((1,), ("tp",)), ("tp",),
                expert_mesh=None)


class TestLocalRegionWithCustomComputeFn:
    """Validate custom compute execution inside local regions."""

    def _wrap(self, mod, spec, mesh, validate_mode):
        boundary = PrecompiledBoundary(spec, mesh, ("tp",))
        compute_fn = _resolve_local_compute_fn(
            mod, spec, mesh, ("tp",), expert_mesh=None)
        _wrap_local_region_forward(
            mod, boundary, spec, mesh, ("tp",),
            validate_mode=validate_mode, compute_fn=compute_fn)

    def test_custom_compute_fn_runs_in_region(self, make_mesh):
        """production：custom compute_fn 收到 (module, local tensor)，
        输出经骨架 boundary 出口返回 local。"""
        mesh = make_mesh((1,), ("tp",))
        calls = []

        @local_compute
        def my_compute(mesh, tp_mesh, cp_mesh, ep_mesh):
            def compute_fn(module, x):
                calls.append((module, x))
                return module.lin(x) * 2   # 自定义逻辑：放大 2 倍
            return compute_fn

        mod = _TinyMod()
        spec = _identity_spec()
        spec.local_compute_fn = my_compute
        spec.region_dispatch = False
        self._wrap(mod, spec, mesh, validate_mode=False)

        x = torch.randn(2, 4)
        out = mod(x)
        assert calls and calls[0][0] is mod
        torch.testing.assert_close(out, mod.lin(x) * 2)

    def test_custom_compute_fn_validate_mode(self, make_mesh):
        """validate：DTensor 输入由骨架 unwrap——compute_fn 收到的仍是
        local tensor（无模式感知），出口重包装后解包返回。"""
        mesh = make_mesh((1,), ("tp",))
        seen = []

        @local_compute
        def my_compute(mesh, tp_mesh, cp_mesh, ep_mesh):
            def compute_fn(module, x):
                seen.append(x)
                return module.lin(x)
            return compute_fn

        mod = _TinyMod()
        spec = _identity_spec()
        spec.local_compute_fn = my_compute
        spec.region_dispatch = False
        self._wrap(mod, spec, mesh, validate_mode=True)

        x = torch.randn(2, 4)
        out = mod(x)
        assert len(seen) == 1
        assert not isinstance(seen[0], DTensor)   # compute_fn 内恒为 local
        assert not isinstance(out, DTensor)       # 骨架出口恒解包
        torch.testing.assert_close(out, mod.lin(x))


class TestRegionDispatchDeclaration:
    """region_dispatch 注入纪律（无默认）：声明注入必须显式给出。"""

    def test_local_compute_fn_without_region_dispatch_fails(self, make_mesh):
        @local_compute
        def my_fn(mesh, tp_mesh, cp_mesh, ep_mesh):
            def compute_fn(module, x):
                return x
            return compute_fn
        spec = _identity_spec()
        spec.local_compute_fn = my_fn
        with pytest.raises(ValueError, match="region_dispatch"):
            _resolve_local_compute_fn(
                _TinyMod(), spec, make_mesh((1,), ("tp",)), ("tp",),
                expert_mesh=None)

    def test_redundant_true_without_injection_fails(self, make_mesh):
        """region_dispatch=True 但无注入 → fail-fast（普通边界天然穿透，
        声明冗余）。"""
        model = _TinyModel()
        spec = _identity_spec()
        spec.region_dispatch = True
        mesh = make_mesh((1,), ("tp",))
        plan = ShardingPlan(modules={"mod": spec}, mesh_dim_names=("tp",))
        with pytest.raises(ValueError, match="冗余"):
            _apply_phase_c(model, plan, mesh, validate_mode=False)


class TestLocalRegionDispatchThrough:
    """region_dispatch=True：validate 穿透注入函数（纯标准算子），
    策略传播覆盖注入物，out_src 从声明式重包升级为真校验。"""

    def _wrap(self, mod, spec, mesh, validate_mode):
        boundary = PrecompiledBoundary(spec, mesh, ("tp",))
        compute_fn = _resolve_local_compute_fn(
            mod, spec, mesh, ("tp",), expert_mesh=None)
        _wrap_local_region_forward(
            mod, boundary, spec, mesh, ("tp",),
            validate_mode=validate_mode, compute_fn=compute_fn)

    def test_dispatch_through_validate(self, make_mesh):
        """validate：DTensor 直入注入函数（不经 to_local），传播结果与
        out_src 声明一致 → 通过；production 行为不变。"""
        mesh = make_mesh((1,), ("tp",))
        seen = {}

        @local_compute
        def my_compute(mesh, tp_mesh, cp_mesh, ep_mesh):
            def compute_fn(module, x):
                seen["x_is_dtensor"] = isinstance(x, DTensor)
                return x * 2 + x              # 纯 pointwise：可 dispatch
            return compute_fn

        mod = _TinyMod()
        spec = _identity_spec()
        spec.local_compute_fn = my_compute
        spec.region_dispatch = True       # 注入物纯标准算子
        self._wrap(mod, spec, mesh, validate_mode=True)

        # size-1 mesh 的边界入口不包 DTensor（退化跳过）——直接喂 DTensor
        # （等价于 D-14 嵌套场景下外层边界传来的输入）
        x = torch.randn(2, 4)
        dt = DTensor.from_local(x, mesh, (Shard(1),))
        out = mod(dt)
        assert seen["x_is_dtensor"] is True    # validate 穿透：注入函数见 DTensor
        assert not isinstance(out, DTensor)    # 骨架出口恒解包
        torch.testing.assert_close(out, x * 3)

    def test_dispatch_through_out_src_mismatch_fails(self, make_mesh):
        """真校验：传播结果（pointwise → Shard(1) 保持）与声明的 out_src
        （Shard(0)）不符 → fail-fast——黑盒模式抓不到这类注入物 bug。"""
        mesh = make_mesh((1,), ("tp",))

        @local_compute
        def my_compute(mesh, tp_mesh, cp_mesh, ep_mesh):
            def compute_fn(module, x):
                return x * 2
            return compute_fn

        mod = _TinyMod()
        spec = _identity_spec()
        spec.out_src = {"output": {TP: Shard(0)}}   # 撒谎的声明
        spec.local_compute_fn = my_compute
        spec.region_dispatch = True
        self._wrap(mod, spec, mesh, validate_mode=True)

        dt = DTensor.from_local(torch.randn(2, 4), mesh, (Shard(1),))
        with pytest.raises(Exception, match="out_src"):
            mod(dt)

    def test_dispatch_through_production_unchanged(self, make_mesh):
        """production：region_dispatch=True 无任何分支变化（local 直通）。"""
        mesh = make_mesh((1,), ("tp",))
        seen = {}

        @local_compute
        def my_compute(mesh, tp_mesh, cp_mesh, ep_mesh):
            def compute_fn(module, x):
                seen["x_is_dtensor"] = isinstance(x, DTensor)
                return x * 2
            return compute_fn

        mod = _TinyMod()
        spec = _identity_spec()
        spec.local_compute_fn = my_compute
        spec.region_dispatch = True
        self._wrap(mod, spec, mesh, validate_mode=False)

        x = torch.randn(2, 4)
        out = mod(x)
        assert seen["x_is_dtensor"] is False   # production 恒 local
        torch.testing.assert_close(out, x * 2)


class _TinyMoeMod(nn.Module):
    """Minimal MoE-shaped module for archetype factory tests (gate + experts
    [+ shared_expert + shared_expert_gate])."""
    def __init__(self, with_shared=False):
        super().__init__()
        self.gate = nn.Linear(4, 4, bias=False)
        self.experts = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        if with_shared:
            self.shared_expert = nn.Linear(4, 4)
            self.shared_expert_gate = nn.Linear(4, 1, bias=False)


class TestEpArchetypeFactories:
    """内置 EP archetype 工厂（ep_compute.py）：mesh 家族上下文的使用 +
    路由显式选择 + apply 期接口断言（accuracy_fix_plan.md §3 E2）。"""

    class _FakeEpMesh:
        def __getitem__(self, name):
            assert name == "ep"
            return self

        def get_group(self, name):
            return f"group-{name}"

        def size(self):
            return 2

    def _capture(self, monkeypatch):
        """Patch EP primitives and return the metadata captured by the fakes."""
        captured = {}

        def fake_compute(module, hidden_states, *, router_fn, ep_group):
            captured.update(router_fn=router_fn, ep_group=ep_group)
            return hidden_states

        def fake_bind(module, ep_size):
            captured.update(bound_module=module, ep_size=ep_size)

        monkeypatch.setattr(ep_compute, "ep_routed_forward", fake_compute)
        monkeypatch.setattr(ep_compute, "bind_local_expert_forward", fake_bind)
        return captured

    def test_mesh_family_used_directly(self, make_mesh, monkeypatch):
        """ep_mesh 由框架填入，工厂直接取 ep_mesh.get_group("ep") 交给
        ep_routed_forward；路由内嵌（默认 softmax top-k）；无 tp_group —
        nested boundary 的 TP 通信由子边界自封（契约见 ep_utils）。"""
        mesh = make_mesh((1,), ("tp",))
        module = _TinyMoeMod()
        captured = self._capture(monkeypatch)
        compute_fn = ep_compute.routed_only_ep_compute_fn(
            module=module, mesh=mesh, tp_mesh=mesh["tp"], cp_mesh=None,
            ep_mesh=self._FakeEpMesh())
        compute_fn(module, torch.randn(2, 4))
        assert captured["ep_group"] == "group-ep"
        assert captured["bound_module"] is module
        assert captured["ep_size"] == 2
        # 路由内嵌：默认 softmax top-k（框架不决定 router，spec 无此字段）
        assert captured["router_fn"] is _softmax_topk_router

    def test_interface_assertion_fails_fast(self, monkeypatch):
        """选错 archetype（模块缺 shared_expert/shared_expert_gate）→
        apply 期 ValueError，报错列出模块实际子模块名。"""
        self._capture(monkeypatch)
        module = _TinyMoeMod(with_shared=False)
        with pytest.raises(ValueError, match="shared_expert") as exc_info:
            ep_compute.qwen2moe_ep_compute_fn(
                module=module, mesh=None, tp_mesh=None, cp_mesh=None,
                ep_mesh=self._FakeEpMesh())
        msg = str(exc_info.value)
        assert "gate" in msg and "experts" in msg  # 实际子模块名可见

    def test_qwen2moe_factory_combines_shared_and_gate(self, monkeypatch):
        """qwen2moe archetype 合并公式：routed + sigmoid(gate(x)) * shared(x)
        —— shared_expert 调用是普通子模块调用（nested boundary 契约），
        不做任何补偿通信。"""
        captured = self._capture(monkeypatch)
        module = _TinyMoeMod(with_shared=True)
        compute_fn = ep_compute.qwen2moe_ep_compute_fn(
            module=module, mesh=None, tp_mesh=None, cp_mesh=None,
            ep_mesh=self._FakeEpMesh())
        x = torch.randn(2, 4)
        out = compute_fn(module, x)
        expected = x + torch.sigmoid(module.shared_expert_gate(x)) * module.shared_expert(x)
        torch.testing.assert_close(out, expected)
        assert captured["router_fn"] is MOE_ROUTER_ADAPTERS["qwen2moe"]

    def test_qwen3_factory_embeds_topk_router(self, monkeypatch):
        """Qwen3-MoE uses its explicit TopKRouter factory."""
        module = _TinyMoeMod()
        captured = self._capture(monkeypatch)
        compute_fn = ep_compute.qwen3moe_ep_compute_fn(
            module=module,
            mesh=None,
            tp_mesh=None,
            cp_mesh=None,
            ep_mesh=self._FakeEpMesh(),
        )
        compute_fn(module, torch.randn(2, 4))

        assert captured["router_fn"] is MOE_ROUTER_ADAPTERS["qwen3moe"]
        assert captured["bound_module"] is module

    def test_mixtral_factory_uses_tuple_router_and_training_jitter(self, monkeypatch):
        """Mixtral 5.12 uses its tuple router and jitters the expert input in training."""
        module = _TinyMoeMod()
        module.jitter_noise = 0.2
        module.train()
        captured = self._capture(monkeypatch)
        compute_fn = ep_compute.mixtral_ep_compute_fn(
            module=module,
            mesh=None,
            tp_mesh=None,
            cp_mesh=None,
            ep_mesh=self._FakeEpMesh(),
        )
        hidden_states = torch.ones(2, 4)
        torch.manual_seed(17)
        output = compute_fn(module, hidden_states)
        torch.manual_seed(17)
        expected = hidden_states * torch.empty_like(hidden_states).uniform_(0.8, 1.2)

        torch.testing.assert_close(output, expected)
        assert captured["router_fn"] is MOE_ROUTER_ADAPTERS["mixtral"]

    def test_mixtral_factory_disables_jitter_in_eval(self, monkeypatch):
        """Mixtral evaluation preserves hidden states even when jitter is configured."""
        module = _TinyMoeMod()
        module.jitter_noise = 0.2
        module.eval()
        self._capture(monkeypatch)
        compute_fn = ep_compute.mixtral_ep_compute_fn(
            module=module,
            mesh=None,
            tp_mesh=None,
            cp_mesh=None,
            ep_mesh=self._FakeEpMesh(),
        )
        hidden_states = torch.randn(2, 4)
        torch.testing.assert_close(compute_fn(module, hidden_states), hidden_states)

    def test_custom_factory_embeds_its_router(self, make_mesh, monkeypatch):
        """路由是注入函数的一部分：自定义工厂按名引用 MOE_ROUTER_ADAPTERS
        的适配器并写进自己的 compute fn——框架不参与选择。"""
        captured = self._capture(monkeypatch)

        @local_compute
        def qwen3moe_ep_factory(mesh, tp_mesh, cp_mesh, ep_mesh):
            ep_group = ep_mesh.get_group("ep")

            def compute_fn(module, hidden_states):
                return ep_compute.ep_routed_forward(
                    module, hidden_states,
                    router_fn=MOE_ROUTER_ADAPTERS["qwen3moe"],
                    ep_group=ep_group)
            return compute_fn

        fn = qwen3moe_ep_factory(
            mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=self._FakeEpMesh())
        fn(_TinyMoeMod(), torch.randn(2, 4))
        assert captured["router_fn"] is MOE_ROUTER_ADAPTERS["qwen3moe"]
        assert captured["ep_group"] == "group-ep"

    def test_factory_requires_ep_mesh(self):
        """非 EP 边界（框架填入 ep_mesh=None）→ 配置错误 fail-fast。"""
        mesh = type("M", (), {"mesh_dim_names": ("tp",)})()
        with pytest.raises(ValueError, match="ep_mesh"):
            ep_compute.routed_only_ep_compute_fn(
                module=_TinyMoeMod(), mesh=mesh, tp_mesh=None, cp_mesh=None,
                ep_mesh=None)


# ==========================================================================
# 来源: test_s4_moe_gate_compile.py
# S4.3: moe_gate 模板 EP redistribute（out_dst {EP: Shard(0)}）编译。
# ==========================================================================

class _FakeMesh:
    mesh_dim_names = ("tp", "ep")


def _moe_gate_spec():
    t = TEMPLATES["moe_gate"]
    spec = ModuleShardingSpec(
        in_src=t.sp_in_src,
        in_dst=t.sp_in_dst,
        out_src=t.sp_out_src,
        out_dst=t.sp_out_dst,
    )
    return _normalize_out_fields(spec)


def test_moe_gate_out_plan_has_ep_redistribute():
    spec = _moe_gate_spec()
    b = PrecompiledBoundary(spec, _FakeMesh(), ("tp", "ep"))
    assert len(b.out_plan) == 1
    op = b.out_plan[0]
    ep_idx = ("tp", "ep").index("ep")
    # out_dst EP 维 Replicate → Shard(0)
    assert op.src_placements[ep_idx] == Replicate()
    assert op.dst_placements[ep_idx] == Shard(0)
    assert op.collective_type == "redistribute"


def test_moe_gate_in_plan_tp_allgather():
    spec = _moe_gate_spec()
    b = PrecompiledBoundary(spec, _FakeMesh(), ("tp", "ep"))
    assert len(b.in_plan) == 1
    assert b.in_plan[0].collective_type == "all_gather"


# ==========================================================================
# 来源: test_s5_hf_native_moe.py
# S5.8: D-09/D-10 HF 原生 MoE EP 直通（05 §6.4.7/§6.4.8）单进程用例。
# ==========================================================================

def _meta_mesh(shape, names):
    """仅元数据的 mesh（planner 测试不需要真实进程组，但 DeviceMesh
    构造需要默认 PG 存在——与 make_mesh 的 _ensure_pg 同理）。"""
    _ensure_pg()
    n = 1
    for s in shape:
        n *= s
    return init_device_mesh("cpu", tuple(shape), mesh_dim_names=tuple(names),
                            rank_list=tuple(range(n)), init_backend=False)


def test_planner_marks_hf_native_moe(tiny_hf_native_moe):
    """per-expert 参数 + ep>1 → stacked 元数据 + TP-extend-EP 契约（D-09a/D-10）。"""
    mesh = _meta_mesh((4, 2), ("dp", "tp"))
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2, ep_size=2)

    spec = plan.modules["model.layers.0.mlp"]
    # 数字段守卫生效：边界聚合在 mlp，无 per-expert 边界
    assert not any("experts.0" in fqn for fqn in plan.modules)

    # stacked 条目（D-10 TP-extend-EP：仅 {EP: S0} expert 维切分，无 TP 键、
    # 无第二轴）
    for proj in ("gate_proj", "up_proj", "down_proj"):
        p = spec.params[f"experts.{proj}"]
        assert p[EP] == Shard(0)
        assert TP not in p and p[CP] == Replicate()

    # per-expert 条目已移除；router 全复制
    assert not any("experts.0" in k for k in spec.params)
    assert spec.params["gate.weight"][TP] == Replicate()

    # _ep_stack 元数据：stacked 名 → 按 expert idx 排序的源路径
    assert set(spec._ep_stack) == {
        "experts.gate_proj", "experts.up_proj", "experts.down_proj"}
    assert spec._ep_stack["experts.gate_proj"] == [
        f"experts.{i}.gate_proj.weight" for i in range(4)]
    # TP-extend-EP：_ep_size = ep_size，边界 identity
    assert spec._ep_size == 2
    assert spec.in_dst["x_BLD"][TP] == Shard(1)


def test_planner_no_mark_without_ep(tiny_hf_native_moe, make_mesh):
    """ep=1 → 不堆叠，per-expert 条目保留（TP-only 语义正确）。"""
    mesh = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_stack == {}
    assert "experts.0.gate_proj.weight" in spec.params
    assert spec.params["experts.0.gate_proj.weight"][TP] == Shard(0)


def test_planner_pre_stacked_d10_ep_extend(tiny_moe):
    """自定义命名（experts.w1 3D）→ D-10 TP-extend-EP 路径：{EP: Shard(0)}，
    无 TP 键，SP-in identity 边界，_ep_stack 为空（已 stacked）。"""
    mesh = _meta_mesh((4, 2), ("dp", "tp"))
    plan = ShardingPlanner().plan(tiny_moe, mesh, tp_size=2, ep_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_size == 2
    assert spec._ep_stack == {}
    # 自定义命名 w1/w2/w3 → expert 参数仅 {EP: Shard(0)}，无 TP 键
    for proj in ("w1", "w2", "w3"):
        p = spec.params[f"experts.{proj}"]
        assert p[EP] == Shard(0)
        assert TP not in p and p[CP] == Replicate()
    assert spec.in_dst["x_BLD"][TP] == Shard(1)   # SP-in identity


def test_stack_moe_experts(tiny_hf_native_moe):
    """堆叠 handler：stacked 值 == 原 per-expert 值，原参数移除。"""
    mesh = _meta_mesh((4, 2), ("dp", "tp"))
    mlp = tiny_hf_native_moe.model.layers[0].mlp
    orig = {
        proj: torch.stack([getattr(mlp.experts[i], proj).weight.data
                           for i in range(4)])
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2, ep_size=2)
    ep_stack = plan.modules["model.layers.0.mlp"]._ep_stack

    _stack_moe_experts(mlp, ep_stack)

    assert isinstance(mlp.experts, _StackedExperts)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        stacked = getattr(mlp.experts, proj)
        assert stacked.shape == orig[proj].shape
        torch.testing.assert_close(stacked, orig[proj])
    # 原 per-expert 参数已移除
    assert not any("experts.0" in n
                   for n, _ in mlp.named_parameters())


def test_stack_moe_experts_rejects_bias(tiny_hf_native_moe):
    """带 bias 的 expert → NotImplementedError（v1 限制）。"""
    mlp = tiny_hf_native_moe.model.layers[0].mlp
    mlp.experts[0].gate_proj.bias = nn.Parameter(torch.zeros(32))
    ep_stack = {"experts.gate_proj": [f"experts.{i}.gate_proj.weight" for i in range(4)]}
    with pytest.raises(NotImplementedError, match="bias"):
        _stack_moe_experts(mlp, ep_stack)


def test_softmax_topk_router(tiny_hf_native_moe):
    """default adapter 与玩具模型 forward 的路由语义一致。"""
    mlp = tiny_hf_native_moe.model.layers[0].mlp
    torch.manual_seed(5)
    hidden = torch.randn(2, 3, 16)
    topk_idx, topk_w = _softmax_topk_router(mlp, hidden)
    logits = mlp.gate(hidden).view(-1, 4)
    w = logits.softmax(-1)
    ref_w, ref_idx = w.topk(2, dim=-1)
    ref_w = ref_w / ref_w.sum(-1, keepdim=True)
    assert torch.equal(topk_idx, ref_idx)
    torch.testing.assert_close(topk_w, ref_w)
    assert MOE_ROUTER_ADAPTERS["default"] is _softmax_topk_router


def testresolve_swiglu_weights_two_naming_families():
    """gate/up/down_proj 与 w1/w2/w3 两套命名均可解析；缺矩阵报错。"""
    class Holder(nn.Module):  # pylint: disable=abstract-method
        pass

    h1 = Holder()
    h1.gate_proj = nn.Parameter(torch.randn(4, 8, 16))
    h1.up_proj = nn.Parameter(torch.randn(4, 8, 16))
    h1.down_proj = nn.Parameter(torch.randn(4, 16, 8))
    g, u, d = resolve_swiglu_weights(h1)
    assert g is h1.gate_proj and u is h1.up_proj and d is h1.down_proj

    h2 = Holder()
    h2.w1 = nn.Parameter(torch.randn(4, 8, 16))
    h2.w3 = nn.Parameter(torch.randn(4, 8, 16))
    h2.w2 = nn.Parameter(torch.randn(4, 16, 8))
    g, u, d = resolve_swiglu_weights(h2)
    assert g is h2.w1 and u is h2.w3 and d is h2.w2

    with pytest.raises(NotImplementedError, match="SwiGLU"):
        resolve_swiglu_weights(Holder())


def testresolve_swiglu_weights_fused_layout():
    """D-11 fused 布局：gate_up_proj + down_proj → (fused, None, down)。"""
    class Holder(nn.Module):  # pylint: disable=abstract-method
        pass

    h = Holder()
    h.gate_up_proj = nn.Parameter(torch.randn(4, 16, 8))
    h.down_proj = nn.Parameter(torch.randn(4, 8, 8))
    g, u, d = resolve_swiglu_weights(h)
    assert g is h.gate_up_proj and u is None and d is h.down_proj

    # automodel 命名（gate_and_up_projs/down_projs）同构
    h2 = Holder()
    h2.gate_and_up_projs = nn.Parameter(torch.randn(4, 16, 8))
    h2.down_projs = nn.Parameter(torch.randn(4, 8, 8))
    g, u, d = resolve_swiglu_weights(h2)
    assert g is h2.gate_and_up_projs and u is None and d is h2.down_projs


def test_local_expert_forward_uses_declared_activation():
    """EP expert computation honors the model activation instead of forcing SiLU."""
    class Experts(nn.Module):  # pylint: disable=abstract-method
        def __init__(self):
            super().__init__()
            self.local_expert_count = 1
            self.gate_up_proj = nn.Parameter(torch.randn(1, 8, 4))
            self.down_proj = nn.Parameter(torch.randn(1, 4, 4))
            self._ep_act_fn = torch.tanh

    experts = Experts()
    hidden_states = torch.randn(3, 4)
    expert_indices = torch.zeros(3, dtype=torch.long)
    output = _local_swiglu_expert_forward(experts, hidden_states, expert_indices)
    gate_states, up_states = F.linear(hidden_states, experts.gate_up_proj[0]).chunk(2, dim=-1)
    expected = F.linear(torch.tanh(gate_states) * up_states, experts.down_proj[0])
    torch.testing.assert_close(output, expected)


def testep_routed_forward_calls_experts_forward(tiny_hf_batched_moe):
    """HF-native EP compute enters experts.__call__ for nested FSDP hooks."""
    _ensure_pg()
    mlp = tiny_hf_batched_moe.model.layers[0].mlp
    hidden_states = torch.randn(2, 3, 16)
    expected_output = mlp(hidden_states)
    bind_local_expert_forward(mlp, ep_size=1)
    forward_call_count = 0

    def count_forward_call(unused_module, unused_inputs):
        del unused_module, unused_inputs
        nonlocal forward_call_count
        forward_call_count += 1

    hook = mlp.experts.register_forward_pre_hook(count_forward_call)
    output = ep_routed_forward(
        mlp,
        hidden_states,
        router_fn=MOE_ROUTER_ADAPTERS["qwen3moe"],
        ep_group=torch.distributed.group.WORLD,
    )
    hook.remove()

    assert output.shape == hidden_states.shape
    assert forward_call_count == 1
    torch.testing.assert_close(output, expected_output)


def test_topk_router_module_adapter(tiny_hf_batched_moe):
    """Qwen2/Qwen3/Mixtral adapter：直接取 TopKRouter 的 indices 和 scores。"""
    mlp = tiny_hf_batched_moe.model.layers[0].mlp
    torch.manual_seed(5)
    hidden = torch.randn(2, 3, 16)
    idx, w = _topk_router_module(mlp, hidden)
    _, ref_w, ref_idx = mlp.gate(hidden)
    assert torch.equal(idx, ref_idx)
    torch.testing.assert_close(w, ref_w)
    assert MOE_ROUTER_ADAPTERS["qwen2moe"] is _topk_router_module
    assert MOE_ROUTER_ADAPTERS["qwen3moe"] is _topk_router_module
    assert MOE_ROUTER_ADAPTERS["mixtral"] is _topk_router_module


def test_sigmoid_group_router_adapter():
    """deepseekv3/glm4moe adapter：sigmoid + correction bias + norm + scaling
    （n_group=1 跳过 group 过滤），与手算参考一致。"""
    class Gate(nn.Module):
        def __init__(self, e, h):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(e, h) * 0.02)
            self.register_buffer("e_score_correction_bias", torch.randn(e) * 0.01)

        def forward(self, x):
            return F.linear(  # pylint: disable=not-callable
                x.view(-1, x.shape[-1]).float(), self.weight.float()
            )

    class MoE(nn.Module):  # pylint: disable=abstract-method
        def __init__(self):
            super().__init__()
            self.gate = Gate(4, 16)
            self.top_k = 2
            self.n_group = 1
            self.norm_topk_prob = True
            self.routed_scaling_factor = 2.5

    torch.manual_seed(5)
    moe = MoE()
    hidden = torch.randn(2, 3, 16)
    idx, w = _sigmoid_group_router(moe, hidden)

    logits = moe.gate(hidden)
    scores = logits.sigmoid()
    choice = scores + moe.gate.e_score_correction_bias
    ref_idx = choice.topk(2, dim=-1, sorted=False)[1]
    ref_w = scores.gather(1, ref_idx)
    ref_w = ref_w / (ref_w.sum(-1, keepdim=True) + 1e-20) * 2.5
    assert torch.equal(idx, ref_idx)
    torch.testing.assert_close(w, ref_w.to(w.dtype))


# ==========================================================================
# 来源: test_s6_ep_extend.py
# S6.1: D-10 TP-extend-EP（05 §6.4.8）单进程用例。
# ==========================================================================

def test_planner_ep_extend_contract(tiny_hf_native_moe):
    """mesh (dp=4, tp=2)，ep=4 → 扩展 EP 组 {0,1,2,3}/{4,5,6,7}（用户示例）。"""
    mesh = _meta_mesh((4, 2), ("dp", "tp"))
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2, ep_size=4)
    spec = plan.modules["model.layers.0.mlp"]

    assert spec._ep_size == 4           # ep_size 即扩展 EP 组大小
    assert spec._ep_stack               # 堆叠元数据不变

    # expert 参数：仅 {EP: Shard(0)}（expert 维切分），无 TP 键、无第二轴
    for proj in ("gate_proj", "up_proj", "down_proj"):
        p = spec.params[f"experts.{proj}"]
        assert p[EP] == Shard(0)
        assert TP not in p and p[CP] == Replicate()
        assert len(p) == 2              # 只有 CP(Replicate) + EP 两个键

    # router 全复制（本地 chunk 计算）
    assert spec.params["gate.weight"][TP] == Replicate()

    # 边界契约 identity（SP-in）：in_dst/out_src/out_dst 均 TP Shard(1)
    assert spec.in_dst["x_BLD"][TP] == Shard(1)
    assert spec.out_src["output"][TP] == Shard(1)
    assert spec.out_dst["output"][TP] == Shard(1)
    # in_src 与上游契约不变（链式校验通过）
    assert spec.in_src["x_BLD"][TP] == Shard(1)


def test_planner_ep1_no_extend(tiny_hf_native_moe, make_mesh):
    """ep=1 → 无 TP-extend-EP，per-expert 条目保留（TP-only 语义正确）。"""
    mesh = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_size == 0
    assert spec._ep_stack == {}
    assert "experts.0.gate_proj.weight" in spec.params
    assert spec.params["experts.0.gate_proj.weight"][TP] == Shard(0)


def test_planner_ep_extend_invalid(tiny_hf_native_moe):
    """ep_size 超过 dense 区域 / 不整除 dense / num_experts 不整除 → ValueError。"""
    # mesh (1,2) D=2：ep=4 > D → 报错
    mesh = _meta_mesh((1, 2), ("dp", "tp"))
    with pytest.raises(ValueError, match="dense"):
        ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2, ep_size=4)
    # mesh (4,2) D=8：ep=3 不整除 D → 报错
    mesh8 = _meta_mesh((4, 2), ("dp", "tp"))
    with pytest.raises(ValueError, match="dense"):
        ShardingPlanner().plan(tiny_hf_native_moe, mesh8, tp_size=2, ep_size=3)
    # mesh (4,2) D=8：ep=8 合法但 num_experts=4 不整除 ep=8 → 报错
    with pytest.raises(ValueError, match="num_experts"):
        ShardingPlanner().plan(tiny_hf_native_moe, mesh8, tp_size=2, ep_size=8)


def test_planner_batched_contract(tiny_hf_batched_moe):
    """D-11 batched 布局（experts.gate_up_proj [E,2I,H]）：无需堆叠，直接标
    {EP: Shard(0)}；arch=qwen3moe → TopKRouter 模块 adapter。"""
    mesh = _meta_mesh((4, 2), ("dp", "tp"))
    plan = ShardingPlanner().plan(tiny_hf_batched_moe, mesh, tp_size=2, ep_size=4)
    spec = plan.modules["model.layers.0.mlp"]

    assert spec._ep_size == 4
    assert spec._ep_stack == {}          # batched 天生 stacked，无需堆叠

    # expert 参数：仅 {EP: Shard(0)}（expert 维切分），无 TP 键、无第二轴
    for proj in ("gate_up_proj", "down_proj"):
        p = spec.params[f"experts.{proj}"]
        assert p[EP] == Shard(0)
        assert TP not in p and p[CP] == Replicate()
        assert len(p) == 2

    # router（TopKRouter.weight）全复制；边界 identity
    assert spec.params["gate.weight"][TP] == Replicate()
    assert spec.in_dst["x_BLD"][TP] == Shard(1)
    assert spec.out_src["output"][TP] == Shard(1)
    assert spec.out_dst["output"][TP] == Shard(1)


def test_planner_batched_ep1_no_mark(tiny_hf_batched_moe, make_mesh):
    """batched 布局 ep=1 → 不标记（_ep_size == 0）。

    融合权重裸 TP Shard(1) 的写法（D-08 旧语义）已被
    _finalize_fused_expert_tp_guard fail-fast——连续块切分与 forward 内
    chunk 不兼容；合法配置需 override 为 TP Replicate（守门解法②，
    须同步把 out_src 的 TP 从模板推导的 Partial 改为 Replicate）。
    """
    rep = lambda: {TP: Replicate(), CP: Replicate()}  # noqa: E731
    overrides = {"*.mlp": ModuleShardingSpec(
        params={
            "gate.weight": rep(),
            "experts.gate_up_proj": rep(),
            "experts.down_proj": rep(),
        },
        out_src={"output": rep()},
    )}
    mesh = make_mesh((1,), ("tp",))
    plan = ShardingPlanner(plan_overrides=overrides).plan(
        tiny_hf_batched_moe, mesh, tp_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_size == 0
    assert spec.params["experts.gate_up_proj"][TP] == Replicate()


def test_expert_mesh_layout_mapping():
    """派生 expert mesh：dense 区域 flatten → (edp, ep)，EP 组 = flatten 连续
    ep_size 个 rank（先跨完 TP 组再向相邻 dp rank 扩展）。"""
    mesh = _meta_mesh((4, 2), ("dp", "tp"))   # rank = d*2 + t

    # ep=4（用户示例）：EP 组 {0,1,2,3} / {4,5,6,7}——跨 2 个 TP 组 × 2 个 dp
    shape, names, rank_list = _expert_mesh_layout(mesh, ("dp", "tp"), 4)
    assert shape == (2, 4)
    assert names == ("edp", "ep")
    assert rank_list == (0, 1, 2, 3, 4, 5, 6, 7)

    # ep=2：EP 组 {0,1}/{2,3}/{4,5}/{6,7}——即 TP 组
    shape, names, _ = _expert_mesh_layout(mesh, ("dp", "tp"), 2)
    assert shape == (4, 2)
    assert names == ("edp", "ep")

    # 不整除报错
    with pytest.raises(ValueError, match="must divide"):
        _expert_mesh_layout(mesh, ("dp", "tp"), 3)
