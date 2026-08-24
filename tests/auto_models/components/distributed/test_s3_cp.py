# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_s3_cp.py: 核心套件合并文件。

来源: test_s3_inner_attn_detect.py, test_s3_shard_batch.py, test_s3_shard_seq_lens.py
"""

import pytest
import torch
import torch.nn as nn
from hyper_parallel.auto_models.components.distributed.cp_utils import (
    _shard_seq_lens_for_cp,
    shard_batch_for_cp,
)
from hyper_parallel.auto_models.components.distributed.cp_wrappers import (
    INNER_WRAPPER_REGISTRY,
    is_flex_attention,
    is_hf_style_attention,
    is_sdpa_attention,
    sdpa_qkv_cp_wrapper,
)
from hyper_parallel.auto_models.components.distributed.sharding_applier import (
    _resolve_inner_target,
    _resolve_inner_wrapper,
    _wrap_inner_attention,
)
from hyper_parallel.auto_models.components.distributed.injection import (
    inner_wrapper,
)
from hyper_parallel.auto_models.components.distributed.sharding_config import (
    ModuleShardingSpec,
    TP,
)
from hyper_parallel.auto_models.trainer.config import Target
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import (
    Replicate,
    Shard,
)


# ==========================================================================
# 来源: test_s3_inner_attn_detect.py
# S3.2: inner-wrap 解析链 + HF/NeMo 风格判定 + 注册表（单进程 mock）。
# ==========================================================================

class _Cfg:
    _attn_implementation = "sdpa"


class HFSdpaAttention(nn.Module):
    """HF 风格：持有 q/k/v_proj，forward(hidden_states)。"""

    def __init__(self):
        super().__init__()
        self.config = _Cfg()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)

    def forward(self, hidden_states):
        return hidden_states


class NeMoAttention(nn.Module):
    """NeMo 风格：inner_attention 子模块 forward(q,k,v)。"""

    class Inner(nn.Module):
        def forward(self, q, k, v):
            return q

    def __init__(self):
        super().__init__()
        self.inner_attention = self.Inner()

    def forward(self, hidden_states):
        return hidden_states


class FlexHFAattention(HFSdpaAttention):
    class _FlexCfg:
        _attn_implementation = "flex_attention"

    def __init__(self):
        super().__init__()
        self.config = self._FlexCfg()


class TestResolveInnerTarget:
    """inner_target 显式解析（attention 域自动定位启发式已删除，2026-08-10）。"""

    def test_undeclared_raises(self):
        """未声明 inner_target → fail-fast（不再自动定位/返回 None）。"""
        with pytest.raises(ValueError, match="inner_target"):
            _resolve_inner_target(NeMoAttention(), spec=ModuleShardingSpec())

    def test_user_inner_target_attr(self):
        """spec.inner_target 显式指定属性名 → 命中该子模块。"""
        m = NeMoAttention()
        spec = ModuleShardingSpec(inner_target="inner_attention")
        assert _resolve_inner_target(m, spec=spec) is m.inner_attention

    def test_user_inner_target_self(self):
        """spec.inner_target='self' → 模块本身即 target。"""
        class Bare(nn.Module):
            def forward(self, x):
                return x
        m = Bare()
        spec = ModuleShardingSpec(inner_target="self")
        assert _resolve_inner_target(m, spec=spec) is m

    def test_user_inner_target_missing_raises(self):
        """spec.inner_target 拼写错误 → fail-fast（不能静默降级）。"""
        class Bare(nn.Module):
            def forward(self, x):
                return x
        spec = ModuleShardingSpec(inner_target="core_atn")  # 拼写错误
        with pytest.raises(ValueError, match="inner_target"):
            _resolve_inner_target(Bare(), spec=spec)


class _FakeCpMesh:
    """单进程 fake cp mesh（size=1，仅用于测试注入路径，不触发通信）。"""

    def size(self):
        return 1


class TestResolveInnerWrapper:
    def test_no_declaration_returns_none(self):
        """无 inner_target/inner_wrapper → None（派生门控）。"""
        resolved = _resolve_inner_wrapper(
            NeMoAttention(), ModuleShardingSpec(), _FakeCpMesh(), None, ())
        assert resolved is None

    def test_needs_cp_attn_alone_returns_none(self):
        """改造后 _needs_cp_attn 只是模板元数据（preflight/内省用），
        不再触发任何注入——无显式 inner_wrapper → None。"""
        spec = ModuleShardingSpec(_needs_cp_attn=True)
        assert _resolve_inner_wrapper(
            NeMoAttention(), spec, _FakeCpMesh(), None, ()) is None
        assert _resolve_inner_wrapper(
            HFSdpaAttention(), spec, _FakeCpMesh(), None, ()) is None

    def test_inner_target_without_wrapper_raises(self):
        """inner_target 只是定位——单独声明（无 inner_wrapper）→ fail-fast
        （改造后不再启发式选择方案）。"""
        spec = ModuleShardingSpec(inner_target="inner_attention")
        with pytest.raises(ValueError, match="inner_wrapper"):
            _resolve_inner_wrapper(
                NeMoAttention(), spec, _FakeCpMesh(), None, ())

    def test_str_registry_lookup(self):
        """inner_wrapper='sdpa_qkv'（str）→ 显式选注册表方案。"""
        spec = ModuleShardingSpec(inner_target="self",
                                  inner_wrapper="sdpa_qkv", region_dispatch=False)
        name, _, _ = _resolve_inner_wrapper(
            HFSdpaAttention(), spec, _FakeCpMesh(), None, ())
        assert name == "sdpa_qkv"   # HF 模块也可显式选 qkv 路（用户负责）

    def test_str_unknown_name_raises(self):
        """inner_wrapper=str 未注册 → fail-fast 并列出可用名。"""
        spec = ModuleShardingSpec(inner_target="self",
                                  inner_wrapper="sdpa_hff", region_dispatch=False)  # 拼写错误
        with pytest.raises(ValueError, match="INNER_WRAPPER_REGISTRY"):
            _resolve_inner_wrapper(
                NeMoAttention(), spec, _FakeCpMesh(), None, ())

    def test_wrapper_without_inner_target_raises(self):
        """声明 inner_wrapper 但未声明 inner_target → fail-fast（自动定位
        启发式已删除，两字段必须成对显式声明）。"""
        class Bare(nn.Module):
            def forward(self, x):
                return x
        spec = ModuleShardingSpec(inner_wrapper="sdpa_qkv", region_dispatch=False)
        with pytest.raises(ValueError, match="inner_target"):
            _resolve_inner_wrapper(Bare(), spec, _FakeCpMesh(), None, ())

    def test_user_registry_extension(self):
        """用户注册命名方案后可按名引用。"""
        calls = []

        @inner_wrapper
        def my_fn(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            calls.append(target_module)

        INNER_WRAPPER_REGISTRY["test_custom"] = my_fn
        try:
            spec = ModuleShardingSpec(inner_target="inner_attention",
                                      inner_wrapper="test_custom", region_dispatch=False)
            name, target, apply_fn = _resolve_inner_wrapper(
                NeMoAttention(), spec, _FakeCpMesh(), None, ())
            assert name == "test_custom"
            apply_fn()
            assert calls == [target]
        finally:
            INNER_WRAPPER_REGISTRY.pop("test_custom")

    def test_callable_custom(self):
        """inner_wrapper=callable → ('custom', target, ...)。"""
        m = NeMoAttention()

        @inner_wrapper
        def my_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            pass

        spec = ModuleShardingSpec(inner_target="inner_attention",
                                  inner_wrapper=my_wrapper, region_dispatch=False)
        name, target, _ = _resolve_inner_wrapper(
            m, spec, _FakeCpMesh(), None, ())
        assert name == "custom"
        assert target is m.inner_attention

    def test_wrong_type_raises(self):
        """inner_wrapper 既非 str/callable/Target → TypeError。"""
        spec = ModuleShardingSpec(inner_target="self",
                                  inner_wrapper=123, region_dispatch=False)
        with pytest.raises(TypeError, match="inner_wrapper"):
            _resolve_inner_wrapper(
                NeMoAttention(), spec, _FakeCpMesh(), None, ())


class TestTargetInnerWrapper:
    """Target 延迟引用形态（YAML sharding.injections 的运行时载体）。"""

    def test_target_builtin_inplace(self):
        """Target 指向仓内内置函数（registry 风格签名）：build 后原地替换，
        名称为 target_path。"""
        m = NeMoAttention()
        spec = ModuleShardingSpec(inner_target="inner_attention",
                                  inner_wrapper=Target(
            sdpa_qkv_cp_wrapper,
            target_path="hyper_parallel.auto_models.components.distributed."
                        "cp_wrappers.sdpa_qkv_cp_wrapper"), region_dispatch=False)
        cp_mesh = _FakeCpMesh()
        name, target, apply_fn = _resolve_inner_wrapper(
            m, spec, cp_mesh, None, ())
        assert name.endswith("sdpa_qkv_cp_wrapper")
        assert target is m.inner_attention
        apply_fn()
        # forward 已被 (q,k,v) wrapper 替换
        q = torch.randn(1, 1, 2, 4)
        assert m.inner_attention(q, q, q) is not None

    def test_target_factory_returning_callable(self):
        """Target 工厂返回 callable → 按自定义 wrapper 契约 (target, cp_mesh) 应用。"""
        received = []

        @inner_wrapper
        def my_factory(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            @inner_wrapper
            def my_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
                received.append((target_module, cp_mesh))
            return my_wrapper

        m = NeMoAttention()
        spec = ModuleShardingSpec(inner_target="inner_attention",
                                  inner_wrapper=Target(
            my_factory, target_path="tests.my_factory"), region_dispatch=False)
        cp_mesh = _FakeCpMesh()
        name, target, apply_fn = _resolve_inner_wrapper(
            m, spec, cp_mesh, None, ())
        apply_fn()
        assert received == [(target, cp_mesh)]

    def test_target_bad_return_raises(self):
        """Target 返回非 None 非 callable → TypeError。"""
        @inner_wrapper
        def bad_factory(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            return 42

        m = NeMoAttention()
        spec = ModuleShardingSpec(inner_target="inner_attention",
                                  inner_wrapper=Target(
            bad_factory, target_path="tests.bad_factory"), region_dispatch=False)
        _, _, apply_fn = _resolve_inner_wrapper(
            m, spec, _FakeCpMesh(), None, ())
        with pytest.raises(TypeError, match="inner_wrapper"):
            apply_fn()

    def test_target_undecorated_factory_raises(self):
        """注入纪律：Target 指向未装饰的函数 → fail-fast 提示 @inner_wrapper。"""
        m = NeMoAttention()
        spec = ModuleShardingSpec(inner_target="self",
                                  inner_wrapper=Target(
            lambda: None, target_path="tests.undecorated"), region_dispatch=False)
        with pytest.raises(TypeError, match="@inner_wrapper"):
            _resolve_inner_wrapper(m, spec, _FakeCpMesh(), None, ())

    def test_context_filled_by_name(self):
        """声明的上下文按名填充：mesh 家族必选参数全部收到框架传入的值
        （无 cp/ep 轴时 cp_mesh/ep_mesh 为 None），用户只管使用。"""
        seen = {}

        @inner_wrapper
        def my_factory(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            seen.update(target_module=target_module, cp_mesh=cp_mesh,
                        ep_mesh=ep_mesh)
            return None   # 原地替换风格（本测试不真的替换）

        m = NeMoAttention()
        spec = ModuleShardingSpec(inner_target="inner_attention",
                                  inner_wrapper=Target(
            my_factory, target_path="tests.my_factory"), region_dispatch=False)
        cp_mesh = _FakeCpMesh()
        _, target, apply_fn = _resolve_inner_wrapper(
            m, spec, cp_mesh, None, ())
        apply_fn()
        assert seen["target_module"] is target
        assert seen["cp_mesh"] is cp_mesh
        assert seen["ep_mesh"] is None

    def test_target_typo_config_key_raises(self):
        """配置了 wrapper 未声明的键（cp_mesg 拼写错误）→ fail-fast 并列出
        合法形参——否则该键被 **kwargs 静默吞掉、不会生效。"""
        m = NeMoAttention()
        spec = ModuleShardingSpec(inner_target="self",
                                  inner_wrapper=Target(
            sdpa_qkv_cp_wrapper,
            target_path="hyper_parallel.auto_models.components.distributed."
                        "cp_wrappers.sdpa_qkv_cp_wrapper",
            cp_mesg="oops"), region_dispatch=False)                      # 拼写错误：应为 cp_mesh
        with pytest.raises(ValueError, match="cp_mesh"):
            _resolve_inner_wrapper(m, spec, _FakeCpMesh(), None, ())


class TestNoCpGeneralization:
    """inner_wrapper 泛化（不再 CP 门控）：声明即应用；cp_mesh=None 时自定义
    wrapper 正常工作，四个内置 CP 方案 fail-fast。"""

    def test_custom_callable_fires_without_cp(self):
        """无 cp 轴（cp_mesh=None）：自定义 callable 照常解析与应用。"""
        m = NeMoAttention()
        received = []

        @inner_wrapper
        def my_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            received.append((target_module, cp_mesh))

        spec = ModuleShardingSpec(inner_target="inner_attention",
                                  inner_wrapper=my_wrapper,
                                  inner_out_src="first_input", region_dispatch=False)
        name = _wrap_inner_attention(m, None, spec=spec)
        assert name == "custom"
        assert received == [(m.inner_attention, None)]
        assert spec._resolved_inner_target == "inner_attention"

    def test_user_registered_name_fires_without_cp(self):
        """用户注册的命名方案不是内置 CP wrapper → 无 cp 轴也可用。"""
        calls = []

        @inner_wrapper
        def my_fn(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            calls.append(cp_mesh)

        INNER_WRAPPER_REGISTRY["test_no_cp"] = my_fn
        try:
            spec = ModuleShardingSpec(inner_target="inner_attention",
                                      inner_wrapper="test_no_cp",
                                      inner_out_src="first_input", region_dispatch=False)
            name = _wrap_inner_attention(NeMoAttention(), None, spec=spec)
            assert name == "test_no_cp"
            assert calls == [None]
        finally:
            INNER_WRAPPER_REGISTRY.pop("test_no_cp")

    def test_builtin_str_without_cp_raises(self):
        """内置 CP 方案（str）+ 无 cp 轴 → wrapper 自检 fail-fast（cp_mesh
        为 None），指引 local_compute_fn。"""
        spec = ModuleShardingSpec(inner_target="self",
                                  inner_wrapper="sdpa_hf", region_dispatch=False)
        with pytest.raises(ValueError, match="local_compute_fn"):
            _wrap_inner_attention(HFSdpaAttention(), None, spec=spec)

    def test_builtin_callable_without_cp_raises(self):
        """内置 CP 方案（callable 直传）+ 无 cp 轴 → fail-fast。"""
        spec = ModuleShardingSpec(inner_target="inner_attention",
                                  inner_wrapper=sdpa_qkv_cp_wrapper, region_dispatch=False)
        with pytest.raises(ValueError, match="active cp axis"):
            _wrap_inner_attention(NeMoAttention(), None, spec=spec)

    def test_builtin_target_without_cp_raises(self):
        """内置 CP 方案（Target 形态）+ 无 cp 轴 → fail-fast。"""
        spec = ModuleShardingSpec(inner_target="inner_attention",
                                  inner_wrapper=Target(
            sdpa_qkv_cp_wrapper,
            target_path="hyper_parallel.auto_models.components.distributed."
                        "cp_wrappers.sdpa_qkv_cp_wrapper"), region_dispatch=False)
        with pytest.raises(ValueError, match="active cp axis"):
            _wrap_inner_attention(NeMoAttention(), None, spec=spec)


class TestCpWrapApply:
    def test_apply_writes_resolved_name(self):
        """应用后 spec._resolved_inner_wrapper 回写（plan 内省）。"""
        spec = ModuleShardingSpec(inner_target="inner_attention",
                                  inner_wrapper="sdpa_qkv",
                                  inner_out_src="first_input", region_dispatch=False)
        _wrap_inner_attention(NeMoAttention(), _FakeCpMesh(), spec=spec)
        assert spec._resolved_inner_wrapper == "sdpa_qkv"

    def test_apply_without_declaration_returns_none(self):
        """无声明 → 返回 None 且不注入（即便 _needs_cp_attn 元数据在）。"""
        spec = ModuleShardingSpec(_needs_cp_attn=True)
        m = NeMoAttention()
        orig_fwd = m.inner_attention.forward
        assert _wrap_inner_attention(
            m, _FakeCpMesh(), spec=spec) is None
        assert m.inner_attention.forward == orig_fwd

    def test_custom_callable_takeover(self):
        """自定义 callable 整体接管：以 (target, cp_mesh) 调用并替换 forward。"""
        class Bare(nn.Module):
            def forward(self, x):
                return x

        calls = []

        @inner_wrapper
        def my_cp_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            calls.append((target_module, cp_mesh))
            orig = target_module.forward

            def wrapped(x, *args, **kwargs):
                return orig(x)

            target_module.forward = wrapped

        m = Bare()
        spec = ModuleShardingSpec(inner_target="self",
                                  inner_wrapper=my_cp_wrapper, region_dispatch=False)
        mesh = _FakeCpMesh()
        _wrap_inner_attention(m, mesh, spec=spec)
        assert calls and calls[0][1] is mesh
        assert spec._resolved_inner_wrapper == "custom"
        assert m.forward(torch.tensor(1), None, None) is not None

    def test_custom_callable_with_inner_target(self):
        """inner_target + inner_wrapper 组合：target 为用户指定的子模块。"""
        m = NeMoAttention()
        received = []

        @inner_wrapper
        def my_cp_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            received.append(target_module)

        spec = ModuleShardingSpec(inner_target="inner_attention",
                                  inner_wrapper=my_cp_wrapper,
                                  inner_out_src="first_input", region_dispatch=False)
        _wrap_inner_attention(m, _FakeCpMesh(), spec=spec)
        assert received == [m.inner_attention]

    def test_resolved_inner_target_written_back(self):
        """定位结果可见化：spec._resolved_inner_target 回写属性名/"self"。
        （inner_target 现为必填显式声明，回写供 plan 内省核对）。"""
        m = NeMoAttention()
        spec = ModuleShardingSpec(inner_target="inner_attention",
                                  inner_wrapper="sdpa_qkv",
                                  inner_out_src="first_input", region_dispatch=False)
        _wrap_inner_attention(m, _FakeCpMesh(), spec=spec)
        assert spec._resolved_inner_target == "inner_attention"

        m2 = HFSdpaAttention()
        spec2 = ModuleShardingSpec(inner_target="self",
                                   inner_wrapper="sdpa_hf", region_dispatch=False)
        _wrap_inner_attention(m2, _FakeCpMesh(), spec=spec2)
        assert spec2._resolved_inner_target == "self"

    def test_str_pinned_wrapper_applied(self):
        """inner_wrapper='sdpa_qkv' 显式固定：应用的是注册表里的函数。"""
        m = NeMoAttention()
        spec = ModuleShardingSpec(inner_target="inner_attention",
                                  inner_wrapper="sdpa_qkv",
                                  inner_out_src="first_input", region_dispatch=False)
        _wrap_inner_attention(m, _FakeCpMesh(), spec=spec)
        assert spec._resolved_inner_wrapper == "sdpa_qkv"
        # forward 已被 (q,k,v) wrapper 替换
        q = torch.randn(1, 1, 2, 4)
        assert m.inner_attention(q, q, q) is not None


class TestInjectionDiscipline:
    """注入纪律（injection.py）：装饰器强制 + 替换 forward 入参兼容校验。"""

    def test_undecorated_callable_raises(self):
        """未装饰的 callable → fail-fast 提示 @inner_wrapper。"""
        spec = ModuleShardingSpec(inner_target="self",
                                  inner_wrapper=lambda t, cm: None, region_dispatch=False)
        with pytest.raises(TypeError, match="@inner_wrapper"):
            _resolve_inner_wrapper(
                NeMoAttention(), spec, _FakeCpMesh(), None, ())

    def test_wrong_kind_decorator_raises(self):
        """装饰器种类不符（@local_compute 用在 inner_wrapper 上）→ fail-fast。"""
        from hyper_parallel.auto_models.components.distributed.injection import local_compute

        @local_compute
        def my_compute(mesh, tp_mesh, cp_mesh, ep_mesh):
            def compute_fn(module, x):
                return x
            return compute_fn

        spec = ModuleShardingSpec(inner_target="self",
                                  inner_wrapper=my_compute, region_dispatch=False)
        with pytest.raises(TypeError, match="wrong decorator kind"):
            _resolve_inner_wrapper(
                NeMoAttention(), spec, _FakeCpMesh(), None, ())

    def test_decorator_requires_mesh_family(self):
        """装饰器强制必选上下文：缺 mesh/tp_mesh/cp_mesh/ep_mesh 任一个
        都在 import 期 fail-fast。"""
        with pytest.raises(TypeError, match="missing required context parameters"):
            @inner_wrapper
            def bad(target_module, mesh):   # 缺 tp_mesh/cp_mesh/ep_mesh
                pass

    def test_decorator_rejects_context_default(self):
        """上下文参数不得有默认值（框架必然填充，默认值无意义）。"""
        with pytest.raises(TypeError, match="must not have a default"):
            @inner_wrapper
            def bad(target_module, mesh, tp_mesh, cp_mesh, ep_mesh=None):
                pass

    def test_decorator_rejects_var_kwargs(self):
        """装饰器在 import 期拒绝 *args/**kwargs（拼写配置键会被静默吞掉）。"""
        with pytest.raises(TypeError, match="\\*args/\\*\\*kwargs"):
            @inner_wrapper
            def bad(target_module, mesh, tp_mesh, cp_mesh, ep_mesh, **ctx):
                pass

    def test_incompatible_replacement_forward_raises(self):
        """原则 1：替换后的 forward 接不住原 forward 的必填入参 → fail-fast。"""
        m = NeMoAttention()   # inner forward(q, k, v)

        @inner_wrapper
        def bad_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            def wrapped(q):     # 丢掉 k/v —— 与原 forward 不兼容
                return q
            target_module.forward = wrapped

        spec = ModuleShardingSpec(inner_target="inner_attention",
                                  inner_wrapper=bad_wrapper,
                                  inner_out_src="first_input", region_dispatch=False)
        with pytest.raises(TypeError, match="incompatible with the original forward"):
            _wrap_inner_attention(m, _FakeCpMesh(), spec=spec)

    def test_undecorated_registry_entry_raises(self):
        """注册表里的未装饰函数 → fail-fast（内置四路已装饰）。"""
        INNER_WRAPPER_REGISTRY["test_undecorated"] = lambda t, cm: None
        try:
            spec = ModuleShardingSpec(inner_target="inner_attention", inner_wrapper="test_undecorated", region_dispatch=False)
            with pytest.raises(TypeError, match="@inner_wrapper"):
                _resolve_inner_wrapper(
                    NeMoAttention(), spec, _FakeCpMesh(), None, ())
        finally:
            INNER_WRAPPER_REGISTRY.pop("test_undecorated")


class TestDualModeAdapter:
    """统一双模适配器：用户 wrapper 只面向 local 张量；转换与重包托管。"""

    def _dtensor(self, mesh, local):
        return DTensor.from_local(local, mesh, (Replicate(),))

    def test_inner_without_declaration_fails(self):
        """情形 B（inner 子模块）未声明 inner_out_src → 安装时 fail-fast。"""
        m = NeMoAttention()

        @inner_wrapper
        def w(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            orig = target_module.forward

            def fwd(q, k, v):
                return orig(q, k, v)
            target_module.forward = fwd

        spec = ModuleShardingSpec(inner_target="inner_attention", inner_wrapper=w, region_dispatch=False)
        with pytest.raises(ValueError, match="inner_out_src"):
            _wrap_inner_attention(m, _FakeCpMesh(), spec=spec)

    def test_bad_sentinel_fails(self):
        """inner_out_src 的非法字符串哨兵 → fail-fast。"""
        m = NeMoAttention()

        @inner_wrapper
        def w(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            orig = target_module.forward

            def fwd(q, k, v):
                return orig(q, k, v)
            target_module.forward = fwd

        spec = ModuleShardingSpec(inner_target="inner_attention", inner_wrapper=w, inner_out_src="bogus", region_dispatch=False)
        with pytest.raises(ValueError, match="first_input"):
            _wrap_inner_attention(m, _FakeCpMesh(), spec=spec)

    def test_first_input_rewrap_validate(self, make_mesh):
        """validate：DTensor 输入被 to_local（用户只见 local），输出按
        首个输入的 placements 重包回 DTensor。"""
        mesh = make_mesh((1,), ("tp",))
        m = NeMoAttention()
        seen = {}

        @inner_wrapper
        def w(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            orig = target_module.forward

            def fwd(q, k, v):
                seen["q_is_dtensor"] = isinstance(q, DTensor)
                return orig(q, k, v)
            target_module.forward = fwd

        spec = ModuleShardingSpec(inner_target="inner_attention", inner_wrapper=w, inner_out_src="first_input", region_dispatch=False)
        _wrap_inner_attention(m, None, spec=spec, mesh=mesh,
                              mesh_dim_names=("tp",))
        q = self._dtensor(mesh, torch.randn(1, 2, 4, 8))
        out = m.inner_attention(q, q, q)
        assert seen["q_is_dtensor"] is False      # 用户只见 local 张量
        assert isinstance(out, DTensor)           # 输出已重包
        assert tuple(out.placements) == (Replicate(),)

    def test_production_passthrough(self):
        """production（local 输入）：直通，零转换。"""
        m = NeMoAttention()
        seen = {}

        @inner_wrapper
        def w(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            orig = target_module.forward

            def fwd(q, k, v):
                seen["q_is_dtensor"] = isinstance(q, DTensor)
                return orig(q, k, v)
            target_module.forward = fwd

        spec = ModuleShardingSpec(inner_target="inner_attention", inner_wrapper=w, inner_out_src="first_input", region_dispatch=False)
        _wrap_inner_attention(m, _FakeCpMesh(), spec=spec)
        x = torch.randn(1, 2, 4, 8)
        out = m.inner_attention(x, x, x)
        assert seen["q_is_dtensor"] is False
        assert not isinstance(out, DTensor)
        torch.testing.assert_close(out, x)

    def test_explicit_placement_rewrap(self, make_mesh):
        """情形 B 显式 placement 声明：按声明重包。"""
        mesh = make_mesh((1,), ("tp",))
        m = NeMoAttention()

        @inner_wrapper
        def w(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            orig = target_module.forward

            def fwd(q, k, v):
                return orig(q, k, v)
            target_module.forward = fwd

        # 注意：inner_out_src 的 spec 形态是 Placement 对象（YAML 字符串在
        # 脱糖时解析）——直接构造：
        from hyper_parallel.core.dtensor.placement_types import Replicate as R
        spec = ModuleShardingSpec(inner_target="inner_attention", inner_wrapper=w,
                                  inner_out_src={TP: R()}, region_dispatch=False)
        _wrap_inner_attention(m, None, spec=spec, mesh=mesh,
                              mesh_dim_names=("tp",))
        q = self._dtensor(mesh, torch.randn(1, 2, 4, 8))
        out = m.inner_attention(q, q, q)
        assert isinstance(out, DTensor)
        assert tuple(out.placements) == (R(),)

    def test_self_target_uses_boundary_out_src(self, make_mesh):
        """情形 A（target=self）：按边界 spec.out_src 声明重包。"""
        mesh = make_mesh((1,), ("tp",))
        m = HFSdpaAttention()

        @inner_wrapper
        def w(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            orig = target_module.forward

            def fwd(hidden_states, *args, **kwargs):
                return orig(hidden_states)
            target_module.forward = fwd

        spec = ModuleShardingSpec(inner_target="self", inner_wrapper=w,
                                  out_src={"output": {TP: Replicate()}}, region_dispatch=False)
        _wrap_inner_attention(m, None, spec=spec, mesh=mesh,
                              mesh_dim_names=("tp",))
        h = self._dtensor(mesh, torch.randn(1, 2, 8))
        out = m(h)
        assert isinstance(out, DTensor)
        assert tuple(out.placements) == (Replicate(),)

    def test_multi_output_declared(self, make_mesh):
        """情形 B 多输出：{name: placement} 按声明键序逐位置重包。"""
        mesh = make_mesh((1,), ("tp",))

        class Pair(nn.Module):
            def forward(self, q, k, v):
                return q, k + v

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner_attention = Pair()

        m = Outer()

        @inner_wrapper
        def w(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            pass   # 不替换（用原 forward 的多输出）

        spec = ModuleShardingSpec(
            inner_target="inner_attention",
            inner_wrapper=w,
            inner_out_src={"a": {TP: Replicate()}, "b": {TP: Replicate()}}, region_dispatch=False)
        # 原 forward 未被替换（w 未赋值 forward）→ 适配器不安装；
        # 手动替换以走多输出路径：
        orig = m.inner_attention.forward

        @inner_wrapper
        def w2(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            def fwd(q, k, v):
                a, b = orig(q, k, v)
                return a, b
            target_module.forward = fwd

        spec = ModuleShardingSpec(
            inner_target="inner_attention",
            inner_wrapper=w2,
            inner_out_src={"a": {TP: Replicate()}, "b": {TP: Replicate()}}, region_dispatch=False)
        _wrap_inner_attention(m, None, spec=spec, mesh=mesh,
                              mesh_dim_names=("tp",))
        q = self._dtensor(mesh, torch.randn(1, 2, 4, 8))
        a, b = m.inner_attention(q, q, q)
        assert isinstance(a, DTensor) and isinstance(b, DTensor)
        assert tuple(b.placements) == (Replicate(),)

    def test_first_input_tuple_output_fails(self, make_mesh):
        """first_input 哨兵 + 多输出 → 运行期 fail-fast 指引显式声明。"""
        mesh = make_mesh((1,), ("tp",))
        m = NeMoAttention()

        @inner_wrapper
        def w(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            orig = target_module.forward

            def fwd(q, k, v):
                return orig(q, k, v), q   # 多输出
            target_module.forward = fwd

        spec = ModuleShardingSpec(inner_target="inner_attention", inner_wrapper=w, inner_out_src="first_input", region_dispatch=False)
        _wrap_inner_attention(m, None, spec=spec, mesh=mesh,
                              mesh_dim_names=("tp",))
        q = self._dtensor(mesh, torch.randn(1, 2, 4, 8))
        with pytest.raises(RuntimeError, match="only supports a single output"):
            m.inner_attention(q, q, q)


class TestAdapterDispatchThrough:
    """region_dispatch=True（inner_wrapper 通道）：validate 穿透——DTensor
    直入用户 forward，传播结果与声明的重包规则做真校验。"""

    def _dtensor(self, mesh, local):
        return DTensor.from_local(local, mesh, (Replicate(),))

    def test_dispatch_through_first_input(self, make_mesh):
        """validate：用户 forward 见 DTensor（穿透），传播结果 == 首个入参
        布局（first_input 声明为真校验基准）→ 通过。"""
        mesh = make_mesh((1,), ("tp",))
        m = NeMoAttention()
        seen = {}

        @inner_wrapper
        def w(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            orig = target_module.forward

            def fwd(q, k, v):
                seen["q_is_dtensor"] = isinstance(q, DTensor)
                return orig(q, k, v)
            target_module.forward = fwd

        spec = ModuleShardingSpec(inner_target="inner_attention", inner_wrapper=w, inner_out_src="first_input",
                                  region_dispatch=True)
        _wrap_inner_attention(m, None, spec=spec, mesh=mesh,
                              mesh_dim_names=("tp",))
        q = self._dtensor(mesh, torch.randn(1, 2, 4, 8))
        out = m.inner_attention(q, q, q)
        assert seen["q_is_dtensor"] is True     # 穿透：用户见 DTensor
        assert isinstance(out, DTensor)
        assert tuple(out.placements) == (Replicate(),)

    def test_dispatch_through_mismatch_fails(self, make_mesh):
        """真校验：显式声明 {tp: Shard(0)}，传播结果是 Replicate →
        PlacementMismatchError（黑盒模式抓不到的注入物布局 bug）。"""
        mesh = make_mesh((1,), ("tp",))
        m = NeMoAttention()

        @inner_wrapper
        def w(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            orig = target_module.forward

            def fwd(q, k, v):
                return orig(q, k, v)
            target_module.forward = fwd

        spec = ModuleShardingSpec(inner_target="inner_attention", inner_wrapper=w,
                                  inner_out_src={TP: Shard(0)},
                                  region_dispatch=True)
        _wrap_inner_attention(m, None, spec=spec, mesh=mesh,
                              mesh_dim_names=("tp",))
        q = self._dtensor(mesh, torch.randn(1, 2, 4, 8))
        with pytest.raises(Exception, match="inner_out_src"):
            m.inner_attention(q, q, q)

    def test_dispatch_through_broken_chain_fails(self, make_mesh):
        """撒谎检测：声明 True 但注入物把 DTensor 解成了 local（脱离
        dispatch 链）→ fail-fast 教学报错。"""
        mesh = make_mesh((1,), ("tp",))
        m = NeMoAttention()

        @inner_wrapper
        def w(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            orig = target_module.forward

            def fwd(q, k, v):
                return orig(q, k, v).to_local()   # 破坏 dispatch 链
            target_module.forward = fwd

        spec = ModuleShardingSpec(inner_target="inner_attention", inner_wrapper=w, inner_out_src="first_input",
                                  region_dispatch=True)
        _wrap_inner_attention(m, None, spec=spec, mesh=mesh,
                              mesh_dim_names=("tp",))
        q = self._dtensor(mesh, torch.randn(1, 2, 4, 8))
        with pytest.raises(RuntimeError, match="not a DTensor"):
            m.inner_attention(q, q, q)

    def test_wrapper_without_region_dispatch_fails(self):
        """注入纪律：inner_wrapper 声明了但 region_dispatch 缺失 →
        fail-fast（无默认）。"""
        @inner_wrapper
        def w(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            pass
        spec = ModuleShardingSpec(inner_wrapper=w)
        with pytest.raises(ValueError, match="region_dispatch"):
            _resolve_inner_wrapper(NeMoAttention(), spec, _FakeCpMesh(),
                                   None, ())


class TestMisfireDetection:
    def test_sdpa_hf_not_fired_raises(self):
        """发火检测：'sdpa_hf' 拦截路但模块内部不调 F.sdpa → RuntimeError。"""
        class FakeHFAttn(HFSdpaAttention):
            def forward(self, hidden_states):
                # 不调 F.scaled_dot_product_attention 的自研实现
                return self.q_proj(hidden_states)

        m = FakeHFAttn()
        spec = ModuleShardingSpec(inner_target="self",
                                  inner_wrapper="sdpa_hf", region_dispatch=False)
        _wrap_inner_attention(m, _FakeCpMesh(), spec=spec)
        with pytest.raises(RuntimeError, match="did not intercept"):
            m(torch.randn(1, 2, 8))


class TestCpSdpaCallCondition:
    """D-04 触发条件：按 CP 语义（cp_mesh.size()>1）而非 q_len≠kv_len 形状比较。"""

    def test_is_causal_kept_when_cp_inactive(self):
        """cp_size=1：is_causal 原样透传，不替换显式 mask。"""
        from hyper_parallel.auto_models.components.distributed.cp_wrappers import (
            _cp_sdpa_call,
        )
        received = {}

        def fake_sdpa(q, k, v, **kwargs):
            received.update(kwargs)
            return q

        q = torch.randn(1, 1, 4, 2)
        _cp_sdpa_call(fake_sdpa, _FakeCpMesh(), q, q, q, {"is_causal": True})
        assert received.get("is_causal") is True
        assert "attn_mask" not in received


class TestStyleDetection:
    """风格判定 helper（cp_wrappers 公开工具，供自定义 wrapper 作者使用——
    框架自身不再据此启发式分派）。"""

    def test_hf_style_by_signature(self):
        assert is_hf_style_attention(HFSdpaAttention()) is True

    def test_nemo_style_not_hf(self):
        inner = NeMoAttention().inner_attention
        assert is_hf_style_attention(inner) is False

    def test_sdpa_detection(self):
        assert is_sdpa_attention(HFSdpaAttention()) is True
        assert is_flex_attention(HFSdpaAttention()) is False

    def test_flex_detection(self):
        assert is_flex_attention(FlexHFAattention()) is True
        assert is_sdpa_attention(FlexHFAattention()) is False


# ==========================================================================
# 来源: test_s3_shard_batch.py
# S3.6: shard_batch_for_cp（FakeMesh 单进程，逐 rank 参数化断言）。
# ==========================================================================

class FakeCpMesh:
    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank


def _batch(S=10):
    return {
        "input_ids": torch.arange(S).unsqueeze(0),
        "labels": torch.arange(100, 100 + S).unsqueeze(0),
        "position_ids": torch.arange(S).unsqueeze(0),
        "qkv_format": "thd",
    }


class TestShardBatch:
    def test_cp_size1_passthrough(self):
        b = _batch()
        out = shard_batch_for_cp(b, FakeCpMesh(1, 0))
        assert out is b

    def test_equal_split_no_pad(self):
        b = _batch(S=8)
        for rank, slc in ((0, slice(0, 4)), (1, slice(4, 8))):
            out = shard_batch_for_cp(b, FakeCpMesh(2, rank))
            torch.testing.assert_close(
                out["input_ids"], b["input_ids"][:, slc])
            torch.testing.assert_close(out["labels"], b["labels"][:, slc])
            assert out["qkv_format"] == "thd"

    def test_pad_to_2cp_multiple(self):
        """S=10 pad 到 12（2*cp=4 的倍数）→ chunk=6；最后 rank 的 chunk 含 pad 区。"""
        b = _batch(S=10)
        out0 = shard_batch_for_cp(b, FakeCpMesh(2, 0))
        out1 = shard_batch_for_cp(b, FakeCpMesh(2, 1))
        assert out0["input_ids"].shape[1] == 6
        assert out1["input_ids"].shape[1] == 6
        # rank0: 原始前 6 个
        torch.testing.assert_close(out0["input_ids"], b["input_ids"][:, :6])
        # N6：rank1 的 pad 区 label=-100、input_ids=0、position_ids 连续递增
        torch.testing.assert_close(out1["labels"][0, -2:],
                                   torch.tensor([-100, -100]))
        torch.testing.assert_close(out1["input_ids"][0, -2:],
                                   torch.tensor([0, 0]))
        torch.testing.assert_close(out1["position_ids"][0, -2:],
                                   torch.tensor([10, 11]))
        # rank1 的有效区 == 原始 [6:10]
        torch.testing.assert_close(out1["input_ids"][0, :4],
                                   b["input_ids"][0, 6:10])

    def test_non_tensor_passthrough(self):
        b = _batch(S=8)
        b["meta"] = "info"
        out = shard_batch_for_cp(b, FakeCpMesh(2, 0))
        assert out["meta"] == "info"

    def test_seq_lens_recomputed(self):
        b = _batch(S=8)
        b["seq_lens"] = torch.tensor([[8, -1000]])
        b["seq_lens_padded"] = torch.tensor([[8, -1000]])
        out = shard_batch_for_cp(b, FakeCpMesh(2, 1))
        # rank1 区间 [4,8)：一个 pack 截断为 4
        assert out["seq_lens"][0, 0].item() == 4


# ==========================================================================
# 来源: test_s3_shard_seq_lens.py
# S3.7: _shard_seq_lens_for_cp（pack 完全在内/跨界/在外/哨兵/防空）。
# ==========================================================================

SENTINEL = -1000


def _run(seq_lens, seq_lens_padded, cp_rank, chunk):
    return _shard_seq_lens_for_cp(seq_lens, seq_lens_padded,
                                  cp_rank=cp_rank, chunk=chunk)


class TestShardSeqLens:
    def test_pack_fully_inside(self):
        """pack [0,4) 完全落在 rank0 [0,4) 内 → 原样保留。"""
        sl = torch.tensor([[4, -1000]])
        slp = torch.tensor([[4, -1000]])
        out_lens, out_pad = _run(sl, slp, cp_rank=0, chunk=4)
        assert out_lens[0, 0].item() == 4
        assert out_pad[0, 0].item() == 4

    def test_pack_crosses_lo_boundary(self):
        """pack [0,6) 跨 rank1 的 lo=4 边界 → 截断为 [4,6) 长度 2。"""
        sl = torch.tensor([[6, -1000]])
        slp = torch.tensor([[6, -1000]])
        out_lens, out_pad = _run(sl, slp, cp_rank=1, chunk=4)
        assert out_lens[0, 0].item() == 2
        assert out_pad[0, 0].item() == 2

    def test_pack_crosses_hi_boundary(self):
        """pack [2,8) 跨 rank0 的 hi=4 → 截断为 [2,4) 长度 2。"""
        sl = torch.tensor([[6, -1000]])
        slp = torch.tensor([[6, -1000]])
        out_lens, out_pad = _run(sl, slp, cp_rank=0, chunk=4)
        # pack_start=2 → 需要另一个前置 pack 构造 offset
        sl = torch.tensor([[2, 6, -1000]])
        slp = torch.tensor([[2, 6, -1000]])
        out_lens, out_pad = _run(sl, slp, cp_rank=0, chunk=4)
        assert out_lens[0, 0].item() == 2   # 第一个 pack 完整
        assert out_lens[0, 1].item() == 2   # 第二个 pack 截断到 hi=4
        # 哨兵填充
        assert out_lens.shape[1] == 2

    def test_pack_fully_outside(self):
        """pack [0,4) 完全在 rank1 [4,8) 外 → 跳过（防空 → 哨兵）。"""
        sl = torch.tensor([[4, -1000]])
        slp = torch.tensor([[4, -1000]])
        out_lens, out_pad = _run(sl, slp, cp_rank=1, chunk=4)
        # max_local_packs=0→1 防空
        assert out_lens.shape == (1, 1)
        assert out_lens[0, 0].item() == SENTINEL

    def test_sentinel_terminates(self):
        """哨兵之后的 pack 不处理。"""
        sl = torch.tensor([[4, -1000, 4]])
        slp = torch.tensor([[4, -1000, 4]])
        out_lens, _ = _run(sl, slp, cp_rank=0, chunk=8)
        assert out_lens.shape[1] == 1
        assert out_lens[0, 0].item() == 4

    def test_padded_covers_separator(self):
        """seq_lens_padded 含 separator：pack 实际 3 + pad 1，跨界按 padded 截断。"""
        sl = torch.tensor([[3, -1000]])
        slp = torch.tensor([[4, -1000]])
        out_lens, out_pad = _run(sl, slp, cp_rank=0, chunk=2)
        # pack [0,4) 跨 hi=2：实际 token 截断 [0,2) → 2；pad 区间 [0,2) → 2
        assert out_lens[0, 0].item() == 2
        assert out_pad[0, 0].item() == 2
        # rank1 [2,4)：实际 token [2,3) → 1；pad [2,4) → 2
        out_lens1, out_pad1 = _run(sl, slp, cp_rank=1, chunk=2)
        assert out_lens1[0, 0].item() == 1
        assert out_pad1[0, 0].item() == 2

    def test_per_rank_asymmetry(self):
        """N5：rank0 与 rank1 的 pack 交集不同 → 重算结果不同（逐 rank 断言）。"""
        sl = torch.tensor([[5, 3, -1000]])
        slp = torch.tensor([[5, 3, -1000]])
        out0, _ = _run(sl, slp, cp_rank=0, chunk=4)
        out1, _ = _run(sl, slp, cp_rank=1, chunk=4)
        # rank0 [0,4)：pack1 [0,5) 截断 → 4；pack2 在外 → 共 1 项
        assert out0[0, 0].item() == 4
        # rank1 [4,8)：pack1 [0,5) 截断 → 1；pack2 [5,8) 完整 → 3
        assert out1[0, 0].item() == 1
        assert out1[0, 1].item() == 3
