# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S3.2: inner-wrap 双解析链 + HF/NeMo 风格判定 + 注册表（单进程 mock）。

覆盖：
- _resolve_inner_target：自动定位三策略 + inner_target 用户指定/fail-fast；
- _resolve_inner_wrapper：无声明→None（派生门控）/ 启发式分派 / str 注册表
  （含未知名 fail-fast、target 缺失 fail-fast）/ callable 自定义；
- 应用侧：_resolved_inner_wrapper 回写、自定义 wrapper 接管；
- 发火检测：'sdpa_hf' 未拦到 F.sdpa 调用 → RuntimeError；
- D-04 触发条件：cp_mesh.size()>1 语义判断。
"""

import pytest
import torch
import torch.nn as nn

from hyper_models.components.distributed.sharding_applier import (
    CP_WRAPPER_REGISTRY,
    _is_flex_attention,
    _is_hf_style_attention,
    _is_sdpa_attention,
    _resolve_inner_target,
    _resolve_inner_wrapper,
    _wrap_cp_inner_attention,
)
from hyper_models.components.distributed.sharding_config import (
    ModuleShardingSpec,
)


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
    def test_explicit_inner_attention_attr(self):
        m = NeMoAttention()
        assert _resolve_inner_target(m) is m.inner_attention

    def test_hf_classname_self(self):
        m = HFSdpaAttention()
        assert _resolve_inner_target(m) is m

    def test_structural_qkv_fallback(self):
        class Plain(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(8, 8)
                self.k_proj = nn.Linear(8, 8)
                self.v_proj = nn.Linear(8, 8)

            def forward(self, hidden_states):
                return hidden_states
        m = Plain()
        assert _resolve_inner_target(m) is m

    def test_not_found_returns_none(self):
        class Bare(nn.Module):
            def forward(self, x):
                return x
        assert _resolve_inner_target(Bare()) is None

    def test_user_inner_target_attr(self):
        """spec.inner_target 显式指定属性名 → 最高优先级命中。"""
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
        """无 inner_target/inner_wrapper/_needs_cp_attn → None（派生门控）。"""
        resolved = _resolve_inner_wrapper(
            NeMoAttention(), ModuleShardingSpec(), _FakeCpMesh(), None, ())
        assert resolved is None

    def test_heuristic_dispatch_nemo(self):
        """_needs_cp_attn → 启发式分派：NeMo 风格 → 'sdpa_qkv'。"""
        spec = ModuleShardingSpec(_needs_cp_attn=True)
        name, target, _ = _resolve_inner_wrapper(
            NeMoAttention(), spec, _FakeCpMesh(), None, ())
        assert name == "sdpa_qkv"
        assert target is not None

    def test_heuristic_dispatch_hf(self):
        """_needs_cp_attn → 启发式分派：HF 风格 → 'sdpa_hf'。"""
        spec = ModuleShardingSpec(_needs_cp_attn=True)
        name, target, _ = _resolve_inner_wrapper(
            HFSdpaAttention(), spec, _FakeCpMesh(), None, ())
        assert name == "sdpa_hf"

    def test_str_registry_lookup(self):
        """inner_wrapper='sdpa_qkv'（str）→ 显式选注册表方案，跳过启发式。"""
        spec = ModuleShardingSpec(inner_wrapper="sdpa_qkv")
        name, _, _ = _resolve_inner_wrapper(
            HFSdpaAttention(), spec, _FakeCpMesh(), None, ())
        assert name == "sdpa_qkv"   # HF 模块也可显式选 qkv 路（用户负责）

    def test_str_unknown_name_raises(self):
        """inner_wrapper=str 未注册 → fail-fast 并列出可用名。"""
        spec = ModuleShardingSpec(inner_wrapper="sdpa_hff")  # 拼写错误
        with pytest.raises(ValueError, match="CP_WRAPPER_REGISTRY"):
            _resolve_inner_wrapper(
                NeMoAttention(), spec, _FakeCpMesh(), None, ())

    def test_str_missing_target_raises(self):
        """inner_wrapper=str 但 target 定位失败 → fail-fast 提示 inner_target。"""
        class Bare(nn.Module):
            def forward(self, x):
                return x
        spec = ModuleShardingSpec(inner_wrapper="sdpa_qkv")
        with pytest.raises(ValueError, match="inner_target"):
            _resolve_inner_wrapper(Bare(), spec, _FakeCpMesh(), None, ())

    def test_user_registry_extension(self):
        """用户注册命名方案后可按名引用。"""
        calls = []

        def my_fn(target, cp_mesh, **ctx):
            calls.append(target)

        CP_WRAPPER_REGISTRY["test_custom"] = my_fn
        try:
            spec = ModuleShardingSpec(inner_wrapper="test_custom")
            name, target, apply_fn = _resolve_inner_wrapper(
                NeMoAttention(), spec, _FakeCpMesh(), None, ())
            assert name == "test_custom"
            apply_fn()
            assert calls == [target]
        finally:
            CP_WRAPPER_REGISTRY.pop("test_custom")

    def test_callable_custom(self):
        """inner_wrapper=callable → ('custom', target, ...)。"""
        m = NeMoAttention()
        spec = ModuleShardingSpec(inner_wrapper=lambda t, cm: None)
        name, target, _ = _resolve_inner_wrapper(
            m, spec, _FakeCpMesh(), None, ())
        assert name == "custom"
        assert target is m.inner_attention

    def test_callable_custom_target_fallback(self):
        """callable + 定位失败 → target 退化为边界模块本身。"""
        class Bare(nn.Module):
            def forward(self, x):
                return x
        m = Bare()
        spec = ModuleShardingSpec(inner_wrapper=lambda t, cm: None)
        name, target, _ = _resolve_inner_wrapper(
            m, spec, _FakeCpMesh(), None, ())
        assert name == "custom"
        assert target is m

    def test_target_missing_fail_fast(self):
        """_needs_cp_attn + 定位失败 → ValueError（静默数值错误必须 fail-fast）。"""
        class Bare(nn.Module):
            def forward(self, x):
                return x
        spec = ModuleShardingSpec(_needs_cp_attn=True)
        with pytest.raises(ValueError, match="inner_target"):
            _resolve_inner_wrapper(Bare(), spec, _FakeCpMesh(), None, ())


class TestCpWrapApply:
    def test_apply_writes_resolved_name(self):
        """应用后 spec._resolved_inner_wrapper 回写（plan 内省）。"""
        spec = ModuleShardingSpec(_needs_cp_attn=True)
        _wrap_cp_inner_attention(NeMoAttention(), _FakeCpMesh(), spec=spec)
        assert spec._resolved_inner_wrapper == "sdpa_qkv"

    def test_custom_callable_takeover(self):
        """自定义 callable 整体接管：以 (target, cp_mesh) 调用并替换 forward。"""
        class Bare(nn.Module):
            def forward(self, x):
                return x

        calls = []

        def my_cp_wrapper(target, cp_mesh):
            calls.append((target, cp_mesh))
            orig = target.forward

            def wrapped(q, k, v, **kwargs):
                return orig(q)

            target.forward = wrapped

        m = Bare()
        spec = ModuleShardingSpec(inner_wrapper=my_cp_wrapper)
        mesh = _FakeCpMesh()
        _wrap_cp_inner_attention(m, mesh, spec=spec)
        assert calls and calls[0][1] is mesh
        assert spec._resolved_inner_wrapper == "custom"
        assert m.forward(torch.tensor(1), None, None) is not None

    def test_custom_callable_with_inner_target(self):
        """inner_target + inner_wrapper 组合：target 为用户指定的子模块。"""
        m = NeMoAttention()
        received = []

        def my_cp_wrapper(target, cp_mesh):
            received.append(target)

        spec = ModuleShardingSpec(inner_target="inner_attention",
                                  inner_wrapper=my_cp_wrapper)
        _wrap_cp_inner_attention(m, _FakeCpMesh(), spec=spec)
        assert received == [m.inner_attention]

    def test_str_pinned_wrapper_applied(self):
        """inner_wrapper='sdpa_qkv' 显式固定：应用的是注册表里的函数。"""
        m = NeMoAttention()
        spec = ModuleShardingSpec(inner_wrapper="sdpa_qkv")
        _wrap_cp_inner_attention(m, _FakeCpMesh(), spec=spec)
        assert spec._resolved_inner_wrapper == "sdpa_qkv"
        # forward 已被 (q,k,v) wrapper 替换
        q = torch.randn(1, 1, 2, 4)
        assert m.inner_attention(q, q, q) is not None


class TestMisfireDetection:
    def test_sdpa_hf_not_fired_raises(self):
        """发火检测：'sdpa_hf' 拦截路但模块内部不调 F.sdpa → RuntimeError。"""
        class FakeHFAttn(HFSdpaAttention):
            def forward(self, hidden_states):
                # 不调 F.scaled_dot_product_attention 的自研实现
                return self.q_proj(hidden_states)

        m = FakeHFAttn()
        spec = ModuleShardingSpec(_needs_cp_attn=True)
        _wrap_cp_inner_attention(m, _FakeCpMesh(), spec=spec)
        with pytest.raises(RuntimeError, match="did not intercept"):
            m(torch.randn(1, 2, 8))


class TestCpSdpaCallCondition:
    """D-04 触发条件：按 CP 语义（cp_mesh.size()>1）而非 q_len≠kv_len 形状比较。"""

    def test_is_causal_kept_when_cp_inactive(self):
        """cp_size=1：is_causal 原样透传，不替换显式 mask。"""
        from hyper_models.components.distributed.sharding_applier import (
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
    def test_hf_style_by_signature(self):
        assert _is_hf_style_attention(HFSdpaAttention()) is True

    def test_nemo_style_not_hf(self):
        inner = NeMoAttention().inner_attention
        assert _is_hf_style_attention(inner) is False

    def test_sdpa_detection(self):
        assert _is_sdpa_attention(HFSdpaAttention()) is True
        assert _is_flex_attention(HFSdpaAttention()) is False

    def test_flex_detection(self):
        assert _is_flex_attention(FlexHFAattention()) is True
        assert _is_sdpa_attention(FlexHFAattention()) is False
