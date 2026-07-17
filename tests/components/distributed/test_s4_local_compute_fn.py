# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S4.6: local_compute_fn 用户自定义 compute_fn + 派生门控（05 §4.4.3，单进程）。

覆盖点：
- 解析链优先级：local_compute_fn > TP-extend-EP 注入意图 > use_local_map
  纯门控（模块自身 forward）> None（不走骨架）；
- 派生门控：use_local_map=False + local_compute_fn → 仍走骨架（门控 =
  解析结果非 None，不读存储的 bool）；
- 骨架注入：custom compute_fn 在 _wrap_local_region_forward 内被调用，
  输入 local tensor、输出经 boundary 出口缝合；
- validate 模式：DTensor 输入由骨架 unwrap，compute_fn 无模式感知。
"""

import functools

import torch
import torch.nn as nn

from hyper_models.components.distributed.precompiled_boundary import (
    PrecompiledBoundary,
)
from hyper_models.components.distributed.sharding_applier import (
    _apply_phase_c,
    _resolve_local_compute_fn,
    _wrap_local_region_forward,
)
from hyper_models.components.distributed.sharding_config import (
    TP,
    ModuleShardingSpec,
    ShardingPlan,
)
from hyper_parallel.core.dtensor.placement_types import Shard


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
    from hyper_models.components.distributed.sharding_config import (
        _normalize_out_fields,
    )
    return _normalize_out_fields(ModuleShardingSpec(
        in_src={"x": {TP: Shard(1)}},
        in_dst={"x": {TP: Shard(1)}},   # identity
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
    ))


class TestResolveLocalComputeFn:
    def test_user_fn_wins_over_ep_injection(self, make_mesh):
        """local_compute_fn 优先于 TP-extend-EP 注入意图（_ep_size>0 也不走
        _hf_native_ep_compute）。"""
        my_fn = lambda module, x: x  # noqa: E731
        spec = _identity_spec()
        spec.local_compute_fn = my_fn
        spec._ep_size = 2   # 即使 EP 意图在，用户 fn 仍优先
        fn = _resolve_local_compute_fn(
            _TinyMod(), spec, make_mesh((1,), ("tp",)), ("tp",),
            expert_mesh=None)
        assert isinstance(fn, functools.partial)
        assert fn.func is my_fn

    def test_use_local_map_resolves_to_module_forward(self, make_mesh):
        """use_local_map 纯门控（无用户 fn / EP 意图）→ 模块自身 forward。"""
        mod = _TinyMod()
        spec = _identity_spec()
        spec.use_local_map = True
        fn = _resolve_local_compute_fn(
            mod, spec, make_mesh((1,), ("tp",)), ("tp",), expert_mesh=None)
        assert fn == mod.forward

    def test_no_declaration_returns_none(self, make_mesh):
        """三个来源皆无 → None（模块不走骨架，门控派生为 False）。"""
        fn = _resolve_local_compute_fn(
            _TinyMod(), _identity_spec(), make_mesh((1,), ("tp",)), ("tp",),
            expert_mesh=None)
        assert fn is None

    def test_derived_gate_via_apply_path(self, make_mesh):
        """派生门控端到端：use_local_map=False + local_compute_fn →
        _apply_phase_c 仍注入骨架并执行 custom fn（门控不读存储的 bool）。"""
        mesh = make_mesh((1,), ("tp",))
        calls = []

        def my_compute(module, x):
            calls.append(x)
            return module.lin(x) * 3

        model = _TinyModel()
        spec = _identity_spec()
        spec.local_compute_fn = my_compute
        assert spec.use_local_map is False   # 门控未被任何声明置位
        plan = ShardingPlan(modules={"mod": spec}, mesh_dim_names=("tp",))
        _apply_phase_c(model, plan, mesh, validate_mode=False)

        x = torch.randn(2, 4)
        out = model.mod(x)
        assert len(calls) == 1               # custom fn 被执行 → 骨架已注入
        torch.testing.assert_close(out, model.mod.lin(x) * 3)


class TestLocalRegionWithCustomComputeFn:
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

        def my_compute(module, x):
            calls.append((module, x))
            return module.lin(x) * 2   # 自定义逻辑：放大 2 倍

        mod = _TinyMod()
        spec = _identity_spec()
        spec.local_compute_fn = my_compute
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

        def my_compute(module, x):
            seen.append(x)
            return module.lin(x)

        mod = _TinyMod()
        spec = _identity_spec()
        spec.local_compute_fn = my_compute
        self._wrap(mod, spec, mesh, validate_mode=True)

        x = torch.randn(2, 4)
        out = mod(x)
        assert len(seen) == 1
        from hyper_parallel.core.dtensor.dtensor import DTensor
        assert not isinstance(seen[0], DTensor)   # compute_fn 内恒为 local
        assert not isinstance(out, DTensor)       # 骨架出口恒解包
        torch.testing.assert_close(out, mod.lin(x))
