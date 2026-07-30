# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_s2_boundary.py: 核心套件合并文件。

来源: test_s2_boundary_compile.py, test_s2_boundary_io.py, test_s2_path_utils.py, test_s2_tp_grad_info.py, test_local_region.py
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from hyper_models.components.distributed.local_region import local_region
from hyper_models.components.distributed.precompiled_boundary import (
    PrecompiledBoundary,
    RedistOp,
    _get_arg,
    _set_arg,
)
from hyper_models.components.distributed.sharding.apply import (
    _get_attr_by_path,
    _resolve_module,
    _set_param_by_path,
)
from hyper_models.components.distributed.sharding_config import (
    CP,
    ModuleShardingSpec,
    ShardingPlan,
    TP,
)
from hyper_models.components.distributed.tp_grad import build_tp_grad_info
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Replicate,
    Shard,
)


# ==========================================================================
# 来源: test_s2_boundary_compile.py
# S2.4: PrecompiledBoundary 编译期（identity 跳过 / 多输出映射 / None 分支）。
# ==========================================================================

class _FakeMesh:
    mesh_dim_names = ("tp", "cp")


def _attention_spec():
    """attention TP×CP 契约：TP 维通信、CP 维 identity。"""
    return ModuleShardingSpec(
        in_src={"hidden_states": {TP: Shard(1), CP: Shard(1)}},
        in_dst={"hidden_states": {TP: Replicate(), CP: Shard(1)}},
        out_src={"output": {TP: Partial(), CP: Shard(1)}},
        out_dst={"output": {TP: Shard(1), CP: Shard(1)}},
    )


class TestCompileInputPlan:
    def test_attention_cp_dim_identity(self):
        """attention CP 维 in_dst=Shard(1) identity 断言：in_plan 只有 TP all-gather。"""
        b = PrecompiledBoundary(_attention_spec(), _FakeMesh(), ("tp", "cp"))
        assert len(b.in_plan) == 1
        op = b.in_plan[0]
        assert op.collective_type == "all_gather"
        # CP 维 src==dst==Shard(1)（identity 体现在 placement 对相等，而非 CP 维通信）
        assert op.src_placements[1] == Shard(1)
        assert op.dst_placements[1] == Shard(1)

    def test_identity_input_still_compiled_as_passthrough(self):
        """in_src==in_dst → identity op（直通）。"""
        spec = ModuleShardingSpec(
            in_src={"x": {TP: Shard(1)}}, in_dst={"x": {TP: Shard(1)}})
        b = PrecompiledBoundary(spec, _FakeMesh(), ("tp",))
        assert len(b.in_plan) == 1
        assert b.in_plan[0].collective_type == "identity"


class TestCompileOutputPlan:
    def test_attention_out_cp_identity_skipped(self):
        """out_plan：CP 维 identity 不产生额外 op，仅 TP reduce-scatter。"""
        b = PrecompiledBoundary(_attention_spec(), _FakeMesh(), ("tp", "cp"))
        assert len(b.out_plan) == 1
        assert b.out_plan[0].collective_type == "reduce_scatter"

    def test_identity_output_plan_empty(self):
        spec = ModuleShardingSpec(
            out_src={"output": {TP: Shard(1)}},
            out_dst={"output": {TP: Shard(1)}},
        )
        b = PrecompiledBoundary(spec, _FakeMesh(), ("tp",))
        assert b.out_plan == []

    def test_multi_output_arg_index_from_out_names(self):
        spec = ModuleShardingSpec(
            out_src={"hidden_states": {TP: Partial()}, "present_kv": {TP: Shard(1)}},
            out_dst={"hidden_states": {TP: Shard(1)}, "present_kv": {TP: Replicate()}},
            out_names=["hidden_states", "present_kv"],
        )
        b = PrecompiledBoundary(spec, _FakeMesh(), ("tp",))
        idx = {op.arg_name: op.arg_index for op in b.out_plan}
        assert idx == {"hidden_states": 0, "present_kv": 1}

    def test_multi_output_arg_index_default_key_order(self):
        spec = ModuleShardingSpec(
            out_src={"a": {TP: Partial()}, "b": {TP: Shard(1)}},
            out_dst={"a": {TP: Shard(1)}, "b": {TP: Replicate()}},
        )
        b = PrecompiledBoundary(spec, _FakeMesh(), ("tp",))
        idx = {op.arg_name: op.arg_index for op in b.out_plan}
        assert idx == {"a": 0, "b": 1}

    def test_out_src_none_no_out_plan(self):
        spec = ModuleShardingSpec(out_src=None, out_dst={"output": {TP: Shard(1)}})
        b = PrecompiledBoundary(spec, _FakeMesh(), ("tp",))
        assert b.out_plan == []

    def test_out_dst_none_no_out_plan(self):
        spec = ModuleShardingSpec(out_src={"output": {TP: Partial()}}, out_dst=None)
        b = PrecompiledBoundary(spec, _FakeMesh(), ("tp",))
        assert b.out_plan == []


# ==========================================================================
# 来源: test_s2_boundary_io.py
# S2.5: redistribute_inputs/outputs + _get_arg/_set_arg 双通道（单进程 mock mesh）。
# ==========================================================================

@pytest.fixture(scope="module")
def mesh():
    if not dist.is_initialized():
        dist.init_process_group(
            "gloo", init_method="tcp://127.0.0.1:29721", rank=0, world_size=1)
    return init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))


class TestGetSetArg:
    def test_kwargs_hit_priority(self):
        args, kwargs = (torch.tensor([1]),), {"x": torch.tensor([2])}
        got = _get_arg(args, kwargs, "x", 0)
        assert got.item() == 2  # kwargs 优先于 args[0]

    def test_args_idx_fallback(self):
        args, kwargs = (torch.tensor([1]),), {}
        got = _get_arg(args, kwargs, "x", 0)
        assert got.item() == 1

    def test_missing_returns_default(self):
        assert _get_arg((), {}, "x", None, default="d") == "d"

    def test_idx_out_of_range_falls_to_kwargs(self):
        args, kwargs = (), {}
        args, kwargs = _set_arg(args, kwargs, "x", 5, "v")
        assert kwargs == {"x": "v"}

    def test_set_arg_kwargs_channel(self):
        args, kwargs = (), {"x": 1}
        args, kwargs = _set_arg(args, kwargs, "x", None, 2)
        assert kwargs["x"] == 2

    def test_set_arg_args_channel(self):
        args, kwargs = (1, 2), {}
        args, kwargs = _set_arg(args, kwargs, "x", 1, 9)
        assert args == (1, 9)


class TestRedistributeIO:
    def test_inputs_kwargs_hit(self, mesh):
        spec = ModuleShardingSpec(
            in_src={"x": {TP: Replicate()}}, in_dst={"x": {TP: Replicate()}})
        b = PrecompiledBoundary(spec, mesh, ("tp",))
        t = torch.tensor([1.0])
        args, kwargs = b.redistribute_inputs((), {"x": t})
        assert kwargs["x"] is t  # identity 直通

    def test_inputs_missing_arg_skipped(self, mesh):
        """arg 未找到（None）→ 跳过，不向 kwargs 注入 None。"""
        spec = ModuleShardingSpec(
            in_src={"input": {TP: Replicate()}}, in_dst={"input": {TP: Replicate()}})
        b = PrecompiledBoundary(spec, mesh, ("tp",))
        args, kwargs = b.redistribute_inputs((torch.tensor([1]),), {})
        assert "input" not in kwargs

    def test_outputs_single(self, mesh):
        spec = ModuleShardingSpec(
            out_src={"output": {TP: Shard(1)}}, out_dst={"output": {TP: Shard(1)}})
        b = PrecompiledBoundary(spec, mesh, ("tp",))
        # identity out_plan 为空 → 原样返回
        t = torch.tensor([1.0])
        assert b.redistribute_outputs(t) is t

    def test_outputs_tuple_order_preserved(self, mesh):
        spec = ModuleShardingSpec(
            out_src={"a": {TP: Shard(1)}, "b": {TP: Replicate()}},
            out_dst={"a": {TP: Shard(1)}, "b": {TP: Replicate()}},
            out_names=["a", "b"],
        )
        b = PrecompiledBoundary(spec, mesh, ("tp",))
        ta, tb = torch.tensor([1.0]), torch.tensor([2.0])
        out = b.redistribute_outputs((ta, tb))
        assert isinstance(out, tuple) and out[0] is ta and out[1] is tb

    def test_outputs_index_out_of_range_warns_and_skips(self, mesh, caplog):
        spec = ModuleShardingSpec(
            out_src={"a": {TP: Shard(1)}, "b": {TP: Partial()}},
            out_dst={"a": {TP: Replicate()}, "b": {TP: Replicate()}},
            out_names=["a", "b"],
        )
        b = PrecompiledBoundary(spec, mesh, ("tp",))
        import logging
        with caplog.at_level(logging.WARNING):
            out = b.redistribute_outputs((torch.tensor([[1.0]]),))
        assert "Skipping" in caplog.text
        assert len(out) == 1

    def test_inputs_as_dtensor(self, mesh):
        spec = ModuleShardingSpec(
            in_src={"x": {TP: Replicate()}}, in_dst={"x": {TP: Replicate()}})
        b = PrecompiledBoundary(spec, mesh, ("tp",))
        from hyper_parallel.core.dtensor.dtensor import DTensor
        args, kwargs = b.redistribute_inputs((), {"x": torch.tensor([1.0])},
                                             as_dtensor=True)
        assert isinstance(kwargs["x"], DTensor)


# ==========================================================================
# 来源: test_s2_path_utils.py
# S2.1: 路径工具 _resolve_module / _get_attr_by_path / _set_param_by_path。
# ==========================================================================

class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.lm_head = nn.Linear(4, 4)


class TestPathUtils:
    def test_resolve_module_nested_modulelist(self):
        net = _Net()
        assert _resolve_module(net, "model.layers.0") is net.model.layers[0]
        assert _resolve_module(net, "model.layers.1") is net.model.layers[1]
        assert _resolve_module(net, "lm_head") is net.lm_head

    def test_resolve_module_no_leaf_strip(self):
        """不剥离末段：传模块 FQN 返回模块本身，而非父模块。"""
        net = _Net()
        mod = _resolve_module(net, "model.layers.0")
        assert isinstance(mod, nn.Linear)

    def test_get_attr_by_path_param(self):
        net = _Net()
        w = _get_attr_by_path(net, "model.layers.0.weight")
        assert w is net.model.layers[0].weight

    def test_set_param_by_path_register_parameter(self):
        net = _Net()
        new_w = nn.Parameter(torch.ones(4, 4))
        _set_param_by_path(net, "model.layers.1.weight", new_w)
        assert net.model.layers[1].weight is new_w
        # register_parameter 路径：在 _parameters 中
        assert net.model.layers[1]._parameters["weight"] is new_w

    def test_set_param_by_path_setattr_branch(self):
        class Plain:
            pass
        obj = Plain()
        p = nn.Parameter(torch.ones(2))
        _set_param_by_path(obj, "w", p)
        assert obj.w is p

    def test_set_param_by_path_numeric_segment(self):
        net = _Net()
        new_w = nn.Parameter(torch.zeros(4, 4))
        _set_param_by_path(net, "model.layers.0.bias",
                           nn.Parameter(torch.zeros(4)))
        _set_param_by_path(net, "model.layers.0.weight", new_w)
        assert net.model.layers[0].weight is new_w


# ==========================================================================
# 来源: test_s2_tp_grad_info.py
# S2.9: build_tp_grad_info + tied 归一化（单进程，mock mesh）。
# ==========================================================================

class _FakeTpMesh:
    pass


def _plan():
    plan = ShardingPlan(mesh_dim_names=("tp",))
    plan.modules["model.embed_tokens"] = ModuleShardingSpec(
        params={"weight": {TP: Shard(0)}})
    plan.modules["model.layers.0.input_layernorm"] = ModuleShardingSpec(
        params={"weight": {TP: Replicate()}})
    plan.modules["lm_head"] = ModuleShardingSpec(
        params={"weight": {TP: Shard(0)}})
    return plan


def test_reads_from_plan_not_dtensor():
    mesh = _FakeTpMesh()
    info = build_tp_grad_info(_plan(), mesh)
    assert info["model.embed_tokens.weight"] == (Shard(0), mesh)
    assert info["model.layers.0.input_layernorm.weight"] == (Replicate(), mesh)
    assert info["lm_head.weight"] == (Shard(0), mesh)


def test_tied_consistent_placements_unchanged():
    plan = _plan()
    plan.tied_pairs = [("model.embed_tokens.weight", "lm_head.weight")]
    info = build_tp_grad_info(plan, _FakeTpMesh())
    assert info["model.embed_tokens.weight"][0] == Shard(0)
    assert info["lm_head.weight"][0] == Shard(0)


def test_tied_inconsistent_shard_wins():
    """tied 对 placement 不一致 → 取 Shard 优先。"""
    plan = _plan()
    plan.modules["lm_head"] = ModuleShardingSpec(
        params={"weight": {TP: Replicate()}})
    plan.tied_pairs = [("model.embed_tokens.weight", "lm_head.weight")]
    info = build_tp_grad_info(plan, _FakeTpMesh())
    assert info["model.embed_tokens.weight"][0] == Shard(0)
    assert info["lm_head.weight"][0] == Shard(0)  # 归一化为 Shard


def test_tied_pair_not_in_plan_ignored():
    plan = _plan()
    plan.tied_pairs = [("ghost.a", "ghost.b")]
    info = build_tp_grad_info(plan, _FakeTpMesh())
    assert "ghost.a" not in info


def test_explicit_tied_pairs_override():
    plan = _plan()
    info = build_tp_grad_info(
        plan, _FakeTpMesh(),
        tied_pairs=[("model.embed_tokens.weight", "lm_head.weight")])
    assert info["lm_head.weight"][0] == Shard(0)


def test_no_tp_axis_mesh_none():
    plan = ShardingPlan(mesh_dim_names=("cp",))
    plan.modules["m"] = ModuleShardingSpec(params={"w": {"cp": Replicate()}})
    info = build_tp_grad_info(plan, None)
    # 无 tp 键 → 默认 Replicate
    assert info["m.w"][0] == Replicate()


# ==========================================================================
# 来源: test_local_region.py
# local_region 单元测试（单进程，torch/cpu 平台）。
# ==========================================================================

@pytest.fixture(scope="module")
def mesh__local_region():
    """单 rank mesh__local_region：world_size=1 时 Replicate 与任意 Shard 语义等价，
    足以验证包装/解包/autograd 逻辑（多进程数值归 distributed UT）。"""
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(
            "gloo", init_method="tcp://127.0.0.1:29511", rank=0, world_size=1
        )
    return DeviceMesh("cpu", (1,), mesh_dim_names=("tp",))


def _make_dtensor(mesh__local_region, data, requires_grad=False):
    local = torch.tensor(data, dtype=torch.float32, requires_grad=requires_grad)
    return DTensor.from_local(local, mesh__local_region, [Replicate()])


class TestWrapUnwrap:
    def test_kwargs_input_and_output_wrap(self, mesh__local_region):
        def fn(hidden_states, scale=None):
            assert not isinstance(hidden_states, DTensor)  # 区域内是 local tensor
            return hidden_states * (scale or 2.0)

        wrapped = local_region(
            fn, device_mesh=mesh__local_region,
            in_placements={"hidden_states": (Replicate(),)},
            out_placements=(Replicate(),),
        )
        dt = _make_dtensor(mesh__local_region, [1.0, 2.0, 3.0])
        out = wrapped(hidden_states=dt, scale=3.0)
        assert isinstance(out, DTensor)
        assert tuple(out.placements) == (Replicate(),)
        assert torch.allclose(out.to_local(), torch.tensor([3.0, 6.0, 9.0]))

    def test_positional_input_via_signature_binding(self, mesh__local_region):
        def fn(x, y):
            return x + y

        wrapped = local_region(
            fn, device_mesh=mesh__local_region,
            in_placements={"x": (Replicate(),), "y": (Replicate(),)},
            out_placements=(Replicate(),),
        )
        dt_x = _make_dtensor(mesh__local_region, [1.0, 2.0])
        dt_y = _make_dtensor(mesh__local_region, [10.0, 20.0])
        out = wrapped(dt_x, dt_y)
        assert isinstance(out, DTensor)
        assert torch.allclose(out.to_local(), torch.tensor([11.0, 22.0]))

    def test_plain_tensor_passthrough_no_wrap(self, mesh__local_region):
        """全部输入非 DTensor（production 参数已解包场景）→ 输出不包装。"""
        def fn(x):
            return x * 2.0

        wrapped = local_region(
            fn, device_mesh=mesh__local_region,
            in_placements={"x": (Replicate(),)},
            out_placements=(Replicate(),),
        )
        out = wrapped(torch.tensor([1.0, 2.0]))
        assert not isinstance(out, DTensor)
        assert torch.allclose(out, torch.tensor([2.0, 4.0]))

    def test_mixed_dtensor_and_plain_args(self, mesh__local_region):
        def fn(x, bias):
            return x + bias

        wrapped = local_region(
            fn, device_mesh=mesh__local_region,
            in_placements={"x": (Replicate(),)},
            out_placements=(Replicate(),),
        )
        dt = _make_dtensor(mesh__local_region, [1.0, 2.0])
        out = wrapped(dt, torch.tensor([100.0, 100.0]))
        assert isinstance(out, DTensor)
        assert torch.allclose(out.to_local(), torch.tensor([101.0, 102.0]))

    def test_tuple_output_with_none_placeholder(self, mesh__local_region):
        def fn(x):
            return x * 2.0, "meta"

        wrapped = local_region(
            fn, device_mesh=mesh__local_region,
            in_placements={"x": (Replicate(),)},
            out_placements=((Replicate(),), None),
        )
        dt = _make_dtensor(mesh__local_region, [1.0, 2.0])
        out_tensor, meta = wrapped(dt)
        assert isinstance(out_tensor, DTensor)
        assert meta == "meta"

    def test_output_already_dtensor_not_rewrapped(self, mesh__local_region):
        def fn(x):
            return DTensor.from_local(x * 2.0, mesh__local_region, [Replicate()])

        wrapped = local_region(
            fn, device_mesh=mesh__local_region,
            in_placements={"x": (Replicate(),)},
            out_placements=(Replicate(),),
        )
        out = wrapped(_make_dtensor(mesh__local_region, [1.0]))
        assert isinstance(out, DTensor)
        assert torch.allclose(out.to_local(), torch.tensor([2.0]))


class TestContractValidation:
    def test_out_placements_count_mismatch(self, mesh__local_region):
        def fn(x):
            return x, x

        wrapped = local_region(
            fn, device_mesh=mesh__local_region,
            in_placements={"x": (Replicate(),)},
            out_placements=((Replicate(),), (Replicate(),), (Replicate(),)),
        )
        with pytest.raises(ValueError, match="does not match"):
            wrapped(_make_dtensor(mesh__local_region, [1.0]))

    def test_flat_out_placements_rejected_for_multi_output(self, mesh__local_region):
        def fn(x):
            return x, x

        wrapped = local_region(
            fn, device_mesh=mesh__local_region,
            in_placements={"x": (Replicate(),)},
            out_placements=(Replicate(),),  # 扁平写法仅允许单输出
        )
        with pytest.raises(ValueError, match="single-output"):
            wrapped(_make_dtensor(mesh__local_region, [1.0]))

    def test_tensor_output_with_none_placement_raises(self, mesh__local_region):
        def fn(x):
            return x

        wrapped = local_region(
            fn, device_mesh=mesh__local_region,
            in_placements={"x": (Replicate(),)},
            out_placements=(None,),
        )
        with pytest.raises(TypeError, match="non-None out_placements"):
            wrapped(_make_dtensor(mesh__local_region, [1.0]))

    def test_out_placements_none_returns_raw(self, mesh__local_region):
        def fn(x):
            return x * 2.0

        wrapped = local_region(
            fn, device_mesh=mesh__local_region,
            in_placements={"x": (Replicate(),)},
            out_placements=None,
        )
        out = wrapped(_make_dtensor(mesh__local_region, [1.0]))
        assert not isinstance(out, DTensor)
        assert torch.allclose(out, torch.tensor([2.0]))
