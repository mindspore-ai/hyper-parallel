# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_s2_boundary.py: core merged suite file.

Sources: test_s2_boundary_compile.py, test_s2_boundary_io.py, test_s2_path_utils.py,
test_s2_source_shard_info.py, test_local_region.py

Grouped by feature family: compile input plan family, compile output plan
family, _get_arg/_set_arg family, path utils family, source shard info family,
tied weights consistency family, error paths family.

The real-process-group redistribute IO / local_region families moved to the
Gate-2 CPU Gloo integration worker
tests/torch/auto_models/distributed/_test_precompiled_boundary_gloo.py —
Gate-1 unit tests must not initialize process groups.
"""

# Pytest injects fixtures through parameters that intentionally reuse fixture names.
# pylint: disable=redefined-outer-name

import pytest
import torch
from torch import nn
from hyper_parallel.distributed._builder.precompiled_boundary import (
    PrecompiledBoundary,
    _get_arg,
    _set_arg,
)
from hyper_parallel.distributed._builder.parameter_sharding import (
    _get_attr_by_path,
    _resolve_module,
    _set_param_by_path,
)
from hyper_parallel.distributed.plan import ShardingPlan
from hyper_parallel.distributed.recipe_spec import (
    CP,
    EP,
    ModuleShardingSpec,
    TP,
)
from hyper_parallel.distributed._builder.source_shard import build_source_shard_info
from hyper_parallel.distributed._builder.tp_collective_lowering import (
    TPExecutionOp,
    create_tp_collective_lowerer,
)
from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Replicate,
    Shard,
)


class _FakeMesh:
    mesh_dim_names = ("tp", "cp")


class _FakeTPMesh:
    """Minimal TP mesh used to verify production lowering without distributed init."""

    mesh_dim_names = ("tp",)
    rank_list = (0, 1)

    def __init__(self):
        """Create a distinct fake process group."""
        self.group = object()

    def get_group(self):
        """Return the fake process group."""
        return self.group

    @staticmethod
    def size():
        """Return the fake TP group size."""
        return 2

    @staticmethod
    def get_local_rank():
        """Return the fake rank inside the TP group."""
        return 1


def _set_fake_group_ranks(monkeypatch, ranks=(0, 1)):
    monkeypatch.setattr(
        "hyper_parallel.distributed._builder.tp_collective_lowering."
        "platform.get_process_group_ranks",
        lambda _group: list(ranks),
    )


def _attention_spec():
    """attention TPxCP contract: communication on the TP axis, identity on the CP axis."""
    return ModuleShardingSpec(
        in_src={"hidden_states": {TP: Shard(1), CP: Shard(1)}},
        in_dst={"hidden_states": {TP: Replicate(), CP: Shard(1)}},
        out_src={"output": {TP: Partial(), CP: Shard(1)}},
        out_dst={"output": {TP: Shard(1), CP: Shard(1)}},
    )


# ==========================================================================
# Source: test_s2_boundary_compile.py
# S2.4: PrecompiledBoundary compile time (identity skip / multi-output mapping / None branches).
# ==========================================================================

def test_compile_input_plan(monkeypatch):
    """Compile input plan family: in_plan structure, TP lowering hit/skip paths."""
    # --- case: attention_cp_dim_identity ---
    # attention CP-axis in_dst=Shard(1) identity assertion: in_plan has only the TP all-gather.
    b = PrecompiledBoundary(_attention_spec(), _FakeMesh(), ("tp", "cp"))
    assert len(b.in_plan) == 1, "case: attention_cp_dim_identity"
    op = b.in_plan[0]
    assert op.collective_type == "all_gather", "case: attention_cp_dim_identity"
    # CP axis src==dst==Shard(1) (identity is expressed by equal placement pairs, not CP-axis communication)
    assert op.src_placements[1] == Shard(1), "case: attention_cp_dim_identity"
    assert op.dst_placements[1] == Shard(1), "case: attention_cp_dim_identity"

    # --- case: identity_input_still_compiled_as_passthrough ---
    # in_src==in_dst -> identity op (pass-through).
    spec = ModuleShardingSpec(
        in_src={"x": {TP: Shard(1)}}, in_dst={"x": {TP: Shard(1)}})
    b = PrecompiledBoundary(spec, _FakeMesh(), ("tp",))
    assert len(b.in_plan) == 1, "case: identity_input_still_compiled_as_passthrough"
    assert b.in_plan[0].collective_type == "identity", \
        "case: identity_input_still_compiled_as_passthrough"

    # --- case: execution_lowers_tp_transition ---
    # Production plans lower common TP transitions before the first forward.
    _set_fake_group_ranks(monkeypatch)
    for src, dst, kind, tensor_dim in [
        (Shard(1), Replicate(), "all_gather", 1),
        (Partial(), Replicate(), "all_reduce", None),
        (Partial(), Shard(1), "reduce_scatter", 1),
    ]:
        spec = ModuleShardingSpec(
            in_src={"x": {TP: src}},
            in_dst={"x": {TP: dst}},
        )
        boundary = PrecompiledBoundary(
            spec,
            _FakeTPMesh(),
            ("tp",),
            op_lowerer=create_tp_collective_lowerer(
                _FakeTPMesh(), ("tp",), collective_backend="hccl"
            ),
        )
        execution_op = boundary.in_plan[0].execution_op
        assert isinstance(execution_op, TPExecutionOp), \
            f"case: execution_lowers_tp_transition[{kind}]"
        assert execution_op.kind == kind, f"case: execution_lowers_tp_transition[{kind}]"
        assert execution_op.tensor_dim == tensor_dim, \
            f"case: execution_lowers_tp_transition[{kind}]"

    # --- case: gloo_reduce_scatter_uses_differentiable_fallback ---
    # Gloo lacks reduce-scatter; use all-reduce followed by local chunking.
    spec = ModuleShardingSpec(
        in_src={"x": {TP: Partial()}},
        in_dst={"x": {TP: Shard(1)}},
    )
    boundary = PrecompiledBoundary(
        spec,
        _FakeTPMesh(),
        ("tp",),
        op_lowerer=create_tp_collective_lowerer(
            _FakeTPMesh(), ("tp",), collective_backend="gloo"
        ),
    )
    execution_op = boundary.in_plan[0].execution_op
    assert isinstance(execution_op, TPExecutionOp), \
        "case: gloo_reduce_scatter_uses_differentiable_fallback"
    assert execution_op.kind == "all_reduce_shard", \
        "case: gloo_reduce_scatter_uses_differentiable_fallback"

    # --- case: validate_plan_does_not_lower_tp_transition ---
    # Placement validation keeps the DTensor redistribution path.
    spec = ModuleShardingSpec(
        in_src={"x": {TP: Shard(1)}},
        in_dst={"x": {TP: Replicate()}},
    )
    boundary = PrecompiledBoundary(spec, _FakeTPMesh(), ("tp",))
    assert boundary.in_plan[0].execution_op is None, \
        "case: validate_plan_does_not_lower_tp_transition"

    # --- case: replicate_to_shard_is_not_lowered ---
    # Local slicing stays generic until its backward all-gather is explicit.
    spec = ModuleShardingSpec(
        in_src={"x": {TP: Replicate()}},
        in_dst={"x": {TP: Shard(1)}},
    )
    boundary = PrecompiledBoundary(
        spec,
        _FakeTPMesh(),
        ("tp",),
        op_lowerer=create_tp_collective_lowerer(
            _FakeTPMesh(), ("tp",), collective_backend="hccl"
        ),
    )
    assert boundary.in_plan[0].execution_op is None, \
        "case: replicate_to_shard_is_not_lowered"

    # --- case: non_tp_difference_is_not_lowered ---
    # A boundary that also changes CP remains on the generic redistribution path.
    class _FakeTPAndCPMesh(_FakeTPMesh):
        mesh_dim_names = ("tp", "cp")

        def __getitem__(self, name):
            """Return the fake submesh for every requested axis."""
            del name
            return self

    mesh = _FakeTPAndCPMesh()
    spec = ModuleShardingSpec(
        in_src={"x": {TP: Shard(1), CP: Shard(1)}},
        in_dst={"x": {TP: Replicate(), CP: Replicate()}},
    )
    boundary = PrecompiledBoundary(
        spec,
        mesh,
        ("tp", "cp"),
        op_lowerer=create_tp_collective_lowerer(
            mesh, ("tp", "cp"), collective_backend="hccl"
        ),
    )
    assert boundary.in_plan[0].execution_op is None, \
        "case: non_tp_difference_is_not_lowered"

    # --- case: rank_order_mismatch_is_not_lowered ---
    # Explicit collectives require process-group order to match TP placement order.
    _set_fake_group_ranks(monkeypatch, ranks=(1, 0))
    spec = ModuleShardingSpec(
        in_src={"x": {TP: Shard(1)}},
        in_dst={"x": {TP: Replicate()}},
    )
    boundary = PrecompiledBoundary(
        spec,
        _FakeTPMesh(),
        ("tp",),
        op_lowerer=create_tp_collective_lowerer(
            _FakeTPMesh(), ("tp",), collective_backend="hccl"
        ),
    )
    assert boundary.in_plan[0].execution_op is None, \
        "case: rank_order_mismatch_is_not_lowered"


def test_compile_output_plan():
    """Compile output plan family: out_plan structure, arg_index mapping, None branches."""
    # --- case: attention_out_cp_identity_skipped ---
    # out_plan: CP-axis identity produces no extra op, only the TP reduce-scatter.
    b = PrecompiledBoundary(_attention_spec(), _FakeMesh(), ("tp", "cp"))
    assert len(b.out_plan) == 1, "case: attention_out_cp_identity_skipped"
    assert b.out_plan[0].collective_type == "reduce_scatter", \
        "case: attention_out_cp_identity_skipped"

    # --- case: identity_output_plan_empty ---
    spec = ModuleShardingSpec(
        out_src={"output": {TP: Shard(1)}},
        out_dst={"output": {TP: Shard(1)}},
    )
    b = PrecompiledBoundary(spec, _FakeMesh(), ("tp",))
    assert not b.out_plan, "case: identity_output_plan_empty"

    # --- cases: multi_output_arg_index_from_out_names / _default_key_order ---
    for case, spec, expected in [
        (
            "multi_output_arg_index_from_out_names",
            ModuleShardingSpec(
                out_src={"hidden_states": {TP: Partial()}, "present_kv": {TP: Shard(1)}},
                out_dst={"hidden_states": {TP: Shard(1)}, "present_kv": {TP: Replicate()}},
                out_names=["hidden_states", "present_kv"],
            ),
            {"hidden_states": 0, "present_kv": 1},
        ),
        (
            "multi_output_arg_index_default_key_order",
            ModuleShardingSpec(
                out_src={"a": {TP: Partial()}, "b": {TP: Shard(1)}},
                out_dst={"a": {TP: Shard(1)}, "b": {TP: Replicate()}},
            ),
            {"a": 0, "b": 1},
        ),
    ]:
        b = PrecompiledBoundary(spec, _FakeMesh(), ("tp",))
        idx = {op.arg_name: op.arg_index for op in b.out_plan}
        assert idx == expected, f"case: {case}"

    # --- cases: out_src_none_no_out_plan / out_dst_none_no_out_plan ---
    for case, spec in [
        (
            "out_src_none_no_out_plan",
            ModuleShardingSpec(out_src=None, out_dst={"output": {TP: Shard(1)}}),
        ),
        (
            "out_dst_none_no_out_plan",
            ModuleShardingSpec(out_src={"output": {TP: Partial()}}, out_dst=None),
        ),
    ]:
        b = PrecompiledBoundary(spec, _FakeMesh(), ("tp",))
        assert not b.out_plan, f"case: {case}"


# ==========================================================================
# Source: test_s2_boundary_io.py
# S2.5: _get_arg/_set_arg dual channels (pure-python, no process group).
# The real-process-group redistribute IO cases moved to Gate-2:
# tests/torch/auto_models/distributed/_test_precompiled_boundary_gloo.py
# ==========================================================================

def test_get_set_arg():
    """_get_arg/_set_arg family: positional/keyword dual-channel read and write-back."""
    # --- case: kwargs_hit_priority ---
    args, kwargs = (torch.tensor([1]),), {"x": torch.tensor([2])}
    got = _get_arg(args, kwargs, "x", 0)
    assert got.item() == 2, "case: kwargs_hit_priority"  # kwargs takes priority over args[0]

    # --- case: args_idx_fallback ---
    args, kwargs = (torch.tensor([1]),), {}
    got = _get_arg(args, kwargs, "x", 0)
    assert got.item() == 1, "case: args_idx_fallback"

    # --- case: missing_returns_default ---
    assert _get_arg((), {}, "x", None, default="d") == "d", \
        "case: missing_returns_default"

    # --- case: idx_out_of_range_falls_to_kwargs ---
    args, kwargs = (), {}
    args, kwargs = _set_arg(args, kwargs, "x", 5, "v")
    assert kwargs == {"x": "v"}, "case: idx_out_of_range_falls_to_kwargs"

    # --- case: set_arg_kwargs_channel ---
    args, kwargs = (), {"x": 1}
    args, kwargs = _set_arg(args, kwargs, "x", None, 2)
    assert kwargs["x"] == 2, "case: set_arg_kwargs_channel"

    # --- case: set_arg_args_channel ---
    args, kwargs = (1, 2), {}
    args, kwargs = _set_arg(args, kwargs, "x", 1, 9)
    assert args == (1, 9), "case: set_arg_args_channel"


# ==========================================================================
# Source: test_s2_path_utils.py
# S2.1: path utils _resolve_module / _get_attr_by_path / _set_param_by_path.
# ==========================================================================

class _Net(nn.Module):  # pylint: disable=abstract-method
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.lm_head = nn.Linear(4, 4)


def test_path_utils():
    """Path utils family: _resolve_module / _get_attr_by_path / _set_param_by_path."""
    # --- case: resolve_module_nested_modulelist ---
    net = _Net()
    assert _resolve_module(net, "model.layers.0") is net.model.layers[0], \
        "case: resolve_module_nested_modulelist"
    assert _resolve_module(net, "model.layers.1") is net.model.layers[1], \
        "case: resolve_module_nested_modulelist"
    assert _resolve_module(net, "lm_head") is net.lm_head, \
        "case: resolve_module_nested_modulelist"

    # --- case: resolve_module_no_leaf_strip ---
    # no leaf stripping: passing a module FQN returns the module itself, not its parent.
    mod = _resolve_module(net, "model.layers.0")
    assert isinstance(mod, nn.Linear), "case: resolve_module_no_leaf_strip"

    # --- case: get_attr_by_path_param ---
    w = _get_attr_by_path(net, "model.layers.0.weight")
    assert w is net.model.layers[0].weight, "case: get_attr_by_path_param"

    # --- case: set_param_by_path_register_parameter ---
    new_w = nn.Parameter(torch.ones(4, 4))
    _set_param_by_path(net, "model.layers.1.weight", new_w)
    assert net.model.layers[1].weight is new_w, "case: set_param_by_path_register_parameter"
    # register_parameter path: present in _parameters
    assert net.model.layers[1]._parameters["weight"] is new_w, \
        "case: set_param_by_path_register_parameter"  # pylint: disable=protected-access

    # --- case: set_param_by_path_setattr_branch ---
    class Plain:
        pass
    obj = Plain()
    p = nn.Parameter(torch.ones(2))
    _set_param_by_path(obj, "w", p)
    assert obj.w is p, "case: set_param_by_path_setattr_branch"  # pylint: disable=no-member

    # --- case: set_param_by_path_numeric_segment ---
    new_w = nn.Parameter(torch.zeros(4, 4))
    _set_param_by_path(net, "model.layers.0.bias",
                       nn.Parameter(torch.zeros(4)))
    _set_param_by_path(net, "model.layers.0.weight", new_w)
    assert net.model.layers[0].weight is new_w, "case: set_param_by_path_numeric_segment"


# ==========================================================================
# Source: test_s2_source_shard_info.py
# S2.9: build_source_shard_info + tied normalization (single process, mock mesh).
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


def test_source_shard_info():
    """Source shard info family: read from plan, FSDP-owned axis, EP extension."""
    # --- case: reads_from_plan_not_dtensor ---
    tp_mesh = _FakeTpMesh()
    info = build_source_shard_info(_plan(), tp_mesh)
    assert info["model.embed_tokens.weight"] == ((Shard(0),), tp_mesh), \
        "case: reads_from_plan_not_dtensor"
    assert info["model.layers.0.input_layernorm.weight"] == ((Replicate(),), tp_mesh), \
        "case: reads_from_plan_not_dtensor"
    assert info["lm_head.weight"] == ((Shard(0),), tp_mesh), \
        "case: reads_from_plan_not_dtensor"

    # --- case: no_tp_axis_mesh_none ---
    plan = ShardingPlan(mesh_dim_names=("cp",))
    plan.modules["m"] = ModuleShardingSpec(params={"w": {"cp": Replicate()}})
    info = build_source_shard_info(plan, None)
    # the cp axis is FSDP-owned -> source dims are empty; no placements to record
    assert info["m.w"][0] == (), "case: no_tp_axis_mesh_none"

    # --- case: tp_extend_ep_uses_real_expert_source_layout ---
    # Routed experts use Shard(0) on EP while dense parameters retain TP.
    plan = ShardingPlan(mesh_dim_names=("tp",))
    moe_spec = ModuleShardingSpec(
        params={
            "experts.gate_proj": {EP: Shard(0)},
            "gate.weight": {TP: Replicate()},
        }
    )
    moe_spec._ep_size = 4  # pylint: disable=protected-access
    plan.modules["model.layers.0.mlp"] = moe_spec
    dense_tp_mesh = _FakeTpMesh()
    expert_ep_mesh = _FakeTpMesh()
    info = build_source_shard_info(
        plan,
        dense_tp_mesh,
        expert_source_mesh=expert_ep_mesh,
    )
    assert info["model.layers.0.mlp.experts.gate_proj"] == (
        (Shard(0),),
        expert_ep_mesh,
    ), "case: tp_extend_ep_uses_real_expert_source_layout"
    assert info["model.layers.0.mlp.gate.weight"] == (
        (Replicate(),),
        dense_tp_mesh,
    ), "case: tp_extend_ep_uses_real_expert_source_layout"


def test_tied_weights_consistency():
    """Tied weights consistency family: tied normalization, ghost ignoring, explicit override."""
    # --- case: tied_consistent_placements_unchanged ---
    plan = _plan()
    plan.tied_pairs = [("model.embed_tokens.weight", "lm_head.weight")]
    info = build_source_shard_info(plan, _FakeTpMesh())
    assert info["model.embed_tokens.weight"][0] == (Shard(0),), \
        "case: tied_consistent_placements_unchanged"
    assert info["lm_head.weight"][0] == (Shard(0),), \
        "case: tied_consistent_placements_unchanged"

    # --- case: tied_inconsistent_shard_wins ---
    # inconsistent placements in a tied pair -> Shard wins.
    plan = _plan()
    plan.modules["lm_head"] = ModuleShardingSpec(
        params={"weight": {TP: Replicate()}})
    plan.tied_pairs = [("model.embed_tokens.weight", "lm_head.weight")]
    info = build_source_shard_info(plan, _FakeTpMesh())
    assert info["model.embed_tokens.weight"][0] == (Shard(0),), \
        "case: tied_inconsistent_shard_wins"
    assert info["lm_head.weight"][0] == (Shard(0),), \
        "case: tied_inconsistent_shard_wins"  # normalized to Shard

    # --- case: tied_pair_not_in_plan_ignored ---
    plan = _plan()
    plan.tied_pairs = [("ghost.a", "ghost.b")]
    info = build_source_shard_info(plan, _FakeTpMesh())
    assert "ghost.a" not in info, "case: tied_pair_not_in_plan_ignored"

    # --- case: explicit_tied_pairs_override ---
    plan = _plan()
    info = build_source_shard_info(
        plan, _FakeTpMesh(),
        tied_pairs=[("model.embed_tokens.weight", "lm_head.weight")])
    assert info["lm_head.weight"][0] == (Shard(0),), \
        "case: explicit_tied_pairs_override"


def test_error_paths():
    """Error paths family: fail-fast raises for source shard contract violations."""
    # --- case: fsdp_owned_axis_shard_rejected ---
    # weight declared Shard on an FSDP-owned axis (e.g. cp) -> fail-fast.
    plan = ShardingPlan(mesh_dim_names=("cp", "tp"))
    plan.modules["m"] = ModuleShardingSpec(
        params={"w": {"cp": Shard(1), TP: Shard(0)}})
    with pytest.raises(ValueError, match="conflicts with FSDP ownership"):
        build_source_shard_info(plan, _FakeTpMesh())

    # --- case: tp_extend_ep_requires_expert_source_mesh ---
    # Routed-expert metadata requires the derived EP child mesh.
    plan = ShardingPlan(mesh_dim_names=("tp",))
    moe_spec = ModuleShardingSpec(
        params={"experts.gate_proj": {EP: Shard(0)}}
    )
    moe_spec._ep_size = 4  # pylint: disable=protected-access
    plan.modules["model.layers.0.mlp"] = moe_spec
    with pytest.raises(
        ValueError,
        match="Routed expert metadata requires an expert EP source mesh",
    ):
        build_source_shard_info(plan, _FakeTpMesh())
