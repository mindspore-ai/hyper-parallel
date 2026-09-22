# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Contract tests for the iterable placements ``hyper_build_tp_grad_info`` returns.

Regression test for the crash seen on the first real codegen-arm run (gate
open): the codegen runtime built ``source_shard_info`` entries as
``(Shard(0), mesh)`` — a *bare* placement — but
``FSDP2._build_source_shard_info_by_param`` iterates the placements
(``any(isinstance(p, Partial) for p in placements)``), blowing up with
``"TypeError: 'Shard' object is not iterable"``.  The native
``build_source_shard_info`` records one placement per source (non-FSDP) mesh
axis as a tuple.

The codegen helper no longer re-implements that builder: it delegates to the
SAME function object the applier's ``apply_sharding_plan`` calls
(``parameter_sharding._build_runtime_source_shard_info``).  These tests pin the
delegation and the resulting contract for dense and routed-expert parameters,
plus the tied-pair normalization path.
"""

from __future__ import annotations

import torch

import hyper_parallel.codegen.runtime as runtime
import hyper_parallel.distributed._builder.parameter_sharding as parameter_sharding
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.distributed._builder.source_shard import build_source_shard_info
from hyper_parallel.distributed.plan import ShardingPlan
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec


class _FakeMesh:
    """A minimal mesh carrying ``mesh_dim_names``, like a TP/EP child mesh."""

    mesh_shape = (2,)

    def __init__(self, dims):
        self.mesh_dim_names = tuple(dims)

    def __getitem__(self, key):
        return self

    def __repr__(self):  # pragma: no cover - debug only
        return f"_FakeMesh({self.mesh_dim_names!r})"


DENSE_SOURCE_MESH = _FakeMesh(("tp",))
EXPERT_SOURCE_MESH = _FakeMesh(("ep",))


def _model() -> torch.nn.Module:
    """A tiny stand-in carrying the plan's dense and expert parameters."""
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    layer = model.layers[0]
    layer.mlp = torch.nn.Module()
    layer.mlp.weight = torch.nn.Parameter(torch.randn(4))
    layer.mlp.gate_up_proj = torch.nn.Parameter(torch.randn(4))
    return model


def _plan(*, ep_size: int, tied_pairs: list | None = None) -> ShardingPlan:
    """A live plan with one dense parameter and one routed-expert parameter."""
    spec = ModuleShardingSpec(
        params={
            "weight": {"tp": Shard(0)},
            "experts.gate_up_proj": {"ep": Shard(0)},
        },
    )
    spec._ep_size = ep_size  # pylint: disable=protected-access
    return ShardingPlan(
        modules={"model.layers.0.mlp": spec},
        mesh_dim_names=("tp",),
        tied_pairs=list(tied_pairs or []),
    )


def _build(model, plan, *, unwrapped: bool = True):
    """Call ``hyper_build_tp_grad_info`` with the parameter unwrap stubbed.

    The unwrap runs inside the native builder the helper delegates to; the fake
    model's parameters are plain tensors, so a non-empty record stands in for
    the DTensors a real Phase A leaves behind.
    """
    original = parameter_sharding._local_params_context  # pylint: disable=protected-access
    parameter_sharding._local_params_context = (  # pylint: disable=protected-access
        lambda model: {"weight": ()} if unwrapped else {}
    )
    try:
        return runtime.hyper_build_tp_grad_info(
            model, plan, DENSE_SOURCE_MESH, EXPERT_SOURCE_MESH
        )
    finally:
        parameter_sharding._local_params_context = original  # pylint: disable=protected-access


def test_codegen_path_delegates_to_the_native_builder(monkeypatch):
    """The codegen helper calls the native builder ``apply_sharding_plan`` calls."""
    calls = {}

    def fake_builder(models, plan, dense_source_mesh, expert_source_mesh, validate_mode):
        calls["args"] = (models, plan, dense_source_mesh, expert_source_mesh, validate_mode)
        return {"sentinel": "native"}

    monkeypatch.setattr(
        parameter_sharding, "_build_runtime_source_shard_info", fake_builder
    )
    plan = _plan(ep_size=2)
    model = _model()

    result = runtime.hyper_build_tp_grad_info(
        model, plan, DENSE_SOURCE_MESH, EXPERT_SOURCE_MESH
    )

    assert result == {"sentinel": "native"}
    assert calls["args"] == (
        [model], plan, DENSE_SOURCE_MESH, EXPERT_SOURCE_MESH, False,
    )


def test_dense_and_expert_placements_agree_with_native_builder():
    """Codegen and native metadata are identical for the same plan and meshes."""
    plan = _plan(ep_size=2)

    info = _build(_model(), plan)
    native = build_source_shard_info(
        plan, DENSE_SOURCE_MESH, expert_source_mesh=EXPERT_SOURCE_MESH
    )

    assert info == native
    assert set(info) == {
        "model.layers.0.mlp.weight",
        "model.layers.0.mlp.experts.gate_up_proj",
    }
    for full_fqn, (placements, mesh) in info.items():
        assert hasattr(placements, "__iter__"), f"{full_fqn}: placements not iterable"
        assert not any(isinstance(p, Partial) for p in placements), full_fqn
        assert mesh.mesh_dim_names in (("tp",), ("ep",)), mesh.mesh_dim_names
    assert info["model.layers.0.mlp.weight"] == ((Shard(0),), DENSE_SOURCE_MESH)
    assert info["model.layers.0.mlp.experts.gate_up_proj"] == (
        (Shard(0),),
        EXPERT_SOURCE_MESH,
    )


def test_experts_without_ep_size_stay_on_the_dense_source_mesh():
    """The native ``spec._ep_size > 0`` rule (not the ``experts.`` name) decides."""
    plan = _plan(ep_size=0)

    info = _build(_model(), plan)

    assert info["model.layers.0.mlp.experts.gate_up_proj"] == (
        (Replicate(),),
        DENSE_SOURCE_MESH,
    )


def test_no_unwrapped_parameter_yields_no_metadata():
    """Nothing to unwrap -> no source metadata, exactly like the native builder."""
    assert _build(_model(), _plan(ep_size=2), unwrapped=False) is None


def test_tied_pair_normalization_keeps_tuples():
    """Tied pairs share one placement, axis by axis (``Shard`` wins)."""
    embed = ModuleShardingSpec(params={"weight": {"tp": Shard(0)}})
    head = ModuleShardingSpec(params={"weight": {"tp": Replicate()}})
    plan = ShardingPlan(
        modules={"embed": embed, "lm_head": head},
        mesh_dim_names=("tp",),
        tied_pairs=[("embed.weight", "lm_head.weight")],
    )

    info = _build(_model(), plan)
    native = build_source_shard_info(
        plan, DENSE_SOURCE_MESH, expert_source_mesh=EXPERT_SOURCE_MESH
    )

    assert info == native
    pa, _ = info["embed.weight"]
    pb, _ = info["lm_head.weight"]
    assert hasattr(pa, "__iter__") and hasattr(pb, "__iter__")
    assert pa == (Shard(0),)
    assert pa == pb
