# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Contract tests for iterable placements emitted by ``hyper_build_tp_grad_info``.

Regression test for the crash seen on the first real codegen-arm run (gate
open): the runtime built ``source_shard_info`` entries as ``(Shard(0), mesh)``
— a *bare* placement — but ``FSDP2._build_source_shard_info_by_param`` iterates
the placements (``any(isinstance(p, Partial) for p in placements)``), blowing up
with ``"TypeError: 'Shard' object is not iterable"``.  The native
``build_source_shard_info`` records one placement per source (non-FSDP) mesh
axis as a tuple, so the runtime must mirror that contract: a ``placements``
value that is always a tuple, resolved over the source mesh's non-FSDP dim
names via ``resolve_placements``.

This test locks the contract shape (item 0 of every ``source_shard_info`` value
is iterable and non-``Partial``) for dense and routed-expert parameters, plus
the tied-pair normalization path.
"""

from __future__ import annotations

import torch  # noqa: F401  (runtime helpers are lazy, but the function runs torch-side)

import hyper_parallel.codegen.runtime as runtime
from hyper_parallel.core.dtensor.placement_types import Partial


class _FakeMesh:
    """A minimal mesh carrying ``mesh_dim_names``, like a TP/EP child mesh."""

    mesh_shape = (2,)

    def __init__(self, dims):
        self.mesh_dim_names = tuple(dims)

    def __getitem__(self, key):
        return self

    def __repr__(self):  # pragma: no cover - debug only
        return f"_FakeMesh({self.mesh_dim_names!r})"


def _build(*args, **kwargs):
    """Call ``hyper_build_tp_grad_info`` with the mesh-resolution hook stubbed.

    We only want to exercise the tuple-building logic, so the meshes are
    hand-supplied rather than resolved from a live ``MeshContext``/device mesh.
    """
    import hyper_parallel.distributed._builder.parameter_sharding as apply_mod

    dense_src = _FakeMesh(("tp",))
    expert_src = _FakeMesh(("ep",))

    orig_local = apply_mod._local_params_context
    orig_resolve = runtime._resolve_runtime_meshes
    try:
        apply_mod._local_params_context = lambda model: None
        runtime._resolve_runtime_meshes = lambda mesh_context, mesh_dim_names=None: (
            _FakeMesh(("cp", "tp")), dense_src, expert_src, ("cp", "tp"),
        )
        return runtime.hyper_build_tp_grad_info(*args, **kwargs)
    finally:
        apply_mod._local_params_context = orig_local
        runtime._resolve_runtime_meshes = orig_resolve


def test_dense_and_expert_placements_are_iterable_tuples():
    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.weight = torch.nn.Parameter(torch.randn(4))
    model.mlp.gate_up_proj = torch.nn.Parameter(torch.randn(4))

    plan = {
        "model.layers.0.mlp": {
            "params": {
                "weight": {"tp": "S(0)"},
                "experts.gate_up_proj": {"ep": "S(0)"},
            },
        },
    }
    info = _build(model, plan, None, tied_pairs=[])
    assert info
    for full_fqn, (placements, mesh) in info.items():
        assert hasattr(placements, "__iter__"), f"{full_fqn}: placements not iterable"
        assert not any(isinstance(p, Partial) for p in placements), full_fqn
        assert mesh.mesh_dim_names in (("tp",), ("ep",)), mesh.mesh_dim_names
    assert info["model.layers.0.mlp.weight"][0] == (info["model.layers.0.mlp.weight"][0][0],)
    assert info["model.layers.0.mlp.experts.gate_up_proj"][0] == (
        info["model.layers.0.mlp.experts.gate_up_proj"][0][0],
    )


def test_tied_pair_normalization_keeps_tuples():
    model = torch.nn.Module()
    model.embed = torch.nn.Module()
    model.embed.weight = torch.nn.Parameter(torch.randn(4))
    model.lm_head = torch.nn.Module()
    model.lm_head.weight = torch.nn.Parameter(torch.randn(4))

    plan = {
        "embed": {"params": {"weight": {"tp": "S(0)"}}},
        "lm_head": {"params": {"weight": {"tp": "S(0)"}}},
    }
    info = _build(model, plan, None, tied_pairs=[("embed.weight", "lm_head.weight")])
    assert info
    pa, _ = info["embed.weight"]
    pb, _ = info["lm_head.weight"]
    assert hasattr(pa, "__iter__") and hasattr(pb, "__iter__")
    assert pa == pb
