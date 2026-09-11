# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Regression tests for install-time boundary compilation and binding.

The generated runtime compiles every frozen boundary once
(``hyper_install_boundaries``) and binds the ``InstalledBoundary`` to the
module instance; the generated forwards call
``self._hyper_boundary.redistribute_inputs/outputs`` with no per-call
re-resolution.  These tests lock the binding behavior and — critically — the
install-time mesh routing: it must match ``hyper_redistribute``'s per-call
routing (empty-active-axes no-op, changing-ep expert-mesh compile, dense-mesh
compile with ep keys dropped), because that routing is runtime-conditional
and must never be baked into generated source text.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hyper_parallel.codegen import runtime
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard
from tests.common.mark_utils import arg_mark


class Leaf(nn.Module):
    def forward(self, x):
        return ("compute", x)


class LoweredLeaf(nn.Module):
    def _forward_impl(self, x):
        return ("impl", x)

    def forward(self, x):
        return ("lowered", x)


class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = Leaf()
        self.self_attn = LoweredLeaf()


class FakeTpMesh:
    def size(self):
        return 2

    def get_local_rank(self):
        return 1


class FakeDenseMesh:
    mesh_dim_names = ("tp",)
    tp_mesh = FakeTpMesh()

    def __getitem__(self, name):
        if name != "tp":
            raise KeyError(name)
        return self.tp_mesh


class FakePlan:
    """Compiled-plan double with the ``in_plan`` list _bind_input_indices walks."""

    def __init__(self):
        self.in_plan = []


class PositionModel(nn.Module):
    """Minimal model that exposes the Transformers position-id call contract."""

    def __init__(self) -> None:
        """Create the embedding owned by the model entrance."""
        super().__init__()
        self.embed_tokens = nn.Embedding(4, 2)

    def forward(
        self,
        input_ids: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        use_cache: Any = None,
        **kwargs: Any,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Return received arguments so tests can inspect wrapper behavior."""
        del input_ids, past_key_values
        return position_ids, use_cache, kwargs


class PositionRoot(nn.Module):
    """Root matching the ``model.embed_tokens`` FQN used by causal LMs."""

    def __init__(self) -> None:
        """Create a nested model matching a causal LM module hierarchy."""
        super().__init__()
        self.model = PositionModel()


class TagBoundary:
    """InstalledBoundary double: tags each side so ordering is assertable."""

    last: "TagBoundary | None" = None

    def __init__(self, entry, mesh_context, mesh_dim_names=None, *, module=None):
        self.entry = entry
        self.mesh_context = mesh_context
        self.mesh_dim_names = mesh_dim_names
        self.module = module
        self.sides = []
        TagBoundary.last = self

    def redistribute_inputs(self, payload):
        self.sides.append("in")
        return (("input-redist",), {})

    def redistribute_outputs(self, outputs):
        self.sides.append("out")
        return ("output-redist", outputs)

    def rewrap_outputs(self, output):
        self.sides.append("rewrap")
        return ("rewrapped", output)


class PassthroughBoundary:
    """InstalledBoundary double: identity plan, payloads untouched."""

    def __init__(self, entry, mesh_context, mesh_dim_names=None, *, module=None):
        self.module = module

    def redistribute_inputs(self, payload):
        return payload

    def redistribute_outputs(self, outputs):
        return outputs

    def rewrap_outputs(self, output):
        return output


def _entry():
    return {
        "is_boundary": True,
        "in_src": {"input": {"tp": "R"}},
        "in_dst": {"input": {"tp": "R"}},
        "out_src": {"output": {"tp": "P(sum)"}},
        "out_dst": {"output": {"tp": "S(1)"}},
    }


def _embedding_entry():
    entry = _entry()
    entry["params"] = {"weight": {"tp": "S(0)"}}
    return entry


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_install_wraps_unlowered_boundary(monkeypatch):
    """A boundary with no TP lowering gets a full forward wrapper.

    Feature: boundary-install
    Description: When ``_compile_boundary`` cannot lower (returns identity),
        ``hyper_install_boundaries`` wraps the module's forward with the
        installed boundary's redistribute inputs/outputs.
    Expectation: The module is tagged ``_codegen_boundary_wrapped`` and calling
        it dispatches through input-redist → compute → output-redist.
    """
    model = Root()
    monkeypatch.setattr(runtime, "InstalledBoundary", TagBoundary)

    runtime.hyper_install_boundaries(
        model,
        {"embed_tokens": _entry()},
        mesh_context="mesh",
        mesh_dim_names=("tp",),
    )

    assert getattr(model.embed_tokens, "_codegen_boundary_wrapped") is True
    assert model.embed_tokens("original") == ("output-redist", ("compute", "input-redist"))
    installed = TagBoundary.last
    assert installed.sides == ["in", "out"]
    assert installed.module is model.embed_tokens


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_install_binds_lowered_forward_without_wrapping(monkeypatch):
    """A lowered boundary binds ``_hyper_boundary`` without extra wrapping.

    Feature: boundary-install
    Description: When the module already has a lowered forward (``_forward_impl``),
        only the ``_hyper_boundary`` attribute is bound; no wrapper is added.
    Expectation: No ``_codegen_boundary_wrapped`` tag; ``_hyper_boundary`` is set
        and the module's forward reads it for redistribute calls.
    """
    model = Root()
    monkeypatch.setattr(runtime, "InstalledBoundary", TagBoundary)

    runtime.hyper_install_boundaries(
        model,
        {"self_attn": _entry()},
        mesh_context="mesh",
        mesh_dim_names=("tp",),
    )

    assert not hasattr(model.self_attn, "_codegen_boundary_wrapped")
    assert model.self_attn(torch.tensor(1)) == ("lowered", torch.tensor(1))
    # The generated forward reads the compiled plan from this attribute.
    assert isinstance(model.self_attn._hyper_boundary, TagBoundary)
    assert model.self_attn._hyper_boundary.module is model.self_attn


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_install_binds_rewrap_for_region_boundaries(monkeypatch):
    """Region FQNs (``local_compute_fn``) also get ``module._hyper_rewrap``.

    Feature: boundary-install
    Description: For local-region boundaries, ``hyper_install_boundaries`` binds
        ``module._hyper_rewrap`` to the boundary's ``rewrap_outputs`` method so
        the generated forward can call it as a function.
    Expectation: ``_hyper_rewrap`` is callable and its underlying function matches
        ``_hyper_boundary.rewrap_outputs``.
    """
    model = Root()
    monkeypatch.setattr(runtime, "InstalledBoundary", TagBoundary)

    runtime.hyper_install_boundaries(
        model,
        {"self_attn": _entry()},
        mesh_context="mesh",
        mesh_dim_names=("tp",),
        injections=[{"match": "self_attn", "local_compute_fn": {"_target_": "some.fn"}}],
    )

    # The lowered forward calls ``self._hyper_rewrap(outputs)`` as a function,
    # so the bound method is installed (not the InstalledBoundary object —
    # that is not callable and would raise TypeError at runtime).  Two
    # ``obj.method`` accesses yield distinct bound-method objects, so compare
    # the underlying function instead.
    assert callable(model.self_attn._hyper_rewrap)
    assert (
        model.self_attn._hyper_rewrap.__func__
        is model.self_attn._hyper_boundary.rewrap_outputs.__func__
    )


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_install_wraps_vocab_parallel_embedding_before_boundary(monkeypatch):
    """Vocab-parallel embedding wrapper is installed before the boundary.

    Feature: boundary-install
    Description: For ``embed_tokens`` with a sharded weight, the install path
        first wraps the embedding for vocab-parallel lookup, then installs the
        boundary wrapper on top.
    Expectation: Both ``_codegen_vocab_parallel_wrapped`` and
        ``_codegen_boundary_wrapped`` are set; output matches the expected
        local slice; re-install is idempotent.
    """
    model = nn.Module()
    model.embed_tokens = nn.Embedding(4, 2)
    with torch.no_grad():
        model.embed_tokens.weight.copy_(
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        )

    local_weight = model.embed_tokens.weight[:2]
    model.embed_tokens.weight.to_local = lambda: local_weight

    monkeypatch.setattr(runtime, "InstalledBoundary", PassthroughBoundary)
    mesh = FakeDenseMesh()
    runtime.hyper_install_boundaries(
        model,
        {"embed_tokens": _embedding_entry()},
        mesh_context=mesh,
        mesh_dim_names=("tp",),
    )

    input_ids = torch.tensor([0, 2, 3])
    expected = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    assert torch.equal(model.embed_tokens(input_ids), expected)
    assert model.embed_tokens._codegen_vocab_parallel_wrapped is True
    assert model.embed_tokens._codegen_boundary_wrapped is True

    runtime.hyper_install_boundaries(
        model,
        {"embed_tokens": _embedding_entry()},
        mesh_context=mesh,
        mesh_dim_names=("tp",),
    )
    assert torch.equal(model.embed_tokens(input_ids), expected)


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_install_builds_default_position_ids_before_embedding_sequence_shard(monkeypatch):
    """Use the pre-reduce-scatter input length while preserving explicit ids.

    Feature: boundary-install
    Description: When the embedding output is sequence-sharded (S(1)→R), the
        install path wraps the model entrance to build default position_ids
        from the pre-scatter input length.
    Expectation: The default position_ids tensor matches ``arange(seq_len)``
        and the ``_codegen_position_ids_wrapped`` tag is set.
    """
    model = PositionRoot()
    model.model.embed_tokens.weight.to_local = lambda: model.model.embed_tokens.weight[:2]
    monkeypatch.setattr(runtime, "InstalledBoundary", PassthroughBoundary)

    runtime.hyper_install_boundaries(
        model,
        {"model.embed_tokens": _embedding_entry()},
        mesh_context=FakeDenseMesh(),
        mesh_dim_names=("tp",),
    )

    input_ids = torch.tensor([[0, 1, 2, 3]])
    generated, use_cache, forwarded = model.model(
        input_ids,
        use_cache=False,
        output_router_logits=False,
    )
    torch.testing.assert_close(generated, torch.arange(4).unsqueeze(0))
    assert use_cache is False
    assert forwarded == {"output_router_logits": False}
    explicit = torch.tensor([[7, 8, 9, 10]])
    returned, _, _ = model.model(input_ids, position_ids=explicit)
    assert returned is explicit
    positional_default, _, _ = model.model(input_ids, None)
    torch.testing.assert_close(positional_default, torch.arange(4).unsqueeze(0))

    class Cache:
        """Minimal Transformers cache exposing the consumed sequence length."""

        @staticmethod
        def get_seq_length() -> int:
            """Return the number of positions already consumed by the cache."""
            return 3

    cached, _, _ = model.model(input_ids, past_key_values=Cache())
    torch.testing.assert_close(cached, torch.arange(3, 7).unsqueeze(0))
    assert model.model._codegen_position_ids_wrapped is True


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_install_does_not_build_default_position_ids_without_sequence_shard(monkeypatch):
    """Leave the model entrance unchanged for a non-SP embedding contract.

    Feature: boundary-install
    Description: When the embedding out_dst is Replicate (no sequence shard),
        the install path must not inject a default position_ids wrapper.
    Expectation: No ``_codegen_position_ids_wrapped`` attribute is set and the
        model entrance forward runs unchanged.
    """
    model = PositionRoot()
    model.model.embed_tokens.weight.to_local = lambda: model.model.embed_tokens.weight[:2]
    entry = _embedding_entry()
    entry["out_dst"] = {"output": {"tp": "R"}}
    monkeypatch.setattr(runtime, "InstalledBoundary", PassthroughBoundary)

    runtime.hyper_install_boundaries(
        model,
        {"model.embed_tokens": entry},
        mesh_context=FakeDenseMesh(),
        mesh_dim_names=("tp",),
    )

    assert not hasattr(model.model, "_codegen_position_ids_wrapped")
    assert model.model(torch.tensor([[0, 1]]))[0] is None


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_install_routing_matches_hyper_redistribute(monkeypatch):
    """Lock the install-time routing against hyper_redistribute's branches.

    Runtime-conditional mesh routing must stay in ``hyper_install_boundaries``
    (never baked into source text): with active axes everything compiles on
    the dense mesh (ep keys are dropped later by ``resolve_placements``);
    without active axes an entry whose ``ep`` placement actually *changes*
    compiles on the expert mesh while everything else — including entries
    carrying only identity ``R -> R`` ep keys — is an exact no-op passthrough
    (the shared ``boundary_forms`` criterion; identity keys are the frozen
    plan's "not EP-dependent" marker, so a degenerate tp=cp=ep=1 topology
    prunes such boundaries to identity form while install is skipped).

    Feature: boundary-install
    Description: Four routing branches are exercised: (1) active axes
        present → dense-mesh compile even for an ep-changing entry; (2) no
        active axes and no expert mesh → exact no-op passthrough; (3) no
        active axes but expert mesh with a changing ep placement →
        expert-mesh compile; (4) expert mesh with only identity ep keys →
        no-op passthrough.
    Expectation: Compiled-boundary mesh/dims match hyper_redistribute's routing
        in all four branches; no-op passthrough returns payloads unchanged.
    """
    from hyper_parallel.distributed._builder import tp_collective_lowering

    monkeypatch.setattr(
        tp_collective_lowering, "create_tp_collective_lowerer", lambda mesh, dims: None
    )

    dense = object()
    expert = object()
    state = {"active": ("cp", "tp"), "expert": expert}

    def fake_resolve(mesh_context, mesh_dim_names=None):
        return dense, None, state["expert"], state["active"]

    monkeypatch.setattr(runtime, "_resolve_runtime_meshes", fake_resolve)
    compiled = []

    def fake_compile(entry, mesh, dims):
        compiled.append((mesh, dims))
        return FakePlan()

    monkeypatch.setattr(runtime, "_compile_boundary", fake_compile)

    ep_entry = {
        "is_boundary": True,
        "in_src": {"input": {"cp": "S(1)", "ep": "R", "tp": "S(1)"}},
        "in_dst": {"input": {"cp": "S(1)", "ep": "S(0)", "tp": "S(1)"}},
        "out_src": {"output": {"cp": "S(1)", "ep": "R", "tp": "S(1)"}},
        "out_dst": {"output": {"cp": "S(1)", "ep": "S(0)", "tp": "S(1)"}},
    }
    identity_ep_entry = {
        "is_boundary": True,
        "in_src": {"input": {"cp": "R", "ep": "R", "tp": "R"}},
        "in_dst": {"input": {"cp": "R", "ep": "R", "tp": "R"}},
        "out_src": {"output": {"cp": "R", "ep": "R", "tp": "R"}},
        "out_dst": {"output": {"cp": "R", "ep": "R", "tp": "R"}},
    }

    # Active axes present: dense compile, even for an ep-changing entry.
    model = Root()
    runtime.hyper_install_boundaries(
        model, {"embed_tokens": ep_entry}, "mesh", ("cp", "tp")
    )
    assert compiled == [(dense, ("cp", "tp"))]
    assert model.embed_tokens._codegen_boundary_wrapped is True

    # Pure DP/FSDP (no active axes, no expert mesh): exact no-op, no plan.
    compiled.clear()
    state = {"active": (), "expert": None}
    model = Root()
    runtime.hyper_install_boundaries(
        model, {"embed_tokens": ep_entry}, "mesh", ()
    )
    assert compiled == []
    installed = model.embed_tokens._hyper_boundary
    assert installed.noop is True
    payload = ((torch.tensor(1),), {})
    assert installed.redistribute_inputs(payload) == payload
    assert installed.redistribute_outputs("out") == "out"

    # No active axes but an expert mesh and a changing ep placement:
    # expert-mesh compile.
    compiled.clear()
    state = {"active": (), "expert": expert}
    model = Root()
    runtime.hyper_install_boundaries(
        model, {"embed_tokens": ep_entry}, "mesh", ()
    )
    assert compiled == [(expert, None)]

    # Expert mesh but only identity ep keys (R -> R): not EP-dependent, so
    # the boundary is an exact no-op passthrough even though ep > 1.
    compiled.clear()
    model = Root()
    runtime.hyper_install_boundaries(
        model, {"embed_tokens": identity_ep_entry}, "mesh", ()
    )
    assert compiled == []
    installed = model.embed_tokens._hyper_boundary
    assert installed.noop is True
    payload = ((torch.tensor(1),), {})
    assert installed.redistribute_inputs(payload) == payload
    assert installed.redistribute_outputs("out") == "out"


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_installed_rewrap_matches_hyper_rewrap_outputs(monkeypatch):
    """The install-time rewrap uses the same parsed plan as the per-call helper.

    Feature: boundary-install
    Description: ``InstalledBoundary.rewrap_outputs`` must parse ``out_src``
        placements identically to ``hyper_rewrap_outputs`` so that local region
        outputs are re-wrapped with the same DTensor layout.
    Expectation: Both paths produce the same ``(Shard(1), Shard(1))`` placements
        when given the same frozen entry and mesh.
    """
    from hyper_parallel.distributed._builder import tp_collective_lowering

    monkeypatch.setattr(
        tp_collective_lowering, "create_tp_collective_lowerer", lambda mesh, dims: None
    )

    class FakeMesh:
        mesh_dim_names = ("tp", "cp")

        def __getitem__(self, _name):
            return self

    captured = []

    def fake_from_local(tensor, mesh, placements):
        captured.append((tensor, mesh, placements))
        return "wrapped"

    monkeypatch.setattr(DTensor, "from_local", staticmethod(fake_from_local))
    entry = {
        "is_boundary": True,
        "out_src": {"output": {"cp": "S(1)", "ep": "R", "tp": "S(1)"}},
        "out_names": ["output"],
    }
    monkeypatch.setattr(
        runtime,
        "_resolve_runtime_meshes",
        lambda mesh_context, mesh_dim_names=None: (
            mesh_context, None, None, ("tp", "cp")
        ),
    )
    installed = runtime.InstalledBoundary(entry, FakeMesh(), ("tp", "cp"))
    local = torch.randn(2, 16, 8)

    installed.rewrap_outputs(local)
    runtime.hyper_rewrap_outputs(local, entry, FakeMesh(), ("tp", "cp"))

    assert captured[0][2] == (Shard(1), Shard(1))
    assert captured[0][2] == captured[1][2]


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_installed_redistribute_inputs_takes_pair_payload(monkeypatch):
    """Generated forwards call ``redistribute_inputs((args, kwargs))`` — one pair.

    The lowered forward builds ``((hidden_states,), {})`` and passes it as a
    single positional argument, mirroring ``hyper_redistribute``'s first arg.
    A signature drift to ``redistribute_inputs(args, kwargs)`` (two positional
    args) breaks every imported-class boundary forward at runtime with
    ``TypeError: missing 1 required positional argument: 'kwargs'``; this test
    pins the pair-payload contract so that regression cannot ship again.

    Feature: boundary-install
    Description: The emitted call site and the runtime signature must agree on
        a single pair-payload argument ``((args, kwargs))`` for
        ``redistribute_inputs``.
    Expectation: ``_render_input_redistribute`` emits one positional pair, and
        ``InstalledBoundary.redistribute_inputs`` accepts exactly one
        parameter named ``payload``.
    """
    import inspect

    from hyper_parallel.codegen.emit.parallel import _render_input_redistribute
    from hyper_parallel.codegen.astkit.index import FunctionInfo

    # The emitted call site passes one positional pair, not two args.
    func = FunctionInfo(
        name="forward", def_line=1, body_start=0, body_end=0,
        def_offset=0, param_names=["hidden_states"],
    )
    body = _render_input_redistribute(func)
    assert "redistribute_inputs(\n            ((hidden_states,), {}),\n        )" in body

    # The runtime signature accepts exactly that one pair.
    sig = inspect.signature(runtime.InstalledBoundary.redistribute_inputs)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    assert len(params) == 1
    assert params[0].name == "payload"


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_legacy_wrap_alias_still_covers_unlowered_boundaries(monkeypatch):
    """``hyper_wrap_module_boundaries`` delegates to the install entry.

    Feature: boundary-install
    Description: The legacy ``hyper_wrap_module_boundaries`` entry point must
        delegate to ``hyper_install_boundaries`` so that callers using the old
        API still get the compiled-boundary binding.
    Expectation: After calling the legacy wrapper, the module has
        ``_codegen_boundary_wrapped`` set and its forward dispatches through the
        installed boundary.
    """
    model = Root()
    monkeypatch.setattr(runtime, "InstalledBoundary", TagBoundary)

    runtime.hyper_wrap_module_boundaries(
        model,
        {"embed_tokens": _entry()},
        mesh_context="mesh",
        mesh_dim_names=("tp",),
    )

    assert getattr(model.embed_tokens, "_codegen_boundary_wrapped") is True
    assert model.embed_tokens("original") == ("output-redist", ("compute", "input-redist"))


# ---------------------------------------------------------------------------
# static tp_collective validation / fallback
# ---------------------------------------------------------------------------


class StaticLeaf(nn.Module):
    """Class-shaped ``tp_collective`` boundary: marker + static template body.

    The body mirrors what ``emit/parallel`` renders for an entry whose input
    side is identity (R -> R, rendered ``to_local``) and whose output side is
    a Partial -> Shard transition (rendered ``reduce_scatter``).
    """

    _hyper_boundary_form = "tp_collective"

    def _forward_impl(self, x):
        return ("impl", x)

    def forward(self, x):
        x = self._hyper_tp.to_local(x)
        outputs = self._forward_impl(x)
        outputs = self._hyper_tp.reduce_scatter(outputs, dim=1)
        return outputs


class StaticInstalledBoundary:
    """InstalledBoundary double exposing the routing facts the validator reads."""

    def __init__(self, entry, mesh_context, mesh_dim_names=None, *, module=None):
        self.entry = entry
        self.module = module
        self.dense_mesh = mesh_context
        self.active_dim_names = tuple(mesh_dim_names or ())
        self.sides = []

    def redistribute_inputs(self, payload):
        self.sides.append("in")
        return (("input-redist",), {})

    def redistribute_outputs(self, outputs):
        self.sides.append("out")
        return ("output-redist", outputs)

    def rewrap_outputs(self, output):
        return output


class RecordingLowerer:
    """Lowerer double recording bare-operator dispatch without a process group."""

    def __init__(self):
        self.calls = []

    def execution_op(self, kind, tensor_dim=None, reduce_op="sum"):
        self.calls.append((kind, tensor_dim))
        return self

    def execute(self, tensor):
        return ("executed", tensor)


def _static_entry():
    """A tp_collective entry: identity input, Partial -> Shard output."""
    return {
        "is_boundary": True,
        "in_src": {"x": {"tp": "R"}},
        "in_dst": {"x": {"tp": "R"}},
        "out_src": {"output": {"tp": "P(sum)"}},
        "out_dst": {"output": {"tp": "S(1)"}},
    }


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_tp_operators_dispatch_and_passthrough():
    """Bare operators dispatch through the lowerer; ``None`` passes through.

    Feature: boundary-install
    Description: ``TPOperators`` mirrors the compiled engine's execution per
        transition — identity inputs unwrap/pass through, ``None`` skips (the
        compiled plan's skip), and collectives dispatch through
        ``TPCollectiveLowerer.execution_op`` with the emitted kind and dim.
    Expectation: Every call records ``(kind, tensor_dim)`` on the lowerer and
        returns its executed result; ``None`` never reaches the lowerer.
    """
    lowerer = RecordingLowerer()
    ops = runtime.TPOperators(lowerer)

    assert ops.to_local(None) is None
    plain = object()
    assert ops.to_local(plain) is plain
    assert ops.all_gather(None, dim=1) is None
    assert ops.all_reduce("t") == ("executed", "t")
    assert ops.all_gather("t", dim=1) == ("executed", "t")
    assert ops.reduce_scatter("t", dim=-1) == ("executed", "t")
    assert lowerer.calls == [
        ("all_reduce", None),
        ("all_gather", 1),
        ("reduce_scatter", -1),
    ]


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_tp_operators_unwrap_dtensor_inputs(monkeypatch):
    """DTensor inputs unwrap to the local shard before any operator runs.

    Feature: boundary-install
    Description: ``RedistOp.execute`` unwraps a DTensor to its local shard
        before running the execution op; ``TPOperators`` must do the same for
        both the identity ``to_local`` and the collective operators.
    Expectation: ``to_local`` returns the shard, and a collective executes on
        the shard (not on the DTensor wrapper).
    """
    from hyper_parallel.core.dtensor import dtensor as dtensor_module

    class FakeDTensor:
        """DTensor double: only ``to_local`` is exercised."""

        def __init__(self, local):
            self._local = local

        def to_local(self):
            return self._local

    monkeypatch.setattr(dtensor_module, "DTensor", FakeDTensor)
    ops = runtime.TPOperators(RecordingLowerer())
    wrapped = FakeDTensor("local-shard")

    assert ops.to_local(wrapped) == "local-shard"
    assert ops.all_reduce(wrapped) == ("executed", "local-shard")


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_install_static_tp_collective_binds_operators_on_match(monkeypatch):
    """A form match binds ``_hyper_tp`` and leaves the static forward in place.

    Feature: boundary-install
    Description: When the emitted marker, the live-mesh re-classification,
        and the TP lowerer all agree, ``hyper_install_boundaries`` binds the
        bare-operator runtime (``module._hyper_tp``) and does NOT touch the
        generated static forward.
    Expectation: ``_hyper_tp`` is a ``TPOperators`` carrying the lowerer; the
        forward still resolves to the class's own method; calling it runs the
        bare-operator pipeline.
    """
    from hyper_parallel.distributed._builder import tp_collective_lowering

    lowerer = RecordingLowerer()
    monkeypatch.setattr(
        tp_collective_lowering,
        "create_tp_collective_lowerer",
        lambda mesh, dims: lowerer,
    )
    monkeypatch.setattr(runtime, "InstalledBoundary", StaticInstalledBoundary)

    model = nn.Module()
    model.self_attn = StaticLeaf()
    runtime.hyper_install_boundaries(
        model,
        {"self_attn": _static_entry()},
        mesh_context=FakeDenseMesh(),
        mesh_dim_names=("tp",),
    )

    assert isinstance(model.self_attn._hyper_tp, runtime.TPOperators)
    assert model.self_attn._hyper_tp._lowerer is lowerer
    # The static forward was not replaced (no instance-level forward).
    assert "forward" not in model.self_attn.__dict__
    assert model.self_attn("x") == ("executed", ("impl", "x"))


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_install_static_tp_collective_falls_back_when_lowerer_unavailable(
    monkeypatch,
):
    """No TP lowerer (rank order / backend) replaces the forward with generic.

    Feature: boundary-install
    Description: ``create_tp_collective_lowerer`` returns ``None`` when the tp
        rank order differs from the group or the backend lacks a collective;
        the baked ``self._hyper_tp`` calls would then crash, so the install
        path must replace the forward with the generic redistribute engine.
    Expectation: ``_hyper_tp`` is not bound; the instance carries a replaced
        forward that dispatches input-redist -> ``_forward_impl`` ->
        output-redist through the compiled plan.
    """
    from hyper_parallel.distributed._builder import tp_collective_lowering

    monkeypatch.setattr(
        tp_collective_lowering,
        "create_tp_collective_lowerer",
        lambda mesh, dims: None,
    )
    monkeypatch.setattr(runtime, "InstalledBoundary", StaticInstalledBoundary)

    model = nn.Module()
    model.self_attn = StaticLeaf()
    runtime.hyper_install_boundaries(
        model,
        {"self_attn": _static_entry()},
        mesh_context=FakeDenseMesh(),
        mesh_dim_names=("tp",),
    )

    assert not hasattr(model.self_attn, "_hyper_tp")
    assert "forward" in model.self_attn.__dict__
    assert model.self_attn("x") == ("output-redist", ("impl", "input-redist"))
    installed = model.self_attn._hyper_boundary
    assert installed.sides == ["in", "out"]


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_install_static_tp_collective_falls_back_when_live_axes_disagree(
    monkeypatch,
):
    """A live mesh without the plan's tp axis demotes the static form.

    Feature: boundary-install
    Description: The plan froze axes ``("tp",)`` (emitted verdict
        tp_collective), but the live mesh routes with no active axes — the
        live re-classification is identity, not tp_collective.  Even with a
        usable lowerer the emitted ops no longer describe the boundary, so
        the forward must fall back to the generic engine.
    Expectation: ``_hyper_tp`` is not bound; the forward is replaced.
    """
    from hyper_parallel.distributed._builder import tp_collective_lowering

    lowerer = RecordingLowerer()
    monkeypatch.setattr(
        tp_collective_lowering,
        "create_tp_collective_lowerer",
        lambda mesh, dims: lowerer,
    )

    module = StaticLeaf()
    installed = StaticInstalledBoundary(_static_entry(), FakeDenseMesh(), ())
    runtime._install_static_tp_operators(
        module, _static_entry(), installed, ("tp",), None
    )

    assert not hasattr(module, "_hyper_tp")
    assert "forward" in module.__dict__
    assert module("x") == ("output-redist", ("impl", "input-redist"))


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_install_leaves_unmarked_lowered_boundary_alone(monkeypatch):
    """A lowered boundary without the marker gets no tp operators, no rewrite.

    Feature: boundary-install
    Description: Generic / region lowered forwards (no
        ``_hyper_boundary_form`` marker) never enter the static validation —
        they already route through ``self._hyper_boundary``.
    Expectation: No ``_hyper_tp`` binding and no instance-level forward.
    """
    from hyper_parallel.distributed._builder import tp_collective_lowering

    monkeypatch.setattr(
        tp_collective_lowering,
        "create_tp_collective_lowerer",
        lambda mesh, dims: RecordingLowerer(),
    )

    module = LoweredLeaf()
    installed = StaticInstalledBoundary(_static_entry(), FakeDenseMesh(), ("tp",))
    runtime._install_static_tp_operators(
        module, _static_entry(), installed, ("tp",), None
    )

    assert not hasattr(module, "_hyper_tp")
    assert "forward" not in module.__dict__
