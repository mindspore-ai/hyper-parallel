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
routing (empty-active-axes no-op, ep-entry expert-mesh compile, dense-mesh
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
    without active axes an ep-keyed entry compiles on the expert mesh while
    everything else is an exact no-op passthrough.

    Feature: boundary-install
    Description: Three routing branches are exercised: (1) active axes
        present → dense-mesh compile even for ep-keyed entries; (2) no active
        axes and no expert mesh → exact no-op passthrough; (3) no active axes
        but expert mesh with ep keys → expert-mesh compile.
    Expectation: Compiled-boundary mesh/dims match hyper_redistribute's routing
        in all three branches; no-op passthrough returns payloads unchanged.
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
        "in_src": {"input": {"cp": "S(1)", "ep": "S(0)", "tp": "S(1)"}},
        "in_dst": {"input": {"cp": "S(1)", "ep": "S(0)", "tp": "S(1)"}},
        "out_src": {"output": {"cp": "S(1)", "ep": "S(0)", "tp": "S(1)"}},
        "out_dst": {"output": {"cp": "S(1)", "ep": "S(0)", "tp": "S(1)"}},
    }

    # Active axes present: dense compile, even for an ep-keyed entry.
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

    # No active axes but an expert mesh and ep keys: expert-mesh compile.
    compiled.clear()
    state = {"active": (), "expert": expert}
    model = Root()
    runtime.hyper_install_boundaries(
        model, {"embed_tokens": ep_entry}, "mesh", ()
    )
    assert compiled == [(expert, None)]


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
