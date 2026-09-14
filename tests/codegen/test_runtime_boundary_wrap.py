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
"""Regression tests for generated runtime instance-level boundary wrapping."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hyper_parallel.codegen import runtime
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


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


def test_runtime_wraps_unlowered_boundary(monkeypatch):
    model = Root()
    calls = []

    def fake_redistribute(payload, entry, mesh_context, mesh_dim_names=None, *, module=None):
        calls.append((payload, entry, mesh_context, mesh_dim_names, module))
        if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[1], dict):
            return (("input-redist",), {})
        return ("output-redist", payload)

    monkeypatch.setattr(runtime, "hyper_redistribute", fake_redistribute)
    runtime.hyper_wrap_module_boundaries(
        model,
        {"embed_tokens": _entry()},
        mesh_context="mesh",
        mesh_dim_names=("tp",),
    )

    assert getattr(model.embed_tokens, "_hyper_codegen_boundary_wrapped") is True
    assert model.embed_tokens("original") == ("output-redist", ("compute", "input-redist"))
    assert len(calls) == 2
    assert calls[0][4] is model.embed_tokens
    assert calls[1][4] is None


def test_runtime_does_not_double_wrap_lowered_forward(monkeypatch):
    model = Root()
    monkeypatch.setattr(runtime, "hyper_redistribute", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))

    runtime.hyper_wrap_module_boundaries(
        model,
        {"self_attn": _entry()},
        mesh_context="mesh",
        mesh_dim_names=("tp",),
    )

    assert not hasattr(model.self_attn, "_hyper_codegen_boundary_wrapped")
    assert model.self_attn(torch.tensor(1)) == ("lowered", torch.tensor(1))


def test_runtime_wraps_vocab_parallel_embedding_before_boundary(monkeypatch):
    model = nn.Module()
    model.embed_tokens = nn.Embedding(4, 2)
    with torch.no_grad():
        model.embed_tokens.weight.copy_(
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        )

    local_weight = model.embed_tokens.weight[:2]
    model.embed_tokens.weight.to_local = lambda: local_weight

    def identity_redistribute(payload, *_args, **_kwargs):
        return payload

    monkeypatch.setattr(runtime, "hyper_redistribute", identity_redistribute)
    mesh = FakeDenseMesh()
    runtime.hyper_wrap_module_boundaries(
        model,
        {"embed_tokens": _embedding_entry()},
        mesh_context=mesh,
        mesh_dim_names=("tp",),
    )

    input_ids = torch.tensor([0, 2, 3])
    expected = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    assert torch.equal(model.embed_tokens(input_ids), expected)
    assert model.embed_tokens._hyper_codegen_vocab_parallel_wrapped is True
    assert model.embed_tokens._hyper_codegen_boundary_wrapped is True

    runtime.hyper_wrap_module_boundaries(
        model,
        {"embed_tokens": _embedding_entry()},
        mesh_context=mesh,
        mesh_dim_names=("tp",),
    )
    assert torch.equal(model.embed_tokens(input_ids), expected)


def test_runtime_builds_default_position_ids_before_embedding_sequence_shard(monkeypatch):
    """Use the pre-reduce-scatter input length while preserving explicit ids."""
    model = PositionRoot()
    model.model.embed_tokens.weight.to_local = lambda: model.model.embed_tokens.weight[:2]
    monkeypatch.setattr(runtime, "hyper_redistribute", lambda payload, *_args, **_kwargs: payload)

    runtime.hyper_wrap_module_boundaries(
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
    assert model.model._hyper_codegen_position_ids_wrapped is True


def test_runtime_does_not_wrap_default_position_ids_without_sequence_shard(monkeypatch):
    """Leave the model entrance unchanged for a non-SP embedding contract."""
    model = PositionRoot()
    model.model.embed_tokens.weight.to_local = lambda: model.model.embed_tokens.weight[:2]
    entry = _embedding_entry()
    entry["out_dst"] = {"output": {"tp": "R"}}
    monkeypatch.setattr(runtime, "hyper_redistribute", lambda payload, *_args, **_kwargs: payload)

    runtime.hyper_wrap_module_boundaries(
        model,
        {"model.embed_tokens": entry},
        mesh_context=FakeDenseMesh(),
        mesh_dim_names=("tp",),
    )

    assert not hasattr(model.model, "_hyper_codegen_position_ids_wrapped")
    assert model.model(torch.tensor([[0, 1]]))[0] is None


def test_rewrap_parses_three_axis_frozen_contract_for_two_axis_mesh(monkeypatch):
    """Ignore absent EP while rewrapping a TP/CP local MoE output."""

    class FakeMesh:
        mesh_dim_names = ("tp", "cp")

        def __getitem__(self, _name):
            return self

    captured = {}

    def fake_from_local(tensor, mesh, placements):
        captured.update(tensor=tensor, mesh=mesh, placements=placements)
        return "wrapped"

    monkeypatch.setattr(DTensor, "from_local", staticmethod(fake_from_local))
    entry = {
        "out_src": {"output": {"cp": "S(1)", "ep": "R", "tp": "S(1)"}},
        "out_names": ["output"],
    }
    local = torch.randn(2, 16, 8)

    output = runtime.hyper_rewrap_outputs(local, entry, FakeMesh(), ("tp", "cp"))

    assert output == "wrapped"
    assert captured["tensor"] is local
    assert captured["placements"] == (Shard(1), Shard(1))
