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
"""Unit tests for the Gated DeltaNet CP inner wrapper."""

import torch

from hyper_parallel.distributed.context_parallel import wrappers as cp_wrappers
from hyper_parallel.distributed.recipe_spec import (
    MeshAxisName,
)
from hyper_parallel.distributed._builder.default_templates import TEMPLATES
from hyper_parallel.distributed._builder.planner import (
    ShardingPlanner,
)
from hyper_parallel.core.dtensor.placement_types import Shard


def causal_conv1d_fn(
        hidden_states=None, weight=None, bias=None, activation=None, *, x=None):
    """Fake Transformers Conv1d primitive used by the fake GDN forward."""
    del weight, bias, activation
    if hidden_states is None:
        hidden_states = x
    return hidden_states + 1


def torch_chunk_gated_delta_rule(
        query, key, value, *, g, beta, initial_state=None,
        output_final_state=False):
    """Fake Transformers GDN primitive used by the fake GDN forward."""
    del key, value, g, beta, initial_state, output_final_state
    return query + 2, None


def _hook_wrapper(forward_func):
    """Mimic the Transformers hook decorator without ``functools.wraps``."""
    def wrapped(self, *args, **kwargs):
        return forward_func(self, *args, **kwargs)

    return wrapped


class _FakeCPMesh:
    """Minimal active CP mesh required by wrapper validation."""

    @staticmethod
    def size():
        """Return an active two-rank CP size."""
        return 2


class FakeGatedDeltaNet:
    """Minimal GDN whose forward calls the two intercepted primitives."""

    def __init__(self):
        """Initialize the fake GDN head metadata."""
        self.num_v_heads = 2

    def forward(self, hidden_states, cache_params=None, **kwargs):
        """Run fake Conv1d and GDN rule primitives."""
        del cache_params, kwargs
        mixed = causal_conv1d_fn(hidden_states, torch.ones(1, 2))
        decay = torch.zeros(mixed.shape[:-1])
        beta = torch.zeros_like(decay)
        output, _ = torch_chunk_gated_delta_rule(
            mixed,
            mixed,
            mixed,
            g=decay,
            beta=beta,
            initial_state=None,
            output_final_state=False,
        )
        return output


class HookedGatedDeltaNet:
    """GDN variant using instance primitives below an opaque hook wrapper."""

    def __init__(self):
        """Initialize instance primitive attributes and head metadata."""
        self.num_v_heads = 2
        self.causal_conv1d_fn = causal_conv1d_fn
        self.chunk_gated_delta_rule = torch_chunk_gated_delta_rule

    @_hook_wrapper
    def forward(self, hidden_states, cache_params=None, **kwargs):
        """Run the same attribute primitive calls as Transformers 5.13."""
        del cache_params, kwargs
        mixed = self.causal_conv1d_fn(
            x=hidden_states,
            weight=torch.ones(1, 2),
        )
        decay = torch.zeros(mixed.shape[:-1])
        beta = torch.zeros_like(decay)
        output, _ = self.chunk_gated_delta_rule(
            mixed,
            mixed,
            mixed,
            g=decay,
            beta=beta,
            initial_state=None,
            output_final_state=False,
        )
        return output


def test_gdn_wrapper_intercepts_primitives_without_copying_forward(monkeypatch):
    """The wrapper delegates model semantics to the original GDN forward."""
    calls = {"conv": 0, "to_hp": 0, "to_cp": 0}

    def fake_cp_conv(original, cp_mesh, *args, **kwargs):
        del cp_mesh
        calls["conv"] += 1
        return original(*args, **kwargs)

    def fake_to_hp(query, key, value, decay, beta, cp_mesh):
        del cp_mesh
        calls["to_hp"] += 1
        return query, key, value, decay, beta

    def fake_to_cp(output, cp_mesh):
        del cp_mesh
        calls["to_cp"] += 1
        return output

    monkeypatch.setattr(cp_wrappers, "_gdn_cp_causal_conv1d", fake_cp_conv)
    monkeypatch.setattr(cp_wrappers, "_gdn_rule_cp_to_hp", fake_to_hp)
    monkeypatch.setattr(cp_wrappers, "_gdn_rule_hp_to_cp", fake_to_cp)

    module = FakeGatedDeltaNet()
    original_conv = causal_conv1d_fn
    original_rule = torch_chunk_gated_delta_rule
    cp_wrappers.gdn_ulysses_cp_wrapper(
        module,
        mesh=None,
        tp_mesh=None,
        cp_mesh=_FakeCPMesh(),
        ep_mesh=None,
    )

    inputs = torch.zeros(1, 2, 2, 1)
    output = module.forward(inputs)

    torch.testing.assert_close(output, inputs + 3)
    assert calls == {"conv": 1, "to_hp": 1, "to_cp": 1}
    assert globals()["causal_conv1d_fn"] is original_conv
    assert globals()["torch_chunk_gated_delta_rule"] is original_rule


def test_gdn_halo_exchange_uses_torch_collective_adapter(monkeypatch):
    """The rebased wrapper routes halo exchange through Torch collectives."""
    tail = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    calls = {}

    class FakeMesh:
        """Provide the two-rank mesh surface used by halo exchange."""

        rank_list = (0, 1)

        @staticmethod
        def size():
            """Return the active CP world size."""
            return 2

        @staticmethod
        def get_local_rank():
            """Place this test rank after rank zero."""
            return 1

        @staticmethod
        def get_group():
            """Return an opaque process-group token."""
            return "cp_group"

    def fake_all_to_all_single(input_tensor, input_splits, output_splits, group):
        """Capture the adapter contract and return the previous-rank halo."""
        calls.update(
            input_tensor=input_tensor,
            input_splits=input_splits,
            output_splits=output_splits,
            group=group,
        )
        return torch.full_like(tail, 7)

    monkeypatch.setattr(cp_wrappers.dist, "get_process_group_ranks", lambda _group: (0, 1))
    monkeypatch.setattr(
        cp_wrappers._collectives,
        "differentiable_all_to_all_single",
        fake_all_to_all_single,
    )

    result = cp_wrappers._exchange_previous_cp_halo(tail, FakeMesh())

    torch.testing.assert_close(result, torch.full_like(tail, 7))
    torch.testing.assert_close(calls["input_tensor"], tail[:0])
    assert calls["input_splits"] == [0, 0]
    assert calls["output_splits"] == [2, 0]
    assert calls["group"] == "cp_group"


def test_planner_recognizes_linear_attention_boundary():
    """The standard planner classifies a linear-attention module."""
    assert ShardingPlanner._explicit_boundary_type(  # pylint: disable=protected-access
        "model.layers.0.linear_attn",
        "linear_attn",
    ) == "linear_attention"


def test_linear_attention_nosp_contract_keeps_cp_sequence_sharded():
    """GDN boundaries retain their CP-local sequence without TP SP."""
    template = TEMPLATES["linear_attention"]
    contracts = (
        template.nosp_in_src["hidden_states"],
        template.nosp_in_dst["hidden_states"],
        template.nosp_out_src,
        template.nosp_out_dst,
    )
    for contract in contracts:
        assert contract[MeshAxisName.CP] == Shard(1)


def test_gdn_wrapper_resolves_hooked_instance_primitives(monkeypatch):
    """The wrapper supports the decorated Transformers 5.13 call structure."""
    monkeypatch.setattr(
        cp_wrappers,
        "_gdn_cp_causal_conv1d",
        lambda original, cp_mesh, *args, **kwargs: original(*args, **kwargs),
    )
    monkeypatch.setattr(
        cp_wrappers,
        "_gdn_rule_cp_to_hp",
        lambda query, key, value, decay, beta, cp_mesh: (
            query, key, value, decay, beta),
    )
    monkeypatch.setattr(
        cp_wrappers,
        "_gdn_rule_hp_to_cp",
        lambda output, cp_mesh: output,
    )

    module = HookedGatedDeltaNet()
    original_conv = module.causal_conv1d_fn
    original_rule = module.chunk_gated_delta_rule
    cp_wrappers.gdn_ulysses_cp_wrapper(
        module,
        mesh=None,
        tp_mesh=None,
        cp_mesh=_FakeCPMesh(),
        ep_mesh=None,
    )

    inputs = torch.zeros(1, 2, 2, 1)
    torch.testing.assert_close(module.forward(inputs), inputs + 3)
    assert module.causal_conv1d_fn is original_conv
    assert module.chunk_gated_delta_rule is original_rule
