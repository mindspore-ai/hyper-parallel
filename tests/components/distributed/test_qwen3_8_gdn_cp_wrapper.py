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
"""Unit tests for the Qwen3.8 Gated DeltaNet CP inner wrapper."""

import torch

from hyper_parallel.auto_models.components.distributed import cp_wrappers
from hyper_parallel.auto_models.components.distributed.sharding_planner import (
    ShardingPlanner,
)


def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None):
    """Fake Transformers Conv1d primitive used by the fake GDN forward."""
    del weight, bias, activation
    return hidden_states + 1


def torch_chunk_gated_delta_rule(
        query, key, value, *, g, beta, initial_state=None,
        output_final_state=False):
    """Fake Transformers GDN primitive used by the fake GDN forward."""
    del key, value, g, beta, initial_state, output_final_state
    return query + 2, None


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


def test_qwen3_8_gdn_wrapper_intercepts_primitives_without_copying_forward(monkeypatch):
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
    cp_wrappers.qwen3_8_gdn_ulysses_cp_wrapper(
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


def test_planner_recognizes_linear_attention_boundary():
    """The standard planner classifies the Qwen3.8 linear-attention module."""
    assert ShardingPlanner._explicit_boundary_type(  # pylint: disable=protected-access
        "model.layers.0.linear_attn",
        "linear_attn",
    ) == "linear_attention"
