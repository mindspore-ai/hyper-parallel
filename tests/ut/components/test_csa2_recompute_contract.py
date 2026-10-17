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
"""Contracts that let a CSA2 attention layer be recomputed as a whole.

Three mechanisms make a recomputed attention legal, and this module exercises
them on CPU against the real classes:

* ``is_recomputing()`` phase guards keep ``publish_*`` from running twice.
  Those calls contain no differentiable op, so skipping them cannot change the
  checkpoint's saved-tensor count.
* the indexer's discrete selection is an exclusion region, because a recompute
  must neither choose a different Top-K set nor drop saved tensors.
* nothing drops a published entry early: a recomputed consumer reads its
  entries a second time during backward, so they stay alive until the graph
  releases the state.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest import mock

import torch
from torch import nn
from torch.utils.checkpoint import DefaultDeviceType

from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedAttentionCPContext,
    SharedCompressedAttentionState,
    SharedCompressedDSAAttentionBase,
    SharedCompressedDSAIndexer,
)
from hyper_parallel.core.activation_memory.api import checkpoint_wrapper
from hyper_parallel.models.deepseek_v41.adapter.policies import (
    activation_checkpointing as attention_policy,
)
from hyper_parallel.models.deepseek_v41.adapter.policies.shared_state import (
    attention_role,
)

HIDDEN, HEADS, HEAD_DIM, GROUPS, OUT_PER_GROUP = 8, 2, 8, 2, 4
BATCH, SEQUENCE, RATIO, LAYER = 1, 8, 2, 20
CP_SIZE = 2


class _CountingState(SharedCompressedAttentionState):
    """Record how often a layer writes into or drops from the shared state."""

    def __init__(self, **kwargs: Any) -> None:
        """Create the counters used by the assertions."""
        super().__init__(**kwargs)
        self.publish_count = 0
        self.topk_count = 0
        self.index_key_count = 0

    def publish_compressed_kv(self, source_layer: int, value: torch.Tensor) -> None:
        """Count one compressed K=V publication."""
        self.publish_count += 1
        super().publish_compressed_kv(source_layer, value)

    def publish_topk_indices(self, source_layer: int, value: torch.Tensor) -> None:
        """Count one Top-K publication."""
        self.topk_count += 1
        super().publish_topk_indices(source_layer, value)

    def publish_index_key(self, source_layer: int, value: torch.Tensor) -> None:
        """Count one indexer-K publication."""
        self.index_key_count += 1
        super().publish_index_key(source_layer, value)


class _Compressor(nn.Module):
    """Compress along the sequence dim so a KV source produces a shorter K=V."""

    def __init__(self) -> None:
        """Create the compressor projection."""
        super().__init__()
        self.wkv = nn.Linear(HIDDEN, HEAD_DIM, bias=False)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the latent and the compressed K=V."""
        del position_embeddings
        pooled = hidden_states.reshape(BATCH, SEQUENCE // RATIO, RATIO, HIDDEN).mean(dim=2)
        compressed = self.wkv(pooled)
        return compressed, compressed


class _Source(nn.Module):
    """Duck-typed attention source carrying the CSA2 structural contract."""

    def __init__(self) -> None:
        """Describe the CSA2 structural contract this layer exposes."""
        super().__init__()
        self.q_a_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.q_a_norm = nn.LayerNorm(HIDDEN, bias=False)
        self.q_b_proj = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.kv_proj = nn.Linear(HIDDEN, HEAD_DIM, bias=False)
        self.kv_norm = nn.LayerNorm(HEAD_DIM, bias=False)
        self.o_a_proj = nn.Linear(GROUPS * OUT_PER_GROUP, HIDDEN, bias=False)
        self.o_b_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.sinks = nn.Parameter(torch.zeros(HEADS))
        self.config = SimpleNamespace(o_groups=GROUPS, qk_rope_head_dim=4)
        self.layer_idx = LAYER
        self.num_heads = HEADS
        self.num_key_value_groups = 1
        self.compress_ratio = RATIO
        self.is_kv_source = True
        self.is_index_source = False
        self.kv_source_layer_idx = LAYER
        self.index_source_layer_idx = LAYER
        self.candidate_source_layer_idx = LAYER
        self.head_dim = HEAD_DIM
        self.sliding_window = 4
        self.scaling = HEAD_DIM ** -0.5


class _IndexerSource(nn.Module):
    """Duck-typed indexer source for a Reindex-style layer."""

    def __init__(self) -> None:
        """Create the projections a Reindex-style indexer needs."""
        super().__init__()
        self.q_b_proj = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.weights_proj = nn.Linear(HIDDEN, HEADS, bias=False)
        self.wk = nn.Linear(HIDDEN, HEAD_DIM, bias=False)
        self.k_norm = nn.LayerNorm(HEAD_DIM, bias=False)
        self.compress_ratio = RATIO
        self.num_heads = HEADS
        self.head_dim = HEAD_DIM
        self.index_topk = 2
        self.owns_key = False
        self.is_candidate_source = False
        self.uses_candidates = False
        self.candidate_topk_blocks = 1
        self.candidate_block_size = 2
        self.loss_coeff = 0.0


class _IndexSource(nn.Module):
    """Attention source whose layer consumes shared state and republishes Top-K."""

    def __init__(self) -> None:
        """Describe the Reindex-style layer this source stands for."""
        super().__init__()
        self.q_a_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.q_a_norm = nn.LayerNorm(HIDDEN, bias=False)
        self.q_b_proj = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.kv_proj = nn.Linear(HIDDEN, HEAD_DIM, bias=False)
        self.kv_norm = nn.LayerNorm(HEAD_DIM, bias=False)
        self.o_a_proj = nn.Linear(GROUPS * OUT_PER_GROUP, HIDDEN, bias=False)
        self.o_b_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.sinks = nn.Parameter(torch.zeros(HEADS))
        self.config = SimpleNamespace(o_groups=GROUPS, qk_rope_head_dim=4)
        self.layer_idx = LAYER
        self.num_heads = HEADS
        self.num_key_value_groups = 1
        self.compress_ratio = RATIO
        self.is_kv_source = False
        self.is_index_source = True
        self.kv_source_layer_idx = LAYER
        self.index_source_layer_idx = LAYER
        self.candidate_source_layer_idx = LAYER
        self.head_dim = HEAD_DIM
        self.sliding_window = 4
        self.scaling = HEAD_DIM ** -0.5


class _GatherCounter:
    """Stand-in for the CP rank-ordered all-gather that counts its calls."""

    def __init__(self) -> None:
        """Start with an empty gather record."""
        self.calls = []

    def gather(self, tensor: torch.Tensor, sequence_dim: int) -> torch.Tensor:
        """Return a two-rank stand-in gather and record the requested shape."""
        self.calls.append(tuple(tensor.shape))
        return torch.cat((tensor, tensor), dim=sequence_dim)

    def launch(self, tensor: torch.Tensor, sequence_dim: int) -> Any:
        """Return a handle whose ``wait`` yields the stand-in gather."""
        gathered = self.gather(tensor, sequence_dim)
        return SimpleNamespace(wait=lambda: gathered)


def _run(layer: nn.Module, state: SharedCompressedAttentionState, gather_counter, inputs):
    """Run one forward/backward and return the loss and the input gradient."""
    hidden, rope = inputs
    kwargs = {"shared_attention_state": state}
    if gather_counter is not None:
        kwargs["shared_attention_cp_context"] = SharedCompressedAttentionCPContext(
            size=CP_SIZE,
            rank=0,
            gather_sequence=gather_counter.gather,
            launch_sequence=gather_counter.launch,
        )
    output, _ = layer(
        hidden,
        {"compress": (rope, rope), "main": (rope, rope)},
        torch.arange(SEQUENCE).unsqueeze(0),
        None,
        **kwargs,
    )
    loss = output.float().square().sum()
    loss.backward()
    return loss.item(), hidden.grad.clone()


def _kv_source_layer(use_cp, state):
    """Build a KV-source attention layer and the inputs for one run."""
    torch.manual_seed(11)
    layer = SharedCompressedDSAAttentionBase(
        _Source(), replace_indexer=False, use_fused_ops=False
    )
    layer.compressor = _Compressor()
    state.publish_topk_indices(LAYER, torch.zeros(BATCH, SEQUENCE, 2, dtype=torch.int32))
    return layer, _inputs(), _GatherCounter() if use_cp else None


def _index_source_layer(use_cp, state):
    """Build a Reindex-style layer that republishes the shared Top-K."""
    torch.manual_seed(11)
    source = _IndexSource()
    source.indexer = SharedCompressedDSAIndexer(_IndexerSource())
    layer = SharedCompressedDSAAttentionBase(
        source, replace_indexer=False, use_fused_ops=False
    )
    state.publish_compressed_kv(LAYER, torch.zeros(BATCH, SEQUENCE // RATIO, HEAD_DIM))
    state.publish_index_key(LAYER, torch.zeros(BATCH, SEQUENCE // RATIO, HEAD_DIM))
    return layer, _inputs(), _GatherCounter() if use_cp else None


def _inputs():
    """Return one deterministic (hidden_states, rope) pair."""
    torch.manual_seed(23)
    return (
        torch.randn(BATCH, SEQUENCE, HIDDEN, requires_grad=True),
        torch.randn(BATCH, SEQUENCE, 2),
    )


def _snapshot(state: _CountingState) -> dict[str, int]:
    """Return the current write counts of one counting state."""
    return {
        "publish_compressed_kv": state.publish_count,
        "publish_topk_indices": state.topk_count,
        "publish_index_key": state.index_key_count,
    }


def _delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Return the writes performed between two snapshots."""
    return {name: after[name] - before[name] for name in before}


def _checkpointed_run(layer: nn.Module, state: SharedCompressedAttentionState, gathers, inputs):
    """Run one checkpointed case with the checkpoint's device type pinned to the CPU.

    Every input of these cases is on the CPU, so the checkpoint falls back to
    torch's default device type. On a host that has an accelerator backend
    installed but no usable device, that fallback makes the recomputation build
    that device's autocast context and initialize the accelerator, which these
    CPU cases neither need nor can succeed at. Pin the type they actually use.
    """
    previous = DefaultDeviceType.get_device_type()
    DefaultDeviceType.set_device_type("cpu")
    try:
        return _run(checkpoint_wrapper(layer), state, gathers, inputs)
    finally:
        DefaultDeviceType.set_device_type(previous)


def _assert_recompute_matches_plain(builder, *, use_cp: bool) -> None:
    """Compare one role's plain and recomputed runs, including its state writes."""
    plain_state = _CountingState()
    plain_layer, plain_inputs, plain_gathers = builder(use_cp, plain_state)
    plain_before = _snapshot(plain_state)
    plain_loss, plain_grad = _run(plain_layer, plain_state, plain_gathers, plain_inputs)
    plain_writes = {name: count for name, count in _delta(plain_before, _snapshot(plain_state)).items() if count}

    recomputed_state = _CountingState()
    recomputed_layer, recomputed_inputs, recomputed_gathers = builder(use_cp, recomputed_state)
    recomputed_before = _snapshot(recomputed_state)
    recomputed_loss, recomputed_grad = _checkpointed_run(
        recomputed_layer, recomputed_state, recomputed_gathers, recomputed_inputs
    )
    recomputed_after = _snapshot(recomputed_state)
    recomputed_writes = {
        name: count for name, count in _delta(recomputed_before, recomputed_after).items() if count
    }

    assert recomputed_loss == plain_loss
    assert torch.equal(recomputed_grad, plain_grad)
    assert plain_writes == recomputed_writes
    # Every write the plain pass performs must happen exactly once under
    # recompute: the phase guards keep the publishes from running twice.
    assert plain_writes and all(value == 1 for value in plain_writes.values())
    if use_cp:
        # The CP gather is pure, so the recompute issues it again with the same shapes.
        assert len(recomputed_gathers.calls) == 2 * len(plain_gathers.calls)
        assert recomputed_gathers.calls[: len(plain_gathers.calls)] == plain_gathers.calls


def test_checkpointed_kv_source_matches_plain_and_writes_state_once():
    """A recomputed KV source reproduces the plain result and writes state once."""
    for use_cp in (False, True):
        _assert_recompute_matches_plain(_kv_source_layer, use_cp=use_cp)


def test_checkpointed_index_source_matches_plain_and_writes_state_once():
    """A recomputed index source republishes its Top-K exactly once."""
    for use_cp in (False, True):
        _assert_recompute_matches_plain(_index_source_layer, use_cp=use_cp)


class _FakeIndexer:
    """Minimal indexer that records whether the selection pin was installed."""

    def __init__(self) -> None:
        """Start unpinned until the model-owned hook decides otherwise."""
        self.selection_pinned = False

    def enable_selection_exclude(self) -> None:
        """Record that the hook pinned the discrete selection."""
        self.selection_pinned = True


class _FakeAttention(nn.Module):
    """Attention stand-in exposing the attributes the CSA2 roles read."""

    def __init__(self, layer_idx: int, *, kv_source: bool = False, index_source: bool = False) -> None:
        """Describe one layer stand-in of a CSA2 role."""
        super().__init__()
        self.compress_ratio = 1
        self.layer_idx = layer_idx
        self.is_kv_source = kv_source
        self.is_index_source = index_source
        self.indexer = _FakeIndexer() if index_source else None


def _role_model():
    """Return a stand-in model with one attention per CSA2 role."""
    attentions = {
        "full": _FakeAttention(2, kv_source=True, index_source=True),
        "reindex": _FakeAttention(24, index_source=True),
        "reuse": _FakeAttention(25),
    }
    model = nn.Module()
    model.norm = nn.LayerNorm(HIDDEN)
    model.layers = nn.ModuleList(list(attentions.values()))
    return model, attentions


_RECOMPUTE_TIERS = (
    # tier, roles kept out of the recompute, index sources that recompute and get pinned
    ("none", ("full", "reindex", "reuse"), ()),
    ("reuse", ("full", "reindex"), ()),
    ("reuse_reindex", ("full",), ("reindex",)),
    ("all", (), ("full", "reindex")),
)


def _keeps_out_of_recompute(policy, module: nn.Module) -> bool:
    """Return whether one policy keeps a module outside the recompute regions."""
    predicate = policy.exclude_predicate
    return bool(predicate is not None and predicate("layer.self_attn", module))


def test_recompute_tiers_exclude_and_pin_one_role_set():
    """The tier selects which attentions recompute and which of them get pinned."""
    for tier, resident, pinned in _RECOMPUTE_TIERS:
        with mock.patch.object(attention_policy, "ATTENTION_RECOMPUTE_TIER", tier):
            policy = attention_policy.build_recompute_policy()
            model, attentions = _role_model()
            policy.on_applied(model)

            assert not _keeps_out_of_recompute(policy, model.norm)
            for role, attention in attentions.items():
                assert attention_role(attention) == role
                assert _keeps_out_of_recompute(policy, attention) == (role in resident)
                assert bool(getattr(attention.indexer, "selection_pinned", False)) == (role in pinned)
