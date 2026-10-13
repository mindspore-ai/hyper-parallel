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

import importlib
from types import SimpleNamespace
from typing import Any
from unittest import mock

import torch
from torch import nn

from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedAttentionCPContext,
    SharedCompressedAttentionState,
    SharedCompressedDSAAttentionBase,
    SharedCompressedDSAIndexer,
)
from hyper_parallel.core.activation_memory.api import checkpoint_wrapper
from hyper_parallel.core.activation_memory.recompute_state import create_recompute_contexts
from hyper_parallel.core.activation_memory.wrapper import checkpoint_exclude_wrapper
from hyper_parallel.models.deepseek_v41.adapter.policies.shared_state import (
    configure_shared_state,
)
from hyper_parallel.models.adapter_spec import RecomputeContext

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


def _build_layer():
    """Build a KV-source attention layer and the shared state it consumes."""
    torch.manual_seed(11)
    layer = SharedCompressedDSAAttentionBase(
        _Source(), replace_indexer=False, use_optimized_sparse_attention=False
    )
    layer.compressor = _Compressor()
    state = _CountingState()
    state.publish_topk_indices(LAYER, torch.zeros(BATCH, SEQUENCE, 2, dtype=torch.int32))
    return layer, state


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
        _Source(), replace_indexer=False, use_optimized_sparse_attention=False
    )
    layer.compressor = _Compressor()
    state.publish_topk_indices(LAYER, torch.zeros(BATCH, SEQUENCE, 2, dtype=torch.int32))
    return layer, _inputs(), _GatherCounter() if use_cp else None


def _index_source_layer(use_cp, state):
    """Build a Reindex-style layer that republishes the shared Top-K."""
    torch.manual_seed(11)
    layer = SharedCompressedDSAAttentionBase(
        _IndexSource(), replace_indexer=False, use_optimized_sparse_attention=False
    )
    layer.indexer = SharedCompressedDSAIndexer(_IndexerSource())
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


def _assert_recompute_matches_plain(builder, *, use_cp: bool) -> None:
    """Compare one role's plain and recomputed runs, including its state writes."""
    plain_state = _CountingState()
    plain_layer, plain_inputs, plain_gathers = builder(use_cp, plain_state)
    plain_before = _snapshot(plain_state)
    plain_loss, plain_grad = _run(plain_layer, plain_state, plain_gathers, plain_inputs)
    plain_writes = {name: count for name, count in _delta(plain_before, _snapshot(plain_state)).items() if count}

    replayed_state = _CountingState()
    replayed_layer, replayed_inputs, replayed_gathers = builder(use_cp, replayed_state)
    replayed_before = _snapshot(replayed_state)
    replayed_loss, replayed_grad = _run(
        checkpoint_wrapper(replayed_layer), replayed_state, replayed_gathers, replayed_inputs
    )
    replayed_after = _snapshot(replayed_state)
    replayed_writes = {
        name: count for name, count in _delta(replayed_before, replayed_after).items() if count
    }

    assert replayed_loss == plain_loss
    assert torch.equal(replayed_grad, plain_grad)
    assert plain_writes == replayed_writes
    # Every write the plain pass performs must happen exactly once under
    # recompute: the phase guards keep the publishes from running twice.
    assert plain_writes and all(value == 1 for value in plain_writes.values())
    if use_cp:
        # The CP gather is pure, so the recompute re-issues it with the same shapes.
        assert len(replayed_gathers.calls) == 2 * len(plain_gathers.calls)
        assert replayed_gathers.calls[: len(plain_gathers.calls)] == plain_gathers.calls


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
    """Attention stand-in exposing the attributes the model-owned hook reads."""

    def __init__(self, layer_idx: int, *, index_source: bool) -> None:
        """Describe one replayed or excluded layer stand-in."""
        super().__init__()
        self.compress_ratio = 1
        self.layer_idx = layer_idx
        self.is_index_source = index_source
        self.indexer = _FakeIndexer() if index_source else None


class _FakeModel(nn.Module):
    """Model stand-in carrying the layers plus the config the hook writes to."""

    def __init__(self) -> None:
        """Build the three attention roles plus an empty config."""
        super().__init__()
        self.config = SimpleNamespace()
        self.layers = nn.ModuleList([
            _FakeAttention(2, index_source=True),     # Full: produces and consumes state
            _FakeAttention(24, index_source=True),    # Reindex: produces state
            _FakeAttention(25, index_source=False),   # Reuse: consumes only
        ])


def test_hook_pins_the_selection_of_replayed_index_sources_only():
    """Every replayed index source is pinned, excluded and reused layers are not."""
    model = _FakeModel()
    model.layers[1] = checkpoint_exclude_wrapper(model.layers[1])
    configure_shared_state(model, RecomputeContext(region_fqns=("layers.0", "layers.2")))

    assert model.layers[0].indexer.selection_pinned is True
    assert model.layers[1]._wrapped_module.indexer.selection_pinned is False
    assert model.layers[2].indexer is None


_KV_SOURCES = (2, 8, 14, 20)
_INDEX_SOURCES = (2, 8, 14, 20, 24, 28, 32, 36)
_CANDIDATE_SOURCE = 20


class _SwapIndexer:
    """Indexer stand-in carrying the reader-relevant flags."""

    def __init__(self, *, owns_key: bool, uses_candidates: bool, is_candidate_source: bool) -> None:
        """Record one indexer role."""
        self.owns_key = owns_key
        self.uses_candidates = uses_candidates
        self.is_candidate_source = is_candidate_source
        self.selection_pinned = False

    def enable_selection_exclude(self) -> None:
        """Record the selection pin the hook installs on every index source."""
        self.selection_pinned = True


class _SwapAttention(nn.Module):
    """Attention stand-in exposing the attributes the swap plan reads."""

    def __init__(
            self,
            layer_idx: int,
            *,
            ratio: int,
            kv_source: int | None,
            index_source: int | None,
            candidate_source: int | None,
            is_kv_source: bool = False,
            is_index_source: bool = False,
    ) -> None:
        """Describe one released V4.1 layer role."""
        super().__init__()
        self.layer_idx = layer_idx
        self.compress_ratio = ratio
        self.is_kv_source = is_kv_source
        self.is_index_source = is_index_source
        self.kv_source_layer_idx = kv_source
        self.index_source_layer_idx = index_source
        self.candidate_source_layer_idx = candidate_source
        self.indexer = (
            _SwapIndexer(
                owns_key=is_kv_source,
                uses_candidates=candidate_source is not None and candidate_source < layer_idx,
                is_candidate_source=layer_idx == _CANDIDATE_SOURCE,
            )
            if is_index_source
            else None
        )


class _SwapModel(nn.Module):
    """Model stand-in with every attention inside a checkpoint region."""

    def __init__(self, attentions: list[nn.Module]) -> None:
        """Register the wrapped layers exactly like the region installer does."""
        super().__init__()
        self.config = SimpleNamespace()
        self.layers = nn.ModuleList([checkpoint_wrapper(attention) for attention in attentions])


def _crop_like_model() -> tuple[_SwapModel, list[nn.Module], tuple[str, ...]]:
    """Build the released 40-layer V4.1 role layout, all layers inside regions."""
    attentions = []
    for layer_idx in range(40):
        kv_source = max((source for source in _KV_SOURCES if source <= layer_idx), default=None)
        index_source = max((source for source in _INDEX_SOURCES if source <= layer_idx), default=None)
        ratio = 0 if layer_idx < _INDEX_SOURCES[0] else (2 if layer_idx < _CANDIDATE_SOURCE else 1)
        attentions.append(
            _SwapAttention(
                layer_idx,
                ratio=ratio,
                kv_source=kv_source,
                index_source=index_source,
                candidate_source=(
                    _CANDIDATE_SOURCE if _CANDIDATE_SOURCE <= layer_idx else None
                ),
                is_kv_source=layer_idx in _KV_SOURCES,
                is_index_source=layer_idx in _INDEX_SOURCES,
            )
        )
    model = _SwapModel(attentions)
    region_fqns = tuple(f"layers.{layer_idx}" for layer_idx in range(40))
    return model, attentions, region_fqns


def _swap_plan(**overrides):
    """Build the crop-like model and return its installed swap plan."""
    model, attentions, region_fqns = _crop_like_model()
    context = RecomputeContext(region_fqns=region_fqns, **{"state_swap": True, **overrides})
    configure_shared_state(model, context)
    # Layer 0 is a sliding-window layer, so the hook skips it; layer 7 is the
    # first replayed compressed layer and carries the plan.
    return attentions[7]._state_swap_plan, attentions


def test_state_swap_plan_keeps_the_encoder_entries_by_default():
    """Scope 'encoder' takes the three Full groups and nothing from the decoder."""
    plan, _ = _swap_plan()

    assert plan.enabled is True
    assert plan.kinds_by_layer == {
        7: frozenset({"compressed_kv", "topk"}),
        13: frozenset({"compressed_kv", "topk"}),
        19: frozenset({"compressed_kv", "topk"}),
    }


def test_state_swap_plan_extends_to_reindex_groups_with_scope_all():
    """Scope 'all' adds the reindex groups and still drops the terminal readers."""
    plan, _ = _swap_plan(state_swap_scope="all")

    assert plan.kinds_by_layer == {
        7: frozenset({"compressed_kv", "topk"}),
        13: frozenset({"compressed_kv", "topk"}),
        19: frozenset({"compressed_kv", "topk"}),
        23: frozenset({"topk"}),
        27: frozenset({"topk"}),
        31: frozenset({"topk"}),
        35: frozenset({"topk"}),
        36: frozenset({"candidate", "index_key"}),
    }


def test_state_swap_plan_is_inert_until_the_trainer_enables_it():
    """Without the Trainer switch the hook only pins selections."""
    plan, attentions = _swap_plan(state_swap=False)

    assert plan.enabled is False
    assert plan.kinds_by_layer == {}
    assert attentions[7]._state_swap_plan is plan


class _RecordingPlan:
    """Stand-in plan that records what the attention hands over."""

    def __init__(self) -> None:
        """Start with an empty record."""
        self.calls: list[tuple[int, str, int, int]] = []

    def register(
            self,
            layer_idx: int,
            kind: str,
            source_layer: int,
            entry: Any,
            _state: Any,
    ) -> bool:
        """Record one registration attempt."""
        self.calls.append((layer_idx, kind, source_layer, id(entry)))
        return True


def test_attention_routes_reads_to_the_plan_only_during_forward():
    """Each read site reports its kind, source and the entry object itself."""
    layer, state = _build_layer()
    plan = _RecordingPlan()
    layer._state_swap_plan = plan  # pylint: disable=W0212
    entry = torch.zeros(BATCH, SEQUENCE, RATIO, dtype=torch.int32)

    layer._maybe_swap_state_entry("topk", entry, state)  # pylint: disable=W0212
    assert plan.calls == [(LAYER, "topk", LAYER, id(entry))]

    plan.calls.clear()
    _, recompute_context = create_recompute_contexts()
    with recompute_context:
        layer._maybe_swap_state_entry("topk", entry, state)  # pylint: disable=W0212
    assert not plan.calls


def test_attention_state_swap_is_a_noop_without_a_plan():
    """A model that installs no plan never registers anything."""
    layer, state = _build_layer()
    entry = torch.zeros(BATCH, SEQUENCE, RATIO, dtype=torch.int32)

    layer._maybe_swap_state_entry("topk", entry, state)  # pylint: disable=W0212

    assert layer._state_swap_plan is None  # pylint: disable=W0212


class _SwapRegistration:
    """Record the entries the plan actually hands to the swap chain."""

    def __init__(self) -> None:
        """Start with an empty record."""
        self.tags: list[str] = []
        self.bytes = 0

    def __call__(self, entry: Any, tag: str) -> bool:
        """Record one registered entry."""
        self.tags.append(tag)
        self.bytes += int(entry.numel() * entry.element_size())
        return True


def test_plan_registers_each_entry_once_and_respects_the_host_budget():
    """Registration is once per entry, per forward, and bounded in bytes."""
    policy_module = importlib.import_module(
        "hyper_parallel.models.deepseek_v41.adapter.policies.shared_state"
    )
    plan = policy_module._StateSwapPlan(  # pylint: disable=W0212
        enabled=True,
        budget_bytes=8 * 4 * 4,
        kinds_by_layer={7: frozenset({"topk"}), 9: frozenset({"topk"})},
    )
    registration = _SwapRegistration()
    state = SharedCompressedAttentionState()
    entry = torch.zeros(8, 4, dtype=torch.float32)
    with mock.patch.object(policy_module, "_register_swap_entry", registration):
        assert plan.register(7, "topk", 2, entry, state) is True
        assert plan.register(7, "topk", 2, entry, state) is False
        assert plan.register(9, "topk", 8, entry, state) is False
        assert plan.register(7, "compressed_kv", 2, entry, state) is False
        assert registration.tags == ["csa2_topk_2"]

        next_state = SharedCompressedAttentionState()
        assert plan.register(7, "topk", 2, entry, next_state) is True
    assert registration.tags == ["csa2_topk_2", "csa2_topk_2"]
