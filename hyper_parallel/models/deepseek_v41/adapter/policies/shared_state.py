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
"""Model-owned recompute hooks for the DeepSeek-V4.1 shared attention state."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from hyper_parallel.models.adapter_spec import RecomputeContext

logger = logging.getLogger(__name__)

_MIB = 1024 * 1024
# Published cross-layer entries, keyed by the attention attribute that names
# their producing layer.
_ENTRY_SOURCES = {
    "compressed_kv": "kv_source_layer_idx",
    "index_key": "kv_source_layer_idx",
    "topk": "index_source_layer_idx",
    "candidate": "candidate_source_layer_idx",
}


@dataclass
class _StateSwapPlan:
    """Decide which published entries the swap chain offloads, and track the cost.

    The attention forward calls :meth:`register` at every shared-state read. The
    plan answers three questions in one place: is this layer the last reader of
    that entry, is the pinned-host budget still open, and is a swap group active
    at all. Registration is per forward, so a micro-batch batch that reuses the
    same modules drains and refills the same groups.
    """

    enabled: bool = False
    budget_bytes: int = 0
    kinds_by_layer: dict[int, frozenset[str]] = field(default_factory=dict)
    _state_id: int | None = None
    _registered: set[tuple[str, int]] = field(default_factory=set)
    _used_bytes: int = 0

    def register(
            self,
            layer_idx: int,
            kind: str,
            source_layer: int,
            entry: Any,
            state: Any,
    ) -> bool:
        """Hand one entry to the current swap group; return whether it was taken."""
        if not self.enabled or kind not in self.kinds_by_layer.get(layer_idx, frozenset()):
            return False
        self._begin_forward(state)
        key = (kind, source_layer)
        if key in self._registered:
            return False
        entry_bytes = int(entry.numel() * entry.element_size())
        if self.budget_bytes and self._used_bytes + entry_bytes > self.budget_bytes:
            logger.debug(
                "CSA2 state swap: keeping %s from layer %d on device (%.1f MiB "
                "exceeds the remaining host budget)",
                kind,
                source_layer,
                entry_bytes / _MIB,
            )
            return False
        if not _register_swap_entry(entry, f"csa2_{kind}_{source_layer}"):
            return False
        self._registered.add(key)
        self._used_bytes += entry_bytes
        return True

    def _begin_forward(self, state: Any) -> None:
        """Reset the per-forward accounting when a new shared state shows up."""
        state_id = id(state)
        if state_id == self._state_id:
            return
        if self._state_id is not None:
            logger.info(
                "CSA2 state swap: handed %d entrie(s) / %.1f MiB to the swap "
                "chain in the previous forward",
                len(self._registered),
                self._used_bytes / _MIB,
            )
        self._state_id = state_id
        self._registered = set()
        self._used_bytes = 0


def _register_swap_entry(entry: Any, tag: str) -> bool:
    """Hand one entry to the swap group of the layer that is running.

    Returns ``False`` when no swap schedule is active for this run, so that the
    caller neither counts the entry against the host budget nor stops trying in
    a later forward.
    """
    from hyper_parallel.core.activation_memory.api import (  # pylint: disable=C0415
        swap_tensor_wrapper,
    )
    from hyper_parallel.core.activation_memory.swap import (  # pylint: disable=C0415
        SwapManager,
    )

    if not SwapManager().get_current_group_name():
        return False
    swap_tensor_wrapper(entry, tag=tag, warn_missing_group=False)
    return True


def configure_shared_state(model: Any, context: RecomputeContext) -> None:
    """Install the must-run-once regions of the layers that will be replayed.

    Replayed index-producing layers get their discrete selection turned into an
    exclusion region, because a replay must neither choose a different key set
    nor change how many tensors the checkpoint sees in each phase. The published
    shared-state entries need no schedule either: the state is an input of every
    region, so the graph keeps it alive until backward is done with every
    consumer.

    When the Trainer enables the cross-layer state swap, the same hook plans
    which published entries the swap chain takes over: each entry is registered
    by the layer that reads it last, which is exactly the layer whose prefetch
    group offloads it after its forward and loads it back before its backward.
    """
    attentions = _replayed_attentions(model)
    for module in attentions:
        _pin_regions(module)
    plan = _build_state_swap_plan(model, attentions, context)
    for module in attentions:
        module._state_swap_plan = plan  # pylint: disable=W0212


def _build_state_swap_plan(
        model: Any,
        attentions: list[Any],
        context: RecomputeContext,
) -> _StateSwapPlan:
    """Return the swap plan for every published entry of one model."""
    if not context.state_swap:
        return _StateSwapPlan()
    wired = _wired_layers(model, attentions, context.region_fqns)
    readers = _entry_readers(attentions)
    selected, skipped = _select_swap_entries(
        readers,
        _published_entries(attentions),
        scope=context.state_swap_scope,
        boundary=_encoder_boundary(attentions),
        wired=wired,
    )
    kinds_by_layer: dict[int, set[str]] = defaultdict(set)
    for (kind, source_layer), last_reader in sorted(selected.items(), key=lambda item: item[1]):
        kinds_by_layer[last_reader].add(kind)
        logger.info(
            "CSA2 state swap: %s from layer %d is read last by layer %d; "
            "layer %d offloads it",
            kind,
            source_layer,
            last_reader,
            last_reader,
        )
    for kind, source_layer, reason in skipped:
        logger.info(
            "CSA2 state swap: keeping %s from layer %d on device (%s)",
            kind,
            source_layer,
            reason,
        )
    return _StateSwapPlan(
        enabled=True,
        budget_bytes=context.state_swap_host_mb * _MIB,
        kinds_by_layer={layer: frozenset(kinds) for layer, kinds in kinds_by_layer.items()},
    )


def _entry_readers(attentions: list[Any]) -> dict[tuple[str, int], set[int]]:
    """Return the reader layers of every published entry, keyed by (kind, source)."""
    readers: dict[tuple[str, int], set[int]] = defaultdict(set)
    for module in sorted(attentions, key=lambda item: int(item.layer_idx)):
        layer_idx = int(module.layer_idx)
        if not getattr(module, "compress_ratio", 0):
            continue
        indexer = getattr(module, "indexer", None)
        owns_key = bool(getattr(indexer, "owns_key", False))
        uses_candidates = bool(getattr(indexer, "uses_candidates", False))
        is_index_source = bool(getattr(module, "is_index_source", False))
        kv_source = getattr(module, "kv_source_layer_idx", None)
        index_source = getattr(module, "index_source_layer_idx", None)
        candidate_source = getattr(module, "candidate_source_layer_idx", None)
        if kv_source is not None:
            readers[("compressed_kv", int(kv_source))].add(layer_idx)
        if index_source is not None:
            readers[("topk", int(index_source))].add(layer_idx)
        if is_index_source and not owns_key and kv_source is not None:
            readers[("index_key", int(kv_source))].add(layer_idx)
        if is_index_source and uses_candidates and candidate_source is not None:
            readers[("candidate", int(candidate_source))].add(layer_idx)
    return readers


def _published_entries(attentions: list[Any]) -> set[tuple[str, int]]:
    """Return every entry a layer publishes, including the never-read ones."""
    published = set()
    for module in attentions:
        layer_idx = int(module.layer_idx)
        if getattr(module, "is_kv_source", False):
            published.add(("compressed_kv", layer_idx))
            published.add(("index_key", layer_idx))
        if getattr(module, "is_index_source", False):
            published.add(("topk", layer_idx))
        indexer = getattr(module, "indexer", None)
        if indexer is not None and getattr(indexer, "is_candidate_source", False):
            published.add(("candidate", layer_idx))
    return published


def _select_swap_entries(
        readers: dict[tuple[str, int], set[int]],
        published: set[tuple[str, int]],
        *,
        scope: str,
        boundary: int,
        wired: list[int],
) -> tuple[dict[tuple[str, int], int], list[tuple[str, int, str]]]:
    """Split published entries into swappable ones and the reasons to keep them."""
    terminal = wired[-1] if wired else None
    wired_set = set(wired)
    selected: dict[tuple[str, int], int] = {}
    skipped: list[tuple[str, int, str]] = []
    for kind, source_layer in sorted(published | set(readers)):
        consumers = readers.get((kind, source_layer))
        if not consumers:
            skipped.append((kind, source_layer, "no layer reads it"))
            continue
        last_reader = max(consumers)
        if scope == "encoder" and last_reader >= boundary:
            skipped.append(
                (kind, source_layer, f"outside the {scope} scope, last reader {last_reader}")
            )
            continue
        if last_reader not in wired_set:
            skipped.append(
                (kind, source_layer, f"last reader {last_reader} is not a replayed region")
            )
            continue
        if last_reader == terminal:
            skipped.append(
                (kind, source_layer, f"last reader {last_reader} is the terminal region")
            )
            continue
        selected[(kind, source_layer)] = last_reader
    return selected, skipped


def _encoder_boundary(attentions: list[Any]) -> int:
    """Return the first layer outside the compressed encoder, or the layer count."""
    ratios = {int(module.layer_idx): int(getattr(module, "compress_ratio", 0)) for module in attentions}
    decoder = [layer for layer, ratio in ratios.items() if ratio == 1]
    if decoder:
        return min(decoder)
    return max(ratios, default=0) + 1


def _wired_layers(model: Any, attentions: list[Any], region_fqns: tuple[str, ...]) -> list[int]:
    """Return the layer indices of attentions that sit inside an installed region."""
    module_by_fqn = dict(model.named_modules())
    attention_ids = {id(module) for module in attentions}
    wired = [
        int(module.layer_idx)
        for fqn, module in module_by_fqn.items()
        if id(module) in attention_ids
        and any(
            fqn == region_fqn or fqn.startswith(f"{region_fqn}.")
            for region_fqn in region_fqns
        )
    ]
    return sorted(wired)


def _pin_regions(module: Any) -> None:
    """Install the must-run-once regions of one replayed layer.

    Only the indexer's discrete selection is pinned. A recompute must not choose
    a different Top-K set, and the block contains differentiable ops, so
    skipping it would also break the checkpoint's forward/recompute saved-tensor
    count; an exclusion region runs it once and replays its cached result.

    Producer-side regions are deliberately not pinned: keeping the compressor
    and indexer projections out of the recompute retains their activations
    (+0.21 GB on the 4096-token crop) while the out-of-order producer recompute
    they would guard against does not reproduce on that crop, so the pin buys
    nothing. Add a pin here only with a measurement that shows otherwise.
    """
    if not getattr(module, "is_index_source", False):
        return
    indexer = getattr(module, "indexer", None)
    if indexer is not None:
        indexer.enable_selection_exclude()


def _replayed_attentions(model: Any) -> list[Any]:
    """Return the attention modules whose forward a backward replay revisits.

    The shipped policy makes every decoder layer a region, so the CSA2
    attentions found under those regions are the ones replay reaches. An
    attention that an exclusion keeps out of replay is not pinned, because an
    exclusion region must not nest inside another exclusion region.
    """
    from hyper_parallel.core.activation_memory.wrapper import (  # pylint: disable=C0415
        CheckpointExcludeWrapper,
    )

    excluded_inner_ids = set()
    for module in model.modules():
        if isinstance(module, CheckpointExcludeWrapper):
            excluded_inner_ids.add(id(module._wrapped_module))  # pylint: disable=W0212
    replayed = []
    for module in model.modules():
        if not getattr(module, "compress_ratio", 0):
            continue
        if getattr(module, "layer_idx", None) is None:
            continue
        if id(module) in excluded_inner_ids or isinstance(module, CheckpointExcludeWrapper):
            continue
        replayed.append(module)
    return replayed


__all__ = ["configure_shared_state"]
