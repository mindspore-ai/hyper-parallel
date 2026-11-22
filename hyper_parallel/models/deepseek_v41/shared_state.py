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

from typing import Any

#: The two CSA2 attention roles that publish cross-layer state.
_INDEX_SOURCE_ROLES = frozenset({"full", "reindex"})


def attention_role(module: Any) -> str | None:
    """Return the CSA2 role of one attention module, or None otherwise.

    ``full`` publishes compressed K=V and the index key, ``reindex`` publishes
    its own Top-K selection, and ``reuse`` consumes what a producer published. A
    layer whose attention carries no compressed state takes the consumer role as
    well, because recomputing it publishes nothing either.
    """
    if getattr(module, "layer_idx", None) is None or not hasattr(module, "compress_ratio"):
        return None
    if getattr(module, "is_kv_source", False):
        return "full"
    if getattr(module, "is_index_source", False):
        return "reindex"
    return "reuse"


def configure_shared_state(model: Any, roles: tuple[str, ...]) -> None:
    """Pin the discrete selection of every recomputed index source.

    The policy keeps the attention roles outside ``roles`` out of the recompute
    regions and hands the same tuple here, so a pinned selection always belongs
    to a layer whose forward runs again during backward. The published
    shared-state entries need no schedule: the state is an input of every
    region, so the graph keeps it and its entries alive until backward is done
    with every consumer.
    """
    for module in model.modules():
        role = attention_role(module)
        if role in _INDEX_SOURCE_ROLES and role in roles:
            _pin_regions(module)


def _pin_regions(module: Any) -> None:
    """Turn the discrete selection of one recomputed index source into a region.

    The selection is the only pinned block: recomputing it must not choose a
    different Top-K set, and the block contains differentiable ops, whose
    saved-tensor count has to match between the forward pass and the
    recomputation. The compressor and the indexer projections recompute with the
    rest of the layer.
    """
    indexer = getattr(module, "indexer", None)
    if indexer is not None:
        indexer.enable_selection_exclude()


__all__ = ["attention_role", "configure_shared_state"]
