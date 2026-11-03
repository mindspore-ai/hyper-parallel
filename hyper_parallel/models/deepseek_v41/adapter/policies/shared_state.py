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

from typing import Any, Iterable

# The two CSA2 attention roles that publish indexer outputs.
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


def configure_shared_state(
    model: Any,
    roles: tuple[str, ...],
    recompute_modules: Iterable[Any] | None = None,
) -> None:
    """Pin discrete indexer selections inside the configured recompute regions.

    Args:
        model: Constructed model containing shared-compressed attention modules.
        roles: CSA2 roles whose indexer selection needs replay protection.
        recompute_modules: Modules that execute during checkpoint replay. When
            omitted, all modules matching ``roles`` are configured.
    """
    recompute_module_ids = (
        None if recompute_modules is None else {id(module) for module in recompute_modules}
    )

    for module in model.modules():
        role = attention_role(module)
        if role not in _INDEX_SOURCE_ROLES or role not in roles:
            continue
        if recompute_module_ids is not None and id(module) not in recompute_module_ids:
            continue
        _pin_regions(module)


def _pin_regions(module: Any) -> None:
    """Keep a recomputed index source's discrete selection fixed across replay."""
    indexer = getattr(module, "indexer", None)
    if indexer is not None:
        indexer.enable_selection_exclude()


__all__ = ["attention_role", "configure_shared_state"]
