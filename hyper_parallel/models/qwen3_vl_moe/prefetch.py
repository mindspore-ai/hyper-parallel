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
"""FSDP all-gather / compute overlap: prefetch chaining across sibling units.

Model ``parallelize`` modules own the policy (which units, in what order);
this module owns the mechanics.
"""
import logging
from typing import Any, Dict, Sequence, Tuple

from hyper_parallel.core.fully_shard.api import HSDPModule

logger = logging.getLogger(__name__)


def resolve_prefetch_depths(cfg: Any) -> Tuple[int, int]:
    """Return ``(forward, backward)`` prefetch depths from ``cfg``.

    Missing fields and ``None`` mean "off"; negative values clamp to 0.
    """
    accelerator = cfg.train.accelerator
    forward = max(int(getattr(accelerator, "fsdp_forward_prefetch", 0) or 0), 0)
    backward = max(int(getattr(accelerator, "fsdp_backward_prefetch", 0) or 0), 0)
    return forward, backward


def chain_fsdp_prefetch(units: Sequence[Any], forward_depth: int,
                        backward_depth: int,
                        label: str = "fsdp units") -> int:
    """Chain all-gather prefetch across an ordered run of sibling FSDP units.

    ``units`` must be in **execution** order: unit ``i`` issues the all-gather
    for units ``i+1 .. i+depth`` from its own pre-hook; backward walks the list
    reversed. Non-:class:`HSDPModule` entries are dropped — callers wrap
    opportunistically and a partially wrapped model is legal.

    Returns:
        The number of units actually chained (0 when there is nothing to do).
    """
    wrapped = [unit for unit in units if isinstance(unit, HSDPModule)]
    if len(wrapped) < 2:
        return 0
    if forward_depth > 0:
        for i, unit in enumerate(wrapped):
            targets = wrapped[i + 1: i + 1 + forward_depth]
            if targets:
                unit.set_modules_to_forward_prefetch(targets)
    if backward_depth > 0:
        reverse = list(reversed(wrapped))
        for i, unit in enumerate(reverse):
            targets = reverse[i + 1: i + 1 + backward_depth]
            if targets:
                unit.set_modules_to_backward_prefetch(targets)
    logger.info(
        "FSDP prefetch wired for %s: %d units (forward depth=%d, backward depth=%d)",
        label, len(wrapped), forward_depth, backward_depth,
    )
    return len(wrapped)


def apply_fsdp_prefetch(units: Sequence[Any], cfg: Any,
                        fsdp_kwargs: Dict[str, Any],
                        label: str = "fsdp units") -> int:
    """Chain ``units`` when the config asks for it *and* it can actually help.

    Skipped for units wrapped with ``reshard_after_forward=False`` (e.g.
    pipeline stages): their parameters stay gathered, so there is no
    all-gather to overlap. Returns the number of units chained.
    """
    forward_depth, backward_depth = resolve_prefetch_depths(cfg)
    if forward_depth <= 0 and backward_depth <= 0:
        return 0
    if not fsdp_kwargs.get("reshard_after_forward", True):
        logger.info(
            "FSDP prefetch skipped for %s: reshard_after_forward=False already "
            "keeps parameters gathered, so there is no all-gather to overlap.",
            label,
        )
        return 0
    return chain_fsdp_prefetch(units, forward_depth, backward_depth, label)
