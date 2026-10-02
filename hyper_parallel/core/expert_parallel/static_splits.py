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
"""Cache one *static* MoE routing plan so the dispatch can stop syncing every layer.

Why this exists
---------------
A MoE dispatch has to give ``all_to_all_single`` Python split lists, which costs a
device-to-host drain of the token counts: an extra counts all-to-all, its wait, and two
``.tolist()`` calls, per layer, in *both* dispatch implementations
(``core/expert_parallel/expert_parallel.py`` and
``distributed/expert_parallel/experts.py``).  A ``.tolist()`` waits for every op already
enqueued on the stream -- including the collective it just issued -- so the host cannot
enqueue the token all-to-all until the counts have landed.

That serialisation is visible in the host API profile of the 2-SN step: **8.8 s blocked in
``aclrtSynchronizeStream``** over a 25 s window (470 calls, average 18.7 ms, maximum
237 ms).

When the plan is *static* the answer never changes, so it can be computed once:

* the benchmark's ``fix_router: true`` adapter assigns slots deterministically, so every
  rank sends and receives exactly the same number of tokens to every peer, in every layer;
* a real router reshuffles tokens every step, and reusing a stale plan there would silently
  mis-route tokens.

Hence the explicit opt-in (``HP_EP_STATIC_SPLITS=1``), a plan-shape-keyed cache, and a log
line the first time a plan is reused.  Benchmark-only, exactly like ``fix_router`` itself.

BLOCKED -- measured, not usable yet
-----------------------------------
On the 2-SN GBS-256 config (``fix_router: true``, so the plan really is uniform: the log
reports ``tokens/peer=512`` = 8192 * 8 / 128 exactly) the fast path is engaged and then the
run dies in the **first forward**, in the loss-normalization all-reduce
(``trainer/runtime/metrics.py:75``), with

    Memory_Allocation_Failure(EL0004): Failed to allocate memory requested by HCCL module

Measured: **4 of 4 full fast-path runs OOM** (r32_static_r1/r2, r33_static_r1/r2) while
**0 of 2 control runs** did (min step 20.6625 s / 20.4525 s).

The bisect (r34) names the responsible half -- and it is *not* the collective:

===========================  ==================================  ==================
probe                        removes                             result
===========================  ==================================  ==================
``HP_EP_STATIC_SPLITS_PROBE=a2a``     the two D2H drains          **OOM** (24 hits)
``HP_EP_STATIC_SPLITS_PROBE=drains``  the counts all-to-all       passes, min 20.78 s
===========================  ==================================  ==================

So the ``.tolist()`` drains are not merely a cost, they are the **host-side pacing**: they
stall the enqueue once per layer, and removing them lets the host run far enough ahead that
the live-tensor peak rises until the first forward's loss all-reduce fails to get its HCCL
buffer.  A drain is a throttle as much as an expense.

Consequences:

* removing only the collective (keeping the drains) is safe but **slower** (20.78 s vs the
  20.45-20.66 s control band), so that half is not worth shipping on its own;
* making the full fast path safe needs an explicit run-ahead bound, not just the removal --
  e.g. a cheaper pacing point, a lower ``forward_prefetch_depth``, or computing the splits
  one layer ahead so the drain is hidden instead of deleted.

**Do not enable this switch on a real run until a pacing solution is measured**; it defaults
to off precisely so a blocked lever cannot leak into the champion.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Hashable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "clear_static_plans",
    "probe_mode",
    "get_static_plan",
    "static_plan_key",
    "static_splits_enabled",
    "store_static_plan",
]

_ENABLED = os.environ.get("HP_EP_STATIC_SPLITS", "0") == "1"
_PLANS: dict[Hashable, Any] = {}
_LOGGED: set[Hashable] = set()


def static_splits_enabled() -> bool:
    """Return whether ``HP_EP_STATIC_SPLITS`` asked for the cached-plan fast path."""
    return _ENABLED


def static_plan_key(*parts: Any, device: Optional[Any] = None) -> Hashable:
    """Build a cache key from the quantities that define a routing plan.

    Args:
        *parts: Shape/config values that pin the plan (token count, top-k, ep size, ...).
        device: Device the plan's tensors live on, so one process cannot mix devices.

    Returns:
        A hashable key.
    """
    return (*parts, str(device))


def get_static_plan(key: Hashable) -> Any:
    """Return the cached plan for ``key``, or ``None`` when there is none.

    Args:
        key: Key from :func:`static_plan_key`.

    Returns:
        The stored value, or ``None``.
    """
    if not _ENABLED:
        return None
    return _PLANS.get(key)


def store_static_plan(key: Hashable, value: Any, description: str = "") -> None:
    """Cache a plan and log it once, so the fast path is never invisible.

    Args:
        key: Key from :func:`static_plan_key`.
        value: The plan to reuse (split lists, counts tensor, ...).
        description: Short human-readable note for the log line.
    """
    if not _ENABLED:
        return
    _PLANS[key] = value
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(
            "EP static splits: reusing one routing plan for every layer %s -- "
            "benchmark-only, requires a static router such as fix_router=True",
            description or f"(key={key})",
        )


def clear_static_plans() -> None:
    """Drop every cached plan (used by tests and by shape changes)."""
    _PLANS.clear()
    _LOGGED.clear()


def probe_mode() -> str:
    """Return the bisect probe selected by ``HP_EP_STATIC_SPLITS_PROBE``.

    The fast path removes two things at once -- the counts all-to-all and the two
    ``.tolist()`` drains -- and a 4/4 OOM on the 2-SN config means one of them (or the
    combination) raises the live-tensor peak.  These modes remove exactly one half, so the
    culprit can be named:

    * ``"a2a"``   -- keep the collective, drop only the host drains;
    * ``"drains"``-- keep the host drains, drop only the collective;
    * ``""``      -- normal fast path (both removed).

    Returns:
        ``"a2a"``, ``"drains"`` or ``""``.
    """
    mode = os.environ.get("HP_EP_STATIC_SPLITS_PROBE", "").strip().lower()
    return mode if mode in ("a2a", "drains") else ""
