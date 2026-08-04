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
"""Group construction: bucket cases by mesh and chunk for the launcher."""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from tests.shard_ops.framework.case_spec import OpShardCase
from tests.shard_ops.framework.registry import load_case_plan_from_package

# Canonical mesh-axis names per ndim. For cases that do NOT reference axis
# names at runtime (everything except ``needs_mesh`` MC2 ops), the names are
# pure labels — placements act by axis index and the fn never reads names. We
# rewrite them to these canonical names at bucketing time so two cases with the
# same topology but different cosmetic naming (e.g. ("sp",) vs ("dp",), or
# ("dp","mp") vs ("dp","tp")) share ONE bucket/launcher instead of splitting
# into extra serial batches. Each launcher startup costs ~50s (NPU/CANN init),
# so collapsing buckets directly cuts wall-clock.
_CANONICAL_NAMES = {1: ("dp",), 2: ("dp", "tp"), 3: ("dp", "cp", "tp")}


@dataclass
class GroupSpec:
    """A batch of cases that run in a single launcher process."""
    id: int
    cases: List[OpShardCase]
    mesh_shape: Tuple[int, ...]
    mesh_dim_names: Tuple[str, ...]
    num_proc: int
    cases_pkg: str = ""
    tags: Tuple[str, ...] = field(default_factory=tuple)
    fail_fast: bool = False


def _bucket_key(
        case: OpShardCase,
        default_shape: Tuple[int, ...],
        default_names: Tuple[str, ...],
):
    """Return the bucket key for a case.

    Base key is ``(mesh_shape, mesh_dim_names, num_proc)`` so each group can
    share one device-mesh init. A ``solo_launcher`` case gets a unique key
    (its own name appended) so it never shares a launcher process — needed for
    3-D MC2 ops whose CANN sub-comm setup conflicts with another MC2 op in the
    same process (see ``OpShardCase.solo_launcher``).

    num_proc is auto-derived from mesh_shape via math.prod() when the case
    does not set it explicitly.
    """
    shape = case.mesh_shape or default_shape
    names = case.mesh_dim_names or default_names
    if case.num_proc is not None:
        nproc = case.num_proc
    else:
        nproc = math.prod(shape)
    if case.solo_launcher:
        return (shape, names, nproc, "solo", case.name)
    if not case.needs_mesh:
        # Axis names are cosmetic for non-mesh cases — canonicalize so
        # same-topology cases merge regardless of how they named their axes.
        names = _CANONICAL_NAMES.get(len(shape), names)
    return (shape, names, nproc)


def _split_evenly(seq: List[OpShardCase],
                  parts: int) -> List[List[OpShardCase]]:
    """Distribute ``seq`` across ``parts`` chunks as evenly as possible.

    The first ``len(seq) % parts`` chunks get one extra element so the
    largest and smallest chunk differ by at most one.
    """
    n = len(seq)
    if parts <= 1 or n <= parts:
        # Degenerate cases: single chunk, or already one case per chunk.
        if parts <= 1:
            return [list(seq)]
        return [[c] for c in seq]
    base, extra = divmod(n, parts)
    chunks: List[List[OpShardCase]] = []
    idx = 0
    for i in range(parts):
        size = base + (1 if i < extra else 0)
        chunks.append(list(seq[idx:idx + size]))
        idx += size
    return chunks


def _plan_group_count(num_cases: int, num_proc: int,
                      max_cases_per_group: int,
                      global_num_proc: int) -> int:
    """Pick a group count for a single mesh bucket.

    Goal: saturate every concurrent launcher slot on the box while still
    respecting ``max_cases_per_group``. Each launcher pays its own
    startup; parallel launchers pay it once (in wall-clock) regardless
    of count, so we always prefer ``K`` groups over ``1`` group as long
    as ``K`` launchers can run concurrently.
    """
    if num_cases <= 0:
        return 0
    concurrent_slots = max(1, global_num_proc // max(1, num_proc))
    # Need at least enough groups so each one fits ``max_cases_per_group``.
    min_groups_for_cap = math.ceil(num_cases / max(1, max_cases_per_group))
    # And at least enough to use every concurrent slot.
    target = max(concurrent_slots, min_groups_for_cap)
    # But never more groups than cases (avoids empty groups).
    return min(target, num_cases)


def build_suite_groups(
        cases_pkg: str,
        mesh_shape: Tuple[int, ...] = (2, 2),
        mesh_dim_names: Tuple[str, ...] = ("dp", "tp"),
        max_cases_per_group: int = 20,
        tag_include: Optional[Set[str]] = None,
        fail_fast: bool = False,
        global_num_proc: int = 8,
) -> List[GroupSpec]:
    """Load cases from ``cases_pkg`` and split into launcher groups.

    Cases are bucketed by ``(mesh_shape, mesh_dim_names, num_proc)`` so each
    group can share one device-mesh init. Within a bucket they are split
    into groups sized at most ``max_cases_per_group``.

    The group count is chosen via ``_plan_group_count`` to saturate
    concurrent launcher slots (``global_num_proc // num_proc``) while
    respecting ``max_cases_per_group``.  This way even a small suite
    that fits in one chunk is split across multiple groups so that
    ``run_groups`` can run their launchers side-by-side on disjoint
    device slices.

    ``tag_include`` selects cases whose ``tags`` intersect the given set.
    A case declares its platform+level routing via tags like
    ``("cpu_level1", "npu_level1")``; a case that should not run on CPU
    simply omits any ``cpu_*`` tag.  When ``tag_include`` is ``None``
    (the default) all registered cases are included.

    ``fail_fast`` is forwarded onto each ``GroupSpec`` so the suite entry
    can stop on first failure (typically for level0 gates).
    """
    all_cases = load_case_plan_from_package(cases_pkg)
    if tag_include:
        all_cases = [c for c in all_cases if set(c.tags) & tag_include]

    buckets: dict = {}
    for c in all_cases:
        key = _bucket_key(c, mesh_shape, mesh_dim_names)
        buckets.setdefault(key, []).append(c)

    groups: List[GroupSpec] = []
    gid = 0
    for key, bucket in buckets.items():
        if not bucket:
            continue
        shape, names, nproc = key[0], key[1], key[2]
        # Order cases by source module then name so adjacent cases from
        # the same file (same op) hit the JIT graph cache (notably MindSpore:
        # first invocation compiles, subsequent reuse the cached graph).
        # Pure-eager torch is unaffected; this also stabilises pytest ids.
        bucket.sort(key=lambda c: (c.source_module, c.name))
        # Plan group count: saturate concurrent launcher slots while
        # respecting the per-group cap.
        num_groups = _plan_group_count(
            len(bucket), nproc, max_cases_per_group, global_num_proc,
        )
        for chunk in _split_evenly(bucket, num_groups):
            groups.append(GroupSpec(
                id=gid, cases=chunk,
                mesh_shape=shape, mesh_dim_names=names, num_proc=nproc,
                cases_pkg=cases_pkg,
                fail_fast=fail_fast,
            ))
            gid += 1
    return groups
