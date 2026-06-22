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
"""Pytest entry for the Torch shard-ops new framework.

Each test function is a ``(device_type, level)`` combination.  ``arg_mark``
decorators route to the appropriate CI gate; ``tag_include`` selects cases
whose ``tags`` intersect — cases declare routing with tags like
``("cpu_level1", "npu_level1")``.

``fail_fast`` is wired per pytest function: level0 stops on the first
failure, level1 runs every case.
"""
import os
from fnmatch import fnmatchcase

import pytest

import tests.torch.shard.ops.framework  # pylint: disable=W0611  # register backends
from tests.common.mark_utils import arg_mark
from tests.shard_ops.framework import RUNNER, build_suite_groups
from tests.shard_ops.framework.suite import GroupSpec

CASES_PKG = "tests.torch.shard.ops.cases"

# Pack everything into a single launcher per level by default. Each
# launcher pays ~25-50s of hccl/CANN/msrun init, which dwarfs the
# typical case exec (ms scale), so amortising one startup across many
# cases is the dominant win. ``RUNNER.run_groups`` still runs multiple
# groups concurrently when they exist — useful once individual cases
# get heavy enough that exec_time approaches startup_time, or when
# different cases need different mesh shapes (which buckets them into
# separate groups automatically).
_MAX_CASES_PER_GROUP = 256

_GROUPS_CPU_LEVEL0 = build_suite_groups(
    cases_pkg=CASES_PKG, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"),
    max_cases_per_group=_MAX_CASES_PER_GROUP,
    tag_include={"cpu_level0"}, fail_fast=True,
)
_GROUPS_CPU_LEVEL1 = build_suite_groups(
    cases_pkg=CASES_PKG, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"),
    max_cases_per_group=_MAX_CASES_PER_GROUP,
    tag_include={"cpu_level1"}, fail_fast=False,
)
_GROUPS_ASCEND_LEVEL0 = build_suite_groups(
    cases_pkg=CASES_PKG, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"),
    max_cases_per_group=_MAX_CASES_PER_GROUP,
    tag_include={"npu_level0"}, fail_fast=True,
)
_GROUPS_ASCEND_LEVEL1 = build_suite_groups(
    cases_pkg=CASES_PKG, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"),
    max_cases_per_group=_MAX_CASES_PER_GROUP,
    tag_include={"npu_level1"}, fail_fast=False,
)


def _run_groups(groups, framework, device_type, fail_fast):
    """Run all groups for one (framework, device, level) via the 8-card
    concurrent path. Skip cleanly when the suite is empty so pytest
    collection stays stable across levels that have no cases yet.

    ``HYPER_PARALLEL_SHARD_CASE_FILTER`` env var (fnmatch glob) filters cases within groups;
    e.g. ``HP_CASE="sort_ops_*" pytest ... -vs`` runs only sort cases.
    """
    if not groups:
        pytest.skip("no shard-ops cases registered at this level yet")
    case_filter = os.environ.get("HYPER_PARALLEL_SHARD_CASE_FILTER")
    if case_filter:
        groups = [
            GroupSpec(
                id=g.id, mesh_shape=g.mesh_shape, mesh_dim_names=g.mesh_dim_names,
                num_proc=g.num_proc, cases_pkg=g.cases_pkg, fail_fast=g.fail_fast,
                cases=[c for c in g.cases if fnmatchcase(c.name, case_filter)],
            )
            for g in groups
        ]
        groups = [g for g in groups if g.cases]
        if not groups:
            pytest.skip(f"no case matched HP_CASE={case_filter!r}")
    RUNNER.run_groups(
        groups,
        framework=framework, device_type=device_type,
        fail_fast=fail_fast,
    )


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_shard_ops_cpu_level0():
    """Torch CPU (gloo) level0 — fail-fast for fast PR feedback."""
    _run_groups(_GROUPS_CPU_LEVEL0, "torch", "cpu", fail_fast=True)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_shard_ops_cpu_level1():
    """Torch CPU (gloo) level1 — run all cases, collect every failure."""
    _run_groups(_GROUPS_CPU_LEVEL1, "torch", "cpu", fail_fast=False)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_shard_ops_ascend_level0():
    """Torch Ascend (hccl) level0 — fail-fast for critical ops."""
    _run_groups(_GROUPS_ASCEND_LEVEL0, "torch", "npu", fail_fast=True)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_shard_ops_ascend_level1():
    """Torch Ascend (hccl) level1 — full coverage daily build."""
    _run_groups(_GROUPS_ASCEND_LEVEL1, "torch", "npu", fail_fast=False)
