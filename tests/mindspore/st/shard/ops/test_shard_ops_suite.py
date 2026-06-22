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
"""Pytest entry for the MindSpore shard-ops new framework.

``tag_include`` selects cases whose ``tags`` intersect — cases declare
routing with tags like ``("npu_level1",)``.

``fail_fast=True`` on level0 stops on first failure; level1 runs all.

``HYPER_PARALLEL_SHARD_CASE_FILTER`` env var (fnmatch glob) filters cases within groups.
"""
import os
from fnmatch import fnmatchcase

import pytest

import tests.mindspore.st.shard.ops.framework  # pylint: disable=W0611  # register backend
from tests.common.mark_utils import arg_mark
from tests.shard_ops.framework import RUNNER, build_suite_groups
from tests.shard_ops.framework.suite import GroupSpec

CASES_PKG = "tests.mindspore.st.shard.ops.cases"

# See torch entry — startup-time dominates, so pack everything into one
# launcher per level by default and let ``run_groups`` activate
# concurrency only when more groups exist.
_MAX_CASES_PER_GROUP = 256

_GROUPS_LEVEL0 = build_suite_groups(
    cases_pkg=CASES_PKG, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"),
    max_cases_per_group=_MAX_CASES_PER_GROUP,
    tag_include={"npu_level0"}, fail_fast=True,
)
_GROUPS_LEVEL1 = build_suite_groups(
    cases_pkg=CASES_PKG, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"),
    max_cases_per_group=_MAX_CASES_PER_GROUP,
    tag_include={"npu_level1"}, fail_fast=False,
)


def _run_groups(groups, fail_fast):
    """Run groups with optional HYPER_PARALLEL_SHARD_CASE_FILTER filter."""
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
        framework="mindspore", device_type="npu",
        fail_fast=fail_fast,
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_shard_ops_ascend_level0():
    """MindSpore Ascend level0 — fail-fast for critical ops."""
    _run_groups(_GROUPS_LEVEL0, fail_fast=True)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_shard_ops_ascend_level1():
    """MindSpore Ascend level1 — full coverage daily build."""
    _run_groups(_GROUPS_LEVEL1, fail_fast=False)
