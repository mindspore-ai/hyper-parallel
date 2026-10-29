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
"""Launcher for snapshot sidecar failure coordination."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_rank0_generation_failure_releases_non_owner():
    """Rank 0 failure must release the waiting peer without a long timeout."""
    worker = str(Path(__file__).with_name("_test_snapshot_failure_coordination.py"))
    torchrun_case(worker, "test_rank0_generation_failure_releases_non_owner", num_proc=2)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_snapshot_signature_mismatch_fails_all_ranks_fast():
    """Rank-local snapshot drift must fail before artifact polling begins."""
    worker = str(Path(__file__).with_name("_test_snapshot_failure_coordination.py"))
    torchrun_case(worker, "test_snapshot_signature_mismatch_fails_all_ranks_fast", num_proc=2)
