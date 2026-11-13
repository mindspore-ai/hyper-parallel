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
"""Launch the standalone-evaluation end-to-end workers."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = str(Path(__file__).resolve().parent / "_test_eval_e2e.py")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_eval_matches_reference_single_card() -> None:
    """Verify standalone evaluation on one NPU against an independent reference.

    Feature: run_mode eval, single card.
    Description: Evaluate a tiny random Qwen3 through the real TextTrainer.
    Expectation: Metrics match a plain cross-entropy loop over the same batches.
    """
    torchrun_case(_WORKER, "test_eval_matches_reference_single_card", num_proc=1)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_eval_dp_ranks_report_identical_global_metrics() -> None:
    """Verify data-parallel evaluation reports one global result.

    Feature: run_mode eval, DP=2.
    Description: Evaluate on two NPUs with FSDP-sharded weights.
    Expectation: Both ranks report identical global metrics.
    """
    torchrun_case(_WORKER, "test_eval_dp_ranks_report_identical_global_metrics", num_proc=2)
