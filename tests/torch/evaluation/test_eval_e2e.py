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
    Description: Evaluate on two NPUs with replicated weights.
    Expectation: Both ranks report identical global metrics.
    """
    torchrun_case(_WORKER, "test_eval_dp_ranks_report_identical_global_metrics", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_native_eval_range_and_empty_lanes() -> None:
    """Feature: standalone finite evaluation plans.
    Description: Evaluate five samples, one sample, and the whole finite split on two native-model lanes.
    Expectation: Every input is selected once and loss matches an independent reference.
    """
    torchrun_case(_WORKER, "test_native_eval_range_and_empty_lanes", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_eval_restores_latest_model_only() -> None:
    """Feature: standalone model-only checkpoint restoration.
    Description: Rebuild a two-lane evaluator from a real LATEST DCP checkpoint.
    Expectation: Saved weights and independent loss/PPL agree without training state.
    """
    torchrun_case(_WORKER, "test_eval_restores_latest_model_only", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_fsdp_eval_range_and_empty_lanes() -> None:
    """Feature: finite evaluation with two-way FSDP parameter sharding.
    Description: Evaluate uneven, empty and natural data ranges against a full-weight native model.
    Expectation: Repeated loss/perplexity and exact token/sample totals agree on both NPUs.
    """
    torchrun_case(_WORKER, "test_fsdp_eval_range_and_empty_lanes", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_hccl_evaluation_preserves_small_increments() -> None:
    """Feature: high-precision statistics on NPU/HCCL.
    Description: Reduce large totals with small increments through the production evaluator.
    Expectation: Exact counts and numerators survive one shared loss/perplexity all-reduce.
    """
    torchrun_case(_WORKER, "test_hccl_evaluation_preserves_small_increments", num_proc=2)
