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
"""Lightweight launchers for real-model prediction acceptance."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = str(Path(__file__).resolve().parent / "_test_predict_e2e.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_greedy_matches_huggingface() -> None:
    """Feature: shared generation parity.
    Description: Compare native Qwen3 and Qwen3-MoE with HF using cache and left padding.
    Expectation: Completion tokens match exactly.
    """
    torchrun_case(_WORKER, "test_greedy_matches_huggingface", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_dp_prediction_is_complete_ordered_and_failure_safe() -> None:
    """Feature: independent DP prediction.
    Description: Exercise non-divisible inputs, empty lanes and a lane exception.
    Expectation: Ordered complete output on success and no output on failure.
    """
    torchrun_case(_WORKER, "test_dp_prediction_is_complete_ordered_and_failure_safe", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_text_trainer_predict_restores_model_only() -> None:
    """Feature: prediction task assembly and model-only checkpoint restore.
    Description: Construct TextTrainer from config without a dataset or optimizer and restore LATEST.
    Expectation: All weights and generated completions match the saved model on both lanes.
    """
    torchrun_case(_WORKER, "test_text_trainer_predict_restores_model_only", num_proc=2)
