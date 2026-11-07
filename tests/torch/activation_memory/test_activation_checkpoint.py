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
"""test activation checkpoint"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

ACTIVATION_CHECKPOINT = str(Path(__file__).with_name("activation_checkpoint.py").resolve())
CHECKPOINT_CASES = str(Path(__file__).with_name("checkpoint_cases.py").resolve())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ac_memory_group():
    """
    Feature: parallel run case in activation_checkpoint
    Description:
        1. test_ac_memory_comparison
        2. test_checkpoint_wrapper_accepts_func
        3. test_wrapper_overlap_detection_cases
        4. test_wrapper_non_overlapping_allowed_cases
        5. TestCheckpoint
        6. TestScheduledRecomputation
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(ACTIVATION_CHECKPOINT, "test_ac_memory_comparison", 12404, 1),
        TorchCase(ACTIVATION_CHECKPOINT, "test_checkpoint_wrapper_accepts_func", num_proc=1),
        TorchCase(ACTIVATION_CHECKPOINT, "test_wrapper_overlap_detection_cases", num_proc=1),
        TorchCase(ACTIVATION_CHECKPOINT, "test_wrapper_non_overlapping_allowed_cases", num_proc=1),
        TorchCase(CHECKPOINT_CASES, "TestCheckpoint", num_proc=1),
        TorchCase(CHECKPOINT_CASES, "TestScheduledRecomputation", num_proc=1),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ac_features():
    """
    Feature: parallel run case in activation_checkpoint
    Description:
        1. test_native_npu_rng_for_closure_only_tensor_is_not_restored
        2. test_hyper_npu_rng_for_closure_only_tensor_matches_native
        3. test_scheduled_recompute_supports_dx_dw_split
        4. test_scheduled_recompute_npu_preserves_rng_state
        5. test_checkpoint_exclude_save_preserves_rng_state
        6. test_scheduled_recompute_npu_restores_autocast
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(ACTIVATION_CHECKPOINT, "test_native_npu_rng_for_closure_only_tensor_is_not_restored", num_proc=1),
        TorchCase(ACTIVATION_CHECKPOINT, "test_hyper_npu_rng_for_closure_only_tensor_matches_native", num_proc=1),
        TorchCase(ACTIVATION_CHECKPOINT, "test_scheduled_recompute_supports_dx_dw_split", num_proc=1),
        TorchCase(ACTIVATION_CHECKPOINT, "test_scheduled_recompute_npu_preserves_rng_state", num_proc=1),
        TorchCase(ACTIVATION_CHECKPOINT, "test_checkpoint_exclude_save_preserves_rng_state", num_proc=1),
        TorchCase(ACTIVATION_CHECKPOINT, "test_scheduled_recompute_npu_restores_autocast", num_proc=1),
    ])
