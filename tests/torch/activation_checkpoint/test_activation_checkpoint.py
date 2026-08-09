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
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase
from tests.torch.activation_checkpoint import activation_checkpoint as activation_checkpoint_cases
from tests.torch.activation_checkpoint import checkpoint_cases

ACTIVATION_CHECKPOINT = "activation_checkpoint.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ac_memory_group():
    """
    Feature: parallel run case in activation_checkpoint
    Description:
        1. test_ac_memory_comparison
        2. test_checkpoint_wrapper_accepts_func
        3. test_wrapper_overlap_detection_cases
        4. test_wrapper_non_overlapping_allowed_cases
        5. test_rmsnorm_matmul_checkpoint_exclude_memory
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(ACTIVATION_CHECKPOINT, "test_ac_memory_comparison", 12404, 1),
        TorchCase(ACTIVATION_CHECKPOINT, "test_checkpoint_wrapper_accepts_func", 12405, 1),
        TorchCase(ACTIVATION_CHECKPOINT, "test_wrapper_overlap_detection_cases", 12406, 1),
        TorchCase(ACTIVATION_CHECKPOINT, "test_wrapper_non_overlapping_allowed_cases", 12407, 1),
        TorchCase("checkpoint_exclude_matmul.py", "test_rmsnorm_matmul_checkpoint_exclude_memory", 12408, 1),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_native_npu_rng_for_closure_only_tensor_is_not_restored():
    """Verify native checkpoint does not restore closure-only NPU RNG state."""
    activation_checkpoint_cases.test_native_npu_rng_for_closure_only_tensor_is_not_restored()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_hyper_npu_rng_for_closure_only_tensor_matches_native():
    """Verify Hyper matches native closure-only NPU RNG behavior."""
    activation_checkpoint_cases.test_hyper_npu_rng_for_closure_only_tensor_matches_native()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_scheduled_recompute_supports_dx_dw_split() -> None:
    """Verify one prefired recomputation serves separate dx and dw autograd calls."""
    activation_checkpoint_cases.test_scheduled_recompute_supports_dx_dw_split()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_scheduled_recompute_npu_preserves_rng_state() -> None:
    """Verify scheduled NPU recomputation preserves random state."""
    activation_checkpoint_cases.test_scheduled_recompute_npu_preserves_rng_state()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_scheduled_recompute_npu_restores_autocast() -> None:
    """Verify scheduled NPU recomputation restores autocast settings."""
    activation_checkpoint_cases.test_scheduled_recompute_npu_restores_autocast()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_checkpoint_npu_semantics() -> None:
    """Run eager checkpoint semantics and scheduling coverage on NPU."""
    checkpoint_cases.run_checkpoint_cases()
