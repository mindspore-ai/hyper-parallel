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

from tests.common.distributed_launcher import torchrun_case, torchrun_cases
from tests.common.mark_utils import arg_mark

ACTIVATION_CHECKPOINT = str(Path(__file__).with_name("activation_checkpoint.py").resolve())
CHECKPOINT_CASES = str(Path(__file__).with_name("checkpoint_cases.py").resolve())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_ac_memory_group() -> None:
    """Compare peak NPU memory with separate allocator state for each measurement."""
    torchrun_case(ACTIVATION_CHECKPOINT, "test_ac_memory_comparison", num_proc=1, timeout=420)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_ac_features() -> None:
    """Verify eager and scheduled checkpoint gradients, NPU RNG, and autocast."""
    torchrun_cases([
        f"{ACTIVATION_CHECKPOINT}::test_hyper_npu_rng_for_closure_only_tensor_matches_native",
        f"{ACTIVATION_CHECKPOINT}::test_scheduled_recompute_supports_dx_dw_split",
        f"{ACTIVATION_CHECKPOINT}::test_scheduled_recompute_npu_preserves_rng_state",
        f"{ACTIVATION_CHECKPOINT}::test_checkpoint_exclude_save_preserves_rng_state",
        f"{ACTIVATION_CHECKPOINT}::test_scheduled_recompute_npu_restores_autocast",
        f"{CHECKPOINT_CASES}::TestCheckpoint",
        f"{CHECKPOINT_CASES}::TestScheduledRecomputation",
    ], num_proc=1, timeout=420)
