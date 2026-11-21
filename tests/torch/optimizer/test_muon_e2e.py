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
"""Launch _test_muon_e2e.py distributed cases."""
from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_TEST_MUON_E2E = str(Path(__file__).resolve().parent / "_test_muon_e2e.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_muon_e2e_sharding_parity() -> None:
    """Compare three sharding layouts and optimizer states in one eight-rank job."""
    torchrun_case(_TEST_MUON_E2E, "test_muon_e2e_sharding_parity", num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_muon_e2e_forward_backward_smoke_8p():
    """
    Feature: Muon end-to-end training smoke under fully_shard with real autograd gradients.
    Description:
        1. test_muon_e2e_forward_backward_smoke_8p
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_MUON_E2E, "test_muon_e2e_forward_backward_smoke_8p", None, 8),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_muon_e2e_callbacks():
    """
    Feature: Muon callback e2e cases (single card each).
    Description:
        1. test_muon_custom_ns_coefficients_parity (custom NS coefficients vs built-in variants)
        2. test_muon_zeropower_fn_parity (zeropower_fn delegation)
        3. test_muon_momentum_update_fn_parity (custom momentum math)
        4. test_muon_apply_lr_in_update_parity (lr folded into the update)
        5. test_muon_zero_rms_scale_mode (zero/use_lr with matched_adamw_rms=0)
        6. test_muon_ns_transform_fn_qkv_split_parity (fused qkv reversible transform)
        7. test_muon_post_update_fn_context (per-parameter post-update callback coverage)
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_MUON_E2E, "test_muon_custom_ns_coefficients_parity", None, 1),
        TorchCase(_TEST_MUON_E2E, "test_muon_zeropower_fn_parity", None, 1),
        TorchCase(_TEST_MUON_E2E, "test_muon_momentum_update_fn_parity", None, 1),
        TorchCase(_TEST_MUON_E2E, "test_muon_apply_lr_in_update_parity", None, 1),
        TorchCase(_TEST_MUON_E2E, "test_muon_zero_rms_scale_mode", None, 1),
        TorchCase(_TEST_MUON_E2E, "test_muon_ns_transform_fn_qkv_split_parity", None, 1),
        TorchCase(_TEST_MUON_E2E, "test_muon_post_update_fn_context", None, 1),
    ])
