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
"""Lightweight launchers for native operator fusion and original-AG CP8 regression."""

from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


def _require_native_packages():
    """Skip optional installations in the parent without importing accelerator frameworks."""
    if find_spec("torch_npu") is None:
        pytest.skip("requires torch-npu and Ascend NPU")
    # This gates the optional ST only; the worker imports every required entry.
    if find_spec("cann_ops_transformer") is None:
        pytest.skip("requires the ops-transformer training extension")



@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_native_fused_operators():
    """Validate real Indexer, causal KL and V4 attention with independent formulas.

    Feature: LI V2, SMLA and fused KL.
    Description: Validate real Indexer, causal KL and V4 attention with independent formulas.
    Expectation: Native dispatch and independent precision checks pass.
    """
    _require_native_packages()
    torchrun_case(str(Path(__file__).with_name("_test_v41_native_ops.py")),
                  "test_native_fused_operators", num_proc=1)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_native_geometry_cp8_gloo():
    """Run the mixed-ratio Full/Reindex/Reuse regression on eight CPU ranks.

    Feature: CP8 shared compressed attention.
    Description: Run the mixed-ratio Full/Reindex/Reuse regression on eight CPU ranks.
    Expectation: Outputs, every gradient and two SGD updates match the unsharded reference.
    """
    # CPU workers need no accelerator backend, including torch's entry-point loading.
    with patch.dict("os.environ", {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}):
        torchrun_case(str(Path(__file__).with_name("_test_v41_native_cp.py")),
                      "test_native_geometry_cp8_gloo", num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_native_cp8_training_smoke():
    """Require real dispatch and finite parameter updates on two packed steps.

    Feature: CP8 native pretraining.
    Description: Require real dispatch and finite parameter updates on two packed steps.
    Expectation: Native calls execute and both steps retain finite outputs, gradients and updates.
    """
    _require_native_packages()
    torchrun_case(str(Path(__file__).with_name("_test_v41_native_cp.py")),
                  "test_native_cp8_training_smoke", num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_native_fused_recompute():
    """Replay the fused region under full and strict selective checkpointing.

    Feature: Fused-region checkpoint replay.
    Description: Replay the fused region under full and strict selective checkpointing.
    Expectation: Loss and all seven input gradients exactly match without recomputation.
    """
    _require_native_packages()
    torchrun_case(str(Path(__file__).with_name("_test_v41_native_ops.py")),
                  "test_native_fused_recompute", num_proc=1)
