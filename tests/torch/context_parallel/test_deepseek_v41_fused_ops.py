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
"""Lightweight launcher for the original-AG CPU CP8 regression."""

from pathlib import Path
from unittest.mock import patch

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


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
        torchrun_case(str(Path(__file__).with_name("_test_deepseek_v41_fused_cp.py")),
                      "test_native_geometry_cp8_gloo", num_proc=8)
