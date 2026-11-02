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
"""Framework-free launcher for Xing40 TP/FSDP QK clipping."""
from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_qk_clip_tp_strided_main_weights():
    """
    Feature: TP-local head maxima and strided FSDP main weights.
    Description: Distinct DP/TP head maxima, BF16 models and FP32 main copies.
    Expectation: Q/K rows follow the same global head's maximum; V rows stay
        unchanged, inactive clipping is exact, and unsupported layouts fail.
    """
    torchrun_case(str(Path(__file__).with_name("_test_xing40_qk_clip_tp.py")),
                  "test_xing40_qk_clip_tp_worker", num_proc=4)
