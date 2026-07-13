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
"""Launch Torch HSDP AVG gradient scaling parity ST."""
from tests.common.mark_utils import arg_mark
from tests.torch.utils import torchrun_case

_FILE_NAME = "_test_hsdp_avg_grad_scale_parity.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_torch_hsdp_avg_grad_scale_parity():
    """
    Feature: HSDP AVG gradient scaling correctness (Torch, comm_fusion=False vs True).
    Description: Launch an 8-card torchrun case mirroring the MindSpore ST.
    Expectation: Run success.
    """
    torchrun_case(_FILE_NAME, "test_torch_hsdp_avg_grad_scale_parity", num_proc=8)
