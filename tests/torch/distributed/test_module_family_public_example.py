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
"""Thin launcher for the documented public module-family example."""

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_public_example_cp_ep_fsdp_npu() -> None:
    """Feature: Real public construction selects and trains CP/EP family members.
    Description: Compare direct and facade CP2, EP2 and CP2+EP2 jobs with dense AdamW references.
    Expectation: Outputs, global weights and optimizer moments agree for two steps.
    """
    torchrun_case(__file__.replace("test_module_family", "_test_module_family"),
                  "test_public_example_cp_ep_fsdp_npu", num_proc=4)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_direct_mesh_construction_cpu() -> None:
    """Feature: Public families accept mesh without a model facade or preparation call.
    Description: Construct CP2, EP2 and CP2+EP2 modules on four Gloo workers.
    Expectation: Expert ownership, outputs and local input gradients match dense references.
    """
    torchrun_case(__file__.replace("test_module_family", "_test_module_family"),
                  "test_direct_mesh_construction_cpu", num_proc=4)
