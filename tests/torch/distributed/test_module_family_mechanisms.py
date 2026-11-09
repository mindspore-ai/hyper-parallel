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
"""Thin launchers for the common module-family mechanism acceptance cases."""

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_contexts_and_dcp_cpu() -> None:
    """Feature: Common CP/EP transport and complete DCP continuation.
    Description: Exercise four actual Gloo workers without accelerator allocation.
    Expectation: Partitions, differentiable transport and restarted updates match references.
    """
    torchrun_case(__file__.replace("test_module_family", "_test_module_family"),
                  "test_parallel_contexts_and_dcp_cpu", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_family_cp_fsdp_dcp_npu() -> None:
    """Feature: Direct CP family plus main HyperParallel FSDP2 and DCP.
    Description: Compare DP2 x CP2 updates and a persisted restart on four NPUs.
    Expectation: Correct global parameters and moments without module-side parameter reductions.
    """
    torchrun_case(__file__.replace("test_module_family", "_test_module_family"),
                  "test_family_cp_fsdp_dcp_npu", num_proc=4)
