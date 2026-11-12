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
"""Lightweight launchers for optional CUDA Gram NS integration tests."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = str(Path(__file__).with_name("_test_muon_gram_ns.py"))


@arg_mark(plat_marks=["platform_gpu"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
def test_muon_gram_ns_single_gpu() -> None:
    """Feature: Optional Gram NS integration on CUDA.

    Description: Compare the adapter with upstream and resume a Muon checkpoint.
    Expectation: Direct integration and resumed updates match exactly.
    """
    torchrun_case(_WORKER, "test_gram_adapter_and_numerics", num_proc=1)
    torchrun_case(_WORKER, "test_gram_muon_and_resume", num_proc=1)


@arg_mark(plat_marks=["platform_gpu"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_muon_gram_ns_two_gpu() -> None:
    """Feature: Distributed Gram NS integration on CUDA.

    Description: Update sharded and replicated matrices using two NCCL ranks.
    Expectation: Gathered parameters match a plain Gram-backed reference.
    """
    torchrun_case(_WORKER, "test_gram_muon_distributed", num_proc=2)
