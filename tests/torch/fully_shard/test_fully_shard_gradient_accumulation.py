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
"""Launch the CPU-only nested FSDP accumulation regression without importing Torch."""
import os

import pytest
from pytest import MonkeyPatch

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = os.path.join(os.path.dirname(__file__), "_test_fully_shard_gradient_accumulation.py")


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_fully_shard_gradient_accumulation(monkeypatch: MonkeyPatch) -> None:
    """
    Feature: Nested FSDP no-sync accumulation.
    Description: Include a unit that participates only before the final microbatch.
    Expectation: Both reduction paths match ordinary PyTorch and consume pending gradients.
    """
    monkeypatch.setenv("HYPER_PARALLEL_PLATFORM", "torch")
    monkeypatch.setenv("HYPER_PARALLEL_TEST_DEVICE_TYPE", "cpu")
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    torchrun_case(_WORKER, "test_nested_no_sync_flushes_inactive_microbatch", num_proc=2)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("case_name", [
    "test_backward_boundaries_preserve_accumulation",
    "test_child_backward_finalizes_the_shared_tree",
])
def test_fully_shard_backward_lifecycle(monkeypatch: MonkeyPatch, case_name: str) -> None:
    """Check backward lifetime and child-only finalization with independent process groups.

    Feature: Module-tree backward lifecycle.
    Description: Exercise nested/grouped ownership, checkpointing, and standalone child backwards.
    Expectation: Each backward finalizes once; accumulated gradients match an independent graph.
    """
    monkeypatch.setenv("HYPER_PARALLEL_PLATFORM", "torch")
    monkeypatch.setenv("HYPER_PARALLEL_TEST_DEVICE_TYPE", "cpu")
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    worker = os.path.join(os.path.dirname(__file__), "_test_fully_shard_backward_lifecycle.py")
    torchrun_case(worker, case_name, num_proc=2)
