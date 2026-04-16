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
"""msrun launchers for MindSpore ContextParallel integration tests."""
import os

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

_WORKER = os.path.join(os.path.dirname(__file__), "context_parallel.py")


def _run_group(*cases):
    """Launch a group of MindSpore distributed worker cases with ``parallel_run``."""
    parallel_run([
        MindSporeCase(_WORKER, case_name, master_port, worker_num, local_worker_num)
        for case_name, master_port, worker_num, local_worker_num in cases
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_context_parallel_group1_msrun():
    """
    Feature: MindSpore ContextParallel grouped 2-card integration tests.
    Description:
        Launch four 2-card workers together to improve 8-card utilization:
        1. sync Ulysses forward parity
        2. async Ulysses forward parity
        3. async Ulysses backward parity
        4. async Ulysses repeated forward stability
    Expectation: Run success.
    """
    _run_group(
        ("test_context_parallel_ulysses_forward", 13320, 2, 2),
        ("test_async_context_parallel_ulysses_forward", 13340, 2, 2),
        ("test_async_context_parallel_ulysses_backward", 13360, 2, 2),
        ("test_async_context_parallel_ulysses_forward_repeat", 13380, 2, 2),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_context_parallel_group2_msrun():
    """
    Feature: MindSpore ContextParallel grouped remaining mixed-card integration tests.
    Description:
        Launch the remaining 2-card and 4-card workers together:
        1. async Ulysses repeated backward stability
        2. async hybrid forward parity
    Expectation: Run success.
    """
    _run_group(
        ("test_async_context_parallel_ulysses_backward_repeat", 13420, 2, 2),
        ("test_async_context_parallel_hybrid_forward", 13440, 4, 4),
    )
