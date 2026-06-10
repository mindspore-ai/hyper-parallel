# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Pytest launchers that smoke-run every Torch llama3 example via torchrun.

One launcher per example so a failure in any example surfaces an isolated
``test_*`` line. The 2-card and one 4-card case share NPUs in a single
``parallel_run`` group (6 ≤ 8 visible cards); the other 4-card and the 8-card
cases run on their own launchers because no other case fits alongside them.
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase
from tests.torch.utils import torchrun_case

# Resolve from this file so the worker path is independent of the launcher's
# cwd (the gate runs pytest from the test file's directory, not the repo root,
# so a repo-root-relative path would surface as pytest exit code 4).
_WORKER = str(Path(__file__).resolve().parent / "_test_llama3_examples.py")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_llama3_tensor_parallel_and_fsdp_tp():
    """
    Feature: smoke coverage for ``tensor_parallel_example.py`` (TP=2) and
        ``fsdp_tp_example.py`` (TP=2 + DP=2 fully_shard).
    Description:
        1. Run ``tensor_parallel_example.main()`` on 2 NPUs.
        2. Run ``fsdp_tp_example.main()`` on 4 NPUs alongside it (disjoint NPU set).
    Expectation: Both example modules complete without raising.
    """
    parallel_run([
        TorchCase(_WORKER, "test_tensor_parallel_example_npu", 13700, 2),
        TorchCase(_WORKER, "test_fsdp_tp_example_npu", 13702, 4),
    ])


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_llama3_tp_cp_example():
    """
    Feature: smoke coverage for ``tp_cp_example.py`` (TP=2 × CP=2 on a 2-D mesh).
    Description: Run ``tp_cp_example.main()`` on 4 NPUs; CP=2 splits the
        sequence dimension and TP=2 shards heads + sequence-parallel norms.
    Expectation: Example completes without raising.
    """
    torchrun_case(_WORKER, "test_tp_cp_example_npu", 13710, num_proc=4)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_llama3_dp_tp_cp_sp_fsdp_example():
    """
    Feature: smoke coverage for ``dp_tp_cp_sp_fsdp_example.py`` —
        full 4-D ``(dp, fsdp, cp, tp) = (1, 2, 2, 2)`` combo on 8 NPUs.
    Description: Run ``dp_tp_cp_sp_fsdp_example.main()`` which composes
        ``parallelize_llama3`` (TP+SP), ``ContextParallel`` on every
        ``sdpa_core``, and ``fully_shard`` over the ``(dp, fsdp)`` HSDP slice.
    Expectation: Example completes without raising.
    """
    torchrun_case(_WORKER, "test_dp_tp_cp_sp_fsdp_example_npu", 13720, num_proc=8)
