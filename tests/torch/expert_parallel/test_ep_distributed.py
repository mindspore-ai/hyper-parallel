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
"""Pytest launchers for Expert Parallelism distributed NPU precision tests.

Each test function uses ``parallel_run`` to spawn ``torchrun`` workers that
execute the actual distributed logic in ``_test_ep_distributed.py``.

Port allocation:
  10480  EP forward/backward precision (4 cards, 4 experts)
  10482  EP multi-experts-per-rank precision (4 cards, 8 experts)
  10483  EP with npu_grouped_matmul kernel (4 cards, 4 experts)
  10484  EP local-shard input: basic forward/backward (4 cards, 4 experts)
  10485  EP local-shard input: multi-expert per rank (4 cards, 8 experts)
  10486  EP local-shard input: grouped_mm kernel (4 cards, 4 experts)
  10487  EP local-shard input: top_k=1 (4 cards, 4 experts)
  10488  EP local-shard input: shared expert (4 cards, 4 experts)
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase


_WORKER = str(Path(__file__).resolve().parent / "_test_ep_distributed.py")


def _run_group(*cases):
    """Launch a group of worker cases with ``parallel_run``.

    Args:
        *cases: Each element is a tuple ``(case_name, master_port, num_proc)``.
    """
    parallel_run([
        TorchCase(_WORKER, case_name, master_port, num_proc)
        for case_name, master_port, num_proc in cases
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_ep_forward_backward():
    """
    Feature: ExpertParallel forward and backward precision on NPU.
    Description:
        4-card EP with 4 experts (one expert per rank). Verifies that EP output
        and input gradients match the standalone (single-rank) reference within
        rtol=1e-3, atol=1e-3.
    Expectation: Run success.
    """
    _run_group(
        ("test_ep_forward_backward_npu", 10480, 4),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="unessential")
def test_ep_multi_experts_per_rank():
    """
    Feature: ExpertParallel with multiple local experts per rank.
    Description:
        4-card EP with 8 experts (2 experts per rank). Verifies that EP output
        and input gradients match the standalone reference.
    Expectation: Run success.
    """
    _run_group(
        ("test_ep_multi_experts_per_rank_npu", 10482, 4),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="unessential")
def test_ep_grouped_mm():
    """
    Feature: ExpertParallel with npu_grouped_matmul kernel precision.
    Description:
        4-card EP with 4 experts (one per rank), use_grouped_mm=True.
        Verifies that npu_grouped_matmul output and gradients match the
        for-loop reference within relaxed tolerance (rtol=1e-2, atol=1e-2).
    Expectation: Run success.
    """
    _run_group(
        ("test_ep_grouped_mm_npu", 10483, 4),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_ep_local_shard_forward_backward():
    """
    Feature: ExpertParallel with local-shard token input, forward/backward precision.
    Description:
        4-card EP with 4 experts (one per rank).  Each rank receives its own
        token slice (slen // 4).  Verifies that EP output and input gradients
        match the standalone (single-rank, same local slice) reference within
        rtol=1e-3, atol=1e-3.
    Expectation: Run success.
    """
    _run_group(
        ("test_ep_local_shard_forward_backward_npu", 10484, 4),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="unessential")
def test_ep_local_shard_multi_expert():
    """
    Feature: ExpertParallel local-shard input with multiple experts per rank.
    Description:
        4-card EP with 8 experts (2 per rank).  Each rank receives its own
        token slice.  Verifies EP output and gradients match standalone.
    Expectation: Run success.
    """
    _run_group(
        ("test_ep_local_shard_multi_expert_npu", 10485, 4),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="unessential")
def test_ep_local_shard_grouped_mm():
    """
    Feature: ExpertParallel local-shard input with npu_grouped_matmul.
    Description:
        4-card EP with 4 experts (one per rank), use_grouped_mm=True, local-shard
        input.  Verifies grouped_mm EP matches for-loop EP within rtol=1e-2.
    Expectation: Run success.
    """
    _run_group(
        ("test_ep_local_shard_grouped_mm_npu", 10486, 4),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="unessential")
def test_ep_local_shard_top1():
    """
    Feature: ExpertParallel local-shard input with top_k=1.
    Description:
        4-card EP with 4 experts (one per rank), top_k=1, local-shard input.
        Verifies EP output and gradients match standalone.
    Expectation: Run success.
    """
    _run_group(
        ("test_ep_local_shard_top1_npu", 10487, 4),
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="unessential")
def test_ep_local_shard_shared_expert():
    """
    Feature: ExpertParallel local-shard input with a shared (always-active) expert.
    Description:
        4-card EP with 4 experts + 1 shared expert, local-shard input.
        Verifies EP output and gradients match standalone.
    Expectation: Run success.
    """
    _run_group(
        ("test_ep_local_shard_shared_expert_npu", 10488, 4),
    )
