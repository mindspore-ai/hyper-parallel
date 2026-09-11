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
"""Lightweight torchrun launchers for SHMEM AllGather system tests."""

import os

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark
from tests.common.port_utils import allocate_port


_WORKER = os.path.join(os.path.dirname(__file__), "_test_all_gather.py")


def _set_unique_bootstrap_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")


def _run_boundary_matrix(monkeypatch, rank_count: int) -> None:
    _set_unique_bootstrap_endpoint(monkeypatch)
    torchrun_case(_WORKER, "test_all_gather_boundary_matrix", num_proc=rank_count)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_all_gather_boundary_matrix_1_rank(monkeypatch) -> None:
    """Run the byte-boundary matrix on one Root PE."""
    _run_boundary_matrix(monkeypatch, 1)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_all_gather_boundary_matrix_2_ranks(monkeypatch) -> None:
    """Run the byte-boundary matrix on two Root PEs."""
    _run_boundary_matrix(monkeypatch, 2)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_all_gather_boundary_matrix_4_ranks(monkeypatch) -> None:
    """Run the byte-boundary matrix on four Root PEs."""
    _run_boundary_matrix(monkeypatch, 4)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_all_gather_boundary_matrix_8_ranks(monkeypatch) -> None:
    """Run the byte-boundary matrix on eight Root PEs."""
    _run_boundary_matrix(monkeypatch, 8)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_all_gather_correctness_large(monkeypatch) -> None:
    """Run the large-message path with four Root PEs."""
    _set_unique_bootstrap_endpoint(monkeypatch)
    torchrun_case(_WORKER, "test_all_gather_correctness_large", num_proc=4)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_all_gather_stream_order_and_visibility(monkeypatch) -> None:
    """Run non-default Stream ordering and rank-skew validation."""
    _set_unique_bootstrap_endpoint(monkeypatch)
    torchrun_case(_WORKER, "test_all_gather_stream_order_and_visibility", num_proc=4)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_runtime_device_guard_on_real_npu(monkeypatch) -> None:
    """Run the Runtime device-drift guard on a host with at least two NPUs."""
    _set_unique_bootstrap_endpoint(monkeypatch)
    torchrun_case(_WORKER, "test_runtime_device_guard_on_real_npu", num_proc=1)
