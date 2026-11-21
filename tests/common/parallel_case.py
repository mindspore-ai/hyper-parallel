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
"""Wrappers that spawn ``torchrun`` ST workers.

Like :mod:`tests.common.distributed_launcher`, this module must stay free of
``torch`` / ``hyper_parallel`` imports so pytest parent launchers only pay the
cost of forking the distributed runner. Worker bodies live in non-launcher
modules and import frameworks as usual.
"""
from __future__ import annotations

import os
import signal
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import pytest

from tests.common.port_utils import allocate_port


class TorchCase:
    """torch case messages"""

    def __init__(
        self, file_name: str, case_name: str | tuple[str, ...],
        master_port: Optional[int] = None, num_proc: int = 1,
    ) -> None:
        """Initialize TorchCase with file path, case name, optional port, and process count."""
        self.file_name = file_name
        self.case_name = case_name
        self.master_port = master_port
        self.num_proc = num_proc


def _parallel_run_context():
    """Use spawn for CPU shards to avoid inheriting fork-only state."""
    device_type = os.environ.get("HYPER_PARALLEL_TEST_DEVICE_TYPE", "").strip().lower()
    if device_type == "cpu":
        return mp.get_context("spawn")
    return mp.get_context()


def run_case(visible_devices: list, case: TorchCase) -> None:
    """Run a single test case in a child process with device visibility set.

    Args:
        visible_devices: List of device indices to expose via ASCEND_RT_VISIBLE_DEVICES.
        case: The test case descriptor.
    """
    # become the leader of a new process group so that os.killpg on timeout
    # kills torchrun worker sub-processes as well as this wrapper
    os.setsid()
    if os.environ.get("HYPER_PARALLEL_TEST_DEVICE_TYPE", "").strip().lower() == "cpu":
        os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    else:
        # set visible devices for current case
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = ','.join(map(str, visible_devices))
    # Import the thin launcher only — never tests.torch.utils (imports torch /
    # torch_npu) in this wrapper process.
    # pylint: disable=C0415
    from tests.common.distributed_launcher import torchrun_case
    torchrun_case(case.file_name, case.case_name, case.master_port, case.num_proc)


def _auto_assign_ports(cases: list) -> None:
    """Assign unique ports to every case whose :attr:`master_port` is ``None``.

    Ports are allocated in the parent process so that concurrent children
    never race — each child receives a globally-unique, pre-allocated port.
    """
    for case in cases:
        if case.master_port is None:
            case.master_port = allocate_port()


def parallel_run(
    cases: list[TorchCase],
    global_num_proc: int = 8,
    *,
    device_ids: Optional[tuple[int, ...]] = None,
) -> None:
    """Run a group of test cases in parallel, assigning disjoint device slices to each.

    Args:
        cases: List of TorchCase descriptors to run concurrently.
            The sum of all ``num_proc`` values must not exceed ``global_num_proc``.
        global_num_proc: Total device budget for this group. Defaults to 8.
        device_ids: Ascend device IDs within the runner. Defaults to contiguous IDs from zero.

    Raises:
        AssertionError: If the total device count exceeds ``global_num_proc``, if any
            case times out (900 s deadline), or if any child process exits with a
            non-zero return code.
    """
    # auto-assign ports before spawning children (avoids cross-process races)
    unassigned = [c for c in cases if c.master_port is None]
    if unassigned:
        _auto_assign_ports(unassigned)

    available_devices = tuple(range(global_num_proc)) if device_ids is None else device_ids
    budget = min(global_num_proc, len(available_devices))
    required = sum(case.num_proc for case in cases)
    assert required <= budget, f"Cases need {required} devices, but the budget is {budget}"

    offset = 0
    assignments = []
    for case in cases:
        end = offset + case.num_proc
        assignments.append(list(available_devices[offset:end]))
        offset = end

    # create child process (run_case calls os.setsid to own a process group,
    # so os.killpg on timeout kills torchrun workers too)
    processes = []
    ctx = _parallel_run_context()
    for _, (case, devices) in enumerate(zip(cases, assignments)):
        p = ctx.Process(target=run_case, args=(devices, case))
        p.start()
        processes.append(p)

    # wait child process terminates (timeout=420s to prevent infinite hang on distributed deadlock)
    timed_out = []
    for i, p in enumerate(processes):
        p.join(timeout=420)
        if p.is_alive():
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.join()
            timed_out.append(cases[i].case_name)

    # check results for all cases
    if timed_out:
        raise AssertionError(f"Cases timed out (possible collective deadlock): {timed_out}")
    failed = [cases[i].case_name for i, p in enumerate(processes) if p.exitcode != 0]
    assert not failed, f"List cases failed: {failed}"


def _ascend_devices() -> tuple[int, ...]:
    """Resolve Ascend logical device IDs without importing the runtime."""
    budget = int(os.environ.get("HYPER_PARALLEL_TEST_CARD_COUNT", "8"))
    if not 1 <= budget <= 8:
        raise ValueError("HYPER_PARALLEL_TEST_CARD_COUNT must be between 1 and 8")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible is None:
        # Container device names can use physical IDs. CANN numbers exposed devices from zero.
        count = sum(
            device.name.removeprefix("davinci").isdecimal()
            for device in Path("/dev").glob("davinci[0-9]*")
        )
        return tuple(range(min(count, budget)))
    tokens = visible.split(",") if visible else []
    if any(not token.isascii() or not token.isdecimal() for token in tokens):
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES must be a comma-separated list of device IDs")
    devices = tuple(int(token) for token in tokens)
    if devices != tuple(sorted(set(devices))):
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES must list distinct device IDs in ascending order")
    return devices[:budget]


def parallel_run_ascend_batches(cases: list[TorchCase]) -> None:
    """Run all cases in order on visible devices, with a skip if any case cannot fit."""
    if not cases:
        return
    device_ids = _ascend_devices()
    budget = len(device_ids)
    required = max(case.num_proc for case in cases)
    if required > budget:
        pytest.skip(f"Requires {required} Ascend devices, but only {budget} are visible")

    batch = []
    used = 0
    for case in cases:
        if used + case.num_proc > budget:
            parallel_run(batch, global_num_proc=budget, device_ids=device_ids)
            batch = []
            used = 0
        batch.append(case)
        used += case.num_proc
    if batch:
        parallel_run(batch, global_num_proc=budget, device_ids=device_ids)
