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
"""Unit tests for CommComputeOverlap's persistent backward worker."""
import threading

import pytest

from hyper_parallel.core.pipeline_parallel.comm_compute_overlap import CommComputeOverlap
from hyper_parallel.core.pipeline_parallel.hook_coordinator import HookRole


def test_reuses_backward_worker_thread():
    """Multiple overlap windows should reuse one persistent BWD worker thread."""
    overlap = CommComputeOverlap()
    worker_ids = []
    main_id = threading.get_ident()
    try:
        for _ in range(3):
            overlap.run(lambda: None, lambda: worker_ids.append(threading.get_ident()))

        assert len(set(worker_ids)) == 1
        assert worker_ids[0] != main_id
    finally:
        overlap.close()


def test_backward_exception_propagates_and_worker_survives():
    """A BWD exception is re-raised on the caller while the worker stays reusable."""
    overlap = CommComputeOverlap()
    worker_ids = []

    def _raising_bwd():
        worker_ids.append(threading.get_ident())
        raise ValueError("bwd failed")

    try:
        with pytest.raises(RuntimeError) as exc_info:
            overlap.run(lambda: None, _raising_bwd)
        assert isinstance(exc_info.value.__cause__, ValueError)

        overlap.run(lambda: None, lambda: worker_ids.append(threading.get_ident()))
        assert len(set(worker_ids)) == 1
    finally:
        overlap.close()


def test_forward_exception_unblocks_waiting_backward_worker():
    """A FWD exception should abort rendezvous so the persistent BWD worker drains."""
    overlap = CommComputeOverlap()
    bwd_started = threading.Event()
    bwd_finished = threading.Event()

    def _blocked_bwd():
        bwd_started.set()
        try:
            overlap.coordinator.rendezvous(HookRole.COMPUTE)
        finally:
            bwd_finished.set()

    def _raising_fwd():
        if not bwd_started.wait(timeout=5.0):
            raise AssertionError("backward worker did not start")
        raise ValueError("fwd failed")

    try:
        with pytest.raises(ValueError):
            overlap.run(_raising_fwd, _blocked_bwd)
        assert bwd_finished.is_set()
    finally:
        overlap.close()
