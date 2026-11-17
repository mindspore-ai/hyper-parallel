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
"""Keep distributed-data logic tests independent of installed NPU hardware."""

import sys
from concurrent.futures import Future, wait
from queue import Queue
from threading import Thread

from typing import Any
import pytest
import torch


@pytest.fixture(autouse=True)
def cpu_data_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable NPU discovery and device hooks injected into CPU DataLoader workers.

    Args:
        monkeypatch: Pytest patch context for process and device substitutes.
    """
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "")
    npu = sys.modules.get("torch_npu")
    if npu is not None:
        monkeypatch.setattr(torch.npu, "is_available", lambda: False)
        # Older torch_npu versions run these hooks even for CPU-only workers.
        monkeypatch.setattr(torch.npu, "synchronize", lambda: None)
        monkeypatch.setattr(npu._C, "_npu_set_thread_affinity", lambda *_: None)
        monkeypatch.setattr(npu._C, "_npu_reset_thread_affinity", lambda: None)


class _LocalProcessStandIn:
    """Model process readiness with a separate source thread so mocks stay local."""

    def __init__(self, source, transport):
        source.bind_transport(transport)
        self.source = source
        self.requests = Queue()
        self.thread = Thread(target=self._run, name="hp-test-data-source", daemon=True)
        self.thread.start()
        self.future = None
        self.closed = False

    def _run(self) -> None:
        while (future := self.requests.get()) is not None:
            try:
                future.set_result(next(self.source))
            except BaseException as exc:
                future.set_exception(exc)

    def prefetch(self) -> None:
        """Request one mock-process result."""
        if self.future is None:
            self.future = Future()
            self.requests.put(self.future)

    def is_ready(self, timeout: float = 0.0) -> bool:
        """Wait briefly for a result without blocking staging.

        Args:
            timeout: Maximum seconds to wait for a ready result.
        """
        return bool(wait([self.future], timeout=timeout).done)

    def __next__(self):
        self.prefetch()
        future, self.future = self.future, None
        return future.result()

    def set_epoch(self, epoch: int) -> None:
        """Reset the test source epoch.

        Args:
            epoch: Non-negative epoch to select before the next source read.
        """
        self.source.set_epoch(epoch)

    def reset_iteration(self) -> None:
        """Re-enter the test source."""
        self.source.reset_iteration()

    def close(self) -> None:
        """Drain the owned thread and release test readers."""
        if not self.closed:
            self.requests.put(None)
            self.thread.join()
            self.source.close()
            self.closed = True


@pytest.fixture(autouse=True)
def inline_data_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep logic mocks local; dedicated process tests exercise real IPC separately.

    Args:
        monkeypatch: Pytest patch context for process and device substitutes.
    """
    workers = []

    def create(source: Any, transport: Any) -> _LocalProcessStandIn:
        """Create and register a deterministic test worker.

        Args:
            source: Serializable, unstarted CPU source with native or external preparation.
            transport: Data-plane collectives owned by this source process.
        """
        worker = _LocalProcessStandIn(source, transport)
        workers.append(worker)
        return worker

    monkeypatch.setattr("hyper_parallel.distributed_data.distributed_dataloader.create_data_worker", create)
    monkeypatch.setattr("hyper_parallel.distributed_data.packed_balancing.create_data_worker", create)
    yield
    for worker in workers:
        worker.close()
