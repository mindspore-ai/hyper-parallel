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
"""One bounded CPU episode batch that can wait behind a backend policy gate."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
import logging
import os
import threading
from typing import Any, Callable, Coroutine, Sequence

from rl.dataset.contracts import PromptRecord


class PreparedProgramBatch:
    """Own an event loop for CPU programs, with explicit consumption and cancellation.

    The work callback must not execute distributed collectives or device operations.
    Model requests must wait at a version-aware gateway until the trainer admits them.
    """

    def __init__(self, prompts: Sequence[PromptRecord], policy_version: int,
                 work: Callable[[], Coroutine[Any, Any, Any]], *, cpu_fraction: float | None = None,
                 max_cpus: int | None = None) -> None:
        """Validate optional CPU limits, then start a batch and retain its ordered identity for later consumption.

        Args:
            prompts: Ordered prompts to prepare.
            policy_version: Policy required by this batch's model requests.
            work: CPU-only asynchronous episode callback.
            cpu_fraction: Fraction of caller-visible logical CPUs, rounded down with a minimum of one.
            max_cpus: Optional positive cap on that CPU subset; the stricter limit wins.

        Note:
            Affinity is inherited by CLI children and lasts for the whole episode.
            It does not reserve CPUs or enforce a CPU-time quota on the host.
        """
        self._cpu_affinity = None
        if cpu_fraction is not None:
            if (isinstance(cpu_fraction, bool) or not isinstance(cpu_fraction, (int, float))
                    or not 0 < cpu_fraction <= 1):
                raise ValueError("Preparation CPU fraction must be in (0, 1]")
        if max_cpus is not None and (isinstance(max_cpus, bool) or not isinstance(max_cpus, int) or max_cpus <= 0):
            raise ValueError("Preparation max CPUs must be a positive integer or null")
        if cpu_fraction is not None or max_cpus is not None:
            if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
                raise RuntimeError("CPU preparation affinity requires Linux affinity APIs")
            available = sorted(os.sched_getaffinity(0))
            count = max(1, int(len(available) * (cpu_fraction if cpu_fraction is not None else 1)))
            if max_cpus is not None:
                count = min(count, max_cpus)
            # Spread preparation across the available topology without changing trainer affinity.
            self._cpu_affinity = {available[index * len(available) // count] for index in range(count)}
        self._identity = self._prompt_identity(prompts)
        self._policy_version = policy_version
        self._work = work
        self._result: Future = Future()
        self._started = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._thread = threading.Thread(target=self._run, name="rl-agent-preparation", daemon=True)
        self._thread.start()
        if not self._started.wait(10):
            raise RuntimeError("Agent preparation thread did not start")

    @staticmethod
    def _prompt_identity(prompts: Sequence[PromptRecord]) -> tuple:
        return tuple((p.prompt_id, p.messages, p.ground_truth) for p in prompts)

    async def _execute(self) -> Any:
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()
        self._started.set()
        return await self._work()

    def _run(self) -> None:
        """Publish thread completion or cancellation to the consuming trainer."""
        try:
            if self._cpu_affinity is not None:
                # Linux affinity is per-thread; executor threads and CLI children inherit this budget.
                os.sched_setaffinity(0, self._cpu_affinity)
                logging.getLogger(__name__).info("CPU preparation uses %d logical CPUs; trainer affinity unchanged",
                                                 len(self._cpu_affinity))
            self._result.set_result(asyncio.run(self._execute()))
        # Thread exits, including cancellation, must wake the result consumer.
        except BaseException as error:  # pylint: disable=broad-exception-caught
            self._result.set_exception(error)
        finally:
            self._started.set()

    def result(self, prompts: Sequence[PromptRecord], policy_version: int) -> Any:
        """Consume only the originally prepared prompts and policy, propagating failures."""
        if policy_version != self._policy_version or self._prompt_identity(prompts) != self._identity:
            raise RuntimeError("Prepared agent batch does not match the requested prompts and policy")
        return self._result.result()

    def close(self) -> None:
        """Cancel CPU tasks and wait for their child-process and session cleanup."""
        if not self._result.done() and self._loop is not None and self._task is not None:
            try:
                self._loop.call_soon_threadsafe(self._task.cancel)
            except RuntimeError:
                if not self._loop.is_closed():
                    raise
        self._thread.join(timeout=60)
        if self._thread.is_alive():
            raise RuntimeError("Agent preparation did not finish cancellation cleanup")
