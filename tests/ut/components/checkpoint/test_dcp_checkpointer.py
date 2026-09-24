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
"""Unit tests for when the DCP checkpointer publishes its latest-checkpoint pointer."""

import os
import tempfile
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional
from unittest.mock import patch

from hyper_parallel.components.checkpoint.dcp_checkpointer import (
    LATEST_CHECKPOINT_FILE,
    STEP_PREFIX,
    DistributedCheckpointer,
)
from tests.common.mark_utils import arg_mark

_CHECKPOINTER_MODULE = "hyper_parallel.components.checkpoint.dcp_checkpointer"


def _pointer_value(checkpoint_dir: str) -> Optional[int]:
    """Return the step the pointer file names, or None while it does not exist."""
    pointer_path = os.path.join(checkpoint_dir, LATEST_CHECKPOINT_FILE)
    if not os.path.isfile(pointer_path):
        return None
    with open(pointer_path, encoding="utf-8") as handle:
        return int(handle.read().strip())


class TestDistributedCheckpointerPublish(unittest.TestCase):
    """Verify that a checkpoint is published as soon as it is durable."""

    @staticmethod
    def _state() -> Dict[str, Any]:
        """Return a payload shaped like the trainer's, small enough to ignore."""
        return {"model": {"weight": 1}, "extra_state": {"global_step": 7}}

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_async_save_publishes_from_its_persist_callback(self) -> None:
        """Publish the pointer when persistence completes, not at the next save.

        Feature: Asynchronous checkpoint publication.
        Description: Dispatch an async save and let its persist callback fire while no
            drain has happened, the way a run continues training after dispatching a save.
        Expectation: The pointer is unwritten while persistence is in flight, the callback
            publishes it on completion, and the later drain leaves it alone.
        """
        captured: Dict[str, Any] = {}

        def _fake_async_save(payload: Dict[str, Any], **kwargs: Any) -> SimpleNamespace:
            """Record the dispatch without persisting anything."""
            del payload
            captured.update(kwargs)
            completion: Future = Future()
            completion.set_result(None)
            return SimpleNamespace(persist_completion=completion)

        checkpointer = DistributedCheckpointer(extra_state_per_rank=False)
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            save_dir = os.path.join(checkpoint_dir, f"{STEP_PREFIX}7")
            with patch(f"{_CHECKPOINTER_MODULE}.dcp_async_save", _fake_async_save):
                checkpointer.save(save_dir, self._state(), global_step=7, save_async=True)

                in_flight = _pointer_value(checkpoint_dir)
                self.assertIsNone(
                    in_flight,
                    f"pointer must not be published before persistence completes, got={in_flight!r}",
                )

                publish: Optional[Callable[[], None]] = captured.get("callback")
                self.assertIsNotNone(
                    publish,
                    f"async save must be handed a persist callback, got kwargs={sorted(captured)}",
                )

                publish()
                published = _pointer_value(checkpoint_dir)
                self.assertEqual(
                    published,
                    7,
                    f"persist callback must publish the checkpoint: expected=7, got={published!r}",
                )

                checkpointer.maybe_wait_for_async_save()
                after_drain = _pointer_value(checkpoint_dir)
                self.assertEqual(
                    after_drain,
                    7,
                    f"draining must leave the published pointer alone: expected=7, got={after_drain!r}",
                )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_a_failed_async_save_is_raised_without_waiting(self) -> None:
        """Re-raise a failed async save on the asking thread, keeping its type.

        Feature: Early reporting of a failed asynchronous save.
        Description: Ask about an async save that is still running, then about one that
            has failed, the way a training loop asks once per step.
        Expectation: The unfinished save is left alone, and the failed one raises the
            exception it failed with rather than a flattened stand-in.
        """
        completion: Future = Future()

        def _fake_async_save(payload: Dict[str, Any], **kwargs: Any) -> SimpleNamespace:
            """Hand back a response whose future the test resolves by hand."""
            del payload, kwargs
            return SimpleNamespace(persist_completion=completion)

        checkpointer = DistributedCheckpointer(extra_state_per_rank=False)
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            save_dir = os.path.join(checkpoint_dir, f"{STEP_PREFIX}7")
            with patch(f"{_CHECKPOINTER_MODULE}.dcp_async_save", _fake_async_save):
                checkpointer.save(save_dir, self._state(), global_step=7, save_async=True)

            # Still running: asking must not block and must not raise.
            checkpointer.raise_for_failed_async_save()

            completion.set_exception(OSError("No space left on device"))
            with self.assertRaises(OSError) as ctx:
                checkpointer.raise_for_failed_async_save()
            self.assertIn(
                "No space left on device",
                str(ctx.exception),
                f"the original failure must reach the caller, got={ctx.exception!r}",
            )
            # Cleared, so the same failure is not reported twice.
            checkpointer.raise_for_failed_async_save()

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_a_finished_async_save_is_retired_by_whoever_notices_first(self) -> None:
        """Retire a save that succeeded without raising, leaving the drain nothing to do.

        Feature: Early reporting of a failed asynchronous save.
        Description: Ask about an async save that has completed successfully, the way the
            training loop asks once per step, then drain.
        Expectation: Nothing is raised and the drain turns into a no-op. A finished save
            needs no waiting, so the step that notices it is free to clear it --- what
            must not happen is a step raising on a save that went fine.
        """
        completion: Future = Future()
        completion.set_result(None)

        def _fake_async_save(payload: Dict[str, Any], **kwargs: Any) -> SimpleNamespace:
            """Hand back a response whose persistence already finished."""
            del payload, kwargs
            return SimpleNamespace(persist_completion=completion)

        checkpointer = DistributedCheckpointer(extra_state_per_rank=False)
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            save_dir = os.path.join(checkpoint_dir, f"{STEP_PREFIX}7")
            with patch(f"{_CHECKPOINTER_MODULE}.dcp_async_save", _fake_async_save):
                checkpointer.save(save_dir, self._state(), global_step=7, save_async=True)

            checkpointer.raise_for_failed_async_save()

            with patch(f"{_CHECKPOINTER_MODULE}.logger") as log:
                checkpointer.maybe_wait_for_async_save()
            waited = [call.args[0] for call in log.info.call_args_list if call.args]
            self.assertEqual(
                waited,
                [],
                f"a retired save leaves the drain nothing to wait on, logged={waited}",
            )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_sync_save_publishes_before_returning(self) -> None:
        """Keep the synchronous path publishing inline, under its barriers.

        Feature: Synchronous checkpoint publication.
        Description: Write a checkpoint with ``save_async=false``.
        Expectation: The pointer names the checkpoint by the time ``save`` returns.
        """
        checkpointer = DistributedCheckpointer(extra_state_per_rank=False)
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            save_dir = os.path.join(checkpoint_dir, f"{STEP_PREFIX}3")
            with patch(f"{_CHECKPOINTER_MODULE}.dcp_save") as fake_save:
                checkpointer.save(save_dir, self._state(), global_step=3, save_async=False)

            self.assertEqual(
                fake_save.call_count,
                1,
                f"sync save must reach the storage layer once, got={fake_save.call_count}",
            )
            published = _pointer_value(checkpoint_dir)
            self.assertEqual(
                published,
                3,
                f"sync save must publish before returning: expected=3, got={published!r}",
            )


if __name__ == "__main__":
    unittest.main()
