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
"""Unit tests for the Trainer checkpoint callback's restore paths."""

import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple
from unittest.mock import patch

from hyper_parallel.components.checkpoint.config import CheckpointingConfig
from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback
from hyper_parallel.trainer.state import TrainerState
from tests.common.mark_utils import arg_mark

_CALLBACK_MODULE = "hyper_parallel.trainer.callbacks.checkpoint_callback"
_SUCCESS_LOG = "Checkpoint loaded successfully"


class _FakeStateful:
    """Carry a state dict the way a model, optimizer or scheduler would."""

    def __init__(self, state: Optional[Dict[str, Any]] = None) -> None:
        """Seed the state a save would collect."""
        self._state = dict(state or {})
        self.loaded: Optional[Dict[str, Any]] = None

    def state_dict(self) -> Dict[str, Any]:
        """Return the state a save collects or a load fills in place."""
        return dict(self._state)

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """Record what a restore handed back."""
        del strict
        self.loaded = dict(state_dict)


class _FakeDataLoader(_FakeStateful):
    """A stateful loader whose length sets the steps-per-epoch divisor."""

    def __init__(self, length: int) -> None:
        """Fix the epoch length the resume position is computed against."""
        super().__init__({"consumed_samples": 0})
        self._length = length

    def __len__(self) -> int:
        """Return the optimizer steps one epoch contains."""
        return self._length


class _FakeCheckpointer:
    """Replay a fixed payload instead of touching a real DCP checkpoint."""

    def __init__(self, extra_state: Optional[Dict[str, Any]] = None) -> None:
        """Hold the extra_state bundle a restore should read back, if any."""
        self.extra_state = extra_state
        self.skeleton_seen: Optional[Dict[str, Any]] = None
        self.async_failure: Optional[BaseException] = None

    def load(
        self,
        path: str,
        state: Dict[str, Any],
        *,
        strict_model: bool = True,
        extra_state_skeleton: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fill ``state`` only when the caller asked for the extra_state bundle."""
        del path, strict_model
        self.skeleton_seen = extra_state_skeleton
        if extra_state_skeleton is not None:
            state["extra_state"] = dict(self.extra_state or {})
        return state

    def maybe_wait_for_async_save(self) -> None:
        """No asynchronous save is ever in flight in these tests."""

    def raise_for_failed_async_save(self) -> None:
        """Report the failure a test planted, the way the real backend would."""
        if self.async_failure is not None:
            raise self.async_failure


class TestCheckpointerCallbackRestore(unittest.TestCase):
    """Verify what each restore mode reads back into the trainer."""

    @staticmethod
    def _build_trainer(restore_from: str, restore_train_state: bool) -> SimpleNamespace:
        """Build the callback's minimal Trainer dependency surface."""
        checkpoint = CheckpointingConfig(
            save_ckpt=False,
            checkpoint_dir=restore_from,
            restore_from=restore_from,
            restore_optimizer=True,
            restore_train_state=restore_train_state,
        )
        return SimpleNamespace(
            mesh=None,
            config=SimpleNamespace(checkpoint=checkpoint, model_init_dtype=None),
            state=TrainerState(global_step=0, epoch=0),
            model=_FakeStateful({"weight": 1}),
            optimizer=_FakeStateful({"state": {}}),
            lr_scheduler=_FakeStateful({"last_epoch": 0}),
            # Deliberately unequal: train_steps counts optimizer steps per epoch,
            # the loader counts micro-batches, and only the former sets the
            # position. Deriving it from len(loader) would report (0, 7) below.
            train_dataloader=_FakeDataLoader(8),
            train_steps=4,
        )

    @staticmethod
    def _restore(trainer: SimpleNamespace, checkpointer: _FakeCheckpointer) -> Any:
        """Run the callback's restore hook and return the stubbed module logger."""
        with patch(f"{_CALLBACK_MODULE}.build_checkpointer", return_value=checkpointer), \
                patch(f"{_CALLBACK_MODULE}.initialize_optimizer_state", return_value=True), \
                patch(f"{_CALLBACK_MODULE}.logger") as log:
            callback = CheckpointerCallback(trainer)
            callback.on_train_begin(trainer.state)
        return log

    def _reported_position(self, log: Any) -> Tuple[int, int, int]:
        """Return the (global_step, start_epoch, start_step) the success log reports.

        The log is the only consumer of the resume position now, so it is also the
        only place a test can observe it.
        """
        for call in log.info.call_args_list:
            if call.args and str(call.args[0]).startswith(_SUCCESS_LOG):
                return call.args[2], call.args[3], call.args[4]
        emitted = [str(call.args[0])[:40] for call in log.info.call_args_list if call.args]
        raise AssertionError(f"no {_SUCCESS_LOG!r} log was emitted, got={emitted}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_weights_only_restore_reports_a_fresh_position(self) -> None:
        """Load weights without the extra_state bundle and still report a position.

        Feature: Weights-only checkpoint restore.
        Description: Restore with ``restore_train_state=false``, the mode the checkpointer
            itself recommends when a checkpoint carries no training state.
        Expectation: The restore completes, the weights are applied, and it reports step 0
            rather than failing on state that only the extra_state path would produce.
        """
        with tempfile.TemporaryDirectory() as restore_from:
            trainer = self._build_trainer(restore_from, restore_train_state=False)
            checkpointer = _FakeCheckpointer()
            log = self._restore(trainer, checkpointer)

        self.assertIsNone(
            checkpointer.skeleton_seen,
            f"weights-only restore must not request an extra_state skeleton, "
            f"got={checkpointer.skeleton_seen!r}",
        )
        self.assertEqual(
            trainer.model.loaded,
            {"weight": 1},
            f"weights were not applied: expected={{'weight': 1}}, got={trainer.model.loaded!r}",
        )
        self.assertEqual(
            trainer.state.global_step,
            0,
            f"weights-only restore must leave progress fresh: expected=0, "
            f"got={trainer.state.global_step}",
        )
        reported = self._reported_position(log)
        self.assertEqual(
            reported,
            (0, 0, 0),
            f"weights-only restore must report a fresh position: expected=(0, 0, 0), got={reported}",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_train_state_restore_maps_the_step_onto_an_epoch_position(self) -> None:
        """Map a restored global step onto its epoch and offset within that epoch.

        Feature: Checkpoint restore of training progress.
        Description: Restore step 7 of epoch 1 with ``restore_train_state=true``, against a
            four-optimizer-step epoch fed by an eight-micro-batch loader.
        Expectation: Progress is restored, the reported position is the loop's own
            ``(state.epoch, global_step - epoch * train_steps)``, and the trainer gains no
            attribute carrying it.
        """
        with tempfile.TemporaryDirectory() as restore_from:
            trainer = self._build_trainer(restore_from, restore_train_state=True)
            checkpointer = _FakeCheckpointer(
                {
                    "global_step": 7,
                    "epoch": 1,
                    "lr_scheduler": {},
                    "train_dataloader": {},
                    "rng_state": {},
                }
            )
            log = self._restore(trainer, checkpointer)

        self.assertEqual(
            trainer.state.global_step,
            7,
            f"restored step mismatch: expected=7, got={trainer.state.global_step}",
        )
        reported = self._reported_position(log)
        self.assertEqual(
            reported,
            (7, 1, 3),
            f"reported resume position mismatch: expected=(7, 1, 3), got={reported}",
        )
        # The loops derive their own position from state, so a restore that also
        # parked it on the trainer would be reintroducing state nobody reads.
        carried = [name for name in ("start_epoch", "start_step") if hasattr(trainer, name)]
        self.assertEqual(
            carried,
            [],
            f"restore must not carry the resume position on the trainer, got={carried}",
        )


class TestCheckpointerCallbackAsyncFailure(unittest.TestCase):
    """Verify when a failed asynchronous save reaches the training loop."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_step_end_reports_a_failed_async_save_before_the_next_save(self) -> None:
        """Raise a failed async save at the next step, not at the next save.

        Feature: Early reporting of a failed asynchronous save.
        Description: End a step that the save cadence does not land on, while an async
            save has already failed.
        Expectation: The step raises that failure, so a long ``save_steps`` interval
            cannot spend thousands of steps on a run whose checkpoint is already lost.
        """
        checkpointer = _FakeCheckpointer()
        trainer = SimpleNamespace(
            mesh=None,
            config=SimpleNamespace(
                checkpoint=CheckpointingConfig(
                    save_ckpt=True,
                    save_steps=1000,
                    checkpoint_dir="unused",
                    is_async=True,
                ),
                model_init_dtype=None,
            ),
            state=TrainerState(global_step=7, epoch=0),
        )
        with patch(f"{_CALLBACK_MODULE}.build_checkpointer", return_value=checkpointer):
            callback = CheckpointerCallback(trainer)

        checkpointer.async_failure = OSError("No space left on device")
        with self.assertRaises(OSError) as ctx:
            callback.on_step_end(trainer.state)
        self.assertIn(
            "No space left on device",
            str(ctx.exception),
            f"the async failure must reach the training loop, got={ctx.exception!r}",
        )


if __name__ == "__main__":
    unittest.main()
