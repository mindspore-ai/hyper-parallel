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
"""Unit tests for checkpoint retention in CheckpointerCallback."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

from hyper_parallel.components.checkpoint.config import CheckpointingConfig
from hyper_parallel.trainer.callbacks.base import TrainerState
from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback


class TestCheckpointCallback(unittest.TestCase):
    """Verify elapsed-time saving and safe checkpoint retention."""

    @staticmethod
    def _build_callback(
            checkpoint_dir: str,
            limit: Optional[int] = None,
            save_steps: int = 0,
            save_time_interval_minutes: float = 0.0,
    ) -> CheckpointerCallback:
        """Build the minimum callback surface needed for retention tests."""
        trainer = SimpleNamespace(
            config=SimpleNamespace(
                checkpoint=CheckpointingConfig(
                    save_ckpt=True,
                    checkpoint_dir=checkpoint_dir,
                    save_steps=save_steps,
                    save_epochs=0,
                    save_time_interval_minutes=save_time_interval_minutes,
                    save_total_limit=limit,
                )
            ),
            mesh=None,
            model=MagicMock(),
        )
        with patch(
                "hyper_parallel.trainer.callbacks.checkpoint_callback.build_checkpointer"
        ):
            return CheckpointerCallback(trainer)

    def test_time_interval_saves_at_next_completed_step(self) -> None:
        """An elapsed interval should save once at the next optimizer step."""
        with tempfile.TemporaryDirectory() as temp_dir:
            callback = self._build_callback(
                temp_dir,
                save_time_interval_minutes=1.0,
            )
            state = TrainerState()
            saved_steps = []
            with (
                patch.object(callback, "_load_checkpoint"),
                patch.object(
                    callback,
                    "_save_checkpoint",
                    side_effect=lambda current_state: saved_steps.append(
                        current_state.global_step
                    ),
                ),
                patch(
                    "hyper_parallel.trainer.callbacks.checkpoint_callback.time.monotonic",
                    side_effect=[100.0, 159.0, 160.0, 160.0, 219.0, 220.0, 220.0],
                ),
            ):
                callback.on_train_begin(state)
                for step in range(1, 5):
                    state.global_step = step
                    callback.on_step_end(state)

            self.assertEqual(saved_steps, [2, 4])

    def test_step_and_time_cadences_save_once_on_same_step(self) -> None:
        """Coincident cadence triggers should produce only one checkpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            callback = self._build_callback(
                temp_dir,
                save_steps=5,
                save_time_interval_minutes=1.0,
            )
            state = TrainerState(global_step=5)
            with (
                patch.object(callback, "_load_checkpoint"),
                patch.object(callback, "_save_checkpoint") as save_checkpoint,
                patch(
                    "hyper_parallel.trainer.callbacks.checkpoint_callback.time.monotonic",
                    side_effect=[100.0, 160.0, 160.0],
                ),
            ):
                callback.on_train_begin(state)
                callback.on_step_end(state)

            save_checkpoint.assert_called_once_with(state)

    def test_time_cadence_uses_distributed_consensus(self) -> None:
        """One rank reaching the deadline should make every rank save together."""
        with tempfile.TemporaryDirectory() as temp_dir:
            callback = self._build_callback(
                temp_dir,
                save_time_interval_minutes=1.0,
            )
            callback._next_time_save = 160.0  # pylint: disable=protected-access
            with (
                patch(
                    "hyper_parallel.trainer.callbacks.checkpoint_callback.time.monotonic",
                    return_value=159.0,
                ),
                patch("torch.distributed.is_available", return_value=True),
                patch("torch.distributed.is_initialized", return_value=True),
                patch(
                    "hyper_parallel.trainer.callbacks.checkpoint_callback.all_reduce",
                    return_value=1,
                ) as reduce_due,
            ):
                self.assertTrue(
                    callback._time_checkpoint_due()  # pylint: disable=protected-access
                )

            reduce_due.assert_called_once_with(0, op="max")

    @staticmethod
    def _make_complete_checkpoint(root: Path, step: int) -> Path:
        """Create a completed DCP-shaped checkpoint directory."""
        checkpoint = root / f"global_step_{step}"
        checkpoint.mkdir()
        (checkpoint / ".metadata").touch()
        return checkpoint

    def test_rotate_keeps_newest_checkpoints_and_unmanaged_paths(self) -> None:
        """Rotation should remove only the oldest completed managed directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_checkpoint = self._make_complete_checkpoint(root, 1)
            kept_checkpoint = self._make_complete_checkpoint(root, 2)
            active_checkpoint = self._make_complete_checkpoint(root, 3)
            incomplete_checkpoint = root / "global_step_4"
            incomplete_checkpoint.mkdir()
            unrelated_checkpoint = root / "global_step_5_backup"
            unrelated_checkpoint.mkdir()
            unrelated_file = root / "notes.txt"
            unrelated_file.touch()

            callback = self._build_callback(temp_dir, limit=2)
            callback._last_saved_step = 3  # pylint: disable=protected-access
            callback._rotate_checkpoints()  # pylint: disable=protected-access

            self.assertFalse(old_checkpoint.exists())
            self.assertTrue(kept_checkpoint.is_dir())
            self.assertTrue(active_checkpoint.is_dir())
            self.assertTrue(incomplete_checkpoint.is_dir())
            self.assertTrue(unrelated_checkpoint.is_dir())
            self.assertTrue(unrelated_file.is_file())

    def test_rotate_skips_checkpoint_symlink(self) -> None:
        """A checkpoint-shaped symlink must never cause its target to be deleted."""
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as target_dir:
            root = Path(temp_dir)
            external_target = Path(target_dir)
            marker = external_target / ".metadata"
            marker.touch()
            symlink = root / "global_step_1"
            os.symlink(external_target, symlink)
            self._make_complete_checkpoint(root, 2)

            callback = self._build_callback(temp_dir, limit=1)
            callback._last_saved_step = 2  # pylint: disable=protected-access
            callback._rotate_checkpoints()  # pylint: disable=protected-access

            self.assertTrue(symlink.is_symlink())
            self.assertTrue(marker.is_file())

    def test_rotate_recognizes_completed_hf_only_checkpoint(self) -> None:
        """A standard HF weight file should make an HF-only step eligible for rotation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_checkpoint = root / "global_step_1"
            hf_dir = old_checkpoint / "hf_ckpt"
            hf_dir.mkdir(parents=True)
            (hf_dir / "model.safetensors").touch()
            active_checkpoint = self._make_complete_checkpoint(root, 2)

            callback = self._build_callback(temp_dir, limit=1)
            callback._last_saved_step = 2  # pylint: disable=protected-access
            callback._rotate_checkpoints()  # pylint: disable=protected-access

            self.assertFalse(old_checkpoint.exists())
            self.assertTrue(active_checkpoint.is_dir())

    def test_rotate_preserves_current_checkpoint_when_steps_go_backwards(self) -> None:
        """The checkpoint written by the active run must survive stale higher steps."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            active_checkpoint = self._make_complete_checkpoint(root, 50)
            stale_checkpoint = self._make_complete_checkpoint(root, 100)

            callback = self._build_callback(temp_dir, limit=1)
            callback._last_saved_step = 50  # pylint: disable=protected-access
            callback._rotate_checkpoints()  # pylint: disable=protected-access

            self.assertTrue(active_checkpoint.is_dir())
            self.assertFalse(stale_checkpoint.exists())

    def test_non_positive_limit_disables_rotation(self) -> None:
        """A non-positive limit should preserve every completed checkpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoints = [self._make_complete_checkpoint(root, step) for step in range(1, 4)]

            callback = self._build_callback(temp_dir, limit=0)
            callback._last_saved_step = 3  # pylint: disable=protected-access
            callback._rotate_checkpoints()  # pylint: disable=protected-access

            self.assertTrue(all(checkpoint.is_dir() for checkpoint in checkpoints))


if __name__ == "__main__":
    unittest.main()
