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
"""Unit tests for the trainer checkpoint callback."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback
from hyper_parallel.trainer.state import TrainerState


def _checkpoint_config(restore_from: str) -> SimpleNamespace:
    """Build the checkpoint configuration used by callback unit tests."""
    return SimpleNamespace(
        save_ckpt=False,
        checkpoint_dir=restore_from,
        save_steps=0,
        save_epochs=0,
        is_async=False,
        is_peft=False,
        save_optimizer=False,
        save_train_state=False,
        save_extra_state_per_rank=False,
        restore_from=restore_from,
        restore_optimizer=False,
        restore_train_state=False,
    )


class TestCheckpointerCallback(unittest.TestCase):
    """Tests for checkpoint restore policy and trainer state handling."""

    @patch("hyper_parallel.trainer.callbacks.checkpoint_callback.empty_cache")
    @patch("hyper_parallel.trainer.callbacks.checkpoint_callback.validate_model_init_dtype")
    @patch("hyper_parallel.trainer.callbacks.checkpoint_callback.build_checkpointer")
    def test_weights_only_restore_initializes_fresh_start_position(
            self,
            mock_build_checkpointer: MagicMock,
            mock_validate_model_init_dtype: MagicMock,
            mock_empty_cache: MagicMock,
    ) -> None:
        """Weights-only restore should retain fresh progress and define its start position."""
        with tempfile.TemporaryDirectory() as restore_path:
            checkpointer = MagicMock()
            mock_build_checkpointer.return_value = checkpointer
            trainer = SimpleNamespace(
                config=SimpleNamespace(
                    checkpoint=_checkpoint_config(restore_path),
                    model_init_dtype="float32",
                ),
                mesh=MagicMock(),
                model=torch.nn.Linear(4, 4),
                optimizer=None,
                lr_scheduler=None,
                train_dataloader=[object()] * 4,
                train_steps=4,
                state=TrainerState(),
            )
            callback = CheckpointerCallback(trainer)

            callback._load_checkpoint()

        self.assertEqual(trainer.state.global_step, 0)
        self.assertEqual(trainer.state.epoch, 0)
        self.assertEqual(trainer.start_epoch, 0)
        self.assertEqual(trainer.start_step, 0)
        checkpointer.load.assert_called_once()
        mock_validate_model_init_dtype.assert_called_once_with(
            trainer.model,
            "float32",
        )
        mock_empty_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
