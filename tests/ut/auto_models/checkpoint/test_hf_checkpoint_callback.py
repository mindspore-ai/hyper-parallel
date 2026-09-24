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
"""Unit tests for Trainer-driven Hugging Face checkpoint export."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hyper_parallel.components.checkpoint.config import CheckpointingConfig
from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback
from hyper_parallel.trainer.state import TrainerState


class TestHuggingFaceCheckpointCallback(unittest.TestCase):
    """Verify DCP and Hugging Face checkpoint orchestration."""

    @staticmethod
    def _build_trainer(checkpoint_config: CheckpointingConfig) -> SimpleNamespace:
        """Build the minimum Trainer surface consumed by the callback."""
        model = MagicMock()
        model.state_dict.return_value = {"weight": MagicMock()}
        tokenizer = MagicMock()
        return SimpleNamespace(
            config=SimpleNamespace(checkpoint=checkpoint_config),
            mesh=None,
            model=model,
            optimizer=None,
            lr_scheduler=None,
            train_dataloader=None,
            tokenizer=tokenizer,
            processor=None,
            chat_template=None,
            train_steps=1,
        )

    @patch("hyper_parallel.trainer.callbacks.checkpoint_callback.CheckpointManager")
    @patch("hyper_parallel.trainer.callbacks.checkpoint_callback.build_checkpointer")
    def test_step_save_writes_dcp_and_hf_weights(
            self, build_checkpointer: MagicMock, checkpoint_manager: MagicMock
    ) -> None:
        """A scheduled save should persist DCP first and then export HF weights."""
        dcp = build_checkpointer.return_value
        checkpoint_manager.return_value.save_pretrained.return_value = True

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            config = CheckpointingConfig(
                save_ckpt=True,
                save_hf_weights=True,
                checkpoint_dir=checkpoint_dir,
                save_steps=1,
                save_epochs=0,
                save_optimizer=False,
                save_train_state=False,
            )
            trainer = self._build_trainer(config)
            callback = CheckpointerCallback(trainer)
            state = TrainerState(global_step=1)
            callback.on_step_end(state)
            callback.on_train_end(state)

            step_dir = os.path.join(checkpoint_dir, "global_step_1")
            dcp.save.assert_called_once()
            self.assertEqual(dcp.maybe_wait_for_async_save.call_count, 2)
            checkpoint_manager.return_value.save_pretrained.assert_called_once_with(
                os.path.join(step_dir, "hf_ckpt")
            )
            trainer.tokenizer.save_pretrained.assert_called_once_with(
                os.path.join(step_dir, "hf_ckpt")
            )

    @patch("hyper_parallel.trainer.callbacks.checkpoint_callback.CheckpointManager")
    @patch("hyper_parallel.trainer.callbacks.checkpoint_callback.build_checkpointer")
    def test_train_end_supports_hf_only_export(
            self, build_checkpointer: MagicMock, checkpoint_manager: MagicMock
    ) -> None:
        """A restored run may export HF weights without writing another DCP."""
        dcp = build_checkpointer.return_value
        checkpoint_manager.return_value.save_pretrained.return_value = True

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            config = CheckpointingConfig(
                save_ckpt=False,
                save_hf_weights=True,
                checkpoint_dir=checkpoint_dir,
            )
            trainer = self._build_trainer(config)
            callback = CheckpointerCallback(trainer)
            callback.on_train_end(TrainerState(global_step=7))

            dcp.save.assert_not_called()
            checkpoint_manager.return_value.save_pretrained.assert_called_once_with(
                os.path.join(checkpoint_dir, "global_step_7", "hf_ckpt")
            )

    @patch("hyper_parallel.trainer.callbacks.checkpoint_callback.build_checkpointer")
    def test_peft_hf_export_is_rejected(self, build_checkpointer: MagicMock) -> None:
        """PEFT export must not silently write an invalid full-model checkpoint."""
        config = CheckpointingConfig(
            save_ckpt=True,
            save_hf_weights=True,
            is_peft=True,
        )
        trainer = self._build_trainer(config)

        with self.assertRaisesRegex(ValueError, "does not yet support PEFT"):
            CheckpointerCallback(trainer)
        build_checkpointer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
