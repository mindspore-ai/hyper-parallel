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
"""Unit tests for trainer lifecycle cleanup."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hyper_parallel.trainer.base import BaseTrainer


class TestBaseTrainerLifecycle(unittest.TestCase):
    """Verify BaseTrainer releases resources on every exit path."""

    @patch("hyper_parallel.trainer.base.destroy_process_group")
    @patch("hyper_parallel.trainer.base.synchronize")
    @patch("hyper_parallel.trainer.base.empty_cache")
    @patch("hyper_parallel.trainer.base.dist")
    def test_destroy_process_group_after_barrier_failure(
        self,
        mock_dist: MagicMock,
        mock_empty_cache: MagicMock,
        mock_synchronize: MagicMock,
        mock_destroy_process_group: MagicMock,
    ) -> None:
        """
        Feature: Distributed Trainer cleanup.
        Description: Simulate a barrier failure during process-group teardown.
        Expectation: The original error propagates after the process group is destroyed.
        """
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        mock_dist.barrier.side_effect = RuntimeError("barrier failed")
        trainer = SimpleNamespace()

        with self.assertRaisesRegex(RuntimeError, "barrier failed"):
            BaseTrainer.destroy_distributed(trainer)

        mock_empty_cache.assert_called_once_with()
        mock_synchronize.assert_not_called()
        mock_destroy_process_group.assert_called_once_with()

    @patch("hyper_parallel.trainer.base.synchronize")
    @patch("hyper_parallel.trainer.base.HyperIter")
    def test_train_cleans_up_after_step_failure(
        self,
        mock_hyper_iter: MagicMock,
        mock_synchronize: MagicMock,
    ) -> None:
        """
        Feature: Trainer lifecycle cleanup.
        Description: Raise a non-StopIteration error from the first training step.
        Expectation: The iterator and every finalizer run before the error propagates.
        """
        data_iterator = MagicMock()
        mock_hyper_iter.return_value = data_iterator
        trainer = SimpleNamespace(
            config=SimpleNamespace(
                dataloader=SimpleNamespace(
                    use_background_prefetcher=True,
                    drop_last=False,
                ),
            ),
            train_dataloader=MagicMock(),
            local_rank=0,
            state=SimpleNamespace(global_step=0, epoch=0),
            train_iters=1,
            train_steps=1,
            train_epochs=1,
            on_train_begin=MagicMock(),
            on_train_end=MagicMock(),
            on_epoch_begin=MagicMock(),
            on_epoch_end=MagicMock(),
            train_step=MagicMock(side_effect=RuntimeError("step failed")),
            destroy_distributed=MagicMock(),
        )

        with self.assertRaisesRegex(RuntimeError, "step failed"):
            BaseTrainer.train(trainer)

        data_iterator.stop.assert_called_once_with()
        trainer.on_train_end.assert_called_once_with()
        mock_synchronize.assert_called_once_with()
        trainer.destroy_distributed.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
