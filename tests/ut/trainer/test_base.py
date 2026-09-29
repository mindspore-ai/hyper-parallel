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

import importlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch


def _load_base_trainer():
    """Import BaseTrainer without requiring the unrelated torchdata dependency."""
    batching_module = ModuleType("hyper_parallel.data.batching")
    batching_module.build_dataloader = MagicMock(name="build_dataloader")
    with patch.dict(sys.modules, {"hyper_parallel.data.batching": batching_module}):
        base_module = importlib.import_module("hyper_parallel.trainer.base")
    return base_module, base_module.BaseTrainer


class TestBaseTrainerLifecycle(unittest.TestCase):
    """Verify BaseTrainer releases resources on every exit path."""

    def test_destroy_process_group_after_barrier_failure(self) -> None:
        """
        Feature: Distributed Trainer cleanup.
        Description: Simulate a barrier failure during process-group teardown.
        Expectation: The original error propagates after the process group is destroyed.
        """
        base_module, base_trainer = _load_base_trainer()
        with (
            patch.object(base_module, "destroy_process_group") as mock_destroy_process_group,
            patch.object(base_module, "synchronize") as mock_synchronize,
            patch.object(base_module, "empty_cache") as mock_empty_cache,
            patch.object(base_module, "dist") as mock_dist,
        ):
            mock_dist.is_available.return_value = True
            mock_dist.is_initialized.return_value = True
            mock_dist.barrier.side_effect = RuntimeError("barrier failed")
            trainer = SimpleNamespace()

            with self.assertRaisesRegex(RuntimeError, "barrier failed"):
                base_trainer.destroy_distributed(trainer)

            mock_empty_cache.assert_called_once_with()
            mock_synchronize.assert_not_called()
            mock_destroy_process_group.assert_called_once_with()

    def test_train_cleans_up_after_step_failure(self) -> None:
        """
        Feature: Trainer lifecycle cleanup.
        Description: Raise a non-StopIteration error from the first training step.
        Expectation: The iterator and every finalizer run before the error propagates.
        """
        base_module, base_trainer = _load_base_trainer()
        with (
            patch.object(base_module, "synchronize") as mock_synchronize,
            patch.object(base_module, "HyperIter") as mock_hyper_iter,
        ):
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
                base_trainer.train(trainer)

            data_iterator.stop.assert_called_once_with()
            trainer.on_train_end.assert_called_once_with()
            mock_synchronize.assert_called_once_with()
            trainer.destroy_distributed.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
