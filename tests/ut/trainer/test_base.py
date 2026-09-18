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

from tests.common.mark_utils import arg_mark

_BATCHING_MODULE = ModuleType("hyper_parallel.data.batching")
_BATCHING_MODULE.build_dataloader = MagicMock()

# This module tests the training lifecycle, not dataloader construction. Stub
# that boundary so collecting the UT does not require the optional torchdata.
with patch.dict(sys.modules, {"hyper_parallel.data.batching": _BATCHING_MODULE}):
    base_module = importlib.import_module("hyper_parallel.trainer.base")


class TestBaseTrainerCleanup(unittest.TestCase):
    """Tests for cleaning up the data iterator on exceptional exits."""

    @staticmethod
    def _make_trainer() -> MagicMock:
        """Build the minimal trainer surface required by ``BaseTrainer.train``."""
        trainer = MagicMock()
        trainer.config = SimpleNamespace(
            dataloader=SimpleNamespace(
                use_background_prefetcher=True,
                drop_last=False,
            )
        )
        trainer.train_dataloader = object()
        trainer.local_rank = 0
        trainer.state = SimpleNamespace(global_step=0, epoch=0)
        trainer.train_iters = 1
        trainer.train_epochs = 1
        trainer.train_steps = 1
        return trainer

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="allcards",
        essential_mark="essential",
    )
    def test_train_stops_data_iterator_when_training_or_callback_raises(self) -> None:
        """
        Feature: BaseTrainer lifecycle cleanup.
        Description: Raise from training and callback hooks while background prefetching is enabled.
        Expectation: The data iterator is stopped once and each original exception propagates unchanged.
        """
        for failure_hook in ("train_step", "on_epoch_end"):
            with self.subTest(failure_hook=failure_hook):
                trainer = self._make_trainer()
                data_iterator = MagicMock()
                expected_error = RuntimeError(f"{failure_hook} failed")
                getattr(trainer, failure_hook).side_effect = expected_error

                with patch.object(base_module, "HyperIter", return_value=data_iterator):
                    with self.assertRaises(RuntimeError) as error_context:
                        base_module.BaseTrainer.train(trainer)

                self.assertIs(error_context.exception, expected_error)
                data_iterator.stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
