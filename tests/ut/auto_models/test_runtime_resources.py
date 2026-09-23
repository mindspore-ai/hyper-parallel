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
"""Model resource ordering, rollback and ownership without native hardware."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from torch import nn

from tests.common.mark_utils import arg_mark

from hyper_parallel.models.runtime_resources import ModelRuntimeModule, ModelRuntimeResources
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.text_trainer import TextTrainer
from hyper_parallel.trainer.vlm_trainer import VLMTrainer


class _Participant(nn.Module, ModelRuntimeModule):
    """Record lifecycle events and inject failures."""

    def __init__(self, events: list, name: str) -> None:
        """Bind a test event list and a participant name."""
        super().__init__()
        self.events = events
        self.name = name
        self.fail_prepare = False
        self.fail_close = False

    @staticmethod
    def prepare_runtime_group(modules: list[_Participant]) -> None:
        """Prepare all test participants or inject a failure."""
        modules[0].events.append(('prepare', tuple(module.name for module in modules)))
        if any(module.fail_prepare for module in modules):
            raise ValueError('prepare failure')

    def close_runtime(self) -> None:
        """Record cleanup or inject a retryable failure."""
        self.events.append(('close', self.name))
        if self.fail_close:
            raise ValueError('close failure')


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestModelRuntimeResources(unittest.TestCase):
    """Only registered participants receive hooks, in deterministic order."""

    def test_aliases_group_once_and_native_model_untouched(self):
        events = []
        first, second = _Participant(events, 'first'), _Participant(events, 'second')
        native = nn.Linear(2, 2)
        native.close = Mock(side_effect=AssertionError('not a participant'))
        model = nn.ModuleList([first, second, first, native])
        resources = ModelRuntimeResources(model)
        resources.prepare()
        resources.prepare()
        resources.close()
        resources.close()
        self.assertEqual(events, [('prepare', ('first', 'second')), ('close', 'second'), ('close', 'first')])
        native.close.assert_not_called()

    def test_prepare_failure_rolls_back_all_modules(self):
        events = []
        modules = nn.ModuleList([_Participant(events, 'a'), _Participant(events, 'b')])
        modules[1].fail_prepare = True
        resources = ModelRuntimeResources(modules)
        with self.assertRaisesRegex(ValueError, 'prepare failure'):
            resources.prepare()
        self.assertEqual(events[-2:], [('close', 'b'), ('close', 'a')])
        resources.close()
        self.assertEqual(len(events), 3)

    def test_close_failure_attempts_remaining_and_can_retry(self):
        events = []
        modules = nn.ModuleList([_Participant(events, 'a'), _Participant(events, 'b')])
        modules[1].fail_close = True
        resources = ModelRuntimeResources(modules)
        resources.prepare()
        with self.assertRaisesRegex(RuntimeError, 'Failed to close 1'):
            resources.close()
        self.assertEqual(events[-2:], [('close', 'b'), ('close', 'a')])
        modules[1].fail_close = False
        resources.close()
        self.assertEqual(events[-1], ('close', 'b'))

    def test_trainer_closes_after_callbacks_before_process_group(self):
        events = []
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.state = None
        trainer._callbacks = [Mock(on_train_end=lambda state: events.append('callback'))]
        trainer.model_runtime_resources = ModelRuntimeResources(nn.ModuleList([_Participant(events, 'expert')]))
        trainer.model_runtime_resources.prepare()
        with (patch('hyper_parallel.trainer.base.dist') as distributed,
              patch('hyper_parallel.trainer.base.empty_cache'),
              patch('hyper_parallel.trainer.base.synchronize'),
              patch('hyper_parallel.trainer.base.destroy_process_group', side_effect=lambda: events.append('destroy'))):
            distributed.is_initialized.return_value = True
            trainer.on_train_end()
            trainer.destroy_distributed()
        self.assertEqual(events[-3:], ['callback', ('close', 'expert'), 'destroy'])

    def test_composed_trainers_close_on_failure(self):
        """Text/VLM own their loops and must both clean up the composed base."""
        for trainer_type in (TextTrainer, VLMTrainer):
            with self.subTest(trainer=trainer_type.__name__):
                events = []
                trainer = trainer_type.__new__(trainer_type)
                trainer.base = BaseTrainer.__new__(BaseTrainer)
                resources = ModelRuntimeResources(nn.ModuleList([_Participant(events, 'expert')]))
                resources.prepare()
                trainer.base.model_runtime_resources = resources
                trainer._train = Mock(side_effect=ValueError('step failed'))
                with self.assertRaisesRegex(ValueError, 'step failed'):
                    trainer.train()
                self.assertEqual(events[-1], ('close', 'expert'))

    def test_shared_model_build_prepares_resources(self):
        """The stage used by both composed Trainers prepares the final model."""
        events = []
        model = nn.ModuleList([_Participant(events, 'expert')])
        model.config = SimpleNamespace()
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.global_rank = 1
        trainer.distributed_setup = None
        trainer.mesh = None
        trainer.config = SimpleNamespace(
            model=Mock(build=Mock(return_value=model)), peft=None,
            activation_checkpoint=SimpleNamespace(mode='off', selection=None, swap_inputs=False),
            activation_swap=None, compile=None, model_init_dtype=None,
            accelerator=SimpleNamespace(loss_parallel=False),
            debug=SimpleNamespace(check_fsdp_runtime=False),
        )
        with patch('hyper_parallel.trainer.base.model_integration_runtime.build_model_integration_session'):
            trainer._build_model()
        self.assertEqual(events, [('prepare', ('expert',))])
        trainer._close_model_runtime_resources()
        self.assertEqual(events[-1], ('close', 'expert'))
