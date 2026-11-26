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
"""Regression tests for the VeRL adapter's inherited trainer lifecycle."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tests.ut.integration.verl._contracts import ADAPTERS

HYPER_PARALLEL_ENGINE_WITH_LMHEAD = ADAPTERS.engine.HyperParallelEngineWithLMHead
VERL_TRAINER_ADAPTER = ADAPTERS.trainer.VerlTrainerAdapter


class TestTrainerLifecycle(unittest.TestCase):
    """Verify optimizer boundaries in VeRL training and inference."""

    def test_native_optimizer_step_updates_once_after_clipping(self) -> None:
        """The native update owns clipping and one optimizer step, including failure handling."""
        trainer = object.__new__(VERL_TRAINER_ADAPTER)
        events = Mock()
        events.clip.return_value = 0.5
        trainer.clip_grad_norm = events.clip
        trainer.optimizer = events.optimizer
        self.assertEqual(trainer.optimizer_step(), 0.5)
        self.assertEqual([event[0] for event in events.mock_calls], ["clip", "optimizer.step"])
        trainer.optimizer.step.assert_called_once_with()
        events.reset_mock()
        events.clip.side_effect = RuntimeError("clipping failed")
        with self.assertRaisesRegex(RuntimeError, "clipping failed"):
            trainer.optimizer_step()
        trainer.optimizer.step.assert_not_called()

    def test_scheduler_container_and_missing_scheduler(self) -> None:
        """Advance the native container and read its optimizer's current learning rate."""
        trainer = VERL_TRAINER_ADAPTER.__new__(VERL_TRAINER_ADAPTER)
        scheduler = Mock()
        trainer.optimizer = SimpleNamespace(param_groups=[{"lr": 0.25}])
        scheduler.step.side_effect = lambda: trainer.optimizer.param_groups[0].update(lr=0.125)
        trainer.lr_scheduler = scheduler
        actual = trainer.lr_scheduler_step()
        self.assertEqual(actual, 0.125, f"lr={actual}, expected=0.125")
        scheduler.step.assert_called_once_with()
        trainer.lr_scheduler = None
        trainer.optimizer.param_groups[0]["lr"] = 0.25
        actual = trainer.lr_scheduler_step()
        self.assertEqual(actual, 0.25, f"lr={actual}, expected=0.25")

    def test_construction_selects_actor_and_reference_components(self) -> None:
        """Build an optimizer only for the training role."""
        config = SimpleNamespace(
            training=SimpleNamespace(train_iters=5, seed=None, enable_full_determinism=True),
            lr_scheduler=Mock(),
        )
        mesh = SimpleNamespace(device_mesh=object(), dp_rank=0, tp_rank=0)
        setup = SimpleNamespace(mesh_context=mesh)
        trainer_cls = VERL_TRAINER_ADAPTER
        for forward_only in (False, True):
            build_optimizer = Mock()
            build_scheduler = Mock()
            with (
                self.subTest(forward_only=forward_only),
                patch.multiple(
                    ADAPTERS.trainer, get_local_rank_safe=Mock(return_value=0),
                    get_global_rank_safe=Mock(return_value=0), get_world_size_safe=Mock(return_value=1),
                    get_device_type=Mock(return_value="cpu"), get_torch_device=Mock(),
                    setup_logging=Mock(), enable_high_precision_for_bf16=Mock(),
                    enable_third_party_logging=Mock(), set_seed=Mock(),
                    create_dataloader_parallel_context=Mock(),
                ),
                patch("torch.use_deterministic_algorithms") as deterministic,
                patch.multiple(
                    trainer_cls, _build_model=Mock(), _build_collate_fn=Mock(), _build_get_batch=Mock(),
                    _build_optimizer=build_optimizer,
                    _build_lr_scheduler=build_scheduler, _build_training_context=Mock(),
                    model=Mock(), model_config=Mock(), optimizer=Mock(), create=True,
                ),
            ):
                trainer = trainer_cls(config, distributed_setup=setup, tokenizer=Mock(), forward_only=forward_only)
                self.assertIs(trainer.mesh, mesh)
                self.assertEqual(trainer.train_iters, 5)
                self.assertIs(trainer.config, config)
                self.assertEqual(len(trainer.model_assets), 2)
                deterministic.assert_called_once_with(True, warn_only=False)
                if forward_only:
                    trainer.model.requires_grad_.assert_called_once_with(False)
                    trainer.model.eval.assert_called_once()
                    build_optimizer.assert_not_called()
                    build_scheduler.assert_not_called()
                else:
                    build_optimizer.assert_called_once()
                    build_scheduler.assert_called_once()
