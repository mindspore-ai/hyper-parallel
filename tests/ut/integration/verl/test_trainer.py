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
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import Mock, patch

# VeRL is an optional integration dependency.
if find_spec("verl") is not None:
    from hyper_parallel.integration.verl.engine.transformer_impl import (
        HyperParallelEngineWithLMHead as HYPER_PARALLEL_ENGINE_WITH_LMHEAD,
    )
    from hyper_parallel.integration.verl.engine.trainer import VerlTrainerAdapter as VERL_TRAINER_ADAPTER
else:
    HYPER_PARALLEL_ENGINE_WITH_LMHEAD = None
    VERL_TRAINER_ADAPTER = None


@unittest.skipUnless(find_spec("verl") is not None, "requires VeRL")
class TestTrainerLifecycle(unittest.TestCase):
    """Verify optimizer boundaries in VeRL training and inference."""

    def test_forward_only_does_not_update_optimizer(self) -> None:
        """Reference inference uses the forward-only execution path."""
        engine = object.__new__(HYPER_PARALLEL_ENGINE_WITH_LMHEAD)
        engine.forward_backward_batch = Mock(return_value={"metrics": {}})
        engine.optimizer_step = Mock()
        engine.optimizer_zero_grad = Mock()
        engine.infer_batch({})
        engine.optimizer_step.assert_not_called()
        engine.optimizer_zero_grad.assert_not_called()
        engine.forward_backward_batch.assert_called_once_with({}, None, forward_only=True)

    def test_train_batch_updates_optimizer_once(self) -> None:
        """Use VeRL's native step to apply one optimizer update."""
        engine = object.__new__(HYPER_PARALLEL_ENGINE_WITH_LMHEAD)
        events = Mock()
        engine.engine_config = SimpleNamespace(forward_only=False)
        engine.optimizer_zero_grad = events.zero
        engine.forward_backward_batch = events.forward_backward
        engine.forward_backward_batch.return_value = {"metrics": {}}
        engine.optimizer_step = events.update
        engine.optimizer_step.return_value = 0.5
        engine.is_mp_src_rank_with_outputs = Mock(return_value=True)
        loss_function = Mock()
        result = engine.train_batch({}, loss_function)
        self.assertEqual(
            [event[0] for event in events.mock_calls],
            ["zero", "forward_backward", "update"],
            f"events={events.mock_calls}, expected=zero/forward_backward/update",
        )
        self.assertEqual(result["metrics"]["grad_norm"], 0.5, f"metrics={result['metrics']}, expected grad_norm=0.5")
        events.reset_mock()
        engine.forward_backward_batch.side_effect = RuntimeError("forward failed")
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            engine.train_batch({}, loss_function)
        events.update.assert_not_called()

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
        module_path = "hyper_parallel.integration.verl.engine.trainer"
        trainer_cls = VERL_TRAINER_ADAPTER
        for forward_only in (False, True):
            build_optimizer = Mock()
            build_scheduler = Mock()
            with (
                self.subTest(forward_only=forward_only),
                patch.multiple(
                    module_path, get_local_rank_safe=Mock(return_value=0),
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
