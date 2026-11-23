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
"""UT for the DiT trainer runtime, assembled without an accelerator.

Stage methods of :class:`BaseTrainer` are stubbed at class level so
:class:`DiTTrainer.__init__` runs its real assembly against a CPU model, and
the step/loop tests drive the real forward-backward and optimizer updates.
"""
import contextlib
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from hyper_parallel.data.dit import DiTBatch
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.dit_trainer import (
    DiTTrainer,
    _as_float,
    _count_parameters,
    _has_condition_inputs,
    _is_rank0,
    _move_nested_to_device,
    _rank0_info,
    _select_forward_inputs,
    _unwrap_single_sample,
)
from hyper_parallel.trainer.state import TrainerState

_STAGE_NAMES = (
    "_setup", "_build_model", "_build_dataset", "_build_dataloader",
    "_compute_train_iters", "_build_optimizer", "_build_lr_scheduler",
    "_build_training_context", "_init_callbacks",
)


def _make_config(**training_overrides):
    """Build a trainer config tree with DiT-sized training options."""
    training = {
        "seed": 7,
        "global_batch_size": 8,
        "micro_batch_size": 2,
        "max_grad_norm": 1.0,
        "empty_cache_before_backward": False,
    }
    training.update(training_overrides)
    return SimpleNamespace(
        model=SimpleNamespace(
            model_provider=None,
            model_type="wan",
            task="t2v",
            pretrained_model_name_or_path="/models/wan",
            condition_model_name_or_path=None,
        ),
        dataset=SimpleNamespace(data_transform=SimpleNamespace(build=Mock(return_value=Mock()))),
        dataloader=SimpleNamespace(
            collate_fn=SimpleNamespace(build=Mock(return_value=Mock())),
            get_batch=None,
            pp_shared_data=False,
            drop_last=True,
        ),
        training=SimpleNamespace(**training),
    )


def _stub_base_stages(calls):
    """Patch every BaseTrainer stage to record itself and seed CPU runtime attrs."""

    def _setup(self):
        """Seed the device, mesh, and default seed the assembly consumes."""
        calls.append("_setup")
        self.device = torch.device("cpu")
        self.mesh = SimpleNamespace(dp_rank=0, dp_size=1, tp_rank=0, cp_rank=0)
        self.default_seed = 1234

    def _build_model(self):
        """Seed a small CPU model for the assembly to inspect."""
        calls.append("_build_model")
        self.model = torch.nn.Linear(4, 4)

    patchers = [patch.object(BaseTrainer, "_setup", _setup),
                patch.object(BaseTrainer, "_build_model", _build_model)]
    for name in _STAGE_NAMES[2:]:
        def _record(self, name=name):  # pylint: disable=unused-argument
            """Record the stage call order."""
            calls.append(name)
        patchers.append(patch.object(BaseTrainer, name, _record))
    return patchers  # callers start the patchers so each test controls its own cleanup


class _KwArgModel(torch.nn.Module):
    """Tiny model accepting arbitrary keyword inputs and returning a sample."""

    def __init__(self) -> None:
        """Build one linear projection used as the flow-matching head."""
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, **kwargs):  # pylint: disable=arguments-differ
        """Project the hidden states into the prediction tensor."""
        return SimpleNamespace(sample=self.linear(kwargs["hidden_states"].float()))


class TestModuleHelpers(unittest.TestCase):
    """Module-level helpers of the DiT trainer."""

    def test_is_rank0_without_process_group(self):
        """A single-process run is rank 0."""
        self.assertTrue(_is_rank0())

    def test_rank0_info_logs_on_rank0(self):
        """Rank-0 info logging forwards to the module logger."""
        with patch("hyper_parallel.trainer.dit_trainer.logger.info") as log:
            _rank0_info("value=%d", 3)
        log.assert_called_once()

    def test_as_float_accepts_tensor_and_number(self):
        """Tensors unwrap their item; numbers convert directly."""
        self.assertEqual(_as_float(torch.tensor(1.5)), 1.5)
        self.assertEqual(_as_float(2), 2.0)

    def test_move_nested_to_device_walks_containers(self):
        """Tensors move; dicts, lists, tuples recurse; scalars pass."""
        device = torch.device("cpu")
        payload = {
            "a": torch.ones(2),
            "b": [torch.ones(1), (torch.ones(1), "keep")],
            "c": 7,
        }
        moved = _move_nested_to_device(payload, device)
        self.assertEqual(moved["c"], 7)
        self.assertTrue(torch.is_tensor(moved["a"]))
        self.assertIsInstance(moved["b"][1], tuple)

    def test_count_parameters_separates_trainable(self):
        """Trainable and frozen parameters are counted separately."""
        module = torch.nn.Linear(2, 2)
        module.bias.requires_grad = False
        total, trainable = _count_parameters(module)
        self.assertEqual(total, 6)
        self.assertEqual(trainable, 4)

    def test_unwrap_single_sample_variants(self):
        """Non-lists pass through; one sample unwraps; more than one raises."""
        self.assertEqual(_unwrap_single_sample("raw", "f"), "raw")
        self.assertEqual(_unwrap_single_sample([{"a": 1}], "f"), {"a": 1})
        with self.assertRaises(ValueError):
            _unwrap_single_sample([1, 2], "f")

    def test_select_forward_inputs_filters_by_signature(self):  # pylint: disable=no-value-for-parameter
        """Batch keys outside the model signature are dropped."""
        def forward(self, hidden_states, timestep, encoder_hidden_states=None):  # pylint: disable=unused-argument
            """Reference signature for the filter."""
            del timestep, encoder_hidden_states
            return hidden_states

        model = torch.nn.Module()
        setattr(model, "forward", forward)  # setattr keeps astroid from inferring the attribute
        batch: dict = {}
        batch["hidden_states"] = torch.ones(1)
        batch["timestep"] = 1
        batch["training_target"] = torch.ones(1)
        selected = _select_forward_inputs(model, batch)
        self.assertEqual(sorted(selected), ["hidden_states", "timestep"])

    def test_select_forward_inputs_prefers_captured_signature(self):
        """A captured forward signature attribute wins over the bound method."""
        captured = inspect.signature(lambda hidden_states: None)
        model = SimpleNamespace(_hp_forward_signature=captured)
        selected = _select_forward_inputs(model, {"hidden_states": 1, "extra": 2})
        self.assertEqual(sorted(selected), ["hidden_states"])

    def test_select_forward_inputs_var_keyword_accepts_all(self):
        """A variadic-keyword forward accepts every batch field."""
        model = _KwArgModel()
        batch = {"hidden_states": 1, "extra": 2}
        selected = _select_forward_inputs(model, batch)
        self.assertEqual(sorted(selected), ["extra", "hidden_states"])

    def test_has_condition_inputs_checks_required_names(self):
        """The condition hook runs only when the batch carries its inputs."""

        class _Condition(torch.nn.Module):
            """Reference condition model with two required inputs."""

            def get_condition(self, inputs, videos, images=None):  # pylint: disable=unused-argument
                """Reference signature with two required parameters."""
                return {}

        self.assertTrue(_has_condition_inputs(_Condition(), {"inputs": 1, "videos": 2}))
        self.assertFalse(_has_condition_inputs(_Condition(), {"inputs": 1}))


class TestDiTTrainerInit(unittest.TestCase):
    """Assembly of the DiT trainer from stubbed BaseTrainer stages."""

    def setUp(self) -> None:
        """Record stage calls while stubbing every BaseTrainer stage."""
        self.calls = []
        for patcher in _stub_base_stages(self.calls):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_init_runs_all_stages_with_condition_model(self):
        """The assembly calls every stage in order and builds the condition model."""
        with patch("hyper_parallel.trainer.dit_trainer.resolve_dit_provider",
                   return_value=SimpleNamespace(
                       build_condition_model=Mock(return_value=torch.nn.Linear(2, 2)))):
            trainer = DiTTrainer(_make_config())
        self.assertEqual(
            self.calls,
            ["_setup", "_build_model", "_build_dataset", "_build_dataloader",
             "_compute_train_iters", "_build_optimizer", "_build_lr_scheduler",
             "_build_training_context", "_init_callbacks"],
        )
        self.assertIsNotNone(trainer.base.condition_model)
        self.assertEqual(trainer.base.num_micro_batches, 4)
        self.assertIsInstance(trainer.base.get_batch, DiTBatch)

    def test_init_without_condition_model_uses_default_get_batch(self):
        """A provider without condition hooks yields no condition model."""
        with patch("hyper_parallel.trainer.dit_trainer.resolve_dit_provider",
                   return_value=SimpleNamespace()):
            trainer = DiTTrainer(_make_config())
        self.assertIsNone(trainer.base.condition_model)
        self.assertIsInstance(trainer.base.get_batch, DiTBatch)

    def test_missing_dataset_section_raises(self):
        """A config without a dataset section is rejected by the assembly."""
        with patch("hyper_parallel.trainer.dit_trainer.resolve_dit_provider",
                   return_value=SimpleNamespace()):
            trainer = DiTTrainer(_make_config())
        trainer.base.config.dataset = None
        with self.assertRaises(ValueError):
            trainer._build_data_transform()  # pylint: disable=protected-access

    def test_missing_collate_fn_raises(self):
        """A config without a collate function target is rejected."""
        with patch("hyper_parallel.trainer.dit_trainer.resolve_dit_provider",
                   return_value=SimpleNamespace()):
            trainer = DiTTrainer(_make_config())
        trainer.base.config.dataloader = None
        with self.assertRaises(ValueError):
            trainer._build_collate_fn()  # pylint: disable=protected-access


class TestDiTTrainerSteps(unittest.TestCase):
    """Forward-backward and optimizer-step behavior of the DiT trainer."""

    def setUp(self) -> None:
        """Build an assembled trainer wired for CPU micro-batches."""
        for patcher in _stub_base_stages([]):
            patcher.start()
            self.addCleanup(patcher.stop)
        with patch("hyper_parallel.trainer.dit_trainer.resolve_dit_provider",
                   return_value=SimpleNamespace()):
            self.trainer = DiTTrainer(_make_config())
        base = self.trainer.base
        base.model = _KwArgModel()
        base.device = torch.device("cpu")
        base.num_micro_batches = 2
        base.condition_model = None
        base.model_fwd_context = contextlib.nullcontext
        base.model_bwd_context = contextlib.nullcontext()
        base.state = TrainerState(global_step=0, epoch=0)
        base.optimizer = Mock()
        base.lr_scheduler = None
        base.on_step_begin = Mock()
        base.on_step_end = Mock()
        base.model_reshard = Mock()
        base.configure_fsdp_gradient_sync = Mock()

    def _micro_batch(self) -> dict:
        """Build one conditioned micro-batch for the tiny model."""
        return {"hidden_states": torch.ones(1, 4), "training_target": torch.zeros(1, 4)}

    def test_postforward_scales_by_micro_batches(self):
        """The flow-matching MSE divides by the micro-batch count."""
        self.trainer.base.num_micro_batches = 2
        loss, loss_dict = self.trainer.postforward(
            SimpleNamespace(sample=torch.ones(1, 4)), {"training_target": torch.zeros(1, 4)},
        )
        self.assertEqual(_as_float(loss_dict["mse_loss"]), 0.5)
        self.assertEqual(_as_float(loss), 0.5)

    def test_postforward_validates_batch_fields(self):
        """Missing targets and malformed model outputs are rejected."""
        with self.assertRaises(ValueError):
            self.trainer.postforward(SimpleNamespace(sample=torch.ones(1)), {})
        with self.assertRaises(ValueError):
            self.trainer.postforward(SimpleNamespace(), {"training_target": torch.ones(1)})
        with self.assertRaises(ValueError):
            self.trainer.postforward(
                SimpleNamespace(sample=torch.ones(1)), {"training_target": "bad"},
            )

    def test_postforward_unwraps_single_sample_targets(self):
        """Collator-preserved target lists unwrap before the loss."""
        loss, _ = self.trainer.postforward(
            SimpleNamespace(sample=torch.zeros(1, 4)),
            {"training_target": [torch.zeros(1, 4)]},
        )
        self.assertEqual(_as_float(loss), 0.0)

    def test_forward_backward_step_runs_condition_and_backward(self):
        """The condition model rewrites the batch before the real forward."""
        condition_model = MagicMock()
        condition_model.get_condition = Mock(
            side_effect=lambda **batch: {**batch, "conditioned": True},
        )
        condition_model.process_condition = Mock(
            side_effect=lambda **batch: {
                key: value for key, value in batch.items() if key != "conditioned"
            },
        )
        self.trainer.base.condition_model = condition_model
        self.trainer.base.config.training.empty_cache_before_backward = True

        with patch("hyper_parallel.trainer.dit_trainer.empty_cache") as empty_cache:
            loss, loss_dict = self.trainer.forward_backward_step(self._micro_batch())

        condition_model.get_condition.assert_called_once()
        condition_model.process_condition.assert_called_once()
        empty_cache.assert_called_once()
        self.assertTrue(torch.is_tensor(loss))
        self.assertIn("mse_loss", loss_dict)
        self.assertIsNotNone(self.trainer.base.model.linear.weight.grad)

    def test_forward_backward_step_without_condition_model(self):
        """A trainer without a condition model runs the model directly."""
        loss, _ = self.trainer.forward_backward_step(self._micro_batch())
        self.assertTrue(torch.is_tensor(loss))

    def test_train_step_accumulates_and_steps_optimizer(self):
        """Two micro-batches accumulate one optimizer update and step count."""
        self.trainer.base.get_batch = Mock(side_effect=lambda _: self._micro_batch())
        result = self.trainer.train_step(iter(range(2)))
        self.trainer.base.model_reshard.assert_called_with(1, 2)
        self.trainer.base.configure_fsdp_gradient_sync.assert_called_with(1, 2)
        self.trainer.base.optimizer.step.assert_called_once()
        self.trainer.base.optimizer.zero_grad.assert_called_once()
        self.trainer.base.on_step_end.assert_called_once()
        self.assertEqual(self.trainer.base.state.global_step, 1)
        self.assertIsInstance(result["grad_norm"], float)

    def test_train_step_handles_optimizer_list_and_scheduler(self):
        """Optimizer lists and schedulers both step inside one update."""
        first = Mock()
        second = Mock()
        self.trainer.base.optimizer = [first, second]
        self.trainer.base.lr_scheduler = Mock()
        self.trainer.base.get_batch = Mock(side_effect=lambda _: self._micro_batch())
        self.trainer.train_step(iter(range(2)))
        first.step.assert_called_once()
        second.step.assert_called_once()
        self.trainer.base.lr_scheduler.step.assert_called_once()

    def test_train_step_without_scheduler(self):
        """A trainer without a scheduler skips scheduler stepping."""
        self.trainer.base.lr_scheduler = None
        self.trainer.base.get_batch = Mock(side_effect=lambda _: self._micro_batch())
        self.trainer.train_step(iter(range(2)))


class TestDiTTrainerLoop(unittest.TestCase):
    """Epoch iteration and early exits of the training loop."""

    def _loop_trainer(self) -> DiTTrainer:
        """Build a trainer whose loop dependencies are stubbed."""
        trainer = DiTTrainer.__new__(DiTTrainer)
        base = BaseTrainer.__new__(BaseTrainer)
        trainer.base = base
        base.config = SimpleNamespace(
            training=SimpleNamespace(seed=1),
            dataloader=SimpleNamespace(drop_last=True),
        )
        base.local_rank = 0
        base.state = TrainerState(global_step=0, epoch=0)
        base.train_iters = 2
        base.train_epochs = 1
        base.train_steps = 2
        base.on_train_begin = Mock()
        base.on_train_end = Mock()
        base.on_epoch_begin = Mock()
        base.on_epoch_end = Mock()
        base.destroy_distributed = Mock()
        return trainer

    def test_train_loop_breaks_when_target_reached(self):
        """Reaching train_iters ends training after the current epoch."""
        trainer = self._loop_trainer()
        dataloader = MagicMock()
        dataloader.__iter__.return_value = iter([])
        trainer.base.train_dataloader = dataloader
        base = trainer.base

        def _step(data_iterator):  # pylint: disable=unused-argument
            """Advance the global step like a real update would."""
            base.state.global_step += 1

        trainer.train_step = _step

        with patch("hyper_parallel.trainer.dit_trainer.print_device_mem_info") as mem_info:
            trainer.train()

        dataloader.set_epoch.assert_called_once_with(0)
        trainer.base.on_epoch_end.assert_called_once()
        trainer.base.on_train_end.assert_called_once()
        trainer.base.destroy_distributed.assert_called_once()
        mem_info.assert_called_once()

    def test_train_loop_natural_epoch_exit(self):
        """Exhausting train_epochs ends the loop without the target break."""
        trainer = self._loop_trainer()
        trainer.base.train_epochs = 1
        trainer.base.train_iters = 5
        class _NoSetEpochDataloader:
            """Iterable dataloader without a ``set_epoch`` attribute."""

            def __iter__(self):
                """Yield no batches."""
                return iter([])
        trainer.base.train_dataloader = _NoSetEpochDataloader()
        base = trainer.base

        def _step(data_iterator):  # pylint: disable=unused-argument
            """Advance the global step like a real update would."""
            base.state.global_step += 1

        trainer.train_step = _step

        with patch("hyper_parallel.trainer.dit_trainer.print_device_mem_info"):
            trainer.train()

        trainer.base.on_train_end.assert_called_once()
        self.assertEqual(trainer.base.state.epoch, 1)
        self.assertEqual(trainer.base.state.global_step, 2)

    def test_train_loop_handles_stop_iteration(self):
        """A StopIteration from the dataloader ends the epoch quietly."""
        trainer = self._loop_trainer()
        trainer.base.train_dataloader = MagicMock()
        trainer.train_step = Mock(side_effect=StopIteration)

        with patch("hyper_parallel.trainer.dit_trainer.print_device_mem_info"):
            trainer.train()

        trainer.base.on_epoch_end.assert_called_once()
        trainer.base.on_train_end.assert_called_once()

    def test_train_loop_resumes_from_global_step(self):
        """A restored global step positions the epoch's first planned step."""
        trainer = self._loop_trainer()
        trainer.base.state.global_step = 1
        trainer.base.train_dataloader = MagicMock()
        steps = []
        base = trainer.base

        def _step(data_iterator):  # pylint: disable=unused-argument
            """Record each executed step and advance the global step."""
            steps.append(1)
            base.state.global_step += 1

        trainer.train_step = _step

        with patch("hyper_parallel.trainer.dit_trainer.print_device_mem_info"):
            trainer.train()

        self.assertEqual(len(steps), 1)


if __name__ == "__main__":
    unittest.main()
