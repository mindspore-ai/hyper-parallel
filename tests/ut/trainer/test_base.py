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
"""Unit tests for hyper_parallel.trainer.base."""
# pylint: disable=protected-access
import os
import types
import unittest
from typing import Any, Optional
from unittest.mock import patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.trainer import base as trainer_base  # pylint: disable=wrong-import-position
from hyper_parallel.trainer.base import BaseTrainer, TrainerState  # pylint: disable=wrong-import-position
from hyper_parallel.trainer.callbacks.base import BaseCallback, TrainerControl  # pylint: disable=wrong-import-position


def _make_args(name: str = "qwen3_5", max_steps: int = 5, local_rank: int = 0):
    """Minimal args object for BaseTrainer.__init__."""
    return types.SimpleNamespace(
        model=types.SimpleNamespace(name=name),
        train=types.SimpleNamespace(max_steps=max_steps, num_train_epochs=1, local_rank=local_rank),
    )


class TestTrainerState(unittest.TestCase):
    """TrainerState serialization and bounded log history."""

    def test_default_construction(self):
        """All counters zero by default."""
        state = TrainerState()
        self.assertEqual(state.global_step, 0)
        self.assertEqual(state.epoch, 0)
        self.assertEqual(state.micro_step, 0)
        self.assertEqual(state.max_steps, 0)
        self.assertEqual(state.log_history, [])

    def test_log_history_is_per_instance(self):
        """log_history must be a fresh list per instance."""
        first = TrainerState()
        second = TrainerState()
        first.add_log({"step": 1})
        self.assertEqual(second.log_history, [])

    def test_kwargs_override_defaults(self):
        """Keyword initialization overrides default state fields."""
        history = [{"loss": 1.0}]
        state = TrainerState(
            global_step=3,
            epoch=2,
            max_steps=10,
            num_train_epochs=4,
            best_metric=0.9,
            log_history=history,
        )

        self.assertEqual(state.global_step, 3)
        self.assertEqual(state.epoch, 2)
        self.assertEqual(state.max_steps, 10)
        self.assertEqual(state.num_train_epochs, 4)
        self.assertEqual(state.best_metric, 0.9)
        self.assertEqual(state.log_history, history)
        self.assertIsNot(state.log_history, history)

    def test_init_rejects_unknown_field(self):
        """Keyword initialization rejects unknown state fields."""
        with self.assertRaises(ValueError):
            TrainerState(unknown=1)

    def test_update_rejects_unknown_field(self):
        """State update only accepts known fields."""
        state = TrainerState()
        with self.assertRaises(ValueError):
            state.update(unknown=1)

    def test_to_dict_from_dict_round_trip(self):
        """Serialized state can be restored."""
        state = TrainerState(max_steps=10)
        state.update(global_step=3, epoch=2, consumed_tokens=99)
        state.add_log({"loss": 1.0})
        restored = TrainerState.from_dict(state.to_dict())
        self.assertEqual(restored.global_step, 3)
        self.assertEqual(restored.epoch, 2)
        self.assertEqual(restored.consumed_tokens, 99)
        self.assertEqual(restored.log_history, [{"loss": 1.0}])


class TestBaseTrainerInit(unittest.TestCase):
    """BaseTrainer.__init__ wires early-bound fields only."""

    def test_init_calls_get_spec_and_stores_state(self):
        """args.model.name -> get_spec and train.max_steps -> state.max_steps."""
        fake_spec = object()
        with patch.object(trainer_base, "get_spec", return_value=fake_spec) as mock_get:
            trainer = BaseTrainer(_make_args(name="my_model", max_steps=42), setup=False)

        mock_get.assert_called_once_with("my_model")
        self.assertIs(trainer.spec, fake_spec)
        self.assertIsInstance(trainer.state, TrainerState)
        self.assertEqual(trainer.state.max_steps, 42)
        self.assertEqual(trainer.state.num_train_epochs, 1)
        self.assertEqual(trainer.callbacks, ())

    def test_init_registers_callbacks_in_manager(self):
        """callbacks constructor argument is registered on CallbackManager."""
        callback = BaseCallback()
        with patch.object(trainer_base, "get_spec", return_value=object()):
            trainer = BaseTrainer(_make_args(), callbacks=[callback], setup=False)
        self.assertEqual(trainer.callbacks, (callback,))

    def test_init_runs_setup_by_default(self):
        """BaseTrainer owns distributed setup by default."""
        with (
            patch.object(trainer_base, "get_spec", return_value=object()),
            patch.object(BaseTrainer, "_setup", return_value=None) as mock_setup,
        ):
            BaseTrainer(_make_args())

        mock_setup.assert_called_once_with()

    def test_init_can_skip_setup_for_unit_tests(self):
        """setup=False keeps isolated unit tests from booting distributed runtime."""
        with (
            patch.object(trainer_base, "get_spec", return_value=object()),
            patch.object(BaseTrainer, "_setup", return_value=None) as mock_setup,
        ):
            BaseTrainer(_make_args(), setup=False)

        mock_setup.assert_not_called()

    def test_init_records_rank_metadata_defaults(self):
        """Before distributed setup, trainer exposes stable rank metadata defaults."""
        with patch.object(trainer_base, "get_spec", return_value=object()):
            trainer = BaseTrainer(_make_args(local_rank=2), setup=False)

        self.assertEqual(trainer.global_rank, 0)
        self.assertEqual(trainer.local_rank, 2)
        self.assertEqual(trainer.world_size, 1)
        self.assertTrue(trainer.is_world_rank0)
        self.assertFalse(trainer.is_local_rank0)


class _RecorderCallback(BaseCallback):
    """Callback double that records callback context and payload."""

    def __init__(self) -> None:
        """Initialize recorder with an empty events list."""
        self.events = []

    def on_log(self, context: "TrainerCallbackContext", **payload: Any) -> Optional["TrainerControl"]:
        """Record on_log event with state and metrics payload."""
        self.events.append(("on_log", context.state, payload["metrics"]))

    def on_save(self, context: "TrainerCallbackContext", **payload: Any) -> Optional["TrainerControl"]:
        """Record on_save event with state and checkpoint path."""
        self.events.append(("on_save", context.state, payload["checkpoint_path"]))

    def on_resume(self, context: "TrainerCallbackContext", **payload: Any) -> Optional["TrainerControl"]:
        """Record on_resume event with state and checkpoint path."""
        self.events.append(("on_resume", context.state, payload["checkpoint_path"]))

    def on_evaluate_end(self, context: "TrainerCallbackContext", **payload: Any) -> Optional["TrainerControl"]:
        """Record on_evaluate_end event with state and metrics payload."""
        self.events.append(("on_evaluate_end", context.state, payload["metrics"]))


class _StopCallback(BaseCallback):
    """Callback that requests training stop."""

    def on_step_end(self, context: "TrainerCallbackContext", **payload: Any) -> Optional["TrainerControl"]:
        """Return a control object requesting training stop."""
        return TrainerControl(should_training_stop=True)


class _MicroDebugCallback(BaseCallback):
    """Callback that listens to the micro-batch debug hook."""

    def on_micro_batch_end(self, context: "TrainerCallbackContext", **payload: Any) -> Optional["TrainerControl"]:
        """No-op micro-batch hook used to enable the debug hook state flag."""
        del context, payload


def _make_trainer():
    """Build a BaseTrainer with get_spec patched."""
    with patch.object(trainer_base, "get_spec", return_value=object()):
        return BaseTrainer(_make_args(), setup=False)


class TestCallbackRegistrationAndDispatch(unittest.TestCase):
    """BaseTrainer callback manager facade."""

    def test_add_and_remove_callback(self):
        """add_callback/register and remove_callback/unregister keep callback tuple updated."""
        trainer = _make_trainer()
        first = BaseCallback()
        second = BaseCallback()
        trainer.add_callback(first)
        trainer.add_callback(second)
        self.assertEqual(trainer.callbacks, (first, second))
        trainer.remove_callback(first)
        self.assertEqual(trainer.callbacks, (second,))

    def test_dispatch_log_save_resume_evaluate(self):
        """Compatibility dispatch methods route through manager-backed facades."""
        trainer = _make_trainer()
        recorder = _RecorderCallback()
        trainer.add_callback(recorder)

        trainer.dispatch_log_event({"step": 7})
        trainer.dispatch_save_event("/tmp/ckpt-1")
        trainer.dispatch_load_event("/tmp/ckpt-2")
        trainer.dispatch_evaluate_event({"acc": 0.9})

        self.assertEqual(
            [event[0] for event in recorder.events],
            ["on_log", "on_save", "on_resume", "on_evaluate_end"],
        )
        for _, state, _ in recorder.events:
            self.assertIs(state, trainer.state)
        self.assertEqual(recorder.events[0][2], {"step": 7})
        self.assertEqual(recorder.events[1][2], "/tmp/ckpt-1")
        self.assertEqual(recorder.events[2][2], "/tmp/ckpt-2")
        self.assertEqual(recorder.events[3][2], {"acc": 0.9})

    def test_step_end_control_is_returned(self):
        """Callback control returned by manager reaches trainer facade caller."""
        trainer = _make_trainer()
        trainer.add_callback(_StopCallback())
        control = trainer.on_step_end(loss=1.0, grad_norm=0.5)
        self.assertTrue(control.should_training_stop)

    def test_refresh_debug_hook_state_caches_listener_flags(self):
        """High-frequency hook enablement is cached per train step."""
        trainer = _make_trainer()
        trainer._refresh_debug_hook_state()
        self.assertFalse(trainer._debug_hook_state.emit_micro_debug)
        trainer.add_callback(_MicroDebugCallback())
        trainer._refresh_debug_hook_state()
        self.assertTrue(trainer._debug_hook_state.emit_micro_debug)


if __name__ == "__main__":
    unittest.main()
