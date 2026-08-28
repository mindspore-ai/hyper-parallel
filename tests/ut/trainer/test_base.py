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
"""Unit tests for ``hyper_parallel.trainer.base``.

``BaseTrainer.__init__`` is the trainer's only constructor — every entry
script reaches it before doing any distributed work. The full lifecycle
(``_setup`` / ``_build_*``) needs a real PG + a model and is exercised by
ST; UT pins the pieces that **must** stay stable without distributed:

1. ``TrainerState`` initial values and the ``log_history`` mutable list.
2. ``BaseTrainer.__init__`` wires ``spec`` from ``get_spec(args.model.name)``,
   defaults ``max_steps`` from ``args.train.max_steps``, and never touches
   the rest of the lifecycle.
3. ``add_callback`` appends to ``user_callbacks`` in registration order.
4. ``_all_callbacks`` returns built-ins first, then user callbacks.
5. The four ``dispatch_*`` fan-outs (log / save / load / evaluate) call
   every callback's matching hook with the trainer state.
"""
# pylint: disable=protected-access
import os
import types
import unittest
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.trainer import base as trainer_base
from hyper_parallel.trainer.base import BaseTrainer, TrainerState


def _make_args(name: str = "qwen3_5", max_steps: int = 5):
    """Minimal ``args`` SimpleNamespace with the fields ``BaseTrainer.__init__`` reads."""
    return types.SimpleNamespace(
        model=types.SimpleNamespace(name=name),
        train=types.SimpleNamespace(max_steps=max_steps),
    )


class TestTrainerState(unittest.TestCase):
    """``TrainerState`` is the shared mutable state passed to every callback."""

    def test_default_construction(self):
        """All counters zero by default; ``log_history`` is an empty list."""
        state = TrainerState()
        self.assertEqual(state.global_step, 0)
        self.assertEqual(state.epoch, 0)
        self.assertEqual(state.max_steps, 0)
        self.assertEqual(state.log_history, [])

    def test_max_steps_propagated(self):
        """``max_steps`` constructor arg is stored verbatim."""
        state = TrainerState(max_steps=123)
        self.assertEqual(state.max_steps, 123, (
            f"max_steps must propagate, got {state.max_steps}"
        ))

    def test_log_history_is_per_instance(self):
        """``log_history`` must be a fresh list per instance (no class-level mutation)."""
        a = TrainerState()
        b = TrainerState()
        a.log_history.append({"step": 1})
        self.assertEqual(b.log_history, [], (
            f"log_history leaked across instances: b.log_history={b.log_history}"
        ))


class TestBaseTrainerInit(unittest.TestCase):
    """``BaseTrainer.__init__`` wires the early-bound fields only."""

    def test_init_calls_get_spec_and_stores_state(self):
        """``args.model.name`` → ``get_spec`` lookup; ``args.train.max_steps`` → ``state.max_steps``."""
        fake_spec = object()
        with patch.object(trainer_base, "get_spec", return_value=fake_spec) as mock_get:
            trainer = BaseTrainer(_make_args(name="my_model", max_steps=42))

        mock_get.assert_called_once_with("my_model")
        self.assertIs(trainer.spec, fake_spec, (
            f"spec must be the registry return value, got {trainer.spec!r}"
        ))
        self.assertIsInstance(trainer.state, TrainerState)
        self.assertEqual(trainer.state.max_steps, 42, (
            f"state.max_steps must mirror args.train.max_steps, got {trainer.state.max_steps}"
        ))


class TestVisionParallelSampling(unittest.TestCase):
    """Vision parallel config is consulted by the shared dataloader path."""

    def test_share_samples_across_dp_flag(self):
        """The helper returns the normalized ``vision_parallel`` flag."""
        with patch.object(trainer_base, "get_spec", return_value=object()), patch.object(
            trainer_base,
            "get_vision_parallel_config",
            return_value={"share_samples_across_dp": True},
        ):
            trainer = BaseTrainer(_make_args())
            self.assertTrue(trainer._share_samples_across_dp())

    def test_share_samples_across_dp_defaults_false(self):
        """Absent ``vision_parallel`` keeps the old per-DP sampling behavior."""
        with patch.object(trainer_base, "get_spec", return_value=object()), patch.object(
            trainer_base,
            "get_vision_parallel_config",
            return_value={},
        ):
            trainer = BaseTrainer(_make_args())

        self.assertFalse(trainer._share_samples_across_dp())


class _RecorderCallback:
    """Callback double that records every dispatched lifecycle event."""

    def __init__(self):
        self.events = []

    def on_log(self, state, *, metrics, **_):
        self.events.append(("on_log", state, metrics))

    def on_save(self, state, *, checkpoint_dir, **_):
        self.events.append(("on_save", state, checkpoint_dir))

    def on_load(self, state, *, checkpoint_dir, **_):
        self.events.append(("on_load", state, checkpoint_dir))

    def on_evaluate(self, state, *, metrics=None, **_):
        self.events.append(("on_evaluate", state, metrics))


def _make_trainer_with_stubbed_callbacks(num_builtins=2):
    """Construct a BaseTrainer with ``_builtin_callbacks`` replaced by recorders."""
    with patch.object(trainer_base, "get_spec", return_value=object()):
        trainer = BaseTrainer(_make_args())
    builtins = [_RecorderCallback() for _ in range(num_builtins)]
    trainer._builtin_callbacks = lambda: list(builtins)
    trainer.user_callbacks = []
    return trainer, builtins


class TestCallbackRegistrationAndDispatch(unittest.TestCase):
    """``add_callback`` + the four ``dispatch_*`` fan-outs."""

    def test_add_callback_appends_in_order(self):
        """``add_callback`` preserves registration order in ``user_callbacks``."""
        trainer, _ = _make_trainer_with_stubbed_callbacks()
        a, b = MagicMock(), MagicMock()
        trainer.add_callback(a)
        trainer.add_callback(b)
        self.assertEqual(trainer.user_callbacks, [a, b], (
            f"user_callbacks order broken: expected [a, b], got {trainer.user_callbacks!r}"
        ))

    def test_all_callbacks_returns_builtins_then_users(self):
        """``_all_callbacks`` returns builtin callbacks first, user callbacks last."""
        trainer, builtins = _make_trainer_with_stubbed_callbacks(num_builtins=3)
        user_cb = MagicMock()
        trainer.add_callback(user_cb)

        all_cbs = trainer._all_callbacks()
        self.assertEqual(all_cbs[: len(builtins)], builtins, (
            f"builtins must come first, got {all_cbs!r}"
        ))
        self.assertEqual(all_cbs[-1], user_cb, (
            f"user callback must come last, got {all_cbs!r}"
        ))

    def test_dispatch_log_event_fans_out_to_every_callback(self):
        """``dispatch_log_event`` invokes ``on_log`` on every builtin + user callback."""
        trainer, builtins = _make_trainer_with_stubbed_callbacks(num_builtins=2)
        user_cb = _RecorderCallback()
        trainer.add_callback(user_cb)

        metrics = {"step": 7, "loss": 1.23}
        trainer.dispatch_log_event(metrics)

        for cb in builtins + [user_cb]:
            self.assertEqual(len(cb.events), 1, (
                f"Each callback must see one on_log call, got {len(cb.events)}"
            ))
            kind, state, payload = cb.events[0]
            self.assertEqual(kind, "on_log")
            self.assertIs(state, trainer.state, (
                f"Dispatch must pass trainer.state, got {state!r}"
            ))
            self.assertEqual(payload, metrics)

    def test_dispatch_save_load_evaluate_use_matching_hooks(self):
        """Each dispatcher hits the matching hook on every callback exactly once."""
        trainer, builtins = _make_trainer_with_stubbed_callbacks(num_builtins=1)
        recorder = _RecorderCallback()
        trainer.add_callback(recorder)

        trainer.dispatch_save_event("/tmp/ckpt-1")
        trainer.dispatch_load_event("/tmp/ckpt-2")
        trainer.dispatch_evaluate_event({"acc": 0.9})

        for cb in builtins + [recorder]:
            kinds = [evt[0] for evt in cb.events]
            self.assertEqual(kinds, ["on_save", "on_load", "on_evaluate"], (
                f"Wrong dispatch order or coverage: got {kinds}"
            ))
            self.assertEqual(cb.events[0][2], "/tmp/ckpt-1")
            self.assertEqual(cb.events[1][2], "/tmp/ckpt-2")
            self.assertEqual(cb.events[2][2], {"acc": 0.9})


if __name__ == "__main__":
    unittest.main()
