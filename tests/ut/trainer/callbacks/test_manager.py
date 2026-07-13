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
"""Unit tests for CallbackManager."""
# pylint: disable=wrong-import-position
import os
import types
import unittest
from typing import Any, Optional

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.trainer.base import TrainerState
from hyper_parallel.trainer.callbacks.base import (
    BaseCallback,
    CallbackHookNames,
    TrainerCallbackContext,
    TrainerControl,
)
from hyper_parallel.trainer.callbacks.manager import CallbackManager


def _context():
    """Build callback context."""
    return TrainerCallbackContext(state=TrainerState(), control=TrainerControl())


class _Recorder(BaseCallback):
    """Recorder callback."""

    def __init__(self, name: str, sink: list, priority: int = 0, control: Optional[TrainerControl] = None) -> None:
        """Initialise recorder with a name, an event sink list, priority, and optional control return value."""
        self.name = name
        self.sink = sink
        self.priority = priority
        self.control = control

    def on_step_end(self, context: TrainerCallbackContext, **payload: Any) -> Optional[TrainerControl]:
        """Append this recorder's name to the sink and return the configured control."""
        self.sink.append(self.name)
        return self.control


class _NoHook(BaseCallback):
    """Callback that does not override any hook."""


class TestCallbackManager(unittest.TestCase):
    """CallbackManager registration and dispatch behavior."""

    def test_dispatch_uses_priority_then_registration_order(self):
        """Lower priority runs first; same priority preserves registration order."""
        order = []
        manager = CallbackManager(
            trainer=types.SimpleNamespace(),
            callbacks=[
                _Recorder("b", order, priority=10),
                _Recorder("a", order, priority=0),
                _Recorder("c", order, priority=10),
            ],
        )
        manager.dispatch(CallbackHookNames.ON_STEP_END, _context())
        self.assertEqual(order, ["a", "b", "c"])

    def test_has_listeners_checks_overrides(self):
        """Only real overrides count as listeners."""
        manager = CallbackManager(trainer=types.SimpleNamespace(), callbacks=[_NoHook()])
        self.assertFalse(manager.has_listeners(CallbackHookNames.ON_STEP_END))
        manager.register(_Recorder("x", []))
        self.assertTrue(manager.has_listeners(CallbackHookNames.ON_STEP_END))

    def test_dispatch_merges_control(self):
        """Control results from callbacks are merged."""
        manager = CallbackManager(
            trainer=types.SimpleNamespace(),
            callbacks=[
                _Recorder("a", [], control=TrainerControl(should_log=True, metadata={"x": 1})),
                _Recorder("b", [], control=TrainerControl(should_save=True, metadata={"x": 2})),
            ],
        )
        control = manager.dispatch(CallbackHookNames.ON_STEP_END, _context())
        self.assertTrue(control.should_log)
        self.assertTrue(control.should_save)
        self.assertEqual(control.metadata["x"], 2)

    def test_unregister_removes_callback(self):
        """unregister removes a callback from dispatch."""
        order = []
        callback = _Recorder("x", order)
        manager = CallbackManager(trainer=types.SimpleNamespace(), callbacks=[callback])
        manager.unregister(callback)
        self.assertEqual(manager.callbacks, ())
        self.assertFalse(manager.has_listeners(CallbackHookNames.ON_STEP_END))

    def test_reset_step_and_epoch_control(self):
        """Manager exposes control reset helpers."""
        manager = CallbackManager(trainer=types.SimpleNamespace())
        manager.control = TrainerControl(
            should_training_stop=True,
            should_epoch_stop=True,
            should_log=True,
            metadata={"x": 1},
        )
        manager.reset_step_control()
        self.assertTrue(manager.control.should_training_stop)
        self.assertTrue(manager.control.should_epoch_stop)
        self.assertFalse(manager.control.should_log)
        self.assertEqual(manager.control.metadata, {})
        manager.reset_epoch_control()
        self.assertFalse(manager.control.should_epoch_stop)


if __name__ == "__main__":
    unittest.main()
