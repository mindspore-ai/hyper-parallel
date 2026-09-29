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
"""Unit tests for Trainer callback ownership."""

import gc
import unittest
import weakref

from hyper_parallel.trainer.callbacks.base import Callback


class _TrainerStub:
    """Minimal weak-referenceable Trainer used by callback ownership tests."""

    def __init__(self) -> None:
        """Initialize the mesh attribute consumed by Callback."""
        self.mesh = object()
        self.callback = None


class TestCallbackOwnership(unittest.TestCase):
    """Verify callbacks do not extend their owning Trainer's lifetime."""

    def test_callback_holds_weak_trainer_reference(self) -> None:
        """
        Feature: Callback ownership lifecycle.
        Description: Remove the final strong Trainer reference while cyclic GC is disabled.
        Expectation: The Trainer is released immediately and the callback proxy expires.
        """
        trainer = _TrainerStub()
        callback = Callback(trainer)
        trainer.callback = callback
        trainer_ref = weakref.ref(trainer)
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            del trainer

            self.assertIsNone(trainer_ref())
            with self.assertRaises(ReferenceError):
                _ = callback.trainer.mesh
        finally:
            if gc_was_enabled:
                gc.enable()


if __name__ == "__main__":
    unittest.main()
