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
"""Unit tests for the device-runtime simulation switch and stand-ins."""
import unittest
import weakref

import torch

from hyper_parallel.tools.dryrun.simulation import (
    SimulatedEvent,
    SimulatedStream,
    hold_for_simulated_stream,
    is_device_runtime_simulated,
    release_simulated_stream_holds,
    simulate_device_runtime,
    simulated_accelerator,
    simulated_device_handle,
)


class TestSimulationSwitch(unittest.TestCase):
    """The process-wide switch toggles and reads back."""

    def tearDown(self):
        simulate_device_runtime(False)

    def test_switch_round_trip(self):
        """Enabling and disabling the switch is observable."""
        self.assertFalse(is_device_runtime_simulated())
        simulate_device_runtime(True)
        self.assertTrue(is_device_runtime_simulated())
        simulate_device_runtime(False)
        self.assertFalse(is_device_runtime_simulated())

    def test_accelerator_lie_round_trip(self):
        """The lied accelerator is readable while enabled and cleared after."""
        simulate_device_runtime(True, accelerator="npu")
        self.assertEqual(simulated_accelerator(), "npu")
        simulate_device_runtime(False)
        self.assertIsNone(simulated_accelerator())

    def test_stream_hold_keeps_tensor_until_wait(self):
        """record_stream stand-in retains the staging buffer until a wait.

        A real caching allocator keeps a record_stream'd buffer alive past
        the last Python reference until the stream's pending work completes;
        the conservative stand-in mirrors that by holding the reference until
        the next simulated synchronization point.
        """
        simulate_device_runtime(True)
        try:
            tensor = torch.empty(16)
            ref = weakref.ref(tensor)
            hold_for_simulated_stream(tensor)
            del tensor
            self.assertIsNotNone(ref(), "hold must outlive the last reference")
            SimulatedEvent().wait()
            self.assertIsNone(ref(), "wait must release the hold")
        finally:
            simulate_device_runtime(False)

    def test_disable_clears_pending_holds(self):
        """Turning simulation off never leaks held references across runs."""
        simulate_device_runtime(True)
        tensor = torch.empty(8)
        ref = weakref.ref(tensor)
        hold_for_simulated_stream(tensor)
        del tensor
        simulate_device_runtime(False)
        self.assertIsNone(ref())
        release_simulated_stream_holds()


class TestSimulatedPrimitives(unittest.TestCase):
    """Stand-ins answer the full stream/event protocol without a device."""

    def test_event_protocol(self):
        """Record/wait/query/synchronize complete immediately."""
        event = SimulatedEvent()
        self.assertIs(event.record(), event)
        event.record(SimulatedStream())
        event.wait()
        event.wait(SimulatedStream())
        self.assertTrue(event.query())
        event.synchronize()

    def test_stream_protocol(self):
        """Streams drain immediately and mint elapsed events."""
        stream = SimulatedStream()
        stream.synchronize()
        self.assertTrue(stream.query())
        stream.wait_event(SimulatedEvent())
        stream.wait_stream(SimulatedStream())
        self.assertTrue(stream.record_event().query())

    def test_device_handle_namespace(self):
        """The handle mirrors the torch.<device> surface swap code touches."""
        handle = simulated_device_handle()
        self.assertIsInstance(handle.Stream(), SimulatedStream)
        self.assertIsInstance(handle.Event(), SimulatedEvent)
        self.assertIs(handle.current_stream(), handle.current_stream())
        with handle.stream(handle.Stream()):
            pass
        handle.synchronize()
        handle.synchronize(0)
        with self.assertRaises(AttributeError):
            handle.get_device_name()  # pylint: disable=no-member

    def test_simulated_backend_and_device_runtime_seams(self):
        """Activation-memory backend and trainer device helpers route to simulation."""
        from hyper_parallel.core.activation_memory import _backend as act_backend  # pylint: disable=import-outside-toplevel
        from hyper_parallel.trainer.runtime import device as runtime_device  # pylint: disable=import-outside-toplevel

        simulate_device_runtime(True, accelerator="npu")
        try:
            stream = act_backend.new_stream()
            self.assertIsInstance(stream, SimulatedStream)
            self.assertIsInstance(act_backend.get_current_stream(), SimulatedStream)
            self.assertIsInstance(act_backend.new_event(), SimulatedEvent)
            with act_backend.get_stream_context()(stream):
                act_backend.record_stream(torch.empty(2), stream)
            runtime_device.set_device(0)
            runtime_device.synchronize()
            runtime_device.stream_synchronize()
            runtime_device.empty_cache()
        finally:
            simulate_device_runtime(False)


if __name__ == "__main__":
    unittest.main()
