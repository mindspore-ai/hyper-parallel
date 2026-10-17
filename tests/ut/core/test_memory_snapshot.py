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
"""Tests for logical storage accounting and native snapshot compatibility."""

import gc
import pickle
import pickletools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.cuda import _memory_viz

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.memory_snapshot import LogicalMemoryTracker, write_memory_snapshot


class TestLogicalMemoryTracker(unittest.TestCase):
    """Validate allocation lifetimes independently of model structure."""

    def _assert_event_totals(self, snapshot: dict) -> None:
        """Replay native events and compare their totals with the logical summary."""
        for device, trace in enumerate(snapshot["device_traces"]):
            allocated = {}
            current_bytes = 0
            peak_bytes = 0
            for event in trace:
                if event["action"] == "alloc":
                    self.assertNotIn(event["addr"], allocated)
                    allocated[event["addr"]] = event["size"]
                    current_bytes += event["size"]
                    peak_bytes = max(peak_bytes, current_bytes)
                elif event["action"] == "free_completed":
                    self.assertEqual(allocated.pop(event["addr"]), event["size"])
                    current_bytes -= event["size"]
            totals = snapshot["summary"]["devices"].get(device, {"current_bytes": 0, "peak_bytes": 0})
            self.assertEqual(totals, {"current_bytes": current_bytes, "peak_bytes": peak_bytes})
            self.assertEqual(current_bytes, sum(
                segment["allocated_size"] for segment in snapshot["segments"] if segment["device"] == device
            ))

    def test_aliases_share_capacity_and_last_reference_releases_storage(self):
        """Count an entire storage once, even when only a small view survives."""
        with FakeTensorMode(), LogicalMemoryTracker("cpu") as tracker:
            tensor = torch.empty(8)
            alias = tensor[:2]
            self.assertEqual(tracker.snapshot()["summary"]["current_bytes"], 32)
            del tensor
            gc.collect()
            self.assertEqual(tracker.snapshot()["summary"]["current_bytes"], 32)
            del alias
            gc.collect()
            snapshot = tracker.snapshot()

        self.assertEqual(snapshot["summary"]["current_bytes"], 0)
        self.assertEqual(snapshot["summary"]["peak_bytes"], 32)
        self._assert_event_totals(snapshot)
        self.assertEqual(snapshot["segments"], [])
        self.assertEqual([event["action"] for event in snapshot["device_traces"][0]], [
            "segment_alloc", "alloc", "free_requested", "free_completed", "segment_free",
        ])

    def test_resize_to_zero_and_regrow_emit_balanced_lifetimes(self):
        """Track FSDP-style capacity release and reuse, including zero-sized seeds."""
        previous_resize = torch.UntypedStorage.resize_
        with FakeTensorMode(), LogicalMemoryTracker("cpu") as tracker:
            tensor = torch.empty(8)
            storage = tensor.untyped_storage()
            storage.resize_(0)
            self.assertEqual(tracker.snapshot()["summary"]["current_bytes"], 0)
            storage.resize_(64)
            self.assertEqual(tracker.snapshot()["summary"]["current_bytes"], 64)
            storage.resize_(16)
            snapshot = tracker.snapshot()
            empty = torch.empty(0)
            empty.untyped_storage().resize_(12)
            self.assertEqual(tracker.snapshot()["summary"]["current_bytes"], 28)

        self.assertIs(torch.UntypedStorage.resize_, previous_resize)
        self.assertEqual(snapshot["summary"]["current_bytes"], 16)
        self.assertEqual(snapshot["summary"]["peak_bytes"], 64)
        self._assert_event_totals(snapshot)
        trace = snapshot["device_traces"][0]
        self.assertEqual([event["size"] for event in trace if event["action"] == "alloc"], [32, 64, 16])
        self.assertEqual([event["size"] for event in trace if event["action"] == "free_completed"], [32, 64])

    def test_seed_inputs_and_pickle_are_compatible_with_native_visualizer(self):
        """Preserve existing state and frames in an independently loadable snapshot."""
        with FakeTensorMode():
            tensor = torch.empty(4)
            with LogicalMemoryTracker("cpu", device_index=2) as tracker:
                tracker.track([tensor, tensor.view(2, 2)])
                snapshot = tracker.snapshot()
                temporary = tensor.clone()
                del temporary
            self.assertEqual(len(snapshot["device_traces"][2]), 2)

        self.assertEqual(snapshot["summary"]["current_bytes"], 16)
        self.assertEqual(len(snapshot["segments"]), 1)
        self.assertEqual(snapshot["segments"][0]["device"], 2)
        self.assertTrue(snapshot["segments"][0]["frames"])
        with tempfile.TemporaryDirectory() as directory:
            destination = write_memory_snapshot(snapshot, str(Path(directory) / "memory.pickle"))
            with open(destination, "rb") as snapshot_file:
                payload = snapshot_file.read()
                self.assertEqual(payload[:2], b"\x80\x04")
                object_opcodes = {"GLOBAL", "STACK_GLOBAL", "REDUCE", "NEWOBJ", "NEWOBJ_EX", "BUILD"}
                self.assertFalse(object_opcodes & {opcode.name for opcode, _, _ in pickletools.genops(payload)})
                snapshot_file.seek(0)
                restored = pickle.load(snapshot_file)
        self.assertEqual(restored, snapshot)
        self._assert_event_totals(restored)
        self.assertIn("alloc", _memory_viz.trace(restored))
        self.assertIn("<html", _memory_viz.trace_plot(restored).lower())
        self.assertIn("<html", _memory_viz.segment_plot(restored).lower())

    def test_dtensor_counts_local_storage_and_actual_gradient(self):
        """Unwrap a global-shape DTensor without creating synthetic gradients."""
        layout = SimpleNamespace(
            mesh=object(), placements=(Shard(0),), alias_placements=(Shard(0),), tensor_map=(0,),
        )
        with FakeTensorMode():
            local = torch.empty(4, requires_grad=True)
            wrapper = DTensor.from_local_with_layout(local, layout, shape=(1024,))
            detached = wrapper.detach()
            self.assertIs(detached.layout, wrapper.layout)
            self.assertIs(detached.device_mesh, wrapper.device_mesh)
            self.assertEqual(detached.placements, wrapper.placements)
            with LogicalMemoryTracker("cpu") as tracker:
                tracker.track([wrapper, local])
                self.assertEqual(tracker.snapshot()["summary"]["current_bytes"], 16)
                local.square().sum().backward()
                self.assertEqual(tracker.snapshot()["summary"]["current_bytes"], 32)
                local.grad = None
                self.assertEqual(tracker.snapshot()["summary"]["current_bytes"], 16)

    def test_fake_wait_does_not_allocate_an_extra_storage(self):
        """Match real collective wait aliasing rather than its allocating Meta kernel."""
        with FakeTensorMode(), LogicalMemoryTracker("cpu") as tracker:
            tensor = torch.empty(4)
            result = torch.ops._c10d_functional.wait_tensor.default(tensor)
            snapshot = tracker.snapshot()
        self.assertIs(result, tensor)
        self.assertEqual(snapshot["summary"]["peak_bytes"], 16)
        self.assertEqual(len(snapshot["segments"]), 1)

    def test_optimizer_creates_state_without_semantic_hooks(self):
        """Capture lazy Adam state and clear gradients using only storage events."""
        with FakeTensorMode():
            parameter = torch.nn.Parameter(torch.empty(4))
            optimizer = torch.optim.Adam([parameter], foreach=False)
            with LogicalMemoryTracker("cpu") as tracker:
                tracker.track([parameter])
                parameter.square().sum().backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            snapshot = tracker.snapshot()
        self.assertIsNone(parameter.grad)
        state_bytes = sum(tensor.untyped_storage().nbytes() for tensor in optimizer.state[parameter].values())
        self.assertEqual(snapshot["summary"]["current_bytes"], 16 + state_bytes)
        self.assertGreater(snapshot["summary"]["peak_bytes"], snapshot["summary"]["current_bytes"])
        self._assert_event_totals(snapshot)
        for event in snapshot["device_traces"][0]:
            self.assertNotIn("category", event)
            self.assertNotIn("task_name", event)

    def test_exception_restores_resize_hook(self):
        """Restore global storage instrumentation even when execution aborts."""
        previous_resize = torch.UntypedStorage.resize_
        with self.assertRaisesRegex(RuntimeError, "abort"):
            with FakeTensorMode(), LogicalMemoryTracker("cpu"):
                raise RuntimeError("abort")
        self.assertIs(torch.UntypedStorage.resize_, previous_resize)


if __name__ == "__main__":
    unittest.main()
