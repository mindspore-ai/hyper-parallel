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
"""CPU coverage for exclusive scheduler storage and immutable tile ownership."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.routed_experts import compute, readiness
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestLinearSchedule(unittest.TestCase):
    """Exercise real frame preparation without loading native kernels or an NPU."""

    def setUp(self) -> None:
        """Replace only device events, metadata uploads and native submissions."""
        self.operations = []
        self.current = self._stream("prepare")
        self.launches = []
        npu = SimpleNamespace(Event=self._event, current_stream=lambda _device: self.current)
        self.enterContext(patch.object(torch, "npu", npu, create=True))
        self.enterContext(patch.object(compute, "_load_native"))
        self.enterContext(patch.object(compute, "upload_metadata", side_effect=lambda value, _device: value.clone()))
        self.enterContext(patch.object(readiness, "upload_metadata", side_effect=lambda value, _device: value.clone()))
        self.enterContext(patch.object(compute, "_linear_config", return_value=torch.empty(200, dtype=torch.uint8)))
        native = SimpleNamespace(
            routed_experts_weight_table=lambda weights: torch.tensor([value.data_ptr() for value in weights]),
            routed_experts_gemm=self._submit,
        )
        self.enterContext(patch.object(torch.ops, "hyper_parallel", native))

    def _stream(self, name: str) -> SimpleNamespace:
        """Record stream-local waits without synchronizing an accelerator."""
        return SimpleNamespace(name=name, wait_event=lambda event: self.operations.append((name, "wait", id(event))))

    def _event(self) -> SimpleNamespace:
        """Make every event distinguishable in the recorded operation order."""
        event = SimpleNamespace()
        event.record = lambda stream: self.operations.append((stream.name, "record", id(event)))
        return event

    def _engine(self, boundaries: tuple[int, ...]) -> compute.NativeExpertCompute:
        """Create only CPU metadata, preserving all preparation implementation logic."""
        engine = compute.NativeExpertCompute.__new__(compute.NativeExpertCompute)
        engine.boundaries, engine.device = boundaries, torch.device("cpu")
        engine.prefix = torch.tensor((0, *boundaries), dtype=torch.int64)
        engine.empty_workspace = torch.empty(0, dtype=torch.uint8)
        engine.initialized = self._event()
        engine.linear_metadata, engine.gradient_metadata = {}, {}
        return engine

    def _submit(self, *_arguments) -> None:
        """Keep launch references so simulated concurrent frames stay alive."""
        workspace = _arguments[5]
        self.launches.append((self.current.name, workspace))
        self.operations.append((self.current.name, "launch", id(workspace)))

    @staticmethod
    def _inputs(boundaries: tuple[int, ...]) -> tuple:
        """Allocate dense operands whose contents do not affect scheduling tests."""
        rows = boundaries[-1] if boundaries else 0
        return (torch.empty((rows, 16), dtype=torch.bfloat16),
                tuple(torch.empty((16, 16), dtype=torch.bfloat16) for _ in boundaries))

    def test_prepared_frames_have_distinct_workspace_on_different_streams(self) -> None:
        """Shared immutable metadata must not imply shared writable scheduler ranges."""
        engine = self._engine((1, 4, 8))
        inputs, weights = self._inputs(engine.boundaries)
        first = engine.prepare_linear(inputs, weights, 16, publish=True)
        self.current = self._stream("second_prepare")
        second = engine.prepare_linear(inputs, weights, 16, publish=True)

        self.assertIs(first.tasks, second.tasks)
        self.assertIs(first.config, second.config)
        self.assertNotEqual(first.schedule_workspace.data_ptr(), second.schedule_workspace.data_ptr())
        self.assertFalse(hasattr(engine, "schedule_workspace"))
        dependency = readiness.ExpertReadiness.create(torch.ones((2, 3), dtype=torch.int32), engine.device)
        with patch.object(torch, "empty", side_effect=AssertionError("Submission allocated scheduler storage")):
            self.current = self._stream("first_submit")
            engine.run_linear(first, dependency)
            self.current = self._stream("second_submit")
            engine.run_linear(second, dependency)

        self.assertEqual([name for name, _ in self.launches], ["first_submit", "second_submit"])
        self.assertIs(self.launches[0][1], first.schedule_workspace)
        self.assertIs(self.launches[1][1], second.schedule_workspace)
        with self.assertRaisesRegex(ValueError, "already submitted"):
            engine.run_linear(first, dependency)


    def test_ready_expert_scan_preserves_each_cores_original_tiles(self) -> None:
        """A blocked first expert does not change unique ownership or ready publication."""
        tasks = compute.linear_tasks((512, 1152, 1537), 528, 128, 256)
        cores = min(24, len(tasks))
        covered = []
        for core in range(cores):
            original = list(range(core, len(tasks), cores))
            ranges = {}
            for index in original:
                expert = int(tasks[index, 0])
                ranges.setdefault(expert, [index, index])
                ranges[expert][1] = index + cores
            executed, published = [], []
            for ready in ({1, 2}, {0, 1, 2}):
                for expert in sorted(ranges):
                    if expert not in ready or expert in published:
                        continue
                    first, end = ranges[expert]
                    owned = list(range(first, end, cores))
                    self.assertTrue(all(int(tasks[index, 0]) == expert for index in owned))
                    executed.extend(owned)
                    published.append(expert)
                if 0 in ranges and 1 in ranges and ready == {1, 2}:
                    self.assertIn(1, published)
                    self.assertNotIn(0, published)
            self.assertEqual(sorted(executed), original)
            self.assertEqual(len(published), len(ranges))
            covered.extend(executed)
        self.assertEqual(sorted(covered), list(range(len(tasks))))
