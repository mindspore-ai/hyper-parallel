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
"""CPU checks for bounded forward storage and the persistent vector admission fence."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from hyper_parallel.core.multicore.modules.routed_experts import forward, readiness
from hyper_parallel.core.multicore.modules.routed_experts.arena import PhaseArena
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestForwardTaskTables(unittest.TestCase):
    """Check the task-table ABI independently of accelerator preparation and dispatch."""

    def test_order_and_tails_keep_original_expert_offsets(self) -> None:
        """An expert starts at its real offset, including row and column-free tail fields."""
        tasks, expected = forward.forward_tasks((3, 133))
        self.assertEqual(tasks.tolist(), [[0, 0, 3], [1, 3, 128], [1, 131, 2]])
        self.assertEqual(expected.tolist(), [[1, 0], [0, 1], [0, 1]])
        self.assertEqual((tasks.dtype, expected.dtype), (torch.int64, torch.int32))
        self.assertEqual((tasks.stride(), expected.stride()), ((3, 1), (2, 1)))

    def test_expected_uses_global_task_ordinal_after_core_wrap(self) -> None:
        """A new expert cannot reset its producer assignment to core zero."""
        tasks, expected = forward.forward_tasks((25 * 128, 25 * 128 + 1))
        self.assertEqual(tasks[-2:].tolist(), [[0, 24 * 128, 128], [1, 25 * 128, 1]])
        self.assertEqual(expected.shape, (24, 2))
        self.assertEqual(expected[:, 0].tolist(), [1] * 24)
        self.assertEqual(expected[:, 1].tolist(), [0, 1] + [0] * 22)

    def test_empty_and_exact_limits_preserve_schema(self) -> None:
        """Empty strides and the maximum legal expert/row extents stay unchanged."""
        tasks, expected = forward.forward_tasks(())
        self.assertEqual((tasks.shape, expected.shape), ((0, 3), (0, 0)))
        self.assertEqual((tasks.stride(), expected.stride()), ((3, 1), (0, 0)))
        self.assertEqual((tasks.dtype, expected.dtype), (torch.int64, torch.int32))
        tasks, _ = forward.forward_tasks(tuple(range(1, 129)))
        self.assertEqual(tasks[-1].tolist(), [127, 127, 1])
        tasks, _ = forward.forward_tasks((2**20,))
        self.assertEqual(tasks.shape, (8192, 3))
        self.assertEqual(tasks[-1].tolist(), [0, 2**20 - 128, 128])

    def test_invalid_types_and_limits_keep_one_host_diagnostic(self) -> None:
        """The expert-count guard short-circuits before malformed entry comparisons."""
        cases = ((True,), (1.5,), (np.int64(1),), (np.bool_(True),), (2**200,),
                 (0,), (2, 2), (2**20 + 1,), tuple(range(1, 130)), (None,) * 129)
        message = "forward boundaries must contain at most 128 positive expert ranges within 2**20 rows"
        for boundaries in cases:
            with self.subTest(boundaries=boundaries), self.assertRaises(ValueError) as error:
                forward.forward_tasks(boundaries)
            self.assertEqual(str(error.exception), message)


class _Stream:
    """Capture stream operations in one shared ordered host history."""

    def __init__(self, name: str, history: list) -> None:
        """Bind a descriptive name to the shared ordered trace."""
        self.name, self.history = name, history

    def wait_event(self, event: _Event) -> None:
        """Require backing resources to be initialized before a queued wait."""
        if not event.records:
            raise AssertionError("wait on an event that has never been recorded")
        self.history.append((self.name, "wait", event))


class _Event:
    """Model event creation separately from first record materialization."""

    def __init__(self, history: list) -> None:
        """Track initial materialization and later recordings separately."""
        self.history, self.records = history, []

    def record(self, stream: _Stream) -> None:
        """Append the stream that materializes or reuses this completion."""
        self.records.append(stream)
        self.history.append((stream.name, "record", self))


class _Npu:
    """Expose CPU-only stream/event primitives without loading an NPU runtime."""

    def __init__(self) -> None:
        """Construct independent C/V histories without device initialization."""
        self.history = []
        self.compute = _Stream("C", self.history)
        self.vector = _Stream("V", self.history)
        self.current = self.compute
        self.events = []

    def Event(self) -> _Event:
        """Create a tracked event with no backing until record is called."""
        event = _Event(self.history)
        self.events.append(event)
        return event

    def Stream(self, **_arguments: Any) -> _Stream:
        """Return the private service stream for standalone compute owners."""
        return self.vector

    def current_stream(self, _device: Any = None) -> _Stream:
        """Report the stream selected by the active context manager."""
        return self.current

    @contextmanager
    def stream(self, stream: _Stream) -> Iterator[None]:
        """Restore C even when one of the two native submissions fails."""
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestForwardPipeline(unittest.TestCase):
    """Check actual row tasks, current pointer ownership, and both stream fences."""

    def setUp(self) -> None:
        """Patch only native boundaries while retaining real tensor validation."""
        self.npu = _Npu()
        self.enterContext(patch.object(torch, "npu", self.npu, create=True))
        self.enterContext(patch.object(forward, "_load_native"))
        self.enterContext(patch.object(forward, "upload_metadata", side_effect=lambda value, _device: value.clone()))
        self.enterContext(patch.object(readiness, "upload_metadata", side_effect=lambda value, _device: value.clone()))
        self.enterContext(patch.object(forward, "_forward_config", return_value=torch.empty(200, dtype=torch.uint8)))
        self.native = SimpleNamespace(
            routed_experts_weight_table=Mock(side_effect=self._table),
            routed_experts_forward_validate=Mock(),
            routed_experts_forward_vector=Mock(side_effect=self._vector),
            routed_experts_forward_cube=Mock(side_effect=self._cube),
        )
        self.enterContext(patch.object(torch.ops, "hyper_parallel", self.native))

    @staticmethod
    def _table(weights: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.tensor([weight.data_ptr() for weight in weights], dtype=torch.int64)

    def _vector(self, *_arguments: Any) -> None:
        self.npu.history.append((self.npu.current.name, "vector", None))

    def _cube(self, *_arguments: Any) -> None:
        self.npu.history.append((self.npu.current.name, "cube", None))

    def _prepare(self, boundaries: tuple[int, ...] = (1, 130), *, transpose: bool = False,
                 allocator: Any = None) -> tuple[forward.ForwardPipeline, forward.PipelineForwardWork]:
        pipeline = forward.ForwardPipeline(boundaries, torch.device("cpu"), self.npu.vector)
        rows = boundaries[-1] if boundaries else 0
        x = torch.empty((rows, 2048), dtype=torch.bfloat16)
        if transpose:
            gate = tuple(torch.empty((1536, 2048), dtype=torch.bfloat16).t() for _ in boundaries)
            down = tuple(torch.empty((2048, 768), dtype=torch.bfloat16).t() for _ in boundaries)
        else:
            gate = tuple(torch.empty((2048, 1536), dtype=torch.bfloat16) for _ in boundaries)
            down = tuple(torch.empty((768, 2048), dtype=torch.bfloat16) for _ in boundaries)
        return pipeline, pipeline.prepare(x, gate, down, 768, allocator)

    def test_tasks_cover_rows_once_and_expected_counts_real_owners(self) -> None:
        """Tails and empty groups never introduce padded or duplicated row writers."""
        for boundaries in ((), (1,), (3, 132, 401), tuple(range(1, 129))):
            tasks, expected = forward.forward_tasks(boundaries)
            coverage = torch.zeros(boundaries[-1] if boundaries else 0, dtype=torch.int32)
            actual = torch.zeros_like(expected)
            for index, (expert, row, count) in enumerate(tasks.tolist()):
                self.assertLessEqual(row + count, boundaries[expert])
                self.assertGreaterEqual(row, (0, *boundaries[:-1])[expert])
                coverage[row:row + count] += 1
                actual[index % expected.shape[0], expert] = 1
            self.assertTrue(torch.equal(coverage, torch.ones_like(coverage)))
            self.assertTrue(torch.equal(expected, actual))
        for boundaries in ((0,), (1, 1), (True,), (2**20 + 1,), tuple(range(1, 130))):
            with self.subTest(boundaries=boundaries), self.assertRaises(ValueError):
                forward.forward_tasks(boundaries)

    def test_prepare_materializes_events_and_run_uses_predecessor_fence(self) -> None:
        """V and reverse polling cannot start before all preceding C producers."""
        pipeline, work = self._prepare()
        self.assertTrue(all(event.records for event in self.npu.events))
        self.native.routed_experts_forward_validate.assert_called_once()
        self.assertFalse(work.ready.poll_after.queued)
        dependency = readiness.ExpertReadiness.create(torch.ones((2, 2), dtype=torch.int32), torch.device("cpu"))
        self.npu.history.clear()
        with patch.object(torch, "empty", side_effect=AssertionError("run allocated tensor")), \
                patch.object(self.npu, "Event", side_effect=AssertionError("run allocated event")), \
                patch.object(self.native, "routed_experts_weight_table",
                             side_effect=AssertionError("run bound weight")):
            result, state, ready = pipeline.run(work, dependency)
        history = self.npu.history
        start = history.index(("C", "record", ready.poll_after.event))
        self.assertLess(history.index(("C", "wait", dependency.poll_after.event)), start)
        self.assertLess(start, history.index(("V", "wait", ready.poll_after.event)))
        self.assertLess(history.index(("V", "wait", ready.poll_after.event)),
                        history.index(("V", "vector", None)))
        self.assertLess(history.index(("V", "vector", None)), history.index(("V", "record", work.vector_complete)))
        self.assertLess(history.index(("V", "record", work.vector_complete)), history.index(("C", "cube", None)))
        self.assertLess(history.index(("C", "cube", None)), history.index(("C", "wait", work.vector_complete)))
        self.assertTrue(ready.poll_after.queued)
        self.assertIs(state.x, work.inputx)
        self.assertIs(state.preact, work.z)
        self.assertIs(result, work.output)
        with self.assertRaisesRegex(ValueError, "already submitted"):
            pipeline.run(work, dependency)

    def test_zero_rank_skips_both_kernels_and_pointer_tables(self) -> None:
        """Empty active sets return valid empty saves without a persistent worker."""
        pipeline, work = self._prepare(())
        output, state, ready = pipeline.run(work)
        self.assertEqual(output.shape, (0, 2048))
        self.assertEqual(state.preact.shape, (0, 1536))
        self.assertIsNone(ready)
        self.native.routed_experts_weight_table.assert_not_called()
        self.native.routed_experts_forward_validate.assert_not_called()
        self.native.routed_experts_forward_cube.assert_not_called()
        self.native.routed_experts_forward_vector.assert_not_called()

    def test_pointer_tables_rebind_every_phase_and_transpose_is_preserved(self) -> None:
        """A replay cannot retain addresses from an earlier FSDP storage generation."""
        pipeline, first = self._prepare((1,), transpose=True)
        self.assertTrue(first.transposed)
        gate = tuple(torch.empty_like(value) for value in first.gate)
        down = tuple(torch.empty_like(value) for value in first.down)
        second = pipeline.prepare(first.inputx, gate, down, 768)
        self.assertEqual(second.gate_table.tolist(), [value.data_ptr() for value in gate])
        self.assertEqual(second.down_table.tolist(), [value.data_ptr() for value in down])
        self.assertNotEqual(first.gate_table.tolist(), second.gate_table.tolist())
        self.assertIs(first.tasks, second.tasks)
        self.assertIsNot(first.mailboxes, second.mailboxes)
        self.assertIsNot(first.ready.flags, second.ready.flags)

    def test_alias_and_oom_fail_before_any_persistent_submission(self) -> None:
        """Admission failures cannot leave V waiting for a producer that never launched."""
        pipeline, work = self._prepare((1,))
        def _alias(name: str, _shape: tuple[int, ...], _dtype: torch.dtype) -> torch.Tensor | None:
            return work.inputx if name == "project" else None
        with self.assertRaisesRegex(ValueError, "alias"):
            pipeline.prepare(work.inputx, work.gate, work.down, 768, _alias)
        with patch.object(torch, "empty", side_effect=RuntimeError("mock OOM")), \
                self.assertRaisesRegex(RuntimeError, "mock OOM"):
            pipeline.prepare(work.inputx, work.gate, work.down, 768)
        self.native.routed_experts_forward_cube.assert_not_called()
        self.native.routed_experts_forward_vector.assert_not_called()

    def test_failed_cube_is_not_marked_ready_or_reusable(self) -> None:
        """A partial launch must propagate to the caller's poison path without fallback."""
        pipeline, work = self._prepare((1,))
        self.native.routed_experts_forward_cube.side_effect = RuntimeError("launch failed")
        with self.assertRaisesRegex(RuntimeError, "launch failed"):
            pipeline.run(work)
        self.assertFalse(work.ready.poll_after.queued)
        self.assertTrue(work.submitted)
        with self.assertRaisesRegex(ValueError, "already submitted"):
            pipeline.run(work)

    def test_arena_uses_distinct_forward_region_and_preserves_saved_z(self) -> None:
        """B activation release cannot remove the separate forward tile capacity."""
        arena = PhaseArena(torch.device("cpu"))
        token = arena.begin()
        pipeline, work = self._prepare((1,), allocator=arena.allocate)
        self.assertEqual(work.activation.shape, (1, 128, 768))
        self.assertNotEqual(work.z.untyped_storage().data_ptr(), work.activation.untyped_storage().data_ptr())
        pipeline.run(work)
        arena.end(token)
        token = arena.begin()
        arena.allocate("activation", (1, 1536), torch.bfloat16)
        arena.end(token)
        before = arena.reserved_bytes
        self.assertEqual(arena.release_regions(("activation",)), 2 * 1024**2)
        self.assertEqual(arena.reserved_bytes, before - 2 * 1024**2)
        self.assertEqual(work.z.shape, (1, 1536))
        arena.close()

    def test_unvalidated_dimensions_and_mixed_layouts_keep_three_op_path(self) -> None:
        """Capabilities do not imply unmeasured precision for arbitrary dimensions."""
        _, work = self._prepare((1,))
        self.assertTrue(forward.ForwardPipeline.supports(2048, 768, work.gate, work.down, 1))
        self.assertFalse(forward.ForwardPipeline.supports(1024, 768, work.gate, work.down, 1))
        mixed = (torch.empty((2048, 768), dtype=torch.bfloat16).t(),)
        self.assertFalse(forward.ForwardPipeline.supports(2048, 768, work.gate, mixed, 1))

    def test_transport_occupancy_gate_covers_topology_boundaries(self) -> None:
        """A persistent consumer cannot consume the service's last vector workers."""
        for ranks, expected in ((16, 8), (20, 8), (21, 0), (24, 0), (25, 8), (40, 8), (41, 0), (48, 0)):
            transport_workers = 2 * ranks if ranks <= 24 else ranks
            with self.subTest(ranks=ranks):
                self.assertEqual(forward.forward_vector_budget(transport_workers), expected)
        self.assertEqual(forward.forward_vector_budget(0), 8)
        for value in (-1, 49, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                forward.forward_vector_budget(value)
        pipeline, work = self._prepare((1,))
        disabled = forward.ForwardPipeline((1,), torch.device("cpu"), vector_workers=0)
        self.assertFalse(disabled.enabled)
        with self.assertRaisesRegex(ValueError, "admitted"):
            disabled.prepare(work.inputx, work.gate, work.down, 768)
        self.assertTrue(pipeline.enabled)
