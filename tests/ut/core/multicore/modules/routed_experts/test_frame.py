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
"""CPU regression coverage for prepared compute storage and submission ordering."""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.routed_experts import compute, readiness
from tests.common.mark_utils import arg_mark


class _Event:
    """Capture event recording without an accelerator runtime."""

    def __init__(self) -> None:
        """Create an empty event-recording history."""
        self.records = []

    def record(self, stream: _Stream) -> None:
        """Record the producer stream and operation order."""
        self.records.append(stream)
        stream.operations.append(("record", self))


class _Stream:
    """Capture the event dependencies used by prepared compute frames."""

    def __init__(self, name: str) -> None:
        """Create a named stream with an independent operation history."""
        self.name = name
        self.operations = []

    def wait_event(self, event: _Event) -> None:
        """Reject missing fences and record the consumer's wait."""
        if event is None:
            raise AssertionError(f"Expected an initialized event, got {event!r}")
        self.operations.append(("wait", event))


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestPreparedComputeFrames(unittest.TestCase):
    """Exercise real preparation with CPU kernels replacing only hardware boundaries."""

    def setUp(self) -> None:
        """Provide isolated streams, deterministic CPU operands and native-op mocks."""
        self.current = _Stream("prepare")
        self.generator = torch.Generator(device="cpu").manual_seed(773)
        self.calls = []
        self.weight_tables = []
        self.npu = SimpleNamespace(Event=_Event, current_stream=lambda _device: self.current)
        self.enterContext(patch.object(torch, "npu", self.npu, create=True))
        self.enterContext(patch.object(compute, "_load_native"))
        self.enterContext(patch.object(compute, "upload_metadata", side_effect=self._upload))
        self.enterContext(patch.object(readiness, "upload_metadata", side_effect=self._upload))
        self.enterContext(patch.object(compute, "_linear_config", return_value=torch.empty(200, dtype=torch.uint8)))
        self.enterContext(patch.object(compute, "_gradient_config", return_value=torch.empty(232, dtype=torch.uint8)))
        self.ops = SimpleNamespace(
            routed_experts_weight_table=self._weight_table,
            routed_experts_gemm=self._gemm,
            routed_experts_gradient=self._gradient,
            routed_experts_swiglu=self._activation,
            routed_experts_swiglu_backward=self._activation_backward,
            routed_experts_combine=lambda *_args: self.calls.append("combine"),
            routed_experts_combine_backward=lambda *_args: self.calls.append("combine_backward"),
        )
        self.enterContext(patch.object(torch.ops, "hyper_parallel", self.ops))

    @staticmethod
    def _upload(value, device):
        """Keep metadata on CPU while retaining the real preparation code."""
        return value.to(device=device).clone()

    @staticmethod
    def _engine(boundaries):
        """Bind CPU storage without invoking the NPU-only constructor."""
        engine = compute.NativeExpertCompute.__new__(compute.NativeExpertCompute)
        engine.boundaries = boundaries
        engine.device = torch.device("cpu")
        engine.swiglu_limit = None
        engine.prefix = torch.tensor((0, *boundaries), dtype=torch.int64)
        engine.empty_workspace = torch.empty(0, dtype=torch.uint8)
        engine.initialized = _Event()
        engine.linear_metadata = {}
        engine.gradient_metadata = {}
        engine.forward_pipeline = compute.ForwardPipeline(boundaries, engine.device)
        return engine

    def _random(self, *shape):
        """Avoid changing global RNG state or selecting an accelerator device."""
        return torch.randn(*shape, dtype=torch.bfloat16, device="cpu", generator=self.generator)

    def _weight_table(self, weights):
        """Capture the actual addresses rebound during each preparation."""
        table = torch.tensor([weight.data_ptr() for weight in weights], dtype=torch.int64)
        self.weight_tables.append(table)
        self.current.operations.append(("weight_table", id(table)))
        return table

    def _gemm(self, value, weights, tasks, output, *arguments):
        """Consume the supplied table and write complete expert results on CPU."""
        self.assertEqual(len(arguments), 9)
        schedule_stride = ((len(weights) * 2 * 8 + 63) // 64) * 64
        self.assertGreaterEqual(arguments[1].numel(), arguments[3] * schedule_stride)
        watchdog, table = arguments[-2], arguments[-1]
        self.assertEqual(watchdog, 60_000_000_000)
        self.assertEqual(table.tolist(), [weight.data_ptr() for weight in weights])
        self.calls.append("gemm")
        self.current.operations.append(("gemm", id(output)))
        for expert, weight in enumerate(weights):
            owned = tasks[tasks[:, 0] == expert]
            first = int(owned[:, 1].min())
            end = int((owned[:, 1] + owned[:, 2]).max())
            output[first:end].copy_((value[first:end].float() @ weight.float()).bfloat16())

    def _gradient(self, value, derivative, boundaries, _tasks, output, *_arguments):
        """Read each input before later frame operations overwrite its shared storage."""
        self.calls.append("gradient")
        self.current.operations.append(("gradient", id(output)))
        for expert, (first, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            output[expert].copy_(value[first:end].float().t() @ derivative[first:end].float())

    def _activation(self, preactivation, output):
        """Use CPU arithmetic solely to check the frame's lifetime and ordering."""
        self.calls.append("activation")
        self.current.operations.append(("activation", id(output)))
        gate, up = preactivation.float().chunk(2, dim=-1)
        output.copy_((torch.nn.functional.silu(gate) * up).bfloat16())

    def _activation_backward(self, derivative, preactivation, output):
        """Read da before the subsequent GEMM reuses its buffer for dx."""
        self.calls.append("activation_backward")
        self.current.operations.append(("activation_backward", id(output)))
        output.copy_(self._activation_derivative(derivative, preactivation))

    @staticmethod
    def _activation_derivative(derivative, preactivation):
        """Compute the CPU reference derivative with the same BF16 boundary."""
        gate, up = preactivation.float().chunk(2, dim=-1)
        sigmoid = gate.sigmoid()
        gate_derivative = derivative.float() * up * sigmoid * (1 + gate * (1 - sigmoid))
        up_derivative = derivative.float() * gate * sigmoid
        return torch.cat((gate_derivative, up_derivative), dim=-1).bfloat16()

    def _forbid_preparation(self):
        """Fail if submission allocates Python-visible storage, metadata or events."""
        stack = ExitStack()
        targets = (
            (torch, "empty"), (torch, "empty_like"), (torch, "zeros"), (torch, "zeros_like"),
            (torch.Tensor, "new_empty"), (compute, "upload_metadata"),
            (compute, "_linear_config"), (compute, "_gradient_config"),
            (readiness.ExpertReadiness, "create"), (self.npu, "Event"),
            (self.ops, "routed_experts_weight_table"),
        )
        for target, name in targets:
            stack.enter_context(patch.object(
                target, name, side_effect=AssertionError(f"Submission called preparation API {name}")))
        return stack

    def _expected_backward(self, derivative, state, gate, down, boundaries):
        """Create independent references before executing the aliased backward frame."""
        down_gradients, gate_gradients, input_gradients = [], [], []
        for expert, (first, end) in enumerate(zip((0, *boundaries[:-1]), boundaries)):
            activation_derivative = (derivative[first:end].float() @ down[expert].float().t()).bfloat16()
            preactivation = state.preact[first:end]
            left, right = preactivation.float().chunk(2, dim=-1)
            activation = (torch.nn.functional.silu(left) * right).bfloat16()
            dz = self._activation_derivative(activation_derivative, preactivation)
            down_gradients.append(activation.float().t() @ derivative[first:end].float())
            gate_gradients.append(state.x[first:end].float().t() @ dz.float())
            input_gradients.append((dz.float() @ gate[expert].float().t()).bfloat16())
        return down_gradients, gate_gradients, input_gradients

    def _exercise_frame(self, hidden, intermediate, boundaries=(2, 5)):
        """Check preparation, submission, reuse ordering and one-shot ownership."""
        engine = self._engine(boundaries)
        rows = boundaries[-1] if boundaries else 0
        inputs = self._random(rows, hidden)
        gate = tuple(self._random(hidden, 2 * intermediate) for _ in boundaries)
        down = tuple(self._random(intermediate, hidden) for _ in boundaries)

        forward = engine.prepare_forward(inputs, gate, down, intermediate)

        self.assertIs(forward.output, forward.down.result)
        self.assertIs(forward.ready, forward.down.ready)
        self.assertEqual(self.calls, [])
        with self._forbid_preparation():
            output, state, ready = engine.run_forward(forward)
        self.assertIs(output, forward.output)
        self.assertIs(ready, forward.ready)
        self.assertIs(state.x, inputs)
        self.assertIs(state.preact, forward.z)
        self.assertEqual(self.calls, ["gemm", "activation", "gemm"] if boundaries else ["activation"])
        with self.assertRaisesRegex(ValueError, "already submitted"):
            engine.run_forward(forward)

        self.calls.clear()
        derivative = self._random(*output.shape)
        backward = engine.prepare_backward(derivative, state, gate, down)
        expected_down, expected_gate, expected_input = self._expected_backward(
            derivative, state, gate, down, boundaries)
        saved_z = state.preact.clone()
        self.assertEqual(self.calls, [])
        self.assertIs(backward.output, backward.dx)
        self.assertEqual(backward.activation.untyped_storage().data_ptr(), backward.dz.untyped_storage().data_ptr())
        self.assertEqual(backward.da.untyped_storage().data_ptr(), backward.dx.untyped_storage().data_ptr())
        for value in (backward.activation, backward.da, backward.dx):
            self.assertTrue(value.is_contiguous())
        if boundaries:
            self.assertNotEqual(backward.down_gradient.untyped_storage().data_ptr(),
                                backward.gate_gradient.untyped_storage().data_ptr())

        with self._forbid_preparation():
            dx, ready = engine.run_backward(
                backward, lambda role, _value: self.calls.append(f"publish_{role}"))

        expected_calls = ["activation", "publish_down", "activation_backward", "publish_gate"]
        if boundaries:
            expected_calls = ["gemm", "activation", "gradient", "publish_down", "activation_backward",
                              "gradient", "publish_gate", "gemm"]
        self.assertEqual(self.calls, expected_calls)
        self.assertIs(dx, backward.output)
        self.assertIs(ready, backward.ready)
        torch.testing.assert_close(state.preact, saved_z, rtol=0, atol=0)
        if boundaries:
            torch.testing.assert_close(dx, torch.cat(expected_input), rtol=0, atol=0)
            torch.testing.assert_close(backward.down_gradient, torch.stack(expected_down), rtol=0, atol=0)
            torch.testing.assert_close(backward.gate_gradient, torch.stack(expected_gate), rtol=0, atol=0)
        else:
            self.assertEqual(self.weight_tables, [])
            self.assertIsNone(ready)
            self.assertEqual(backward.down_gradient.shape, (0, intermediate, hidden))
        with self.assertRaisesRegex(ValueError, "already submitted"):
            engine.run_backward(backward, lambda _role, _value: None)

    def test_forward_backward_submit_without_preparation(self):
        """Normal H>I frames preserve values and issue operations in dependency order."""
        self._exercise_frame(hidden=32, intermediate=16)

    def test_backward_project_storage_handles_intermediate_wider_than_hidden(self):
        """The shared da/dx allocation covers both contiguous views when H<I."""
        self._exercise_frame(hidden=16, intermediate=32)

    def test_empty_experts_need_no_weight_tables_or_cube_launches(self):
        """A zero-row rank retains valid outputs without constructing empty cube work."""
        self._exercise_frame(hidden=32, intermediate=16, boundaries=())

    def test_weight_tables_rebind_and_fence_cached_metadata_across_streams(self):
        """Every new weight lease has a fresh address table and upload completion event."""
        for publish in (False, True):
            with self.subTest(publish=publish):
                engine = self._engine((2,))
                inputs = self._random(2, 32)
                first_weights = (self._random(32, 32),)
                second_weights = (self._random(32, 32),)
                preparation_stream = self.current

                first = engine.prepare_linear(inputs, first_weights, 32, publish=publish)
                second = engine.prepare_linear(inputs, second_weights, 32, publish=publish)

                self.assertIs(first.tasks, second.tasks)
                self.assertIs(first.config, second.config)
                self.assertIsNot(first.weight_table, second.weight_table)
                self.assertIsNot(first.initialized, second.initialized)
                for work, weights in ((first, first_weights), (second, second_weights)):
                    self.assertEqual(work.weight_table.tolist(), [weights[0].data_ptr()])
                    self.assertEqual(work.initialized.records, [preparation_stream])
                    operations = preparation_stream.operations
                    self.assertLess(operations.index(("weight_table", id(work.weight_table))),
                                    operations.index(("record", work.initialized)))
                self.current = _Stream("submit")
                dependency = readiness.ExpertReadiness.create(torch.ones((1, 1), dtype=torch.int32), engine.device)

                with self._forbid_preparation():
                    engine.run_linear(first, dependency)
                    engine.run_linear(second, dependency)

                for work in (first, second):
                    operations = self.current.operations
                    launch = operations.index(("gemm", id(work.result)))
                    self.assertLess(operations.index(("wait", work.initialized)), launch)
                    self.assertLess(operations.index(("wait", dependency.admitted)), launch)

    def test_arena_slots_preserve_saved_z_and_separate_gradient_storage(self):
        """Arena allocation covers named scratch only and rejects aliases with saved state."""
        engine = self._engine((2,))
        inputs = self._random(2, 16)
        gate, down = (self._random(16, 64),), (self._random(32, 16),)
        slots, requests = {}, []

        def _allocate(name, shape, dtype):
            requests.append((name, shape, dtype))
            slots[name] = torch.empty(shape, dtype=dtype)
            return slots[name]

        forward = engine.prepare_forward(inputs, gate, down, 32, allocator=_allocate)

        self.assertEqual([request[0] for request in requests], ["activation", "project"])
        self.assertIs(forward.activation, slots["activation"])
        self.assertIs(forward.output, slots["project"])
        self.assertNotIn(forward.z.data_ptr(), [value.data_ptr() for value in slots.values()])
        engine.run_forward(forward)
        requests.clear()
        derivative = self._random(*forward.output.shape)

        backward = engine.prepare_backward(derivative, forward.state, gate, down, allocator=_allocate)

        self.assertEqual([request[0] for request in requests],
                         ["activation", "project", "down_gradient", "gate_gradient"])
        self.assertEqual(requests[1][1], (2, 32))
        self.assertIs(backward.dz, slots["activation"])
        self.assertIs(backward.down_gradient, slots["down_gradient"])
        self.assertIs(backward.gate_gradient, slots["gate_gradient"])
        self.assertNotEqual(backward.down_gradient.data_ptr(), backward.gate_gradient.data_ptr())
        saved_z = forward.z.clone()
        with self._forbid_preparation():
            engine.run_backward(backward, lambda _role, _value: None)
        torch.testing.assert_close(forward.z, saved_z, rtol=0, atol=0)

        fallback = engine.prepare_forward(inputs, gate, down, 32, allocator=lambda *_args: None)
        self.assertEqual(fallback.output.shape, (2, 16))
        with self.assertRaises(ValueError):
            engine.prepare_forward(inputs, gate, down, 32,
                                   allocator=lambda _name, shape, _dtype: torch.empty(shape, dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "alias"):
            engine.prepare_backward(derivative, forward.state, gate, down,
                                    allocator=lambda name, _shape, _dtype: forward.z if name == "activation" else None)

    def test_prepare_rejects_invalid_or_aliased_destinations(self):
        """Invalid destinations fail during preparation before any native payload call."""
        engine = self._engine((2,))
        inputs, weights = self._random(2, 32), (self._random(32, 32),)
        invalid = (
            inputs, torch.empty(2, 32), torch.empty(2, 16, dtype=torch.bfloat16),
            torch.empty(32, 2, dtype=torch.bfloat16).t(),
            torch.empty(2, 32, dtype=torch.bfloat16, device="meta"),
        )
        for output in invalid:
            with self.subTest(shape=output.shape, dtype=output.dtype, device=output.device):
                with self.assertRaises(ValueError):
                    engine.prepare_linear(inputs, weights, 32, output=output)
        with self.assertRaises(ValueError):
            engine.prepare_gradient(inputs, inputs, output=torch.empty(1, 32, 16))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.weight_tables, [])

    def test_combine_uses_preallocated_outputs_and_rejects_aliases(self):
        """Both vector wrappers preserve supplied output identity without allocating."""
        values, scores = self._random(8, 32), self._random(4, 2)
        order = torch.arange(8)
        output = torch.empty(4, 32, dtype=torch.bfloat16)
        value_gradient, score_gradient = torch.empty_like(values), torch.empty_like(scores)

        with self._forbid_preparation():
            combined = compute.combine(values, order, scores, 4, 2, output=output)
            gradients = compute.combine_gradient(
                output, values, order, scores, outputs=(value_gradient, score_gradient))

        self.assertIs(combined, output)
        self.assertIs(gradients[0], value_gradient)
        self.assertIs(gradients[1], score_gradient)
        self.assertEqual(self.calls, ["combine", "combine_backward"])
        invalid = (torch.empty(4, 16, dtype=torch.bfloat16), torch.empty(4, 32), values[:4],
                   torch.empty(32, 4, dtype=torch.bfloat16).t(),
                   torch.empty(4, 32, dtype=torch.bfloat16, device="meta"))
        for candidate in invalid:
            with self.subTest(shape=candidate.shape, dtype=candidate.dtype, device=candidate.device):
                with self.assertRaises(ValueError):
                    compute.combine(values, order, scores, 4, 2, output=candidate)
        with self.assertRaisesRegex(ValueError, "alias"):
            compute.combine_gradient(output, values, order, scores, outputs=(values, score_gradient))
        self.assertEqual(self.calls, ["combine", "combine_backward"])

    def test_poll_event_is_preallocated_and_initialized_for_standalone_readiness(self):
        """Generic and empty readiness can be consumed without a linear producer."""
        for shape in ((2, 3), (0, 0)):
            with self.subTest(shape=shape):
                ready = readiness.ExpertReadiness.create(torch.zeros(shape, dtype=torch.int32), torch.device("cpu"))
                self.assertTrue(ready.poll_after.queued)
                self.assertIsNot(ready.poll_after.event, ready.admitted)
                self.assertEqual(ready.poll_after.event.records, [self.current])
                self.assertEqual(ready.admitted.records, [self.current])

    def test_linear_poll_event_is_rerecorded_before_launch_on_submission_stream(self):
        """The preallocated fence moves past prior work without fencing GEMM completion."""
        engine = self._engine((2,))
        work = engine.prepare_linear(self._random(2, 32), (self._random(32, 32),), 32, publish=True)
        self.assertFalse(work.ready.poll_after.queued)
        prepared_on = self.current
        self.current = _Stream("producer")
        self.current.operations.append(("prior_vector", None))

        with self._forbid_preparation():
            engine.run_linear(work)

        self.assertTrue(work.ready.poll_after.queued)
        self.assertEqual(work.ready.admitted.records, [prepared_on])
        self.assertEqual(work.ready.poll_after.event.records, [prepared_on, self.current])
        operations = self.current.operations
        record = operations.index(("record", work.ready.poll_after.event))
        self.assertLess(operations.index(("wait", work.initialized)), record)
        self.assertLess(operations.index(("prior_vector", None)), record)
        self.assertLess(record, operations.index(("gemm", id(work.result))))
        with self.assertRaisesRegex(ValueError, "already submitted"):
            engine.run_linear(work)
        self.assertEqual(len(work.ready.poll_after.event.records), 2)

    def test_forward_backward_poll_fence_follows_all_gemm_predecessors(self):
        """F fences SwiGLU and B fences SwiGLUGrad and dW before down/Bx begins."""
        engine = self._engine((2,))
        gate, down = (self._random(32, 32),), (self._random(16, 32),)
        forward = engine.prepare_forward(self._random(2, 32), gate, down, 16)
        self.current = _Stream("forward")
        with self._forbid_preparation():
            engine.run_forward(forward)
        operations = self.current.operations
        fence = operations.index(("record", forward.ready.poll_after.event))
        self.assertLess(operations.index(("activation", id(forward.activation))), fence)
        self.assertLess(fence, operations.index(("gemm", id(forward.output))))

        backward = engine.prepare_backward(self._random(2, 32), forward.state, gate, down)
        self.current = _Stream("backward")
        with self._forbid_preparation():
            engine.run_backward(backward, lambda _role, _value: None)
        operations = self.current.operations
        fence = operations.index(("record", backward.ready.poll_after.event))
        self.assertLess(operations.index(("activation_backward", id(backward.dz))), fence)
        self.assertLess(operations.index(("gradient", id(backward.gate_gradient))), fence)
        self.assertLess(fence, operations.index(("gemm", id(backward.output))))

    def test_failed_gemm_submission_leaves_poll_fence_unavailable(self):
        """An event record alone cannot admit consumers of a failed producer."""
        engine = self._engine((2,))
        work = engine.prepare_linear(self._random(2, 32), (self._random(32, 32),), 32, publish=True)
        with patch.object(self.ops, "routed_experts_gemm", side_effect=RuntimeError("submission failed")):
            with self.assertRaisesRegex(RuntimeError, "submission failed"):
                engine.run_linear(work)
        self.assertFalse(work.ready.poll_after.queued)
        self.assertTrue(work.submitted)
        self.assertEqual(len(work.ready.poll_after.event.records), 2)

    def test_linear_consumer_rejects_pending_producer_before_submission(self):
        """Generic GEMM consumers honor the same pre-poll admission contract as T."""
        engine = self._engine((2,))
        producer = engine.prepare_linear(self._random(2, 32), (self._random(32, 32),), 32, publish=True)
        consumer = engine.prepare_linear(producer.result, (self._random(32, 32),), 32)
        self.current = _Stream("consumer")
        with self.assertRaisesRegex(ValueError, "producer to be submitted first"):
            engine.run_linear(consumer, producer.ready)
        self.assertFalse(consumer.submitted)
        self.assertEqual(self.current.operations, [])
        self.assertEqual(self.calls, [])
        engine.run_linear(producer)
        engine.run_linear(consumer, producer.ready)
        self.assertIn(("wait", producer.ready.poll_after.event), self.current.operations)
