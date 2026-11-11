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
"""Native traceable collectives without changing eager dispatch."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.utils import communication


class TestDifferentiableCollectives(unittest.TestCase):
    """Preserve collective arguments and gradients through the native APIs."""

    def test_graph_all_reduce_uses_native_autograd(self):
        """Native all-reduce retains reduction names and tensor gradients."""
        for op, expected_op, scale in (("sum", "sum", 2), ("avg", "sum", 2), ("mean", "avg", 1)):
            with self.subTest(op=op):
                inputs = torch.randn(2, 3, requires_grad=True)
                collective = Mock(side_effect=lambda value, *_args, factor=scale: value * factor)
                with patch.object(communication, "_is_tracing_or_compiling", return_value=True), \
                        patch.object(communication, "_functional_collectives", SimpleNamespace(all_reduce=collective)):
                    actual = communication.differentiable_all_reduce(inputs, op, "tp")
                    actual.sum().backward()
                collective.assert_called_once_with(inputs, expected_op, "tp")
                torch.testing.assert_close(actual, inputs * scale)
                torch.testing.assert_close(inputs.grad, torch.full_like(inputs, scale))

    def test_graph_all_gather_keeps_nonzero_dim_and_autograd(self):
        """Negative axes normalize before native gathering without changing shard order."""
        for axis in (0, 2, -1, -2):
            with self.subTest(axis=axis):
                inputs = torch.arange(24.0).reshape(2, 3, 4).requires_grad_()
                collective = Mock(side_effect=lambda value, dim, _group: torch.cat((value, value + 100), dim=dim))
                backend = SimpleNamespace(all_gather_single=collective)
                with patch.object(communication, "_is_tracing_or_compiling", return_value=True), \
                        patch.object(communication, "_functional_collectives", backend):
                    actual = communication.differentiable_all_gather_concat(inputs, "tp", 2, axis)
                    upstream = torch.arange(actual.numel(), dtype=inputs.dtype).reshape(
                        tuple(reversed(actual.shape))
                    ).permute(2, 1, 0)
                    self.assertFalse(upstream.is_contiguous())
                    actual.backward(upstream)
                collective.assert_called_once_with(inputs, axis % inputs.ndim, "tp")
                torch.testing.assert_close(actual, torch.cat((inputs, inputs + 100), dim=axis), rtol=0, atol=0)
                left, right = upstream.chunk(2, dim=axis)
                torch.testing.assert_close(inputs.grad, left + right, rtol=0, atol=0)

        with patch.object(communication, "_is_tracing_or_compiling", return_value=True), \
                patch.object(communication, "_functional_collectives") as backend:
            for axis in (3, -4):
                with self.subTest(invalid_axis=axis), self.assertRaises(IndexError):
                    communication.differentiable_all_gather_concat(inputs, "tp", 2, axis)
        backend.all_gather_single.assert_not_called()

    def test_graph_reduce_scatter_keeps_average_and_autograd(self):
        """Average normalization follows the native SUM collective."""
        for op, scale in (("sum", 2), ("avg", 1)):
            with self.subTest(op=op):
                inputs = torch.arange(8.0).reshape(2, 4).requires_grad_()
                collective = Mock(side_effect=lambda value, _op, axis, _group: value.chunk(2, axis)[0] * 2)
                backend = SimpleNamespace(reduce_scatter_single=collective)
                with patch.object(communication, "_is_tracing_or_compiling", return_value=True), \
                        patch.object(communication, "_functional_collectives", backend):
                    actual = communication.differentiable_reduce_scatter(inputs, 2, 1, op, "tp")
                    actual.sum().backward()
                collective.assert_called_once_with(inputs, "sum", 1, "tp")
                torch.testing.assert_close(actual, inputs[:, :2] * scale)
                torch.testing.assert_close(
                    inputs.grad, torch.tensor([[1., 1., 0., 0.], [1., 1., 0., 0.]]) * scale
                )

    def test_eager_collectives_keep_legacy_dispatch(self):
        """Native functional collectives remain absent from the eager path."""
        inputs = torch.ones(2, 4)
        with patch.object(communication, "_is_tracing_or_compiling", return_value=False), \
                patch.object(communication, "_functional_collectives") as backend, \
                patch.object(communication.dist_func, "all_gather", return_value=(inputs, inputs)) as gather, \
                patch.object(communication.dist_func, "reduce_scatter", return_value=inputs[:, :2] * 2) as scatter, \
                patch.object(communication.dist_func, "all_reduce", return_value=inputs * 2) as reduce:
            actual_gather = communication.differentiable_all_gather_concat(inputs, "tp", 2, 1)
            actual_scatter = communication.differentiable_reduce_scatter(inputs, 2, 1, "avg", "tp")
            actual_reduce = communication.differentiable_all_reduce(inputs, "sum", "tp")
        gather.assert_called_once_with(inputs, group="tp")
        self.assertEqual(scatter.call_count, 1)
        reduce.assert_called_once_with(inputs, op=torch.distributed.ReduceOp.SUM, group="tp")
        self.assertEqual(backend.mock_calls, [])
        torch.testing.assert_close(actual_gather, torch.cat((inputs, inputs), dim=1))
        torch.testing.assert_close(actual_scatter, inputs[:, :2])
        torch.testing.assert_close(actual_reduce, inputs * 2)


class TestCommunication(unittest.TestCase):
    """Keep old/new API selection and HP layout semantics independent."""

    def test_new_and_old_names(self):
        """Prefer new APIs without evaluating a missing legacy attribute."""
        for operation in ("all_gather", "reduce_scatter"):
            for autograd in (True, False):
                suffix = "_autograd" if autograd else ""
                for names, expected in ((["single"], "single"), (["tensor"], "tensor"),
                                        (["single", "tensor"], "single")):
                    with self.subTest(operation=operation, autograd=autograd, names=names):
                        functions = {f"{operation}_{name}{suffix}": Mock() for name in names}
                        with patch.object(communication, "_functional_collectives", SimpleNamespace(**functions)):
                            actual = communication._resolve_functional_collective(operation, autograd=autograd)
                        self.assertIs(actual, functions[f"{operation}_{expected}{suffix}"])
        with patch.object(communication, "_functional_collectives", SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, "functional all_gather_autograd"):
                communication._resolve_functional_collective("all_gather")

    def test_compiled_selection_uses_grad_mode(self):
        """No-grad uses forward APIs even if the input requires grad; training retains autograd."""
        value = torch.ones(4, 4, requires_grad=True)
        for context in (torch.no_grad, torch.inference_mode):
            with self.subTest(context=context.__name__), context(), \
                    patch.object(communication, "_is_tracing_or_compiling", return_value=True), \
                    patch.object(communication.dist, "get_world_size", return_value=2), \
                    patch.object(communication, "_functional_all_gather",
                                 side_effect=AssertionError("autograd selected")), \
                    patch.object(communication, "_functional_reduce_scatter",
                                 side_effect=AssertionError("autograd selected")), \
                    patch.object(communication, "_functional_all_gather_forward", return_value=value) as gather, \
                    patch.object(communication, "_functional_reduce_scatter_forward", return_value=value) as scatter:
                communication.differentiable_all_gather_concat(value, "group", 2, 0)
                communication.differentiable_reduce_scatter(value, 2, 0, "sum", "group")
                gather.assert_called_once_with(value, 0, "group")
                scatter.assert_called_once_with(value, "sum", 0, "group")
        with torch.enable_grad(), \
                patch.object(communication, "_is_tracing_or_compiling", return_value=True), \
                patch.object(communication.dist, "get_world_size", return_value=2), \
                patch.object(communication, "_functional_all_gather", return_value=value) as gather, \
                patch.object(communication, "_functional_all_gather_forward",
                             side_effect=AssertionError("forward selected")):
            detached = value.detach()
            communication.differentiable_all_gather_concat(detached, "group", 2, 0)
            gather.assert_called_once_with(detached, 0, "group")

    def test_compiled_gather_preserves_dimensions_and_gradients(self):
        """Gather on dim zero, then restore rank-ordered concatenation."""
        for axis in (0, 1, 2):
            with self.subTest(axis=axis):
                value = torch.arange(24.0).reshape(2, 3, 4).requires_grad_()
                backend = Mock(side_effect=lambda data, *_args: torch.cat([data, data + 100], dim=0))
                with patch.object(communication, "_is_tracing_or_compiling", return_value=True), \
                        patch.object(communication.dist, "get_world_size", return_value=2), \
                        patch.object(communication, "_functional_all_gather", backend):
                    actual = communication.differentiable_all_gather_concat(value, "group", 2, axis)
                expected = torch.cat([value, value + 100], dim=axis)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual.sum().backward()
                torch.testing.assert_close(value.grad, torch.full_like(value, 2), rtol=0, atol=0)
                backend.assert_called_once_with(value, 0, "group")

    def test_compiled_reduce_scatter_sum_and_average(self):
        """The version adapter retains axis, group, average scaling and gradients."""
        for operation, scale in (("sum", 2), ("avg", 1)):
            with self.subTest(operation=operation):
                value = torch.arange(24.0).reshape(2, 3, 4).requires_grad_()
                backend = Mock(side_effect=lambda data, _op, axis, _group: data.chunk(2, axis)[0] * 2)
                with patch.object(communication, "_is_tracing_or_compiling", return_value=True), \
                        patch.object(communication, "_functional_reduce_scatter", backend):
                    actual = communication.differentiable_reduce_scatter(value, 2, 2, operation, "group")
                torch.testing.assert_close(actual, value[..., :2] * scale, rtol=0, atol=0)
                actual.sum().backward()
                expected_grad = torch.zeros_like(value)
                expected_grad[..., :2] = scale
                torch.testing.assert_close(value.grad, expected_grad, rtol=0, atol=0)
                backend.assert_called_once_with(value, "sum", 2, "group")

    def test_compiled_reduce_scatter_rejects_unsupported_reduction(self):
        """Do not silently substitute SUM for a different differentiable reduction."""
        with patch.object(communication, "_is_tracing_or_compiling", return_value=True):
            with self.assertRaisesRegex(ValueError, "only sum"):
                communication.differentiable_reduce_scatter(torch.ones(4), 2, 0, "max", "group")

    def test_compiled_all_reduce_preserves_string_aliases(self):
        """String operations must retain their meaning without hashing ReduceOp enums."""
        value = torch.arange(8.0)
        cases = {"sum": "sum", "prod": "product", "max": "max", "min": "min",
                 "all": "min", "avg": "sum", "mean": "avg", "unknown": "sum"}
        for operation, expected in cases.items():
            with self.subTest(operation=operation), \
                    patch.object(communication, "_is_tracing_or_compiling", return_value=True), \
                    patch.object(communication, "_reduce_op_to_str", side_effect=AssertionError("enum conversion")), \
                    patch.object(communication._functional_collectives, "all_reduce", return_value=value) as functional:
                actual = communication.differentiable_all_reduce(value, operation, "group")
                functional.assert_called_once_with(value, expected, "group")
                self.assertIs(actual, value)
        with patch.object(communication, "_is_tracing_or_compiling", return_value=True), \
                patch.object(communication.dist, "ReduceOp", SimpleNamespace(SUM=communication.dist.ReduceOp.SUM)), \
                patch.object(communication._functional_collectives, "all_reduce", return_value=value) as functional:
            communication.differentiable_all_reduce(value, "mean", "group")
            functional.assert_called_once_with(value, "sum", "group")

    def test_all_reduce_keeps_eager_and_enum_paths(self):
        """Leave eager autograd and the explicit enum path unchanged."""
        value = torch.arange(8.0)
        for operation in (*communication._OP_MAP, communication.dist.ReduceOp.SUM):
            with self.subTest(operation=operation), \
                    patch.object(communication, "_is_tracing_or_compiling", return_value=False), \
                    patch.object(communication.dist_func, "all_reduce", return_value=value) as eager, \
                    patch.object(communication._functional_collectives, "all_reduce") as functional:
                actual = communication.differentiable_all_reduce(value, operation, "group")
                eager.assert_called_once_with(value, op=communication.resolve_reduce_op(operation), group="group")
                functional.assert_not_called()
                self.assertIs(actual, value)
        with patch.object(communication, "_is_tracing_or_compiling", return_value=True), \
                patch.object(communication._functional_collectives, "all_reduce", return_value=value) as functional:
            communication.differentiable_all_reduce(value, communication.dist.ReduceOp.MAX, "group")
            functional.assert_called_once_with(value, "max", "group")

    def test_eager_keeps_original_collectives(self):
        """Version adaptation must not route eager calls through functional APIs."""
        value = torch.arange(8.0).reshape(2, 4)
        with patch.object(communication, "_is_tracing_or_compiling", return_value=False), \
                patch.object(communication.dist_func, "all_gather", return_value=[value, value + 10]), \
                patch.object(communication, "_functional_all_gather") as functional:
            actual = communication.differentiable_all_gather_concat(value, "group", 2, 1)
        torch.testing.assert_close(actual, torch.cat([value, value + 10], dim=1), rtol=0, atol=0)
        functional.assert_not_called()
        with patch.object(communication, "_is_tracing_or_compiling", return_value=False), \
                patch.object(communication.dist_func, "reduce_scatter", return_value=value[:, :2] * 2), \
                patch.object(communication, "_functional_reduce_scatter") as functional:
            actual = communication.differentiable_reduce_scatter(value, 2, 1, "avg", "group")
        torch.testing.assert_close(actual, value[:, :2], rtol=0, atol=0)
        functional.assert_not_called()
