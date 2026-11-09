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
