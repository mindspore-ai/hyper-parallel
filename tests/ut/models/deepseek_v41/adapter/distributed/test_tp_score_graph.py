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
"""Native functional Indexer score reductions support eager and graph capture."""

from collections.abc import Callable
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from hyper_parallel.models.deepseek_v41.adapter.distributed import shared_attention_context_parallel as adapter
from tests.common.mark_utils import arg_mark


class TestTPScoreGraph(unittest.TestCase):
    """Regression contracts for the model-owned functional Indexer score reducer."""

    @staticmethod
    def _callback() -> Callable[[torch.Tensor], torch.Tensor]:
        """Build a reducer using a fake two-rank TP group."""
        mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: 0, get_group=lambda: "mock-group")
        return adapter._build_shared_attention_tp_context(mesh).reduce_sum

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_eager_preserves_native_functional_autograd(self) -> None:
        """Feature: Functional Indexer score SUM in ordinary eager execution.

        Description: Mock the native collective with a differentiable reduction.
        Expectation: Values and gradients follow the collective without an identity override.
        """
        callback = self._callback()
        tensor = torch.ones(3, requires_grad=True)
        with (
            patch.object(adapter.funcol, "all_reduce", side_effect=lambda data, *_: data * 2) as reduce,
            patch.object(adapter.funcol, "wait_tensor") as wait,
            patch.object(adapter.torch.distributed, "all_reduce") as imperative,
        ):
            output = callback(tensor)
            output.sum().backward()
        torch.testing.assert_close(output, torch.full((3,), 2.0))
        torch.testing.assert_close(tensor.grad, torch.full((3,), 2.0))
        reduce.assert_called_once_with(tensor, "sum", "mock-group")
        wait.assert_not_called()
        imperative.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_make_fx_preserves_functional_result(self) -> None:
        """Feature: Functional Indexer score reduction during direct FX capture.

        Description: Trace and replay the reducer on inputs different from the capture input.
        Expectation: FX consumes the functional result without an external wait or imperative reduction.
        """
        callback = self._callback()
        with (
            patch.object(adapter.funcol, "all_reduce", side_effect=lambda data, *_: data + 7),
            patch.object(adapter.funcol, "wait_tensor") as wait,
            patch.object(adapter.torch.distributed, "all_reduce") as imperative,
        ):
            graph = make_fx(callback)(torch.ones(3))
        torch.testing.assert_close(graph(torch.full((3,), 4.0)), torch.full((3,), 11.0))
        wait.assert_not_called()
        imperative.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_contiguous_input_preserves_native_result(self) -> None:
        """Feature: Native result ownership for noncontiguous Indexer score inputs.

        Description: Reduce a transposed input through a mocked native collective.
        Expectation: The collective receives contiguous values and its result is returned unchanged.
        """
        callback = self._callback()
        tensor = torch.arange(12.0).reshape(3, 4).transpose(0, 1)
        expected = torch.full_like(tensor, 5.0)
        with (
            patch.object(adapter.funcol, "all_reduce", return_value=expected) as reduce,
            patch.object(adapter.funcol, "wait_tensor") as wait,
        ):
            output = callback(tensor)
        self.assertIs(output, expected)
        self.assertTrue(reduce.call_args.args[0].is_contiguous())
        torch.testing.assert_close(reduce.call_args.args[0], tensor)
        self.assertEqual(reduce.call_args.args[1:], ("sum", "mock-group"))
        wait.assert_not_called()
