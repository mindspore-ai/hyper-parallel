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
"""CPU coverage for capacity-independent cube task compilation."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.routed_experts import compute
from hyper_parallel.core.multicore.modules.routed_experts.compute import gradient_tasks, linear_tasks
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestComputeTasks(unittest.TestCase):
    """Check exact output coverage, tails, empty ranks and unique producer ownership."""

    def test_linear_exact_coverage(self):
        """Real ragged spans create each output element once, without padded rows."""
        boundaries, cols = (7, 136, 397), 272
        tasks = linear_tasks(boundaries, cols, 128, 256)
        coverage = torch.zeros((397, cols), dtype=torch.int32)
        for expert, row, rows, col, width in tasks.tolist():
            self.assertGreaterEqual(row, (0, *boundaries)[expert])
            self.assertLessEqual(row + rows, boundaries[expert])
            coverage[row:row + rows, col:col + width] += 1
        self.assertTrue(torch.equal(coverage, torch.ones_like(coverage)))
        for core in range(24):
            experts = tasks[core::24, 0].tolist()
            self.assertEqual(experts, sorted(experts))

    def test_gradient_unique_writer(self):
        """Every dW tile reduces its full actual token interval with one writer."""
        boundaries = (7, 136, 397)
        tasks = gradient_tasks(boundaries, 144, 272)
        coverage = torch.zeros((3, 144, 272), dtype=torch.int32)
        for expert, row, rows, col, width, begin, tokens in tasks.tolist():
            self.assertEqual(begin, (0, *boundaries)[expert])
            self.assertEqual(begin + tokens, boundaries[expert])
            coverage[expert, row:row + rows, col:col + width] += 1
        self.assertTrue(torch.equal(coverage, torch.ones_like(coverage)))

    def test_empty_and_invalid(self):
        """Empty ranks launch no tasks; invalid metadata fails on the host."""
        self.assertEqual(linear_tasks((), 256).shape, (0, 5))
        self.assertEqual(gradient_tasks((), 128, 256).shape, (0, 7))
        for boundaries in ((0,), (2, 2), (-1,), (True,), (2**20 + 1,)):
            with self.subTest(boundaries=boundaries), self.assertRaises(ValueError):
                linear_tasks(boundaries, 256)
        for width in (0, -1, True, 1.5, 32769):
            with self.subTest(width=width), self.assertRaises(ValueError):
                gradient_tasks((1,), width, 256)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestComputeTaskTables(unittest.TestCase):
    """Reject malformed native configurations during phase preparation."""

    def setUp(self) -> None:
        """Keep CPU preparation from loading an unmocked native payload."""
        self.enterContext(patch.object(compute, "_load_native", side_effect=AssertionError("CPU tasks loaded native")))


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestLinearConfig(unittest.TestCase):
    """Reject incomplete ABI tiling records before a phase can be admitted."""

    def setUp(self) -> None:
        """Keep cached tiling tests independent and avoid loading accelerator libraries."""
        compute._linear_config.cache_clear()
        self.addCleanup(compute._linear_config.cache_clear)
        self.enterContext(patch.object(compute, "_load_native"))

    @staticmethod
    def _native(config: torch.Tensor, config_bytes: int = 200, workspace: int = 0) -> SimpleNamespace:
        """Model the loaded payload's ABI size independently of the returned tensor."""
        return SimpleNamespace(
            routed_experts_gemm_config=Mock(return_value=(config, workspace)),
            routed_experts_gemm_config_bytes=Mock(return_value=config_bytes),
            routed_experts_weight_table=Mock(),
            routed_experts_gemm=Mock(),
        )


    def test_short_config_fails_during_prepare(self) -> None:
        """Preparation fails before weights bind, ready cells publish or GEMM submits."""
        native = self._native(torch.empty(128, dtype=torch.uint8))
        stream, event = Mock(), Mock()
        npu = SimpleNamespace(Event=Mock(return_value=event), current_stream=lambda _device: stream)
        engine = compute.NativeExpertCompute.__new__(compute.NativeExpertCompute)
        engine.boundaries, engine.device = (1,), torch.device("cpu")
        engine.initialized = event
        engine.linear_metadata = {}
        inputs = torch.empty((1, 16), dtype=torch.bfloat16)
        weights = (torch.empty((16, 16), dtype=torch.bfloat16),)
        with patch.object(torch.ops, "hyper_parallel", native), patch.object(torch, "npu", npu, create=True), \
                patch.object(compute, "upload_metadata", side_effect=lambda value, _device: value.clone()), \
                self.assertRaisesRegex(ValueError, "linear configuration"):
            engine.prepare_linear(inputs, weights, 16, publish=True)
        self.assertEqual(engine.linear_metadata, {})
        event.record.assert_not_called()
        native.routed_experts_weight_table.assert_not_called()
        native.routed_experts_gemm.assert_not_called()


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestClippedActivation(unittest.TestCase):
    """Check that clipped activations dispatch to the native forward and backward."""

    def test_forward_and_gradient_use_native_clipped_kernels(self) -> None:
        """Forward and backward pass the same clipping limit and admitted buffers."""
        for limit in (0.5, 1.5):
            with self.subTest(limit=limit):
                preact = torch.empty(32, 64, dtype=torch.bfloat16)
                derivative = torch.empty(32, 32, dtype=torch.bfloat16)
                engine = compute.NativeExpertCompute.__new__(compute.NativeExpertCompute)
                engine.swiglu_limit = limit
                actual = torch.empty_like(derivative)
                gradient = torch.empty_like(preact)
                native = SimpleNamespace(routed_experts_clipped_swiglu=Mock(),
                                         routed_experts_clipped_swiglu_backward=Mock())
                with patch.object(torch.ops, "hyper_parallel", native):
                    engine._activate(preact, actual)
                    engine._activation_gradient(derivative, preact, gradient)
                native.routed_experts_clipped_swiglu.assert_called_once_with(preact, actual, limit)
                native.routed_experts_clipped_swiglu_backward.assert_called_once_with(
                    derivative, preact, gradient, limit)
