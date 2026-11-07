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
"""Stage-owned split backward for reentrant checkpoints."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.activation_memory import checkpoint, checkpoint_wrapper, checkpoint_exclude_wrapper
from hyper_parallel.core.pipeline_parallel._stage import PipelineStageBase


class _CheckpointBlock(torch.nn.Module):
    """Two sequential checkpoints with a shared excluded parameter."""

    def __init__(self, counts: dict[str, int]) -> None:
        """Initialize scalar weights and invocation counters."""
        super().__init__()
        self.first = torch.nn.Parameter(torch.tensor(2.0))
        self.second = torch.nn.Parameter(torch.tensor(3.0))
        self.shared = torch.nn.Parameter(torch.tensor(4.0))
        self.counts = counts
        self.excluded = checkpoint_exclude_wrapper(self._exclude)
        self.first_ckpt = checkpoint_wrapper(self._first, use_reentrant=True)
        self.second_ckpt = checkpoint_wrapper(self._second, use_reentrant=True)

    def _exclude(self, value):
        self.counts["exclude"] += 1
        return value * self.shared

    def _first(self, value):
        self.counts["first"] += 1
        return self.excluded(value * self.first)

    def _second(self, value):
        self.counts["second"] += 1
        return self.excluded(value * self.second)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Chain two reentrant checkpoint nodes."""
        return self.second_ckpt(self.first_ckpt(value))


class _DirectCheckpoint(torch.nn.Module):
    """Exercise a direct checkpoint function call without a registered wrapper."""

    def __init__(self) -> None:
        """Initialize a weight owned outside the checkpoint function."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Checkpoint through the function API."""
        return checkpoint(lambda item: item * self.weight, value, use_reentrant=True)


class _ParameterlessCheckpoint(torch.nn.Module):
    """Checkpoint whose only gradient source is a keyword stage input."""

    def forward(self, *, value: dict[str, torch.Tensor]) -> torch.Tensor:
        """Checkpoint an activation reached through nested keyword arguments."""
        return checkpoint(lambda item: item["hidden"].square(), value, use_reentrant=True)


class _ClosureOnlyCheckpoint(torch.nn.Module):
    """Own no parameters while referring to a trainable external tensor."""

    def __init__(self, external: torch.nn.Parameter) -> None:
        """Keep the external parameter outside the module's parameter registry."""
        super().__init__()
        self.compute = lambda item: item * external

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Use the function API to checkpoint the captured parameter."""
        return checkpoint(self.compute, value, use_reentrant=True)


def _stage(module, micros, requires_grad=True):
    """Build a pipeline stage with synthetic receive buffers."""
    stage = PipelineStageBase(module, stage_index=1, stage_num=3)
    stage.args_recv_info = {}
    stage.grad_recv_info = {}
    inputs = []
    for index in range(micros):
        value = torch.tensor(float(index + 1), requires_grad=requires_grad)
        inputs.append(value)
        stage.args_recv_info[index] = [SimpleNamespace(buffer=value, requires_grad=requires_grad)]
        stage.grad_recv_info[index] = [SimpleNamespace(buffer=torch.tensor(1.0))]
        stage.forward_one_chunk(index)
    return stage, inputs


class TestReentrantStage(unittest.TestCase):
    """Check real stage entry points and their per-microbatch graph lifetime."""

    def test_repeated_stage_split_clears_microbatch_state(self):
        """Serial stage microbatches keep dx ahead of dw without retaining earlier frames."""
        counts = {"first": 0, "second": 0, "exclude": 0}
        module = _CheckpointBlock(counts)
        hits = []
        handles = [param.register_hook(lambda grad, name=name: hits.append(name))
                   for name, param in module.named_parameters()]
        stage = PipelineStageBase(module, stage_index=1, stage_num=3)
        stage.args_recv_info = {}
        stage.grad_recv_info = {}
        try:
            for micro_index in range(4):
                module.zero_grad(set_to_none=True)
                hits.clear()
                value = torch.tensor(float(micro_index + 1), requires_grad=True)
                stage.args_recv_info[micro_index] = [SimpleNamespace(buffer=value, requires_grad=True)]
                stage.grad_recv_info[micro_index] = [SimpleNamespace(buffer=torch.tensor(1.0))]
                stage.forward_one_chunk(micro_index)
                stage.backward_input_one_chunk(micro_index)
                torch.testing.assert_close(value.grad, torch.tensor(96.0))
                self.assertEqual(hits, [])
                self.assertTrue(all(param.grad is None for param in module.parameters()))
                stage.backward_weight_one_chunk(micro_index)
                self.assertEqual(set(hits), {"first", "second", "shared"})
                self.assertEqual(counts, {"first": 2 * (micro_index + 1),
                                          "second": 2 * (micro_index + 1),
                                          "exclude": 2 * (micro_index + 1)})
                torch.testing.assert_close(module.first.grad, torch.tensor(48.0 * (micro_index + 1)))
                torch.testing.assert_close(module.second.grad, torch.tensor(32.0 * (micro_index + 1)))
                torch.testing.assert_close(module.shared.grad, torch.tensor(48.0 * (micro_index + 1)))
                self.assertFalse(stage._reentrant_frames)
                self.assertFalse(stage._dw_cache)
        finally:
            stage.clear_cache()
            for handle in handles:
                handle.remove()

    def test_split_interleaved_sequential_and_hooks(self):
        """dx has no parameter hooks; dw crosses both nodes once per microbatch."""
        counts = {"first": 0, "second": 0, "exclude": 0}
        module = _CheckpointBlock(counts)
        hits = []
        handles = [param.register_hook(lambda grad, name=name: hits.append(name))
                   for name, param in module.named_parameters()]
        stage, inputs = _stage(module, 2)
        self.assertEqual(counts, {"first": 2, "second": 2, "exclude": 4})
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(stage.backward_input_one_chunk, 1).result()
            stage.backward_input_one_chunk(0)
            self.assertEqual(hits, [])
            for param in module.parameters():
                self.assertIsNone(param.grad)
            for index, value in enumerate(inputs):
                torch.testing.assert_close(value.grad, torch.tensor(96.0))
                torch.testing.assert_close(stage.bwd_cache[index][0], value.grad)

            stage.backward_weight_one_chunk(1)
            stage.backward_weight_one_chunk(0)
            self.assertEqual(counts, {"first": 4, "second": 4, "exclude": 4})
            for name, param in module.named_parameters():
                self.assertGreater(hits.count(name), 0)
                expected = {"first": 144.0, "second": 96.0, "shared": 144.0}[name]
                torch.testing.assert_close(param.grad, torch.tensor(expected))
            self.assertFalse(stage._reentrant_frames)
            self.assertFalse(stage._dw_cache)
        finally:
            for handle in handles:
                handle.remove()
            stage.clear_cache()

    def test_full_backward_and_clear_cache(self):
        """Warmup full backward sends dx; incomplete forward state is released."""
        counts = {"first": 0, "second": 0, "exclude": 0}
        module = _CheckpointBlock(counts)
        stage, (value,) = _stage(module, 1)
        stage.backward_one_chunk(0)
        torch.testing.assert_close(value.grad, torch.tensor(96.0))
        torch.testing.assert_close(stage.bwd_cache[0][0], value.grad)
        self.assertFalse(stage._reentrant_frames)

        stage, _ = _stage(_CheckpointBlock({"first": 0, "second": 0, "exclude": 0}), 1)
        frames = stage._reentrant_frames[0]
        stage.clear_cache()
        self.assertTrue(all(frame.released for frame in frames))

    def test_no_grad_stage_input_keeps_parameter_gradients(self):
        """An input without gradient edge still permits delayed parameter gradients."""
        counts = {"first": 0, "second": 0, "exclude": 0}
        module = _CheckpointBlock(counts)
        stage, _ = _stage(module, 1, requires_grad=False)
        stage.backward_input_one_chunk(0)
        self.assertTrue(all(param.grad is None for param in module.parameters()))
        stage.backward_weight_one_chunk(0)
        self.assertTrue(all(param.grad is not None for param in module.parameters()))

    def test_backward_exception_releases_frames_and_buffers(self):
        """A failed replay cannot leave a microbatch graph or receive buffer alive."""
        counts = {"first": 0, "second": 0, "exclude": 0}
        stage, _ = _stage(_CheckpointBlock(counts), 1)
        frames = stage._reentrant_frames[0]
        with patch.object(frames[0], "set_phase", side_effect=RuntimeError("phase failure")):
            with self.assertRaisesRegex(RuntimeError, "phase failure"):
                stage.backward_input_one_chunk(0)
        self.assertFalse(stage._reentrant_frames)
        self.assertFalse(stage._dw_cache)
        self.assertTrue(all(frame.released for frame in frames))
        self.assertIsNone(stage.args_recv_info[0][0].buffer)
        self.assertIsNone(stage.grad_recv_info[0][0].buffer)

    def test_hsdp_reentrant_rejected_before_forward(self):
        """Only the split path rejects unvalidated sharded parameter identity."""
        module = _CheckpointBlock({"first": 0, "second": 0, "exclude": 0})
        with patch("hyper_parallel.core.pipeline_parallel._stage.HSDPModule", _CheckpointBlock):
            stage, _ = _stage(module, 1)
            with self.assertRaisesRegex(ValueError, "Reentrant PP split with FSDP/HSDP is unsupported"):
                stage.backward_input_one_chunk(0)
            self.assertFalse(stage._reentrant_frames)

    def test_direct_checkpoint_and_first_stage_full_backward(self):
        """Direct checkpoint invocations and stage-zero input leaves keep their dx."""
        module = _DirectCheckpoint()
        stage = PipelineStageBase(module, stage_index=0, stage_num=2)
        value = torch.tensor(3.0, requires_grad=True)
        stage.args_recv_info = {}
        stage.grad_recv_info = {0: [SimpleNamespace(buffer=torch.tensor(4.0))]}
        stage.forward_one_chunk(0, args=(value,))
        self.assertEqual(len(stage._reentrant_frames[0]), 1)
        stage.backward_one_chunk(0)
        torch.testing.assert_close(value.grad, torch.tensor(8.0))
        torch.testing.assert_close(module.weight.grad, torch.tensor(12.0))

    def test_direct_checkpoint_split_defers_parameter_gradient(self):
        """A direct checkpoint call selects the public split path even without a wrapper."""
        module = _DirectCheckpoint()
        stage, (value,) = _stage(module, 1)
        stage.backward_input_one_chunk(0)
        torch.testing.assert_close(value.grad, torch.tensor(2.0))
        self.assertIsNone(module.weight.grad)
        stage.backward_weight_one_chunk(0)
        torch.testing.assert_close(module.weight.grad, torch.tensor(1.0))

    def test_dw_trigger_avoids_none_hooks_and_duplicate_real_parameter_edges(self):
        """Transforming parameter hooks see only the real DW gradient once."""
        module = _DirectCheckpoint()
        received = []

        def transform(grad: torch.Tensor) -> torch.Tensor:
            """Record real gradients and scale once to expose duplicate delivery."""
            self.assertIsInstance(grad, torch.Tensor)
            received.append(grad.clone())
            return grad * 2

        hook = module.weight.register_hook(transform)
        try:
            stage, _ = _stage(module, 1)
            stage.backward_input_one_chunk(0)
            self.assertEqual(received, [])
            stage.backward_weight_one_chunk(0)
            self.assertEqual(len(received), 1)
            torch.testing.assert_close(received[0], torch.tensor(1.0))
            torch.testing.assert_close(module.weight.grad, torch.tensor(2.0))
        finally:
            hook.remove()

    def test_dw_trigger_reaches_closure_only_parameter(self):
        """A checkpoint with no stage-owned parameters still performs DW."""
        external = torch.nn.Parameter(torch.tensor(2.0))
        stage, (value,) = _stage(_ClosureOnlyCheckpoint(external), 1)
        self.assertEqual(stage._get_trainable_params(), ())
        stage.backward_input_one_chunk(0)
        torch.testing.assert_close(value.grad, torch.tensor(2.0))
        self.assertIsNone(external.grad)
        stage.backward_weight_one_chunk(0)
        torch.testing.assert_close(external.grad, torch.tensor(1.0))

    def test_first_stage_keyword_without_trainable_params(self):
        """Unrestricted full backward reaches nested kwargs without parameter targets."""
        stage = PipelineStageBase(_ParameterlessCheckpoint(), stage_index=0, stage_num=2)
        value = torch.tensor(3.0, requires_grad=True)
        stage.args_recv_info = {}
        stage.grad_recv_info = {0: [SimpleNamespace(buffer=torch.tensor(4.0))]}
        stage.forward_one_chunk(0, args=(), kwargs={"value": {"hidden": value}})
        stage.backward_one_chunk(0)
        torch.testing.assert_close(value.grad, torch.tensor(24.0))

    def test_plain_stage_scope_does_not_scan_parameters(self):
        """Direct-call detection does not enumerate ordinary stage parameters."""
        module = torch.nn.Linear(1, 1, bias=False)
        stage = PipelineStageBase(module, stage_index=0, stage_num=2)
        with patch.object(module, "parameters", wraps=module.parameters) as parameters:
            for index in range(3):
                stage.forward_one_chunk(index, args=(torch.ones(1, 1),))
        parameters.assert_not_called()

    def test_reentrant_retained_backward_graph_is_rejected(self):
        """A retained outer graph cannot reuse a frame released by its first backward."""
        stage, _ = _stage(_DirectCheckpoint(), 1)
        stage.retain_backward_graph = True
        with self.assertRaisesRegex(ValueError, "does not support retain_backward_graph"):
            stage.backward_one_chunk(0)
        self.assertFalse(stage._reentrant_frames)
