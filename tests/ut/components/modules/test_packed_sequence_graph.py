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
"""Packing-only graph regressions independent of Compressor and Indexer."""

import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from hyper_parallel.compile.tracer.graph_tracer import trace_model_graph, run_traced_graph
from hyper_parallel.models.deepseek_v41.adapter.data.runtime import DeepseekV41Runtime
from hyper_parallel.models.deepseek_v41.aux_loss import sequence_load_balancing_loss
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.runtime.model_integration import DisabledModelIntegrationSession


def _inputs(boundaries: list[int], *, graph: bool = True) -> dict:
    """Build metadata using the standard adapter without compression configuration binding."""
    ids = torch.arange(boundaries[-1]).unsqueeze(0)
    runtime = DeepseekV41Runtime(include_position_ids=False, include_image_sequence_start=False)
    result = runtime.build(
        batch={"input_ids": ids, "cu_seq_lens": torch.tensor(boundaries, dtype=torch.int32)},
        parallel_context=SimpleNamespace(tp_rank=0, tp_world_size=1, cp_rank=0, cp_world_size=1),
    )
    inputs = {"input_ids": ids, **result}
    return runtime.prepare_graph_inputs(inputs) if graph else inputs


class _PackingModel(nn.Module):
    """Consume only packing metadata, preserving a real parameter-gradient path."""

    def __init__(self) -> None:
        """Initialize a deterministic scalar parameter."""
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(
            self, input_ids: torch.Tensor, packed_seq_params: object, use_cache: bool = False,
    ) -> torch.Tensor:
        """Read prepared geometry without any compression or indexing operations."""
        del use_cache
        packed = packed_seq_params.prepare(input_ids.device, (0,))
        document_ids, _ = packed.router_metadata(input_ids.device)
        values = (input_ids.float() + packed.local_segment_starts(input_ids.device)
                  + packed.local_positions(input_ids.device) + document_ids
                  + packed.segment_start_mask(input_ids.device))
        return (values * self.weight).square().mean()


def _train_fn(model: nn.Module, model_inputs: dict) -> tuple:
    """Use the graph compiler's production training protocol."""
    return model(**model_inputs), {}


class TestPackingOnlyGraph(unittest.TestCase):
    """Exercise changing lengths and documents with no compression/indexing modules."""

    def test_trainer_converts_packing_only_for_graph_execution(self) -> None:
        """The production dispatch invokes model-owned conversion only in graph mode."""
        for graph in (False, True):
            with self.subTest(graph=graph):
                runtime = DeepseekV41Runtime(include_position_ids=False, include_image_sequence_start=False)
                inputs = _inputs([0, 3, 8], graph=False)
                trainer = BaseTrainer.__new__(BaseTrainer)
                trainer.config = SimpleNamespace(training=SimpleNamespace(empty_cache_before_backward=False))
                trainer.device, trainer.state, trainer.mesh = torch.device("cpu"), None, None
                trainer.current_token_counts, trainer.step_token_counts = {}, {}
                trainer.model_fwd_context, trainer.model_bwd_context = nullcontext(), nullcontext()
                trainer.model_integration = DisabledModelIntegrationSession()
                trainer.model = _PackingModel()
                trainer.preforward = lambda batch: batch
                trainer.postforward = lambda output, labels, **kwargs: (output, {})
                trainer.get_batch = SimpleNamespace(runtime_input_adapter=runtime)
                trainer.graph_compiler = Mock() if graph else None
                if graph:
                    trainer.graph_compiler.forward_backward.return_value = (torch.tensor(1.), {})
                with patch.object(runtime, "prepare_graph_inputs", wraps=runtime.prepare_graph_inputs) as prepare:
                    trainer.forward_backward_step(inputs, loss_inputs={})
                if graph:
                    prepare.assert_called_once()
                    converted = trainer.graph_compiler.forward_backward.call_args.kwargs["model_inputs"]
                    self.assertIsNot(converted["packed_seq_params"], inputs["packed_seq_params"])
                else:
                    prepare.assert_not_called()
                torch.testing.assert_close(inputs["packed_seq_params"].cu_seq_lens,
                                           torch.tensor([0, 3, 8], dtype=torch.int32))

    def test_eager_fsdp_hook_preserves_compression_boundaries(self) -> None:
        """Real FSDP autograd input wrapping retains eager compression geometry."""
        for boundaries in ([0, 4, 8], [0, 3, 8]):
            with self.subTest(boundaries=boundaries):
                inputs = _inputs(boundaries, graph=False)
                packed = inputs["packed_seq_params"]
                hidden = torch.ones(1, boundaries[-1], 2, requires_grad=True)
                _, kwargs = HSDPSchedulerV2._register_post_backward_hook(
                    SimpleNamespace(), (hidden,), {"packed_seq_params": packed},
                )
                self.assertIs(kwargs["packed_seq_params"], packed)
                restored = kwargs["packed_seq_params"].prepare(torch.device("cpu"), (2,))
                torch.testing.assert_close(restored.cu_seq_lens, torch.tensor(boundaries, dtype=torch.int32))
                torch.testing.assert_close(restored.minimum_key_indices(torch.device("cpu"), 2),
                                           packed.prepare(torch.device("cpu"), (2,)).minimum_key_indices(
                                               torch.device("cpu"), 2))

    def test_graph_preparation_does_not_mutate_eager_inputs(self) -> None:
        """Preparing graph inputs leaves the source usable by eager compression."""
        source = _inputs([0, 3, 8], graph=False)
        runtime = DeepseekV41Runtime(include_position_ids=False, include_image_sequence_start=False)
        prepared = runtime.prepare_graph_inputs(source)
        self.assertIsNot(prepared["packed_seq_params"], source["packed_seq_params"])
        leaves, spec = tree_flatten(source["packed_seq_params"])
        eager = tree_unflatten(leaves, spec).prepare(torch.device("cpu"), (2,))
        torch.testing.assert_close(eager.cu_seq_lens, torch.tensor([0, 3, 8], dtype=torch.int32))
        leaves, spec = tree_flatten(prepared["packed_seq_params"])
        graph = tree_unflatten(leaves, spec)
        self.assertEqual(graph.cu_seq_lens.numel(), 0)
        with self.assertRaisesRegex(ValueError, "packing-only graph inputs"):
            graph.prepare(torch.device("cpu"), (2,))

    def test_symbolic_joint_graph_replays_lengths_boundaries_and_gradients(self) -> None:
        """Reuse one guarded graph across odd/even lengths and different document counts."""
        model = _PackingModel()
        initial = _inputs([0, 4, 8])
        graph = trace_model_graph(model, _train_fn, {"model_inputs": initial}, dynamic=True)
        for boundaries in ([0, 3, 8], [0, 1, 3, 8], [0, 3, 10], [0, 3, 7], [0, 7]):
            with self.subTest(boundaries=boundaries):
                current = _inputs(boundaries)
                values = []
                for doc, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
                    values.extend(2 * position + begin + doc + int(position == begin)
                                  for position in range(begin, end))
                expected = (torch.tensor(values).float() * model.weight).square().mean()
                gradient = torch.autograd.grad(expected, model.weight)[0]
                loss, gradients, _ = run_traced_graph(graph, model, {"model_inputs": current})
                torch.testing.assert_close(loss, expected.detach())
                torch.testing.assert_close(gradients[0], gradient)

    def test_pytree_schema_does_not_depend_on_compression_or_length(self) -> None:
        """Packing metadata is a Tensor-only schema even without compressed blocks."""
        first, spec = tree_flatten(_inputs([0, 4, 8])["packed_seq_params"])
        second, updated = tree_flatten(_inputs([0, 3, 10])["packed_seq_params"])
        self.assertEqual(spec, updated)
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in first + second))

    def test_router_capacity_preserves_original_document_loss_and_gradients(self) -> None:
        """Empty sequence capacity does not dilute the original per-document mean."""
        packed = _inputs([0, 3, 8])["packed_seq_params"]
        leaves, spec = tree_flatten(packed)
        restored = tree_unflatten(leaves, spec)
        actual_ids, capacity = restored.router_metadata(torch.device("cpu"))
        ids, count = packed.router_metadata(torch.device("cpu"))
        self.assertEqual(count, 2)
        self.assertEqual(capacity, 8)
        self.assertTrue(packed._use_prepared_packing_inputs)
        self.assertFalse(packed._restored_from_packing_tensors)
        self.assertTrue(restored._use_prepared_packing_inputs)
        self.assertTrue(restored._restored_from_packing_tensors)
        torch.testing.assert_close(ids, actual_ids)
        scores = torch.rand(8, 4, requires_grad=True)
        experts = scores.topk(2, -1).indices
        actual = sequence_load_balancing_loss(scores, experts, actual_ids.flatten(), capacity)
        expected = sequence_load_balancing_loss(scores, experts, ids.flatten(), count)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, scores, retain_graph=True)[0],
                                   torch.autograd.grad(expected, scores)[0])

    def test_invalid_boundaries_are_rejected_before_capture(self) -> None:
        """The ordinary runtime still owns validation for every real batch."""
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            _inputs([0, 9, 8])

    def test_boundary_mutation_invalidates_owned_snapshot(self) -> None:
        """Caller mutation creates fresh metadata while existing snapshots stay unchanged."""
        packed = _inputs([0, 4, 8])["packed_seq_params"]
        before = packed.local_segment_starts(torch.device("cpu")).clone()
        packed.cu_seq_lens[1] = 2
        updated = packed.prepare(torch.device("cpu"), (0,))
        torch.testing.assert_close(updated.local_segment_starts(torch.device("cpu")),
                                   torch.tensor([[0, 0, 2, 2, 2, 2, 2, 2]]))
        torch.testing.assert_close(before, torch.tensor([[0, 0, 0, 0, 4, 4, 4, 4]]))


if __name__ == "__main__":
    unittest.main()
