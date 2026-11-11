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
"""Dynamic-static unified packing and DSA compression graph regressions."""

import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from hyper_parallel.compile.tracer.graph_tracer import trace_model_graph, run_traced_graph
from hyper_parallel.components.modules import shared_compressed_dsa_attention as attention
from hyper_parallel.models.deepseek_v41.adapter.data.runtime import DeepseekV41Runtime
from hyper_parallel.models.deepseek_v41.adapter.aux_loss import sequence_load_balancing_loss
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.runtime.model_integration import DisabledModelIntegrationSession


def _inputs(
        boundaries: list[int], *, graph: bool = True,
        compress_ratios: tuple[int, ...] = (),
) -> dict:
    """Build metadata using the standard adapter, optionally with compression."""
    ids = torch.arange(boundaries[-1]).unsqueeze(0)
    runtime = DeepseekV41Runtime(
        include_position_ids=False, include_image_sequence_start=False,
        compress_ratios=compress_ratios,
    )
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


class _CompressionPackingModel(nn.Module):
    """Consume compressed-key metadata exactly as attention layers do."""

    def __init__(self, compress_ratio: int) -> None:
        """Keep the consumed ratio and one deterministic parameter."""
        super().__init__()
        self.compress_ratio = compress_ratio
        self.weight = nn.Parameter(torch.tensor(0.25))

    def forward(
            self, input_ids: torch.Tensor, packed_seq_params: object, use_cache: bool = False,
    ) -> torch.Tensor:
        """Read per-token minima and causal bounds shared by eager and graph."""
        del use_cache
        packed = packed_seq_params.prepare(input_ids.device, (self.compress_ratio,))
        minima = packed.minimum_key_indices(input_ids.device, self.compress_ratio)
        upper = packed.causal_key_upper_bounds(input_ids.device, self.compress_ratio)
        values = (minima + upper + input_ids.long()).float()
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
        runtime = DeepseekV41Runtime(
            include_position_ids=False, include_image_sequence_start=False,
        )
        prepared = runtime.prepare_graph_inputs(source)
        self.assertIsNot(prepared["packed_seq_params"], source["packed_seq_params"])
        leaves, spec = tree_flatten(source["packed_seq_params"])
        eager = tree_unflatten(leaves, spec).prepare(torch.device("cpu"), (2,))
        torch.testing.assert_close(eager.cu_seq_lens, torch.tensor([0, 3, 8], dtype=torch.int32))
        leaves, spec = tree_flatten(prepared["packed_seq_params"])
        graph = tree_unflatten(leaves, spec)
        self.assertEqual(graph.cu_seq_lens.numel(), 0)
        with self.assertRaisesRegex(ValueError, "no compressed-key metadata for ratio 2"):
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


class TestCompressionGraphInputs(unittest.TestCase):
    """Graph packing metadata carries compressed-key geometry for DSA layers."""

    def test_graph_metadata_matches_eager_compressed_keys(self) -> None:
        """Restored graph inputs reproduce eager minima and causal bounds exactly."""
        device = torch.device("cpu")
        # Ratio 1 carries no alignment contract, so natural lengths replay too.
        cases = [([0, 3, 8], 1), ([0, 1, 3, 8], 1), ([0, 7], 1), ([0, 3, 10], 1)]
        # Ratio 2 requires the data-side per-document alignment the collate pads.
        cases += [([0, 4, 8], 2), ([0, 2, 4, 8], 2), ([0, 8], 2), ([0, 6, 10], 2)]
        for boundaries, ratio in cases:
            with self.subTest(boundaries=boundaries, ratio=ratio):
                eager = _inputs(boundaries, graph=False)["packed_seq_params"]
                prepared = eager.prepare(device, (ratio,))
                graph = _inputs(boundaries, compress_ratios=(1, ratio))["packed_seq_params"]
                torch.testing.assert_close(
                    graph.minimum_key_indices(device, ratio),
                    prepared.minimum_key_indices(device, ratio),
                )
                torch.testing.assert_close(
                    graph.causal_key_upper_bounds(device, ratio),
                    prepared.causal_key_upper_bounds(device, ratio),
                )

    def test_causal_bounds_reproduce_segment_math_including_misaligned_docs(self) -> None:
        """The shared tensor bound equals the legacy Python segment derivation."""
        device = torch.device("cpu")
        for boundaries in ([0, 3, 8], [0, 1, 4, 9], [0, 2, 7, 8]):
            for ratio in (2, 3):
                with self.subTest(boundaries=boundaries, ratio=ratio):
                    packed = _inputs(boundaries, graph=False)["packed_seq_params"]
                    prepared = packed.prepare(device, (ratio,))
                    query = torch.zeros(1, boundaries[-1], 1, 4)
                    segment_upper, lower = attention._causal_key_bounds(  # pylint: disable=protected-access
                        query, ratio, 0,
                        prepared.minimum_key_indices(device, ratio),
                        prepared.indexer_segments(ratio),
                    )
                    tensor_upper, tensor_lower = attention._causal_key_bounds(  # pylint: disable=protected-access
                        query, ratio, 0,
                        prepared.minimum_key_indices(device, ratio),
                        None,
                        prepared.causal_key_upper_bounds(device, ratio),
                    )
                    self.assertEqual(tensor_upper.ndim, 2)
                    torch.testing.assert_close(tensor_upper.long(), segment_upper.long())
                    torch.testing.assert_close(tensor_lower, lower)

    def test_graph_boundary_rejects_misaligned_documents(self) -> None:
        """Compression capture refuses boundaries that cross a complete block."""
        with self.assertRaisesRegex(ValueError, "multiple of 2"):
            _inputs([0, 3, 8], compress_ratios=(1, 2))
        # Even boundaries pass and restore one alignment fact for in-graph checks.
        leaves, spec = tree_flatten(_inputs([0, 4, 8], compress_ratios=(1, 2))["packed_seq_params"])
        restored = tree_unflatten(leaves, spec)
        self.assertTrue(restored._restored_from_packing_tensors)  # pylint: disable=protected-access
        self.assertTrue(restored.is_compression_aligned(2))
        self.assertTrue(restored.is_compression_aligned(1))

    def test_restored_inputs_keep_graph_only_semantics(self) -> None:
        """Restored inputs answer alignment, bank length and plan questions tensor-side."""
        leaves, spec = tree_flatten(_inputs([0, 4, 8], compress_ratios=(1, 2))["packed_seq_params"])
        restored = tree_unflatten(leaves, spec)
        self.assertEqual(restored.block_plan(2, 1, 0), None)
        self.assertEqual(restored.compressed_key_length(2), 4)
        self.assertEqual(restored.compressed_key_length(1), 8)
        self.assertEqual(restored.compressed_key_length(0), 8)
        self.assertEqual(restored.raw_document_starts(), ())
        with self.assertRaisesRegex(NotImplementedError, "Python document segments"):
            restored.indexer_segments(2)

    def test_compression_schema_is_stable_across_batches(self) -> None:
        """Adding compressed-key metadata keeps one pytree schema for every batch."""
        first, spec = tree_flatten(_inputs([0, 4, 8], compress_ratios=(1, 2))["packed_seq_params"])
        second, updated = tree_flatten(_inputs([0, 2, 4, 8], compress_ratios=(1, 2))["packed_seq_params"])
        self.assertEqual(spec, updated)
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in first + second))

    def test_symbolic_graph_replays_compressed_keys_across_batches(self) -> None:
        """One guarded graph serves every document count and length with DSA metadata."""
        model = _CompressionPackingModel(compress_ratio=2)
        initial = _inputs([0, 4, 8], compress_ratios=(1, 2))
        graph = trace_model_graph(model, _train_fn, {"model_inputs": initial}, dynamic=True)
        for boundaries in ([0, 2, 4, 8], [0, 4, 8], [0, 8], [0, 6, 10]):
            with self.subTest(boundaries=boundaries):
                current = _inputs(boundaries, compress_ratios=(1, 2))
                eager_packed = _inputs(boundaries, graph=False, compress_ratios=(1, 2))["packed_seq_params"]
                device = torch.device("cpu")
                prepared = eager_packed.prepare(device, (2,))
                ids = current["input_ids"]
                values = (prepared.minimum_key_indices(device, 2)
                          + prepared.causal_key_upper_bounds(device, 2) + ids.long()).float()
                expected = (values * model.weight).square().mean()
                gradient = torch.autograd.grad(expected, model.weight)[0]
                loss, gradients, _ = run_traced_graph(graph, model, {"model_inputs": current})
                torch.testing.assert_close(loss, expected.detach())
                torch.testing.assert_close(gradients[0], gradient)

    def test_topk_multichunk_slices_bounds_along_queries(self) -> None:
        """Chunked reference selection stays query-aligned when S exceeds the chunk."""
        device = torch.device("cpu")
        torch.manual_seed(7)
        # Reviewer regression shape: two 256-token documents, default chunk 256,
        # plus a ragged chunk tail; the packed tensor bound must match segments.
        boundaries = [0, 256, 512]
        packed = _inputs(boundaries, graph=False)["packed_seq_params"]
        prepared = packed.prepare(device, (2,))
        query = torch.randn(1, 512, 2, 8)
        key = torch.randn(1, 256, 8)
        weights = torch.rand(1, 512, 2)
        shared = {
            "compress_ratio": 2, "sparse_count": 3,
            "minimum_key_indices": prepared.minimum_key_indices(device, 2),
        }
        for chunk in (256, 100):
            with self.subTest(chunk=chunk):
                from_segments = attention.compressed_causal_topk(
                    query, key, weights, query_segments=prepared.indexer_segments(2),
                    query_chunk_size=chunk, **shared,
                )
                from_tensors = attention.compressed_causal_topk(
                    query, key, weights,
                    causal_upper=prepared.causal_key_upper_bounds(device, 2),
                    query_chunk_size=chunk, **shared,
                )
                self.assertEqual(tuple(from_tensors.shape), (1, 512, 3))
                torch.testing.assert_close(from_tensors, from_segments)
        # Multi-document, multi-chunk tail on a tiny sequence.
        small = _inputs([0, 4, 8], graph=False)["packed_seq_params"].prepare(device, (2,))
        small_shared = {
            "compress_ratio": 2, "sparse_count": 2,
            "minimum_key_indices": small.minimum_key_indices(device, 2),
        }
        small_query = torch.randn(1, 8, 1, 4)
        small_key = torch.ones(1, 4, 4)
        small_weights = torch.ones(1, 8, 1)
        from_segments = attention.compressed_causal_topk(
            small_query, small_key, small_weights, query_segments=small.indexer_segments(2),
            query_chunk_size=3, **small_shared,
        )
        from_tensors = attention.compressed_causal_topk(
            small_query, small_key, small_weights,
            causal_upper=small.causal_key_upper_bounds(device, 2),
            query_chunk_size=3, **small_shared,
        )
        torch.testing.assert_close(from_tensors, from_segments)

    def test_topk_selection_matches_between_tensor_and_segment_bounds(self) -> None:
        """Reference selection picks the same keys from either bound source."""
        boundaries = [0, 3, 8]
        ratio = 2
        device = torch.device("cpu")
        packed = _inputs(boundaries, graph=False)["packed_seq_params"]
        prepared = packed.prepare(device, (ratio,))
        query = torch.ones(1, 8, 1, 2)
        key = torch.ones(1, 3, 2)
        weights = torch.ones(1, 8, 1)
        shared = {
            "compress_ratio": ratio, "sparse_count": 3,
            "minimum_key_indices": prepared.minimum_key_indices(device, ratio),
        }
        from_segments = attention.compressed_causal_topk(
            query, key, weights, query_segments=prepared.indexer_segments(ratio), **shared,
        )
        from_tensors = attention.compressed_causal_topk(
            query, key, weights, causal_upper=prepared.causal_key_upper_bounds(device, ratio), **shared,
        )
        torch.testing.assert_close(from_tensors, from_segments)


if __name__ == "__main__":
    unittest.main()
