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
"""Compressor/Indexer graph geometry, precision, and eager pytree identity."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from hyper_parallel.compile.tracer.graph_tracer import trace_model_graph, run_traced_graph
from hyper_parallel.components.modules import shared_compressed_dsa_attention as attention
from hyper_parallel.models.deepseek_v41.adapter.data.runtime import DeepseekV41Runtime
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41Compressor, DeepseekV41Indexer
from tests.common.mark_utils import arg_mark


class TestIndexerKLGraph(unittest.TestCase):
    """Compare the captured analytic KL backward with an independent autograd formula."""

    @staticmethod
    def _inputs():
        generator = torch.Generator().manual_seed(71)
        return (torch.randn(1, 4, 2, 3, generator=generator),
                torch.randn(1, 3, 3, generator=generator),
                torch.randn(1, 4, 2, generator=generator),
                torch.randn(1, 2, 4, 5, generator=generator),
                torch.randn(1, 3, 5, generator=generator),
                torch.randn(2, generator=generator))

    @staticmethod
    def _oracle(query, key, weights, teacher_query, teacher_key, sinks, indices):
        safe = indices.clamp_min(0).long()
        valid = indices >= 0
        batch_index = torch.arange(query.shape[0]).view(-1, 1, 1)
        selected = key[batch_index, safe]
        dots = torch.einsum("bshd,bskd->bshk", query, selected).relu()
        logits = (dots * weights.unsqueeze(-1)).sum(2).masked_fill(~valid, -1.0e9)
        with torch.no_grad():
            selected_teacher = teacher_key[batch_index, safe]
            teacher_scores = torch.einsum("bhsd,bskd->bhsk", teacher_query, selected_teacher) * 0.5
            teacher_scores = teacher_scores.masked_fill(~valid.unsqueeze(1), -1.0e9)
            sink_logits = sinks.view(1, -1, 1, 1).expand(*teacher_scores.shape[:-1], 1)
            target = torch.cat((teacher_scores, sink_logits), -1).softmax(-1)[..., :-1]
            target = target.masked_fill(~valid.unsqueeze(1), 0.).mean(1)
            target = target / target.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        loss = (target * (target.clamp_min(torch.finfo(torch.float32).tiny).log() - logits.log_softmax(-1))).sum(-1)
        return loss.masked_fill(~valid.any(-1), 0.).mean() * 0.001

    def test_analytic_loss_and_gradients_match_autograd_with_empty_rows(self) -> None:
        """Mixed and fully empty causal rows retain finite losses and exact zero contributions."""
        for indices in (torch.tensor([[[-1, -1], [0, -1], [1, 2], [0, 2]]]),
                        torch.full((1, 4, 2), -1)):
            with self.subTest(indices=indices.tolist()):
                values = self._inputs()
                actual_args = [value.clone().requires_grad_() for value in values[:3]]
                expected_args = [value.clone().requires_grad_() for value in values[:3]]
                teacher = [value.clone().requires_grad_() for value in values[3:]]
                actual = attention.shared_compressed_indexer_kl_loss(
                    *actual_args, *teacher[:2], indices, teacher[2], attention_scale=0.5, loss_coeff=0.001,
                )
                expected = self._oracle(*expected_args, *teacher, indices)
                actual.backward()
                expected.backward()
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-8)
                for observed, reference in zip(actual_args, expected_args):
                    torch.testing.assert_close(observed.grad, reference.grad, rtol=2e-5, atol=1e-8)
                    self.assertTrue(bool(torch.isfinite(observed.grad).all()))
                for value in teacher:
                    self.assertIsNone(value.grad)

    def test_joint_graph_replays_changed_validity_without_recompiling(self) -> None:
        """The same forward/backward graph handles both valid and entirely masked rows."""
        values = self._inputs()

        class IndexerLoss(nn.Module):
            def __init__(self) -> None:
                """Initialize the regression fixture."""
                super().__init__()
                self.query = nn.Parameter(values[0])
                self.key = nn.Parameter(values[1])
                self.weights = nn.Parameter(values[2])

            def forward(self, indices: torch.Tensor) -> torch.Tensor:
                """Evaluate the graph regression fixture."""
                return attention.shared_compressed_indexer_kl_loss(
                    self.query, self.key, self.weights, values[3], values[4], indices, values[5],
                    attention_scale=0.5, loss_coeff=0.001,
                )

        model = IndexerLoss()
        first = torch.tensor([[[-1, -1], [0, -1], [1, 2], [0, 2]]])
        graph = trace_model_graph(model, lambda module, indices: module(indices), {"indices": first})
        for indices in (torch.full_like(first, -1), first):
            actual, grads, _ = run_traced_graph(graph, model, {"indices": indices})
            expected = self._oracle(model.query, model.key, model.weights, *values[3:], indices)
            expected_grads = torch.autograd.grad(expected, tuple(model.parameters()))
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-8)
            for observed, reference in zip(grads, expected_grads):
                torch.testing.assert_close(observed, reference, rtol=2e-5, atol=1e-8)


class _AttentionModel(nn.Module):
    """Exercise production Compressor, Indexer, sparse attention, and KL together."""

    def __init__(self, compress_ratio: int = 2) -> None:
        """Construct a CPU-sized attention contract without a model-wide crop."""
        super().__init__()
        config = SimpleNamespace(hidden_size=8, head_dim=4, rms_norm_eps=1e-6, q_lora_rank=4,
                                 index_n_heads=2, index_head_dim=4, index_topk=2,
                                 v41_kv_source_layer_ids=[0], v41_candidate_source_layer_id=-1,
                                 v41_indexer_loss_coeff=0.1, o_groups=1, qk_rope_head_dim=2)
        source = nn.Module()
        values = {"config": config, "layer_idx": 0, "num_heads": 2, "num_key_value_groups": 2,
                  "compress_ratio": compress_ratio, "is_kv_source": True, "is_index_source": True,
                  "kv_source_layer_idx": 0,
                  "index_source_layer_idx": 0, "candidate_source_layer_idx": None, "head_dim": 4,
                  "sliding_window": 3, "scaling": 0.5}
        for key, value in values.items():
            setattr(source, key, value)
        source.q_a_proj = nn.Linear(8, 4, bias=False)
        source.q_a_norm = nn.LayerNorm(4)
        source.q_b_proj = nn.Linear(4, 8, bias=False)
        source.kv_proj = nn.Linear(8, 4, bias=False)
        source.kv_norm = nn.LayerNorm(4)
        source.o_a_proj = nn.Linear(8, 4, bias=False)
        source.o_b_proj = nn.Linear(4, 8, bias=False)
        source.sinks = nn.Parameter(torch.zeros(2))
        source.compressor = DeepseekV41Compressor(config, compress_ratio)
        source.indexer = DeepseekV41Indexer(config, compress_ratio, 0)
        self.attention = attention.SharedCompressedDSAAttentionBase(
            source, replace_indexer=False, use_fused_ops=False,
        )

    def forward(self, hidden: torch.Tensor, packed: object) -> torch.Tensor:
        """Return a real attention objective including the injected Indexer gradient."""
        positions = torch.arange(hidden.shape[1]).unsqueeze(0)
        phase = positions.unsqueeze(-1).float() * 0.1
        embeddings = (phase.cos(), phase.sin())
        output, _ = self.attention(
            hidden, {"main": embeddings, "compress": embeddings}, positions, None,
            packed_seq_params=packed, shared_attention_state=attention.SharedCompressedAttentionState(),
        )
        return output.square().mean()


def _packed(boundaries: list[int], *, graph: bool, compress_ratio: int = 2) -> object:
    """Use the production graph-boundary conversion with final model ratios."""
    ids = torch.arange(boundaries[-1]).unsqueeze(0)
    packed = attention.SharedCompressedPackedSequence(torch.tensor(boundaries), 0, boundaries[-1], boundaries[-1])
    if not graph:
        return packed
    runtime = DeepseekV41Runtime(include_position_ids=False, include_image_sequence_start=False)
    return runtime.prepare_graph_inputs(
        {"input_ids": ids, "packed_seq_params": packed},
        model_config=SimpleNamespace(v41_compress_ratios=[compress_ratio]),
    )["packed_seq_params"]


class TestCompressedPackingGraph(unittest.TestCase):
    """Verify shared packed inputs with raw boundaries or prepared tensor geometry."""

    def test_query_bounds_match_document_oracle_for_local_shards(self) -> None:
        """Shared eager/graph bounds preserve document prefixes and CP query offsets."""
        boundaries = [0, 3, 9, 14]
        for ratio in (1, 2, 3):
            lower, upper = [], []
            base = 0
            for begin, end in zip(boundaries, boundaries[1:]):
                lower.extend([base] * (end - begin))
                upper.extend(base + (position - begin + 1) // ratio for position in range(begin, end))
                base += (end - begin) // ratio
            for start, length in ((0, 14), (0, 5), (4, 5), (9, 5)):
                with self.subTest(ratio=ratio, start=start, length=length):
                    packed = attention.SharedCompressedPackedSequence(
                        torch.tensor(boundaries), start, length, boundaries[-1],
                    ).prepare(torch.device("cpu"), (ratio,))
                    torch.testing.assert_close(
                        packed.minimum_key_indices(torch.device("cpu"), ratio),
                        torch.tensor([lower[start:start + length]]),
                    )
                    torch.testing.assert_close(
                        packed.maximum_key_indices(torch.device("cpu"), ratio),
                        torch.tensor([upper[start:start + length]]),
                    )

    def test_pooling_and_topk_exclude_padding_for_natural_documents(self) -> None:
        """Complete blocks and document-local causal visibility match the host oracle."""
        for boundaries in ([0, 4, 8], [0, 3, 8], [0, 1, 3, 7], list(range(9))):
            with self.subTest(boundaries=boundaries):
                packed = _packed(boundaries, graph=True)
                flat, spec = tree_flatten(packed)
                packed = tree_unflatten(flat, spec)
                length = boundaries[-1]
                key = torch.arange(length * 3).reshape(1, length, 3).float().requires_grad_()
                gate = (key.detach() / 10).requires_grad_()
                actual = packed.block_plan(2, 1, 0).pool(key, gate, None)
                blocks = [key[:, start:start+2] * gate[:, start:start+2].softmax(1)
                          for begin, end in zip(boundaries, boundaries[1:]) for start in range(begin, end-1, 2)]
                expected = torch.stack([block.sum(1) for block in blocks], 1) if blocks else key[:, :0]
                torch.testing.assert_close(actual[:, :len(blocks)], expected)
                selected = attention.compressed_causal_topk(
                    torch.ones(1, length, 1, 3), actual.detach(), torch.ones(1, length, 1),
                    compress_ratio=2, sparse_count=2,
                    minimum_key_indices=packed.minimum_key_indices(torch.device("cpu"), 2),
                    maximum_key_indices=packed.maximum_key_indices(torch.device("cpu"), 2),
                )
                base = 0
                for begin, end in zip(boundaries, boundaries[1:]):
                    for position in range(begin, end):
                        valid = selected[0, position]
                        valid = valid[valid >= 0]
                        self.assertTrue(bool(((valid >= base) & (valid < base+(position-begin+1)//2)).all()))
                    base += (end-begin)//2

    def test_joint_graph_matches_eager_attention_loss_and_all_parameter_gradients(self) -> None:
        """A single guarded graph covers alignment, document-count, and length changes."""
        torch.manual_seed(94)
        model = _AttentionModel()
        first = {"hidden": torch.randn(1, 8, 8), "packed": _packed([0, 4, 8], graph=True)}
        graph = trace_model_graph(model, lambda module, **inputs: module(**inputs), first, dynamic=True)
        for boundaries in ([0, 3, 8], [0, 1, 3, 8], [0, 3, 10], [0, 3, 7], list(range(9))):
            with self.subTest(boundaries=boundaries):
                hidden = torch.randn(1, boundaries[-1], 8)
                expected = model(hidden, _packed(boundaries, graph=False))
                gradients = torch.autograd.grad(expected, tuple(model.parameters()), allow_unused=True)
                actual, observed, _ = run_traced_graph(
                    graph, model, {"hidden": hidden, "packed": _packed(boundaries, graph=True)},
                )
                torch.testing.assert_close(actual, expected.detach(), rtol=2e-5, atol=1e-7)
                for (name, parameter), result, reference in zip(model.named_parameters(), observed, gradients):
                    reference = torch.zeros_like(parameter) if reference is None else reference
                    torch.testing.assert_close(result, reference, rtol=3e-4, atol=3e-6, msg=name)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_compressed_tree_schema_and_eager_boundaries_are_stable(self) -> None:
        """The graph subtype carries tensors while ordinary packing remains an opaque leaf.

        Feature: Graph-only pytree registration with shared packing implementation.
        Description: Round-trip prepared metadata and ordinary eager inputs through pytree.
        Expectation: Graph subtype and eager object identity survive their respective round-trips.
        """
        packed = _packed([0, 4, 8], graph=True)
        self.assertIs(type(packed), attention.GraphPackedSequence)
        first, spec = tree_flatten(packed)
        second, other = tree_flatten(_packed([0, 3, 10], graph=True))
        self.assertEqual(spec, other)
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in first+second))
        restored_graph = tree_unflatten(first, spec)
        self.assertIs(type(restored_graph), attention.GraphPackedSequence)
        self.assertIs(restored_graph.prepare(torch.device("cpu"), (2,)), restored_graph)
        eager = _packed([0, 3, 8], graph=False)
        leaves, schema = tree_flatten(eager)
        restored = tree_unflatten(leaves, schema)
        self.assertIs(restored, eager)
        self.assertIs(type(restored), attention.SharedCompressedPackedSequence)
        self.assertEqual(len(leaves), 1)
        self.assertIs(leaves[0], eager)
        torch.testing.assert_close(restored.cu_seq_lens, eager.cu_seq_lens)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_eager_tensor_mapping_preserves_metadata_identity_loss_and_gradients(self) -> None:
        """Tensor input mapping keeps packing opaque and preserves real attention semantics.

        Feature: Eager packing identity under input pytree transformations.
        Description: Clone input tensors for aligned, unaligned, and empty compression banks.
        Expectation: Packing remains the same object and loss plus all parameter gradients match.
        """
        for ratio in (1, 2):
            torch.manual_seed(94)
            model = _AttentionModel(ratio)
            for boundaries in ([0, 4, 8], [0, 3, 8], list(range(9))):
                with self.subTest(ratio=ratio, boundaries=boundaries):
                    eager = _packed(boundaries, graph=False).prepare(torch.device("cpu"), (ratio,))
                    hidden = torch.randn(1, boundaries[-1], 8)
                    mapped = tree_map(lambda value: value.clone() if isinstance(value, torch.Tensor) else value,
                                      {"hidden": hidden, "packed": eager})
                    self.assertIs(mapped["packed"], eager)
                    self.assertIsNot(mapped["hidden"], hidden)
                    expected = model(hidden, eager)
                    expected_grads = torch.autograd.grad(expected, tuple(model.parameters()), allow_unused=True)
                    actual = model(mapped["hidden"], mapped["packed"])
                    actual_grads = torch.autograd.grad(actual, tuple(model.parameters()), allow_unused=True)
                    torch.testing.assert_close(actual, expected)
                    for (name, parameter), result, reference in zip(
                            model.named_parameters(), actual_grads, expected_grads,
                    ):
                        result = torch.zeros_like(parameter) if result is None else result
                        reference = torch.zeros_like(parameter) if reference is None else reference
                        torch.testing.assert_close(result, reference, msg=name)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_prepared_geometry_rejects_unsupported_host_dispatch(self) -> None:
        """Unifying the input type does not enable fused, candidate, or CP graph execution.

        Feature: Prepared packing capability validation.
        Description: Request fused selection, candidate pools, and host CP geometry.
        Expectation: Unsupported requests raise explicit errors without reading boundaries.
        """
        packed = _packed([0, 3, 8], graph=True)
        for options in ({"use_fused": True}, {"use_candidates": True}):
            with self.subTest(options=options):
                with self.assertRaisesRegex(NotImplementedError, "reference DSA without candidate"):
                    packed.indexer_segments(2, **options)
        with self.assertRaisesRegex(NotImplementedError, "CP1"):
            packed.block_plan(2, 2, 0)
        with self.assertRaisesRegex(NotImplementedError, "CP1"):
            packed.document_boundaries()

    def test_multi_chunk_graph_replays_changed_lengths_and_parameter_gradients(self) -> None:
        """A two-chunk graph preserves symbolic tails for ratio-one and pooled banks."""
        for ratio in (1, 2):
            torch.manual_seed(94)
            model = _AttentionModel(ratio)
            first = {"hidden": torch.randn(1, 270, 8),
                     "packed": _packed([0, 103, 270], graph=True, compress_ratio=ratio)}
            graph = trace_model_graph(model, lambda module, **inputs: module(**inputs), first, dynamic=True)
            for boundaries in ([0, 103, 300], [0, 51, 103, 260], [0, 103, 512]):
                with self.subTest(ratio=ratio, boundaries=boundaries):
                    hidden = torch.randn(1, boundaries[-1], 8)
                    expected = model(hidden, _packed(boundaries, graph=False, compress_ratio=ratio))
                    gradients = torch.autograd.grad(expected, tuple(model.parameters()), allow_unused=True)
                    actual, observed, _ = run_traced_graph(
                        graph, model,
                        {"hidden": hidden, "packed": _packed(boundaries, graph=True, compress_ratio=ratio)},
                    )
                    torch.testing.assert_close(actual, expected.detach(), rtol=2e-5, atol=1e-7)
                    for (name, parameter), result, reference in zip(model.named_parameters(), observed, gradients):
                        reference = torch.zeros_like(parameter) if reference is None else reference
                        torch.testing.assert_close(result, reference, rtol=3e-4, atol=3e-6, msg=name)
            with self.assertRaisesRegex(ValueError, "shape guards"):
                run_traced_graph(
                    graph, model,
                    {"hidden": torch.randn(1, 513, 8),
                     "packed": _packed([0, 103, 513], graph=True, compress_ratio=ratio)},
                )


if __name__ == "__main__":
    unittest.main()
