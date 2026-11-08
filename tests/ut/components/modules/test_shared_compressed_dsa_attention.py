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
"""Independent sink attention and explicit fused-operator selection checks."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from hyper_parallel.core.dtensor.placement_types import Shard

from hyper_parallel.components.functional import sparse_flash_mla as attention_ops
from hyper_parallel.distributed.context_parallel.batch_plan import CPBatchPlanConfig, prepare_cp_batch
from hyper_parallel.distributed.context_parallel.compressed_document import document_compression_plan
from hyper_parallel.components.modules import CompressedSparseAttention2
from hyper_parallel.components.modules.module_family import FamilyShardingContext, get_family_build_info
from hyper_parallel.distributed.module_binding import MeshBinding, ParallelBinding, validate_parallel_module
from hyper_parallel.components.modules.attention.compressed_sparse_attention2 import indexer as attention
from hyper_parallel.components.modules.attention.compressed_sparse_attention2 import indexer_kl_loss as loss_impl
from hyper_parallel.components.modules.attention.compressed_sparse_attention2 import (
    compressed_sparse_attention2_family as family,
)
from hyper_parallel.components.modules.attention.compressed_sparse_attention2.document_layout import (
    SharedCompressedPackedSequence,
)
from hyper_parallel.components.modules.attention.compressed_sparse_attention2 import (
    compressed_sparse_attention2 as attention_impl,
)
from hyper_parallel.components.modules.attention.compressed_sparse_attention2.compressed_sparse_attention2 import (
    reference_sparse_attention,
)
from hyper_parallel.distributed.context_parallel.indexer_balance import indexer_task_plan, indexer_route
from hyper_parallel.distributed.context_parallel.index_key_dispatch import index_key_plan
from hyper_parallel.distributed.context_parallel.sequence_dispatch import document_prefix_route, sliding_window_route
from hyper_parallel.trainer.config import PlanOverride, ReplaceAction, Target, entries_to_module_replacements


def _source(index_source=False):
    """Provide only the source contract, without importing a Transformers model."""
    module = nn.Module()
    values = {"config": SimpleNamespace(o_groups=1, qk_rope_head_dim=0), "layer_idx": 0,
              "num_heads": 1, "num_key_value_groups": 1, "compress_ratio": int(index_source),
              "is_kv_source": index_source, "is_index_source": index_source, "kv_source_layer_idx": 0,
              "index_source_layer_idx": 0, "candidate_source_layer_idx": None, "head_dim": 512,
              "sliding_window": 128, "scaling": 512**-.5}
    for name, value in values.items():
        setattr(module, name, value)
    module.sinks = nn.Parameter(torch.zeros(1))
    if index_source:
        module.indexer = nn.Module()
        module.indexer.q_b_proj = nn.Linear(2, 128)
        module.indexer.weights_proj = nn.Linear(2, 1)
        values = {"compress_ratio": 1, "num_heads": 1, "head_dim": 128, "index_topk": 512, "owns_key": True,
                  "is_candidate_source": False, "uses_candidates": False, "candidate_topk_blocks": 1,
                  "candidate_block_size": 1, "loss_coeff": .001}
        for name, value in values.items():
            setattr(module.indexer, name, value)
    return attention_impl.CompressedSparseAttention2Module(module=module)


class TestSharedCompressedFusionPolicy(unittest.TestCase):
    """Explicit fusion must either run native code or explain why it cannot."""

    # Check internal dispatch boundaries before optional device kernels execute.
    # pylint: disable=protected-access

    def setUp(self) -> None:
        """Prepare small causal inputs for dispatch checks."""
        self.query = torch.ones(1, 2, 8, 128)
        self.key = torch.ones(1, 2, 128)
        self.weights = torch.ones(1, 2, 8)
        self.indices = torch.tensor([[[0, -1], [0, 1]]], dtype=torch.int32)

    def test_index_key_transport_reaches_family_and_shared_consumers(self):
        """The CP member owns dispatch and rejects incompatible shared bank layouts."""
        binding = ParallelBinding(MeshBinding(SimpleNamespace(tp_size=1)),
                                  cp_context=SimpleNamespace(size=2, rank=0))
        for transport in ("all_gather", "all_to_all_v"):
            with self.subTest(transport=transport):
                module = CompressedSparseAttention2(module=_source(True), context={"cp_size": 2},
                                                     index_k_comm=transport)
                module.bind_parallel(binding)
                self.assertEqual(module.indexer.index_k_comm, transport)
                plan = module.indexer._resolve_key_plan((3, 5), None, None)
                self.assertEqual(plan is not None, transport == "all_to_all_v")
                module.indexer.owns_key = False
                self.assertEqual(module.indexer._resolve_key_plan((3, 5), plan, None), plan)
                if plan is not None:
                    module.indexer.index_k_comm = "all_gather"
                    with self.assertRaisesRegex(ValueError, "Full/Reindex"):
                        module.indexer._resolve_key_plan((3, 5), plan, None)
                    module.indexer.index_k_comm = transport
                    module.indexer.owns_key = True
                    module.indexer.is_candidate_source = True
                    with self.assertRaisesRegex(NotImplementedError, "ordinary causal selection"):
                        module.indexer._resolve_key_plan((3, 5), None, None)
        with self.assertRaisesRegex(ValueError, "index_k_comm"):
            CompressedSparseAttention2(module=_source(), index_k_comm="invalid")
        with self.assertRaisesRegex(ValueError, "indexer_paged_k requires"):
            CompressedSparseAttention2(module=_source(), indexer_paged_k=True)
        for size in (1, 2):
            paged = CompressedSparseAttention2(module=_source(True), context={"cp_size": size},
                                              use_fused_ops=True, indexer_paged_k=True)
            self.assertTrue(paged.indexer.paged_key)

    def test_compact_index_key_aligns_kl_with_either_main_bank_layout(self):
        """Index K compaction must preserve teacher/student IDs with main AG or A2AV."""
        module = CompressedSparseAttention2(module=_source(index_source=True))
        plan = index_key_plan((3, 5), 2, 1, 1, "greedy", 2)
        main = torch.arange(8 * 128, dtype=torch.float32).view(1, 8, 128)
        compact = main[:, plan.global_positions(main.device)].clone().requires_grad_()
        selected = torch.tensor([[[3, -1], [7, 4]]], dtype=torch.int32)
        output = attention.CSA2IndexerOutput(
            selected, compact.detach(), None, self.query, self.weights,
            plan, plan.owner_view(compact), plan.owner_range[0],
        )
        segments = ((0, 2, 3, 8, 3),)
        expected_segments = ((0, 2, 0, 5, 3),)
        for offset in (0, 3):
            with self.subTest(main_key_offset=offset):
                with patch.object(attention_impl, "shared_compressed_indexer_kl_loss",
                                  return_value=torch.zeros(())) as loss:
                    module._apply_indexer_loss(output, self.query, main[:, offset:], segments, offset)
                args, kwargs = loss.call_args
                torch.testing.assert_close(args[1], main[:, 3:], rtol=0, atol=0)
                torch.testing.assert_close(args[4], main[:, 3:], rtol=0, atol=0)
                torch.testing.assert_close(args[5], torch.tensor([[[0, -1], [4, 1]]], dtype=torch.int32))
                self.assertEqual(kwargs["query_segments"], expected_segments)
                self.assertTrue(args[1].requires_grad)

    def test_indexer_requires_native_when_explicitly_selected(self):
        """Unsupported CPU inputs cannot silently satisfy native mode."""
        kwargs = {"compress_ratio": 1, "sparse_count": 2}
        reference = attention.compressed_causal_topk(self.query, self.key, self.weights, **kwargs)
        torch.testing.assert_close(reference, self.indices)
        with self.assertRaisesRegex(RuntimeError, "Fused LightningIndexer requires NPU inputs; got device=cpu"):
            attention.compressed_causal_topk(self.query, self.key, self.weights, use_fused=True, **kwargs)
        cases = (({"reduce_sum": lambda value: value}, NotImplementedError, "TP score reduction"),
                 ({"minimum_key_indices": torch.zeros(1, 2, dtype=torch.int32)}, ValueError, "query_segments"))
        for options, error, message in cases:
            with self.subTest(options=options), self.assertRaisesRegex(error, message):
                attention.compressed_causal_topk(self.query, self.key, self.weights,
                                                use_fused=True, **kwargs, **options)

    def test_kl_required_mode_never_invokes_reference(self):
        """Reject unsupported active KL before entering the reference function."""
        with patch.object(loss_impl._SharedCompressedIndexerKLLoss, "apply") as reference:
            with self.assertRaisesRegex(RuntimeError, "Fused KL requires NPU inputs; got device=cpu"):
                loss_impl.shared_compressed_indexer_kl_loss(
                    self.query, self.key, self.weights, torch.ones(1, 1, 2, 128), self.key,
                    self.indices, torch.zeros(1), attention_scale=1., loss_coeff=1.,
                    use_fused=True, query_segments=((0, 2, 0, 2, 0),),
                )
            reference.assert_not_called()

    def test_attention_required_mode_never_invokes_fallback(self):
        """Unsupported explicit fusion must leave the reference path untouched."""
        module = CompressedSparseAttention2(module=_source(), use_fused_ops=True)
        query = torch.ones(1, 1, 2, 512)
        with patch.object(attention_impl, "reference_sparse_attention") as reference:
            with self.assertRaisesRegex(RuntimeError, "fused attention is required.*layer=0"):
                module._run_sparse_attention(query, query, None, None, 0, 2, None, ((0, 2, 0, 2, 0),))
            reference.assert_not_called()

    def test_cpu_baseline_never_loads_optional_operators(self):
        """The eager baseline produces gradients without querying optional packages."""
        module = CompressedSparseAttention2(module=_source(), use_fused_ops=False)
        query = torch.ones(1, 1, 2, 512, requires_grad=True)
        with patch.object(
                attention_ops, "_load_attention_ops", side_effect=AssertionError("native package queried"),
        ):
            output = module._run_sparse_attention(query, query, None, None, 0, 2, None, ((0, 2, 0, 2, 0),))
            output.sum().backward()
            self.assertTrue(bool(output.isfinite().all()))
            self.assertTrue(bool(query.grad.isfinite().all()))

    def test_native_failure_is_not_converted_to_reference_execution(self):
        """A native exception is propagated after successful dispatch checks."""
        module = CompressedSparseAttention2(module=_source(), use_fused_ops=True)
        query = torch.ones(1, 1, 2, 512)
        with patch.object(attention_impl, "supports_fused_attention_inputs", return_value=True):
            with patch.object(attention_impl, "sparse_flash_mla", side_effect=RuntimeError("native failure")):
                with self.assertRaisesRegex(RuntimeError, "^native failure$"):
                    module._run_sparse_attention(query, query, None, None, 0, 2, None, ((0, 2, 0, 2, 0),))

    def test_attention_prefix_banks_have_canonical_bsnd_strides(self):
        """Check singleton stride normalization without requiring native kernels.

        Feature: SMLA BSND input adaptation.
        Description: Inspect raw/compressed prefixes for B=1/B=2 and r0/r1/r2.
        Expectation: Strides are canonical and values are unchanged; B=1 raw shares storage.
        """
        for batch, ratio in ((1, 0), (1, 1), (1, 2), (2, 2)):
            with self.subTest(batch=batch, ratio=ratio):
                query = torch.zeros(batch, 8, 4, 512, dtype=torch.bfloat16)
                raw = torch.arange(batch * 64 * 512).reshape(batch, 1, 64, 512).to(torch.bfloat16)
                main = raw[:, 0, :32].contiguous()
                indices = torch.zeros(batch, 4, 2, dtype=torch.int32)
                prefix = 4 // max(ratio, 1)
                with patch.object(attention_ops._SparseFlashMla, "apply") as native:
                    attention_ops._bsnd_sparse_flash_mla(
                        query, raw, main, indices, torch.zeros(8), 1., ratio, 128, 0,
                        (0, 4, 0, prefix, 0),
                    )
                native_raw, native_main = native.call_args.args[1:3]
                self.assertEqual(native_raw.stride(), (4 * 512, 512, 512, 1))
                torch.testing.assert_close(native_raw.squeeze(2), raw[:, 0, :4], rtol=0, atol=0)
                if batch == 1:
                    self.assertEqual(native_raw.data_ptr(), raw.data_ptr())
                if ratio:
                    self.assertEqual(native_main.stride(), (prefix * 512, 512, 512, 1))
                    torch.testing.assert_close(native_main.squeeze(2), main[:, :prefix], rtol=0, atol=0)
                else:
                    self.assertIsNone(native_main)

    def test_replacement_config_selects_fused_kernels(self):
        """Resolve the original baseline and fused kernels through the same replacement Target."""
        for use_fused_ops in (False, True):
            with self.subTest(use_fused_ops=use_fused_ops):
                entry = PlanOverride(
                    match="model.layers.*.self_attn",
                    replace=ReplaceAction(
                        kind="module", source_type="torch.nn.Module",
                        implementation=Target(
                            CompressedSparseAttention2,
                            target_path="hyper_parallel.components.modules.CompressedSparseAttention2",
                            use_fused_ops=use_fused_ops,
                        ),
                    ),
                )
                factory = entries_to_module_replacements([entry])[0].factory
                source = _source(True)
                parameters = {name: id(value) for name, value in source.named_parameters()}
                module = factory(module=source, module_fqn="model.layers.0.self_attn", context={})
                self.assertEqual(module.use_fused_ops, use_fused_ops)
                self.assertEqual(module.indexer.use_fused, use_fused_ops)
                self.assertEqual({name: id(value) for name, value in module.named_parameters()}, parameters)
        default = CompressedSparseAttention2(module=_source(True))
        self.assertFalse(default.use_fused_ops)
        self.assertFalse(default.indexer.use_fused)
        for value in ("false", "true", "auto", None, 0, 1):
            with self.subTest(invalid=value), self.assertRaisesRegex(TypeError, "use_fused_ops"):
                CompressedSparseAttention2(module=_source(), use_fused_ops=value)

    def test_candidate_execution_is_rejected_without_reference_fallback(self):
        """A candidate producer or consumer cannot enter its reference branch."""
        for produces, consumes in ((True, False), (False, True)):
            with self.subTest(produces=produces, consumes=consumes):
                source = _source(True)
                source.indexer.is_candidate_source = produces
                source.indexer.uses_candidates = consumes
                module = CompressedSparseAttention2(module=source, use_fused_ops=True)
                with self.assertRaisesRegex(NotImplementedError, "candidate-pool"):
                    module.indexer._select_indices(self.query, self.key, self.weights, None, 0, None, None, None)

    def test_zero_loss_does_not_require_an_unused_kernel(self):
        """A zero objective is not an attempted fused calculation."""
        query = self.query.clone().requires_grad_()
        loss = loss_impl.shared_compressed_indexer_kl_loss(
            query, self.key, self.weights, torch.ones(1, 1, 2, 128), self.key,
            self.indices, torch.zeros(1), attention_scale=1., loss_coeff=0.,
            use_fused=True,
        )
        loss.backward()
        torch.testing.assert_close(loss, torch.tensor(0.))
        torch.testing.assert_close(query.grad, torch.zeros_like(query))


def _selected_attention(query, key_value, indices, sinks, scale):
    """Enumerate visible keys without constructing the production dense mask."""
    batches = []
    for batch in range(query.shape[0]):
        rows = []
        for row in range(query.shape[2]):
            visible = indices[batch, row]
            visible = visible[visible >= 0].unique().long()
            keys = key_value[batch, 0].index_select(0, visible)
            scores = query[batch, :, row] @ keys.T * scale
            scores = torch.cat((scores, sinks[:, None]), dim=-1)
            probabilities = scores.softmax(dim=-1, dtype=torch.float32)
            rows.append(probabilities[:, :-1] @ keys)
        batches.append(torch.stack(rows))
    return torch.stack(batches)


class TestCompressedAttentionReference(unittest.TestCase):
    """The unfused path preserves sink, duplicate-index and empty-row semantics."""

    def test_selected_formula_and_all_gradients(self):
        """Compare a selected-key oracle with duplicate, padded and empty rows."""
        torch.manual_seed(1016)
        values = [torch.randn(2, 3, 4, 5), torch.randn(2, 1, 6, 5), torch.tensor([-3., 0., 8.])]
        indices = torch.tensor([
            [[-1, -1, -1, -1], [0, 0, 1, -1], [2, 3, -1, -1], [1, 3, 4, 5]],
            [[0, -1, -1, -1], [1, 2, -1, -1], [0, 3, 3, -1], [2, 4, 5, -1]],
        ])
        reference = [value.clone().requires_grad_() for value in values]
        candidate = [value.clone().requires_grad_() for value in values]
        expected = _selected_attention(reference[0], reference[1], indices, reference[2], .4)
        actual = reference_sparse_attention(candidate[0], candidate[1], indices, candidate[2], .4)
        gradient = torch.randn_like(expected)
        expected.backward(gradient)
        actual.backward(gradient)
        torch.testing.assert_close(actual, expected)
        for current, target in zip(candidate, reference):
            torch.testing.assert_close(current.grad, target.grad)
        torch.testing.assert_close(actual[0, 0], torch.zeros_like(actual[0, 0]), rtol=0, atol=0)


class TestNaturalDocumentCompression(unittest.TestCase):
    """Odd documents retain their tokens without creating cross-document compressed keys."""

    def test_pool_and_rotary_positions_match_independent_documents(self):
        """A length-three document and a length-five document produce one and two keys."""
        key = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2).requires_grad_()
        gate = torch.zeros_like(key, requires_grad=True)
        packed = SharedCompressedPackedSequence(torch.tensor([0, 3, 8]), 0, 8, 8)
        plan = packed.block_plan(2)
        actual = attention_impl.CSA2Compressor.pool(torch.cat((key, gate), dim=-1), plan)
        expected = torch.stack((key[:, :2].mean(1), key[:, 3:5].mean(1), key[:, 5:7].mean(1)), dim=1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward()
        expected_grad = torch.full_like(key, .5)
        expected_grad[:, [2, 7]] = 0
        torch.testing.assert_close(key.grad, expected_grad, rtol=0, atol=0)
        positions = torch.arange(8).reshape(1, 8, 1).float()
        cos, sin = packed.compressed_positions(plan, (positions, -positions))
        torch.testing.assert_close(cos.flatten(), torch.tensor([0., 3., 5.]), rtol=0, atol=0)
        torch.testing.assert_close(sin, -cos, rtol=0, atol=0)

    def test_causal_topk_excludes_other_documents_and_incomplete_blocks(self):
        """Compressed coordinates use cumulative complete blocks, not global token division."""
        packed = SharedCompressedPackedSequence(torch.tensor([0, 3, 8]), 0, 8, 8)
        packed = packed.prepare(torch.device("cpu"), (1, 2))
        selected = attention.compressed_causal_topk(
            torch.ones(1, 8, 1, 2), torch.ones(1, 3, 2), torch.ones(1, 8, 1),
            compress_ratio=2, sparse_count=3, query_segments=packed.indexer_segments(2),
            minimum_key_indices=packed.minimum_key_indices(torch.device("cpu"), 2),
        )
        expected = torch.tensor([[[-1, -1, -1], [0, -1, -1], [0, -1, -1], [-1, -1, -1],
                                  [1, -1, -1], [1, -1, -1], [1, 2, -1], [1, 2, -1]]], dtype=torch.int32)
        torch.testing.assert_close(selected, expected, rtol=0, atol=0)

    def test_cross_rank_block_ownership_includes_empty_owners(self):
        """The last-token owner completes each group, even when another owner has no group."""
        plans = [document_compression_plan((0, 3, 8), 2, 2, 4, rank) for rank in range(4)]
        self.assertEqual(plans[0].counts, (1, 0, 1, 1))
        expected_starts = ([0], [], [3], [5])
        for plan, starts in zip(plans, expected_starts):
            actual = (plan.indices(torch.device("cpu"))[1] + plan.route.global_start).tolist()
            self.assertEqual(actual, starts)
        for sender, source in enumerate(plans):
            for receiver, target in enumerate(plans):
                self.assertEqual(source.route.input_splits[receiver], target.route.output_splits[sender])

    def test_fused_empty_bank_retains_the_collective_gradient(self):
        """An all-empty prefix uses native SWA and returns an explicit zero bank gradient."""
        contract = SimpleNamespace(device=SimpleNamespace(type="npu"), dtype=torch.bfloat16, shape=(1, 8, 1, 512))
        self.assertTrue(attention_ops.supports_fused_attention_inputs(contract, 2, 128, 0))
        source = _source(True)
        source.compress_ratio = 2
        module = CompressedSparseAttention2(module=source, use_fused_ops=True)
        query = torch.ones(1, 1, 1, 512, requires_grad=True)
        key = torch.empty(1, 0, 512, requires_grad=True)
        indices = torch.empty(1, 1, 0, dtype=torch.int32)
        with patch.object(attention_impl, "supports_fused_attention_inputs", return_value=True):
            with patch.object(attention_impl, "sparse_flash_mla", return_value=query.transpose(1, 2)):
                output = module._run_sparse_attention(  # pylint: disable=protected-access
                    query, query, key, indices, 0, 1, None, ((0, 1, 0, 0, 1),),
                )
        output.sum().backward()
        self.assertIsNotNone(key.grad)
        torch.testing.assert_close(key.grad, torch.zeros_like(key), rtol=0, atol=0)


    def test_bsnd_natural_document_start_and_bank_offsets(self):
        """Canonical K ID two belongs to raw document six, despite compact prefix routing."""
        query = torch.zeros(1, 8, 2, 512, dtype=torch.bfloat16)
        raw = torch.arange(3, 8, dtype=torch.bfloat16).view(1, 1, 5, 1).expand(1, 1, 5, 512).contiguous()
        main = torch.arange(1, 3, dtype=torch.bfloat16).view(1, 2, 1).expand(1, 2, 512).contiguous()
        indices = torch.full((1, 2, 1), 2, dtype=torch.int32)
        with patch.object(attention_ops._SparseFlashMla, "apply") as native:  # pylint: disable=protected-access
            attention_ops._bsnd_sparse_flash_mla(  # pylint: disable=protected-access
                query, raw, main, indices, torch.zeros(8), 1., 2, 128, 6,
                (0, 2, 2, 3, 0), raw_key_offset=3, main_key_offset=1, raw_document_starts=((0, 6),),
            )
        native_raw, native_main, selected = native.call_args.args[1:4]
        torch.testing.assert_close(native_raw[:, :, 0], raw[:, 0, 3:5], rtol=0, atol=0)
        torch.testing.assert_close(native_main[:, :, 0], main[:, 1:2], rtol=0, atol=0)
        torch.testing.assert_close(selected, torch.zeros_like(selected), rtol=0, atol=0)
        self.assertEqual(native_raw.stride(), (1024, 512, 512, 1))
        self.assertEqual(native_main.stride(), (512, 512, 512, 1))

    def test_tnd_natural_document_ranges_use_raw_coordinates(self):
        """Packed native segments combine document-local raw ranges with compact K prefixes."""
        query = torch.zeros(1, 8, 4, 512, dtype=torch.bfloat16)
        raw = torch.arange(3, 8, dtype=torch.bfloat16).view(1, 1, 5, 1).expand(1, 1, 5, 512).contiguous()
        main = torch.arange(1, 3, dtype=torch.bfloat16).view(1, 2, 1).expand(1, 2, 512).contiguous()
        indices = torch.tensor([[[1], [1], [-1], [2]]], dtype=torch.int32)
        with patch.object(attention_ops._SparseFlashMlaTND, "apply",  # pylint: disable=protected-access
                          side_effect=lambda *args: args[0]) as native:
            output = attention_ops.sparse_flash_mla(
                query, raw, main, indices, torch.zeros(8), 1., 2, 128, 4,
                ((0, 2, 1, 2, 1), (2, 4, 2, 3, 0)), raw_key_offset=3, main_key_offset=1,
                raw_document_starts=((0, 3), (2, 6)),
            )
        native_raw, native_main = native.call_args.args[1:3]
        torch.testing.assert_close(native_raw[:, 0], raw[0, 0], rtol=0, atol=0)
        torch.testing.assert_close(native_main[:, 0], main[0], rtol=0, atol=0)
        torch.testing.assert_close(output, query.transpose(1, 2), rtol=0, atol=0)

    def test_fused_topk_width_uses_global_compact_key_count(self):
        """Logical K is three for documents 3/3/2, even when this rank stores only two keys."""
        source = _source(True)
        source.compress_ratio = 2
        module = CompressedSparseAttention2(module=source, use_fused_ops=True)
        query = torch.ones(1, 1, 4, 512)
        bank = torch.ones(1, 2, 512)
        indices = torch.full((1, 4, 3), -1, dtype=torch.int32)
        with patch.object(attention_impl, "supports_fused_attention_inputs", return_value=True):
            with patch.object(attention_impl, "sparse_flash_mla", return_value=query.transpose(1, 2)) as native:
                output = module._run_sparse_attention(  # pylint: disable=protected-access
                    query, query, bank, indices, 4, 8, None, ((0, 2, 1, 2, 1), (2, 4, 2, 3, 0)),
                    raw_key_offset=3, main_key_offset=1, raw_document_starts=((0, 3), (2, 6)), global_key_length=3,
                )
        native.assert_called_once()
        torch.testing.assert_close(output, query.transpose(1, 2), rtol=0, atol=0)


class TestSharedCompressedCPPlans(unittest.TestCase):
    """Check CPU-only configuration and document routing without operator packages."""

    def test_balance_options(self):
        """Greedy defaults to 2048; irrelevant or invalid options are rejected."""
        self.assertEqual(CompressedSparseAttention2(module=
            _source(), context={"cp_size": 2}, indexer_load_balance="greedy").indexer_balance_block_size, 2048)
        cases = (
            ({"indexer_balance_block_size": 2048}, "requires indexer_load_balance"),
            ({"indexer_load_balance": "zigzag", "indexer_balance_block_size": 2048}, "requires indexer"),
            ({"indexer_load_balance": "hybrid"}, "explicit positive"),
            ({"main_kv_comm": "invalid"}, "main_kv_comm"),
        )
        for options, message in cases:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                CompressedSparseAttention2(module=_source(), **options)

    def test_task_coverage_and_row_budget(self):
        """All real rows appear once, including leading zigzag remainders."""
        lengths, size = (10, 182, 64), 8
        ratio = 2
        for mode in ("zigzag", "greedy", "hybrid"):
            block = 8 if mode != "zigzag" else None
            threshold = 128 if mode == "hybrid" else None
            plan = indexer_task_plan(lengths, size, ratio, mode, block, threshold)
            rows = [(task.document, row) for tasks in plan for task in tasks
                    for row in range(task.start, task.end)]
            self.assertEqual(sorted(rows), [(doc, row) for doc, length in enumerate(lengths)
                                            for row in range(length)])
            for rank, tasks in enumerate(plan):
                route = indexer_route(lengths, size, rank, ratio, mode, block, threshold)
                count = sum(task.length for task in tasks)
                self.assertEqual(route.query_count, count)
                self.assertLessEqual(count, 32 if mode == "zigzag" else 64)
            routes = [indexer_route(lengths, size, rank, ratio, mode, block, threshold)
                      for rank in range(size)]
            for owner in range(size):
                for worker in range(size):
                    self.assertEqual(routes[owner].input_splits[worker], routes[worker].output_splits[owner])

    def test_document_bank_routes(self):
        """Halo and main-prefix routes match independently derived document bounds."""
        boundaries, local, size = (0, 10, 192, 256), 32, 8
        for ratio in (0, 1, 2):
            routes = [sliding_window_route(boundaries, local, size, rank, 8) if ratio == 0
                      else document_prefix_route(boundaries, local, size, rank, ratio) for rank in range(size)]
            for rank, route in enumerate(routes):
                document_start = max(start for start in boundaries if start <= rank * local)
                expected_start = max(document_start, rank * local - 7) if ratio == 0 else document_start // ratio
                self.assertEqual(route.global_start, expected_start)
                self.assertEqual(route.global_end, (rank+1) * local // max(1, ratio))
                for peer in range(size):
                    self.assertEqual(route.input_splits[peer], routes[peer].output_splits[rank])
        natural = document_prefix_route((0, 11, 256), local, size, 0, 2)
        self.assertEqual((natural.global_start, natural.global_end), (0, 15))


class TestCSA2Family(unittest.TestCase):
    """Check selection, identity, layouts and single communication ownership on CPU."""

    def test_source_identity_and_four_members(self):
        """Source parameters/buffers and frozen state survive every allowlisted selection."""
        for size in (1, 2):
            for fused in (False, True):
                with self.subTest(cp=size, fused=fused):
                    source = CompressedSparseAttention2(module=_source(True))
                    source.register_buffer("calibration", torch.ones(1), persistent=False)
                    source.sinks.requires_grad_(False)
                    source.eval()
                    source.indexer.q_b_proj.train()
                    selected = CompressedSparseAttention2(
                        module=source, context={"cp_size": size}, use_fused_ops=fused,
                    )
                    expected = ("cp_" if size > 1 else "") + ("fused" if fused else "reference")
                    self.assertEqual(get_family_build_info(selected).resolved.variant, expected)
                    self.assertEqual(source.state_dict().keys(), selected.state_dict().keys())
                    for name, parameter in source.named_parameters():
                        self.assertIs(selected.get_parameter(name), parameter)
                    self.assertIs(selected.calibration, source.calibration)
                    self.assertFalse(selected.training)
                    self.assertTrue(selected.indexer.q_b_proj.training)
                    self.assertFalse(selected.sinks.requires_grad)
                    self.assertNotIn("calibration", selected.state_dict())
                    checkpoint_keys = tuple(selected.state_dict())
                    selected.indexer.enable_selection_exclude()
                    self.assertEqual(tuple(selected.state_dict()), checkpoint_keys)

    def test_prepared_plan_is_bound_to_call_and_validates_document_and_transport(self):
        """Keep two live micro-batch plans independent and reject stale metadata."""
        module = CompressedSparseAttention2(module=_source(True), context={"cp_size": 2},
                                            index_k_comm="all_to_all_v")
        module.bind_parallel(ParallelBinding(MeshBinding(SimpleNamespace(tp_size=1)),
                                             cp_context=SimpleNamespace(size=2, rank=1)))
        config = CPBatchPlanConfig(2, 1, ratios=(1,), windows=(128,), index_k_comm="all_to_all_v")
        documents = []
        for lengths in ((3, 5), (1, 7)):
            packed = SharedCompressedPackedSequence(torch.tensor([0, lengths[0], 8]), 4, 4, 8)
            document = module.prepare_document_inputs(torch.zeros(1, 4, 2), torch.arange(4).unsqueeze(0), packed)
            plan = prepare_cp_batch(lengths, config)
            bound = module.attach_batch_plan(document, plan)
            self.assertIsNone(document.batch_plan)
            self.assertIs(bound.batch_plan, plan)
            self.assertIs(module.indexer_runtime_options(bound, attention_impl.SharedCompressedAttentionState())[
                "prepared_batch"], plan)
            documents.append(bound)
        self.assertEqual(documents[0].batch_plan.lengths, (3, 5))
        with self.assertRaisesRegex(ValueError, "document lengths"):
            module.attach_batch_plan(documents[1], documents[0].batch_plan)
        mismatched = prepare_cp_batch((3, 5), replace(config, index_k_comm="all_gather"))
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            module.attach_batch_plan(documents[0], mismatched)

    def test_complete_layout_and_invalid_override(self):
        """Every parameter is owned; invalid TP/CP placement overrides are rejected."""
        module = CompressedSparseAttention2(module=_source(True))
        context = FamilyShardingContext(("tp", "cp"), {"tp": 1, "cp": 2}, {})
        layout = family._sharding(module=module, module_fqn="", member=None, context=context)
        self.assertEqual(set(layout.params), set(dict(module.named_parameters())))
        validate_parallel_module(module, family._parallel_spec(module=module, member=None), layout, context)
        layout.params["sinks"] = {"cp": Shard(0)}
        with self.assertRaisesRegex(ValueError, "CSA2"):
            validate_parallel_module(module, family._parallel_spec(module=module, member=None), layout, context)

    def test_binding_keeps_class_forward_and_rejects_duplicate_owner(self):
        """Binding must not monkeypatch forward or permit a second communication owner."""
        module = CompressedSparseAttention2(module=_source(), context={"cp_size": 2})
        binding = ParallelBinding(MeshBinding(SimpleNamespace(tp_size=1)),
                                  cp_context=SimpleNamespace(size=2, rank=1))
        before = module.forward.__func__
        module.bind_parallel(binding)
        self.assertIs(module.forward.__func__, before)
        self.assertNotIn("forward", module.__dict__)
        # Document-local RoPE positions do not change the physical CP row offset.
        self.assertEqual(module.query_partition(2, torch.tensor([[0, 1]])), (2, 4))
        with self.assertRaisesRegex(ValueError, "already bound"):
            module.bind_parallel(binding)

    def test_unsupported_selectors_and_unbound_cp_fail(self):
        """Reject unsupported topology/kernel combinations before any collective runs."""
        source = CompressedSparseAttention2(module=_source())
        with self.assertRaises((ValueError, TypeError)):
            CompressedSparseAttention2(module=source, use_fused_ops="true")
        with self.assertRaisesRegex(NotImplementedError, "TP=1"):
            CompressedSparseAttention2(module=source, context={"tp_size": 2}, use_fused_ops=True)
        cp = CompressedSparseAttention2(module=source, context={"cp_size": 2})
        with self.assertRaisesRegex(RuntimeError, "must be bound"):
            cp.query_partition(2, torch.arange(2).unsqueeze(0))
