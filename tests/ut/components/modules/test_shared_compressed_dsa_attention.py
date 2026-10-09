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
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.components.functional import compressed_smla
from hyper_parallel.components.modules import shared_compressed_dsa_attention as attention
from hyper_parallel.components.modules.shared_compressed_dsa_attention import reference_sparse_attention
from hyper_parallel.trainer.config import PlanOverride, Target, entries_to_module_replacements


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
    return module


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

    def test_indexer_requires_native_when_explicitly_selected(self):
        """Unsupported CPU inputs cannot silently satisfy native mode."""
        kwargs = {"compress_ratio": 1, "sparse_count": 2}
        reference = attention.compressed_causal_topk(self.query, self.key, self.weights, **kwargs)
        torch.testing.assert_close(reference, self.indices)
        with self.assertRaisesRegex(RuntimeError, "fused Indexer is required.*device=cpu"):
            attention.compressed_causal_topk(self.query, self.key, self.weights, use_fused=True, **kwargs)

    def test_kl_required_mode_never_invokes_reference(self):
        """Reject unsupported active KL before entering the reference function."""
        with patch.object(attention._SharedCompressedIndexerKLLoss, "apply") as reference:
            with self.assertRaisesRegex(RuntimeError, "fused KL is required.*device=cpu"):
                attention.shared_compressed_indexer_kl_loss(
                    self.query, self.key, self.weights, torch.ones(1, 1, 2, 128), self.key,
                    self.indices, torch.zeros(1), attention_scale=1., loss_coeff=1.,
                    use_fused=True, query_segments=((0, 2, 0, 2, 0),),
                )
            reference.assert_not_called()

    def test_attention_required_mode_never_invokes_fallback(self):
        """Unsupported explicit fusion must leave the reference path untouched."""
        module = attention.SharedCompressedDSAAttention(_source(), use_fused_ops=True)
        query = torch.ones(1, 1, 2, 512)
        with patch.object(attention, "reference_sparse_attention") as reference:
            with self.assertRaisesRegex(RuntimeError, "fused attention is required.*layer=0"):
                module._run_sparse_attention(query, query, None, None, 0, 2, None, ((0, 2, 0, 2, 0),))
            reference.assert_not_called()

    def test_cpu_baseline_never_loads_optional_operators(self):
        """The eager baseline produces gradients without querying optional packages."""
        module = attention.SharedCompressedDSAAttention(_source(), use_fused_ops=False)
        query = torch.ones(1, 1, 2, 512, requires_grad=True)
        with patch.object(compressed_smla, "_load_attention_ops", side_effect=AssertionError("native package queried")):
            output = module._run_sparse_attention(query, query, None, None, 0, 2, None, ((0, 2, 0, 2, 0),))
            output.sum().backward()
            self.assertTrue(bool(output.isfinite().all()))
            self.assertTrue(bool(query.grad.isfinite().all()))

    def test_native_failure_is_not_converted_to_reference_execution(self):
        """A native exception is propagated after successful dispatch checks."""
        module = attention.SharedCompressedDSAAttention(_source(), use_fused_ops=True)
        query = torch.ones(1, 1, 2, 512)
        with patch.object(attention, "supports_fused_attention_inputs", return_value=True):
            with patch.object(attention, "fused_sparse_mla_attention", side_effect=RuntimeError("native failure")):
                with self.assertRaisesRegex(RuntimeError, "^native failure$"):
                    module._run_sparse_attention(query, query, None, None, 0, 2, None, ((0, 2, 0, 2, 0),))

    def test_replacement_config_selects_fused_kernels(self):
        """Resolve the original baseline and fused kernels through the same replacement Target."""
        for use_fused_ops in (False, True):
            with self.subTest(use_fused_ops=use_fused_ops):
                entry = PlanOverride(
                    match="model.layers.*.self_attn", module_type="torch.nn.Module",
                    replace_module=Target(attention.SharedCompressedDSAAttention,
                                          target_path=f"{attention.__name__}.SharedCompressedDSAAttention",
                                          use_fused_ops=use_fused_ops),
                )
                factory = entries_to_module_replacements([entry])[0].factory
                source = _source(True)
                parameters = {name: id(value) for name, value in source.named_parameters()}
                module = factory(module=source, module_fqn="model.layers.0.self_attn", context={})
                self.assertEqual(module.use_fused_ops, use_fused_ops)
                self.assertEqual(module.indexer.use_fused, use_fused_ops)
                self.assertEqual({name: id(value) for name, value in module.named_parameters()}, parameters)
        default = attention.SharedCompressedDSAAttention(_source(True))
        self.assertFalse(default.use_fused_ops)
        self.assertFalse(default.indexer.use_fused)
        for value in ("false", "true", "auto", None, 0, 1):
            with self.subTest(invalid=value), self.assertRaisesRegex(ValueError, "use_fused_ops must be a bool"):
                attention.SharedCompressedDSAAttention(_source(), use_fused_ops=value)

    def test_candidate_execution_is_rejected_without_reference_fallback(self):
        """A candidate producer or consumer cannot enter its reference branch."""
        for produces, consumes in ((True, False), (False, True)):
            with self.subTest(produces=produces, consumes=consumes):
                source = _source(True)
                source.indexer.is_candidate_source = produces
                source.indexer.uses_candidates = consumes
                module = attention.SharedCompressedDSAAttention(source, use_fused_ops=True)
                with self.assertRaisesRegex(NotImplementedError, "candidate-pool"):
                    module.indexer._select_indices(self.query, self.key, self.weights, None, 0, None, None, None)

    def test_zero_loss_does_not_require_an_unused_kernel(self):
        """A zero objective is not an attempted fused calculation."""
        query = self.query.clone().requires_grad_()
        loss = attention.shared_compressed_indexer_kl_loss(
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
