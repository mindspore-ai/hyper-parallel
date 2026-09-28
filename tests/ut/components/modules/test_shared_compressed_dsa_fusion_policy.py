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
"""CPU checks for an explicit native-fusion requirement and long-K dispatch."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.components.functional import compressed_indexer_ops
from hyper_parallel.components.modules import shared_compressed_dsa_attention as attention
from hyper_parallel.trainer.config import PlanOverride, Target, entries_to_module_replacements
from tests.common.mark_utils import arg_mark


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

    def setUp(self):
        self.query = torch.ones(1, 2, 8, 128)
        self.key = torch.ones(1, 2, 128)
        self.weights = torch.ones(1, 2, 8)
        self.indices = torch.tensor([[[0, -1], [0, 1]]], dtype=torch.int32)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_indexer_requires_native_when_explicitly_selected(self):
        """Feature: Required Indexer.
        Description: Unsupported CPU inputs cannot silently satisfy native mode.
        Expectation: Fused mode raises, while explicit reference mode retains reference indices.
        """
        kwargs = {"compress_ratio": 1, "sparse_count": 2}
        reference = attention.compressed_causal_topk(self.query, self.key, self.weights, **kwargs)
        torch.testing.assert_close(reference, self.indices)
        with self.assertRaisesRegex(RuntimeError, "fused Indexer is required.*device=cpu"):
            attention.compressed_causal_topk(self.query, self.key, self.weights, use_fused=True, **kwargs)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_kl_required_mode_never_invokes_reference(self):
        """Feature: Required KL.
        Description: Reject unsupported active KL before entering the reference function.
        Expectation: A diagnostic is raised and the reference implementation is not called.
        """
        with patch.object(attention._SharedCompressedIndexerKLLoss, "apply") as reference:
            with self.assertRaisesRegex(RuntimeError, "fused KL is required.*device=cpu"):
                attention.shared_compressed_indexer_kl_loss(
                    self.query, self.key, self.weights, torch.ones(1, 1, 2, 128), self.key,
                    self.indices, torch.zeros(1), attention_scale=1., loss_coeff=1.,
                    use_fused=True, query_segments=((0, 2, 0, 2, 0),),
                )
            reference.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_attention_required_mode_never_invokes_fallback(self):
        """Feature: Required main attention.
        Description: Unsupported CPU input is not redirected to the numerical oracle.
        Expectation: The error identifies the layer and both fallback paths stay unused.
        """
        module = attention.SharedCompressedDSAAttention(_source(), use_fused_kernels=True)
        query = torch.ones(1, 1, 2, 512)
        with patch.object(attention, "_reference_sparse_attention") as reference:
            with patch.object(attention, "npu_sparse_attention_with_scalar_sink") as omni:
                with self.assertRaisesRegex(RuntimeError, "fused attention is required.*layer=0"):
                    module._apply_sparse_attention(query, query, None, None, 0, 2, None, ((0, 2, 0, 2, 0),))
                reference.assert_not_called()
                omni.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_cpu_baseline_never_loads_optional_operators(self):
        """Feature: Reference execution without NPU dependencies.
        Description: Execute attention through the same replacement used by CP.
        Expectation: Reference outputs and gradients are finite and no optional package is queried.
        """
        module = attention.SharedCompressedDSAAttention(_source(), use_fused_kernels=False)
        query = torch.ones(1, 1, 2, 512, requires_grad=True)
        with patch.object(compressed_indexer_ops, "_operators", side_effect=AssertionError("native package queried")):
            with patch.object(attention, "npu_sparse_attention_with_scalar_sink") as omni:
                output = module._apply_sparse_attention(query, query, None, None, 0, 2, None, ((0, 2, 0, 2, 0),))
                output.sum().backward()
                self.assertTrue(bool(output.isfinite().all()))
                self.assertTrue(bool(query.grad.isfinite().all()))
                omni.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_npu_baseline_uses_original_omni_bridge(self):
        """Feature: Original NPU baseline.
        Description: Mock the NPU query device while retaining real bank and index assembly.
        Expectation: Disabled fusion dispatches to Omni with original inputs and propagates Omni failures.
        """
        module = attention.SharedCompressedDSAAttention(_source(True), use_fused_kernels=False)
        query = SimpleNamespace(shape=(1, 1, 2, 512), device=SimpleNamespace(type="npu"))
        raw = torch.zeros(1, 1, 2, 512)
        compressed = torch.ones(1, 2, 512)
        expected = torch.ones(1, 2, 1, 512)
        with (
            patch.object(attention, "build_sliding_window_indices", return_value=self.indices),
            patch.object(attention, "npu_sparse_attention_with_scalar_sink", return_value=expected) as omni,
            patch.object(attention, "_reference_sparse_attention") as reference,
        ):
            output = module._apply_sparse_attention(query, raw, compressed, self.indices, 0, 2, None, ())
            self.assertIs(output, expected)
            args = omni.call_args.args
            self.assertIs(args[0], query)
            torch.testing.assert_close(args[1], torch.cat((raw, compressed.unsqueeze(1)), dim=2))
            torch.testing.assert_close(args[2], torch.tensor([[[0, -1, 2, -1], [0, 1, 2, 3]]]))
            self.assertIs(args[3], module.sinks)
            self.assertEqual(args[4:], (module.rope_head_dim, module.scaling))
            omni.side_effect = RuntimeError("Omni unavailable")
            with self.assertRaisesRegex(RuntimeError, "^Omni unavailable$"):
                module._apply_sparse_attention(query, raw, compressed, self.indices, 0, 2, None, ())
            reference.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_native_failure_is_not_converted_to_reference_execution(self):
        """Feature: Native runtime failures.
        Description: A native exception is propagated after successful dispatch checks.
        Expectation: The original exception reaches the caller.
        """
        module = attention.SharedCompressedDSAAttention(_source(), use_fused_kernels=True)
        query = torch.ones(1, 1, 2, 512)
        with patch.object(attention, "fused_attention_available", return_value=True):
            with patch.object(attention, "fused_sparse_mla_attention", side_effect=RuntimeError("native failure")):
                with self.assertRaisesRegex(RuntimeError, "^native failure$"):
                    module._apply_sparse_attention(query, query, None, None, 0, 2, None, ((0, 2, 0, 2, 0),))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_replacement_config_selects_fused_kernels(self):
        """Feature: Declarative implementation selection.
        Description: Resolve the original baseline and fused kernels through the same replacement Target.
        Expectation: Configuration sets the intended implementations without changing parameter identity.
        """
        for use_fused_kernels in (False, True):
            with self.subTest(use_fused_kernels=use_fused_kernels):
                entry = PlanOverride(
                    match="model.layers.*.self_attn", module_type="torch.nn.Module",
                    replace_module=Target(attention.SharedCompressedDSAAttention,
                                          target_path=f"{attention.__name__}.SharedCompressedDSAAttention",
                                          use_fused_kernels=use_fused_kernels),
                )
                factory = entries_to_module_replacements([entry])[0].factory
                source = _source(True)
                parameters = {name: id(value) for name, value in source.named_parameters()}
                module = factory(module=source, module_fqn="model.layers.0.self_attn", context={})
                self.assertEqual(module.use_optimized_sparse_attention, use_fused_kernels)
                self.assertEqual(module.indexer.use_fused, use_fused_kernels)
                self.assertEqual({name: id(value) for name, value in module.named_parameters()}, parameters)
        default = attention.SharedCompressedDSAAttention(_source(True))
        self.assertFalse(default.use_optimized_sparse_attention)
        self.assertFalse(default.indexer.use_fused)
        for value in ("false", "true", "auto", None, 0, 1):
            with self.subTest(invalid=value), self.assertRaisesRegex(ValueError, "use_fused_kernels must be a bool"):
                attention.SharedCompressedDSAAttention(_source(), use_fused_kernels=value)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_candidate_execution_is_rejected_without_reference_fallback(self):
        """Feature: Native topology boundary.
        Description: A candidate producer or consumer cannot enter its reference branch.
        Expectation: Unsupported candidate selection raises before reference execution.
        """
        for produces, consumes in ((True, False), (False, True)):
            with self.subTest(produces=produces, consumes=consumes):
                source = _source(True)
                source.indexer.is_candidate_source = produces
                source.indexer.uses_candidates = consumes
                module = attention.SharedCompressedDSAAttention(source, use_fused_kernels=True)
                with self.assertRaisesRegex(NotImplementedError, "candidate-pool"):
                    module.indexer._select_indices(self.query, self.key, self.weights, None, 0, None, None, None)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_zero_loss_does_not_require_an_unused_kernel(self):
        """Feature: Disabled KL.
        Description: A zero objective is not an attempted fused calculation.
        Expectation: Its scalar and student query gradient remain zero in strict mode.
        """
        query = self.query.clone().requires_grad_()
        loss = attention.shared_compressed_indexer_kl_loss(
            query, self.key, self.weights, torch.ones(1, 1, 2, 128), self.key,
            self.indices, torch.zeros(1), attention_scale=1., loss_coeff=0.,
            use_fused=True,
        )
        loss.backward()
        torch.testing.assert_close(loss, torch.tensor(0.))
        torch.testing.assert_close(query.grad, torch.zeros_like(query))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_long_key_bank_is_not_rejected_before_native_validation(self):
        """Feature: Long-K native KL dispatch.
        Description: Check a 1M shape without allocating a large CPU bank.
        Expectation: The installed interface is usable; missing interfaces still fail dispatch.
        """
        query = SimpleNamespace(device=SimpleNamespace(type="npu"), dtype=torch.bfloat16, shape=(1, 8192, 32, 128))
        key = SimpleNamespace(dtype=torch.bfloat16, shape=(1, 1048576, 128))
        indices = SimpleNamespace(shape=(1, 8192, 512))
        operators = SimpleNamespace(sparse_lightning_indexer_kl_loss_grad=object(),
                                    sparse_lightning_indexer_kl_loss_grad_metadata=object())
        with patch.object(compressed_indexer_ops, "_operators", return_value=operators):
            self.assertTrue(compressed_indexer_ops.fused_kl_available(query, key, indices))
        with patch.object(compressed_indexer_ops, "_operators", return_value=None):
            self.assertFalse(compressed_indexer_ops.fused_kl_available(query, key, indices))
