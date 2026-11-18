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
"""CPU regression tests for canonical module checks after recompute wrapping."""

import unittest
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch
from torch import nn

from hyper_parallel.core.activation_memory.wrapper import CheckpointExcludeWrapper, CheckpointWrapper
from hyper_parallel.models.deepseek_v41.adapter.validation import model_validation_spec
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import _hc_post
from hyper_parallel.tools.model_integration.integration_validator import _validate_parameter_probes


class TestEngramValidationScope(unittest.TestCase):
    """Validate explicit Engram ablations without weakening retained-layer checks."""

    def test_no_engram_crop_keeps_remaining_parameter_probes(self) -> None:
        """An empty Engram layer list must not require omitted branch parameters."""
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.sinks = nn.Parameter(torch.zeros(2))
        layer.mlp = nn.Module()
        layer.mlp.gate = nn.Linear(2, 2)
        model = nn.Module()
        model.model = nn.Module()
        model.model.layers = nn.ModuleList([layer])
        model.config = SimpleNamespace(v41_model_mode="validation_crop", v41_engram_layer_ids=[])

        spec = model_validation_spec.get_validation_spec()
        self.assertEqual(_validate_parameter_probes(model, spec), [])
        self.assertIsNone(model_validation_spec._validate_retained_engram_parameters(model, {}))
        self.assertIsNone(model_validation_spec._validate_engram_scratch_state(model, {}))

        del layer.self_attn.sinks
        findings = _validate_parameter_probes(model, spec)
        self.assertEqual([finding.owner_fqn for finding in findings], ["model.layers.*.self_attn.sinks"])

    def test_retained_engram_requires_every_declared_layer_parameter(self) -> None:
        """A valid earlier layer must not hide missing parameters in another layer."""
        model = nn.Module()
        model.model = nn.Module()
        model.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        model.config = SimpleNamespace(v41_engram_layer_ids=[0, 1])
        for layer in model.model.layers:
            layer.engram = nn.Module()
            layer.engram.q_weight = nn.Parameter(torch.ones(2))
            layer.engram.k_weight = nn.Parameter(torch.ones(2))
            layer.engram.wkv = nn.Linear(2, 2, bias=False)
        self.assertIsNone(model_validation_spec._validate_retained_engram_parameters(model, {}))

        for name in ("q_weight", "k_weight", "wkv"):
            with self.subTest(parameter=name):
                module = model.model.layers[1].engram
                value = getattr(module, name)
                delattr(module, name)
                suffix = "wkv.weight" if name == "wkv" else name
                self.assertEqual(
                    model_validation_spec._validate_retained_engram_parameters(model, {}),
                    {"missing_engram_parameters": [f"model.layers.1.engram.{suffix}"]},
                )
                setattr(module, name, value)


class TestCanonicalModuleTree(unittest.TestCase):
    """Check source module identities through canonical wrapper traversal."""

    def test_recompute_wrappers_preserve_validation(self) -> None:
        """Accept wrapped modules and retain rejection of invalid source types."""
        for wrapped in (False, True):
            for invalid in (None, "attention", "engram"):
                with self.subTest(wrapped=wrapped, invalid=invalid):
                    layer = nn.Module()
                    layer.self_attn = nn.Identity() if invalid == "attention" else nn.Linear(2, 2)
                    layer.engram = nn.Identity() if invalid == "engram" else nn.Embedding(4, 2)
                    if wrapped:
                        layer.self_attn = CheckpointExcludeWrapper(layer.self_attn)
                        layer.engram = CheckpointExcludeWrapper(layer.engram)
                        layer = CheckpointWrapper(layer)
                    model = nn.Module()
                    model.model = nn.Module()
                    model.model.layers = nn.ModuleList([layer])
                    model.config = SimpleNamespace(v41_engram_layer_ids=(0,))
                    with patch.multiple(
                        model_validation_spec,
                        DeepseekV41Attention=nn.Linear,
                        SharedCompressedDSAAttention=nn.Linear,
                        DeepseekV41Engram=nn.Embedding,
                        EngramModule=nn.Embedding,
                    ):
                        result = model_validation_spec._validate_canonical_module_tree(model, {})
                    expected = None if invalid is None else {
                        "invalid_attention_modules": ["model.layers.0.self_attn"] if invalid == "attention" else [],
                        "invalid_engram_modules": ["model.layers.0.engram"] if invalid == "engram" else [],
                    }
                    self.assertEqual(result, expected, f"expected={expected}, got={result}")

    def test_reference_mhc_post_keeps_portable_forward_backward(self) -> None:
        """An installed optional extension must not change the reference path."""
        torch.manual_seed(42)
        inputs = [
            torch.randn(1, 3, 5, requires_grad=True),
            torch.randn(1, 3, 4, 5, requires_grad=True),
            torch.randn(1, 3, 4, requires_grad=True),
            torch.randn(1, 3, 4, 4, requires_grad=True),
        ]
        reference = [value.detach().clone().requires_grad_() for value in inputs]
        states, residual, post, combine = reference
        expected = torch.einsum("bsij,bsih->bsjh", combine, residual) + post.unsqueeze(-1) * states.unsqueeze(-2)
        with (
            patch("torch.Tensor.device", new_callable=PropertyMock, return_value=SimpleNamespace(type="npu")),
            patch("hyper_parallel.components.functional.mhc_post.omni_training_custom_ops", object()),
            patch("hyper_parallel.components.functional.mhc_post._MhcPost.apply") as fused,
        ):
            actual = _hc_post(*inputs)
        fused.assert_not_called()
        torch.testing.assert_close(actual, expected)
        grad = torch.randn_like(actual)
        actual.backward(grad)
        expected.backward(grad)
        for value, oracle in zip(inputs, reference):
            torch.testing.assert_close(value.grad, oracle.grad)
