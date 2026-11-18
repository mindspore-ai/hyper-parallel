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
"""Hardware-independent tests for GDN fused primitive replacement."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from hyper_parallel.components.modules import TritonGDN
from hyper_parallel.components.ops import ops
from hyper_parallel.components.ops.catalog import builtin_registry
from hyper_parallel.distributed.context_parallel import wrappers as cp_wrappers
from hyper_parallel.distributed.context_parallel import gated_delta_net as cp_gdn


def _original_rule(query, key, value, **kwargs):
    """Provide a differentiable primitive without accelerator dependencies."""
    del key, value, kwargs
    return query * 2, None


def _conv(x, **kwargs):
    """Stand in for causal convolution without changing tensor layouts."""
    del kwargs
    return x


class FakeGatedDeltaNet(nn.Module):
    """Use the same instance primitive boundary as Transformers GDN."""

    def __init__(self) -> None:
        """Create a parameter and the two replaceable primitive callables."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.num_v_heads = 2
        self.chunk_gated_delta_rule = _original_rule
        self.causal_conv1d_fn = _conv

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the chunk rule after a primitive convolution."""
        hidden_states = self.causal_conv1d_fn(x=hidden_states) * self.weight
        gate = torch.zeros_like(hidden_states[..., 0])
        output, _ = self.chunk_gated_delta_rule(hidden_states, hidden_states, hidden_states, g=gate, beta=gate)
        return output


class TestGatedDeltaNet(unittest.TestCase):
    """Check replacement identity, dispatch and CP composition."""

    def test_replacement_calls_gdn_and_preserves_weights(self):
        """Select the zero-state operator at runtime without changing model state."""
        original = FakeGatedDeltaNet()
        replacement = TritonGDN(module=original)
        self.assertIs(original.chunk_gated_delta_rule, _original_rule)
        self.assertIs(replacement.weight, original.weight)
        self.assertEqual(list(replacement.state_dict()), list(original.state_dict()))
        inputs = torch.ones(1, 4, 2, 3, requires_grad=True)
        with patch("hyper_parallel.components.modules.gdn_triton.gdn", side_effect=lambda q, *a, **kw: q * 2) as rule:
            replacement(inputs).sum().backward()
        rule.assert_called_once()
        torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 2))

    def test_triton_replacement_rejects_missing_primitive(self):
        """Reject an incompatible source before loading optional Triton modules."""
        with self.assertRaisesRegex(TypeError, "requires callable"):
            TritonGDN(module=nn.Linear(2, 2))

    def test_hf_forward_selects_both_local_operator_families(self) -> None:
        """Qwen's global-call forward uses the configured rule and convolution on the replacement only."""
        config = Qwen3_5TextConfig(
            hidden_size=16, num_hidden_layers=1, layer_types=["linear_attention"],
            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
            linear_value_head_dim=8, linear_conv_kernel_dim=3,
        )
        original = Qwen3_5GatedDeltaNet(config, 0)
        replacement = TritonGDN(module=original)
        runtime = ops.configure(
            registry=builtin_registry, selections={"gdn": "native", "causal_conv1d": "native"}, target_device="cpu",
        )
        inputs = torch.randn(2, 5, 16, requires_grad=True)
        with runtime.activate():
            baseline = original(inputs)
            self.assertEqual(runtime.used_backends(), ())
            output = replacement(inputs)
        torch.testing.assert_close(output, baseline, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(
            torch.autograd.grad(output.sum(), inputs)[0], torch.autograd.grad(baseline.sum(), inputs)[0],
            atol=2e-6, rtol=2e-5,
        )
        self.assertIs(replacement.in_proj_qkv.weight, original.in_proj_qkv.weight)
        self.assertEqual(set(runtime.used_backends()), {("causal_conv1d", "standard", "native"),
                                                       ("gdn", "zero_state", "native")})

    def test_cp_wraps_direct_triton_primitive(self):
        """CP calls the selected Triton primitive without changing its binding."""
        rule = Mock(side_effect=lambda q, *args, **kwargs: q * 2)
        replacement = TritonGDN(module=FakeGatedDeltaNet())
        configured_rule = replacement.chunk_gated_delta_rule
        mesh = SimpleNamespace(size=lambda: 2)
        cp_wrappers.gdn_ulysses_cp_wrapper(replacement, None, None, mesh, None)
        inputs = torch.ones(1, 4, 2, 3, requires_grad=True)
        with patch.object(cp_gdn, "_gdn_cp_causal_conv1d", side_effect=lambda fn, mesh, x, *a, **kw: x), \
                patch.object(cp_gdn, "_gdn_rule_cp_to_hp", side_effect=lambda *args: args[:5]), \
                patch.object(cp_gdn, "_gdn_rule_hp_to_cp", side_effect=lambda x, *a, **kw: x), \
                patch("hyper_parallel.components.modules.gdn_triton.gdn", rule):
            replacement(inputs).sum().backward()
        rule.assert_called_once()
        self.assertIs(replacement.chunk_gated_delta_rule, configured_rule)
        torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 2))
