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

from hyper_parallel.components.modules import TritonGDN
from hyper_parallel.distributed.context_parallel import wrappers as cp_wrappers


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

    def test_triton_replacement_uses_direct_kernel_and_preserves_weights(self):
        """Bind the complete Triton implementation without altering model state."""
        original = FakeGatedDeltaNet()
        triton_rule = Mock(side_effect=_original_rule)
        backend = SimpleNamespace(chunk_gated_delta_rule=triton_rule)
        with patch("hyper_parallel.components.modules.gdn_triton.import_module", return_value=backend) as loader:
            replacement = TritonGDN(module=original)
        loader.assert_called_once_with("hyper_parallel.components.functional.gated_delta_net")
        self.assertIs(replacement.chunk_gated_delta_rule, triton_rule)
        self.assertIs(original.chunk_gated_delta_rule, _original_rule)
        self.assertIs(replacement.weight, original.weight)
        self.assertEqual(list(replacement.state_dict()), list(original.state_dict()))
        inputs = torch.ones(1, 4, 2, 3, requires_grad=True)
        replacement(inputs).sum().backward()
        triton_rule.assert_called_once()
        torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 2))

    def test_triton_replacement_rejects_missing_primitive(self):
        """Reject an incompatible source before loading optional Triton modules."""
        with patch("hyper_parallel.components.modules.gdn_triton.import_module") as loader:
            with self.assertRaisesRegex(TypeError, "requires callable"):
                TritonGDN(module=nn.Linear(2, 2))
        loader.assert_not_called()

    def test_cp_wraps_direct_triton_primitive(self):
        """CP calls the selected Triton primitive without changing its binding."""
        triton_rule = Mock(side_effect=_original_rule)
        backend = SimpleNamespace(chunk_gated_delta_rule=triton_rule)
        with patch("hyper_parallel.components.modules.gdn_triton.import_module", return_value=backend):
            replacement = TritonGDN(module=FakeGatedDeltaNet())
        mesh = SimpleNamespace(size=lambda: 2)
        cp_wrappers.gdn_ulysses_cp_wrapper(replacement, None, None, mesh, None)
        inputs = torch.ones(1, 4, 2, 3, requires_grad=True)
        with patch.object(cp_wrappers, "_gdn_cp_causal_conv1d", side_effect=lambda fn, mesh, x, *a, **kw: x), \
                patch.object(cp_wrappers, "_gdn_rule_cp_to_hp", side_effect=lambda *args: args[:5]), \
                patch.object(cp_wrappers, "_gdn_rule_hp_to_cp", side_effect=lambda x, *a, **kw: x):
            replacement(inputs).sum().backward()
        triton_rule.assert_called_once()
        self.assertIs(replacement.chunk_gated_delta_rule, triton_rule)
        torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 2))
