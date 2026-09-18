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

from hyper_parallel.distributed.context_parallel import wrappers as cp_wrappers
from hyper_parallel.components.modules.gdn_ascendc import replace_gdn_chunk_rule
from hyper_parallel.components.functional import gated_delta_rule


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

    def test_preserves_weights_and_forward(self):
        """Replacement changes only the selected instance primitive."""
        original = FakeGatedDeltaNet()
        fused = Mock(side_effect=_original_rule)
        with patch("hyper_parallel.components.modules.gdn_ascendc.chunk_gated_delta_rule", fused):
            replacement = replace_gdn_chunk_rule(module=original)
        self.assertIs(type(replacement), type(original))
        self.assertIs(replacement.weight, original.weight)
        self.assertEqual(list(replacement.state_dict()), list(original.state_dict()))
        self.assertIs(replacement.forward.__func__, original.forward.__func__)
        self.assertIs(original.chunk_gated_delta_rule, _original_rule)
        inputs = torch.ones(1, 4, 2, 3, requires_grad=True)
        replacement(inputs).sum().backward()
        torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 2))
        fused.assert_called_once()

    def test_rejects_missing_primitive(self):
        """Unsupported modules cannot be silently relabeled as fused GDN."""
        with self.assertRaisesRegex(TypeError, "requires callable"):
            replace_gdn_chunk_rule(module=nn.Linear(2, 2))

    def test_cp_wraps_replaced_primitive(self):
        """CP retains the fused callable and restores it after forward."""
        fused = Mock(side_effect=_original_rule)
        with patch("hyper_parallel.components.modules.gdn_ascendc.chunk_gated_delta_rule", fused):
            replacement = replace_gdn_chunk_rule(module=FakeGatedDeltaNet())
        mesh = SimpleNamespace(size=lambda: 2)
        cp_wrappers.gdn_ulysses_cp_wrapper(replacement, None, None, mesh, None)
        inputs = torch.ones(1, 4, 2, 3, requires_grad=True)
        with patch.object(cp_wrappers, "_gdn_cp_causal_conv1d", side_effect=lambda fn, mesh, x, *a, **kw: x), \
                patch.object(cp_wrappers, "_gdn_rule_cp_to_hp", side_effect=lambda *args: args[:5]), \
                patch.object(cp_wrappers, "_gdn_rule_hp_to_cp", side_effect=lambda x, *a, **kw: x):
            replacement(inputs).sum().backward()
        fused.assert_called_once()
        self.assertIs(replacement.chunk_gated_delta_rule, fused)
        torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 2))


class TestGatedDeltaRuleAdapter(unittest.TestCase):
    """Validate optional dependency errors and supported training options."""

    @staticmethod
    def _inputs():
        """Provide NPU metadata without allocating device tensors."""
        query = Mock(device=SimpleNamespace(type="npu"), dtype=torch.bfloat16, ndim=4, shape=(1, 64, 2, 128))
        gate = Mock(device=query.device, shape=query.shape[:3])
        return query, query, query, gate, gate

    def test_dispatches_ascendc_without_layout_change(self):
        """The adapter delegates normalization and preserves sequence-first layout."""
        inputs = self._inputs()
        kernel = Mock(return_value=("output", None))
        backend = SimpleNamespace(chunk_gated_delta_rule=kernel)
        with patch.object(gated_delta_rule, "import_module", return_value=backend):
            result = gated_delta_rule.chunk_gated_delta_rule(*inputs, use_qk_l2norm_in_kernel=True)
        self.assertEqual(result, ("output", None))
        kernel.assert_called_once_with(*inputs, scale=None, use_qk_l2norm_in_kernel=True)

    def test_missing_dependency_is_actionable(self):
        """Missing AscendC does not select the forward-only torch_npu operator."""
        with patch.object(gated_delta_rule, "import_module", side_effect=ModuleNotFoundError(name="fla_npu")):
            with self.assertRaisesRegex(RuntimeError, "requires fla_npu"):
                gated_delta_rule.chunk_gated_delta_rule(*self._inputs())

    def test_incompatible_registration_does_not_fallback(self):
        """An incompatible native extension must not silently select another backend."""
        kernel = Mock(side_effect=RuntimeError("missing torch.ops.npu registrations"))
        with patch.object(gated_delta_rule, "import_module",
                          return_value=SimpleNamespace(chunk_gated_delta_rule=kernel)) as loader:
            with self.assertRaisesRegex(RuntimeError, "missing torch.ops.npu"):
                gated_delta_rule.chunk_gated_delta_rule(*self._inputs())
        loader.assert_called_once_with("hyper_parallel.components.functional._ascendc_gdn")
        kernel.assert_called_once()

    def test_missing_runtime_dependency_is_actionable(self):
        """Report an extension missing during lazy operator registration."""
        kernel = Mock(side_effect=ModuleNotFoundError(name="fla_npu"))
        with patch.object(gated_delta_rule, "import_module",
                          return_value=SimpleNamespace(chunk_gated_delta_rule=kernel)):
            with self.assertRaisesRegex(RuntimeError, "requires fla_npu"):
                gated_delta_rule.chunk_gated_delta_rule(*self._inputs())
        kernel.assert_called_once()

    def test_rejects_unsupported_state_and_varlen(self):
        """Do not silently discard recurrent state gradients or sequence boundaries."""
        options = ({"initial_state": torch.ones(1)}, {"output_final_state": True},
                   {"cu_seqlens": torch.tensor([0, 64])}, {"chunk_size": 32})
        for kwargs in options:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                gated_delta_rule.chunk_gated_delta_rule(*self._inputs(), **kwargs)

    def test_cpu_fails_before_import(self):
        """CPU use must not initialize an optional NPU backend."""
        query = torch.zeros(1, 64, 2, 128)
        gate = query[..., 0]
        with patch.object(gated_delta_rule, "import_module") as loader:
            with self.assertRaisesRegex(ValueError, "requires NPU"):
                gated_delta_rule.chunk_gated_delta_rule(query, query, query, gate, gate)
        loader.assert_not_called()
