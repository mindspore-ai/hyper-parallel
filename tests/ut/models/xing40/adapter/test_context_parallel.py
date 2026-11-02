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
"""CPU contracts for explicit MLA callbacks and Xing40 rotary precision."""
from typing import Any
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers.models.deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave
from hyper_parallel.distributed.context_parallel import mla_context_parallel as mla
from hyper_parallel.models.xing40.adapter.distributed.context_parallel import xing40_rotary
from tests.common.mla_cp_utils import make_mla


class _FalseCallable:
    """A callable's truth value must not select an unrelated default."""
    def __bool__(self) -> bool:
        """Return false while retaining an explicitly supplied callback."""
        return False

    def __call__(self, value: torch.Tensor, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        """Return the input unchanged for callback selection tests."""
        return value


class TestXing40CPCallbacks(unittest.TestCase):
    """Cover new callback selection and the family-specific rotary contract."""

    def test_optional_callbacks_are_selected_by_none(self):
        """
        Feature: Optional MLA local compute callbacks.
        Description: Check callback selection and execute CPU attention with explicit callbacks.
        Expectation: Q and K use rotary; attention outputs and gradients match the unwrapped module.
        """
        plan = mla.MLACPPlan(mla.MLADimensions(4, 5, 6, 3, 2, 4), 1)
        default = mla.MLAContextParallel(plan, None)
        self.assertIs(default.rotary_fn, mla._apply_mla_rope)
        self.assertIs(default.attention_interface, mla.npu_fusion_attention_forward)
        callback = _FalseCallable()
        custom = mla.MLAContextParallel(plan, None, rotary_fn=callback, attention_interface=callback)
        self.assertIs(custom.rotary_fn, callback)
        self.assertIs(custom.attention_interface, callback)
        with self.assertRaisesRegex(TypeError, 'rotary_fn must be callable'):
            mla.MLAContextParallel(plan, None, rotary_fn=1)
        with self.assertRaisesRegex(TypeError, 'attention_interface must be callable'):
            mla.MLAContextParallel(plan, None, attention_interface=1)

        module = make_mla(dtype=torch.float64, interleaved=True, bias=False)
        reference_input = torch.randn(1, 3, 12, dtype=torch.float64, requires_grad=True)
        local_input = reference_input.detach().clone().requires_grad_()
        rotary_heads, attention_options = [], []

        def rotary(value, positions, *, interleaved):
            self.assertIsNone(positions)
            self.assertTrue(interleaved)
            rotary_heads.append(value.shape[2])
            return value

        def attention(_module, query, key, value, _mask, **options):
            attention_options.append(options)
            result = F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=options["scaling"])
            return result.transpose(1, 2), None

        module.attention_interface = attention
        expected = module(reference_input)[0]
        attention_options.clear()
        runtime = mla.MLAContextParallel(plan, None, rotary_fn=rotary, attention_interface=attention)
        validate_state = mla._validate_state

        def validate_on_cpu(target, cp_plan, state):
            # Keep the production checks before its NPU-only activation guard.
            try:
                validate_state(target, cp_plan, state)
            except ValueError as error:
                if str(error) != "NPU MLA CP requires NPU BF16/FP16 activations":
                    raise

        with patch.object(mla, "_validate_state", side_effect=validate_on_cpu):
            output = runtime(module, *module.project_latent_inputs(local_input))[0]
        actual = module.o_proj(output.reshape(1, 3, -1))
        self.assertEqual(rotary_heads, [1, 4])
        self.assertEqual(len(attention_options), 1)
        self.assertEqual(attention_options[0]["scaling"], module.scaling)
        self.assertIs(attention_options[0]["is_causal"], True)
        self.assertIsNone(attention_options[0]["actual_seq_qlen"])
        self.assertIsNone(attention_options[0]["actual_seq_kvlen"])
        torch.testing.assert_close(actual, expected)
        upstream = torch.linspace(-1, 1, actual.numel(), dtype=actual.dtype).reshape_as(actual)
        parameters = tuple(module.parameters())
        actual_gradients = torch.autograd.grad(actual, (local_input, *parameters), upstream)
        expected_gradients = torch.autograd.grad(expected, (reference_input, *parameters), upstream)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_xing40_rotary_matches_family_reference_and_gradient(self):
        """
        Feature: Xing40 interleaved rotary dtype contract.
        Description: Compare explicit FP32 family rotation and its input gradient.
        Expectation: The callback rounds once to the input dtype and differentiates correctly.
        """
        value = torch.linspace(-2, 2, 48).reshape(2, 3, 1, 8).to(torch.bfloat16).requires_grad_()
        reference = value.detach().clone().requires_grad_()
        angle = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8) / 19
        positions = angle.cos(), angle.sin()
        output = xing40_rotary(value, positions, interleaved=True)
        expected, _ = apply_rotary_pos_emb_interleave(reference.float(), reference.float(), *positions, unsqueeze_dim=2)
        expected = expected.to(reference.dtype)
        self.assertEqual(output.dtype, torch.bfloat16)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.float().square().sum().backward()
        expected.float().square().sum().backward()
        torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'position embeddings'):
            xing40_rotary(value.detach(), None, interleaved=True)
        with self.assertRaisesRegex(ValueError, 'interleaved'):
            xing40_rotary(value.detach(), positions)
