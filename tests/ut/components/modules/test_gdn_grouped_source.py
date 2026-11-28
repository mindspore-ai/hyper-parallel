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
"""Level0 numerical parity for grouped GDN weight conversion and gradients."""
from copy import deepcopy
from functools import partial
from itertools import product
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

from hyper_parallel.components.checkpoint.weight_conversion import revert_weight_conversion
from hyper_parallel.components.modules import GatedDeltaNet
from hyper_parallel.models.module_build import prepare_model_components
from hyper_parallel.models.replacement import ModuleReplacementRequest
from tests.common.mark_utils import arg_mark


class TestGdnGroupedSource(unittest.TestCase):
    """Check source-format export, parameter ownership and gradients against native Qwen3.5."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_gdn_grouped_source_conversion_and_gradients(self):
        """Feature: TP-compatible GDN weight layout.
        Description: Compare grouped QKV/conv with raw HF GDN, including packed resets and export.
        Expectation: Outputs, dX, parameter gradients and inverse checkpoint conversion agree.
        """
        # Qwen3.5 uses value/key head ratios 1, 2, 3 and 4 across model sizes.
        for ratio, offsets in product((1, 2, 3, 4), ((0, 9), (0, 2, 6, 9))):
            config = qwen.Qwen3_5TextConfig.from_dict({
                "hidden_size": 16,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2 * ratio,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 3,
            })
            # Some Transformers versions select the fused norm through this optional module global.
            with self.subTest(ratio=ratio, offsets=offsets), patch.object(qwen, "FusedRMSNormGated", None, create=True):
                torch.manual_seed(319)
                reference = qwen.Qwen3_5GatedDeltaNet(config, 0)
                self.assertIsInstance(reference.norm, qwen.Qwen3_5RMSNormGated)
                reference.causal_conv1d_fn = None
                reference.chunk_gated_delta_rule = partial(qwen.torch_chunk_gated_delta_rule, chunk_size=4)
                reference.in_proj_qkv.bias = nn.Parameter(torch.randn(reference.conv_dim) * .01)
                reference.conv1d.bias = nn.Parameter(torch.randn(reference.conv_dim) * .01)
                reference.A_log.requires_grad_(False)
                source = deepcopy(reference)
                original = {name: value.clone() for name, value in source.state_dict().items()}
                model = nn.ModuleDict({"attention": source})
                request = ModuleReplacementRequest(("attention",), GatedDeltaNet, {"chunk_size": 4})
                model, transforms = prepare_model_components(model, {"module_replacement_requests": (request,)})
                model.config = config
                # The checkpoint loader retains the applied mapping for source-format export.
                model._weight_conversions = transforms  # pylint: disable=protected-access
                candidate = model.attention
                with self.assertRaisesRegex(TypeError, "hidden_states"):
                    candidate()
                with self.assertRaisesRegex(TypeError, "positional arguments"):
                    candidate(torch.randn(1, 9, 16), None, None, None)
                promoted = GatedDeltaNet(module=candidate, chunk_size=4, context={"cp_size": 2},
                                         ulysses_degree=None)
                self.assertIs(promoted.in_proj_qkv.weight, candidate.in_proj_qkv.weight)
                self.assertFalse(candidate.A_log.requires_grad)
                for name, tensor in source.state_dict().items():
                    torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
                restored = revert_weight_conversion(model, model.state_dict())
                for name, tensor in original.items():
                    torch.testing.assert_close(restored[f"attention.{name}"], tensor, rtol=0, atol=0)
                inputs = (torch.randn(1, 9, 16) * .1).requires_grad_()
                copied = inputs.detach().clone().requires_grad_()
                expected = torch.cat([reference(inputs[:, start:end])
                                      for start, end in zip(offsets, offsets[1:])], dim=1)
                actual = (candidate(copied, None, None, cu_seqlens=offsets) if len(offsets) == 2 else
                          candidate(hidden_states=copied, cu_seqlens=offsets))
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                gradient = torch.randn_like(actual)
                expected.backward(gradient)
                actual.backward(gradient)
                torch.testing.assert_close(copied.grad, inputs.grad, rtol=3e-5, atol=3e-6)
                gradients = {name: parameter.grad for name, parameter in reference.named_parameters()}
                expected_gradients = dict(gradients)
                for transform in candidate.make_transforms():
                    expected_gradients.update(transform.operations[0].convert(
                        gradients, transform.source_patterns, transform.target_patterns))
                for name, parameter in candidate.named_parameters():
                    torch.testing.assert_close(parameter.grad, expected_gradients[name], rtol=5e-5, atol=5e-6)
