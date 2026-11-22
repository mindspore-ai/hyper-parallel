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
"""Essential contracts for Qwen3.5 GDN operators and CP injection."""
# pylint: disable=wrong-import-position

import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5TextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeAttention, Qwen3_5MoeTextConfig

from hyper_parallel.components.functional.gated_delta_net import (
    torch_chunk_gated_delta_rule,
)
from hyper_parallel.distributed._builder.forward_rewriter import (
    _ForwardRewriteRequest,
)
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.recipe_spec import (
    CP,
    INNER_WRAPPER,
    ModuleShardingSpec,
)
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed.context_parallel import wrappers as adapter_context_parallel
from hyper_parallel.models.registry import get_model_adapter
from hyper_parallel.components.modules import GatedGroupedQueryAttention
from hyper_parallel.models.module_build import prepare_model_components
from hyper_parallel.models.replacement import ModuleReplacementRequest
from hyper_parallel.trainer.config import entries_to_plan_overrides
from hyper_parallel.trainer.config.parser import parse_training_args
from tests.ut.auto_models.distributed.conftest import FakeDeviceMesh
from tests.common.mark_utils import arg_mark












class TestQwen35GdnCpContracts(unittest.TestCase):
    """Pin the public Qwen3.5 GDN adapter and eager numerical oracle."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_gated_attention_matches_native_source_and_gradients(self):
        """Feature: Real Qwen3.5 dense/MoE full attention uses the gated GQA family.
        Description: Convert loaded weights and compare B2 partial RoPE with native HF attention.
        Expectation: Outputs, input/parameter gradients and retained Norm parameters agree.
        """
        def attention(module: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                      mask: Any, **kwargs: Any) -> tuple:
            """CPU kernel oracle; projections, gate, Norm and RoPE remain production code."""
            del module
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, scale=kwargs["scaling"], enable_gqa=True)
            return output.transpose(1, 2), None

        for config_type, source_type in ((Qwen3_5TextConfig, Qwen3_5Attention),
                                         (Qwen3_5MoeTextConfig, Qwen3_5MoeAttention)):
            with self.subTest(source=source_type.__name__):
                torch.manual_seed(35)
                config = config_type.from_dict({
                    "hidden_size": 32, "num_attention_heads": 4, "num_key_value_heads": 2,
                    "head_dim": 8, "num_hidden_layers": 1, "attention_bias": True,
                })
                config._attn_implementation = "eager"
                reference = source_type(config, 0)
                with torch.no_grad():
                    reference.q_norm.weight.normal_(0, 0.1)
                    reference.k_norm.weight.normal_(0, 0.1)
                model = nn.ModuleDict({"self_attn": deepcopy(reference), "linear_attn": nn.Identity()})
                original_norm = model.self_attn.q_norm.weight
                request = ModuleReplacementRequest(
                    match=("self_attn",), target=GatedGroupedQueryAttention, module_type=source_type, exact_type=True)
                model, _ = prepare_model_components(model, {"module_replacement_requests": (request,)})
                candidate = model.self_attn
                self.assertIs(candidate.q_norm.weight, original_norm)
                self.assertIsInstance(model.linear_attn, nn.Identity)
                # Preparation declares conversion; checkpoint loading applies it later.
                state = reference.state_dict()
                converted = dict(state)
                for transform in candidate.make_transforms():
                    converted.update(transform.operations[0].convert(
                        state, transform.source_patterns, transform.target_patterns))
                candidate.load_state_dict({name: converted[name] for name in candidate.state_dict()})
                candidate.attention_interface = attention
                inputs = torch.randn(2, 7, 32, requires_grad=True)
                parallel_inputs = inputs.detach().clone().requires_grad_()
                phase = torch.randn(2, 7, 2).repeat(1, 1, 2)
                positions = phase.cos(), phase.sin()
                mask = torch.zeros(1, 1, 7, 7).masked_fill(
                    torch.ones(7, 7, dtype=torch.bool).triu(1), float("-inf"))
                expected = reference(inputs, positions, mask)[0]
                actual = candidate(parallel_inputs, positions, mask)[0]
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                gradient = torch.randn_like(expected)
                expected.backward(gradient)
                actual.backward(gradient)
                torch.testing.assert_close(parallel_inputs.grad, inputs.grad, atol=3e-6, rtol=3e-5)
                gradients = {name: parameter.grad for name, parameter in reference.named_parameters()}
                expected_gradients = dict(gradients)
                for transform in candidate.make_transforms():
                    expected_gradients.update(transform.operations[0].convert(
                        gradients, transform.source_patterns, transform.target_patterns))
                for name, parameter in candidate.named_parameters():
                    torch.testing.assert_close(parameter.grad, expected_gradients[name], atol=5e-6, rtol=5e-5)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_registration_and_wrapper_metadata(self):
        """Feature: Qwen packed metadata forwarding.

        Description: Inspect adapter registration and the decoder-only rewrite.
        Expectation: The decoder forwards invocation metadata while the GDN family owns computation.
        """
        # Resolve the nested text identity first to exercise cold discovery
        # through the single qwen3_5 provider directory.
        text_spec = get_model_adapter("qwen3_5_text")
        top_spec = get_model_adapter("Qwen3_5ForConditionalGeneration")
        self.assertIsNotNone(top_spec)
        self.assertIsNotNone(text_spec)
        self.assertEqual(top_spec.model_type, "qwen3_5")
        self.assertEqual(text_spec.model_type, "qwen3_5_text")
        self.assertIs(top_spec.context_parallel(), text_spec.context_parallel())
        self.assertIsNotNone(get_model_adapter("Qwen3_5ForCausalLM"))
        for name in (
            "packed_linear_decoder_wrapper",
        ):
            wrapper = getattr(adapter_context_parallel, name)
            self.assertEqual(wrapper._injection_meta.kind, INNER_WRAPPER)




    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_eager_chunk_partitions_have_matching_forward_and_backward(self):
        """Feature: GDN chunk partition independence.

        Description: Evaluate the same tensors with different eager chunk partitions.
        Expectation: Outputs and all input gradients match the reference partition.
        """
        torch.manual_seed(7)
        inputs = [
            torch.randn(1, 8, 2, 4, requires_grad=True),
            torch.randn(1, 8, 2, 4, requires_grad=True),
            torch.randn(1, 8, 2, 3, requires_grad=True),
            (-torch.rand(1, 8, 2)).requires_grad_(),
            torch.rand(1, 8, 2, requires_grad=True),
        ]
        copies = [tensor.detach().clone().requires_grad_() for tensor in inputs]

        output_a, state_a = torch_chunk_gated_delta_rule(
            *inputs, chunk_size=1, output_final_state=True
        )
        output_b, state_b = torch_chunk_gated_delta_rule(
            *copies, chunk_size=4, output_final_state=True
        )
        torch.testing.assert_close(output_a, output_b, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(state_a, state_b, rtol=2e-5, atol=2e-5)

        output_weight = torch.randn_like(output_a)
        state_weight = torch.randn_like(state_a)
        ((output_a * output_weight).sum() + (state_a * state_weight).sum()).backward()
        ((output_b * output_weight).sum() + (state_b * state_weight).sum()).backward()
        for actual, expected in zip(inputs, copies):
            torch.testing.assert_close(
                actual.grad,
                expected.grad,
                rtol=5e-5,
                atol=5e-5,
            )


if __name__ == "__main__":
    unittest.main()
