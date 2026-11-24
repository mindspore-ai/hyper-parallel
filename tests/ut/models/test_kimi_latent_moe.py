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
"""Independent token-wise gradient oracle for the trainable latent MoE family."""

import unittest
from types import FunctionType, SimpleNamespace

import torch
from transformers import Qwen3Config
from transformers.masking_utils import create_causal_mask

from hyper_parallel.models.kimi_k3.components.latent_moe import KimiLatentMoE, KimiLatentMoEModule
from hyper_parallel.models.kimi_k3.source_compat import prepare_training_mask_protocol


class TestKimiLatentMoE(unittest.TestCase):
    """Check dispatch/reduction gradients and the independent optimizer matrix contract."""

    def test_token_oracle_and_source_identities(self) -> None:
        """Feature: Differentiable latent experts.
        Description: Compare sorted dispatch against an independent per-token expert loop in both compute dtypes.
        Expectation: Outputs and all used gradients agree, and source adoption preserves parameter objects.
        """
        torch.set_num_threads(2)
        config = SimpleNamespace(
            hidden_size=8,
            moe_intermediate_size=12,
            hidden_act="situ",
            activation_situ_beta=1.5,
            activation_situ_linear_beta=2.0,
            num_experts=4,
            num_experts_per_token=2,
            num_expert_group=2,
            topk_group=1,
            num_shared_experts=1,
            routed_expert_hidden_size=4,
            latent_moe_use_norm=True,
            rms_norm_eps=1e-6,
            moe_router_activation_func="sigmoid",
            moe_renormalize=True,
            routed_scaling_factor=1.0,
        )
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(19)
                model = KimiLatentMoE(config=config).to(dtype)
                reference = KimiLatentMoEModule(config=config).to(dtype)
                reference.load_state_dict(model.state_dict())
                hidden = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
                other = hidden.detach().clone().requires_grad_()
                ids, weights = reference.gate(other)
                latent = reference.routed_expert_down_proj(other.flatten(0, 1))
                rows = []
                for token, choices in enumerate(ids.tolist()):
                    values = torch.stack([reference.experts[index](latent[token : token + 1])[0] for index in choices])
                    rows.append((values.float() * weights[token, :, None]).sum(0).to(dtype))
                routed = reference.routed_expert_up_proj(reference.routed_expert_norm(torch.stack(rows)))
                expected = routed.view_as(other) + reference.shared_experts(other)
                actual = model(hidden)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                actual.float().square().mean().backward()
                expected.float().square().mean().backward()
                tolerance = 0.02 if dtype == torch.bfloat16 else 1e-6
                torch.testing.assert_close(hidden.grad, other.grad, atol=tolerance, rtol=tolerance)
                for name, param in model.named_parameters():
                    grad = reference.get_parameter(name).grad
                    if grad is None:
                        self.assertIsNone(param.grad)
                    else:
                        torch.testing.assert_close(param.grad, grad, atol=tolerance, rtol=tolerance)
                for expert in reference.experts:
                    expert.config = config
                reference.shared_experts.config = config
                adopted = KimiLatentMoE(module=reference)
                for name, param in reference.named_parameters():
                    self.assertIs(adopted.get_parameter(name), param)
                self.assertEqual(set(adopted.state_dict()), set(model.state_dict()))


    def test_real_hf_training_masks(self) -> None:
        """Feature: Actual HF causal/padding/packed mask semantics.
        Description: Translate the pinned source call into the real installed HF helper on three visibility cases.
        Expectation: Standard causal, padded and packed-position masks exactly match direct native calls.
        """
        namespace = {"create_causal_mask": create_causal_mask}

        def constructor(_instance):
            pass

        for name in ("KimiLinearModel", "KimiLinearForCausalLM", "Root"):
            namespace[name] = type(name, (), {"__init__": FunctionType(constructor.__code__, namespace)})
        prepare_training_mask_protocol(namespace["Root"])
        bound = getattr(SimpleNamespace(**namespace), "create_causal_mask")
        config = Qwen3Config.from_dict(
            {"hidden_size": 8, "num_attention_heads": 2, "num_hidden_layers": 1, "intermediate_size": 16}
        )
        config._attn_implementation = "eager"
        hidden = torch.randn(1, 4, 8)
        for padding, positions in (
            (None, torch.tensor([[0, 1, 2, 3]])),
            (torch.tensor([[0, 1, 1, 1]]), torch.tensor([[0, 1, 2, 3]])),
            (None, torch.tensor([[0, 1, 0, 1]])),
        ):
            options = {"config": config, "attention_mask": padding, "position_ids": positions, "past_key_values": None}
            expected = create_causal_mask(inputs_embeds=hidden, **options)
            actual = bound(input_embeds=hidden, cache_position=torch.arange(4), **options)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
