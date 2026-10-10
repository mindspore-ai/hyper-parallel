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
"""CPU tests for DeepSeek-V4.1 model-owned shared-expert semantics."""
# pylint: disable=wrong-import-position

import copy
import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

import torch
from torch.nn import functional

try:
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4MLP
except ImportError as exc:
    raise unittest.SkipTest(f"DeepSeek-V4 Transformers dependency unavailable: {exc}") from exc

from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import (
    deepseek_v41_ep_compute_fn,
)
from hyper_parallel.models.deepseek_v41.adapter.validation.native_module_parity import HyperParallelMoE
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41Model,
    bind_shared_expert_forward,
)


def _config() -> DeepseekV4Config:
    """Build a tiny real backbone without Engram assets or compressed attention."""
    config = DeepseekV4Config(  # pylint: disable=unexpected-keyword-arg
        vocab_size=32, hidden_size=16, moe_intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=1, head_dim=8, q_lora_rank=8,
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
        layer_types=["sliding_attention"], mlp_layer_types=["moe"],
        swiglu_limit=1.5, sliding_window=4, o_groups=2, o_lora_rank=4,
        index_n_heads=2, index_head_dim=4, index_topk=2, partial_rotary_factor=0.5,
    )
    config.v41_compress_ratios = [0]
    config.v41_kv_source_layer_ids = []
    config.v41_index_source_layer_ids = []
    config.v41_engram_assets_path = "unused-engram-assets.json"
    config.v41_vision_hidden_size = 16
    config.v41_vision_intermediate_size = 24
    config.v41_vision_patch_size = 2
    config.v41_vision_num_attention_heads = 2
    config.v41_vision_rope_theta = 10000.0
    config.v41_vision_num_hidden_layers = 1
    config.v41_vision_downsample_ratio = 1
    return config


def _native_shared_oracle(module: torch.nn.Module, states: torch.Tensor) -> torch.Tensor:
    """Express released Expert.forward without using the bound candidate forward."""
    # Keep the oracle independent of the HF activation and candidate binder.
    gate = functional.linear(  # pylint: disable=not-callable
        states, module.gate_proj.weight, module.gate_proj.bias,
    ).float()
    up = functional.linear(  # pylint: disable=not-callable
        states, module.up_proj.weight, module.up_proj.bias,
    ).float()
    if module.limit > 0:
        up = torch.clamp(up, min=-module.limit, max=module.limit)
        gate = torch.clamp(gate, max=module.limit)
    intermediate = (functional.silu(gate) * up).to(states.dtype)
    return functional.linear(  # pylint: disable=not-callable
        intermediate, module.down_proj.weight, module.down_proj.bias,
    )


class TestV41SharedExpert(unittest.TestCase):
    """Check activation precision, autograd, state compatibility, and entry points."""

    def test_forward_and_backward_match_native_formula(self) -> None:
        """FP32/BF16, positive/nonpositive limits, and empty inputs retain parity."""
        for dtype, limit, shape in product(
                (torch.float32, torch.bfloat16), (-1.0, 0.0, 1.5, 10.0),
                ((3, 16), (2, 3, 16), (0, 16))):
            with self.subTest(dtype=dtype, limit=limit, shape=shape):
                torch.manual_seed(82)
                config = _config()
                config.swiglu_limit = limit
                shared = DeepseekV4MLP(config).to(dtype=dtype)
                for parameter in shared.parameters():
                    torch.nn.init.normal_(parameter)
                reference = copy.deepcopy(shared)
                bind_shared_expert_forward(shared)
                states = torch.randn(shape, dtype=dtype, requires_grad=True)
                expected_states = states.detach().clone().requires_grad_()
                actual = shared(states)
                expected = _native_shared_oracle(reference, expected_states)
                upstream = torch.randn_like(actual)
                (actual * upstream).sum().backward()
                (expected * upstream).sum().backward()
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(states.grad, expected_states.grad, rtol=0, atol=0)
                for parameter, expected_parameter in zip(shared.parameters(), reference.parameters()):
                    torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=0, atol=0)

    def test_bfloat16_rounds_after_swiglu(self) -> None:
        """A scalar witness distinguishes native FP32 activation from the HF path."""
        config = SimpleNamespace(
            hidden_size=1, intermediate_size=1, mlp_bias=False, hidden_act="silu", swiglu_limit=10.0,
        )
        shared = DeepseekV4MLP(config).to(dtype=torch.bfloat16)
        with torch.no_grad():
            shared.gate_proj.weight.fill_(2.421875)
            shared.up_proj.weight.fill_(-4.40625)
            shared.down_proj.weight.fill_(1.0)
        states = torch.ones(1, 1, dtype=torch.bfloat16)
        old_result = shared(states)
        bind_shared_expert_forward(shared)
        self.assertEqual(shared(states).item(), -9.8125)
        self.assertNotEqual(shared(states).item(), old_result.item())

    def test_down_projection_receives_input_dtype(self) -> None:
        """Do not promote projection weights or pass FP32 activations to BF16 linear."""
        shared = DeepseekV4MLP(_config()).to(dtype=torch.bfloat16)
        bind_shared_expert_forward(shared)
        seen = []
        handle = shared.down_proj.register_forward_pre_hook(lambda _module, args: seen.append(args[0].dtype))
        try:
            shared(torch.ones(2, 16, dtype=torch.bfloat16))
        finally:
            handle.remove()
        self.assertEqual(seen, [torch.bfloat16])
        self.assertTrue(all(parameter.dtype == torch.bfloat16 for parameter in shared.parameters()))

    def test_binding_preserves_parameters_state_and_hooks(self) -> None:
        """Binding twice retains module identity, state keys, modes, and hook calls."""
        shared = DeepseekV4MLP(_config()).eval()
        parameters = dict(shared.named_parameters())
        state = copy.deepcopy(shared.state_dict())
        calls = []
        shared.register_forward_pre_hook(lambda _module, _args: calls.append("pre"))
        shared.register_forward_hook(lambda _module, _args, _output: calls.append("post"))
        bind_shared_expert_forward(shared)
        bind_shared_expert_forward(shared)
        self.assertIs(type(shared), DeepseekV4MLP)
        self.assertFalse(shared.training)
        self.assertEqual(list(shared.named_buffers()), [])
        self.assertEqual(list(shared.state_dict()), list(state))
        for name, parameter in shared.named_parameters():
            self.assertIs(parameter, parameters[name])
            torch.testing.assert_close(parameter, state[name], rtol=0, atol=0)
        states = torch.randn(2, 16)
        torch.testing.assert_close(shared(states), _native_shared_oracle(shared, states), rtol=0, atol=0)
        self.assertEqual(calls, ["pre", "post"])
        copied = copy.deepcopy(shared)
        self.assertIs(copied.forward.__self__, copied)
        torch.testing.assert_close(copied(states), shared(states), rtol=0, atol=0)

    def test_meta_materialization_and_checkpoint_load(self) -> None:
        """Binding adds no state and survives meta construction plus strict loading."""
        reference = DeepseekV4MLP(_config())
        with torch.device("meta"):
            shared = DeepseekV4MLP(_config())
            bind_shared_expert_forward(shared)
        shared.to_empty(device="cpu")
        shared.load_state_dict(reference.state_dict(), strict=True)
        states = torch.randn(2, 16)
        torch.testing.assert_close(shared(states), _native_shared_oracle(reference, states), rtol=0, atol=0)

    def test_model_constructor_binds_text_and_multimodal_shared_experts(self) -> None:
        """The real backbone installs the math before any EP recipe runs."""
        for vision_enabled in (False, True):
            with self.subTest(vision_enabled=vision_enabled):
                config = _config()
                config.v41_vision_enabled = vision_enabled
                config.swiglu_limit = -1.0
                with patch(
                        "hyper_parallel.models.deepseek_v41.modeling_deepseek_v41.Path.open",
                        mock_open(read_data='{"layer_ids": []}')):
                    model = DeepseekV41Model(config).to(dtype=torch.bfloat16)
                for layer in model.layers:
                    shared = layer.mlp.shared_experts
                    self.assertIs(shared.forward.__self__, shared)
                    states = torch.randn(2, 3, config.hidden_size, dtype=torch.bfloat16)
                    torch.testing.assert_close(shared(states), _native_shared_oracle(shared, states), rtol=0, atol=0)

    def test_ep_factories_keep_bound_shared_branch(self) -> None:
        """Both EP factories execute the same real shared branch and its gradients."""
        for grouped in (False, True):
            with self.subTest(grouped=grouped):
                config = _config()
                config.swiglu_limit = -1.0
                module = HyperParallelMoE(config).to(dtype=torch.bfloat16)
                module.is_hash = False
                reference = copy.deepcopy(module.shared_experts)
                shared_forward = module.shared_experts.forward
                ep_mesh = Mock()
                ep_mesh.__getitem__ = Mock(return_value=SimpleNamespace(size=lambda: 2))
                compute = deepseek_v41_ep_compute_fn(
                    module=module, mesh=None, tp_mesh=None, cp_mesh=None,
                    ep_mesh=ep_mesh, use_grouped_gemm=grouped,
                )
                self.assertIs(module.shared_experts.forward, shared_forward)
                states = torch.randn(2, 3, 16, dtype=torch.bfloat16, requires_grad=True)
                expected_states = states.detach().clone().requires_grad_()
                routed = torch.zeros_like(states)
                with patch(
                        "hyper_parallel.models.deepseek_v41.adapter.distributed."
                        "moe_engram_expert_parallel.ep_routed_forward", return_value=routed):
                    actual = compute(module, states)
                expected = _native_shared_oracle(reference, expected_states)
                actual.float().sum().backward()
                expected.float().sum().backward()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(states.grad, expected_states.grad, rtol=0, atol=0)
                for parameter, expected_parameter in zip(module.shared_experts.parameters(), reference.parameters()):
                    torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=0, atol=0)
