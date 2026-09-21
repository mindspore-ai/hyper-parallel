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
"""CPU contracts for unclamped DeepSeek-V4.1 and its MegaMoe block adapter."""

import copy
import json
from pathlib import Path
import tempfile
from types import MethodType
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers.initialization import no_init_weights
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4SparseMoeBlock

from examples.training_demo.cropped_deepseek_v41 import build_deepseek_v41_validation_config
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore.examples.mega_moe.deepseek_v41_accuracy import fp32_accuracy
from hyper_parallel.core.multicore.examples.mega_moe.deepseek_v41_oracle import evaluate_fp32_moe
from hyper_parallel.models.deepseek_v41.adapter.activation import configure_deepseek_v41_swiglu
from hyper_parallel.models.deepseek_v41.adapter.megamoe import DeepseekV41MegaMoe
from hyper_parallel.models.deepseek_v41.configuration import validate_swiglu_limit
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41CroppedForCausalLM,
    DeepseekV41TopKRouter,
    _v41_sparse_moe_forward,
)
from tests.common.mark_utils import arg_mark
from tests.ut.auto_models.models.deepseek_v41.test_deepseek_v41_crop import _tiny_config, _write_engram_assets


def _make_source(limit=0.0, use_v41_router=True):
    """Build the actual HF expert modules and V4.1 multimodal learned router."""
    config = DeepseekV4Config(  # pylint: disable=unexpected-keyword-arg
        hidden_size=8, moe_intermediate_size=4, n_routed_experts=4,
        n_shared_experts=1, num_experts_per_tok=2, num_hidden_layers=1,
        mlp_layer_types=["moe"], swiglu_limit=limit, scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.7,
    )
    config.v41_vision_enabled = True
    module = DeepseekV4SparseMoeBlock(config, 0)
    if use_v41_router:
        module.gate = DeepseekV41TopKRouter(config)
        module.forward = MethodType(_v41_sparse_moe_forward, module)
    generator = torch.Generator().manual_seed(19)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.2, generator=generator)
    return module


def _cpu_experts(module, hidden, indices, weights):
    """Replace only native execution with a differentiable full-expert oracle."""
    flat = hidden.reshape(-1, hidden.shape[-1])
    output = torch.zeros_like(flat)
    for expert_id in range(module.num_experts):
        rows, slots = torch.where(indices == expert_id)
        gate, up = (flat[rows] @ module.gate_up_weight[expert_id]).chunk(2, dim=-1)
        values = (F.silu(gate) * up) @ module.down_weight[expert_id]
        output = output.index_add(0, rows, values * weights[rows, slots, None])
    return output.reshape_as(hidden)


class TestDeepseekV41SwiGLU(unittest.TestCase):
    """A zero validation limit disables clipping, with positive limits unchanged."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_limits_are_rejected(self):
        """
        Feature: DeepSeek-V4.1 MegaMoe validation.
        Description: Reject invalid values instead of silently selecting a different activation.
        Expectation: Invalid limits raise ValueError.
        """
        for value in (-1, float("inf"), float("nan"), True, "0", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_swiglu_limit(value)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_zero_limit_routed_outputs_and_gradients_are_unclamped(self):
        """
        Feature: DeepSeek-V4.1 MegaMoe validation.
        Description: Large gate/up values preserve the ordinary SwiGLU derivative at limit zero.
        Expectation: Large-value outputs and derivatives equal ordinary SwiGLU.
        """
        source = _make_source()
        inputs = torch.tensor([[20.0, -20.0, 11.0, -11.0, -30.0, 30.0, 12.0, -12.0]], requires_grad=True)
        # HF 5.13 treats zero as a clipping bound, not as a disabled clamp.
        self.assertEqual(torch.count_nonzero(source.experts._apply_gate(inputs)).item(), 0)
        configure_deepseek_v41_swiglu(source)
        actual = source.experts._apply_gate(inputs)
        gate, up = inputs.chunk(2, dim=-1)
        expected = F.silu(gate) * up
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual_grad = torch.autograd.grad(actual.sum(), inputs, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected.sum(), inputs)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
        self.assertGreater(actual.abs().max().item(), 100)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_shared_experts_disable_clamp_and_preserve_parameters(self):
        """
        Feature: DeepSeek-V4.1 MegaMoe validation.
        Description: Zero applies to shared experts and preserves checkpoint/optimizer identities.
        Expectation: Shared outputs and gradients match without replacing parameters.
        """
        source = _make_source()
        identities = dict(source.named_parameters())
        configure_deepseek_v41_swiglu(source)
        for name, parameter in source.named_parameters():
            self.assertIs(parameter, identities[name])
        inputs = (torch.randn(2, 4, 8) * 100).requires_grad_()
        shared = source.shared_experts
        actual = shared(inputs)
        expected = (F.silu(inputs @ shared.gate_proj.weight.T)
                    * (inputs @ shared.up_proj.weight.T)) @ shared.down_proj.weight.T
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual_grads = torch.autograd.grad(actual.sum(), (inputs, *shared.parameters()), retain_graph=True)
        expected_grads = torch.autograd.grad(expected.sum(), (inputs, *shared.parameters()))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_positive_limit_keeps_both_hf_methods(self):
        """
        Feature: DeepSeek-V4.1 MegaMoe validation.
        Description: Retain the existing HF clamp path when the source requests limit ten.
        Expectation: Limit ten preserves the HF methods and clipped results.
        """
        source = _make_source(10.0)
        gate_fn = source.experts._apply_gate.__func__
        shared_fn = source.shared_experts.forward.__func__
        inputs = torch.randn(3, 8) * 100
        expected = source.experts._apply_gate(inputs)
        configure_deepseek_v41_swiglu(source)
        self.assertIs(source.experts._apply_gate.__func__, gate_fn)
        self.assertIs(source.shared_experts.forward.__func__, shared_fn)
        torch.testing.assert_close(source.experts._apply_gate(inputs), expected, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mismatched_branch_limits_are_rejected(self):
        """
        Feature: DeepSeek-V4.1 MegaMoe validation.
        Description: Routed/shared semantics cannot diverge during configuration.
        Expectation: Different routed and shared limits raise ValueError.
        """
        source = _make_source()
        source.shared_experts.limit = 10
        with self.assertRaisesRegex(ValueError, "same swiglu_limit"):
            configure_deepseek_v41_swiglu(source)


class TestDeepseekV41FP32Oracle(unittest.TestCase):
    """Validate independent math against the actual unrounded HF block."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_output_and_all_gradients_for_learned_and_hotspot_routes(self):
        """
        Feature: DeepSeek-V4.1 MegaMoe validation.
        Description: Fixed selected IDs preserve routing gradients and leave correction biases unused.
        Expectation: Independent FP32 math matches the real FP32 block and all gradients.
        """
        for learned in (True, False):
            with self.subTest(learned=learned):
                source = _make_source()
                configure_deepseek_v41_swiglu(source)
                inputs = torch.randn(2, 4, 8, requires_grad=True)
                dy = torch.randn_like(inputs)
                if learned:
                    _, weights, indices = source.gate(inputs)
                    weights.retain_grad()
                else:
                    indices = torch.arange(2).expand(8, -1)
                    weights = torch.full((8, 2), 0.85, requires_grad=True)
                output = (source.experts(inputs.reshape(-1, 8), indices, weights).reshape_as(inputs)
                          + source.shared_experts(inputs))
                (output * dy).sum().backward()
                oracle = evaluate_fp32_moe(
                    inputs, dy, dict(source.named_parameters()), indices, weights,
                    learned_routing=learned, scaling_factor=1.7,
                )
                torch.testing.assert_close(oracle["output"], output)
                torch.testing.assert_close(oracle["input_grad"], inputs.grad)
                torch.testing.assert_close(oracle["route_weights"], weights)
                torch.testing.assert_close(oracle["route_weight_grad"], weights.grad)
                for name, parameter in source.named_parameters():
                    expected = oracle["gradients"][name]
                    if parameter.grad is None:
                        self.assertIsNone(expected)
                    else:
                        self.assertEqual(expected.dtype, torch.float32)
                        self.assertEqual(expected.device.type, "cpu")
                        torch.testing.assert_close(expected, parameter.grad)


class TestDeepseekV41MegaMoe(unittest.TestCase):
    """Preserve model semantics while changing routed expert layout/execution."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rejects_clamp_and_hash_routing_before_allocation(self):
        """
        Feature: DeepSeek-V4.1 MegaMoe validation.
        Description: Unsupported semantics must fail before creating a native execution owner.
        Expectation: Unsupported source semantics fail before execution resources exist.
        """
        source = _make_source(10.0)
        with self.assertRaisesRegex(ValueError, "swiglu_limit=0"):
            DeepseekV41MegaMoe(source, local_num_tokens=128)
        source = _make_source()
        source.is_hash = True
        with self.assertRaisesRegex(ValueError, "learned routing"):
            DeepseekV41MegaMoe(source, local_num_tokens=128)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_ep_group_rank_controls_expert_slice_and_layout(self):
        """
        Feature: DeepSeek-V4.1 MegaMoe validation.
        Description: Group-local rank one receives experts two and three exactly once.
        Expectation: The native weights equal the exact group-local slice and transpose.
        """
        source = _make_source()
        source.experts.down_proj.requires_grad_(False)
        group = object()
        with patch("torch.distributed.get_world_size", return_value=2), \
                patch("torch.distributed.get_rank", return_value=1):
            adapter = DeepseekV41MegaMoe(source, local_num_tokens=128, ep_group=group)
        try:
            torch.testing.assert_close(adapter.experts.gate_up_weight.transpose(1, 2),
                                       source.experts.gate_up_proj[2:], rtol=0, atol=0)
            torch.testing.assert_close(adapter.experts.down_weight.transpose(1, 2),
                                       source.experts.down_proj[2:], rtol=0, atol=0)
            self.assertTrue(adapter.experts.gate_up_weight.is_contiguous())
            self.assertFalse(adapter.experts.down_weight.requires_grad)
        finally:
            adapter.close()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_text_and_visual_forward_backward_and_optimizer_match(self):
        """
        Feature: DeepSeek-V4.1 MegaMoe validation.
        Description: The real V4.1 router and shared branch agree through three optimizer steps.
        Expectation: Both paths agree through three SGD steps for every router variant.
        """
        for use_v41_router, visual in ((False, False), (True, False), (True, True)):
            with self.subTest(v41_router=use_v41_router, visual=visual):
                source = _make_source(use_v41_router=use_v41_router)
                configure_deepseek_v41_swiglu(source)
                adapter = DeepseekV41MegaMoe(copy.deepcopy(source), local_num_tokens=128)
                source_optimizer = torch.optim.SGD(source.parameters(), lr=0.01)
                adapter_optimizer = torch.optim.SGD(adapter.parameters(), lr=0.01)
                mask = torch.arange(128).reshape(2, 64).remainder(3) == 0 if visual else None
                try:
                    for _ in range(3):
                        inputs = torch.randn(2, 64, 8, requires_grad=True)
                        other_inputs = inputs.detach().clone().requires_grad_()
                        source_optimizer.zero_grad(set_to_none=True)
                        adapter_optimizer.zero_grad(set_to_none=True)
                        expected = source(inputs, image_mask=mask) if visual else source(inputs)
                        with patch.object(MegaMoeExperts, "forward", _cpu_experts):
                            actual = adapter(other_inputs, image_mask=mask)
                        torch.testing.assert_close(actual, expected)
                        expected.square().mean().backward()
                        actual.square().mean().backward()
                        torch.testing.assert_close(other_inputs.grad, inputs.grad)
                        expected_params = dict(source.named_parameters())
                        for name, parameter in adapter.named_parameters():
                            source_name = name.replace("gate_up_weight", "gate_up_proj").replace(
                                "down_weight", "down_proj")
                            expected_grad = expected_params[source_name].grad
                            if expected_grad is None:
                                self.assertIsNone(parameter.grad)
                            else:
                                actual_grad = (parameter.grad.transpose(1, 2)
                                               if name.endswith("_weight") else parameter.grad)
                                torch.testing.assert_close(actual_grad, expected_grad)
                        source_optimizer.step()
                        adapter_optimizer.step()
                        for name, parameter in adapter.named_parameters():
                            source_name = name.replace("gate_up_weight", "gate_up_proj").replace(
                                "down_weight", "down_proj")
                            actual_weight = parameter.transpose(1, 2) if name.endswith("_weight") else parameter
                            torch.testing.assert_close(actual_weight, expected_params[source_name])
                finally:
                    adapter.close()


class TestDeepseekV41LimitOverride(unittest.TestCase):
    """The validation override reaches every constructed MoE branch."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_limit_override_keeps_source_and_configures_the_model(self):
        """
        Feature: DeepSeek-V4.1 MegaMoe validation.
        Description: Preserve source limit ten while explicitly disabling both expert clamps.
        Expectation: All layers honor the explicit override and retain the source asset.
        """
        with tempfile.TemporaryDirectory() as directory:
            assets_path = _write_engram_assets(directory)
            tiny = _tiny_config(assets_path)
            text = tiny.to_dict()
            text.update(
                rope_scaling=tiny.rope_parameters,
                compress_ratios=tiny.v41_compress_ratios,
                kv_source_layer_ids=tiny.v41_kv_source_layer_ids,
                index_source_layer_ids=tiny.v41_index_source_layer_ids,
                candidate_source_layer_id=tiny.v41_candidate_source_layer_id,
                candidate_topk_blocks=tiny.v41_candidate_topk_blocks,
                candidate_block_size=tiny.v41_candidate_block_size,
            )
            vision = {key.removeprefix("v41_vision_"): value for key, value in text.items()
                      if key.startswith("v41_vision_")}
            source = {"model_type": "deepseek_v41", "text_config": text, "vision_config": vision,
                      "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "image_token_id": 3}
            config_path = Path(directory) / "config.json"
            source_bytes = json.dumps(source)
            config_path.write_text(source_bytes, encoding="utf-8")
            default = build_deepseek_v41_validation_config(directory, str(assets_path))
            self.assertEqual(default.swiglu_limit, 10.0)
            for limit in (0.0, 10.0):
                with self.subTest(limit=limit):
                    config = build_deepseek_v41_validation_config(
                        directory, str(assets_path), swiglu_limit=limit)
                    self.assertEqual(config.swiglu_limit, limit)
                    self.assertEqual(config.v41_source_swiglu_limit, 10.0)
                    with no_init_weights():
                        model = DeepseekV41CroppedForCausalLM(config)
                    for layer in model.model.layers:
                        block = layer.mlp
                        self.assertEqual(block.experts.limit, limit)
                        self.assertEqual(block.shared_experts.limit, limit)
                        gate_up = torch.full((1, 32), 20.0)
                        value = block.experts._apply_gate(gate_up)
                        expected_gate = torch.full((1, 16), 20.0 if limit == 0 else 10.0)
                        torch.testing.assert_close(value, F.silu(expected_gate) * expected_gate)
                        shared = block.shared_experts
                        with torch.no_grad():
                            for parameter in shared.parameters():
                                parameter.fill_(0.25)
                        inputs = torch.full((1, 32), 20.0)
                        projected = shared.gate_proj(inputs)
                        bound = projected if limit == 0 else projected.clamp(max=limit)
                        expected = shared.down_proj(F.silu(bound) * bound)
                        torch.testing.assert_close(shared(inputs), expected)
            self.assertEqual(config_path.read_text(encoding="utf-8"), source_bytes)


class TestDeepseekV41Accuracy(unittest.TestCase):
    """Reject invalid evidence and preserve both independent error budgets."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fp32_contract_rejects_invalid_evidence(self):
        """
        Feature: Per-tensor FP32 acceptance.
        Description: Exercise missing gradients, broadcasting, nonfinite and zero references.
        Expectation: Only matching absence and exact zero candidates pass these cases.
        """
        cases = [(None, None, True), (None, torch.zeros(1), False),
                 (torch.zeros(1), None, False), (torch.zeros(2), torch.zeros(1), False),
                 (torch.tensor([float("nan")]), torch.ones(1), False),
                 (torch.ones(1), torch.tensor([float("inf")]), False),
                 (torch.zeros(2), torch.zeros(2), True),
                 (torch.tensor([1e-20]), torch.zeros(1), False)]
        for actual, expected, passed in cases:
            with self.subTest(actual=actual, expected=expected):
                self.assertEqual(fp32_accuracy(actual, expected)["passed"], passed,
                                 "Expected the shape, presence, finite and zero-reference contract")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fp32_contract_requires_both_error_budgets(self):
        """
        Feature: Per-tensor FP32 acceptance.
        Description: Separate widespread error from a sparse outlier and check scale invariance.
        Expectation: Relative L2 above 1% or peak-normalized error above 2% fails independently.
        """
        reference = torch.ones(10000)
        spike = reference.clone()
        spike[0] += 0.03
        cases = [(reference * 1.005, True), (reference * 1.015, False), (spike, False)]
        for actual, passed in cases:
            for scale in (1.0, 1e-6, 1e6):
                with self.subTest(passed=passed, scale=scale):
                    self.assertEqual(fp32_accuracy(actual * scale, reference * scale)["passed"], passed,
                                     "Expected each tensor to satisfy both scale-independent error budgets")
