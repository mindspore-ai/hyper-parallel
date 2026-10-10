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
"""CPU tests for the DeepSeek-V4.1 clamped grouped-GEMM EP adapter."""
# pylint: disable=wrong-import-position

import copy
import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import yaml
from torch.nn import functional

try:
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts
except ImportError as exc:
    raise unittest.SkipTest(f"DeepSeek-V4 Transformers dependency unavailable: {exc}") from exc

from hyper_parallel.components import functional as hp_functional
from hyper_parallel.distributed._builder.rule_resolver import _resolve_local_compute_fn
from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import (
    _deepseek_v41_grouped_expert_forward,
    deepseek_v41_ep_compute_fn,
)
from hyper_parallel.trainer.config.parallelism import PlanOverride, entries_to_plan_overrides
from hyper_parallel.trainer.config.resolver import resolve_component


# This fixture exercises the installed package's configuration contract, not the
# contents of repository-only training examples.
_EP_OVERRIDE_YAML = """
match: "*.mlp"
when: ep
region_dispatch: false
local_compute_fn:
  _target_: >-
    hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel.deepseek_v41_ep_compute_fn
  use_grouped_gemm: {use_grouped_gemm}
"""


def _expert_config() -> DeepseekV4Config:
    """Build an expert-only configuration independent of the native LM and aux loss."""
    return DeepseekV4Config(  # pylint: disable=unexpected-keyword-arg
        hidden_size=16, moe_intermediate_size=24, n_routed_experts=4, swiglu_limit=1.5,
    )


class TestV41ClampedGroupedGemm(unittest.TestCase):
    """Verify grouped GEMM retains the authoritative V4.1 clamp semantics."""

    @staticmethod
    def _cpu_grouped_matmul(
            inputs: torch.Tensor,
            weight: torch.Tensor,
            *,
            bias: None,
            group_list: torch.Tensor,
            group_type: int,
            group_list_type: int,
    ) -> torch.Tensor:
        """Emulate m-axis grouped GEMM while retaining ordinary autograd."""
        del bias, group_type, group_list_type
        outputs = []
        group_start = 0
        for expert_id, group_end in enumerate(group_list.tolist()):
            outputs.append(inputs[group_start:group_end] @ weight[expert_id])
            group_start = group_end
        return torch.cat(outputs, dim=0)

    def test_hf_grouped_forward_and_backward_preserve_clamp(self) -> None:
        """CPU grouped-mm fallback matches an explicit clamped expert oracle."""
        torch.manual_seed(82)
        config = _expert_config()
        config._experts_implementation = "grouped_mm"  # pylint: disable=protected-access
        experts = DeepseekV4Experts(config)
        for parameter in experts.parameters():
            torch.nn.init.normal_(parameter)
        reference = copy.deepcopy(experts)
        inputs = torch.randn(3, config.hidden_size, requires_grad=True)
        expected_inputs = inputs.detach().clone().requires_grad_()
        indices = torch.tensor([[0, 1], [1, 2], [2, 0]])  # Expert 3 is empty.
        weights = torch.rand(3, 2, requires_grad=True)
        expected_weights = weights.detach().clone().requires_grad_()
        output = experts(inputs, indices, weights)
        expected = torch.zeros_like(expected_inputs)
        for expert_id in range(config.n_routed_experts):
            gate, up = functional.linear(  # pylint: disable=not-callable
                expected_inputs, reference.gate_up_proj[expert_id],
            ).chunk(2, dim=-1)
            intermediate = functional.silu(gate.clamp(max=config.swiglu_limit)) * up.clamp(
                min=-config.swiglu_limit, max=config.swiglu_limit,
            )
            projected = functional.linear(intermediate, reference.down_proj[expert_id])  # pylint: disable=not-callable
            route_weight = ((indices == expert_id) * expected_weights).sum(-1, keepdim=True)
            expected = expected + projected * route_weight
        upstream = torch.randn_like(output)
        (output * upstream).sum().backward()
        (expected * upstream).sum().backward()
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(inputs.grad, expected_inputs.grad)
        torch.testing.assert_close(weights.grad, expected_weights.grad)
        for actual, oracle in zip(experts.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, oracle.grad)

    def test_grouped_forward_and_backward_match_clamped_oracle(self) -> None:
        """FP32/BF16 grouped outputs and gradients match an independent FP32 gate."""
        for dtype in (torch.float32, torch.bfloat16):
            for limit in (-1.0, 0.0, 1.5, 10.0):
                with self.subTest(dtype=dtype, limit=limit):
                    self._check_grouped_parity(dtype, limit)

    def _check_grouped_parity(self, dtype: torch.dtype, limit: float) -> None:
        """Compare packed projections around the official FP32 activation formula."""
        torch.manual_seed(83)
        config = _expert_config()
        config.swiglu_limit = limit
        experts = DeepseekV4Experts(config).to(dtype=dtype)
        for parameter in experts.parameters():
            torch.nn.init.normal_(parameter)
        reference = copy.deepcopy(experts)
        counts = torch.tensor([2, 0, 3, 1])
        inputs = torch.randn(counts.sum(), experts.hidden_dim, dtype=dtype, requires_grad=True)
        expected_inputs = inputs.detach().clone().requires_grad_()
        grouped = Mock(side_effect=self._cpu_grouped_matmul)
        # Patch the lazy export without importing the optional NPU backend on CPU.
        with patch.dict(hp_functional.__dict__, {"grouped_matmul": grouped}):
            output = _deepseek_v41_grouped_expert_forward(experts, inputs, counts)
        self.assertEqual(grouped.call_count, 2)
        self.assertEqual(output.dtype, dtype)
        for call in grouped.call_args_list:
            self.assertEqual(call.args[0].dtype, dtype)
            self.assertIsNone(call.kwargs["bias"])
            self.assertEqual(call.kwargs["group_type"], 0)
            self.assertEqual(call.kwargs["group_list_type"], 0)
            torch.testing.assert_close(call.kwargs["group_list"], counts.cumsum(0))

        expected_parts = []
        group_start = 0
        for expert_id, token_count in enumerate(counts.tolist()):
            group_end = group_start + token_count
            gate, up = functional.linear(  # pylint: disable=not-callable
                expected_inputs[group_start:group_end], reference.gate_up_proj[expert_id],
            ).chunk(2, dim=-1)
            gate, up = gate.float(), up.float()
            if limit > 0:
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            intermediate = functional.silu(gate) * up
            expected_parts.append(
                functional.linear(  # pylint: disable=not-callable
                    intermediate.to(dtype),
                    reference.down_proj[expert_id],
                )
            )
            group_start = group_end
        expected = torch.cat(expected_parts, dim=0)
        upstream = torch.randn_like(output)
        (output * upstream).sum().backward()
        (expected * upstream).sum().backward()
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(inputs.grad, expected_inputs.grad)
        for actual, oracle in zip(experts.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, oracle.grad)

    def test_ep_nonpositive_limit_matches_unclamped_oracle(self) -> None:
        """Actual eager/grouped bindings skip disabled clamps and preserve gradients."""
        for dtype, limit, grouped in product((torch.float32, torch.bfloat16), (-1.0, 0.0), (False, True)):
            with self.subTest(dtype=dtype, limit=limit, grouped=grouped):
                self._check_nonpositive_ep_parity(dtype, limit, grouped)

    def _check_nonpositive_ep_parity(self, dtype: torch.dtype, limit: float, grouped: bool) -> None:
        """Compare unsorted local tokens and an empty expert through the real binder."""
        torch.manual_seed(84)
        config = _expert_config()
        config.swiglu_limit = limit
        experts = DeepseekV4Experts(config).to(dtype=dtype)
        for parameter in experts.parameters():
            torch.nn.init.normal_(parameter)
        reference = copy.deepcopy(experts)
        parameters = dict(experts.named_parameters())
        state_keys = set(experts.state_dict())
        module = SimpleNamespace(
            gate=Mock(), experts=experts, shared_experts=Mock(), is_hash=False,
            forward=lambda hidden_states, input_ids=None: hidden_states,
        )
        ep_mesh = Mock()
        ep_mesh.__getitem__ = Mock(return_value=SimpleNamespace(size=lambda: 1))
        deepseek_v41_ep_compute_fn(
            module=module, mesh=None, tp_mesh=None, cp_mesh=None,
            ep_mesh=ep_mesh, use_grouped_gemm=grouped,
        )
        self.assertEqual(set(experts.state_dict()), state_keys)
        for name, parameter in experts.named_parameters():
            self.assertIs(parameter, parameters[name])
        indices = torch.tensor([2, 0, 2, 3, 0, 2])
        inputs = torch.randn(len(indices), config.hidden_size, dtype=dtype, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_()
        matmul = Mock(side_effect=self._cpu_grouped_matmul)
        with patch.dict(hp_functional.__dict__, {"grouped_matmul": matmul}):
            actual = experts(inputs, indices)
        self.assertEqual(matmul.call_count, 2 if grouped else 0)
        expected = torch.zeros_like(reference_inputs)
        for expert_id in range(config.n_routed_experts):
            selected = (indices == expert_id).nonzero(as_tuple=True)[0]
            gate, up = functional.linear(  # pylint: disable=not-callable
                reference_inputs[selected], reference.gate_up_proj[expert_id],
            ).float().chunk(2, dim=-1)
            projected = functional.linear(  # pylint: disable=not-callable
                (functional.silu(gate) * up).to(dtype), reference.down_proj[expert_id],
            )
            expected = expected.index_add(0, selected, projected)
        upstream = torch.randn_like(actual)
        (actual * upstream).sum().backward()
        (expected * upstream).sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(inputs.grad, reference_inputs.grad)
        for actual_parameter, reference_parameter in zip(experts.parameters(), reference.parameters()):
            torch.testing.assert_close(actual_parameter.grad, reference_parameter.grad)

    def test_grouped_bfloat16_rounds_only_after_swiglu(self) -> None:
        """A BF16 witness distinguishes FP32 activation from the old BF16 gate."""
        config = SimpleNamespace(
            num_local_experts=1, hidden_size=1, intermediate_size=1,
            hidden_act="silu", swiglu_limit=10.0,
        )
        experts = DeepseekV4Experts(config).to(dtype=torch.bfloat16)
        with torch.no_grad():
            experts.gate_up_proj.copy_(torch.tensor([[[2.421875], [-4.40625]]]))
            experts.down_proj.fill_(1.0)
        grouped = Mock(side_effect=self._cpu_grouped_matmul)
        with patch.dict(hp_functional.__dict__, {"grouped_matmul": grouped}):
            actual = _deepseek_v41_grouped_expert_forward(
                experts, torch.ones(1, 1, dtype=torch.bfloat16), torch.tensor([1]),
            )
        expected = torch.tensor([[-9.8125]], dtype=torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gate, up = experts.gate_up_proj.flatten().unbind()
        old_result = functional.silu(gate) * up
        self.assertNotEqual(actual.item(), old_result.item())

    def test_ep_grouped_path_binds_model_owned_forward(self) -> None:
        """The EP factory selects the clamped grouped callable without changing generic EP."""
        experts = DeepseekV4Experts(_expert_config())
        module = SimpleNamespace(
            gate=Mock(), experts=experts, shared_experts=Mock(), is_hash=False,
            forward=lambda hidden_states, input_ids=None: hidden_states,
        )
        ep_mesh = Mock()
        ep_mesh.__getitem__ = Mock(return_value=SimpleNamespace(size=lambda: 2))
        prefix = "hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel"
        with patch(f"{prefix}.bind_local_expert_forward") as bind:
            compute = deepseek_v41_ep_compute_fn(
                module=module, mesh=None, tp_mesh=None, cp_mesh=None,
                ep_mesh=ep_mesh, use_grouped_gemm=True,
            )
        self.assertTrue(callable(compute))
        self.assertIs(experts.forward_expert_major.__func__, _deepseek_v41_grouped_expert_forward)
        bind.assert_called_once_with(module, 2, use_grouped_gemm=True, apply_gate=None)

    def test_yaml_override_selects_grouped_or_eager_factory(self) -> None:
        """Typed YAML flags reach the real EP factory for text and VLM contracts."""
        for grouped, multimodal in product((True, False), (False, True)):
            with self.subTest(grouped=grouped, multimodal=multimodal):
                raw = yaml.safe_load(_EP_OVERRIDE_YAML.format(use_grouped_gemm=str(grouped).lower()))
                entry = resolve_component(raw, annotation=PlanOverride, path="$.plan_overrides[0]")
                spec = entries_to_plan_overrides([entry], ep_size=2)["*.mlp"]
                self.assertIs(spec.local_compute_fn.callable, deepseek_v41_ep_compute_fn)
                self.assertIs(spec.local_compute_fn.use_grouped_gemm, grouped)
                self.assertIs(spec.region_dispatch, False)
                module = SimpleNamespace(
                    gate=Mock(), experts=DeepseekV4Experts(_expert_config()), shared_experts=Mock(), is_hash=False,
                    forward=lambda hidden_states, input_ids=None: hidden_states,
                )
                if multimodal:
                    module.forward = lambda hidden_states, input_ids=None, image_mask=None: hidden_states
                ep_mesh = Mock()
                ep_mesh.__getitem__ = Mock(return_value=SimpleNamespace(size=lambda: 2))
                prefix = "hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel"
                with patch(f"{prefix}.bind_local_expert_forward") as bind:
                    compute = _resolve_local_compute_fn(
                        module, spec, mesh=None, mesh_dim_names=(), expert_mesh=ep_mesh,
                    )
                self.assertTrue(callable(compute))
                bind.assert_called_once()
                self.assertEqual(bind.call_args.args, (module, 2))
                self.assertIs(bind.call_args.kwargs["use_grouped_gemm"], grouped)
                states = torch.randn(2, 16)
                mask = torch.tensor([False, True]) if multimodal else None
                with patch(f"{prefix}._routed_and_shared_forward", return_value=states) as routed:
                    output = compute(states, image_mask=mask) if multimodal else compute(states)
                self.assertIs(output, states)
                routed.assert_called_once_with(module, states, mask, ep_mesh.get_group.return_value)

    def test_yaml_override_requires_active_ep(self) -> None:
        """The same parsed EP override is inactive when expert parallelism is off."""
        raw = yaml.safe_load(_EP_OVERRIDE_YAML.format(use_grouped_gemm="true"))
        entry = resolve_component(raw, annotation=PlanOverride, path="$.plan_overrides[0]")
        self.assertEqual(entries_to_plan_overrides([entry], ep_size=1), {})
        self.assertIn("*.mlp", entries_to_plan_overrides([entry], ep_size=2))

    def test_eager_path_retains_source_gate(self) -> None:
        """Disabling grouped GEMM still binds the exact clamped eager activation."""
        experts = DeepseekV4Experts(_expert_config())
        module = SimpleNamespace(
            gate=Mock(), experts=experts, shared_experts=Mock(), is_hash=False,
            forward=lambda hidden_states, input_ids=None: hidden_states,
        )
        ep_mesh = Mock()
        ep_mesh.__getitem__ = Mock(return_value=SimpleNamespace(size=lambda: 2))
        prefix = "hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel"
        with patch(f"{prefix}.bind_local_expert_forward") as bind:
            deepseek_v41_ep_compute_fn(
                module=module, mesh=None, tp_mesh=None, cp_mesh=None,
                ep_mesh=ep_mesh, use_grouped_gemm=False,
            )
        self.assertFalse(hasattr(experts, "forward_expert_major"))
        bind.assert_called_once_with(
            module, 2, use_grouped_gemm=False,
            apply_gate=experts._apply_gate,  # pylint: disable=protected-access
        )
