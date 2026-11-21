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

from hyper_parallel.components.modules.gated_delta_net import (
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
from hyper_parallel.models.qwen3_5.adapter.distributed import (
    context_parallel as adapter_context_parallel,
)
from hyper_parallel.models.registry import get_model_adapter
from hyper_parallel.components.modules import GatedGroupedQueryAttention
from hyper_parallel.models.module_build import prepare_model_components
from hyper_parallel.models.replacement import ModuleReplacementRequest
from hyper_parallel.trainer.config import entries_to_plan_overrides
from hyper_parallel.trainer.config.parser import parse_training_args
from tests.ut.auto_models.distributed.conftest import FakeDeviceMesh


class _FakeMesh:
    """Minimal mesh metadata used without distributed initialization."""

    def __init__(self, size: int) -> None:
        """Store one synthetic CP extent."""
        self._size = size

    def size(self) -> int:
        """Return the synthetic mesh size."""
        return self._size


class _Target(nn.Module):
    """Small forward target for testing atomic wrapper installation."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Preserve the target signature used by the rewrite request."""
        return hidden_states


class _RecordingExecutor(nn.Module):
    """Executor stub that records construction without using collectives."""

    def __init__(
        self,
        module: nn.Module,
        mesh: Any,
        *,
        backend: str,
        chunk_size: int,
    ) -> None:
        """Record executor construction without touching collectives."""
        super().__init__()
        self.module = module
        self.mesh = mesh
        self.backend = backend
        self.chunk_size = chunk_size

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Return arguments so this metadata-only executor remains callable."""
        return args, kwargs


class _TinyGatedDeltaNet(nn.Module):
    """Parameter-only Qwen3.5 GDN boundary for Planner validation."""

    def __init__(self) -> None:
        """Create the projection, Conv, state, norm and output parameters."""
        super().__init__()
        self.in_proj_qkv = nn.Linear(8, 24, bias=False)
        self.in_proj_z = nn.Linear(8, 8, bias=False)
        self.in_proj_b = nn.Linear(8, 2, bias=False)
        self.in_proj_a = nn.Linear(8, 2, bias=False)
        self.conv1d = nn.Conv1d(24, 24, 3, groups=24, bias=False)
        # Keep the Hugging Face parameter name so Planner sees the real contract.
        self.A_log = nn.Parameter(torch.zeros(2))  # pylint: disable=invalid-name
        self.dt_bias = nn.Parameter(torch.zeros(2))
        self.norm = nn.LayerNorm(8)
        self.out_proj = nn.Linear(8, 8, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Provide a callable boundary; Planner does not execute it."""
        return self.out_proj(hidden_states)


class _TinyQwen35Model(nn.Module):
    """Small official-identity model containing one GDN boundary."""

    def __init__(self) -> None:
        """Expose the architecture and stable ``linear_attn`` FQN."""
        super().__init__()
        self.config = SimpleNamespace(
            architectures=["Qwen3_5ForConditionalGeneration"]
        )
        self.model = nn.Module()
        self.model.visual = nn.Linear(8, 8, bias=False)
        self.linear_attn = _TinyGatedDeltaNet()


class TestQwen35GdnCpContracts(unittest.TestCase):
    """Pin the public Qwen3.5 GDN adapter and eager numerical oracle."""

    def test_gated_attention_matches_native_source_and_gradients(self):
        """Feature: Real Qwen3.5 dense/MoE full attention uses the gated GQA family.
        Description: Convert loaded weights and compare B2 partial RoPE with native HF attention.
        Expectation: Outputs, input/parameter gradients and retained Norm parameters agree.
        """
        def attention(module, query, key, value, mask, **kwargs):
            """CPU kernel oracle; projections, gate, Norm and RoPE remain production code."""
            del module
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, scale=kwargs["scaling"], enable_gqa=True)
            return output.transpose(1, 2), None

        for config_type, source_type in ((Qwen3_5TextConfig, Qwen3_5Attention),
                                         (Qwen3_5MoeTextConfig, Qwen3_5MoeAttention)):
            with self.subTest(source=source_type.__name__):
                torch.manual_seed(35)
                config = config_type(hidden_size=32, num_attention_heads=4, num_key_value_heads=2,
                                     head_dim=8, num_hidden_layers=1, attention_bias=True)
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

    def test_registration_and_wrapper_metadata(self):
        """Both HF identities discover the same pair of GDN CP wrappers."""
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
            "qwen3_5_gdn_ulysses_cp_wrapper",
            "qwen3_5_gdn_p2p_cp_wrapper",
        ):
            wrapper = getattr(adapter_context_parallel, name)
            self.assertEqual(wrapper._injection_meta.kind, INNER_WRAPPER)

    def test_yaml_cp_condition_selects_gdn_wrapper(self):
        """The new Trainer activates the GDN rule only when CP is enabled."""
        yaml_text = """
model:
  _target_: torch.nn.Identity
optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-3
plan_overrides:
  - match: "*.linear_attn.forward"
    when: cp
    replace:
      kind: method
      implementation:
        _target_: hyper_parallel.models.qwen3_5.adapter.distributed.context_parallel.qwen3_5_gdn_p2p_cp_wrapper
        backend: triton
        chunk_size: 64
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qwen3_5_gdn_cp.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            config = parse_training_args([str(path)])

        entry = config.plan_overrides[0]
        self.assertFalse(entry.region_dispatch)
        self.assertNotIn("region_dispatch", entry.to_dict())
        self.assertEqual(entries_to_plan_overrides([entry], cp_size=1), {})
        active = entries_to_plan_overrides([entry], cp_size=4)
        self.assertIs(active["*.linear_attn"].inner_wrapper, entry.replace.implementation)
        self.assertEqual(entry.replace.implementation.backend, "triton")

    def test_cp4_plan_derives_gdn_execution_boundary(self):
        """Planner derives the GDN boundary and preserves its CP injection."""
        model = _TinyQwen35Model()
        override = ModuleShardingSpec(
            inner_target="self",
            inner_wrapper=adapter_context_parallel.qwen3_5_gdn_p2p_cp_wrapper,
            region_dispatch=False,
        )
        plan = ShardingPlanner(plan_overrides={"linear_attn": override}).plan(
            model,
            FakeDeviceMesh((4,), ("cp",)),
            cp_size=4,
        )
        spec = plan.modules["linear_attn"]
        self.assertIs(
            spec.inner_wrapper,
            adapter_context_parallel.qwen3_5_gdn_p2p_cp_wrapper,
        )
        self.assertFalse(spec.region_dispatch)
        expected_params = {
            name for name, _ in model.linear_attn.named_parameters()
            if not name.startswith("norm.")
        }
        self.assertEqual(set(spec.params), expected_params)
        for placements in spec.params.values():
            self.assertEqual(placements[CP], Replicate())

    def test_wrapper_is_atomic_and_rejects_invalid_composition(self):
        """Return an atomic rewrite and reject duplicate or TP+CP wrapping."""
        target = _Target()
        original_forward = target.forward
        with patch.object(
            adapter_context_parallel,
            "GatedDeltaNetP2PCP",
            _RecordingExecutor,
        ):
            request = adapter_context_parallel.qwen3_5_gdn_p2p_cp_wrapper(
                target,
                None,
                None,
                _FakeMesh(4),
                None,
                backend="eager",
                chunk_size=8,
            )
        self.assertIsInstance(request, _ForwardRewriteRequest)
        self.assertIs(request.target, target)
        self.assertIs(target.forward.__func__, original_forward.__func__)
        self.assertEqual(request.companion_attrs["_hp_gdn_cp_config"]["mode"], "p2p")

        target._hp_gdn_cp_config = {"mode": "p2p"}
        with self.assertRaisesRegex(RuntimeError, "already been applied"):
            adapter_context_parallel.qwen3_5_gdn_p2p_cp_wrapper(
                target, None, None, _FakeMesh(4), None
            )

        with self.assertRaisesRegex(NotImplementedError, "simultaneous TP and CP"):
            adapter_context_parallel.qwen3_5_gdn_p2p_cp_wrapper(
                _Target(), None, _FakeMesh(2), _FakeMesh(4), None
            )

    def test_eager_chunk_partitions_have_matching_forward_and_backward(self):
        """Different chunk partitions implement the same recurrent GDN map."""
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
