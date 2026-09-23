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
"""CPU contracts for the generic DeepSeek grouped-experts replacement."""

import unittest
from unittest import mock

import torch
from torch import nn

from hyper_parallel.components.quantization.config import LowPrecisionDtypeScheme
from hyper_parallel.components.quantization.modules import GroupedExperts
from hyper_parallel.models.deepseek_v3.adapter.replacements import (
    replace_grouped_experts,
)
from tests.common.mark_utils import arg_mark


cpu_test = arg_mark(
    plat_marks=["cpu_linux", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)


class _SourceExperts(nn.Module):
    """Packed source module with the registration order used by DeepSeek."""

    def __init__(self, experts=2, hidden=4, intermediate=3):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.randn(experts, 2 * intermediate, hidden)
        )
        self.down_proj = nn.Parameter(
            torch.randn(experts, hidden, intermediate)
        )
        self.act_fn = nn.SiLU()
        self.register_buffer("sentinel", torch.tensor(11), persistent=False)
        self.config = {"name": "source"}


def _dense_grouped_apply(inputs, weight, group_list, strategy, group_list_type):
    """Autograd-native replacement for the low-precision GMM call in shell tests."""
    del strategy
    if group_list_type == 0:
        boundaries = group_list.tolist()
        counts = [end - (boundaries[index - 1] if index else 0)
                  for index, end in enumerate(boundaries)]
    else:
        counts = group_list.tolist()
    parts = inputs.split(counts, dim=0)
    return torch.cat(
        [part @ weight[index].transpose(-2, -1) for index, part in enumerate(parts)],
        dim=0,
    )


def _reference_experts(module, hidden_states, top_k_index, top_k_weights):
    """Straightforward token/expert loop used as a routing oracle."""
    token_outputs = []
    for token_index in range(hidden_states.shape[0]):
        combined = torch.zeros_like(hidden_states[token_index])
        for route_index in range(top_k_index.shape[1]):
            expert = int(top_k_index[token_index, route_index])
            hidden = hidden_states[token_index]
            gate_up = hidden @ module.gate_up_proj[expert].transpose(-2, -1)
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = (module.act_fn(gate) * up) @ module.down_proj[
                expert
            ].transpose(-2, -1)
            combined = combined + top_k_weights[token_index, route_index] * expert_output
        token_outputs.append(combined)
    return torch.stack(token_outputs)


class TestGroupedExpertsConversion(unittest.TestCase):
    """Conversion preserves checkpoint identity while changing only computation."""

    @cpu_test
    def test_from_module_preserves_registered_state_and_metadata(self):
        """The shell is new, but parameters/buffers/submodules remain identical."""
        source = _SourceExperts()
        source.eval()
        converted = GroupedExperts.from_module(
            source, fqn="model.layers.0.mlp.experts", grouped_linear=object()
        )

        self.assertIsNot(converted, source)
        self.assertIs(converted.gate_up_proj, source.gate_up_proj)
        self.assertIs(converted.down_proj, source.down_proj)
        self.assertIs(converted.sentinel, source.sentinel)
        self.assertIs(converted.act_fn, source.act_fn)
        self.assertIs(converted.config, source.config)
        self.assertFalse(converted.training)
        self.assertEqual(tuple(converted.state_dict()), ("gate_up_proj", "down_proj"))
        self.assertEqual(converted.fqn, "model.layers.0.mlp.experts")

    @cpu_test
    def test_routing_forward_and_backward_match_dense_reference(self):
        """Sorting/restoration and top-k weighting preserve dense MoE semantics."""
        torch.manual_seed(23)
        source = _SourceExperts()
        converted = GroupedExperts.from_module(
            source, fqn="experts", grouped_linear=object()
        )
        hidden = torch.randn(4, 4, requires_grad=True)
        indices = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])
        weights = torch.tensor(
            [[0.7, 0.3], [0.8, 0.2], [0.6, 0.4], [0.55, 0.45]]
        )

        reference = _reference_experts(source, hidden, indices, weights)
        with mock.patch(
            "hyper_parallel.components.quantization.modules.grouped_experts."
            "_GroupedLinearFunction.apply",
            side_effect=_dense_grouped_apply,
        ):
            actual = converted(hidden, indices, weights)

        torch.testing.assert_close(actual, reference)
        actual.sum().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertIsNotNone(converted.gate_up_proj.grad)
        self.assertIsNotNone(converted.down_proj.grad)

    @cpu_test
    def test_source_contract_rejects_registration_and_shape_mismatch(self):
        """The adapter fails before partially converting an incompatible module."""
        missing = nn.Module()
        missing.register_parameter("gate_up_proj", nn.Parameter(torch.randn(2, 6, 4)))
        with self.assertRaisesRegex(TypeError, "must register packed expert parameters"):
            GroupedExperts.from_module(missing, fqn="bad")

        bad_shape = _SourceExperts()
        bad_shape.down_proj = nn.Parameter(torch.randn(2, 5, 3))
        with self.assertRaisesRegex(ValueError, "incompatible gate/up and down"):
            GroupedExperts.from_module(bad_shape, fqn="bad")


class TestGroupedExpertsFactory(unittest.TestCase):
    """The replacement factory owns policy selection and topology gates."""

    @cpu_test
    def test_factory_passes_policy_and_live_weight_shapes(self):
        """Strategy construction receives the selected policy and both live tiles."""
        source = _SourceExperts(experts=2, hidden=32, intermediate=32)
        policy = LowPrecisionDtypeScheme(
            weight_format="mxfp4", act_format="mxfp8"
        )
        strategy = object()
        patch_path = (
            "hyper_parallel.models.deepseek_v3.adapter.replacements."
            "build_low_precision_strategy"
        )
        with mock.patch(patch_path, return_value=strategy) as build:
            converted = replace_grouped_experts(
                module=source,
                module_fqn="model.layers.0.mlp.experts",
                context={"low_precision": policy},
            )

        build.assert_called_once_with(
            policy,
            tile_shapes=((2, 64, 32), (2, 32, 32)),
        )
        self.assertIsInstance(converted, GroupedExperts)
        self.assertIs(converted.grouped_linear, strategy)
        self.assertIs(converted.gate_up_proj, source.gate_up_proj)

    @cpu_test
    def test_factory_rejects_every_active_parallel_axis(self):
        """Packed experts currently support only TP=CP=EP=PP=1."""
        source = _SourceExperts(experts=2, hidden=32, intermediate=32)
        for axis in ("tp", "cp", "ep", "pp"):
            with self.subTest(axis=axis):
                with self.assertRaisesRegex(NotImplementedError, axis.upper()):
                    replace_grouped_experts(
                        module=source,
                        module_fqn="experts",
                        context={axis: object()},
                    )


if __name__ == "__main__":
    unittest.main()
