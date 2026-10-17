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
from hyper_parallel.components.quantization.functional import (
    FakeW4A8GroupedLinear,
    HiFloat8GroupedLinear,
)
from hyper_parallel.components.quantization.functional.base_linear_func import LinearStrategy
from hyper_parallel.components.quantization.functional.hifloat8_linear_func import (
    HiFloat8LinearStrategy,
)
from hyper_parallel.components.quantization.functional.linear_strategy_factory import (
    build_linear_strategy,
)
from hyper_parallel.components.quantization.functional.mxfp8_linear_func import (
    MXFP8LinearStrategy,
)
from hyper_parallel.components.quantization.modules.grouped_experts import GroupedExperts
from hyper_parallel.components.quantization.modules.linear import LowPrecisionLinear
from hyper_parallel.components.quantization.tensor import QuantizedTensorStorage
from hyper_parallel.models.deepseek_v3.adapter.conversion.module_replacement import (
    replace_grouped_experts,
    replace_linear,
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
            source,
            fqn="model.layers.0.mlp.experts",
            grouped_linear=FakeW4A8GroupedLinear(),
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
            source,
            fqn="experts",
            grouped_linear=FakeW4A8GroupedLinear(),
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
            GroupedExperts.from_module(
                missing,
                fqn="bad",
                grouped_linear=FakeW4A8GroupedLinear(),
            )

        bad_shape = _SourceExperts()
        bad_shape.down_proj = nn.Parameter(torch.randn(2, 5, 3))
        with self.assertRaisesRegex(ValueError, "incompatible gate/up and down"):
            GroupedExperts.from_module(
                bad_shape,
                fqn="bad",
                grouped_linear=FakeW4A8GroupedLinear(),
            )

    @cpu_test
    def test_invalid_expert_ids_fail_before_grouped_compute(self):
        """Out-of-range routes never reach a grouped-linear kernel."""
        source = _SourceExperts()
        converted = GroupedExperts.from_module(
            source,
            fqn="experts",
            grouped_linear=FakeW4A8GroupedLinear(),
        )
        hidden = torch.ones(1, 4)
        weights = torch.ones(1, 1)
        for expert_id in (-1, 2):
            with self.subTest(expert_id=expert_id):
                with mock.patch.object(converted, "_grouped_forward") as compute:
                    with self.assertRaises((ValueError, RuntimeError)):
                        converted(hidden, torch.tensor([[expert_id]]), weights)
                    compute.assert_not_called()


class TestGroupedExpertsFactory(unittest.TestCase):
    """The replacement factory owns policy selection and topology gates."""

    @cpu_test
    def test_factory_passes_policy_and_live_weight_shapes(self):
        """Strategy construction receives the selected policy and both live tiles."""
        source = _SourceExperts(experts=2, hidden=32, intermediate=32)
        policy = LowPrecisionDtypeScheme(
            weight_format="mxfp4", act_format="mxfp8"
        )
        strategy = FakeW4A8GroupedLinear()
        patch_path = (
            "hyper_parallel.models.deepseek_v3.adapter.conversion.module_replacement."
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

    @cpu_test
    def test_factory_selects_hif8_without_mx_tile_gate(self):
        """The HiFloat8 policy reaches its strategy on unaligned expert shapes."""
        source = _SourceExperts(experts=2, hidden=31, intermediate=7)
        policy = LowPrecisionDtypeScheme(weight_format="hif8", act_format="hif8")
        path = (
            "hyper_parallel.components.quantization.functional.strategy_factory."
            "validate_hifloat8_gmm_runtime"
        )
        with mock.patch(path) as validate:
            converted = replace_grouped_experts(
                module=source,
                module_fqn="model.layers.0.mlp.experts",
                context={"low_precision": policy},
            )

        self.assertIsInstance(converted.grouped_linear, HiFloat8GroupedLinear)
        self.assertIs(converted.gate_up_proj, source.gate_up_proj)
        validate.assert_called_once_with()


class _ContractStrategy(LinearStrategy):
    """Concrete strategy used to test shell construction without an accelerator."""

    def quantize_input(
        self,
        inputs: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> QuantizedTensorStorage:
        raise NotImplementedError

    def quantize_weight(
        self,
        weight: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> QuantizedTensorStorage:
        raise NotImplementedError

    def quantize_grad_output(
        self,
        grad_output: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> QuantizedTensorStorage:
        raise NotImplementedError

    def matmul(
        self,
        left: QuantizedTensorStorage,
        right: QuantizedTensorStorage,
        *,
        layout: str,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        raise NotImplementedError


class TestLinearReplacement(unittest.TestCase):
    """The DeepSeek adapter replacement preserves source Linear state."""

    @cpu_test
    def test_replacement_reuses_parameters_and_training_state(self):
        """Conversion does not allocate new weights or alter module mode."""
        source = nn.Linear(8, 4)
        source.eval()
        strategy = _ContractStrategy()
        policy = object()
        path = (
            "hyper_parallel.models.deepseek_v3.adapter.conversion.module_replacement."
            "build_linear_strategy"
        )
        with mock.patch(path, return_value=strategy) as build:
            converted = replace_linear(
                module=source,
                module_fqn="model.proj",
                context={"low_precision": policy},
            )

        build.assert_called_once_with(policy, in_features=8, out_features=4)
        self.assertIsInstance(converted, LowPrecisionLinear)
        self.assertIs(converted.weight, source.weight)
        self.assertIs(converted.bias, source.bias)
        self.assertIs(converted.strategy, strategy)
        self.assertFalse(converted.training)

    @cpu_test
    def test_replacement_rejects_non_exact_linear_types(self):
        """Subclasses are excluded because the adapter contract is exact nn.Linear."""
        class DerivedLinear(nn.Linear):
            pass

        with self.assertRaisesRegex(TypeError, "must be exact nn.Linear"):
            replace_linear(
                module=DerivedLinear(8, 4),
                module_fqn="model.proj",
                context={},
            )

    @cpu_test
    def test_replacement_rejects_pipeline_parallel(self):
        """The direct Dense replacement rejects PP as specified by its contract."""
        source = nn.Linear(8, 4)
        with self.assertRaisesRegex(NotImplementedError, "pipeline parallelism"):
            replace_linear(
                module=source,
                module_fqn="model.proj",
                context={"pp": 2},
            )

    @cpu_test
    def test_replacement_adds_target_context_to_invalid_policy(self):
        """Policy errors identify the FQN selected by the replacement plan."""
        source = nn.Linear(8, 4)
        path = (
            "hyper_parallel.models.deepseek_v3.adapter.conversion.module_replacement."
            "build_linear_strategy"
        )
        with mock.patch(path, side_effect=ValueError("bad policy")):
            with self.assertRaisesRegex(ValueError, "model.proj.*bad policy"):
                replace_linear(
                    module=source,
                    module_fqn="model.proj",
                    context={},
                )


class TestLinearStrategyFactory(unittest.TestCase):
    """Dense policy dispatch stays independent of grouped strategy selection."""

    @cpu_test
    def test_factory_selects_mxfp8_after_alignment_gate(self):
        """MXFP8 runtime validation follows a successful 32-feature shape check."""
        policy = LowPrecisionDtypeScheme(weight_format="mxfp8", act_format="mxfp8")
        path = (
            "hyper_parallel.components.quantization.functional."
            "linear_strategy_factory.validate_npu_runtime"
        )
        with mock.patch(path) as validate:
            strategy = build_linear_strategy(policy, in_features=64, out_features=32)

        self.assertIsInstance(strategy, MXFP8LinearStrategy)
        validate.assert_called_once_with()
        with mock.patch(path) as validate:
            with self.assertRaisesRegex(ValueError, "multiples of 32"):
                build_linear_strategy(policy, in_features=63, out_features=32)
        validate.assert_not_called()

    @cpu_test
    def test_factory_selects_hif8_without_tile_alignment(self):
        """Native HiFloat8 accepts non-MX feature sizes and validates its runtime."""
        policy = LowPrecisionDtypeScheme(weight_format="hif8", act_format="hif8")
        path = (
            "hyper_parallel.components.quantization.functional."
            "linear_strategy_factory.validate_hifloat8_runtime"
        )
        with mock.patch(path) as validate:
            strategy = build_linear_strategy(policy, in_features=17, out_features=13)

        self.assertIsInstance(strategy, HiFloat8LinearStrategy)
        validate.assert_called_once_with()

    @cpu_test
    def test_factory_rejects_fake_dense_policy(self):
        """Dense fake QAT remains outside the current Linear design scope."""
        policy = LowPrecisionDtypeScheme(
            is_fake_quantize=True,
            weight_format="mxfp4",
            act_format="mxfp8",
        )
        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            build_linear_strategy(policy, in_features=64, out_features=32)


if __name__ == "__main__":
    unittest.main()
