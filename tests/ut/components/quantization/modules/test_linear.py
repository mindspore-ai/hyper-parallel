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
"""CPU contracts for the policy-selected low-precision Linear replacement."""

import copy
import unittest
from unittest import mock

import torch
from torch import nn

from hyper_parallel.components import quantization
from hyper_parallel.components.quantization import functional as quantized_functional
from hyper_parallel.components.quantization import modules as quantized_modules
from hyper_parallel.components.quantization.config import (
    LowPrecisionConfig,
    LowPrecisionDtypeScheme,
)
from hyper_parallel.components.quantization.functional.base_linear_func import (
    LinearStrategy,
)
from hyper_parallel.components.quantization.functional.hifloat8_linear_func import (
    HiFloat8LinearStrategy,
)
from hyper_parallel.components.quantization.functional.linear_strategy_factory import (
    build_linear_strategy,
)
from hyper_parallel.components.quantization.functional.mxfp8_linear_func import (
    MXFP8LinearStrategy,
)
from hyper_parallel.components.quantization.modules.linear import (
    LowPrecisionLinear,
    replace_linear,
)
from hyper_parallel.models.replacement import (
    apply_module_replacements,
    compile_module_replacements,
)
from hyper_parallel.trainer.config.parallelism import (
    PlanOverride,
    entries_to_module_replacements,
)
from hyper_parallel.trainer.config.target import Target
from tests.common.mark_utils import arg_mark


cpu_test = arg_mark(
    plat_marks=["cpu_linux", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)


class _DenseStorage:
    """Minimal directional storage for the CPU strategy."""

    def __init__(self, tensor: torch.Tensor) -> None:
        """Store the full-precision value used by CPU matrix multiplies."""

        self.tensor = tensor

    def update_usage(self, rowwise: bool = True, colwise: bool = True) -> None:
        """Accept the lifecycle release request used by the shared flow."""

        del rowwise, colwise


class _DenseLinearStrategy(LinearStrategy):
    """Full-precision hooks used to verify the canonical shell on CPU."""

    @staticmethod
    def _quantize(tensor: torch.Tensor) -> _DenseStorage:
        return _DenseStorage(tensor)

    def quantize_input(
        self,
        inputs: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> _DenseStorage:
        """Wrap one input value without changing it."""

        del rowwise, colwise
        return self._quantize(inputs)

    def quantize_weight(
        self,
        weight: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> _DenseStorage:
        """Wrap one weight value without changing it."""

        del rowwise, colwise
        return self._quantize(weight)

    def quantize_grad_output(
        self,
        grad_output: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> _DenseStorage:
        """Wrap one gradient-output value without changing it."""

        del rowwise, colwise
        return self._quantize(grad_output)

    def matmul(
        self,
        left: _DenseStorage,
        right: _DenseStorage,
        *,
        layout: str,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Execute the requested Dense layout in full precision."""

        left_tensor = left.tensor.transpose(-1, -2) if layout[0] == "T" else left.tensor
        right_tensor = right.tensor.transpose(-1, -2) if layout[1] == "T" else right.tensor
        return (left_tensor @ right_tensor).to(output_dtype)


class _TwoLinearModel(nn.Module):
    """Small model with valid MXFP8 and HiFloat8 Linear shapes."""

    def __init__(self) -> None:
        """Create one biasless MXFP8 target and one HiFloat8 target."""

        super().__init__()
        self.mxfp = nn.Linear(32, 64, bias=False)
        self.hifloat8 = nn.Linear(31, 17, bias=True)


class TestLinearReplacement(unittest.TestCase):
    """The public replacement selects compute without changing model state."""

    @cpu_test
    def test_default_policy_selects_mxfp8_and_preserves_state(self):
        """The legacy default installs MXFP8 behind the canonical shell."""
        source = nn.Linear(32, 64, bias=True)
        source.eval()

        with mock.patch(
            "hyper_parallel.components.quantization.functional.linear_strategy_factory."
            "validate_npu_runtime"
        ):
            converted = replace_linear(
                module=source,
                module_fqn="model.proj",
                context={},
            )

        self.assertIsInstance(converted, LowPrecisionLinear)
        self.assertIsInstance(converted.strategy, MXFP8LinearStrategy)
        self.assertIs(converted.weight, source.weight)
        self.assertIs(converted.bias, source.bias)
        self.assertFalse(converted.training)
        self.assertEqual(tuple(converted.state_dict()), tuple(source.state_dict()))
        self.assertEqual(tuple(converted.named_children()), ())

    @cpu_test
    def test_public_module_exposes_only_one_linear_replacement(self):
        """Callers see one Linear replacement instead of format-specific targets."""
        self.assertIs(quantized_modules.replace_linear, replace_linear)
        self.assertIs(quantization.replace_linear, replace_linear)
        for old_name in ("replace_mxfp8_linear", "replace_hifloat8_linear"):
            with self.subTest(old_name=old_name):
                self.assertNotIn(old_name, quantized_modules.__all__)
                self.assertFalse(hasattr(quantized_modules, old_name))
                self.assertNotIn(old_name, quantization.__all__)
                self.assertFalse(hasattr(quantization, old_name))

    @cpu_test
    def test_functional_surface_exposes_factory_without_format_wrappers(self):
        """The public functional surface has one policy-based strategy builder."""

        self.assertIs(
            quantized_functional.build_linear_strategy,
            build_linear_strategy,
        )
        for wrapper_name in ("mxfp8_linear", "hifloat8_linear"):
            with self.subTest(wrapper_name=wrapper_name):
                self.assertNotIn(wrapper_name, quantized_functional.__all__)
                self.assertFalse(hasattr(quantized_functional, wrapper_name))

    @cpu_test
    def test_replacement_builds_strategy_once_and_forward_reuses_it(self):
        """Replacement owns strategy creation; the forward hot path only reuses it."""

        source = nn.Linear(2, 3, bias=False, dtype=torch.float64)
        strategy = _DenseLinearStrategy()
        policy = LowPrecisionDtypeScheme(
            weight_format="hif8",
            act_format="hif8",
        )
        with mock.patch(
            "hyper_parallel.components.quantization.modules.linear.build_linear_strategy",
            return_value=strategy,
        ) as build_strategy:
            converted = replace_linear(
                module=source,
                module_fqn="model.proj",
                context={"low_precision": policy},
            )
            first_output = converted(torch.ones(2, 2, dtype=torch.float64))
            second_output = converted(torch.ones(2, 2, dtype=torch.float64))

        self.assertIs(converted.strategy, strategy)
        build_strategy.assert_called_once_with(
            policy,
            in_features=2,
            out_features=3,
        )
        torch.testing.assert_close(first_output, second_output)

    @cpu_test
    def test_typed_and_legacy_configs_select_the_expected_strategy(self):
        """The factory accepts resolved schemes and the legacy global config."""
        cases = (
            (LowPrecisionConfig(), 32, 64, MXFP8LinearStrategy),
            (
                LowPrecisionConfig(format="hif8", scaling="current"),
                31,
                17,
                HiFloat8LinearStrategy,
            ),
            (
                LowPrecisionDtypeScheme(
                    weight_format="hif8",
                    act_format="hif8",
                    block_size=128,
                ),
                31,
                17,
                HiFloat8LinearStrategy,
            ),
        )

        with mock.patch(
            "hyper_parallel.components.quantization.functional.linear_strategy_factory."
            "validate_npu_runtime"
        ), mock.patch(
            "hyper_parallel.components.quantization.functional.linear_strategy_factory."
            "validate_hifloat8_runtime"
        ):
            for config, in_features, out_features, strategy_type in cases:
                with self.subTest(config=config):
                    strategy = build_linear_strategy(
                        config,
                        in_features=in_features,
                        out_features=out_features,
                    )

                    self.assertIsInstance(strategy, strategy_type)

    @cpu_test
    def test_plan_policy_installs_both_strategies_and_keeps_checkpoint(self):
        """Named plan policies reach the factory through the real replacement engine."""
        model = _TwoLinearModel()
        mxfp_source = model.mxfp
        hifloat8_source = model.hifloat8
        target = Target(
            replace_linear,
            target_path=(
                "hyper_parallel.components.quantization.modules.linear.replace_linear"
            ),
        )
        entries = (
            PlanOverride(
                match="mxfp",
                when="low_precision",
                low_precision_dtype_scheme="mxfp8",
                module_type="torch.nn.Linear",
                exact_type=True,
                replace_module=target,
            ),
            PlanOverride(
                match="hifloat8",
                when="low_precision",
                low_precision_dtype_scheme="hifloat8",
                module_type="torch.nn.Linear",
                exact_type=True,
                replace_module=target,
            ),
        )
        config = LowPrecisionConfig(
            enabled=True,
            dtype_schemes={
                "mxfp8": LowPrecisionDtypeScheme(),
                "hifloat8": LowPrecisionDtypeScheme(
                    weight_format="hif8",
                    act_format="hif8",
                ),
            },
        )
        specs = entries_to_module_replacements(
            list(entries),
            low_precision_enabled=True,
            low_precision_config=config,
        )
        plan = compile_module_replacements(model, specs)

        with mock.patch(
            "hyper_parallel.components.quantization.functional.linear_strategy_factory."
            "validate_npu_runtime"
        ), mock.patch(
            "hyper_parallel.components.quantization.functional.linear_strategy_factory."
            "validate_hifloat8_runtime"
        ):
            converted_model, _ = apply_module_replacements(
                model,
                plan,
                context={},
            )

        self.assertIsInstance(converted_model.mxfp, LowPrecisionLinear)
        self.assertIsInstance(converted_model.mxfp.strategy, MXFP8LinearStrategy)
        self.assertIs(converted_model.mxfp.weight, mxfp_source.weight)
        self.assertIsNone(converted_model.mxfp.bias)
        self.assertIsInstance(converted_model.hifloat8, LowPrecisionLinear)
        self.assertIsInstance(
            converted_model.hifloat8.strategy,
            HiFloat8LinearStrategy,
        )
        self.assertIs(converted_model.hifloat8.weight, hifloat8_source.weight)
        self.assertIs(converted_model.hifloat8.bias, hifloat8_source.bias)
        self.assertEqual(
            tuple(converted_model.state_dict()),
            ("mxfp.weight", "hifloat8.weight", "hifloat8.bias"),
        )
        restored = _TwoLinearModel()
        load_result = restored.load_state_dict(converted_model.state_dict(), strict=True)
        self.assertEqual(load_result.missing_keys, [])
        self.assertEqual(load_result.unexpected_keys, [])

    @cpu_test
    def test_bf16_checkpoint_loads_into_replaced_linear_and_resumes_step(self):
        """A BF16 model and optimizer checkpoint resumes through the new shell."""

        base = nn.Linear(32, 64, bias=True, dtype=torch.bfloat16)
        base_optimizer = torch.optim.SGD(
            base.parameters(),
            lr=0.01,
            momentum=0.9,
            foreach=False,
        )
        for parameter in base.parameters():
            parameter.grad = torch.ones_like(parameter)
        base_optimizer.step()
        model_state = copy.deepcopy(base.state_dict())
        optimizer_state = copy.deepcopy(base_optimizer.state_dict())
        checkpoint_momentum = {
            name: base_optimizer.state[parameter]["momentum_buffer"].clone()
            for name, parameter in base.named_parameters()
        }

        target = nn.Linear(32, 64, bias=True, dtype=torch.bfloat16)
        with mock.patch(
            "hyper_parallel.components.quantization.functional.linear_strategy_factory."
            "validate_npu_runtime"
        ):
            reference = replace_linear(
                module=base,
                module_fqn="reference.proj",
                context={},
            )
            converted = replace_linear(
                module=target,
                module_fqn="model.proj",
                context={},
            )
        load_result = converted.load_state_dict(model_state, strict=True)
        optimizer = torch.optim.SGD(
            converted.parameters(),
            lr=0.01,
            momentum=0.9,
            foreach=False,
        )
        optimizer.load_state_dict(optimizer_state)
        reference.strategy = _DenseLinearStrategy()
        converted.strategy = _DenseLinearStrategy()
        converted_parameters = dict(converted.named_parameters())

        for name, parameter in converted_parameters.items():
            torch.testing.assert_close(
                optimizer.state[parameter]["momentum_buffer"],
                checkpoint_momentum[name],
                rtol=0,
                atol=0,
            )

        step_input = torch.ones(2, 32, dtype=torch.bfloat16)
        base_optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        reference_output = reference(step_input)
        actual_output = converted(step_input)
        reference_output.float().sum().backward()
        actual_output.float().sum().backward()
        base_optimizer.step()
        optimizer.step()

        torch.testing.assert_close(actual_output, reference_output, rtol=0, atol=0)
        self.assertEqual(load_result.missing_keys, [])
        self.assertEqual(load_result.unexpected_keys, [])
        self.assertEqual(converted.weight.dtype, torch.bfloat16)
        for name, parameter in converted_parameters.items():
            reference_parameter = dict(reference.named_parameters())[name]
            torch.testing.assert_close(parameter, reference_parameter, rtol=0, atol=0)
            torch.testing.assert_close(
                optimizer.state[parameter]["momentum_buffer"],
                base_optimizer.state[reference_parameter]["momentum_buffer"],
                rtol=0,
                atol=0,
            )

    @cpu_test
    def test_canonical_shell_preserves_forward_and_backward_semantics(self):
        """Strategy compute and one shell bias add produce all expected gradients."""
        source = nn.Linear(2, 2, bias=True, dtype=torch.float64)
        with torch.no_grad():
            source.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
            source.bias.copy_(torch.tensor([0.5, -0.5]))
        converted = LowPrecisionLinear.from_linear(
            source,
            strategy=_DenseLinearStrategy(),
        )
        inputs = torch.tensor(
            [[1.0, 2.0], [-1.0, 3.0]],
            dtype=torch.float64,
            requires_grad=True,
        )

        output = converted(inputs)
        output.sum().backward()

        torch.testing.assert_close(
            output,
            torch.tensor([[5.5, 10.5], [5.5, 8.5]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            inputs.grad,
            torch.tensor([[4.0, 6.0], [4.0, 6.0]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            converted.weight.grad,
            torch.tensor([[0.0, 5.0], [0.0, 5.0]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            converted.bias.grad,
            torch.tensor([2.0, 2.0], dtype=torch.float64),
        )

    @cpu_test
    def test_unsupported_policy_shape_source_and_pipeline_fail_explicitly(self):
        """Invalid Dense requests fail without falling back to another strategy."""
        with self.assertRaisesRegex(ValueError, "not tile aligned"):
            build_linear_strategy(None, in_features=31, out_features=64)
        with self.assertRaisesRegex(ValueError, "model.misaligned"):
            replace_linear(
                module=nn.Linear(31, 64),
                module_fqn="model.misaligned",
                context={},
            )

        unsupported_policies = (
            LowPrecisionDtypeScheme(
                weight_format="mxfp4",
                act_format="mxfp8",
            ),
            LowPrecisionDtypeScheme(
                is_fake_quantize=True,
                weight_format="mxfp4",
                act_format="mxfp8",
            ),
        )
        for policy in unsupported_policies:
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(NotImplementedError, "not implemented"):
                    build_linear_strategy(
                        policy,
                        in_features=32,
                        out_features=64,
                    )

        class LinearSubclass(nn.Linear):
            """Distinct type used to verify the exact source contract."""

        with self.assertRaisesRegex(TypeError, "must be exact nn.Linear"):
            replace_linear(
                module=LinearSubclass(32, 64),
                module_fqn="model.subclass",
                context={},
            )
        with self.assertRaisesRegex(NotImplementedError, "pipeline parallelism"):
            replace_linear(
                module=nn.Linear(32, 64),
                module_fqn="model.proj",
                context={"pp": object()},
            )


if __name__ == "__main__":
    unittest.main()
