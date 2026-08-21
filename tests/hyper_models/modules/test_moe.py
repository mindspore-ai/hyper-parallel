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
"""NPU alignment tests for high-performance MoE modules."""

# Importing Hyper interfaces after importorskip keeps CPU-only collection usable.
# pylint: disable=wrong-import-position

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

configuration = pytest.importorskip("transformers.models.qwen3_5_moe.configuration_qwen3_5_moe")
modeling = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
pytest.importorskip("torch_npu")

from transformers.core_model_loading import WeightConverter, WeightRenaming

from hyper_models.components.model_transform import (
    ModuleReplacementSpec,
    apply_module_replacements,
    compile_module_replacements,
)
from hyper_models.modules import GroupedExperts, SharedExpert
from tests.common.mark_utils import arg_mark


pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required")


def _tiny_config():
    """Build a one-layer Qwen3.5-MoE configuration for expert tests."""
    config = configuration.Qwen3_5MoeTextConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["full_attention"],
    )
    config._experts_implementation = "eager"
    return config


def _convert_transform(
    source_tensors: dict[str, torch.Tensor],
    transform: WeightRenaming | WeightConverter,
) -> dict[str, torch.Tensor]:
    """Apply one declared checkpoint transform to named tensors."""
    if isinstance(transform, WeightRenaming):
        return {transform.target_patterns[0]: source_tensors[transform.source_patterns[0]]}

    converted = {pattern: source_tensors[pattern] for pattern in transform.source_patterns}
    for operation in transform.operations:
        converted = operation.convert(
            converted,
            transform.source_patterns,
            transform.target_patterns,
        )
    return converted


def _convert_named_tensors(
    source_tensors: dict[str, torch.Tensor],
    transforms: list[WeightRenaming | WeightConverter],
) -> dict[str, torch.Tensor]:
    """Apply all declared transforms and collect tensors by target name."""
    converted = {}
    for transform in transforms:
        converted.update(_convert_transform(source_tensors, transform))
    return converted


def _load_converted_parameters(source: nn.Module, replacement: nn.Module) -> None:
    """Simulate checkpoint conversion after replacing an MoE module."""
    source_parameters = dict(source.named_parameters())
    target_parameters = dict(replacement.named_parameters())
    converted = _convert_named_tensors(
        source_parameters,
        replacement.make_transforms(),
    )
    with torch.no_grad():
        for name, value in converted.items():
            target_parameters[name].copy_(value)


def _named_gradients(
    output: torch.Tensor,
    inputs: tuple[torch.Tensor, ...],
    module: nn.Module,
) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
    """Return input and named parameter gradients for one MoE path."""
    parameters = dict(module.named_parameters())
    gradients = torch.autograd.grad(
        output.float().sum(),
        (*inputs, *parameters.values()),
    )
    input_count = len(inputs)
    return (
        gradients[:input_count],
        dict(zip(parameters, gradients[input_count:])),
    )


def _assert_converted_parameter_gradients(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
    replacement: nn.Module,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Compare target gradients with transformed source gradients."""
    converted = _convert_named_tensors(expected, replacement.make_transforms())
    assert set(actual) == set(converted)
    for name, actual_gradient in actual.items():
        torch.testing.assert_close(
            actual_gradient,
            converted[name],
            rtol=rtol,
            atol=atol,
        )


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """Return tolerances for grouped NPU kernels and fused SwiGLU."""
    if dtype == torch.float32:
        return 1e-3, 1e-3
    return 1e-2, 1e-2


class _PackedExpertsSource(nn.Module):
    """Minimal Transformers-shaped packed experts with deterministic weights."""

    def __init__(self) -> None:
        """Build non-transposed expert projections."""
        super().__init__()
        self.config = SimpleNamespace(
            hidden_act="silu",
            use_fused_swiglu=True,
        )
        self.num_experts = 2
        self.hidden_dim = 4
        self.intermediate_dim = 3
        self.has_gate = True
        self.has_bias = False
        self.is_transposed = False
        self.is_concatenated = True
        self.act_fn = F.silu
        self.gate_up_proj = nn.Parameter(torch.arange(48, dtype=torch.float32).reshape(2, 6, 4))
        self.down_proj = nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(2, 4, 3))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Provide the Transformers Experts signature for replacement checks."""
        del top_k_index, top_k_weights
        return hidden_states


class _SeparateSharedSource(nn.Module):
    """Minimal separate shared expert with deterministic biased projections."""

    def __init__(self) -> None:
        """Build separate gate, up, and down projections."""
        super().__init__()
        self.config = SimpleNamespace(
            hidden_act="silu",
            use_fused_swiglu=True,
        )
        self.gate_proj = nn.Linear(4, 3, bias=True)
        self.up_proj = nn.Linear(4, 3, bias=True)
        self.down_proj = nn.Linear(3, 4, bias=True)
        self.act_fn = F.silu
        with torch.no_grad():
            offset = 0
            for parameter in self.parameters():
                values = torch.arange(
                    offset,
                    offset + parameter.numel(),
                    dtype=parameter.dtype,
                ).reshape_as(parameter)
                parameter.copy_(values)
                offset += parameter.numel()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the separate shared-expert formula."""
        gate = self.act_fn(self.gate_proj(hidden_states))
        return self.down_proj(gate * self.up_proj(hidden_states))


class _DeferredBiasLinear(nn.Linear):
    """Linear layer returning output and deferred bias separately."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """Build a bias-enabled deferred-bias projection."""
        super().__init__(in_features, out_features, bias=True)
        self.skip_bias_add = True

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the matrix multiplication result and bias."""
        return torch.matmul(hidden_states, self.weight.t().contiguous()), self.bias


def _silu_gate(hidden_states: torch.Tensor) -> torch.Tensor:
    """Apply the standard SwiGLU formula."""
    gate, up = hidden_states.chunk(2, dim=-1)
    return F.silu(gate) * up


class _FusedSharedSource(nn.Module):
    """Minimal fused shared expert using deferred-bias projections."""

    def __init__(self) -> None:
        """Build the fused shared-expert layout."""
        super().__init__()
        self.config = SimpleNamespace()
        self.linear_fc1 = _DeferredBiasLinear(4, 6)
        self.linear_fc2 = _DeferredBiasLinear(3, 4)
        self.activation_func = _silu_gate

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the fused shared-expert path."""
        intermediate, bias = self.linear_fc1(hidden_states)
        intermediate = self.activation_func(intermediate + bias)
        return self.linear_fc2(intermediate)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_experts_matches_qwen35_moe_experts(dtype: torch.dtype) -> None:
    """Match Transformers experts forward, input gradients, and packed gradients."""
    torch.manual_seed(7)
    config = _tiny_config()
    source = modeling.Qwen3_5MoeExperts(config).to(device="npu", dtype=dtype).eval()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.normal_(mean=0.0, std=config.initializer_range)
    replacement = GroupedExperts(module=source).eval()
    _load_converted_parameters(source, replacement)

    expected_input = torch.randn(8, config.hidden_size, device="npu", dtype=dtype, requires_grad=True)
    actual_input = expected_input.detach().clone().requires_grad_(True)
    top_k_index = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3], [2, 0], [3, 1]],
        device="npu",
        dtype=torch.long,
    )
    routing_weights = torch.softmax(
        torch.randn(8, 2, device="npu", dtype=torch.float32),
        dim=-1,
    ).to(dtype)
    expected_weights = routing_weights.detach().clone().requires_grad_(True)
    actual_weights = routing_weights.detach().clone().requires_grad_(True)

    expected = source(expected_input, top_k_index, expected_weights)
    actual = replacement(actual_input, top_k_index, actual_weights)
    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    expected_input_grads, expected_parameter_grads = _named_gradients(
        expected,
        (expected_input, expected_weights),
        source,
    )
    actual_input_grads, actual_parameter_grads = _named_gradients(
        actual,
        (actual_input, actual_weights),
        replacement,
    )
    for actual_gradient, expected_gradient in zip(
        actual_input_grads,
        expected_input_grads,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=rtol,
            atol=atol,
        )
    _assert_converted_parameter_gradients(
        expected_parameter_grads,
        actual_parameter_grads,
        replacement,
        rtol=rtol,
        atol=atol,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_shared_expert_matches_qwen35_moe_mlp(dtype: torch.dtype) -> None:
    """Match the Transformers shared MLP forward and transformed gradients."""
    torch.manual_seed(11)
    config = _tiny_config()
    source = (
        modeling.Qwen3_5MoeMLP(
            config,
            intermediate_size=config.shared_expert_intermediate_size,
        )
        .to(device="npu", dtype=dtype)
        .eval()
    )
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.normal_(mean=0, std=config.initializer_range)
    replacement = SharedExpert(module=source).eval()
    _load_converted_parameters(source, replacement)

    expected_input = torch.randn(
        2,
        5,
        config.hidden_size,
        device="npu",
        dtype=dtype,
        requires_grad=True,
    )
    actual_input = expected_input.detach().clone().requires_grad_(True)
    expected = source(expected_input)
    actual = replacement(actual_input)
    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    expected_input_grads, expected_parameter_grads = _named_gradients(
        expected,
        (expected_input,),
        source,
    )
    actual_input_grads, actual_parameter_grads = _named_gradients(
        actual,
        (actual_input,),
        replacement,
    )
    torch.testing.assert_close(
        actual_input_grads[0],
        expected_input_grads[0],
        rtol=rtol,
        atol=atol,
    )
    _assert_converted_parameter_gradients(
        expected_parameter_grads,
        actual_parameter_grads,
        replacement,
        rtol=rtol,
        atol=atol,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_grouped_experts_weight_conversion_loads_target_and_reverses() -> None:
    """Packed expert projections must transpose and restore exactly."""
    source = _PackedExpertsSource()
    replacement = GroupedExperts(module=source)
    _load_converted_parameters(source, replacement)
    source_parameters = dict(source.named_parameters())
    target_parameters = dict(replacement.named_parameters())

    for transform in replacement.make_transforms():
        converted = _convert_transform(source_parameters, transform)
        target_name = transform.target_patterns[0]
        source_name = transform.source_patterns[0]
        torch.testing.assert_close(
            target_parameters[target_name],
            converted[target_name],
            rtol=0.0,
            atol=0.0,
        )
        restored = transform.operations[0].reverse_op.convert(
            {target_name: converted[target_name]},
            transform.target_patterns,
            transform.source_patterns,
        )
        torch.testing.assert_close(
            restored[source_name],
            source_parameters[source_name],
            rtol=0.0,
            atol=0.0,
        )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_shared_expert_weight_conversion_loads_target_and_reverses() -> None:
    """Separate shared projections must load and restore their original layout."""
    source = _SeparateSharedSource()
    replacement = SharedExpert(module=source)
    _load_converted_parameters(source, replacement)

    torch.testing.assert_close(
        replacement.linear_fc1.weight,
        torch.cat((source.gate_proj.weight, source.up_proj.weight), dim=0),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        replacement.linear_fc1.bias,
        torch.cat((source.gate_proj.bias, source.up_proj.bias), dim=0),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        replacement.linear_fc2.weight,
        source.down_proj.weight,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        replacement.linear_fc2.bias,
        source.down_proj.bias,
        rtol=0.0,
        atol=0.0,
    )

    source_parameters = dict(source.named_parameters())
    for transform in replacement.make_transforms():
        if not isinstance(transform, WeightConverter):
            reverse = transform.reverse_transform()
            assert reverse.source_patterns == transform.target_patterns
            assert reverse.target_patterns == transform.source_patterns
            continue
        converted = _convert_transform(source_parameters, transform)
        restored = transform.operations[0].reverse_op.convert(
            converted,
            transform.target_patterns,
            transform.source_patterns,
        )
        for source_name in transform.source_patterns:
            torch.testing.assert_close(
                restored[source_name],
                source_parameters[source_name],
                rtol=0.0,
                atol=0.0,
            )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_grouped_experts_replacement_registers_scoped_transforms() -> None:
    """Generic replacement must collect scoped expert weight transforms."""
    model = nn.Sequential(_PackedExpertsSource())
    spec = ModuleReplacementSpec(
        match=("0",),
        factory=GroupedExperts,
        module_type=_PackedExpertsSource,
        exact_type=True,
    )
    plan = compile_module_replacements(model, [spec])
    weights_mapping = []
    model, weights_mapping = apply_module_replacements(
        model,
        plan,
        weights_mapping=weights_mapping,
    )

    assert isinstance(model[0], GroupedExperts)
    assert len(weights_mapping) == 2
    assert all(transform.scope_prefix == "0" for transform in weights_mapping)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_grouped_experts_rejects_bias() -> None:
    """Declared expert bias must fail before unsupported computation is used."""
    source = _PackedExpertsSource()
    source.has_bias = True
    with pytest.raises(ValueError, match="has_bias=True"):
        GroupedExperts(module=source)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_grouped_experts_rejects_custom_apply_gate() -> None:
    """Custom source gating must require a dedicated replacement."""
    source = _PackedExpertsSource()

    def custom_apply_gate(gate_up_output: torch.Tensor) -> torch.Tensor:
        """Represent a source-specific gate implementation."""
        return gate_up_output

    source._apply_gate = custom_apply_gate
    with pytest.raises(ValueError, match="custom _apply_gate"):
        GroupedExperts(module=source)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_fused_shared_expert_reuses_source_projections() -> None:
    """Fused shared experts must retain projections without checkpoint transforms."""
    source = _FusedSharedSource().to("npu").eval()
    model = nn.Sequential(source)
    spec = ModuleReplacementSpec(
        match=("0",),
        factory=SharedExpert,
        module_type=_FusedSharedSource,
        exact_type=True,
    )
    plan = compile_module_replacements(model, [spec])
    model, _ = apply_module_replacements(model, plan)
    replacement = model[0]

    assert isinstance(replacement, SharedExpert)
    assert replacement.linear_fc1 is source.linear_fc1
    assert replacement.linear_fc2 is source.linear_fc2
    assert not replacement.make_transforms()

    expected_input = torch.randn(2, 4, device="npu", requires_grad=True)
    actual_input = expected_input.detach().clone().requires_grad_(True)
    expected = source(expected_input)
    actual = replacement(actual_input)
    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)
