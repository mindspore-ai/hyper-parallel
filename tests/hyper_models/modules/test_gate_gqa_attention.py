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
"""NPU alignment tests for gated-query grouped-query attention."""

# pylint: disable=wrong-import-position

import pytest
import torch
from torch import nn

configuration = pytest.importorskip(
    "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe"
)
modeling = pytest.importorskip(
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
)
pytest.importorskip("torch_npu")

from hyper_models.modules import GatedGQAAttention
from tests.common.mark_utils import arg_mark


pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend NPU is required"
)


def _load_converted_parameters(source: nn.Module, replacement: nn.Module) -> None:
    """Load source Q/gate, K, and V parameters through declared transforms."""
    source_parameters = dict(source.named_parameters())
    target_parameters = dict(replacement.named_parameters())
    with torch.no_grad():
        for transform in replacement.make_transforms():
            converted = {
                pattern: source_parameters[pattern]
                for pattern in transform.source_patterns
            }
            for operation in transform.operations:
                converted = operation.convert(
                    converted,
                    transform.source_patterns,
                    transform.target_patterns,
                )
            for name, value in converted.items():
                target_parameters[name].copy_(value)


def _tiny_config():
    """Build a one-layer Qwen3.5-MoE text configuration for attention tests."""
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
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return config


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gated_gqa_matches_qwen35_moe_attention(dtype: torch.dtype) -> None:
    """Match the Transformers attention module without constructing a model."""
    torch.manual_seed(7)
    config = _tiny_config()
    source = modeling.Qwen3_5MoeAttention(config, 0).to(
        device="npu", dtype=dtype
    ).eval()
    replacement = GatedGQAAttention(
        module=source,
        attention_interface=modeling.eager_attention_forward,
    ).eval()
    _load_converted_parameters(source, replacement)
    rotary = modeling.Qwen3_5MoeTextRotaryEmbedding(config).to("npu")
    hidden_states = torch.randn(
        2, 5, config.hidden_size, device="npu", dtype=dtype, requires_grad=True
    )
    position_ids = torch.arange(5, device="npu").expand(2, -1)
    position_embeddings = rotary(hidden_states, position_ids)
    attention_mask = torch.full(
        (1, 1, 5, 5),
        torch.finfo(dtype).min,
        device="npu",
        dtype=dtype,
    ).triu(diagonal=1)

    expected = source(hidden_states, position_embeddings, attention_mask)[0]
    actual = replacement(hidden_states, position_embeddings, attention_mask)[0]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    if dtype == torch.float32:
        expected_input_grad = torch.autograd.grad(
            expected.sum(), hidden_states, retain_graph=True
        )[0]
        actual_input_grad = torch.autograd.grad(actual.sum(), hidden_states)[0]
        torch.testing.assert_close(
            actual_input_grad, expected_input_grad, rtol=0.0, atol=5e-7
        )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_gated_gqa_runs_forward_and_backward_with_npu_fusion_attention() -> None:
    """The fused NPU backend must preserve output accuracy and support training."""
    torch.manual_seed(11)
    config = _tiny_config()
    source = modeling.Qwen3_5MoeAttention(config, 0).to(
        device="npu", dtype=torch.bfloat16
    ).eval()
    replacement = GatedGQAAttention(module=source).eval()
    _load_converted_parameters(source, replacement)
    rotary = modeling.Qwen3_5MoeTextRotaryEmbedding(config).to("npu")
    hidden_states = torch.randn(
        2,
        5,
        config.hidden_size,
        device="npu",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    position_ids = torch.arange(5, device="npu").expand(2, -1)
    position_embeddings = rotary(hidden_states, position_ids)
    attention_mask = torch.full(
        (1, 1, 5, 5),
        torch.finfo(torch.bfloat16).min,
        device="npu",
        dtype=torch.bfloat16,
    ).triu(diagonal=1)

    expected = source(hidden_states, position_embeddings, attention_mask)[0]
    actual = replacement(hidden_states, position_embeddings, attention_mask)[0]

    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=2e-3)
    actual.float().square().mean().backward()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert all(
        parameter.grad is not None
        for parameter in replacement.parameters()
        if parameter.requires_grad
    )
