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
"""NPU alignment tests for high-performance attention modules."""

# Importing Hyper interfaces after importorskip keeps CPU-only collection usable.
# pylint: disable=wrong-import-position

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
    DeepseekV3Config,
)
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3RotaryEmbedding,
    eager_attention_forward as deepseek_eager_attention,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3RotaryEmbedding,
    eager_attention_forward as qwen_eager_attention,
)

pytest.importorskip("torch_npu")

import hyper_parallel.auto_models.modules.dsa_attention as dsa_attention_module
from hyper_parallel.auto_models.modules import (
    DeepseekV32DSAAttention,
    GQAAttention,
    MLAAttention,
    DSAAttention,
)
from tests.common.mark_utils import arg_mark


pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend NPU is required"
)


@pytest.mark.parametrize(
    "module_type",
    (MLAAttention, DeepseekV32DSAAttention, DSAAttention),
)
def test_attention_module_supports_declarative_replacement(module_type: type[nn.Module]) -> None:
    """Expose every Attention replacement through Trainer plan_overrides."""
    assert getattr(module_type, "_hp_module_replacement", False)


def _position_ids(batch_size: int, sequence_length: int) -> torch.Tensor:
    """Build position ids on the active NPU."""
    return torch.arange(sequence_length, device="npu").expand(batch_size, -1)


def _load_converted_parameters(source: nn.Module, replacement: nn.Module) -> None:
    """Simulate the checkpoint conversion performed after module replacement."""
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


def _named_gradients(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    module: nn.Module,
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return input and named parameter gradients for one attention path."""
    parameters = dict(module.named_parameters())
    gradients = torch.autograd.grad(
        output.sum(),
        (hidden_states, *parameters.values()),
        retain_graph=retain_graph,
    )
    return gradients[0], dict(zip(parameters, gradients[1:]))


def _assert_shared_parameter_gradients(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
    *,
    fused_name: str,
    fused_expected: torch.Tensor,
) -> None:
    """Compare a fused replacement gradient and all unchanged parameters."""
    for name, actual_gradient in actual.items():
        expected_gradient = fused_expected if name == fused_name else expected[name]
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=0.0, atol=5e-7
        )


class _DSAIndexer(nn.Module):
    """Minimal source indexer with the dimensions required by the custom op."""

    def __init__(self, hidden_size: int, q_lora_rank: int) -> None:
        """Build the indexer projections."""
        super().__init__()
        self.head_dim = 128
        self.n_heads = 64
        self.index_topk = 2048
        self.wq_b = nn.Linear(q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.weights_proj = nn.Linear(hidden_size, self.n_heads, bias=False)


class _DSASource(nn.Module):
    """Minimal DeepSeek-V3.2-shaped source attention module."""

    def __init__(self) -> None:
        """Build the source projections with supported DSA dimensions."""
        super().__init__()
        hidden_size = 256
        q_lora_rank = 256
        kv_lora_rank = 512
        num_heads = 128
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        value_head_dim = 128
        self.config = SimpleNamespace(
            num_attention_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=value_head_dim,
            index_head_dim=128,
            index_n_heads=64,
            index_topk=2048,
            dsa_loss_coeff=0.01,
            freeze_dsa=False,
        )
        self.layer_idx = 0
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = value_head_dim
        self.scaling = (qk_nope_head_dim + qk_rope_head_dim) ** -0.5
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(q_lora_rank)
        self.q_b_proj = nn.Linear(
            q_lora_rank,
            num_heads * (qk_nope_head_dim + qk_rope_head_dim),
            bias=False,
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + value_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(num_heads * value_head_dim, hidden_size, bias=False)
        self.indexer = _DSAIndexer(hidden_size, q_lora_rank)


class _PanguLinear(nn.Linear):
    """Minimal Pangu linear returning the output and deferred bias."""

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, object]:
        """Apply the Pangu matmul path."""
        output = torch.matmul(hidden_states, self.weight.t().contiguous())
        return output, self.bias


class _PanguDSASource(nn.Module):
    """Minimal source module implementing the supported Pangu DSA contract."""

    def __init__(self, *, use_mome: bool = False, param_sink_number: int = 0) -> None:
        """Build the standard Pangu DSA projection layout."""
        super().__init__()
        source = _DSASource()
        self.config = source.config
        self.layer_number = 1
        self.num_heads = source.num_heads
        self.q_lora_rank = source.q_lora_rank
        self.kv_lora_rank = source.kv_lora_rank
        self.qk_rope_head_dim = source.qk_rope_head_dim
        self.qk_nope_head_dim = source.qk_nope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = source.v_head_dim
        self.index_head_dim = source.indexer.head_dim
        self.num_index_heads = source.indexer.n_heads
        self.index_topk = source.indexer.index_topk
        self.rotary_interleaved = False
        self.dsa_loss_coeff = source.config.dsa_loss_coeff
        self.freeze_dsa = source.config.freeze_dsa
        self.mla_mm_split = False
        self.use_flash_attn = True
        self.use_mome = use_mome
        self.use_fused_mome = use_mome
        self.param_sink_number = param_sink_number
        self.param_sink_scalar = None
        self.apply_FA_rescale = param_sink_number > 0
        self.attention_dropout = nn.Dropout(0.0)
        self.dsa_dense_warm_up = False

        if use_mome:
            window = 3
            output_size = self.num_heads * self.v_head_dim
            self.qa_conv = nn.Conv1d(
                self.q_lora_rank,
                self.q_lora_rank,
                window,
                groups=self.q_lora_rank,
                bias=False,
            )
            self.compresskv_conv = nn.Conv1d(
                self.kv_lora_rank,
                self.kv_lora_rank,
                window,
                groups=self.kv_lora_rank,
                bias=False,
            )
            self.o_conv = nn.Conv1d(
                output_size,
                output_size,
                window,
                groups=output_size,
                bias=False,
            )

        if param_sink_number > 0:
            self.param_sink_k_pe = nn.Parameter(
                torch.empty(param_sink_number, self.qk_rope_head_dim)
            )
            self.param_sink_compressed_kv = nn.Parameter(
                torch.empty(param_sink_number, self.kv_lora_rank)
            )
            nn.init.normal_(self.param_sink_k_pe, std=0.02)
            nn.init.normal_(self.param_sink_compressed_kv, std=0.02)

        self.linear_qkv = _PanguLinear(
            source.q_a_proj.in_features,
            source.q_a_proj.out_features + source.kv_a_proj_with_mqa.out_features,
            bias=False,
        )
        with torch.no_grad():
            self.linear_qkv.weight.copy_(
                torch.cat(
                    (source.q_a_proj.weight, source.kv_a_proj_with_mqa.weight), dim=0
                )
            )
        self.q_layernorm = source.q_a_layernorm
        self.k_layernorm = source.kv_a_layernorm
        self.linear_qb = _PanguLinear(
            source.q_b_proj.in_features, source.q_b_proj.out_features, bias=False
        )
        self.linear_kvb = _PanguLinear(
            source.kv_b_proj.in_features, source.kv_b_proj.out_features, bias=False
        )
        self.linear_proj = _PanguLinear(
            source.o_proj.in_features, source.o_proj.out_features, bias=False
        )
        self.index_linear_qb = _PanguLinear(
            source.indexer.wq_b.in_features,
            source.indexer.wq_b.out_features,
            bias=False,
        )
        self.index_linear_k = _PanguLinear(
            source.indexer.wk.in_features, source.indexer.wk.out_features, bias=False
        )
        self.index_k_layernorm = source.indexer.k_norm
        self.linear_merge_weight = _PanguLinear(
            source.indexer.weights_proj.in_features,
            source.indexer.weights_proj.out_features,
            bias=False,
        )
        with torch.no_grad():
            for target, origin in (
                (self.linear_qb, source.q_b_proj),
                (self.linear_kvb, source.kv_b_proj),
                (self.linear_proj, source.o_proj),
                (self.index_linear_qb, source.indexer.wq_b),
                (self.index_linear_k, source.indexer.wk),
                (self.linear_merge_weight, source.indexer.weights_proj),
            ):
                target.weight.copy_(origin.weight)


class _FusedGQASource(nn.Module):
    """Minimal fused-QKV source used to verify reversible packing."""

    def __init__(self) -> None:
        """Build deterministic fused input and output projections."""
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=8,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        self.head_dim = 2
        self.qkv_proj = nn.Linear(8, 16, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            values = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
            self.qkv_proj.weight.copy_(values)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_gqa_module_matches_qwen3_eager_attention() -> None:
    """The grouped projection and RoPE path must preserve Qwen3 behavior."""
    config = Qwen3Config(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    source = Qwen3Attention(config, 0).to(device="npu", dtype=torch.float32).eval()
    replacement = GQAAttention(
        module=source, attention_interface=qwen_eager_attention
    ).eval()
    _load_converted_parameters(source, replacement)
    rotary = Qwen3RotaryEmbedding(config).to("npu")
    hidden_states = torch.randn(
        2, 5, config.hidden_size, device="npu", requires_grad=True
    )
    position_embeddings = rotary(hidden_states, _position_ids(2, 5))

    expected = source(hidden_states, position_embeddings, None)[0]
    actual = replacement(hidden_states, position_embeddings, None)[0]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    expected_grad, expected_parameter_grads = _named_gradients(
        expected, hidden_states, source, retain_graph=True
    )
    actual_grad, actual_parameter_grads = _named_gradients(
        actual, hidden_states, replacement, retain_graph=False
    )
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0.0, atol=5e-7)
    pack = replacement.make_transforms()[0].operations[0]
    expected_packed = pack.convert(
        {
            name: expected_parameter_grads[name]
            for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight")
        },
        ["q_proj.weight", "k_proj.weight", "v_proj.weight"],
        ["linear_qkv.weight"],
    )["linear_qkv.weight"]
    _assert_shared_parameter_gradients(
        expected_parameter_grads,
        actual_parameter_grads,
        fused_name="linear_qkv.weight",
        fused_expected=expected_packed,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_fused_gqa_weight_conversion_loads_target_and_reverses() -> None:
    """Fused source checkpoints must load and restore the grouped layout exactly."""
    source = _FusedGQASource()
    replacement = GQAAttention(module=source)
    transform = replacement.make_transforms()[0]
    operation = transform.operations[0]
    converted = operation.convert(
        {"qkv_proj.weight": source.qkv_proj.weight.detach()},
        ["qkv_proj.weight"],
        ["linear_qkv.weight"],
    )["linear_qkv.weight"]
    _load_converted_parameters(source, replacement)
    torch.testing.assert_close(
        converted, replacement.linear_qkv.weight, rtol=0.0, atol=0.0
    )

    restored = operation.reverse_op.convert(
        {"linear_qkv.weight": converted},
        ["linear_qkv.weight"],
        ["qkv_proj.weight"],
    )["qkv_proj.weight"]
    torch.testing.assert_close(
        restored, source.qkv_proj.weight.detach(), rtol=0.0, atol=0.0
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_mla_module_matches_deepseek_v3_eager_attention() -> None:
    """The fused latent projection and interleaved RoPE must preserve MLA behavior."""
    config = DeepseekV3Config(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        intermediate_size=128,
        n_routed_experts=4,
        num_experts_per_tok=2,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    source = DeepseekV3Attention(config, 0).to(
        device="npu", dtype=torch.float32
    ).eval()
    replacement = MLAAttention(
        module=source, attention_interface=deepseek_eager_attention
    ).eval()
    _load_converted_parameters(source, replacement)
    rotary = DeepseekV3RotaryEmbedding(config).to("npu")
    hidden_states = torch.randn(
        2, 5, config.hidden_size, device="npu", requires_grad=True
    )
    position_embeddings = rotary(hidden_states, _position_ids(2, 5))

    expected = source(hidden_states, position_embeddings, None)[0]
    actual = replacement(hidden_states, position_embeddings, None)[0]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    expected_grad, expected_parameter_grads = _named_gradients(
        expected, hidden_states, source, retain_graph=True
    )
    actual_grad, actual_parameter_grads = _named_gradients(
        actual, hidden_states, replacement, retain_graph=False
    )
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0.0, atol=5e-7)
    concatenate = replacement.make_transforms()[0].operations[0]
    expected_fused = concatenate.convert(
        {
            name: expected_parameter_grads[name]
            for name in ("q_a_proj.weight", "kv_a_proj_with_mqa.weight")
        },
        ["q_a_proj.weight", "kv_a_proj_with_mqa.weight"],
        ["linear_qkv.weight"],
    )["linear_qkv.weight"]
    _assert_shared_parameter_gradients(
        expected_parameter_grads,
        actual_parameter_grads,
        fused_name="linear_qkv.weight",
        fused_expected=expected_fused,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_deepseek_v32_attention_runs_real_custom_ops_forward_and_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the complete DSA indexer and sparse-attention custom-op chain."""
    torch.manual_seed(3)
    source = _DSASource().to(device="npu", dtype=torch.bfloat16).train()
    source.scaling *= 1.25
    replacement = DeepseekV32DSAAttention(module=source).train()
    _load_converted_parameters(source, replacement)
    sparse_attention = Mock(wraps=dsa_attention_module.dsa_sparse_attention)
    kl_loss = Mock(wraps=dsa_attention_module.dsa_kl_loss)
    monkeypatch.setattr(dsa_attention_module, "dsa_sparse_attention", sparse_attention)
    monkeypatch.setattr(dsa_attention_module, "dsa_kl_loss", kl_loss)
    index_norm_calls = []
    replacement.indexer.k_norm.register_forward_hook(
        lambda _module, _inputs, _output: index_norm_calls.append(None)
    )
    sequence_length = 2048
    hidden_states = torch.randn(
        1,
        sequence_length,
        256,
        device="npu",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    frequencies = torch.randn(
        1, sequence_length, 32, device="npu", dtype=torch.float32
    )
    cos = torch.cat((frequencies.cos(), frequencies.cos()), dim=-1).to(
        torch.bfloat16
    )
    sin = torch.cat((frequencies.sin(), frequencies.sin()), dim=-1).to(
        torch.bfloat16
    )
    packed_length = sequence_length // 2
    position_ids = torch.arange(packed_length, device="npu").repeat(2).unsqueeze(0)
    sequence_ids = torch.arange(2, device="npu").repeat_interleave(packed_length)
    allowed = torch.ones(
        (sequence_length, sequence_length), dtype=torch.bool, device="npu"
    ).tril_()
    allowed &= sequence_ids.unsqueeze(-1) == sequence_ids.unsqueeze(-2)
    attention_mask = torch.where(
        allowed.unsqueeze(0).unsqueeze(0),
        torch.tensor(0.0, device="npu", dtype=torch.bfloat16),
        torch.tensor(
            torch.finfo(torch.bfloat16).min,
            device="npu",
            dtype=torch.bfloat16,
        ),
    )

    output, attention_weights = replacement(
        hidden_states,
        (cos, sin),
        attention_mask,
        position_ids=position_ids,
        actual_seq_len=torch.tensor(
            [packed_length, sequence_length], device="npu", dtype=torch.int32
        ),
    )
    assert output.shape == hidden_states.shape
    assert attention_weights is None
    assert replacement.scaling == source.scaling
    assert sparse_attention.call_args.args[5] == source.scaling
    assert kl_loss.call_args.args[12] == source.scaling
    assert len(index_norm_calls) == 1
    assert torch.isfinite(output).all()

    output.float().mean().backward()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in replacement.parameters()
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential",
)
def test_deepseek_v32_attention_builds_default_batch_boundaries() -> None:
    """Build one cumulative sequence boundary for each unpacked batch item."""
    actual_seq_len = DeepseekV32DSAAttention._get_actual_seq_len(
        None, 2, 3, torch.device("npu")
    )

    torch.testing.assert_close(
        actual_seq_len,
        torch.tensor([3, 6], dtype=torch.int32, device="npu"),
        rtol=0.0,
        atol=0.0,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_deepseek_v32_attention_rejects_nonzero_dropout() -> None:
    """Reject configurations that the sparse operator cannot reproduce."""
    source = _DSASource()
    source.attention_dropout = 0.1

    with pytest.raises(ValueError, match="requires attention_dropout=0"):
        DeepseekV32DSAAttention(module=source)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_dsa_attention_runs_real_custom_ops_forward_and_backward() -> None:
    """Exercise standard DSA with the real custom operators."""
    torch.manual_seed(4)
    source = _PanguDSASource().to(device="npu", dtype=torch.bfloat16).train()
    replacement = DSAAttention(module=source).train()
    assert not hasattr(replacement, "make_transforms")
    probe = torch.empty((1, 1, 256), device="npu", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="does not consume attention_mask"):
        replacement(probe, attention_mask=torch.ones(1, 1, device="npu"))
    with pytest.raises(NotImplementedError, match="does not support KV reuse"):
        replacement(probe, kv_reuse_states=object())
    index_norm_calls = []
    replacement.index_k_layernorm.register_forward_hook(
        lambda _module, _inputs, _output: index_norm_calls.append(None)
    )

    sequence_length = 2048
    hidden_states = torch.randn(
        1,
        sequence_length,
        256,
        device="npu",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    frequencies = torch.randn(
        1, sequence_length, 32, device="npu", dtype=torch.float32
    )
    cos = torch.cat((frequencies.cos(), frequencies.cos()), dim=-1).to(
        torch.bfloat16
    )
    sin = torch.cat((frequencies.sin(), frequencies.sin()), dim=-1).to(
        torch.bfloat16
    )

    output, output_bias = replacement(
        hidden_states,
        (cos, sin),
        None,
        actual_seq_len=[sequence_length],
        return_bias=True,
    )
    assert output.shape == hidden_states.shape
    assert output_bias is None
    assert len(index_norm_calls) == 1
    assert torch.isfinite(output).all()

    output.float().mean().backward()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in replacement.parameters()
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_dsa_mome_and_parameter_sink_forward_and_backward() -> None:
    """Exercise the MOME and rescaled parameter-sink paths used by 92B Pangu."""
    torch.manual_seed(5)
    source = _PanguDSASource(use_mome=True, param_sink_number=128).to(
        device="npu", dtype=torch.bfloat16
    ).train()
    replacement = DSAAttention(module=source).train()

    sequence_length = 2048
    hidden_states = torch.randn(
        1,
        sequence_length,
        256,
        device="npu",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    frequencies = torch.randn(
        1, sequence_length, 32, device="npu", dtype=torch.float32
    )
    cos = torch.cat((frequencies.cos(), frequencies.cos()), dim=-1).to(
        torch.bfloat16
    )
    sin = torch.cat((frequencies.sin(), frequencies.sin()), dim=-1).to(
        torch.bfloat16
    )
    mome_mask = torch.ones(
        1, sequence_length, device="npu", dtype=torch.bool
    )

    output, output_bias = replacement(
        hidden_states,
        (cos, sin),
        None,
        actual_seq_len=[sequence_length],
        return_bias=True,
        mome_mask=mome_mask,
    )
    assert output.shape == hidden_states.shape
    assert output_bias is None
    assert torch.isfinite(output).all()

    output.float().mean().backward()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in replacement.parameters()
    )
