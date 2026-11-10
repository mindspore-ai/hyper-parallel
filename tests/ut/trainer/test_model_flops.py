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
"""Unit tests for PaLM model-FLOPs accounting."""

from fractions import Fraction
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.runtime.model_flops import (
    AttentionLayer,
    DeltaRuleLayer,
    ModelFlopsProfile,
    attention_layers_from_config,
    count_matmul_params,
)
from tests.common.mark_utils import arg_mark


class _Norm(nn.Module):
    """Hold a one-dimensional scale like an RMSNorm."""

    def __init__(self, dim: int) -> None:
        """Allocate the scale vector."""
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))


class _StackedExperts(nn.Module):
    """Hold routed experts as stacked tensors like a grouped-matmul module."""

    def __init__(self, num_experts: int, hidden: int, intermediate: int) -> None:
        """Allocate stacked gate-up and down projections."""
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, hidden, 2 * intermediate))
        self.down_proj = nn.Parameter(torch.empty(num_experts, intermediate, hidden))


class _Attention(nn.Module):
    """Grouped-query attention projections with per-head norms."""

    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int) -> None:
        """Allocate the four projections and the per-head norms."""
        super().__init__()
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.q_norm = _Norm(head_dim)
        self.k_norm = _Norm(head_dim)


class _MoeBlock(nn.Module):
    """Router plus stacked routed experts."""

    def __init__(self, hidden: int, num_experts: int, intermediate: int) -> None:
        """Allocate the router and the routed experts."""
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = _StackedExperts(num_experts, hidden, intermediate)


class _Layer(nn.Module):
    """One MoE decoder layer."""

    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, num_experts: int,
                 intermediate: int) -> None:
        """Allocate attention, the MoE block and both norms."""
        super().__init__()
        self.self_attn = _Attention(hidden, heads, kv_heads, head_dim)
        self.mlp = _MoeBlock(hidden, num_experts, intermediate)
        self.input_layernorm = _Norm(hidden)
        self.post_attention_layernorm = _Norm(hidden)


class _Decoder(nn.Module):
    """Embedding, decoder layers and final norm."""

    def __init__(self, vocab: int, hidden: int, layers: list[nn.Module]) -> None:
        """Allocate the embedding table, layers and final norm."""
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(layers)
        self.norm = _Norm(hidden)


class _CausalLM(nn.Module):
    """Causal LM with an untied or tied output head."""

    def __init__(self, decoder: _Decoder, vocab: int, hidden: int, tie: bool = False) -> None:
        """Allocate the output head, tying it to the embedding when requested."""
        super().__init__()
        self.model = decoder
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        if tie:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_output_embeddings(self) -> nn.Module:
        """Return the output projection as Hugging Face models do."""
        return self.lm_head


def _qwen3_30b_a3b() -> tuple[nn.Module, SimpleNamespace]:
    """Build Qwen3-30B-A3B parameter shapes on the meta device."""
    config = SimpleNamespace(
        num_hidden_layers=48, num_attention_heads=32, num_key_value_heads=4, head_dim=128,
        hidden_size=2048, num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768,
        vocab_size=151936,
    )
    with torch.device("meta"):
        layers = [_Layer(2048, 32, 4, 128, 128, 768) for _ in range(48)]
        model = _CausalLM(_Decoder(151936, 2048, layers), 151936, 2048)
    return model, config


class _DenseLayer(nn.Module):
    """Small dense layer whose MLP may hold per-expert modules."""

    def __init__(self, hidden: int, mlp: nn.Module) -> None:
        """Allocate attention next to the supplied MLP."""
        super().__init__()
        self.self_attn = _Attention(hidden, 2, 2, hidden // 2)
        self.mlp = mlp


def _config_model(width: int) -> None:  # pylint: disable=unused-argument
    """Model target for the configuration-parsing case."""


def _config_optimizer(learning_rate: float) -> None:  # pylint: disable=unused-argument
    """Optimizer target for the configuration-parsing case."""


_CONFIG_YAML = """
model:
  _target_: {module}._config_model
  width: 8
optimizer:
  _target_: {module}._config_optimizer
  learning_rate: 0.01
"""


class TestMfuConfig(unittest.TestCase):
    """Verify the training.mfu configuration section."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_mfu_section_defaults_and_overrides(self) -> None:
        """Keep MFU reporting off by default and coerce its overrides.

        Feature: PaLM MFU configuration.
        Description: Parse a training YAML without and with training.mfu overrides.
        Expectation: Defaults disable reporting without an assumed peak; overrides set both fields.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "train.yaml"
            yaml_path.write_text(_CONFIG_YAML.format(module=__name__), encoding="utf-8")
            default = parse_training_args([str(yaml_path)]).training.mfu
            enabled = parse_training_args([
                str(yaml_path), "--training.mfu.enabled=true", "--training.mfu.peak_tflops_per_device=400",
                "--training.mfu.hfu_extra_tflops_per_step=12.5", "--training.mfu.target_mfu_percent=16",
                "--training.mfu.target_tokens_per_s_per_device=5200", "--training.mfu.target_step_time_s=2.1",
            ]).training.mfu
        self.assertFalse(default.enabled, f"case: off_by_default, got={default.enabled!r}")
        self.assertIsNone(default.peak_tflops_per_device, f"case: no_assumed_peak, got={default!r}")
        self.assertIsNone(default.target_mfu_percent, f"case: no_default_target, got={default!r}")
        self.assertIsNone(default.hfu_extra_tflops_per_step, f"case: no_default_extra, got={default!r}")
        self.assertTrue(enabled.enabled, f"case: enabled_override, got={enabled.enabled!r}")
        self.assertEqual(enabled.peak_tflops_per_device, 400.0, f"case: peak_override, got={enabled!r}")
        self.assertEqual(enabled.hfu_extra_tflops_per_step, 12.5, f"case: extra_override, got={enabled!r}")
        self.assertEqual(enabled.target_mfu_percent, 16.0, f"case: target_mfu, got={enabled!r}")
        self.assertEqual(enabled.target_tokens_per_s_per_device, 5200.0, f"case: target_tps, got={enabled!r}")
        self.assertEqual(enabled.target_step_time_s, 2.1, f"case: target_step_time, got={enabled!r}")


class TestModelFlops(unittest.TestCase):
    """Verify the PaLM per-token FLOPs inputs and formula."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_qwen3_30b_a3b_matches_palm_hand_count(self) -> None:
        """Reproduce the hand-counted PaLM model FLOPs of Qwen3-30B-A3B.

        Feature: PaLM model-FLOPs accounting.
        Description: Count a meta-device model with the released Qwen3-30B-A3B shapes.
        Expectation: N excludes the embedding table and norms and weights routed experts by 8/128;
            per-token FLOPs, recompute FLOPs and the resulting MFU match the hand count.
        """
        model, config = _qwen3_30b_a3b()
        count = count_matmul_params([model], config)
        self.assertEqual(
            count.matmul_params, 3_041_656_832,
            f"case: active_matmul_params, got={count.matmul_params}")
        self.assertEqual(
            count.output_head_params, 311_164_928,
            f"case: output_head_params, got={count.output_head_params}")
        self.assertEqual(count.unsupported, (), f"case: supported_model, got={count.unsupported}")

        layers, reason = attention_layers_from_config(config)
        self.assertIsNone(reason, f"case: attention_supported, got={reason!r}")
        profile = ModelFlopsProfile(float(count.matmul_params), float(count.output_head_params), layers)
        self.assertEqual(
            profile.flops_per_token(4096), 27_913_617_408,
            f"case: palm_flops_per_token, got={profile.flops_per_token(4096)}")
        self.assertEqual(
            profile.recompute_flops_per_token(4096), 8_682_209_280,
            f"case: recompute_flops_per_token, got={profile.recompute_flops_per_token(4096)}")
        mfu = profile.flops_per_token(4096) * 14095.938232982639 / 8 / 400e12 * 100
        self.assertAlmostEqual(mfu, 12.296, places=3, msg=f"case: audited_run_mfu, got={mfu}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_embedding_counts_only_when_tied_to_output_head(self) -> None:
        """Count an embedding table only when it is also the output projection.

        Feature: PaLM model-FLOPs accounting.
        Description: Count the same small model with an untied and a tied output head.
        Expectation: The untied embedding lookup is excluded; the tied weight is counted once as the head.
        """
        def build(tie: bool) -> nn.Module:
            """Build the small model with or without a tied output head."""
            layer = _DenseLayer(4, nn.Identity())
            return _CausalLM(_Decoder(10, 4, [layer]), 10, 4, tie=tie)

        config = SimpleNamespace()
        attention_params = 4 * (4 * 4)
        untied = count_matmul_params([build(tie=False)], config)
        tied = count_matmul_params([build(tie=True)], config)
        for case, count in (("untied", untied), ("tied", tied)):
            self.assertEqual(
                count.matmul_params, attention_params + 40,
                f"case: {case}_matmul_params, got={count.matmul_params}")
            self.assertEqual(
                count.output_head_params, 40,
                f"case: {case}_output_head_params, got={count.output_head_params}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_per_expert_modules_and_shared_experts(self) -> None:
        """Weight per-expert module lists by top-k while keeping shared experts whole.

        Feature: PaLM model-FLOPs accounting.
        Description: Count routed experts stored as a module list next to a shared expert.
        Expectation: Routed expert parameters are scaled by top_k / n_routed_experts; shared ones are not.
        """
        mlp = nn.Module()
        mlp.experts = nn.ModuleList(nn.Linear(4, 4, bias=False) for _ in range(4))
        mlp.shared_experts = nn.Linear(4, 4, bias=False)
        model = _DenseLayer(4, mlp)
        count = count_matmul_params([model], SimpleNamespace(num_experts_per_tok=1, n_routed_experts=4))
        expected = 4 * 16 + Fraction(1, 4) * 4 * 16 + 16
        self.assertEqual(count.matmul_params, expected, f"case: routed_and_shared, got={count.matmul_params}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_uncovered_structures_are_reported(self) -> None:
        """Report structures the formula would misstate instead of counting them.

        Feature: PaLM model-FLOPs accounting.
        Description: Count models with routed experts but no routing fields, MTP layers and a vision config.
        Expectation: Each structure yields a reason, so no utilization is reported.
        """
        mlp = nn.Module()
        mlp.experts = nn.ModuleList(nn.Linear(4, 4, bias=False) for _ in range(2))
        routed = count_matmul_params([_DenseLayer(4, mlp)], SimpleNamespace())
        self.assertTrue(
            any("top-k" in reason for reason in routed.unsupported),
            f"case: routed_without_fields, got={routed.unsupported}")

        holder = nn.Module()
        holder.mtp = nn.Linear(4, 4, bias=False)
        mtp = count_matmul_params([holder], SimpleNamespace())
        self.assertTrue(
            any("multi-token" in reason for reason in mtp.unsupported),
            f"case: mtp_layers, got={mtp.unsupported}")

        vision = count_matmul_params([nn.Linear(4, 4)], SimpleNamespace(vision_config=SimpleNamespace()))
        self.assertTrue(
            any("vision" in reason for reason in vision.unsupported),
            f"case: unidentified_vision_encoder, got={vision.unsupported}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_vision_encoder_is_excluded_for_decoder_only_mfu(self) -> None:
        """Leave vision-encoder parameters out of the text-decoder count.

        Feature: PaLM model-FLOPs accounting for multimodal models.
        Description: Count a model holding a visual encoder, a projector and a text decoder.
        Expectation: Only decoder matmuls enter N; the visual parameters are reported separately.
        """
        model = nn.Module()
        model.visual = nn.Module()
        model.visual.patch_embed = nn.Linear(4, 8, bias=False)
        model.visual.merger = nn.Linear(8, 4, bias=True)
        model.language_model = _DenseLayer(4, nn.Identity())
        count = count_matmul_params([model], SimpleNamespace(vision_config=SimpleNamespace()))
        self.assertEqual(count.unsupported, (), f"case: decoder_only_supported, got={count.unsupported}")
        self.assertEqual(count.matmul_params, 4 * 16, f"case: decoder_matmul_params, got={count.matmul_params}")
        self.assertEqual(count.vision_params, 32 + 32 + 4, f"case: vision_params, got={count.vision_params}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_attention_geometry_from_config(self) -> None:
        """Read head widths and layer kinds from Hugging Face style configs.

        Feature: PaLM model-FLOPs accounting.
        Description: Resolve GQA, implicit head_dim, MLA, sliding-window and linear-attention configs.
        Expectation: Covered geometries yield per-layer attention terms; uncovered ones yield a reason.
        """
        gqa, _ = attention_layers_from_config(
            SimpleNamespace(num_hidden_layers=2, num_attention_heads=32, num_key_value_heads=4, head_dim=128,
                            hidden_size=2048))
        self.assertEqual(gqa, (AttentionLayer(32, 128, 128),) * 2, f"case: gqa_query_heads, got={gqa}")

        implicit, _ = attention_layers_from_config(
            SimpleNamespace(num_hidden_layers=1, num_attention_heads=16, hidden_size=2048))
        self.assertEqual(implicit, (AttentionLayer(16, 128, 128),), f"case: implicit_head_dim, got={implicit}")

        mla, _ = attention_layers_from_config(
            SimpleNamespace(num_hidden_layers=1, num_attention_heads=16, hidden_size=2048,
                            qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128))
        self.assertEqual(mla, (AttentionLayer(16, 192, 128),), f"case: mla_head_dims, got={mla}")
        self.assertEqual(
            mla[0].forward_flops_per_token(4096) * 3, 12 * 16 * 160 * 4096,
            f"case: mla_palm_term, got={mla[0].forward_flops_per_token(4096) * 3}")

        sliding, _ = attention_layers_from_config(
            SimpleNamespace(num_hidden_layers=2, num_attention_heads=8, head_dim=64, sliding_window=1024,
                            layer_types=["sliding_attention", "full_attention"]))
        self.assertEqual(
            [layer.forward_flops_per_token(4096) for layer in sliding],
            [2 * 8 * 128 * 1024, 2 * 8 * 128 * 4096],
            f"case: sliding_window_attended_tokens, got={sliding}")

        uncovered = (
            ("linear_without_geometry", SimpleNamespace(num_hidden_layers=2, num_attention_heads=8, head_dim=64,
                                                        layer_types=["linear_attention", "full_attention"])),
            ("sparse_attention", SimpleNamespace(num_hidden_layers=1, num_attention_heads=8, head_dim=64,
                                                 index_topk=2048)),
            ("unknown_layer_type", SimpleNamespace(num_hidden_layers=1, num_attention_heads=8, head_dim=64,
                                                   layer_types=["chunked_attention"])),
            ("layer_types_length", SimpleNamespace(num_hidden_layers=3, num_attention_heads=8, head_dim=64,
                                                   layer_types=["full_attention"])),
            ("missing_fields", SimpleNamespace(hidden_size=64)),
        )
        for case, config in uncovered:
            layers, reason = attention_layers_from_config(config)
            self.assertEqual(layers, (), f"case: {case}_no_layers, got={layers}")
            self.assertIsNotNone(reason, f"case: {case}_reason, got={reason!r}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_hybrid_delta_rule_layers(self) -> None:
        """Add the delta-rule recurrence for linear-attention layers of a hybrid model.

        Feature: PaLM model-FLOPs accounting for hybrid linear attention.
        Description: Resolve the Qwen3.5-9B layer layout of three delta-rule layers per softmax layer.
        Expectation: Delta-rule layers cost 18 * value_heads * key_dim * value_dim per token, independent
            of sequence length, and softmax layers keep the PaLM attention term.
        """
        layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8
        config = SimpleNamespace(
            num_hidden_layers=32, num_attention_heads=16, num_key_value_heads=4, head_dim=256, hidden_size=4096,
            linear_num_key_heads=16, linear_num_value_heads=32, linear_key_head_dim=128, linear_value_head_dim=128,
            layer_types=layer_types)
        layers, reason = attention_layers_from_config(SimpleNamespace(text_config=config))
        self.assertIsNone(reason, f"case: hybrid_supported, got={reason!r}")
        self.assertEqual(
            [type(layer) for layer in layers].count(DeltaRuleLayer), 24, f"case: delta_rule_layers, got={layers}")
        profile = ModelFlopsProfile(0.0, 0.0, layers)
        expected = 8 * 12 * 16 * 256 * 4096 + 24 * 18 * 32 * 128 * 128
        actual = profile.flops_per_token(4096)
        self.assertEqual(actual, expected, f"case: hybrid_term, got={actual}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_sliding_windows_without_layer_types(self) -> None:
        """Apply the Qwen2 and Mistral sliding-window conventions when layer_types is absent.

        Feature: PaLM model-FLOPs accounting for sliding-window attention.
        Description: Resolve configs with use_sliding_window plus max_window_layers, a bare
            sliding_window, and a disabled use_sliding_window switch.
        Expectation: Qwen2-style configs window the layers from max_window_layers on, Mistral-style
            configs window every layer, and a disabled switch keeps full attention.
        """
        base = {"num_hidden_layers": 2, "num_attention_heads": 8, "head_dim": 64}
        cases = (
            ("qwen2_style", {"use_sliding_window": True, "sliding_window": 1024, "max_window_layers": 1},
             [None, 1024]),
            ("mistral_style", {"sliding_window": 1024}, [1024, 1024]),
            ("switch_off", {"use_sliding_window": False, "sliding_window": 1024}, [None, None]),
        )
        for case, fields, windows in cases:
            layers, reason = attention_layers_from_config(SimpleNamespace(**base, **fields))
            self.assertIsNone(reason, f"case: {case}_supported, got={reason!r}")
            self.assertEqual([layer.window for layer in layers], windows, f"case: {case}_windows, got={layers}")


if __name__ == "__main__":
    unittest.main()
