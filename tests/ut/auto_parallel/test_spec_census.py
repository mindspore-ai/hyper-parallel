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
"""Tests for the census's model spec: the block walk, its kinds, its fields and its op counts.

How to run this:
    pytest tests/ut/auto_parallel/test_spec_census.py -v
"""
import unittest
from unittest.mock import patch

from hyper_parallel.auto_parallel._hf_model_spec import resolve_hf_model_spec
from hyper_parallel.auto_parallel._model_spec import ModelSpec, ModelSpecError
from hyper_parallel.auto_parallel._op_profiles import family_profile, resolve_ops
from hyper_parallel.auto_parallel._spec_census import census_model_spec

_HF_CONFIG = "hyper_parallel.auto_parallel._hf_model_spec._get_hf_config"

# The model facts a spec states as true or false.
_FLAGS = ("attn_output_gate", "tie_word_embeddings", "qk_norm", "qkv_bias", "o_bias", "mlp_bias", "norm_bias",
          "shared_expert_gate")

# What a spec states that the comparison leaves out: its name, what the
# AutoModels parser drops (MTP layers, which Transformers does not build),
# the census records, and the vision tower, which the census does not walk.
_NOT_COMPARED = ("name", "mtp_depth", "layer_types", "activations", "output_activations", "vision", "ops", "layers")


def _as_priced(data):
    """The fields of the model spec *data*, each as ND prices it.

    A field left unstated is compared as the value ND prices for it: a head
    as wide as derive prices it, an MLA family's at its value heads' width;
    a flag as false; the routed and shared experts' widths as the spec
    resolves them; each kind's op counts as its profile states them, one
    field per op.  The layer stack is its body, the groups in model order:
    the AutoModels parser prices no MTP layer.
    """
    spec = ModelSpec.from_dict(data)
    out = {key: value for key, value in spec.to_dict().items() if key not in _NOT_COMPARED}
    value_width = family_profile(spec.arch).model.get("v_head_dim")
    out["head_dim"] = (spec.v_head_dim or value_width) if value_width is not None else spec.effective_head_dim
    out.update({flag: bool(getattr(spec, flag)) for flag in _FLAGS})
    if spec.is_moe:
        out["moe_intermediate_size"] = spec.routed_expert_width
    if spec.num_shared_experts:
        out["shared_expert_intermediate_size"] = spec.shared_expert_width
    for kind, counts in resolve_ops(spec.arch, spec.ops).items():
        out.update({f"ops.{kind}.{op}": count for op, count in counts.to_dict().items()})
    out["layers"] = tuple((group.kind, group.count) for group in spec.layers or () if not group.mtp)
    return out


def _qwen35_text():
    """A two-layer Qwen3.5-MoE text config of width 64, a linear-attention layer then a full one, gated."""
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # pylint: disable=C0415
        Qwen3_5MoeTextConfig,
    )
    return Qwen3_5MoeTextConfig.from_dict({
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32, "linear_num_key_heads": 2, "linear_key_head_dim": 16,
        "linear_num_value_heads": 4, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
        "vocab_size": 128, "max_position_embeddings": 256, "layer_types": ["linear_attention", "full_attention"],
        "attn_output_gate": True,
    })


def _deepseek_v3():
    """A two-layer DeepSeek-V3 config of width 64, a dense layer then one of 4 experts, MLA heads 12, 4 and 8 wide."""
    from transformers import DeepseekV3Config  # pylint: disable=C0415
    return DeepseekV3Config(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, q_lora_rank=32,
        kv_lora_rank=16, qk_nope_head_dim=12, qk_rope_head_dim=4, v_head_dim=8, n_routed_experts=4,
        num_experts_per_tok=2, moe_intermediate_size=16, n_shared_experts=1, first_k_dense_replace=1,
        intermediate_size=32, vocab_size=128, n_group=1, topk_group=1)


def _one_kind(config_cls, **fields):
    """A two-layer config of *config_cls* of width 64, its layers of one kind."""
    return config_cls(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      intermediate_size=128, vocab_size=128, **fields)


def _resolved(config, **model):
    """The resolver's spec of *config*, as a Transformers checkpoint states it."""
    with patch(_HF_CONFIG, return_value=config):
        return resolve_hf_model_spec({"pretrained_model_name_or_path": "local/model", **model})


class TestSpecCensus(unittest.TestCase):
    """The census states a model from the layers Transformers builds of it."""

    def test_a_hybrid_model_is_stated_as_the_resolver_states_it(self):
        """
        Feature: census_model_spec.
        Description: The two-layer hybrid Qwen3.5-MoE model, a linear-attention
            layer then a full-attention one, its query projection gated.
        Expectation: The family's kinds in the model's order; the linear
            kind runs the delta rule and no score, the full kind two batched
            matmuls and a softmax; every field as ND prices it is the
            resolver's.
        """
        config = _qwen35_text()
        census = census_model_spec(config)
        self.assertEqual(census["layers"], [{"kind": "linear_attention", "count": 1},
                                            {"kind": "full_attention", "count": 1}])
        linear, full = census["ops"]["linear_attention"], census["ops"]["full_attention"]
        self.assertEqual((linear["attBMM"], linear["softmax"], linear["headCast"], linear["linrec"]), (0, 0, 0, 1))
        self.assertEqual((full["attBMM"], full["softmax"], full["headCast"], full["linrec"]), (2, 1, 1, 0))
        self.assertIs(census["attn_output_gate"], True)
        self.assertEqual(_as_priced(census), _as_priced(_resolved(config)))

    def test_dense_then_moe_layers_and_mla(self):
        """
        Feature: census_model_spec of DeepSeek-V3.
        Description: A dense layer then one of 4 experts and a shared one,
            under MLA: a query latent 32 wide, a key and value latent 16
            wide, non-rotary key heads 12 wide, rotary 4, value heads 8.
        Expectation: The family's dense and moe kinds; the latents' and the
            heads' widths as the layers build them, a head as wide as its
            values, as ND prices an MLA head; every field as ND prices it is
            the resolver's.
        """
        config = _deepseek_v3()
        census = census_model_spec(config)
        self.assertEqual(census["layers"], [{"kind": "dense", "count": 1}, {"kind": "moe", "count": 1}])
        self.assertEqual(census["first_k_dense_replace"], 1)
        self.assertEqual({key: census[key] for key in ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim",
                                                         "qk_rope_head_dim", "v_head_dim", "head_dim")},
                         {"q_lora_rank": 32, "kv_lora_rank": 16, "qk_nope_head_dim": 12, "qk_rope_head_dim": 4,
                          "v_head_dim": 8, "head_dim": 8})
        self.assertEqual((census["num_experts"], census["num_experts_per_tok"], census["num_shared_experts"]),
                         (4, 2, 1))
        self.assertEqual(_as_priced(census), _as_priced(_resolved(config)))

    def test_flags_are_read_off_the_layers(self):
        """
        Feature: census_model_spec's flags.
        Description: A Llama biasing its projections and feed-forward; a
            Qwen2, whose query, key and value projections alone hold a
            bias; a Qwen3 with a QK-norm, its output tied to its embedding.
        Expectation: Each flag is what the layers hold, and every field as
            ND prices it is the resolver's.
        """
        from transformers import LlamaConfig, Qwen2Config, Qwen3Config  # pylint: disable=C0415
        cases = (
            (_one_kind(LlamaConfig, attention_bias=True, mlp_bias=True),
             {"qkv_bias": True, "o_bias": True, "mlp_bias": True, "qk_norm": False}),
            (_one_kind(Qwen2Config), {"qkv_bias": True, "o_bias": False, "mlp_bias": False, "qk_norm": False}),
            (_one_kind(Qwen3Config, head_dim=32, tie_word_embeddings=True),
             {"qkv_bias": False, "qk_norm": True, "tie_word_embeddings": True, "head_dim": 32}),
        )
        for config, flags in cases:
            census = census_model_spec(config)
            self.assertEqual({key: census[key] for key in flags}, flags, type(config).__name__)
            self.assertEqual(_as_priced(census), _as_priced(_resolved(config)), type(config).__name__)

    def test_a_profile_pricing_two_shapes_as_one_kind_is_refused(self):
        """
        Feature: census_model_spec's kinds.
        Description: A Qwen2-MoE whose second layer runs a dense
            feed-forward (mlp_only_layers), priced by the qwen family's one
            kind, which keeps the model's own feed-forward.
        Expectation: A ModelSpecError naming the two layers: the family's
            kind would price the dense one as MoE.
        """
        from transformers import Qwen2MoeConfig  # pylint: disable=C0415
        config = _one_kind(Qwen2MoeConfig, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
                           shared_expert_intermediate_size=32, mlp_only_layers=[1])
        with self.assertRaisesRegex(ModelSpecError, "layers 0 and 1 as one kind 'decoder'"):
            census_model_spec(config)

    def test_the_resolver_states_the_census_spec(self):
        """
        Feature: resolve_hf_model_spec(census_spec=True).
        Description: A Llama checkpoint whose overrides name the qwen
            family, resolved with and without the census's spec.
        Expectation: With it, the spec is the census's, its kinds the
            named family's, and a valid spec; as ND prices them, the two
            agree.
        """
        from transformers import LlamaConfig  # pylint: disable=C0415
        config = _one_kind(LlamaConfig)
        with patch(_HF_CONFIG, return_value=config):
            model = {"pretrained_model_name_or_path": "local/model", "config_overrides": {"arch": "qwen"}}
            census = resolve_hf_model_spec(model, census_spec=True)
            resolved = resolve_hf_model_spec(model)
        self.assertEqual((census["arch"], list(census["ops"])), ("qwen", ["decoder"]))
        ModelSpec.from_dict(census)
        self.assertEqual(_as_priced(census), _as_priced(resolved))


if __name__ == "__main__":
    unittest.main()
