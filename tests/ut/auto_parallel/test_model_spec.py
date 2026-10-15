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
"""Tests for the model IR schema."""
import unittest

import yaml

from hyper_parallel.auto_parallel._model_spec import LayerGroup, ModelSpec, ModelSpecError


def _dense() -> dict:
    """A plain dense decoder, the smallest spec that validates."""
    return {
        "name": "llama2",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
    }


def _qwen35() -> dict:
    """Qwen3.5-35B-A3B: all-MoE, explicit head_dim, one shared expert, MTP."""
    return {
        "name": "qwen3_5_moe",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "num_shared_experts": 1,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "mtp_depth": 1,
    }


def _counts(**overrides) -> dict:
    """The default decoder's op counts, with *overrides* applied."""
    counts = {
        "attMM": 4, "attBMM": 2, "ffMM": 3, "softmax": 1,
        "dropout": 0, "normOp": 2, "gather": 4, "headCast": 1, "ffAct": 1,
        "linrec": 0,
    }
    counts.update(overrides)
    return counts


class TestRefused(unittest.TestCase):
    """A spec that is incomplete or incoherent is refused at parse time, not priced as a zero."""

    def test_an_incomplete_or_incoherent_spec_is_refused(self):
        """
        Feature: ModelSpec.from_dict's checks.
        Description: A spec without vocab_size; Qwen3.5 with no width its experts could take; 7
            key-value heads for 32 attention heads; a stack of 39 body layers against 40; an op
            vector without softmax; and compute_dtype, which states the run's precision (D17).
        Expectation: Each refused by name; compute_dtype with the keys a train yaml states the
            run's precision in.
        """
        stack = [{"kind": "decoder", "count": 39}, {"kind": "decoder", "count": 1, "mtp": True}]
        no_softmax = {op: count for op, count in _counts().items() if op != "softmax"}
        cases = (
            ({key: value for key, value in _dense().items() if key != "vocab_size"}, "vocab_size"),
            ({key: value for key, value in _qwen35().items() if not key.endswith("intermediate_size")},
             "moe_intermediate_size"),
            (dict(_dense(), num_key_value_heads=7), "num_key_value_heads"),
            (dict(_qwen35(), layers=stack), "39 body layers"),
            (dict(_dense(), ops={"decoder": no_softmax}), "softmax"),
            (dict(_dense(), compute_dtype="float32"), r"mix_precision\.param_dtype, else model_init_dtype"),
        )
        for data, named in cases:
            with self.subTest(named=named), self.assertRaisesRegex(ModelSpecError, named):
                ModelSpec.from_dict(data)


class TestAbsentIsNotZero(unittest.TestCase):
    """Optional fields distinguish 'not declared' from 'declared as zero'."""

    def test_name_defaults_when_absent(self):
        """Only the name has a default; every other required field does not."""
        data = _dense()
        del data["name"]
        self.assertEqual(ModelSpec.from_dict(data).name, "custom")

    def test_dense_model_has_no_expert_count(self):
        """A dense model's expert count is None, so no consumer reads 0."""
        spec = ModelSpec.from_dict(_dense())
        self.assertIsNone(spec.num_experts)
        self.assertFalse(spec.is_moe)

    def test_single_expert_is_not_moe(self):
        """num_experts of 1 is a dense feed-forward, so MoE rules do not bind."""
        data = _dense()
        data["num_experts"] = 1
        self.assertFalse(ModelSpec.from_dict(data).is_moe)

    def test_absent_optional_is_omitted_from_dump(self):
        """An unset field does not appear as null in the serialised form."""
        dumped = ModelSpec.from_dict(_dense()).to_dict()
        for name in ("num_experts", "mtp_depth", "arch", "ops", "vision"):
            self.assertNotIn(name, dumped)


class TestDerivedReads(unittest.TestCase):
    """The reads that were wrong for Qwen3.5 before the spec existed."""

    def test_head_dim_explicit_beats_derived(self):
        """An explicit head_dim is used, not hidden_size / heads."""
        spec = ModelSpec.from_dict(_qwen35())
        self.assertEqual(spec.effective_head_dim, 256)

    def test_all_moe_config_still_sizes_its_shared_expert(self):
        """Qwen3.5 declares no dense width; the shared expert still resolves.

        This is the regression the schema exists for: the old path read
        intermediate_size, found nothing, and priced the shared expert at
        width zero without saying so.
        """
        data = _qwen35()
        self.assertNotIn("intermediate_size", data)
        self.assertEqual(ModelSpec.from_dict(data).shared_expert_width, 512)


class TestLayersOnTheSpec(unittest.TestCase):
    """A spec may state its layer stack, and it must add up."""

    def test_layers_round_trip_through_yaml_text(self):
        """
        Feature: ModelSpec.layers.
        Description: 40 body layers and one MTP layer, dumped and reloaded.
        Expectation: The same typed groups, and mtp written only where true.
        """
        layers = [{"kind": "decoder", "count": 40}, {"kind": "decoder", "count": 1, "mtp": True}]
        spec = ModelSpec.from_dict(dict(_qwen35(), layers=layers))
        self.assertEqual(spec.layers, (LayerGroup("decoder", 40), LayerGroup("decoder", 1, mtp=True)))
        text = yaml.safe_dump(spec.to_dict(), sort_keys=True)
        self.assertEqual(ModelSpec.from_dict(yaml.safe_load(text)), spec)
        self.assertEqual(spec.to_dict()["layers"], layers)


if __name__ == "__main__":
    unittest.main()
