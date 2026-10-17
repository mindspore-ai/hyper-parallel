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

from hyper_parallel.auto_parallel._model_spec import (
    LayerGroup,
    ModelSpec,
    ModelSpecError,
    OpCounts,
    VisionSpec,
    model_fields,
)


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


class TestRequiredFields(unittest.TestCase):
    """A required fact that nobody supplied is an error, not a zero."""

    def test_missing_field_raises_by_name(self):
        """The message names the field a producer failed to supply."""
        data = _dense()
        del data["vocab_size"]
        with self.assertRaises(ModelSpecError) as ctx:
            ModelSpec.from_dict(data)
        self.assertIn("vocab_size", str(ctx.exception))

    def test_zero_is_rejected_like_absence(self):
        """A required field declared as zero is refused, not carried."""
        data = _dense()
        data["hidden_size"] = 0
        with self.assertRaises(ModelSpecError) as ctx:
            ModelSpec.from_dict(data)
        self.assertIn("hidden_size", str(ctx.exception))

    def test_non_integer_raises_by_name(self):
        """A field that cannot be an integer names itself."""
        data = _dense()
        data["num_hidden_layers"] = "many"
        with self.assertRaises(ModelSpecError) as ctx:
            ModelSpec.from_dict(data)
        self.assertIn("num_hidden_layers", str(ctx.exception))

    def test_name_defaults_when_absent(self):
        """Only the name has a default; every other required field does not."""
        data = _dense()
        del data["name"]
        self.assertEqual(ModelSpec.from_dict(data).name, "custom")


class TestAbsentIsNotZero(unittest.TestCase):
    """Optional fields distinguish 'not declared' from 'declared as zero'."""

    def test_dense_model_has_no_expert_count(self):
        """A dense model's expert count is None, so no consumer reads 0."""
        spec = ModelSpec.from_dict(_dense())
        self.assertIsNone(spec.num_experts)
        self.assertFalse(spec.is_moe)

    def test_absent_optional_is_omitted_from_dump(self):
        """An unset field does not appear as null in the serialised form."""
        dumped = ModelSpec.from_dict(_dense()).to_dict()
        self.assertNotIn("num_experts", dumped)
        self.assertNotIn("mtp_depth", dumped)


class TestDerivedReads(unittest.TestCase):
    """The reads that were wrong for Qwen3.5 before the spec existed."""

    def test_head_dim_explicit_beats_derived(self):
        """An explicit head_dim is used, not hidden_size / heads."""
        spec = ModelSpec.from_dict(_qwen35())
        self.assertEqual(spec.effective_head_dim, 256)
        self.assertEqual(spec.attention_width, 16 * 256)

    def test_head_dim_derived_when_absent(self):
        """Without an explicit head_dim the classic ratio still holds."""
        spec = ModelSpec.from_dict(_dense())
        self.assertEqual(spec.effective_head_dim, 128)
        self.assertEqual(spec.attention_width, 4096)

    def test_shared_expert_width_from_its_own_field(self):
        """An all-MoE config resolves the shared expert to a real width."""
        self.assertEqual(ModelSpec.from_dict(_qwen35()).shared_expert_width, 512)

    def test_shared_expert_width_falls_back_to_routed(self):
        """A config declaring only the routed width shares it."""
        data = _qwen35()
        del data["shared_expert_intermediate_size"]
        self.assertEqual(ModelSpec.from_dict(data).shared_expert_width, 512)

    def test_no_shared_expert_is_zero_width(self):
        """No shared expert genuinely is no width, and says so."""
        data = _qwen35()
        del data["num_shared_experts"]
        self.assertEqual(ModelSpec.from_dict(data).shared_expert_width, 0)


class TestCoherence(unittest.TestCase):
    """Fields that contradict each other are refused at parse time."""

    def test_moe_with_no_resolvable_width_raises(self):
        """Routed experts nothing can size are refused, not priced at zero."""
        data = _qwen35()
        del data["moe_intermediate_size"]
        del data["shared_expert_intermediate_size"]
        with self.assertRaises(ModelSpecError) as ctx:
            ModelSpec.from_dict(data)
        self.assertIn("routed", str(ctx.exception))

    def test_moe_falls_back_to_the_dense_width(self):
        """Mixtral runs its experts at the dense width and declares no other."""
        data = _dense()
        data.update(num_experts=8, num_experts_per_tok=2)
        spec = ModelSpec.from_dict(data)
        self.assertTrue(spec.is_moe)
        self.assertEqual(spec.routed_expert_width, 11008)

    def test_shared_expert_without_any_width_raises(self):
        """A shared expert nothing can size is refused, not priced at zero."""
        data = _dense()
        del data["intermediate_size"]
        data["num_shared_experts"] = 1
        with self.assertRaises(ModelSpecError) as ctx:
            ModelSpec.from_dict(data)
        self.assertIn("num_shared_experts", str(ctx.exception))

    def test_all_moe_config_still_sizes_its_shared_expert(self):
        """Qwen3.5 declares no dense width; the shared expert still resolves.

        This is the regression the schema exists for: the old path read
        intermediate_size, found nothing, and priced the shared expert at
        width zero without saying so.
        """
        data = _qwen35()
        self.assertNotIn("intermediate_size", data)
        self.assertEqual(ModelSpec.from_dict(data).shared_expert_width, 512)

    def test_kv_heads_must_divide_attention_heads(self):
        """Grouped-query attention that does not group is a typo, not a model."""
        data = _dense()
        data["num_key_value_heads"] = 7
        with self.assertRaises(ModelSpecError) as ctx:
            ModelSpec.from_dict(data)
        self.assertIn("num_key_value_heads", str(ctx.exception))

    def test_single_expert_is_not_moe(self):
        """num_experts of 1 is a dense feed-forward, so MoE rules do not bind."""
        data = _dense()
        data["num_experts"] = 1
        self.assertFalse(ModelSpec.from_dict(data).is_moe)


class TestRoundTrip(unittest.TestCase):
    """The serialised form is the interface, so it has to survive a round trip."""

    def test_dense_round_trip(self):
        """from_dict(to_dict(spec)) reproduces the spec exactly."""
        spec = ModelSpec.from_dict(_dense())
        self.assertEqual(ModelSpec.from_dict(spec.to_dict()), spec)

    def test_moe_round_trip(self):
        """A MoE spec with MTP and an explicit head_dim survives unchanged."""
        spec = ModelSpec.from_dict(_qwen35())
        self.assertEqual(ModelSpec.from_dict(spec.to_dict()), spec)

    def test_round_trip_through_yaml_text(self):
        """The round trip holds through real YAML, not just a dict."""
        spec = ModelSpec.from_dict(_qwen35())
        text = yaml.safe_dump(spec.to_dict(), sort_keys=True)
        self.assertEqual(ModelSpec.from_dict(yaml.safe_load(text)), spec)

    def test_dump_is_stable(self):
        """Dumping twice gives the same text, so a spec can be diffed."""
        spec = ModelSpec.from_dict(_qwen35())
        first = yaml.safe_dump(spec.to_dict(), sort_keys=True)
        second = yaml.safe_dump(ModelSpec.from_dict(yaml.safe_load(first)).to_dict(),
                                sort_keys=True)
        self.assertEqual(first, second)


class TestModelKeysOnly(unittest.TestCase):
    """A model spec states the model; a run's keys belong in the ExecSpec."""

    def test_an_unknown_key_is_refused(self):
        """A key that is not the model's is refused, by name, not carried."""
        data = _dense()
        data["compute_dtype"] = "bfloat16"
        with self.assertRaisesRegex(ModelSpecError, "compute_dtype"):
            ModelSpec.from_dict(data)

    def test_a_precision_key_is_refused_with_the_keys_a_run_reads(self):
        """compute_dtype states the run's precision: the refusal says where a train yaml states it."""
        data = _dense()
        data["compute_dtype"] = "float32"
        with self.assertRaisesRegex(ModelSpecError, r"mix_precision\.param_dtype, else model_init_dtype"):
            ModelSpec.from_dict(data)
        with self.assertRaises(ModelSpecError) as refused:
            ModelSpec.from_dict(dict(_dense(), offset=[1, -1]))
        self.assertNotIn("precision", str(refused.exception))

    def test_model_fields_takes_the_model_from_a_run(self):
        """A mapping that states the run too gives up only the model's keys."""
        data = dict(_dense(), local_batch_size=1, offset=[1, -1], compute_dtype="bfloat16")
        fields = model_fields(data)
        self.assertEqual(sorted(set(data) - set(fields)), ["compute_dtype", "local_batch_size", "offset"])
        self.assertEqual(ModelSpec.from_dict(fields).hidden_size, _dense()["hidden_size"])

    def test_the_facts_extra_held_are_fields(self):
        """attn_output_gate, qk_nope_head_dim and layer_types are typed, and round-trip."""
        data = dict(_dense(), attn_output_gate=True, qk_nope_head_dim=128,
                    layer_types=["full_attention"] * _dense()["num_hidden_layers"])
        spec = ModelSpec.from_dict(data)
        got = (spec.attn_output_gate, spec.qk_nope_head_dim, spec.layer_types[0])
        self.assertEqual(got, (True, 128, "full_attention"), f"fields={got}")
        self.assertEqual(ModelSpec.from_dict(spec.to_dict()), spec)

    def test_a_non_boolean_gate_is_refused(self):
        """attn_output_gate is true or false, not a string that reads as one."""
        with self.assertRaisesRegex(ModelSpecError, "attn_output_gate"):
            ModelSpec.from_dict(dict(_dense(), attn_output_gate="yes"))

    def test_a_tie_is_a_boolean_fact(self):
        """tie_word_embeddings is typed, round-trips, and refuses a string."""
        spec = ModelSpec.from_dict(dict(_dense(), tie_word_embeddings=True))
        self.assertIs(spec.tie_word_embeddings, True)
        self.assertEqual(ModelSpec.from_dict(spec.to_dict()), spec)
        with self.assertRaisesRegex(ModelSpecError, "tie_word_embeddings"):
            ModelSpec.from_dict(dict(_dense(), tie_word_embeddings="yes"))


class TestVisionTower(unittest.TestCase):
    """A multimodal spec nests its tower rather than flattening it."""

    def _vl(self) -> dict:
        """The Qwen3.5 spec with a vision tower of 27 layers nested under ``vision``."""
        data = _qwen35()
        data["vision"] = {
            "hidden_size": 1152,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "num_position_embeddings": 2304,
            "max_position_embeddings": 576,
        }
        return data

    def test_vision_is_parsed_and_typed(self):
        """The tower arrives as a VisionSpec, not a nested dict."""
        spec = ModelSpec.from_dict(self._vl())
        self.assertIsInstance(spec.vision, VisionSpec)
        self.assertEqual(spec.vision.num_hidden_layers, 27)

    def test_vision_round_trip(self):
        """A nested tower survives the round trip with its parent."""
        spec = ModelSpec.from_dict(self._vl())
        self.assertEqual(ModelSpec.from_dict(spec.to_dict()), spec)

    def test_vision_layers_round_trip(self):
        """A tower's stated stack survives the round trip, typed as groups."""
        data = self._vl()
        data["vision"]["layers"] = [{"kind": "encoder", "count": 27}]
        spec = ModelSpec.from_dict(data)
        self.assertEqual(spec.vision.layers, (LayerGroup("encoder", 27),))
        self.assertEqual(spec.to_dict()["vision"]["layers"], [{"kind": "encoder", "count": 27}])
        self.assertEqual(ModelSpec.from_dict(spec.to_dict()), spec)

    def test_vision_layers_must_cover_the_tower(self):
        """A tower's stack sums to its own layer count, and has no MTP layers."""
        for layers, words in (
            ([{"kind": "encoder", "count": 26}], "vision.layers list 26"),
            ([{"kind": "encoder", "count": 27, "mtp": True}], "MTP"),
        ):
            data = self._vl()
            data["vision"]["layers"] = layers
            with self.subTest(layers=layers):
                with self.assertRaises(ModelSpecError) as ctx:
                    ModelSpec.from_dict(data)
                self.assertIn(words, str(ctx.exception))

    def test_vision_missing_field_raises(self):
        """The tower is held to the same standard as the parent."""
        data = self._vl()
        del data["vision"]["num_hidden_layers"]
        with self.assertRaises(ModelSpecError) as ctx:
            ModelSpec.from_dict(data)
        self.assertIn("num_hidden_layers", str(ctx.exception))

    def test_plain_model_has_no_tower(self):
        """A text-only spec leaves vision absent rather than empty."""
        spec = ModelSpec.from_dict(_dense())
        self.assertIsNone(spec.vision)
        self.assertNotIn("vision", spec.to_dict())


class TestOpCounts(unittest.TestCase):
    """An op vector names every op the cost model prices, and nothing else."""

    def test_round_trip(self):
        """The mapping form comes back exactly, in declaration order."""
        counts = OpCounts.from_dict(_counts())
        self.assertEqual(counts.to_dict(), _counts())
        self.assertEqual(list(counts.to_dict()), list(_counts()))

    def test_missing_op_raises_by_name(self):
        """An op left out would be priced at zero, so it is refused."""
        data = _counts()
        del data["softmax"]
        with self.assertRaises(ModelSpecError) as ctx:
            OpCounts.from_dict(data, "ops.decoder")
        self.assertIn("softmax", str(ctx.exception))
        self.assertIn("ops.decoder", str(ctx.exception))

    def test_unknown_op_raises_by_name(self):
        """A misspelt op is refused rather than silently ignored."""
        data = _counts()
        data["sofmax"] = data.pop("softmax")
        with self.assertRaises(ModelSpecError) as ctx:
            OpCounts.from_dict(data)
        self.assertIn("sofmax", str(ctx.exception))

    def test_count_must_be_a_whole_non_negative_number(self):
        """Negative, fractional and boolean counts all raise by op name."""
        for bad in (-1, 1.5, True, "four"):
            with self.subTest(bad=bad):
                with self.assertRaises(ModelSpecError) as ctx:
                    OpCounts.from_dict(_counts(gather=bad))
                self.assertIn("gather", str(ctx.exception))

    def test_zero_is_a_declared_count(self):
        """An op the layer does not run is declared as 0 and kept as 0."""
        self.assertEqual(OpCounts.from_dict(_counts(dropout=0)).dropout, 0)


class TestOpsOnTheSpec(unittest.TestCase):
    """A spec may name its op profile and declare counts of its own."""

    def _with_ops(self) -> dict:
        data = _dense()
        data["arch"] = "default"
        data["ops"] = {"decoder": _counts(softmax=0)}
        return data

    def test_absent_by_default(self):
        """A spec that says nothing about ops carries neither field."""
        spec = ModelSpec.from_dict(_dense())
        self.assertIsNone(spec.arch)
        self.assertIsNone(spec.ops)
        self.assertNotIn("ops", spec.to_dict())
        self.assertNotIn("arch", spec.to_dict())

    def test_ops_are_parsed_and_typed(self):
        """Each layer kind arrives as an OpCounts, not a nested dict."""
        spec = ModelSpec.from_dict(self._with_ops())
        self.assertEqual(spec.arch, "default")
        self.assertIsInstance(spec.ops["decoder"], OpCounts)
        self.assertEqual(spec.ops["decoder"].softmax, 0)

    def test_ops_round_trip_through_yaml_text(self):
        """Declared counts survive a dump and a reload through real YAML."""
        spec = ModelSpec.from_dict(self._with_ops())
        text = yaml.safe_dump(spec.to_dict(), sort_keys=True)
        self.assertEqual(ModelSpec.from_dict(yaml.safe_load(text)), spec)
        self.assertEqual(spec.to_dict()["ops"], {"decoder": _counts(softmax=0)})

    def test_malformed_ops_raise(self):
        """Ops must map at least one kind to a complete vector."""
        for bad in ({}, {"decoder": 4}, [_counts()]):
            with self.subTest(bad=bad):
                data = _dense()
                data["ops"] = bad
                with self.assertRaises(ModelSpecError):
                    ModelSpec.from_dict(data)


class TestLayersOnTheSpec(unittest.TestCase):
    """A spec may state its layer stack, and it must add up."""

    @staticmethod
    def _with_layers(layers) -> dict:
        data = _qwen35()
        data["layers"] = layers
        return data

    def test_layers_round_trip_through_yaml_text(self):
        """
        Feature: ModelSpec.layers.
        Description: 40 body layers and one MTP layer, dumped and reloaded.
        Expectation: The same typed groups, and mtp written only where true.
        """
        layers = [{"kind": "decoder", "count": 40}, {"kind": "decoder", "count": 1, "mtp": True}]
        spec = ModelSpec.from_dict(self._with_layers(layers))
        self.assertEqual(spec.layers, (LayerGroup("decoder", 40), LayerGroup("decoder", 1, mtp=True)))
        text = yaml.safe_dump(spec.to_dict(), sort_keys=True)
        self.assertEqual(ModelSpec.from_dict(yaml.safe_load(text)), spec)
        self.assertEqual(spec.to_dict()["layers"], layers)

    def test_the_body_must_add_up(self):
        """
        Feature: layer counts.
        Description: The groups hold 39 body layers against 40 declared.
        Expectation: Refused, naming both counts.
        """
        layers = [{"kind": "decoder", "count": 39}, {"kind": "decoder", "count": 1, "mtp": True}]
        with self.assertRaises(ModelSpecError) as ctx:
            ModelSpec.from_dict(self._with_layers(layers))
        self.assertIn("39", str(ctx.exception))
        self.assertIn("40", str(ctx.exception))

    def test_the_mtp_layers_must_add_up(self):
        """
        Feature: layer counts.
        Description: No MTP group although mtp_depth is 1.
        Expectation: Refused, naming mtp_depth.
        """
        with self.assertRaises(ModelSpecError) as ctx:
            ModelSpec.from_dict(self._with_layers([{"kind": "decoder", "count": 40}]))
        self.assertIn("mtp_depth", str(ctx.exception))

    def test_mtp_layers_come_last(self):
        """
        Feature: layer order.
        Description: An MTP group before a body group.
        Expectation: Refused.
        """
        layers = [
            {"kind": "decoder", "count": 20},
            {"kind": "decoder", "count": 1, "mtp": True},
            {"kind": "decoder", "count": 20},
        ]
        with self.assertRaises(ModelSpecError):
            ModelSpec.from_dict(self._with_layers(layers))

    def test_malformed_groups_raise(self):
        """
        Feature: LayerGroup.
        Description: An empty group, a missing kind, an unknown key, a string mtp.
        Expectation: Each refused.
        """
        for bad in (
            [{"kind": "decoder", "count": 0}],
            [{"count": 40}],
            [{"kind": "decoder", "count": 40, "flavour": "moe"}],
            [{"kind": "decoder", "count": 40, "mtp": "yes"}],
            [],
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ModelSpecError):
                    ModelSpec.from_dict(self._with_layers(bad))

    def test_linear_dimensions_are_fields(self):
        """
        Feature: linear-attention fields.
        Description: A Qwen3.5 spec declares its gated-DeltaNet dimensions.
        Expectation: Typed fields, and absent when not declared.
        """
        data = _qwen35()
        data.update(linear_num_key_heads=16, linear_key_head_dim=128, linear_num_value_heads=32,
                    linear_value_head_dim=128, linear_conv_kernel_dim=4)
        spec = ModelSpec.from_dict(data)
        self.assertEqual(spec.linear_num_value_heads, 32)
        self.assertNotIn("linear_num_value_heads", ModelSpec.from_dict(_dense()).to_dict())


class TestNormalizedConfigAccessor(unittest.TestCase):
    """NormalizedConfig stores the mapping and hands out the typed spec."""

    @staticmethod
    def _config(**model):
        from hyper_parallel.auto_parallel.config_adapter._normalized_config import (
            NormalizedConfig,
        )
        return NormalizedConfig(model_spec=model)

    def test_model_returns_the_typed_spec(self):
        """The mapping a caller filled comes back validated and typed."""
        spec = self._config(**_qwen35()).model()
        self.assertIsInstance(spec, ModelSpec)
        self.assertEqual(spec.effective_head_dim, 256)

    def test_model_raises_on_an_incomplete_section(self):
        """A section the cost model cannot use fails here, by name."""
        data = _dense()
        del data["vocab_size"]
        with self.assertRaises(ModelSpecError) as ctx:
            self._config(**data).model()
        self.assertIn("vocab_size", str(ctx.exception))

    def test_model_leaves_the_run_keys_to_the_adapter(self):
        """The adapter's own keys in the mapping do not stop the typed spec."""
        spec = self._config(**_dense(), local_batch_size=1, compute_dtype="bfloat16").model()
        self.assertEqual(spec.num_hidden_layers, _dense()["num_hidden_layers"])

if __name__ == "__main__":
    unittest.main()
