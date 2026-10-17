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
"""Tests for the op profiles and how a spec settles the one it is priced with."""
import contextlib
import os
import tempfile
import unittest
from unittest.mock import patch

from hyper_parallel.auto_parallel import _op_profiles
from hyper_parallel.auto_parallel._hf_model_spec import resolve_hf_model_spec
from hyper_parallel.auto_parallel._model_spec import ModelSpecError, OpCounts
from hyper_parallel.auto_parallel._op_profiles import (
    DEFAULT_ARCH,
    DEFAULT_RUN,
    family_profile,
    infer_arch,
    known_archs,
    load_op_profile,
    resolve_ops,
)


def _counts(**overrides) -> dict:
    """The default decoder's op counts, with *overrides* applied."""
    counts = {
        "attMM": 4, "attBMM": 2, "ffMM": 3, "softmax": 1,
        "dropout": 0, "normOp": 2, "gather": 4, "headCast": 1, "ffAct": 1,
        "linrec": 0,
    }
    counts.update(overrides)
    return counts


def _overrides(name: str = "unit", **extra) -> dict:
    """A ``model`` section the offline config_overrides path can resolve."""
    overrides = {
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "vocab_size": 32000,
    }
    overrides.update(extra)
    return {"name": name, "config_overrides": overrides}


class TestProfiles(unittest.TestCase):
    """Every family's op counts are a file, and every file is a valid vector."""

    def test_the_families(self):
        """Six families the arch hooks knew, Qwen3.5, the default and the tower."""
        self.assertEqual(set(known_archs()), {
            "default", "llama2", "mixtral", "t5", "pangualpha",
            "deepseek", "qwen", "qwen3_5", "vision",
        })

    def test_every_profile_loads(self):
        """Each profile declares at least one kind, each a full OpCounts."""
        for arch in known_archs():
            with self.subTest(arch=arch):
                profile = load_op_profile(arch)
                self.assertEqual(profile.arch, arch)
                self.assertTrue(profile.kinds)
                for kind in profile.layer_kinds.values():
                    self.assertIsInstance(kind.ops, OpCounts)
                    # Only a linear-attention layer runs the delta rule's state update.
                    self.assertEqual(kind.ops.linrec, int(kind.attention == "linear"))

    def test_unknown_arch_raises_naming_the_profiles(self):
        """A misspelt arch is refused, and the message lists the real ones."""
        with self.assertRaises(ModelSpecError) as ctx:
            load_op_profile("qwne")
        self.assertIn("qwne", str(ctx.exception))
        self.assertIn("qwen", str(ctx.exception))

    def test_unknown_kind_raises_naming_the_kinds(self):
        """Asking a profile for a kind it lacks names the kinds it has."""
        self.assertEqual(load_op_profile("t5").counts("decoder").attMM, 8)
        with self.assertRaises(ModelSpecError) as ctx:
            load_op_profile("t5").counts("block")
        self.assertIn("encoder", str(ctx.exception))


@contextlib.contextmanager
def _profile_file(text: str):
    """Load profiles from a folder holding one ``unit.yaml`` with *text*."""
    with tempfile.TemporaryDirectory() as folder:
        with open(os.path.join(folder, "unit.yaml"), "w", encoding="utf-8") as handle:
            handle.write(text)
        load_op_profile.cache_clear()
        known_archs.cache_clear()
        try:
            with patch.object(_op_profiles, "PROFILE_DIR", folder):
                yield
        finally:
            load_op_profile.cache_clear()
            known_archs.cache_clear()


class TestKindsAndFlavours(unittest.TestCase):
    """A kind states how its layers are shaped, from a closed set."""

    def test_deepseek_kinds_differ_in_their_feed_forward_only(self):
        """
        Feature: ffn flavour.
        Description: DeepSeek's dense and MoE layers run the same ops.
        Expectation: Two kinds, dense and moe flavours, equal counts, moe by default.
        """
        profile = load_op_profile("deepseek")
        self.assertEqual((profile.kind("dense").ffn, profile.kind("moe").ffn), ("dense", "moe"))
        self.assertEqual(profile.counts("dense"), profile.counts("moe"))
        self.assertEqual(profile.default, "moe")

    def test_attention_is_full_unless_stated(self):
        """
        Feature: attention flavour.
        Description: Only Qwen3.5's linear-attention kind declares linear attention.
        Expectation: Every other kind is full attention, and every kind but
            DeepSeek's keeps the model's feed-forward.
        """
        for arch in known_archs():
            for kind in load_op_profile(arch).layer_kinds.values():
                with self.subTest(arch=arch, kind=kind.name):
                    linear = (arch, kind.name) == ("qwen3_5", "linear_attention")
                    self.assertEqual(kind.attention, "linear" if linear else "full")
                    if arch != "deepseek":
                        self.assertIsNone(kind.ffn)

    def test_qwen3_5_linear_layers_keep_no_score_matrix(self):
        """
        Feature: Qwen3.5 profile.
        Description: A linear-attention layer against a full one.
        Expectation: No score matmuls, softmax or score casts, one state
            update, and every other op as in full attention.
        """
        profile = load_op_profile("qwen3_5")
        full, linear = profile.counts("full_attention"), profile.counts("linear_attention")
        self.assertEqual(profile.default, "full_attention")
        self.assertEqual((linear.attBMM, linear.softmax, linear.headCast, linear.linrec), (0, 0, 0, 1))
        self.assertEqual(full, load_op_profile("qwen").counts("decoder"))
        for name in ("attMM", "ffMM", "dropout", "normOp", "gather", "ffAct"):
            self.assertEqual(getattr(linear, name), getattr(full, name), name)

    def test_a_single_kind_is_its_own_default(self):
        """
        Feature: default kind.
        Description: A family with one kind, and t5 with two.
        Expectation: The one kind, and the encoder t5 names, whose TP
            gathers the model's embedding and output layer run.
        """
        self.assertEqual(load_op_profile("qwen").default, "decoder")
        self.assertEqual(load_op_profile("t5").default, "encoder")

    def test_an_unknown_flavour_is_refused(self):
        """
        Feature: profile loader.
        Description: A kind declares a flavour outside the closed set.
        Expectation: Refused, naming the flavours there are.
        """
        text = f"kinds:\n  decoder:\n    attention: sliding\n    ops: {_counts()}\n"
        with _profile_file(text):
            with self.assertRaises(ModelSpecError) as ctx:
                load_op_profile("unit")
        self.assertIn("linear", str(ctx.exception))

    def test_a_default_that_is_not_a_kind_is_refused(self):
        """
        Feature: profile loader.
        Description: The default names a kind the profile does not declare.
        Expectation: Refused, naming the kinds.
        """
        text = f"default: moe\nkinds:\n  decoder:\n    ops: {_counts()}\n"
        with _profile_file(text):
            with self.assertRaises(ModelSpecError) as ctx:
                load_op_profile("unit")
        self.assertIn("decoder", str(ctx.exception))


class TestFamilyDefaults(unittest.TestCase):
    """What a family's model has where its producer states nothing."""

    def test_each_family_states_what_its_hook_set(self):
        """
        Feature: run and model defaults.
        Description: The byte widths, gradient accumulation and activation
            sharding the family hooks set in code, and DeepSeek's head width.
        Expectation: Each profile states its own over DEFAULT_RUN, and only
            the MLA family a model default.
        """
        runs = {
            "llama2": {"grad_bytes": 2, "grad_accumulation": True},
            "mixtral": {"grad_bytes": 2},
            "pangualpha": {"dropout_bytes": 1},
            "t5": {"dropout_bytes": 1},
            "qwen": {"shard_activations": True},
            "qwen3_5": {"shard_activations": True},
        }
        models = {"deepseek": {"v_head_dim": 128}}
        for arch in known_archs():
            with self.subTest(arch=arch):
                profile = load_op_profile(arch)
                self.assertEqual(dict(profile.run), {**DEFAULT_RUN, **runs.get(arch, {})})
                self.assertEqual(dict(profile.model), models.get(arch, {}))

    def test_a_run_default_it_cannot_state_is_refused(self):
        """
        Feature: profile loader.
        Description: A run default outside DEFAULT_RUN, and one that is not
            a whole number of bytes.
        Expectation: Refused, naming the key.
        """
        for run, named in (("dp: 2", "dp"), ("grad_bytes: two", "grad_bytes")):
            text = f"run:\n  {run}\nkinds:\n  decoder:\n    ops: {_counts()}\n"
            with self.subTest(run=run), _profile_file(text):
                with self.assertRaises(ModelSpecError) as ctx:
                    load_op_profile("unit")
                self.assertIn(named, str(ctx.exception))

    def test_a_model_default_it_cannot_state_is_refused(self):
        """
        Feature: profile loader.
        Description: A model default other than v_head_dim, and a zero
            value-head width.
        Expectation: Refused, naming the key.
        """
        for model, named in (("hidden_size: 64", "hidden_size"), ("v_head_dim: 0", "v_head_dim")):
            text = f"model:\n  {model}\nkinds:\n  decoder:\n    ops: {_counts()}\n"
            with self.subTest(model=model), _profile_file(text):
                with self.assertRaises(ModelSpecError) as ctx:
                    load_op_profile("unit")
                self.assertIn(named, str(ctx.exception))

    def test_a_name_no_profile_has_is_the_default_family(self):
        """
        Feature: family_profile.
        Description: No arch, an arch no profile has, and a known one.
        Expectation: The default family twice, then the named one.
        """
        got = [family_profile(arch).arch for arch in (None, "gpt", "qwen")]
        self.assertEqual(got, [DEFAULT_ARCH, DEFAULT_ARCH, "qwen"])


class TestInferArch(unittest.TestCase):
    """A free-text model name maps to the family the cost model always chose."""

    def test_names_in_use(self):
        """The model names the repo's configs and tests carry."""
        cases = {
            "deepseekV3": "deepseek",
            "deepseek_v3": "deepseek",
            "qwen2_72b": "qwen",
            "Qwen3": "qwen",
            "qwen3_moe": "qwen",
            "qwen3_vl_moe": "qwen",
            "qwen3_5_moe": "qwen3_5",
            "qwen3_5_moe_text": "qwen3_5",
            "Qwen3.5-35B-A3B": "qwen3_5",
            "mixtral-8x7b": "mixtral",
            "llama2_7b": "llama2",
            "pangualpha_2_6b": "pangualpha",
            "t5_small": "t5",
        }
        for name, arch in cases.items():
            with self.subTest(name=name):
                self.assertEqual(infer_arch(name), arch)

    def test_the_family_a_name_opens_with_wins(self):
        """A name holding two families takes the one it opens with, and one
        holding a family's letters further in takes none: matched
        anywhere, in list order, qwen2_deepseek_distill took DeepSeek's and
        acme_t5 took t5's."""
        self.assertEqual(infer_arch("deepseek_qwen_distill"), "deepseek")
        self.assertEqual(infer_arch("qwen2_deepseek_distill"), "qwen")
        self.assertEqual(infer_arch("llama2_t5_hybrid"), "llama2")
        with patch.object(_op_profiles, "_DEFAULTED_NAMES", set()):
            self.assertEqual(infer_arch("acme_t5"), DEFAULT_ARCH)

    def test_no_match_is_the_default_and_says_so_once(self):
        """A name nothing claims is priced as the default, with one warning a name."""
        with patch.object(_op_profiles, "_DEFAULTED_NAMES", set()):
            with self.assertLogs("hyper_parallel.auto_parallel._op_profiles", "WARNING") as said:
                for _ in range(2):
                    self.assertEqual(infer_arch("llama"), DEFAULT_ARCH)
                self.assertEqual(infer_arch("glm4_moe"), DEFAULT_ARCH)
        self.assertEqual(len(said.output), 2)


class TestResolveOps(unittest.TestCase):
    """Declared counts replace a profile's, but only kind for kind."""

    def test_profile_counts_when_nothing_is_declared(self):
        """Without declared ops the profile's kinds are returned."""
        self.assertEqual(set(resolve_ops("t5")), {"encoder", "decoder"})

    def test_declared_counts_win(self):
        """Counts the spec declares are the ones it is priced with."""
        declared = {"decoder": OpCounts.from_dict(_counts(softmax=0))}
        self.assertEqual(resolve_ops("qwen", declared)["decoder"].softmax, 0)

    def test_declared_kinds_must_match_the_profile(self):
        """Counts for a kind the family never assigns would be dropped."""
        declared = {"block": OpCounts.from_dict(_counts())}
        with self.assertRaises(ModelSpecError) as ctx:
            resolve_ops("qwen", declared)
        self.assertIn("block", str(ctx.exception))
        self.assertIn("decoder", str(ctx.exception))


class TestResolvedSpecStatesItsArch(unittest.TestCase):
    """The producer writes the profile it settled on into the spec."""

    def test_inferred_arch_is_written_out(self):
        """A spec that declares no arch gets the one its name implies."""
        spec = resolve_hf_model_spec(_overrides(name="qwen3_moe"))
        self.assertEqual(spec["arch"], "qwen")
        self.assertNotIn("ops", spec)

    def test_declared_arch_wins_over_the_name(self):
        """An explicit arch is kept even when the name implies another."""
        spec = resolve_hf_model_spec(_overrides(name="qwen3_moe", arch="default"))
        self.assertEqual(spec["arch"], "default")

    def test_unknown_declared_arch_raises(self):
        """A misspelt arch fails at parse time instead of pricing as default."""
        with self.assertRaises(ModelSpecError):
            resolve_hf_model_spec(_overrides(arch="qwne"))

    def test_declared_ops_round_trip(self):
        """Counts declared in config_overrides come back in the spec."""
        ops = {"decoder": _counts(softmax=0)}
        spec = resolve_hf_model_spec(_overrides(arch="default", ops=ops))
        self.assertEqual(spec["ops"], ops)

    def test_declared_ops_for_the_wrong_kinds_raise(self):
        """Counts that no layer of the family would use are refused."""
        with self.assertRaises(ModelSpecError):
            resolve_hf_model_spec(_overrides(arch="t5", ops={"decoder": _counts()}))


if __name__ == "__main__":
    unittest.main()
