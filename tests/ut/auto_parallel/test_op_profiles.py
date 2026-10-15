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
import unittest
from unittest.mock import patch

from hyper_parallel.auto_parallel import _op_profiles
from hyper_parallel.auto_parallel._hf_model_spec import resolve_hf_model_spec
from hyper_parallel.auto_parallel._model_spec import ModelSpecError, OpCounts
from hyper_parallel.auto_parallel._op_profiles import (
    DEFAULT_ARCH,
    family_profile,
    infer_arch,
    known_archs,
    load_op_profile,
    resolve_ops,
)


class TestProfiles(unittest.TestCase):
    """Every family's op counts are a file, and every file is a valid vector."""

    def test_the_families(self):
        """The seven families the arch hooks knew, Qwen3.5, the default and the tower."""
        self.assertEqual(set(known_archs()), {
            "default", "llama2", "mixtral", "t5", "pangualpha",
            "deepseek", "qwen", "qwen3_5", "cm", "vision",
        })

    def test_every_profile_loads(self):
        """Each profile declares at least one kind, each a full OpCounts; every kind is full attention
        but Qwen3.5's linear one, and keeps the model's feed-forward but DeepSeek's and cm's."""
        for arch in known_archs():
            with self.subTest(arch=arch):
                profile = load_op_profile(arch)
                self.assertEqual(profile.arch, arch)
                self.assertTrue(profile.kinds)
                for kind in profile.layer_kinds.values():
                    self.assertIsInstance(kind.ops, OpCounts)
                    # Only a linear-attention layer runs the delta rule's state update.
                    self.assertEqual(kind.ops.linrec, int(kind.attention == "linear"))
                    linear = (arch, kind.name) == ("qwen3_5", "linear_attention")
                    self.assertEqual(kind.attention, "linear" if linear else "full", kind.name)
                    if arch not in ("deepseek", "cm"):
                        self.assertIsNone(kind.ffn, kind.name)

    def test_what_a_profile_cannot_price_is_refused(self):
        """
        Feature: the op profiles' checks.
        Description: A spec declaring a misspelt arch, a kind asked of a profile
            that lacks it, and declared counts for a kind the family never assigns.
        Expectation: Each refused by name: the misspelt arch at parse time,
            rather than priced as the default family.
        """
        model = {"hidden_size": 1024, "num_hidden_layers": 4, "num_attention_heads": 8, "vocab_size": 32000}
        cases = (
            ("qwne", lambda: resolve_hf_model_spec({"name": "unit", "config_overrides": dict(model, arch="qwne")})),
            ("no layer kind 'block'", lambda: load_op_profile("t5").counts("block")),
            ("arch 'qwen' prices", lambda: resolve_ops("qwen", {"block": load_op_profile("qwen").counts("decoder")})),
        )
        for named, refused in cases:
            with self.subTest(named=named), self.assertRaisesRegex(ModelSpecError, named):
                refused()


class TestFamilyDefaults(unittest.TestCase):
    """What a family's model has where its producer states nothing."""

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
        """Qwen3.5's release name takes its own family, not the Qwen one it opens with too."""
        self.assertEqual(infer_arch("Qwen3.5-35B-A3B"), "qwen3_5")

    def test_the_family_a_name_opens_with_wins(self):
        """A name holding two families takes the one it opens with, and one
        holding a family's letters further in takes none (I8): matched
        anywhere, in list order, qwen2_deepseek_distill took DeepSeek's and
        acme_lm took cm's."""
        self.assertEqual(infer_arch("qwen2_deepseek_distill"), "qwen")
        with patch.object(_op_profiles, "_DEFAULTED_NAMES", set()):
            self.assertEqual(infer_arch("acme_lm"), DEFAULT_ARCH)

    def test_no_match_is_the_default_and_says_so_once(self):
        """A name nothing claims is priced as the default, with one warning a name."""
        with patch.object(_op_profiles, "_DEFAULTED_NAMES", set()):
            with self.assertLogs("hyper_parallel.auto_parallel._op_profiles", "WARNING") as said:
                for _ in range(2):
                    self.assertEqual(infer_arch("llama"), DEFAULT_ARCH)
                self.assertEqual(infer_arch("glm4_moe"), DEFAULT_ARCH)
        self.assertEqual(len(said.output), 2)


if __name__ == "__main__":
    unittest.main()
