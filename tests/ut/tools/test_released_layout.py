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
"""Model-agnostic released-layout pin: the skill script on a tiny model.

The fixtures build a tiny Llama configuration on the meta device, then play
the released side with a real (tiny) safetensors shard plus the standard
index file, so every comparison path runs: name sets, shard-header shapes,
and the failure modes (missing entry, invented entry, shape drift).
"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, LlamaConfig

from tests.common.mark_utils import arg_mark

_SCRIPT = (Path(__file__).resolve().parents[3]
           / ".agent" / "skills" / "hf-model-integration" / "scripts"
           / "check_released_layout.py")
_SPEC = importlib.util.spec_from_file_location("check_released_layout", _SCRIPT)
layout = importlib.util.module_from_spec(_SPEC)
sys.modules["check_released_layout"] = layout
_SPEC.loader.exec_module(layout)

_TINY = {
    "vocab_size": 64,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "max_position_embeddings": 32,
    "tie_word_embeddings": False,
}


def _write_release(model_dir: Path) -> dict:
    """Materialize a tiny model and write config + shard + index files."""
    config = LlamaConfig.from_dict(dict(_TINY))
    model = AutoModelForCausalLM.from_config(config)
    state = {name: tensor.contiguous() for name, tensor in model.state_dict().items()}
    (model_dir / "config.json").write_text(config.to_json_string(), encoding="utf-8")
    shard = "model-00001-of-00001.safetensors"
    save_file(state, str(model_dir / shard))
    index = {"metadata": {"total_size": sum(t.numel() * t.element_size()
                                            for t in state.values())},
             "weight_map": {name: shard for name in state}}
    (model_dir / layout.INDEX_NAME).write_text(json.dumps(index), encoding="utf-8")
    return state


class TestReleasedLayoutScript(unittest.TestCase):
    """The layout pin passes on a faithful release and names each violation."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_faithful_release_passes_with_shapes(self):
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            _write_release(model_dir)
            released = layout.released_entries(model_dir)
            self.assertTrue(released)
            self.assertTrue(all(shape is not None for shape in released.values()),
                            "shard headers must supply every shape")
            self.assertEqual(layout.main(["--model-dir", str(model_dir)]), 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_missing_and_invented_entries_fail(self):
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            state = _write_release(model_dir)
            index = json.loads((model_dir / layout.INDEX_NAME).read_text(encoding="utf-8"))
            victim = next(iter(state))
            del index["weight_map"][victim]
            index["weight_map"]["model.layers.9.invented.weight"] = (
                "model-00001-of-00001.safetensors")
            (model_dir / layout.INDEX_NAME).write_text(json.dumps(index), encoding="utf-8")
            (model_dir / "model-00001-of-00001.safetensors").unlink()  # names only
            released = layout.released_entries(model_dir)
            problems = layout.compare(released, layout.built_entries(model_dir))
            text = "\n".join(problems)
            self.assertIn("released entries absent", text)
            self.assertIn(victim, "\n".join(problems) + victim)  # build still has it
            self.assertIn("invented", text)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_shape_drift_is_reported(self):
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            _write_release(model_dir)
            released = layout.released_entries(model_dir)
            victim = next(name for name, shape in released.items()
                          if shape is not None and len(shape) == 2)
            released[victim] = (released[victim][0] + 1, released[victim][1])
            problems = layout.compare(released, layout.built_entries(model_dir))
            self.assertTrue(any("shape drift" in p for p in problems))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tied_release_passes(self):
        """A tied-embedding release (no lm_head entry) is not a false alarm."""
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            tied = dict(_TINY)
            tied["tie_word_embeddings"] = True
            config = LlamaConfig.from_dict(tied)
            model = AutoModelForCausalLM.from_config(config)
            state = {name: tensor.contiguous()
                     for name, tensor in model.state_dict().items()}
            state.pop("lm_head.weight")  # HF releases store the tensor once
            (model_dir / "config.json").write_text(config.to_json_string(),
                                                   encoding="utf-8")
            shard = "model-00001-of-00001.safetensors"
            save_file(state, str(model_dir / shard))
            index = {"metadata": {"total_size": 0},
                     "weight_map": {name: shard for name in state}}
            (model_dir / layout.INDEX_NAME).write_text(json.dumps(index),
                                                       encoding="utf-8")
            entries, aliases, _ = layout.built_entries_and_aliases(model_dir)
            self.assertIn("lm_head.weight", entries)
            self.assertIn("lm_head.weight", aliases)
            self.assertEqual(layout.main(["--model-dir", str(model_dir)]), 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fused_expert_reconciliation_matches_and_names_violations(self):
        """Per-expert released weights pair with stacked/concatenated builds."""
        released = {"model.norm.weight": (16,)}
        for layer in (1, 2):
            for i in range(4):
                pre = f"model.layers.{layer}.mlp.experts.{i}"
                released[f"{pre}.gate_proj.weight"] = (3, 8)
                released[f"{pre}.up_proj.weight"] = (3, 8)
                released[f"{pre}.down_proj.weight"] = (8, 3)
        built = {"model.norm.weight": (16,)}
        for layer in (1, 2):
            built[f"model.layers.{layer}.mlp.experts.gate_up_proj"] = (4, 6, 8)
            built[f"model.layers.{layer}.mlp.experts.down_proj"] = (4, 8, 3)
        rel_rest, built_rest, notes, problems = layout.reconcile_fused_experts(
            released, built)
        self.assertEqual(problems, [])
        self.assertEqual(set(rel_rest), {"model.norm.weight"})
        self.assertEqual(set(built_rest), {"model.norm.weight"})
        self.assertTrue(notes and "2 expert groups" in notes[0])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fused_expert_shape_drift_and_missing_expert_fail(self):
        base = {}
        for i in range(4):
            pre = f"model.layers.1.mlp.experts.{i}"
            base[f"{pre}.gate_proj.weight"] = (3, 8)
            base[f"{pre}.up_proj.weight"] = (3, 8)
            base[f"{pre}.down_proj.weight"] = (8, 3)
        fused = {"model.layers.1.mlp.experts.gate_up_proj": (4, 6, 8),
                 "model.layers.1.mlp.experts.down_proj": (4, 8, 3)}
        drifted = dict(fused)
        drifted["model.layers.1.mlp.experts.gate_up_proj"] = (4, 7, 8)
        _, _, _, problems = layout.reconcile_fused_experts(base, drifted)
        self.assertTrue(any("gate_up_proj" in p for p in problems))
        short = dict(base)
        del short["model.layers.1.mlp.experts.3.down_proj.weight"]
        _, _, _, problems = layout.reconcile_fused_experts(short, fused)
        self.assertTrue(any("down_proj" in p for p in problems))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fused_expert_names_only_mode_matches_counts(self):
        """Without shard headers (shapes None) counts still reconcile."""
        released = {}
        for i in range(4):
            pre = f"model.layers.1.mlp.experts.{i}"
            for leaf in ("gate_proj", "up_proj", "down_proj"):
                released[f"{pre}.{leaf}.weight"] = None
        built = {"model.layers.1.mlp.experts.gate_up_proj": (4, 6, 8),
                 "model.layers.1.mlp.experts.down_proj": (4, 8, 3)}
        rel_rest, built_rest, notes, problems = layout.reconcile_fused_experts(
            released, built)
        self.assertEqual(problems, [])
        self.assertEqual(rel_rest, {})
        self.assertEqual(built_rest, {})
        self.assertTrue(any("name only" in n for n in notes))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_auto_class_chain_falls_through_to_a_supporting_class(self):
        """A config the causal-LM head rejects still builds via the chain."""
        config = LlamaConfig.from_dict(dict(_TINY))

        class _Rejecting:
            """Stand-in for an auto-class that does not support this config."""

            @staticmethod
            def from_config(_config):
                raise ValueError("Unrecognized configuration class\nsecond line")

        built = layout._meta_build(config, torch, _Rejecting)
        self.assertTrue(built.state_dict())
        self.assertIn("AutoModelForImageTextToText", layout.AUTO_CLASS_NAMES)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_declared_architecture_wins_over_a_buildable_auto_class(self):
        """config.architectures decides, so a subset build cannot shadow it."""
        config = LlamaConfig.from_dict(dict(_TINY))
        config.architectures = ["LlamaForCausalLM"]
        marker = object()

        class _SubsetBuild:
            """An auto-class that would build something else entirely."""

            @staticmethod
            def from_config(_config):
                return marker

        built = layout._meta_build(config, torch, _SubsetBuild)
        self.assertIsNot(built, marker, "declared architecture must be used")
        self.assertTrue(built.state_dict())

        config.architectures = ["NoSuchArchitectureV9"]
        self.assertIs(layout._meta_build(config, torch, _SubsetBuild), marker,
                      "an unexported architecture must fall back")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_auto_class_chain_reports_every_failure(self):
        """When nothing builds it, the error names each class tried."""

        class _Broken:
            @staticmethod
            def from_config(_config):
                raise ValueError("nope")

        original = layout.AUTO_CLASS_NAMES
        layout.AUTO_CLASS_NAMES = ()
        try:
            with self.assertRaises(ValueError) as caught:
                layout._meta_build(object(), torch, _Broken)
        finally:
            layout.AUTO_CLASS_NAMES = original
        self.assertIn("_Broken: nope", str(caught.exception))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_wholesale_prefix_rename_reconciles(self):
        """A moved subtree pairs by suffix+shape instead of failing twice."""
        released = {"model.embed_tokens.weight": (64, 16),
                    "model.layers.0.mlp.up_proj.weight": (32, 16),
                    "visual.patch_embed.weight": (8, 3),
                    "lm_head.weight": (64, 16)}
        built = {"model.language_model.embed_tokens.weight": (64, 16),
                 "model.language_model.layers.0.mlp.up_proj.weight": (32, 16),
                 "model.visual.patch_embed.weight": (8, 3),
                 "lm_head.weight": (64, 16)}
        self.assertTrue(layout.compare(released, built), "strict must fail first")
        rel_rest, built_rest, notes, problems = layout.reconcile_renamed_prefixes(
            released, built)
        self.assertEqual(problems, [])
        self.assertEqual(rel_rest, {})
        self.assertEqual(built_rest, {})
        self.assertTrue(any("rename reconciliation" in n for n in notes))
        self.assertTrue(any("model -> model.language_model" in n for n in notes),
                        notes)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rename_reconciles_in_names_only_mode(self):
        """Without shard headers the released side has no shapes; still pairs."""
        released = {"model.embed_tokens.weight": None,
                    "model.layers.0.mlp.up_proj.weight": None,
                    "visual.patch_embed.weight": None}
        built = {"model.language_model.embed_tokens.weight": (64, 16),
                 "model.language_model.layers.0.mlp.up_proj.weight": (32, 16),
                 "model.visual.patch_embed.weight": (8, 3)}
        rel_rest, built_rest, notes, problems = layout.reconcile_renamed_prefixes(
            released, built)
        self.assertEqual(problems, [])
        self.assertEqual((rel_rest, built_rest), ({}, {}))
        self.assertTrue(any("shapes unverified" in n for n in notes), notes)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rename_pairing_still_reports_real_absence_and_drift(self):
        """Renaming never hides a missing entry or a changed shape."""
        released = {"model.embed_tokens.weight": (64, 16),
                    "model.layers.0.mlp.up_proj.weight": (32, 16)}
        missing = {"model.language_model.embed_tokens.weight": (64, 16)}
        rel_rest, built_rest, _, _ = layout.reconcile_renamed_prefixes(
            released, missing)
        self.assertIn("model.layers.0.mlp.up_proj.weight", rel_rest)
        self.assertTrue(layout.compare(rel_rest, built_rest))

        drifted = {"model.language_model.embed_tokens.weight": (64, 16),
                   "model.language_model.layers.0.mlp.up_proj.weight": (33, 16)}
        rel_rest, built_rest, _, _ = layout.reconcile_renamed_prefixes(
            released, drifted)
        self.assertTrue(rel_rest and built_rest,
                        "shape drift must stay unpaired")
        self.assertTrue(layout.compare(rel_rest, built_rest))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_meta_build_loads_no_tensor_data(self):
        """The build side stays on the meta device end to end."""
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            _write_release(model_dir)
            built = layout.built_entries(model_dir)
            self.assertTrue(built)
            config = LlamaConfig.from_dict(dict(_TINY))
            with torch.device("meta"):
                reference = AutoModelForCausalLM.from_config(config)
            self.assertEqual(set(built), set(reference.state_dict()))


if __name__ == "__main__":
    unittest.main()


class TestDeclaredIgnoredSubtree(unittest.TestCase):
    """A subtree the architecture declares it will not load is labelled, not failed.

    Field case: a released 9B checkpoint carries 15 `mtp.*` tensors for a
    prediction head the modelling code does not implement, and the class says
    so itself with `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`. Calling
    that "absent from the build" reports upstream's own contract as a defect
    and buries any real difference behind a known one.
    """

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_released_only_entries_matching_the_declaration_are_split_out(self):
        released = {"model.norm.weight": (16,), "mtp.fc.weight": (16, 16),
                    "mtp.layers.0.mlp.up_proj.weight": (32, 16)}
        built = {"model.norm.weight": (16,)}
        kept, declared = layout.split_declared_ignored(released, built,
                                                       (r"^mtp.*",))
        self.assertEqual(set(declared), {"mtp.fc.weight",
                                         "mtp.layers.0.mlp.up_proj.weight"})
        self.assertEqual(set(kept), {"model.norm.weight"})
        self.assertEqual(layout.compare(kept, built), [])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_a_declaration_never_excuses_an_entry_the_build_has(self):
        """Shape drift stays a failure however broadly the patterns read."""
        released = {"mtp.fc.weight": (16, 16)}
        built = {"mtp.fc.weight": (8, 16)}
        kept, declared = layout.split_declared_ignored(released, built,
                                                       (r".*",))
        self.assertEqual(declared, {})
        self.assertTrue(layout.compare(kept, built))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_entries_the_build_invented_are_still_a_failure(self):
        """The split only ever removes released-only entries."""
        released = {"mtp.fc.weight": (16, 16)}
        built = {"invented.weight": (4,)}
        kept, declared = layout.split_declared_ignored(released, built,
                                                       (r"^mtp.*",))
        self.assertEqual(set(declared), {"mtp.fc.weight"})
        problems = layout.compare(kept, built)
        self.assertTrue(any("invented" in p for p in problems), problems)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_no_declaration_changes_nothing(self):
        released = {"mtp.fc.weight": (16, 16)}
        kept, declared = layout.split_declared_ignored(released, {}, ())
        self.assertEqual(declared, {})
        self.assertEqual(kept, released)
