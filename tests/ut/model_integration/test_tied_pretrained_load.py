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
"""Deferred pretrained loading must keep tied weights one storage.

Regression for a defect reproduced independently by five clean-room audits:
a config-first meta build never re-tied ``_tied_weights_keys`` (current
Transformers resolves tying outside ``post_init``), and ``to_empty``
materializes every module's view of a shared parameter separately — so a
tied-embedding checkpoint crashed the loader's alias validation with
"Tied parameters no longer share local storage after distributed wrapping"
on a plain single-process CPU load.
"""

import unittest
from tempfile import TemporaryDirectory

import torch
from torch import nn
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from hyper_parallel.models._transformers.auto_model import HyperAutoModelForCausalLM
from hyper_parallel.models._transformers.model_builder import (
    _restore_parameter_aliases,
    _tied_parameter_groups,
)
from tests.common.mark_utils import arg_mark


def _tiny_tied_config() -> LlamaConfig:
    return LlamaConfig(  # pylint: disable=unexpected-keyword-arg
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )


class _SharedHead(nn.Module):
    """Two modules deliberately sharing one weight object."""

    def __init__(self) -> None:
        """Tie the head's weight to the embed's at construction."""
        super().__init__()
        self.embed = nn.Linear(4, 4, bias=False)
        self.head = nn.Linear(4, 4, bias=False)
        self.head.weight = self.embed.weight


class TestTiedAliasRestoration(unittest.TestCase):
    """The materialization helpers restore exactly the observed tying."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_split_alias_is_restored(self):
        module = _SharedHead()
        groups = _tied_parameter_groups(module)
        self.assertEqual(sorted(groups[0]), ["embed.weight", "head.weight"])
        # Whether a device move preserves sharing depends on torch internals
        # and on what wrapped the model first; simulate the observed split
        # and require the restoration to repair it either way.
        module.head.weight = nn.Parameter(torch.empty_like(module.embed.weight))
        self.assertIsNot(module.head.weight, module.embed.weight)
        _restore_parameter_aliases(module, groups)
        self.assertIs(module.head.weight, module.embed.weight)
        self.assertEqual(module.head.weight.untyped_storage().data_ptr(),
                         module.embed.weight.untyped_storage().data_ptr())

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_untied_module_untouched(self):
        module = _SharedHead()
        module.head.weight = nn.Parameter(module.embed.weight.detach().clone())
        self.assertEqual(_tied_parameter_groups(module), [])
        _restore_parameter_aliases(module, [])
        self.assertIsNot(module.head.weight, module.embed.weight)


class TestTiedPretrainedLoad(unittest.TestCase):
    """End-to-end: deferred load of a tied checkpoint matches plain HF."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_deferred_load_keeps_tie_and_matches_reference(self):
        torch.manual_seed(7)
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            source = LlamaForCausalLM(_tiny_tied_config())
            source.save_pretrained(tmp, safe_serialization=True)

            model = HyperAutoModelForCausalLM.from_pretrained(
                tmp, torch_dtype="float32", attn_implementation="eager",
                local_files_only=True).eval()
            reference = AutoModelForCausalLM.from_pretrained(
                tmp, dtype=torch.float32, attn_implementation="eager",
                local_files_only=True).eval()

        embed = model.get_parameter("model.embed_tokens.weight")
        head = model.get_parameter("lm_head.weight")
        self.assertIs(head, embed)
        self.assertEqual(head.untyped_storage().data_ptr(),
                         embed.untyped_storage().data_ptr())

        model_state = dict(model.named_parameters())
        for name, expected in reference.named_parameters():
            self.assertIn(name, model_state)
            self.assertTrue(torch.equal(model_state[name], expected),
                            f"parameter mismatch after deferred load: {name}")

        tokens = torch.arange(8, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            ours = model(input_ids=tokens).logits
            theirs = reference(input_ids=tokens).logits
        self.assertTrue(torch.equal(ours, theirs))


if __name__ == "__main__":
    unittest.main()
