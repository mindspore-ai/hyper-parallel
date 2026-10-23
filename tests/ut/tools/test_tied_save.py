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
"""The tied-save predictor separates tied models from untied ones.

The prediction is checked against the behaviour it predicts: safetensors is
asked to write the same state dict, and must refuse exactly when the script
says it will.
"""

import importlib.util
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
           / "check_tied_save.py")
_SPEC = importlib.util.spec_from_file_location("check_tied_save", _SCRIPT)
tied = importlib.util.module_from_spec(_SPEC)
sys.modules["check_tied_save"] = tied
_SPEC.loader.exec_module(tied)

_TINY = {"vocab_size": 64, "hidden_size": 16, "intermediate_size": 32,
         "num_hidden_layers": 2, "num_attention_heads": 2,
         "num_key_value_heads": 2, "max_position_embeddings": 32}


def _build(tie: bool):
    config = LlamaConfig.from_dict(dict(_TINY, tie_word_embeddings=tie))
    return AutoModelForCausalLM.from_config(config)


class TestTiedSavePredictor(unittest.TestCase):
    """Aliased-parameter detection matches what safetensors actually does."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tied_model_is_flagged_and_safetensors_refuses_it(self):
        model = _build(tie=True)
        groups = tied.aliased_parameters(model)
        self.assertTrue(groups, "tied embeddings must be detected")
        names = {name for group in groups.values() for name in group}
        self.assertIn("lm_head.weight", names)

        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            target = str(Path(tmp) / "model.safetensors")
            with self.assertRaises(RuntimeError) as caught:
                save_file(model.state_dict(), target)
            self.assertIn("share memory", str(caught.exception).lower())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_untied_model_is_clean_and_safetensors_accepts_it(self):
        model = _build(tie=False)
        self.assertEqual(tied.aliased_parameters(model), {})

        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            target = str(Path(tmp) / "model.safetensors")
            state = {k: v.contiguous() for k, v in model.state_dict().items()}
            save_file(state, target)
            self.assertTrue(Path(target).is_file())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_meta_device_parameters_are_grouped_by_identity(self):
        """On meta there is no storage pointer; identity must still group."""
        config = LlamaConfig.from_dict(dict(_TINY, tie_word_embeddings=True))
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
        groups = tied.aliased_parameters(model)
        self.assertTrue(groups, "meta build must still reveal the tie")
