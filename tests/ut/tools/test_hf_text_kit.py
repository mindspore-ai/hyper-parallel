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
"""Generic bring-up kit: multi-node launcher plan and pretrained loading."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tests.common.mark_utils import arg_mark

_KIT = (Path(__file__).resolve().parents[3]
        / "examples" / "training_demo" / "hf_text_bringup")


class TestDistLauncherDryRun(unittest.TestCase):
    """DRYRUN prints the all-or-nothing guard plan and per-rank commands."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dryrun_emits_guards_and_per_rank_commands(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash not available on this platform")
        env = dict(os.environ, NODES="nodeA nodeB", DRYRUN="1",
                   MASTER_PORT="29777", NPROC_PER_NODE="8")
        done = subprocess.run(
            [bash, str(_KIT / "run_hf_text_dist.sh"), "/models/m", "7",
             "--model_init_dtype=bfloat16"],
            capture_output=True, text=True, env=env, check=False)
        out = done.stdout
        self.assertEqual(done.returncode, 0, out + done.stderr)
        for needle in ("guard: ssh nodeA", "guard: ssh nodeB",
                       "--nnodes=2", "--node-rank=0", "--node-rank=1",
                       "--master-addr=nodeA", "--master-port=29777",
                       '--training.train_iters="7"',
                       "--model_init_dtype=bfloat16",
                       "post: ssh nodeB"):
            self.assertIn(needle, out, f"missing {needle!r}")


class TestPretrainedLoading(unittest.TestCase):
    """`load_pretrained` returns released weights, not a random init."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_load_pretrained_matches_released_tensors(self):
        pytest.importorskip("hyper_parallel")
        import torch
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, LlamaConfig

        spec = importlib.util.spec_from_file_location(
            "hf_text_build_model", _KIT / "build_model.py")
        build_model = importlib.util.module_from_spec(spec)
        sys.modules["hf_text_build_model"] = build_model
        spec.loader.exec_module(build_model)

        tiny = {"vocab_size": 64, "hidden_size": 16, "intermediate_size": 32,
                "num_hidden_layers": 2, "num_attention_heads": 2,
                "num_key_value_heads": 2, "max_position_embeddings": 32,
                "tie_word_embeddings": False}
        # ignore_cleanup_errors: safetensors maps the shard, and on Windows
        # the mapping outlives the model long enough to block the rmtree.
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            model_dir = Path(tmp)
            config = LlamaConfig.from_dict(dict(tiny))
            released = AutoModelForCausalLM.from_config(config)
            state = {k: v.contiguous() for k, v in released.state_dict().items()}
            (model_dir / "config.json").write_text(config.to_json_string(),
                                                   encoding="utf-8")
            shard = "model-00001-of-00001.safetensors"
            save_file(state, str(model_dir / shard))
            index = {"metadata": {"total_size": 0},
                     "weight_map": {k: shard for k in state}}
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps(index), encoding="utf-8")

            # The Trainer injects its selection object on every build, in its
            # default form; that must not block load_pretrained (it did, twice).
            from hyper_parallel.trainer.config.parallelism import (
                ActivationCheckpointSelection)
            built = build_model.build_hf_causal_lm(
                str(model_dir),
                options=build_model.HfCausalLmOptions(
                    load_pretrained=True, torch_dtype="float32",
                    attn_implementation="eager",
                    activation_checkpoint_selection=ActivationCheckpointSelection()))
            selective = ActivationCheckpointSelection(
                source="model_adapter_safe_regions", layer_count=1)
            for requested in (selective, ["model.layers.0"]):
                with self.assertRaises(ValueError):
                    build_model.build_hf_causal_lm(
                        str(model_dir),
                        options=build_model.HfCausalLmOptions(
                            load_pretrained=True,
                            activation_checkpoint_selection=requested))
            got = built.state_dict()
            self.assertTrue(torch.equal(
                got["model.embed_tokens.weight"].float(),
                state["model.embed_tokens.weight"].float()))
            self.assertTrue(torch.equal(
                got["lm_head.weight"].float(), state["lm_head.weight"].float()))
