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
"""Reproduction kit: every planned command resolves, and parity sees buffers."""
# Tests reach into the build pipeline to disable the fix for a positive control.
# pylint: disable=protected-access

import dataclasses
import importlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from examples.training_demo.hf_text_bringup.build_model import HfCausalLmOptions
from examples.training_demo.hf_text_bringup.reproduce import adamw_recipe, parity_check
from tests.common.mark_utils import arg_mark

_ROOT = Path(__file__).resolve().parents[3]
_KIT = _ROOT / "examples" / "training_demo" / "hf_text_bringup"
_DRIVER = _KIT / "reproduce" / "reproduce.sh"
_TEXT_RECIPE = _KIT / "train_hf_text.yaml"
_VLM_RECIPE = _ROOT / "examples" / "training_demo" / "train_vlm.yaml"
_CPP_HELPERS = "hyper_parallel.data.indexed._indexed_helpers_cpp"
_MODEL_KEYS = {"config_path"} | {field.name for field in dataclasses.fields(HfCausalLmOptions)}
_CELL = re.compile(r"^\[([A-Za-z0-9_]+)\] (.*)$")


def _bash() -> str:
    bash = shutil.which("bash")
    if bash is None:
        raise unittest.SkipTest("bash not available on this platform")
    return bash


def _dry_run(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, DRYRUN="1", MODEL_ROOT="/models", OUT="/out", NODES="nodeA nodeB")
    env.pop("S3_EVIDENCE", None)
    return subprocess.run([_bash(), str(_DRIVER), *args], capture_output=True, text=True,
                          env=env, check=False)


def _plans():
    """Yield (model, stage, stdout) for every model and stage the driver lists."""
    listing = _dry_run("list")
    assert listing.returncode == 0, listing.stderr
    for line in listing.stdout.splitlines()[1:]:
        model, _hub, *stages = line.split()
        for stage in stages:
            done = _dry_run(model, stage)
            assert done.returncode == 0, f"{model} {stage}: {done.stderr}"
            yield model, stage, done.stdout


def _launches(stdout: str):
    """Yield (cell, env, script, args) for every printed launcher command."""
    for line in stdout.splitlines():
        match = _CELL.match(line)
        if not match:
            continue
        tokens = shlex.split(match.group(2))
        env = {}
        if tokens[0] == "env":
            tokens = tokens[1:]
            while "=" in tokens[0] and not tokens[0].startswith("-"):
                key, value = tokens.pop(0).split("=", 1)
                env[key] = value
        if tokens[0].startswith("NODES="):
            tokens = tokens[1:]
        if tokens[0] != "bash":
            continue
        yield match.group(1), env, Path(tokens[1]).name, tokens[2:]


def _last_value(args: list[str], key: str) -> Optional[str]:
    """The value a repeated ``--key=value`` override ends with (the last one wins)."""
    values = [item.split("=", 1)[1] for item in args if item.startswith(f"--{key}=")]
    return values[-1] if values else None


_REAL_TO_EMPTY = torch.nn.Module.to_empty


def _poisoned_to_empty(self, *, device, recurse=True):
    """to_empty whose fresh storage holds a sentinel instead of allocator leftovers.

    Uninitialized memory can by chance hold the right bytes (a freed tensor
    of the same size is often handed straight back), which makes a check of
    "was this restored" pass or fail by luck. Poisoning makes it decide.
    """
    module = _REAL_TO_EMPTY(self, device=device, recurse=recurse)
    with torch.no_grad():
        for tensor in (*module.parameters(), *module.buffers()):
            tensor.fill_(float("nan") if tensor.is_floating_point() else -1)
    return module


def _stub_cpp_helpers():
    """Stand in for the compiled dataset helpers config resolution imports."""
    try:
        importlib.import_module(_CPP_HELPERS)
        return {}
    except ImportError:
        stub = types.ModuleType(_CPP_HELPERS)
        for name in ("build_blending_indices", "build_sample_index_int32", "build_sample_index_int64"):
            setattr(stub, name, lambda *args, **kwargs: None)
        return {_CPP_HELPERS: stub}


class TestReproducePlans(unittest.TestCase):
    """Every command the driver would run resolves against its recipe."""

    @classmethod
    def setUpClass(cls) -> None:
        """Dry-run every model and stage once for all tests in the class."""
        cls.plans = list(_plans())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_every_override_resolves(self):
        """Feature: no typo reaches a device.

        Description: Resolve each planned launch with the real trainer config parser.
        Expectation: All overrides resolve, and model options name real builder fields.
        """
        from hyper_parallel.trainer.config.parser import (  # pylint: disable=import-outside-toplevel
            load_training_config,
        )
        resolved = 0
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, _stub_cpp_helpers()):
            adamw = Path(tmp) / "train_adamw.yaml"
            adamw_recipe.main(["--output", str(adamw)])
            for model, stage, stdout in self.plans:
                for cell, env, script, args in _launches(stdout):
                    with self.subTest(model=model, stage=stage, cell=cell):
                        if script == "run_vlm.sh":
                            nproc = env.get("NPROC_PER_NODE", "8")
                            recipe = _VLM_RECIPE
                            overrides = ["--model.pretrained_model_name_or_path=/m",
                                         "--dataset.model_assets.pretrained_model_name_or_path=/m",
                                         "--dataset.data_path=/d", f"--training.global_batch_size={nproc}",
                                         f"--fsdp_config.dp_shard_size={nproc}", *args[2:]]
                        else:
                            recipe = adamw if "RECIPE" in env else _TEXT_RECIPE
                            overrides = ["--model.config_path=/m", "--dataset.data_path=/d",
                                         "--dataset.model_assets.tokenizer.pretrained_model_name_or_path=/m",
                                         f"--training.train_iters={args[1]}", *args[2:]]
                            model_keys = {item.split("=", 1)[0][len("--model."):]
                                          for item in overrides if item.startswith("--model.")}
                            self.assertLessEqual(model_keys, _MODEL_KEYS)
                        load_training_config(recipe, overrides)
                        resolved += 1
        self.assertGreater(resolved, 40)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_equivalence_cells_hold_inputs_fixed(self):
        """Feature: S3 compares layouts, not inputs.

        Description: Inspect every S3 plan's cells and judged pairs.
        Expectation: Text cells load released weights; judged pairs share global batch and data.
        """
        checked = 0
        for model, stage, stdout in self.plans:
            if not stage.startswith("s3"):
                continue
            cells = {}
            for cell, env, script, args in _launches(stdout):
                if script == "run_vlm.sh":
                    # The VLM launcher takes the data positionally and sets the
                    # global batch to its process count before the overrides.
                    nproc = env.get("NPROC_PER_NODE", "8")
                    args = [f"--training.global_batch_size={nproc}", f"--dataset.data_path={args[1]}",
                            *args[2:]]
                cells[cell] = args
            for line in stdout.splitlines():
                pair = re.search(r"--(?:early-)?identical=([A-Za-z0-9_]+)\\?,([A-Za-z0-9_]+)", line)
                if not pair:
                    continue
                with self.subTest(model=model, stage=stage, pair=pair.groups()):
                    first, second = (cells[name] for name in pair.groups())
                    for key in ("training.global_batch_size", "dataset.data_path", "training.seed"):
                        self.assertEqual(_last_value(first, key), _last_value(second, key), key)
                    if model != "qwen25-vl-3b":
                        self.assertEqual(_last_value(first, "model.load_pretrained"), "true")
                    checked += 1
        self.assertGreater(checked, 10)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unknown_model_or_stage_is_refused(self):
        """Feature: the driver fails loudly on a bad request.

        Description: Ask for a model and a stage that do not exist.
        Expectation: Non-zero exit for both.
        """
        self.assertNotEqual(_dry_run("no-such-model", "s1").returncode, 0)
        self.assertNotEqual(_dry_run("smollm2-360m", "s9").returncode, 0)


class TestParityCheck(unittest.TestCase):
    """The S2 parity builds through the training path and compares buffers."""

    def _tiny_llama(self, directory: str) -> None:
        config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32)
        torch.manual_seed(0)
        LlamaForCausalLM(config).save_pretrained(directory)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_kit_path_matches_transformers_bitwise(self):
        """Feature: S2 parity on the pretrained construction path.

        Description: Run the parity on a tiny released Llama.
        Expectation: Parameters, buffers and logits all match bitwise.
        """
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            self._tiny_llama(directory)
            with patch.object(torch.nn.Module, "to_empty", _poisoned_to_empty):
                report = parity_check.run_parity(directory, dtype="float32", seq_len=12)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["buffers"]["compared"], 2)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_uninitialized_buffers_are_caught(self):
        """Feature: positive control for the buffer comparison.

        Description: Disable the constructed-buffer recipe, as before the fix.
        Expectation: Parity fails and names the RoPE buffers while parameters still match.
        """
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            self._tiny_llama(directory)
            with patch("hyper_parallel.models._transformers.model_builder._declare_constructed_buffers"), \
                    patch.object(torch.nn.Module, "to_empty", _poisoned_to_empty):
                report = parity_check.run_parity(directory, dtype="float32", seq_len=12)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["parameters"]["different"], [])
        self.assertEqual(report["buffers"]["different"],
                         ["model.rotary_emb.inv_freq", "model.rotary_emb.original_inv_freq"])


class TestAdamwRecipe(unittest.TestCase):
    """The attribution recipe differs from the kit recipe in the optimizer only."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_only_the_optimizer_changes(self):
        """Feature: AdamW-only variant of the kit recipe.

        Description: Derive the variant and compare it with its source.
        Expectation: Same blocks except an AdamW optimizer with no Muon settings.
        """
        import yaml  # pylint: disable=import-outside-toplevel
        source = yaml.safe_load(_TEXT_RECIPE.read_text(encoding="utf-8"))
        variant = adamw_recipe.adamw_recipe()
        self.assertEqual(set(source), set(variant))
        for key in source:
            if key != "optimizer":
                self.assertEqual(source[key], variant[key], key)
        self.assertTrue(variant["optimizer"]["_target_"].endswith(".AdamW"))
        self.assertNotIn("muon_config", variant["optimizer"])


if __name__ == "__main__":
    unittest.main()
