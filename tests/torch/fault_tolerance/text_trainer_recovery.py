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
"""Observe the real TextTrainer entry point without replacing its training loop."""

import argparse
import faulthandler
import json
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, Qwen3MoeConfig

from examples.training_demo.qwen3_moe.prepare_parallel_data import prepare_parallel_data
from examples.training_demo import train_text
from hyper_parallel.trainer.text_trainer import TextTrainer


def prepare_assets(directory: Path) -> Path:
    """Derive a small offline MoE recipe from the full training demo.

    Args:
        directory: New directory for the local model, dataset and recipe.

    Returns:
        Path to the generated Trainer YAML configuration.
    """
    directory.mkdir(parents=True)
    model_path = directory / "model"
    torch.manual_seed(42)
    config = Qwen3MoeConfig(vocab_size=1024, hidden_size=64, intermediate_size=128,
                           moe_intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                           num_key_value_heads=2, head_dim=16, num_experts=4, num_experts_per_tok=2,
                           max_position_embeddings=256, tie_word_embeddings=False,
                           bos_token_id=1022, eos_token_id=1023, pad_token_id=0)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
    model.save_pretrained(model_path)
    data_path = directory / "data"
    prepare_parallel_data(data_path, num_samples=256, seq_length=32)
    recipe = Path(__file__).resolve().parents[3] / "examples/training_demo/qwen3_moe/train_parallel_full_offline.yaml"
    raw = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    raw["model"].update(pretrained_model_name_or_path=str(model_path), torch_dtype="float32")
    raw["training"]["train_iters"] = 12
    raw["fsdp_config"]["dp_shard_size"] = 4
    raw["activation_checkpoint"]["mode"] = "off"
    raw["checkpoint"].update(save_ckpt=False, is_async=False, restore_from=None)
    raw["stress_detect"]["op_stress_detect"]["detection_interval"] = 0
    raw["optimizer"]["muon_config"]["muon_hsdp_replica_count"] = 1
    raw["dataset"]["data_path"] = str(data_path / "parallel_offline_text_document")
    raw["dataset"]["data_config"]["seq_length"] = 32
    raw["dataset"]["model_assets"]["tokenizer"].update(
        pretrained_model_name_or_path=str(model_path), local_files_only=True, vocab_size=1024, eod_token_id=1023)
    raw["worker_recovery"] = {"enable_worker_reboot": True}
    path = directory / "trainer.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def worker(options: argparse.Namespace) -> None:
    """Run the demo entry point with fault gates and acceptance exports.

    Args:
        options: Rank, world size, run directory and rendezvous settings.
    """
    faulthandler.enable()
    faulthandler.dump_traceback_later(150, repeat=True)
    directory = Path(options.directory)
    rank = options.rank
    faults = tuple(int(value) for value in os.getenv("HP_RECOVERY_TEST_FAULT_RANKS", "1").split(","))
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(options.world),
                      WORLD_SIZE=str(options.world), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(options.store_port),
                      TORCHELASTIC_USE_AGENT_STORE="True")
    original_step = TextTrainer.train_step
    original_begin = TextTrainer.on_train_begin
    original_epoch_begin = TextTrainer.on_epoch_begin
    original_end = TextTrainer.on_train_end

    def _consume_rng(trainer):
        random.random()
        np.random.rand()
        torch.rand(1)
        torch.rand(1, device="npu")
        generator = getattr(trainer.base.train_dataloader, "generator", None)
        if generator is not None:
            torch.rand(1, generator=generator)

    def _begin(trainer):
        original_begin(trainer)
        _consume_rng(trainer)
        active = trainer.base.recovery
        record = {"pid": os.getpid(), "step": trainer.base.state.global_step,
                  "generation": active.groups.generation}
        with (directory / f"begin{rank}.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")

    def _epoch_begin(trainer):
        original_epoch_begin(trainer)
        _consume_rng(trainer)

    def _step(trainer, iterator):
        active = trainer.base.recovery
        step = trainer.base.state.global_step + 1
        if options.fault and step == 9 and active.groups.generation == 0:
            while rank in faults or not (directory / "forward_go").exists():
                active.pause()
                time.sleep(0.02)
            (directory / f"forward_started{rank}").touch()
        result = original_step(trainer, iterator)
        record = {"rank": rank, "pid": os.getpid(), "step": active.step,
                  "global_step": trainer.base.state.global_step, "epoch": trainer.base.state.epoch}
        temporary = directory / f"rank{rank}.json.tmp"
        temporary.write_text(json.dumps(record), encoding="utf-8")
        temporary.replace(directory / f"rank{rank}.json")
        if options.fault and active.step == 12:
            time.sleep(5)
        return result

    def _end(trainer):
        original_end(trainer)
        active = trainer.base.recovery
        torch.save({"step": active.step,
                    "model": {name: tensor.detach().cpu().clone()
                              for name, tensor in active.model_state.tensor_items()},
                    "optimizer_metadata": active.model_state.metadata(), "training": active.train_state.capture()},
                   directory / f"final{rank}.pt")

    arguments = ["examples.training_demo.train_text", os.environ["HP_RECOVERY_TEST_TEXT_TRAINER"],
                 f"--checkpoint.checkpoint_dir={directory / 'checkpoints'}"]
    with patch.object(TextTrainer, "on_train_begin", _begin), \
            patch.object(TextTrainer, "on_epoch_begin", _epoch_begin), \
            patch.object(TextTrainer, "train_step", _step), patch.object(TextTrainer, "on_train_end", _end), \
            patch.object(sys, "argv", arguments):
        train_text.main()
