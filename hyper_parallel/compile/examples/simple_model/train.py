#!/usr/bin/env python3
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
"""
Simple Model Training Script - HyperParallel Graph Mode Example

Demonstrates how to use HyperParallel Graph Mode for training a simple model.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hyper_parallel.compile import (  # pylint: disable=C0413
    GraphTrainer,
    PassConfig,
    PassPlan,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple Model Training with HyperParallel Graph Mode"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training"""
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(dist.get_rank() % torch.npu.device_count())


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_pass_config(config: dict) -> PassConfig:
    pass_config = config["parallel"]
    return PassConfig(
        enable_overlap=pass_config.get("enable_overlap", True),
    )


def build_pass_plan(config: dict) -> PassPlan:
    """Build sharding plan from YAML config"""
    if "sharding" in config:
        # Use YAML configuration
        from hyper_parallel.compile import create_sharding_plan_from_yaml  # pylint: disable=C0415
        import tempfile  # pylint: disable=C0415

        # Write sharding config to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config["sharding"], f)
            temp_path = f.name

        plan = create_sharding_plan_from_yaml(config_path=temp_path)

        # Clean up temp file
        import os  # pylint: disable=C0415

        os.unlink(temp_path)

        return plan

    # Default: FSDP all modules
    plan = PassPlan()
    plan.fsdp_wrap_pattern("*")
    return plan


def train_fn(model, input_ids, labels):
    """Training function"""
    logits = model(input_ids)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


def main():
    args = parse_args()
    config = load_config(args.config)

    # Setup distributed training
    setup_distributed()

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    print("=" * 80)
    print("Simple Model Training with HyperParallel Graph Mode")
    print("=" * 80)
    print(f"Rank: {rank}/{world_size}")
    print(f"FSDP: {world_size}")
    print("=" * 80)

    # Create dummy model (example)
    class DummyModel(torch.nn.Module):
        def __init__(self, vocab_size, dim):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, dim)
            self.linear = torch.nn.Linear(dim, vocab_size)

        def forward(self, x):
            return self.linear(self.embed(x))

    # Materialize the model on a real device. GraphTrainer executes the
    # compiled graph and steps the optimizer with real parameters, so the model
    # must not stay on the ``meta`` device.
    device = "npu" if (hasattr(torch, "npu") and torch.npu.is_available()) else "cpu"
    model = DummyModel(config["model"]["vocab_size"], config["model"]["dim"]).to(device)

    pass_config = build_pass_config(config)
    pass_plan = build_pass_plan(config)

    trainer = GraphTrainer(
        model=model,
        train_fn=train_fn,
        pass_config=pass_config,
        pass_plan=pass_plan,
        optimizer_config={
            "lr": config["train"]["optimizer"]["lr"],
            "grad_clip": config["train"]["grad_clip"],
        },
    )

    vocab_size = config["model"]["vocab_size"]
    max_seq_len = config["model"]["max_seq_len"]
    max_steps = config["train"]["max_steps"]
    log_interval = config["logging"]["log_interval"]

    # The data iterator yields ``(input, label)`` batches. train drives the whole
    # loop: it compiles on the first batch, moves each batch onto the trainer's
    # device, runs a step + optimizer update, and logs on ``log_interval``.
    # Batches are produced on CPU; ``train`` moves them onto ``trainer.device``.
    g_input_ids = torch.randint(0, vocab_size, (1, max_seq_len))
    g_labels = torch.randint(0, vocab_size, (1, max_seq_len))

    def data_iter():
        for _ in range(max_steps):
            input_ids = g_input_ids
            labels = g_labels
            yield input_ids, labels

    print("\nStarting training...")
    trainer.train(data_iter(), max_steps=max_steps, log_interval=log_interval)
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)

    # Cleanup distributed training
    cleanup_distributed()


if __name__ == "__main__":
    main()
