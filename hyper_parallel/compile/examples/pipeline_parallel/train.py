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
Pipeline Parallel Training - HyperParallel Graph Mode Example

Demonstrates graph-pass pipeline parallelism: the joint fwd+bwd graph is
sliced into per-rank stages by ``PpPass`` (no model deepcopy), boundary
activations cross stages via P2P, and a self-contained GPipe schedule drives
microbatched training. ``GraphTrainer`` itself stays PP-unaware.

Launch with world_size == pp_degree (v1 pure-PP contract), e.g. 2 stages:

    torchrun --nproc_per_node=2 train.py --config config.yaml

Stage split comes from ``sharding.pp.stages`` in the YAML when declared, or
falls back to the automatic even-by-layers split (the ``layers`` ModuleList
is distributed evenly; modules before it go to stage 0, modules after it to
the last stage).
"""

import argparse
import sys
from pathlib import Path
from typing import Iterator, Tuple

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from hyper_parallel.compile import (  # pylint: disable=C0413,C0415,E0611
    GraphTrainer,
    PassConfig,
    PassPlan,
    create_sharding_plan_from_yaml,
)


def parse_args() -> argparse.Namespace:
    """Parse example CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Pipeline-Parallel Training with HyperParallel Graph Mode"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def setup_distributed() -> None:
    """Initialize the process group: hccl on NPU, gloo elsewhere."""
    if dist.is_initialized():
        return
    backend = "gloo"
    if hasattr(torch, "npu") and torch.npu.is_available():
        backend = "hccl"
    dist.init_process_group(backend=backend)
    # ``set_device`` needs the rank, which only exists after init; guard it so
    # gloo (CPU) runs don't touch the NPU stack.
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(dist.get_rank() % torch.npu.device_count())


def cleanup_distributed() -> None:
    """Destroy the process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path: str) -> dict:
    """Load the example YAML config."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class PPDemoModel(nn.Module):
    """Tiny LLM-shaped model: embedding -> layer container -> norm -> head.

    The ``layers`` ModuleList is what the automatic PP split distributes:
    modules registered before it go to stage 0 and modules after it to the
    last stage. Each block keeps a residual stream, so exactly ONE tensor
    (the hidden state) crosses every stage boundary — the single-tensor cut
    contract of the graph-pass PP split.
    """

    def __init__(self, vocab_size: int, dim: int, num_layers: int) -> None:
        """Initialize embedding, ``num_layers`` residual blocks, head."""
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim), nn.GELU()) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the LM trunk: embed -> residual blocks -> norm -> logits."""
        hidden = self.tok_embeddings(input_ids)
        for layer in self.layers:
            hidden = hidden + layer(hidden)
        return self.lm_head(self.norm(hidden))


def train_fn(
    model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Next-token loss; computed at the root, so it lands on the LAST stage."""
    logits = model(input_ids)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def build_pass_config(config: dict) -> PassConfig:
    """Build the pass config; PP is opt-in and pure (FSDP off) in v1."""
    parallel = config.get("parallel", {})
    return PassConfig(
        enable_overlap=parallel.get("enable_overlap", True),
        fsdp_enabled=False,
        pp_enabled=parallel.get("pp_enabled", True),
        pp_degree=parallel.get("pp_degree"),
        pp_microbatch_size=parallel.get("pp_microbatch_size", 1),
    )


def build_pass_plan(config_path: str) -> PassPlan:
    """Parse the YAML plan (fsdp + pp sections; extra keys are ignored).

    With ``pp.stages`` declared the plan is manual; without it ``PpPass``
    falls back to the automatic even-by-layers split.
    """
    return create_sharding_plan_from_yaml(config_path=config_path)


def main() -> None:
    """Set up distributed PP training and run the trainer loop."""
    args = parse_args()
    config = load_config(args.config)

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = "npu" if (hasattr(torch, "npu") and torch.npu.is_available()) else "cpu"
    model_cfg = config["model"]
    model = PPDemoModel(
        vocab_size=model_cfg["vocab_size"],
        dim=model_cfg["dim"],
        num_layers=model_cfg["num_layers"],
    ).to(device)

    pass_config = build_pass_config(config)
    pass_plan = build_pass_plan(args.config)

    if rank == 0:
        stages = pass_plan.pp_module_fqns_per_stage or "auto (even by layers)"
        print("=" * 72)
        print(
            f"PP demo: world_size={world_size} (stages), "
            f"microbatch={pass_config.pp_microbatch_size}"
        )
        print(f"Stage plan: {stages}")
        print("=" * 72)

    trainer = GraphTrainer(
        model=model,
        train_fn=train_fn,
        pass_config=pass_config,
        pass_plan=pass_plan,
        optimizer_config={
            "lr": config["train"]["optimizer"]["lr"],
            "grad_clip": config["train"].get("grad_clip"),
        },
        device=torch.device(device),
    )

    vocab = model_cfg["vocab_size"]
    seq_len = model_cfg["seq_len"]
    steps = config["train"]["max_steps"]
    microbatch = pass_config.pp_microbatch_size
    gen = torch.Generator().manual_seed(42)
    batch = torch.randint(
        0, vocab, (config["train"]["batch_size"], seq_len), generator=gen
    )
    labels = torch.roll(batch, shifts=-1, dims=1)

    # Compile ONCE with a MICROBATCH-shaped sample: the traced graph bakes
    # view sizes for one microbatch, and the GPipe schedule then slices
    # every full batch it receives into matching microbatches at runtime.
    # ``compile`` does NOT move its sample tensors, so place them on the
    # trainer's device here (``train_step`` does the same per batch).
    sample_input, sample_label = trainer._place_on_device(  # pylint: disable=protected-access
        (batch[:microbatch], labels[:microbatch])
    )
    trainer.compile(sample_input, sample_label)

    def data_iter() -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield the same synthetic batch each step (demo workload)."""
        for _ in range(steps):
            yield batch, labels

    losses = trainer.train(
        data_iter(), max_steps=steps, log_interval=config["logging"]["log_interval"]
    )
    # The real loss lives on the LAST stage (other stages return a zero
    # placeholder); report convergence from there.
    if rank == world_size - 1 and losses:
        print(f"Last-stage loss: {losses[0].item():.4f} -> {losses[-1].item():.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
