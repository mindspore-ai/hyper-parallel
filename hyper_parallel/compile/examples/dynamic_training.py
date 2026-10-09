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
"""Train a tiny causal LM with variable shapes and verify each step against eager.

Run with ``python -m hyper_parallel.compile.examples.dynamic_training --device npu``.
Launching the same module with torchrun enables FSDP across the participating devices.
Synthetic token sequences require no model weights, tokenizer or dataset downloads.
"""

import argparse
import copy
import importlib
import json
import os
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional

from hyper_parallel.compile import GraphTrainer, PassConfig


class TinyLanguageModel(nn.Module):
    """Learn next-token prediction through embeddings and a small feed-forward network."""

    def __init__(self) -> None:
        """Initialize the small embedding and prediction layers."""
        super().__init__()
        self.embedding = nn.Embedding(32, 16)
        self.layers = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 32))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return token logits for variable input dimensions.

        Args:
            input_ids: Token ids with shape (batch, sequence).

        Returns:
            Per-token vocabulary logits.
        """
        return self.layers(self.embedding(input_ids))


def causal_loss(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute shifted-label cross entropy, ignoring the final token.

    Args:
        model: Language model producing vocabulary logits.
        input_ids: Input token ids.
        labels: Labels shifted by one token for next-token prediction.

    Returns:
        Scalar mean cross entropy over valid labels.
    """
    shifted = functional.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    return functional.cross_entropy(model(input_ids).float().flatten(0, 1), shifted.flatten())


def _check_gradients(model: nn.Module, reference: nn.Module, rank: int, world_size: int) -> None:
    """Compare graph gradients with eager gradients summed across the FSDP group."""
    for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
        if world_size > 1:
            dist.all_reduce(ref_parameter.grad)
        expected_grad = ref_parameter.grad.chunk(world_size, dim=0)[rank]
        torch.testing.assert_close(parameter.grad, expected_grad)


def run_training(device: torch.device, steps: int = 12, dtype: torch.dtype = torch.float32) -> dict:
    """Check graph loss, reduced gradient shards and optimizer updates against eager.

    A process group is optional. FSDP uses SUM reduce-scatter, so the eager
    reference sums gradients across ranks before its optimizer update as well.
    Both paths consume the current weights and current batch on every step.

    Args:
        device: Device for model state and generated token batches.
        steps: Number of optimizer steps, at least four.
        dtype: Floating-point parameter dtype.

    Returns:
        Training audit containing shapes, losses and capture count.
    """
    if steps < 4:
        raise ValueError("steps must be at least 4 to exercise changing and repeated shapes")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    torch.manual_seed(42)
    model = TinyLanguageModel().to(device=device, dtype=dtype)
    reference = copy.deepcopy(model)
    initial = [parameter.detach().clone() for parameter in reference.parameters()]
    trainer = GraphTrainer(
        model, causal_loss, pass_config=PassConfig(fsdp_enabled=world_size > 1), device=device,
        dynamic_arg_dims={"input_ids": [0, 1], "labels": [0, 1]},
        optimizer_config={"lr": 1e-2},
    )
    optimizer = torch.optim.Adam(reference.parameters(), lr=1e-2, foreach=False)
    shapes, losses = [], []
    with patch.object(trainer, "compile", wraps=trainer.compile) as capture:
        for step in range(steps):
            batch = 2 if step % 6 < 3 else 3
            length = (5, 7, 7, 11, 9, 11)[step % 6] + rank
            input_ids = (torch.arange(batch * length, device=device).reshape(batch, length) + rank) % 32
            labels = input_ids.clone()
            actual = trainer.train_step(input_ids=input_ids, labels=labels)
            expected = causal_loss(reference, input_ids, labels)
            expected.backward()
            torch.testing.assert_close(actual, expected)
            _check_gradients(model, reference, rank, world_size)
            trainer.optimizer_step()
            optimizer.step()
            optimizer.zero_grad()
            for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter, ref_parameter.chunk(world_size, dim=0)[rank])
            shapes.append([batch, length])
            losses.append(float(actual.detach().cpu()))
            if rank == 0:
                print(f"step={step + 1}, shape=({batch}, {length}), loss={losses[-1]:.6f}")
        captures = capture.call_count
    if captures != 1:
        raise RuntimeError(f"Expected one graph capture, got {captures}")
    if not any(not torch.equal(parameter, old) for parameter, old in zip(reference.parameters(), initial)):
        raise RuntimeError("Training did not update any parameters")
    if not all(torch.isfinite(torch.tensor(losses))):
        raise RuntimeError(f"Training produced non-finite losses: {losses}")
    audit = {
        "rank": rank, "world_size": world_size, "device": str(device), "dtype": str(dtype),
        "optimizer_steps": steps, "captures": captures, "unique_shapes": sorted({tuple(shape) for shape in shapes}),
        "losses": losses, "eager_loss_gradients_and_weights_match": True,
    }
    print(json.dumps(audit))
    return audit


def main() -> None:
    """Select a device, optionally initialize FSDP, and run the training check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "npu"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    if args.device == "npu":
        # torch_npu is optional for CPU/CUDA users and registers the NPU backend.
        importlib.import_module("torch_npu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cpu" if args.device == "cpu" else f"{args.device}:{local_rank}")
    if args.device != "cpu":
        getattr(torch, args.device).set_device(device)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        backend = {"cpu": "gloo", "cuda": "nccl", "npu": "hccl"}[args.device]
        dist.init_process_group(backend, timeout=timedelta(seconds=120))
    try:
        run_training(device, args.steps, getattr(torch, args.dtype))
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
