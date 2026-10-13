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
"""Compare eager and graph recompute on a small residual MLP.

Run from the repository root:
    python -m hyper_parallel.compile.examples.recompute.train --device cpu
    python -m hyper_parallel.compile.examples.recompute.train --device npu --dropout 0.2
"""

# Torch-only graph examples follow the repository backend policy.
# pylint: disable=forbidden-backend-import

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn

from hyper_parallel.compile import GraphCompiler, PassConfig, RecomputeConfig


class ResidualBlock(nn.Module):
    """Two bias-free projections make SAC's mm alternation visible."""

    def __init__(self, width: int, dropout: float) -> None:
        """Build the example model components."""
        super().__init__()
        self.up = nn.Linear(width, width * 4, bias=False)
        self.down = nn.Linear(width * 4, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return a residual update."""
        return value + self.down(self.dropout(self.up(value).sigmoid())).tanh()


class ResidualMLP(nn.Module):
    """A small model with explicit checkpoint region boundaries."""

    def __init__(self, width: int, dropout: float) -> None:
        """Build the example model components."""
        super().__init__()
        self.layers = nn.ModuleList([ResidualBlock(width, dropout) for _ in range(3)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the residual blocks."""
        for layer in self.layers:
            value = layer(value)
        return value


def train_loss(model: nn.Module, *, value: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Return a scalar regression loss."""
    return (model(value) - label).square().mean()


def seed_step(seed: int, device: torch.device) -> None:
    """Reset both CPU and accelerator generators before each comparison."""
    torch.manual_seed(seed)
    if device.type != "cpu":
        getattr(torch, device.type).manual_seed_all(seed)


def rng_state(device: torch.device) -> list[torch.Tensor]:
    """Read generator states for parity checking."""
    states = [torch.get_rng_state()]
    if device.type != "cpu":
        states.append(getattr(torch, device.type).get_rng_state())
    return states


def compare_mode(initial: nn.Module, batch: dict, args: argparse.Namespace,
                 mode: str, device: torch.device) -> dict:
    """Check every gradient and parameter update against eager execution."""
    reference, model = copy.deepcopy(initial), copy.deepcopy(initial)
    config = RecomputeConfig(mode=mode, selective_policy=args.selective_policy,
                             exclude_module_patterns=tuple(args.exclude))
    compiler = GraphCompiler(model, train_loss, PassConfig(
        fsdp_enabled=False, enable_overlap=False, recompute=config), device=device)
    compiler.compile(**batch)
    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        graph = compiler._joint_graph.graph_module  # pylint: disable=protected-access
        (args.dump_dir / f"{mode}.txt").write_text(graph.print_readable(print_output=False))
    eager_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    graph_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    max_grad_error = 0.0
    losses = []
    for step in range(args.steps):
        eager_optimizer.zero_grad(set_to_none=True)
        graph_optimizer.zero_grad(set_to_none=True)
        seed_step(1234 + step, device)
        expected = train_loss(reference, **batch)
        expected.backward()
        expected_rng = rng_state(device)
        seed_step(1234 + step, device)
        actual, _ = compiler.forward_backward(**batch)
        torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)
        for actual_state, expected_state in zip(rng_state(device), expected_rng):
            torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
        for parameter, original in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, original.grad, rtol=args.rtol, atol=args.atol)
            max_grad_error = max(max_grad_error, (parameter.grad - original.grad).abs().max().item())
        graph_optimizer.step()
        eager_optimizer.step()
        for parameter, original in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter, original, rtol=args.rtol, atol=args.atol)
        losses.append(actual.item())
    return {"mode": mode, "steps": args.steps, "losses": losses,
            "max_grad_error": max_grad_error, "rng_equal": True,
            "recompute": compiler.recompute_report}


def main() -> None:
    """Run all graph modes and print machine-readable validation results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--selective-policy", choices=("eager", "save_compute"), default="eager")
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()
    if args.steps < 1 or args.width < 1:
        parser.error("steps and width must be positive")
    device = torch.device(args.device)
    seed_step(42, device)
    model = ResidualMLP(args.width, args.dropout).to(device)
    batch = {"value": torch.randn(8, args.width, device=device),
             "label": torch.randn(8, args.width, device=device)}
    for mode in ("none", "full", "selective"):
        print(json.dumps(compare_mode(model, batch, args, mode, device)))


if __name__ == "__main__":
    main()
