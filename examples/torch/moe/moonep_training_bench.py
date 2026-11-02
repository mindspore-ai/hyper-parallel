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
"""Measure complete Qwen3-MoE training windows with reproducible routing."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.utils.clip_grad import _get_local_grad
from hyper_parallel.distributed.expert_parallel.routing import MOE_ROUTER_ADAPTERS
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.text_trainer import TextTrainer


class RoutingExperiment:
    """Own replay state and device-side counts without changing production APIs."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.step = 0
        self.micro = 0
        self.layers: dict[int, int] = {}
        self.counts: dict[tuple[int, int], torch.Tensor] = {}
        self.seen: set[tuple[int, int, int]] = set()
        self.original = MOE_ROUTER_ADAPTERS["qwen3moe"]

    def route(self, module: Any, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay IDs across recomputation while preserving router gradients."""
        layer = self.layers.setdefault(id(module), len(self.layers))
        if self.args.routing == "natural":
            indices, scores = self.original(module, hidden)
            experts = int(module.experts.num_experts)
        else:
            logits, original_scores, _ = module.gate(hidden)
            logits = logits.reshape(-1, logits.shape[-1])
            tokens, experts = logits.shape
            top_k = original_scores.shape[-1]
            rank = dist.get_rank()
            seed = self.args.route_seed + self.step * 100003 + self.micro * 997 + layer * 31 + rank
            generator = torch.Generator(device=logits.device).manual_seed(seed)
            if self.args.routing == "uniform":
                positions = torch.arange(tokens * top_k, device=logits.device).reshape(tokens, top_k)
                indices = (positions + self.step + rank * top_k + layer) % experts
            else:
                local_experts = experts // dist.get_world_size()
                hot_rank = 0
                if self.args.routing == "dynamic":
                    hot_rank = max(0, self.step - self.args.warmup - 1) // 10 % dist.get_world_size()
                log_popularity = torch.zeros(experts, device=logits.device)
                if self.args.routing == "expert":
                    log_popularity[0] = self.args.hot_log_weight
                elif self.args.routing == "dispersed":
                    hot = torch.arange(local_experts, device=logits.device)
                    hot = (hot % dist.get_world_size()) * local_experts + hot // dist.get_world_size()
                    log_popularity[hot] = self.args.hot_log_weight
                else:
                    start = hot_rank * local_experts
                    log_popularity[start:start + local_experts] = self.args.hot_log_weight
                uniform = torch.rand(tokens, experts, device=logits.device, generator=generator)
                gumbel = -torch.log(-torch.log(uniform.clamp(1e-6, 1 - 1e-6)))
                indices = (gumbel + log_popularity).topk(top_k, dim=-1).indices
            scores = logits.float().softmax(-1).gather(-1, indices)
            scores = scores / scores.sum(-1, keepdim=True).clamp_min(1e-20)
        indices = indices.reshape(-1, indices.shape[-1])
        key = (self.step, self.micro, layer)
        if self.step > self.args.warmup and key not in self.seen:
            self.seen.add(key)
            counts = torch.bincount(indices.reshape(-1).long(), minlength=experts).detach()
            count_key = (self.step, layer)
            if count_key in self.counts:
                self.counts[count_key].add_(counts)
            else:
                self.counts[count_key] = counts
        return indices, scores


class MeasuredTextTrainer(TextTrainer):
    """Time full optimizer steps including fetch and ordinary callbacks."""

    def __init__(self, config: Any, experiment: RoutingExperiment) -> None:
        self.experiment = experiment
        self.window_start = 0.0
        self.host_steps: list[dict[str, Any]] = []
        self.measured_tokens = 0.0
        self.full_peak: dict[str, int] = {}
        super().__init__(config)
        if experiment.args.gradient_diagnostic:
            self.original_prepare = self.base.prepare_optimizer_step
            self.base.prepare_optimizer_step = self.prepare_with_snapshot

    def prepare_with_snapshot(self) -> Any:
        """Use the norm helper's gradient convention for diagnostic sampling."""
        norm = self.original_prepare()
        torch.npu.synchronize()
        snapshots = {}
        for name, param in self.base.model.named_parameters():
            grad = _get_local_grad(param)
            if grad is None:
                snapshots[name] = None
                continue
            flat = grad.detach().reshape(-1).float()
            stride = max(1, flat.numel() // 4096)
            snapshots[name] = {
                "shape": list(grad.shape), "dtype": str(grad.dtype),
                "max_abs": flat.abs().max().item(),
                "l2": flat.norm().item(),
                "sample": flat[::stride][:4096].cpu(),
            }
        folder = Path(self.experiment.args.output)
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({"norm": float(norm), "gradients": snapshots},
                   folder / f"grad_step{self.experiment.step}_rank{dist.get_rank()}.pt")
        return norm

    def forward_backward_step(self, data_iterator: Any, num_micro_steps: int) -> tuple:
        """Keep the microbatch index fixed throughout its backward replay."""
        result = super().forward_backward_step(data_iterator, num_micro_steps)
        self.experiment.micro += 1
        return result

    def train_step(self, data_iterator: Any) -> dict[str, float]:
        """Synchronize only at window boundaries, never at each measured step."""
        step = self.base.state.global_step + 1
        self.experiment.step = step
        self.experiment.micro = 0
        if step == self.experiment.args.warmup + 1:
            dist.barrier()
            torch.npu.synchronize()
            self.full_peak = {
                "allocated": torch.npu.max_memory_allocated(),
                "reserved": torch.npu.max_memory_reserved(),
            }
            torch.npu.reset_peak_memory_stats()
            self.window_start = time.perf_counter()
        started = time.perf_counter()
        result = super().train_step(data_iterator)
        host_seconds = time.perf_counter() - started
        if step > self.experiment.args.warmup:
            metrics = dict(self.base.step_env_metrics)
            self.measured_tokens += metrics["data/step_tokens"]
            self.host_steps.append({"step": step, "host_seconds": host_seconds, **metrics})
        if step == self.base.train_iters:
            torch.npu.synchronize()
            seconds = time.perf_counter() - self.window_start
            self.write_results(seconds)
        return result

    def write_results(self, seconds: float) -> None:
        """Reduce outside timing and save every rank's metrics and routing counts."""
        args = self.experiment.args
        device = self.base.device
        window_peak = {
            "allocated": torch.npu.max_memory_allocated(),
            "reserved": torch.npu.max_memory_reserved(),
        }
        elapsed = torch.tensor(seconds, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        window_seconds = elapsed.item()
        rank = dist.get_rank()
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        layers = []
        for (step, layer), counts in sorted(self.experiment.counts.items()):
            source_counts = counts.cpu().tolist()
            dist.all_reduce(counts)
            global_counts = counts.cpu()
            owners = global_counts.reshape(dist.get_world_size(), -1).sum(-1)
            mean = global_counts.float().mean().item()
            layers.append({
                "step": step, "layer": layer, "source_counts": source_counts,
                "global_counts": global_counts.tolist(), "owner_counts": owners.tolist(),
                "expert_ratio": global_counts.max().item() / mean,
                "owner_ratio": owners.max().item() / owners.float().mean().item(),
            })
        result = {
            "rank": rank, "backend": os.environ.get("HP_MOONEP_EP"),
            "routing": args.routing, "hot_log_weight": args.hot_log_weight,
            "warmup": args.warmup, "steps": len(self.host_steps),
            "window_seconds": window_seconds, "effective_tokens": self.measured_tokens,
            "tokens_per_second": self.measured_tokens / window_seconds,
            "window_peak_bytes": window_peak,
            "full_peak_bytes": {k: max(v, window_peak[k]) for k, v in self.full_peak.items()},
            "layers": layers,
        }
        (output / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
        with (output / f"rank{rank}_steps.jsonl").open("w") as stream:
            for row in self.host_steps:
                stream.write(json.dumps(row) + "\n")
        if rank == 0:
            print("BENCH_WINDOW " + json.dumps({k: v for k, v in result.items() if k != "layers"}), flush=True)


def main() -> None:
    """Install a process-local router adapter and run the normal text trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing", choices=["natural", "uniform", "expert", "rank",
                                              "dispersed", "dynamic"], default="natural")
    parser.add_argument("--hot-log-weight", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--route-seed", type=int, default=20261026)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gradient-diagnostic", action="store_true")
    args, config_args = parser.parse_known_args()
    config = parse_training_args(config_args)
    if args.warmup < 0 or config.training.train_iters <= args.warmup:
        raise ValueError("train_iters must exceed non-negative warmup")
    if config.accelerator.tp_size != 1 or config.accelerator.cp_size != 1 or config.accelerator.pp_size != 1:
        raise ValueError("this benchmark currently supports TP=CP=PP=1 only")
    experiment = RoutingExperiment(args)
    MOE_ROUTER_ADAPTERS["qwen3moe"] = experiment.route
    trainer = MeasuredTextTrainer(config, experiment)
    trainer.train()


if __name__ == "__main__":
    main()
