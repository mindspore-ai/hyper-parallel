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
"""Fresh-process, full-width DSV4.1 crop benchmark; launch with torchrun."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
from torch.utils.data import DataLoader, Dataset  # pylint: disable=forbidden-backend-import
import yaml

from examples.training_demo.train_deepseek_v41_megamoe import MegaMoeTextTrainer
from examples.training_demo.deepseek_v41_diagnostics import MoeTrainingDiagnostics
from hyper_parallel.trainer.config.manager import parse_training_args
from hyper_parallel.trainer.text_trainer import TextTrainer

_TARGET = "examples.training_demo.benchmark_deepseek_v41_megamoe."


class FixedTokens(Dataset):
    """Deterministic text IDs with identical samples in every backend run."""

    def __init__(self, tokens: int, data_config: dict | None = None) -> None:
        """Set sequence length; batching consumes data_config separately."""
        del data_config
        self.tokens = tokens

    def __len__(self) -> int:
        """Provide enough samples for warmup and measurement."""
        return 65536

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return valid vocabulary IDs and shifted next-token labels."""
        tokens = (torch.arange(self.tokens + 1) * 31 + index * 131) % 120000 + 3
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}


def build_loader(dataset: Dataset, collate_fn: Any, batch_sampler: Any) -> DataLoader:
    """Use the Trainer sampler and production packed-text collator.

    Args:
        dataset: Deterministic token samples.
        collate_fn: Production packed-text collator.
        batch_sampler: Trainer-owned distributed sampler.
    """
    return DataLoader(dataset, collate_fn=collate_fn, batch_sampler=batch_sampler)


def _recipe(args, work, world):
    recipe = yaml.safe_load(Path(__file__).with_name("train_deepseek_v41_megamoe.yaml").read_text(encoding="utf-8"))
    recipe["model"].update(config_path=args.model_dir, engram_assets_path=args.engram_assets,
                           num_routed_experts=args.experts)
    recipe["model_init_dtype"] = "bfloat16"
    recipe["training"].update(train_iters=args.schedule_steps or args.warmup + args.steps, global_batch_size=world)
    recipe["accelerator"]["ep_size"] = world
    recipe["fsdp_config"].update(dp_shard_size=world, edp_shard_size=1)
    if args.backend == "owner_ep":
        recipe["plan_overrides"].pop(0)
        recipe["plan_overrides"][-1]["local_compute_fn"] = {
            "_target_": "hyper_parallel.models.deepseek_v41.adapter.expert_parallel.deepseek_v41_ep_compute_fn",
            "use_grouped_gemm": True,
        }
    else:
        recipe["plan_overrides"][-1]["local_compute_fn"].update(
            local_num_tokens=args.tokens, dispatch_mode=args.dispatch_mode)
    recipe["dataset"] = {"_target_": _TARGET + "FixedTokens", "tokens": args.tokens, "data_config": {
        "create_attention_mask_in_dataloader": True, "labels_are_shifted": True,
    }}
    recipe["dataloader"] = {"_target_": _TARGET + "build_loader",
                            "batch_adapter": recipe["dataloader"]["batch_adapter"],
                            "collate_fn": recipe["dataloader"]["collate_fn"],
                            "get_batch": recipe["dataloader"]["get_batch"], "dataloader_type": "single"}
    recipe["checkpoint"]["checkpoint_dir"] = str(work / "unused-checkpoint")
    path = work / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    return parse_training_args([str(path)])


def _local(value):
    return value.to_local() if hasattr(value, "to_local") else value


def _initial_weights(trainer, args, rank):
    """Read identical rank-local shards, transposing only MegaMoe expert weights."""
    if not args.weights:
        return
    directory = Path(args.weights)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank{rank}.pt"
    state = trainer.base.model.state_dict()
    if args.write_weights:
        if args.backend != "owner_ep" or path.exists():
            raise ValueError("Write canonical weights once, from owner_ep, into an empty directory")
        torch.save({name: _local(value).detach().cpu() for name, value in state.items()}, path)
    else:
        source = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if source.keys() != state.keys():
            raise ValueError("Backend state names differ from canonical weights")
        with torch.no_grad():
            for name, parameter in state.items():
                value = source[name]
                if args.backend == "megamoe" and ".mlp.experts." in name:
                    value = value.transpose(-1, -2).contiguous()
                target = _local(parameter)
                if target.shape != value.shape:
                    raise ValueError(f"Canonical weight shape mismatch for {name}: {value.shape} vs {target.shape}")
                target.copy_(value)
        optimizers = trainer.base.optimizer
        for optimizer in optimizers if isinstance(optimizers, list) else [optimizers]:
            optimizer.reload_model_params()
    dist.barrier()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--engram-assets", required=True)
    parser.add_argument("--backend", choices=("owner_ep", "megamoe"), required=True)
    parser.add_argument("--experts", type=int, default=192)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--dispatch-mode", choices=("push", "pull"), default="push")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--schedule-steps", type=int, help="Keep the LR schedule horizon fixed in shorter diagnostics")
    parser.add_argument("--weights", help="Shared canonical rank-local BF16 weights directory")
    parser.add_argument("--write-weights", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostics", choices=("none", "moe"), default="none")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0 or args.tokens < 128 or args.tokens % 128:
        parser.error("steps must be positive, warmup nonnegative, and tokens a positive multiple of 128")
    world = int(os.environ["WORLD_SIZE"])
    if world < 2 or args.experts < 6 or args.experts % world:
        parser.error("requires EP >= 2 and experts >= TopK=6 divisible by EP")
    if args.write_weights and not args.weights:
        parser.error("--write-weights requires --weights")
    if args.schedule_steps is not None and args.schedule_steps < args.warmup + args.steps:
        parser.error("schedule-steps must cover all warmup and measured steps")
    return args


def main() -> None:
    """Time synchronized full optimizer steps; exclude model/weight initialization."""
    args = _arguments()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    work = Path(args.output) / f"rank{rank}"
    work.mkdir(parents=True, exist_ok=True)
    config = _recipe(args, work, world)
    trainer_type = MegaMoeTextTrainer if args.backend == "megamoe" else TextTrainer
    trainer = trainer_type(config)
    _initial_weights(trainer, args, rank)
    trainer.on_train_begin()
    iterator = iter(trainer.base.train_dataloader)
    rows = []
    diagnostics = None
    torch.npu.reset_peak_memory_stats()
    for step in range(args.warmup + args.steps):
        if step == args.warmup and args.diagnostics != "none":
            diagnostics = MoeTrainingDiagnostics(trainer)
        if diagnostics is not None:
            diagnostics.step = step
        dist.barrier()
        torch.npu.synchronize()
        start = time.perf_counter()
        metrics = trainer.train_step(iterator)
        torch.npu.synchronize()
        elapsed = time.perf_counter() - start
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError(f"Nonfinite step metrics: {metrics}")
        times = torch.tensor([elapsed], device="npu", dtype=torch.float32)
        dist.all_reduce(times, op=dist.ReduceOp.MAX)
        rows.append({"step": step, "warmup": step < args.warmup, "local_seconds": elapsed,
                     "max_rank_seconds": times.item(), "allocated_bytes": torch.npu.memory_allocated(),
                     "reserved_bytes": torch.npu.memory_reserved(), **metrics})
        (work / "progress.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    measured = [row["max_rank_seconds"] for row in rows if not row["warmup"]]
    result = {**vars(args), "rank": rank, "world_size": world, "steps_detail": rows,
              "median_seconds": statistics.median(measured), "mean_seconds": statistics.mean(measured),
              "tokens_per_second": world * args.tokens / statistics.median(measured),
              "peak_allocated_bytes": torch.npu.max_memory_allocated(),
              "peak_reserved_bytes": torch.npu.max_memory_reserved(),
              "attention_mhc": "production_fused", "optimizer": "Muon_fp32_main_params",
              "canonical_weights": bool(args.weights), "workspace_shared": args.backend == "megamoe",
              "owner_use_grouped_gemm": args.backend == "owner_ep"}
    if diagnostics is not None:
        result["phase_timings"] = diagnostics.rows
        result["throughput_valid"] = False
        diagnostics.close()
    trainer.on_train_end()
    (work / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
