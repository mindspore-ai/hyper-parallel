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

"""Paired full Qwen3-MoE training benchmark for internal collective EP.

Run with torchrun on one NPU node. Synthetic tokens, random initialization;
this is a small full-model benchmark, not the pretrained 30B/LIMO workload.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.distributed._builder.forward_rewriter import _install_bound_forward
from hyper_parallel.distributed.expert_parallel.recipes import build_ep_compute
from hyper_parallel.distributed.expert_parallel.routing import MOE_ROUTER_ADAPTERS


def build_model(backend, mesh, args):
    """Build identical global initialization, then retain owner expert shards."""
    torch.manual_seed(42)
    config = Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=16,
        num_experts_per_tok=2,
        max_position_embeddings=65536,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3MoeForCausalLM(config)
    rank, size = dist.get_rank(), dist.get_world_size()
    if config.num_experts % size:
        raise ValueError("EP size must divide 16 experts")
    count = config.num_experts // size
    for layer in model.model.layers:
        block = layer.mlp
        for name in ("gate_up_proj", "down_proj"):
            weight = getattr(block.experts, name)
            setattr(
                block.experts,
                name,
                torch.nn.Parameter(
                    weight.detach()[rank * count : (rank + 1) * count].clone()
                ),
            )
        compute = build_ep_compute(
            block,
            mesh,
            router_fn=MOE_ROUTER_ADAPTERS["qwen3moe"],
            archetype_key="benchmark_qwen3",
            expected_attrs=["gate", "experts"],
            combine=lambda _m, _x, y: y,
            token_dispatcher=backend,
            use_grouped_gemm=args.grouped,
        )
        _install_bound_forward(block, compute)
    return model.to(device="npu", dtype=torch.float32).train()


def run(backend, mesh, args, repeat):
    """Measure complete optimizer windows with global replicated-grad reduction."""
    torch.npu.empty_cache()
    model = build_model(backend, mesh, args)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    rank, size = dist.get_rank(), dist.get_world_size()
    generator = torch.Generator().manual_seed(123 + rank)
    # Fixed resident batch excludes data-loading cost equally for both backends.
    tokens = torch.randint(
        0, 256, (args.micro_batch, args.seq_length), generator=generator
    ).npu()
    dense = [p for name, p in model.named_parameters() if ".experts." not in name]
    expert = [p for name, p in model.named_parameters() if ".experts." in name]

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = None
        for _ in range(args.accumulation):
            loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss
            (loss / args.accumulation).backward()
        # Every EP rank has distinct examples. Dense replicas average gradients;
        # expert owners already receive summed gradients through EP backward.
        for parameter in dense:
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(size)
        for parameter in expert:
            if parameter.grad is not None:
                parameter.grad.div_(size)
        optimizer.step()
        return loss

    for _ in range(args.warmup):
        step()
    torch.npu.synchronize()
    dist.barrier()
    torch.npu.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(args.steps):
        loss = step()
    torch.npu.synchronize()
    elapsed = time.perf_counter() - started
    metrics = torch.tensor(
        [elapsed, torch.npu.max_memory_allocated(), torch.npu.max_memory_reserved()],
        device="npu",
        dtype=torch.float32,
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    duration, allocated, reserved = metrics.cpu().tolist()
    final_loss = float(loss.detach().cpu())
    if not torch.isfinite(loss).all():
        raise RuntimeError("nonfinite training loss")
    result = {
        "backend": backend,
        "repeat": repeat,
        "seq_length": args.seq_length,
        "micro_batch": args.micro_batch,
        "global_batch": size * args.micro_batch * args.accumulation,
        "accumulation": args.accumulation,
        "ep_size": size,
        "steps": args.steps,
        "warmup": args.warmup,
        "step_seconds": duration / args.steps,
        "tokens_per_second": size
        * tokens.numel()
        * args.accumulation
        * args.steps
        / duration,
        "allocated_gib": allocated / 2**30,
        "reserved_gib": reserved / 2**30,
        "final_loss_rank0": final_loss,
        "grouped": args.grouped,
    }
    if rank == 0:
        print(json.dumps(result), flush=True)
        with Path(args.output).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result) + "\n")
    dist.barrier()


def main():
    """Run alternating backend pairs with fixed initialization and data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (
        min(
            args.seq_length,
            args.micro_batch,
            args.accumulation,
            args.steps,
            args.repeats,
        )
        < 1
        or args.warmup < 0
    ):
        parser.error(
            "positive dimensions/steps/repeats and nonnegative warmup required"
        )
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    mesh = init_device_mesh("npu", (dist.get_world_size(),), mesh_dim_names=("ep",))
    for repeat in range(args.repeats):
        backends = ["all_to_all", "moonep_collective"]
        for backend in backends[:: (-1 if repeat % 2 else 1)]:
            run(backend, mesh, args, repeat)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
