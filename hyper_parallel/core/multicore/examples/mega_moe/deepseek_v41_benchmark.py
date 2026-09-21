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
"""Original-width DSV4.1 MoE forward/backward benchmark, without optimizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from types import MethodType

import torch
import torch.distributed as dist
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4SparseMoeBlock

from hyper_parallel.core.multicore.examples.mega_moe.deepseek_v41_accuracy import fp32_accuracy
from hyper_parallel.models.deepseek_v41.adapter.activation import configure_deepseek_v41_swiglu
from hyper_parallel.models.deepseek_v41.adapter.expert_parallel import deepseek_v41_ep_compute_fn
from hyper_parallel.models.deepseek_v41.adapter.megamoe_training import (
    DeepseekV41TrainingExperts,
    deepseek_v41_megamoe_compute_fn,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41TopKRouter


class _EpMesh:
    """Expose the existing WORLD group to the production EP factories."""

    def __getitem__(self, _axis: str) -> _EpMesh:
        """Expose the EP axis requested by the model adapter."""
        return self

    @staticmethod
    def size() -> int:
        """Return the degree of the owner EP group."""
        return dist.get_world_size()

    @staticmethod
    def get_group(_axis: str) -> dist.ProcessGroup:
        """Return the explicit group used by the EP collectives.

        Args:
            _axis: EP axis requested by the adapter.
        """
        return dist.group.WORLD


def _build(args):
    config = DeepseekV4Config(  # pylint: disable=unexpected-keyword-arg
        hidden_size=5120, moe_intermediate_size=2304, n_routed_experts=args.experts,
        n_shared_experts=1, num_experts_per_tok=6, num_hidden_layers=1,
        mlp_layer_types=["moe"], swiglu_limit=0.0, scoring_func="sqrtsoftplus", routed_scaling_factor=1.5,
    )
    with torch.device("meta"):
        model = DeepseekV4SparseMoeBlock(config, 0)
        model.gate = DeepseekV41TopKRouter(config)
        for name, parameter in tuple(model.experts.named_parameters()):
            shape = (args.experts // dist.get_world_size(), *parameter.shape[1:])
            setattr(model.experts, name, torch.nn.Parameter(torch.empty(shape)))
    model.to_empty(device="cpu")
    checksum = hashlib.sha256()
    # Initialize canonical HF-layout local shards before any backend conversion.
    with torch.no_grad():
        for index, (name, parameter) in enumerate(model.named_parameters()):
            seed = 42 + index * 100 + (dist.get_rank() if name.startswith("experts.") else 0)
            if name.endswith("bias"):
                parameter.zero_()
            else:
                parameter.normal_(std=0.02, generator=torch.Generator().manual_seed(seed))
    model.to(dtype=torch.bfloat16)
    for name, value in model.state_dict().items():
        checksum.update(name.encode())
        checksum.update(value.view(torch.uint8).numpy().tobytes())
    configure_deepseek_v41_swiglu(model)
    if args.backend == "megamoe":
        model.experts = DeepseekV41TrainingExperts(module=model.experts)
        compute = deepseek_v41_megamoe_compute_fn(
            module=model, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=_EpMesh(),
            local_num_tokens=args.tokens, dispatch_mode=args.dispatch_mode,
            expert_capacity_factor=args.expert_capacity_factor)
    else:
        compute = deepseek_v41_ep_compute_fn(module=model, mesh=None, tp_mesh=None,
                                           cp_mesh=None, ep_mesh=_EpMesh(), use_grouped_gemm=True)
    model.forward = MethodType(compute, model)
    return model.to(device="npu"), checksum.hexdigest()


def _reference_evidence(args, model, hidden, output):
    if not args.evidence_dir:
        return None
    directory = Path(args.evidence_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank{dist.get_rank()}.pt"
    actual = {"output": output.detach().cpu(), "input_grad": hidden.grad.cpu()}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is not None:
            if args.backend == "megamoe" and name.startswith("experts."):
                gradient = gradient.transpose(-1, -2)
            gradient = gradient.detach().cpu().contiguous()
        actual[f"grad/{name}"] = gradient
    if not path.exists():
        if args.backend != "owner_ep":
            raise ValueError("Generate owner_ep evidence before checking MegaMoe")
        torch.save(actual, path)
        return {"reference": "owner_ep_bf16", "written": True, "passed": True}
    expected = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if actual.keys() != expected.keys():
        raise ValueError("Cross-backend gradient names differ")
    # Supplemental BF16 comparison uses the same norm bounds; the independent
    # FP32 oracle remains a separate validation and is explicitly labeled.
    checks = [{"name": name, **fp32_accuracy(value, expected[name])} for name, value in actual.items()]
    return {"reference": "owner_ep_bf16", "checks": checks, "passed": all(c["passed"] for c in checks)}


def _check_finite(model, hidden, output):
    finite = torch.isfinite(output).all() & torch.isfinite(hidden.grad).all()
    for parameter in model.parameters():
        if parameter.grad is not None:
            finite &= torch.isfinite(parameter.grad).all()
    if not finite.item():
        raise RuntimeError("Nonfinite MoE output or gradients")


def _capacity_factor(value: str) -> float | None:
    """Parse ``none`` or a finite factor of at least one."""
    if value.lower() == "none":
        return None
    try:
        factor = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"capacity factor must be 'none' or a number, got {value!r}."
        ) from error
    if not math.isfinite(factor) or factor < 1.0:
        raise argparse.ArgumentTypeError(
            f"capacity factor must be finite and at least 1.0, got {value!r}."
        )
    return factor


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("owner_ep", "megamoe"), required=True)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--dispatch-mode", choices=("push", "pull"), default="push")
    parser.add_argument("--expert-capacity-factor", type=_capacity_factor, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--evidence-dir", help="Optional canonical BF16 output/gradient evidence directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    world = int(os.environ["WORLD_SIZE"])
    if world < 2 or args.experts < 6 or args.experts % world or args.tokens < 128 or args.tokens % 128:
        parser.error("requires EP >= 2, divisible experts >= 6 and tokens a positive multiple of 128")
    if args.warmup < 0 or args.steps < 1:
        parser.error("warmup must be nonnegative and steps positive")
    return args


def main() -> None:
    """Time learned router, routed/shared experts and their backward together."""
    args = _arguments()
    world = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    model, checksum = _build(args)
    generator = torch.Generator().manual_seed(7000 + dist.get_rank())
    hidden = torch.randn(1, args.tokens, 5120, generator=generator).to("npu", torch.bfloat16).requires_grad_()
    dy = torch.randn(hidden.shape, generator=generator).to("npu", torch.bfloat16) / args.tokens ** 0.5
    with torch.no_grad():
        logits, weights, indices = model.gate(hidden.detach())
    route_hash = hashlib.sha256(indices.cpu().numpy().tobytes()).hexdigest()
    counts = torch.bincount(indices.flatten(), minlength=args.experts)
    dist.all_reduce(counts)
    owner_counts = counts.reshape(world, -1).sum(1).cpu().tolist()
    del logits, weights, indices, counts
    rows = []
    torch.npu.reset_peak_memory_stats()
    try:
        for step in range(args.warmup + args.steps):
            model.zero_grad(set_to_none=True)
            hidden.grad = None
            dist.barrier()
            torch.npu.synchronize()
            start = time.perf_counter()
            output = model(hidden)
            output.backward(dy)
            torch.npu.synchronize()
            elapsed = time.perf_counter() - start
            times = torch.tensor([elapsed], device="npu")
            dist.all_reduce(times, op=dist.ReduceOp.MAX)
            rows.append({"step": step, "warmup": step < args.warmup,
                         "local_seconds": elapsed, "max_rank_seconds": times.item()})
        peak_allocated = torch.npu.max_memory_allocated()
        peak_reserved = torch.npu.max_memory_reserved()
        _check_finite(model, hidden, output)
        evidence = _reference_evidence(args, model, hidden, output)
        measured = [row["max_rank_seconds"] for row in rows if not row["warmup"]]
        result = {**vars(args), "output": str(args.output), "rank": dist.get_rank(), "world_size": world,
                  "scope": "moe_forward_backward_no_optimizer_no_dense_dp_reduction",
                  "owner_use_grouped_gemm": args.backend == "owner_ep",
                  "hidden_size": 5120, "intermediate_size": 2304, "top_k": 6, "swiglu_limit": 0,
                  "initial_weight_sha256": checksum, "route_sha256": route_hash,
                  "owner_received_tokens": owner_counts, "finite": True, "steps_detail": rows,
                  "median_seconds": statistics.median(measured),
                  "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
                  "reference_evidence": evidence}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank{dist.get_rank()}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if evidence is not None and not evidence["passed"]:
            raise RuntimeError("Supplemental owner EP BF16 output/gradient comparison failed")
    finally:
        if isinstance(model.experts, DeepseekV41TrainingExperts):
            model.experts.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
