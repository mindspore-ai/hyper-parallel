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
"""Audit graph capture behavior for SimpleFSDP, DTensor, and formal FSDP paths.

Run one target with:
    python -m torch.distributed.run --nproc_per_node=4 \
        examples/torch/fully_shard/graph_comm_audit_demo.py \
        --target simplefsdp_functional --dump-graph

This demo is a diagnostic probe, not a stable regression test. The stable
regression baseline is ``tests/torch/fully_shard/test_simple_fsdp_graph_correctness.py``.
"""
# pylint: disable=C0413
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"


def _set_default_hccl_socket_port_range() -> None:
    """Avoid Ascend HCCL default socket port collisions across local jobs."""
    if "HCCL_NPU_SOCKET_PORT_RANGE" in os.environ:
        return
    master_port = int(os.environ.get("MASTER_PORT", "29500"))
    start_port = 20000 + (master_port % 30000)
    end_port = start_port + 127
    os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = f"{start_port}-{end_port}"


_set_default_hccl_socket_port_range()


def _restore_env(name: str, value: str | None) -> None:
    """Restore an environment variable saved before one target runs."""
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


import argparse
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
# pylint: disable=W0611
import torch_npu
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx

from hyper_parallel import DTensor, SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.experiments.graph_trainer import (
    GraphTrainerConfig,
    SimpleGraphTrainer,
    make_fwd_bwd_step,
)
from hyper_parallel.experiments.graph_trainer.graph_debug import (
    CollectiveGraphSummary,
    dump_graph_debug,
    summarize_collectives,
)
from hyper_parallel.experiments.simple_fsdp import simple_fsdp

_TARGETS = (
    "simplefsdp_functional",
    "simplefsdp_c10d",
    "dtensor_redistribute",
    "dtensor_redistribute_warmup_compile",
    "dtensor_redistribute_warmup_lowered_compile",
    "dtensor_redistribute_warmup_legacy_make_fx",
    "functional_all_gather",
    "functional_reduce_scatter",
    "functional_all_reduce",
    "functional_all_to_all",
    "formal_fully_shard",
)


@dataclass
class AuditResult:
    """Compact per-target graph communication audit result."""

    target: str
    phase: str
    trace_ok: bool
    correctness_ok: bool
    contains_all_gather: bool
    contains_reduce_scatter: bool
    contains_all_reduce: bool
    contains_all_to_all: bool
    collective_node_count: int
    max_abs_error: float
    max_rel_error: float
    failure_type: str
    failure_message: str
    dump_dir: str


class MLPBlock(nn.Module):
    """Single MLP block: Linear, GELU, Linear."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MiniModel(nn.Module):
    """Small model covering linear, residual, layernorm, and head parameters."""

    def __init__(self, dim: int = 64, depth: int = 2):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.layers = nn.ModuleList([
            MLPBlock(dim, dim * 4) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for layer in self.layers:
            x = x + layer(x)
        x = self.norm(x)
        return self.head(x)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Audit graph capture of distributed communication paths.")
    parser.add_argument("--target", choices=(*_TARGETS, "all"), required=True)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--dump-graph", action="store_true")
    parser.add_argument("--dump-dir", type=Path, default=Path("logs/graph_comm_audit"))
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="For dtensor_redistribute_warmup_compile, verify strict cache miss without eager warmup.",
    )
    return parser.parse_args()


def init_dist() -> tuple[int, int]:
    """Initialize process group and bind one NPU per process."""
    if not dist.is_initialized():
        dist.init_process_group()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_rank)
    return rank, local_rank


def _loss_fn(output: torch.Tensor) -> torch.Tensor:
    """Use a scalar mean loss so the train step includes backward."""
    return output.mean()


def _build_mesh():
    """Build a one-dimensional DP mesh from the current process group."""
    world_size = dist.get_world_size()
    return init_device_mesh(
        device_type="npu",
        mesh_shape=(world_size,),
        mesh_dim_names=("dp",),
    )


def _build_simplefsdp_model(seed: int, dim: int, depth: int, mesh) -> nn.Module:
    """Build a deterministic SimpleFSDP-wrapped model."""
    torch.manual_seed(seed)
    model = MiniModel(dim=dim, depth=depth).npu()
    return simple_fsdp(model, mesh, shard_dim=0)


def _build_formal_fully_shard_model(
    seed: int,
    dim: int,
    depth: int,
    mesh,
    reshard_after_forward: bool = True,
) -> nn.Module:
    """Build a deterministic model wrapped by formal HyperParallel fully_shard."""
    torch.manual_seed(seed)
    model = MiniModel(dim=dim, depth=depth).npu()
    return fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=reshard_after_forward,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
            cast_forward_inputs=True,
        ),
    )


def _detach_outputs(outputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Detach and clone graph/eager outputs before another run mutates state."""
    return tuple(output.detach().clone() for output in outputs)


def _max_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return max absolute and relative errors on device."""
    abs_error = (actual - expected).abs().max()
    denom = expected.abs().max().clamp_min(1e-12)
    return abs_error, abs_error / denom


def _all_reduce_max(value: torch.Tensor) -> torch.Tensor:
    """Return maximum scalar value across ranks."""
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
    return reduced


def _all_ranks_pass(local_pass: bool, device: torch.device) -> bool:
    """Return whether every rank reported local pass."""
    world_size = dist.get_world_size()
    pass_tensor = torch.tensor(1 if local_pass else 0, device=device, dtype=torch.int32)
    dist.all_reduce(pass_tensor, op=dist.ReduceOp.SUM)
    return pass_tensor.item() == world_size


def _compare_outputs(
    actual_outputs: tuple[torch.Tensor, ...],
    expected_outputs: tuple[torch.Tensor, ...],
    rtol: float,
    atol: float,
) -> tuple[bool, torch.Tensor, torch.Tensor]:
    """Compare output tuples and return all-rank pass plus max errors."""
    if len(actual_outputs) != len(expected_outputs):
        device = actual_outputs[0].device if actual_outputs else expected_outputs[0].device
        return False, torch.full((), float("inf"), device=device), torch.full((), float("inf"), device=device)

    local_pass = True
    max_abs_error = torch.zeros((), device=actual_outputs[0].device)
    max_rel_error = torch.zeros((), device=actual_outputs[0].device)
    for actual, expected in zip(actual_outputs, expected_outputs):
        if actual.shape != expected.shape:
            local_pass = False
            max_abs_error = torch.full((), float("inf"), device=actual_outputs[0].device)
            max_rel_error = torch.full((), float("inf"), device=actual_outputs[0].device)
            break
        abs_error, rel_error = _max_errors(actual, expected)
        max_abs_error = torch.maximum(max_abs_error, abs_error)
        max_rel_error = torch.maximum(max_rel_error, rel_error)
        local_pass = local_pass and bool(torch.allclose(actual, expected, rtol=rtol, atol=atol))

    global_abs_error = _all_reduce_max(max_abs_error)
    global_rel_error = _all_reduce_max(max_rel_error)
    global_pass = _all_ranks_pass(local_pass, actual_outputs[0].device)
    return global_pass, global_abs_error, global_rel_error


def _run_eager_fwd_bwd(
    model: nn.Module,
    param_items: list[tuple[str, torch.Tensor]],
    data: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Run eager fwd+bwd and return detached loss plus grads."""
    step_fn = make_fwd_bwd_step(model, _loss_fn)
    grad_params = tuple(param for _, param in param_items)
    outputs = _detach_outputs(step_fn(grad_params, data))
    torch.npu.synchronize()
    return outputs


def _empty_summary() -> CollectiveGraphSummary:
    """Return an empty collective summary."""
    return CollectiveGraphSummary(total_nodes=0, collective_nodes=[])


def _failure_result(
    target: str,
    dump_dir: Path,
    error: BaseException,
    phase: str = "train_step",
) -> AuditResult:
    """Build an AuditResult for an exception."""
    return AuditResult(
        target=target,
        phase=phase,
        trace_ok=False,
        correctness_ok=False,
        contains_all_gather=False,
        contains_reduce_scatter=False,
        contains_all_reduce=False,
        contains_all_to_all=False,
        collective_node_count=0,
        max_abs_error=float("inf"),
        max_rel_error=float("inf"),
        failure_type=type(error).__name__,
        failure_message=str(error).splitlines()[0] if str(error) else repr(error),
        dump_dir=str(dump_dir),
    )


def _result_from_summary(
    target: str,
    summary: CollectiveGraphSummary,
    correctness_ok: bool,
    max_abs_error: torch.Tensor,
    max_rel_error: torch.Tensor,
    dump_dir: Path,
    phase: str = "train_step",
    failure_type: str = "",
    failure_message: str = "",
) -> AuditResult:
    """Build a successful trace AuditResult."""
    return AuditResult(
        target=target,
        phase=phase,
        trace_ok=True,
        correctness_ok=correctness_ok,
        contains_all_gather=summary.contains("all_gather"),
        contains_reduce_scatter=summary.contains("reduce_scatter"),
        contains_all_reduce=summary.contains("all_reduce"),
        contains_all_to_all=summary.contains("all_to_all"),
        collective_node_count=summary.collective_node_count,
        max_abs_error=float(max_abs_error.detach().cpu().item()),
        max_rel_error=float(max_rel_error.detach().cpu().item()),
        failure_type=failure_type,
        failure_message=failure_message,
        dump_dir=str(dump_dir),
    )


def _functional_collectives_module():
    """Return PyTorch functional collectives used by the graph-capture probes."""
    # pylint: disable=C0415
    from torch.distributed import _functional_collectives as functional_collectives

    return functional_collectives


def _wait_functional_output(output: torch.Tensor, functional_collectives) -> torch.Tensor:
    """Materialize an AsyncCollectiveTensor-like output when the backend returns one."""
    wait_tensor = getattr(functional_collectives, "wait_tensor", None)
    if wait_tensor is not None:
        return wait_tensor(output)
    wait = getattr(output, "wait", None)
    if wait is not None:
        return wait()
    return output


def _functional_collective_step(target: str, group) -> Callable[[torch.Tensor], tuple[torch.Tensor, ...]]:
    """Build a raw functional collective step for one communication target."""
    functional_collectives = _functional_collectives_module()

    if target == "functional_all_gather":
        all_gather_tensor = functional_collectives.all_gather_tensor

        def step(input_tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
            gathered = all_gather_tensor(input_tensor, 0, group)
            return (_wait_functional_output(gathered, functional_collectives),)

        return step

    if target == "functional_reduce_scatter":
        reduce_scatter_tensor = functional_collectives.reduce_scatter_tensor

        def step(input_tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
            reduced = reduce_scatter_tensor(input_tensor, "sum", 0, group)
            return (_wait_functional_output(reduced, functional_collectives),)

        return step

    if target == "functional_all_reduce":
        all_reduce = functional_collectives.all_reduce

        def step(input_tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
            reduced = all_reduce(input_tensor, "sum", group)
            return (_wait_functional_output(reduced, functional_collectives),)

        return step

    if target == "functional_all_to_all":
        all_to_all_single = functional_collectives.all_to_all_single

        def step(input_tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
            exchanged = all_to_all_single(input_tensor, None, None, group)
            return (_wait_functional_output(exchanged, functional_collectives),)

        return step

    raise ValueError(f"Unsupported functional collective target: {target}")


def _c10d_collective_reference(
    target: str,
    group,
    world_size: int,
) -> Callable[[torch.Tensor], tuple[torch.Tensor, ...]]:
    """Build a c10d eager reference for one functional collective target."""
    if target == "functional_all_gather":

        def reference(input_tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
            output_shape = (input_tensor.shape[0] * world_size, *input_tensor.shape[1:])
            output = torch.empty(output_shape, device=input_tensor.device, dtype=input_tensor.dtype)
            dist.all_gather_into_tensor(output, input_tensor, group=group)
            return (output,)

        return reference

    if target == "functional_reduce_scatter":

        def reference(input_tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
            output_shape = (input_tensor.shape[0] // world_size, *input_tensor.shape[1:])
            output = torch.empty(output_shape, device=input_tensor.device, dtype=input_tensor.dtype)
            dist.reduce_scatter_tensor(output, input_tensor, group=group)
            return (output,)

        return reference

    if target == "functional_all_reduce":

        def reference(input_tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
            output = input_tensor.detach().clone()
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
            return (output,)

        return reference

    if target == "functional_all_to_all":

        def reference(input_tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
            output = torch.empty_like(input_tensor)
            dist.all_to_all_single(output, input_tensor, group=group)
            return (output,)

        return reference

    raise ValueError(f"Unsupported c10d reference target: {target}")


def _functional_collective_input(target: str, args: argparse.Namespace) -> torch.Tensor:
    """Build deterministic rank-distinct input for a functional collective target."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rows = max(1, args.batch_size)
    feature_dim = max(1, args.dim)
    rows = local_rows * world_size if target in (
        "functional_reduce_scatter",
        "functional_all_to_all",
    ) else local_rows
    values = torch.arange(rows * feature_dim, dtype=torch.float32).reshape(rows, feature_dim).npu()
    return values + rank * 1000.0


def _merge_compare_results(
    results: tuple[tuple[bool, torch.Tensor, torch.Tensor], ...]
) -> tuple[bool, torch.Tensor, torch.Tensor]:
    """Merge multiple comparison results into one pass flag and max error pair."""
    correctness_ok = all(result[0] for result in results)
    max_abs_error = results[0][1]
    max_rel_error = results[0][2]
    for _, abs_error, rel_error in results[1:]:
        max_abs_error = torch.maximum(max_abs_error, abs_error)
        max_rel_error = torch.maximum(max_rel_error, rel_error)
    return correctness_ok, max_abs_error, max_rel_error


def _audit_functional_collective(target: str, args: argparse.Namespace, mesh, dump_dir: Path) -> AuditResult:
    """Audit eager, make_fx, and torch.compile behavior for one functional collective."""
    group = mesh.get_comm_group_by_axis("dp")
    world_size = dist.get_world_size()
    input_tensor = _functional_collective_input(target, args)
    step = _functional_collective_step(target, group)
    reference = _c10d_collective_reference(target, group, world_size)

    reference_outputs = _detach_outputs(reference(input_tensor))
    eager_outputs = _detach_outputs(step(input_tensor))

    graph_module = make_fx(step)(input_tensor)
    graph_summary = summarize_collectives(graph_module)
    if args.dump_graph and (not dist.is_initialized() or dist.get_rank() == 0):
        dump_graph_debug(graph_module, dump_dir, target, graph_summary)
    graph_outputs = _detach_outputs(graph_module(input_tensor))

    compiled_step = torch.compile(step, fullgraph=True, backend="aot_eager")
    compiled_outputs = _detach_outputs(compiled_step(input_tensor))
    torch.npu.synchronize()

    correctness_ok, max_abs_error, max_rel_error = _merge_compare_results(
        (
            _compare_outputs(eager_outputs, reference_outputs, args.rtol, args.atol),
            _compare_outputs(graph_outputs, reference_outputs, args.rtol, args.atol),
            _compare_outputs(compiled_outputs, reference_outputs, args.rtol, args.atol),
        )
    )
    return _result_from_summary(
        target,
        graph_summary,
        correctness_ok,
        max_abs_error,
        max_rel_error,
        dump_dir,
        phase="functional_eager_make_fx_compile",
    )


def _audit_simplefsdp(
    target: str,
    args: argparse.Namespace,
    mesh,
    dump_dir: Path,
) -> AuditResult:
    """Audit SimpleFSDP functional or c10d communication in a full train-step graph."""
    os.environ["HYPER_PARALLEL_SIMPLE_FSDP_FUNCTIONAL"] = (
        "1" if target == "simplefsdp_functional" else "0"
    )
    eager_model = _build_simplefsdp_model(args.seed, args.dim, args.depth, mesh)
    graph_model = _build_simplefsdp_model(args.seed, args.dim, args.depth, mesh)
    torch.manual_seed(args.seed + 1)
    data = torch.randn(args.batch_size, args.seq_len, args.dim).npu()

    trainer = SimpleGraphTrainer(
        graph_model,
        _loss_fn,
        config=GraphTrainerConfig(dump_graph=args.dump_graph, dump_dir=dump_dir),
    )
    eager_outputs = _run_eager_fwd_bwd(
        eager_model,
        list(eager_model.named_parameters(remove_duplicate=False)),
        data,
    )
    trainer.trace(data)
    graph_result = trainer.step(data)
    torch.npu.synchronize()
    graph_outputs = (graph_result.loss.detach(), *[grad.detach() for grad in graph_result.grads])
    correctness_ok, max_abs_error, max_rel_error = _compare_outputs(
        graph_outputs,
        eager_outputs,
        args.rtol,
        args.atol,
    )
    return _result_from_summary(
        target,
        trainer.graph_summary,
        correctness_ok,
        max_abs_error,
        max_rel_error,
        dump_dir,
    )


def _dtensor_redistribute_step(mesh) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Build a DTensor redistribute function that exercises all-gather and reduce-scatter."""

    def step(shard_input: torch.Tensor, partial_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shard_dtensor = DTensor.from_local(shard_input, mesh, [Shard(0)])
        gathered = shard_dtensor.redistribute(mesh, [Replicate()]).to_local()
        partial_dtensor = DTensor.from_local(partial_input, mesh, [Partial("sum")])
        reduced = partial_dtensor.redistribute(mesh, [Shard(0)]).to_local()
        return gathered, reduced

    return step


def _dtensor_redistribute_lowered_step(
    mesh,
) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Build graph-friendly raw tensor replay of the warmed DTensor redistribute plan."""
    # pylint: disable=C0415
    from torch.distributed import _functional_collectives as functional_collectives
    group = mesh.get_comm_group_by_axis("dp")
    all_gather_tensor = functional_collectives.all_gather_tensor
    reduce_scatter_tensor = functional_collectives.reduce_scatter_tensor
    wait_tensor = getattr(functional_collectives, "wait_tensor", None)
    if wait_tensor is None:
        raise RuntimeError("functional collective wait_tensor is unavailable")

    def step(shard_input: torch.Tensor, partial_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gathered = all_gather_tensor(shard_input, 0, group)
        gathered = wait_tensor(gathered)
        reduced = reduce_scatter_tensor(partial_input, "sum", 0, group)
        reduced = wait_tensor(reduced)
        return gathered, reduced

    return step


def _clear_dtensor_metadata_caches() -> None:
    """Clear private DTensor metadata caches so warmup owns the observed cache hits."""
    # pylint: disable=C0415,W0212
    from hyper_parallel.core.dtensor import dtensor as dtensor_module
    from hyper_parallel.core.dtensor.tensor_redistribution import _tensor_redistribution

    dtensor_module._LAYOUT_CACHE.clear()
    _tensor_redistribution.clear_warmup_cache()


def _audit_dtensor_redistribute(args: argparse.Namespace, mesh, dump_dir: Path) -> AuditResult:
    """Audit formal DTensor.redistribute graph capture and eager equivalence."""
    os.environ["HYPER_PARALLEL_DTENSOR_FUNCTIONAL_COLLECTIVES"] = "1"
    os.environ["HYPER_PARALLEL_DTENSOR_WARMUP_CACHE"] = "0"
    os.environ["HYPER_PARALLEL_DTENSOR_TRACE_CACHE_STRICT"] = "0"
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(args.seed)
    full_shape = (args.dim * world_size, args.dim)
    full_tensor = torch.randn(full_shape).npu()
    shard_input = full_tensor.chunk(world_size, dim=0)[rank].contiguous()

    torch.manual_seed(args.seed + 1)
    partial_input = torch.randn(full_shape).npu()

    step = _dtensor_redistribute_step(mesh)
    eager_outputs = _detach_outputs(step(shard_input, partial_input))
    graph_module = make_fx(step)(shard_input, partial_input)
    graph_summary = summarize_collectives(graph_module)
    if args.dump_graph and (not dist.is_initialized() or dist.get_rank() == 0):
        dump_graph_debug(graph_module, dump_dir, "dtensor_redistribute", graph_summary)
    graph_outputs = _detach_outputs(graph_module(shard_input, partial_input))
    torch.npu.synchronize()

    correctness_ok, max_abs_error, max_rel_error = _compare_outputs(
        graph_outputs,
        eager_outputs,
        args.rtol,
        args.atol,
    )
    return _result_from_summary(
        "dtensor_redistribute",
        graph_summary,
        correctness_ok,
        max_abs_error,
        max_rel_error,
        dump_dir,
    )


def _audit_dtensor_redistribute_warmup_compile(
    args: argparse.Namespace,
    mesh,
    dump_dir: Path,
) -> AuditResult:
    """Audit whether eager warmup helps fullgraph compile of DTensor.redistribute."""
    old_functional = os.environ.get("HYPER_PARALLEL_DTENSOR_FUNCTIONAL_COLLECTIVES")
    old_warmup = os.environ.get("HYPER_PARALLEL_DTENSOR_WARMUP_CACHE")
    old_strict = os.environ.get("HYPER_PARALLEL_DTENSOR_TRACE_CACHE_STRICT")
    try:
        os.environ["HYPER_PARALLEL_DTENSOR_FUNCTIONAL_COLLECTIVES"] = "1"
        os.environ["HYPER_PARALLEL_DTENSOR_WARMUP_CACHE"] = "1"
        os.environ["HYPER_PARALLEL_DTENSOR_TRACE_CACHE_STRICT"] = "0"
        # pylint: disable=C0415
        from hyper_parallel.core.dtensor.tensor_redistribution import _tensor_redistribution
        _tensor_redistribution.clear_warmup_cache()

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.manual_seed(args.seed)
        full_shape = (args.dim * world_size, args.dim)
        full_tensor = torch.randn(full_shape).npu()
        shard_input = full_tensor.chunk(world_size, dim=0)[rank].contiguous()

        torch.manual_seed(args.seed + 1)
        partial_input = torch.randn(full_shape).npu()

        step = _dtensor_redistribute_step(mesh)

        if args.skip_warmup:
            os.environ["HYPER_PARALLEL_DTENSOR_TRACE_CACHE_STRICT"] = "1"
            compiled_step = torch.compile(step, fullgraph=True, backend="aot_eager")
            compiled_outputs = _detach_outputs(compiled_step(shard_input, partial_input))
            torch.npu.synchronize()
            zero_error = torch.zeros((), device=compiled_outputs[0].device)
            return _result_from_summary(
                "dtensor_redistribute_warmup_compile",
                _empty_summary(),
                False,
                zero_error,
                zero_error,
                dump_dir,
                phase="strict_cache_miss_probe",
                failure_type="expected_cache_miss_not_raised",
                failure_message="strict compile unexpectedly succeeded without eager warmup",
            )

        # Warmup intentionally runs the exact same HP DTensor path in eager
        # before compile. This populates layout, mesh/group, and reduce plans.
        warmup_outputs = _detach_outputs(step(shard_input, partial_input))
        torch.npu.synchronize()

        os.environ["HYPER_PARALLEL_DTENSOR_TRACE_CACHE_STRICT"] = "1"
        compiled_step = torch.compile(step, fullgraph=True, backend="aot_eager")
        compiled_outputs = _detach_outputs(compiled_step(shard_input, partial_input))
        torch.npu.synchronize()

        # torch.compile(aot_eager) does not directly expose its graph here, so
        # keep a post-warmup make_fx probe only for collective-node inspection.
        graph_summary = _empty_summary()
        failure_type = ""
        failure_message = ""
        try:
            graph_module = make_fx(step)(shard_input, partial_input)
            graph_summary = summarize_collectives(graph_module)
            if args.dump_graph and (not dist.is_initialized() or dist.get_rank() == 0):
                dump_graph_debug(graph_module, dump_dir, "dtensor_redistribute_warmup_compile", graph_summary)
        except Exception as graph_error:  # pylint: disable=broad-except
            failure_type = f"make_fx:{type(graph_error).__name__}"
            failure_message = str(graph_error).splitlines()[0] if str(graph_error) else repr(graph_error)

        correctness_ok, max_abs_error, max_rel_error = _compare_outputs(
            compiled_outputs,
            warmup_outputs,
            args.rtol,
            args.atol,
        )
        return _result_from_summary(
            "dtensor_redistribute_warmup_compile",
            graph_summary,
            correctness_ok,
            max_abs_error,
            max_rel_error,
            dump_dir,
            phase="warmup_then_fullgraph_compile",
            failure_type=failure_type,
            failure_message=failure_message,
        )
    finally:
        _restore_env("HYPER_PARALLEL_DTENSOR_FUNCTIONAL_COLLECTIVES", old_functional)
        _restore_env("HYPER_PARALLEL_DTENSOR_WARMUP_CACHE", old_warmup)
        _restore_env("HYPER_PARALLEL_DTENSOR_TRACE_CACHE_STRICT", old_strict)


def _audit_dtensor_redistribute_warmup_lowered_compile(
    args: argparse.Namespace,
    mesh,
    dump_dir: Path,
) -> AuditResult:
    """Audit warmup plus graph-friendly lowered replay of DTensor.redistribute."""
    old_functional = os.environ.get("HYPER_PARALLEL_DTENSOR_FUNCTIONAL_COLLECTIVES")
    old_warmup = os.environ.get("HYPER_PARALLEL_DTENSOR_WARMUP_CACHE")
    old_strict = os.environ.get("HYPER_PARALLEL_DTENSOR_TRACE_CACHE_STRICT")
    try:
        os.environ["HYPER_PARALLEL_DTENSOR_FUNCTIONAL_COLLECTIVES"] = "1"
        os.environ["HYPER_PARALLEL_DTENSOR_WARMUP_CACHE"] = "1"
        os.environ["HYPER_PARALLEL_DTENSOR_TRACE_CACHE_STRICT"] = "0"
        # pylint: disable=C0415
        from hyper_parallel.core.dtensor.tensor_redistribution import _tensor_redistribution
        _tensor_redistribution.clear_warmup_cache()

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.manual_seed(args.seed)
        full_shape = (args.dim * world_size, args.dim)
        full_tensor = torch.randn(full_shape).npu()
        shard_input = full_tensor.chunk(world_size, dim=0)[rank].contiguous()

        torch.manual_seed(args.seed + 1)
        partial_input = torch.randn(full_shape).npu()

        warmup_step = _dtensor_redistribute_step(mesh)
        warmup_outputs = _detach_outputs(warmup_step(shard_input, partial_input))
        torch.npu.synchronize()

        os.environ["HYPER_PARALLEL_DTENSOR_TRACE_CACHE_STRICT"] = "1"
        lowered_step = _dtensor_redistribute_lowered_step(mesh)
        compiled_step = torch.compile(lowered_step, fullgraph=True, backend="aot_eager")
        compiled_outputs = _detach_outputs(compiled_step(shard_input, partial_input))
        torch.npu.synchronize()

        graph_module = make_fx(lowered_step)(shard_input, partial_input)
        graph_summary = summarize_collectives(graph_module)
        if args.dump_graph and (not dist.is_initialized() or dist.get_rank() == 0):
            dump_graph_debug(graph_module, dump_dir, "dtensor_redistribute_warmup_lowered_compile", graph_summary)

        correctness_ok, max_abs_error, max_rel_error = _compare_outputs(
            compiled_outputs,
            warmup_outputs,
            args.rtol,
            args.atol,
        )
        return _result_from_summary(
            "dtensor_redistribute_warmup_lowered_compile",
            graph_summary,
            correctness_ok,
            max_abs_error,
            max_rel_error,
            dump_dir,
            phase="warmup_then_lowered_fullgraph_compile",
        )
    finally:
        _restore_env("HYPER_PARALLEL_DTENSOR_FUNCTIONAL_COLLECTIVES", old_functional)
        _restore_env("HYPER_PARALLEL_DTENSOR_WARMUP_CACHE", old_warmup)
        _restore_env("HYPER_PARALLEL_DTENSOR_TRACE_CACHE_STRICT", old_strict)


def _audit_dtensor_redistribute_warmup_legacy_make_fx(
    args: argparse.Namespace,
    mesh,
    dump_dir: Path,
) -> AuditResult:
    """Audit eager warmup using the current Layout and platform collective implementations."""
    old_functional = os.environ.get("HYPER_PARALLEL_DTENSOR_FUNCTIONAL_COLLECTIVES")
    old_warmup = os.environ.get("HYPER_PARALLEL_DTENSOR_WARMUP_CACHE")
    old_strict = os.environ.get("HYPER_PARALLEL_DTENSOR_TRACE_CACHE_STRICT")
    try:
        os.environ["HYPER_PARALLEL_DTENSOR_FUNCTIONAL_COLLECTIVES"] = "0"
        os.environ["HYPER_PARALLEL_DTENSOR_WARMUP_CACHE"] = "1"
        os.environ["HYPER_PARALLEL_DTENSOR_TRACE_CACHE_STRICT"] = "0"
        _clear_dtensor_metadata_caches()

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.manual_seed(args.seed)
        full_shape = (args.dim * world_size, args.dim)
        full_tensor = torch.randn(full_shape).npu()
        shard_input = full_tensor.chunk(world_size, dim=0)[rank].contiguous()

        torch.manual_seed(args.seed + 1)
        partial_input = torch.randn(full_shape).npu()

        step = _dtensor_redistribute_step(mesh)

        # Warmup uses the current workspace implementation. Trace strict mode
        # below must then hit the warmed caches instead of rebuilding
        # layouts/plans.
        warmup_outputs = _detach_outputs(step(shard_input, partial_input))
        torch.npu.synchronize()

        os.environ["HYPER_PARALLEL_DTENSOR_TRACE_CACHE_STRICT"] = "1"
        graph_module = make_fx(step)(shard_input, partial_input)
        graph_summary = summarize_collectives(graph_module)
        if args.dump_graph and (not dist.is_initialized() or dist.get_rank() == 0):
            dump_graph_debug(graph_module, dump_dir, "dtensor_redistribute_warmup_legacy_make_fx", graph_summary)
        graph_outputs = _detach_outputs(graph_module(shard_input, partial_input))
        torch.npu.synchronize()

        correctness_ok, max_abs_error, max_rel_error = _compare_outputs(
            graph_outputs,
            warmup_outputs,
            args.rtol,
            args.atol,
        )
        return _result_from_summary(
            "dtensor_redistribute_warmup_legacy_make_fx",
            graph_summary,
            correctness_ok,
            max_abs_error,
            max_rel_error,
            dump_dir,
            phase="current_layout_platform_warmup_then_make_fx",
        )
    finally:
        _restore_env("HYPER_PARALLEL_DTENSOR_FUNCTIONAL_COLLECTIVES", old_functional)
        _restore_env("HYPER_PARALLEL_DTENSOR_WARMUP_CACHE", old_warmup)
        _restore_env("HYPER_PARALLEL_DTENSOR_TRACE_CACHE_STRICT", old_strict)


def _audit_formal_fully_shard_train_step(args: argparse.Namespace, mesh, dump_dir: Path) -> AuditResult:
    """Probe formal fully_shard full train-step capture."""
    eager_model = _build_formal_fully_shard_model(args.seed, args.dim, args.depth, mesh)
    graph_model = _build_formal_fully_shard_model(args.seed, args.dim, args.depth, mesh)
    torch.manual_seed(args.seed + 1)
    data = torch.randn(args.batch_size, args.seq_len, args.dim).npu()

    trainer = SimpleGraphTrainer(
        graph_model,
        _loss_fn,
        config=GraphTrainerConfig(dump_graph=args.dump_graph, dump_dir=dump_dir),
    )
    with SkipDTensorDispatch():
        eager_outputs = _run_eager_fwd_bwd(
            eager_model,
            list(eager_model.named_parameters(remove_duplicate=False)),
            data,
        )
        trainer.trace(data)
        graph_result = trainer.step(data)
    torch.npu.synchronize()

    graph_outputs = (graph_result.loss.detach(), *[grad.detach() for grad in graph_result.grads])
    correctness_ok, max_abs_error, max_rel_error = _compare_outputs(
        graph_outputs,
        eager_outputs,
        args.rtol,
        args.atol,
    )
    return _result_from_summary(
        "formal_fully_shard",
        trainer.graph_summary,
        correctness_ok,
        max_abs_error,
        max_rel_error,
        dump_dir,
        phase="train_step",
    )


def _audit_formal_fully_shard_forward_only(
    args: argparse.Namespace,
    mesh,
    dump_dir: Path,
    train_step_error: BaseException,
) -> AuditResult:
    """Probe formal fully_shard forward graph capture after train-step probe fails."""
    eager_model = _build_formal_fully_shard_model(
        args.seed,
        args.dim,
        args.depth,
        mesh,
        reshard_after_forward=False,
    )
    graph_model = _build_formal_fully_shard_model(
        args.seed,
        args.dim,
        args.depth,
        mesh,
        reshard_after_forward=False,
    )
    torch.manual_seed(args.seed + 1)
    data = torch.randn(args.batch_size, args.seq_len, args.dim).npu()

    with torch.no_grad(), SkipDTensorDispatch():
        eager_loss = _loss_fn(eager_model(data)).detach().clone()

    def forward_loss(input_data: torch.Tensor) -> torch.Tensor:
        with SkipDTensorDispatch():
            return _loss_fn(graph_model(input_data))

    graph_module = make_fx(forward_loss)(data)
    graph_summary = summarize_collectives(graph_module)
    if args.dump_graph and (not dist.is_initialized() or dist.get_rank() == 0):
        dump_graph_debug(graph_module, dump_dir, "formal_fully_shard_forward_loss", graph_summary)
    graph_loss = graph_module(data).detach().clone()
    torch.npu.synchronize()

    correctness_ok, max_abs_error, max_rel_error = _compare_outputs(
        (graph_loss,),
        (eager_loss,),
        args.rtol,
        args.atol,
    )
    return _result_from_summary(
        "formal_fully_shard",
        graph_summary,
        correctness_ok,
        max_abs_error,
        max_rel_error,
        dump_dir,
        phase="forward_only_after_train_step_failure",
        failure_type=f"train_step:{type(train_step_error).__name__}",
        failure_message=(
            "train-step probe failed; forward-only probe used "
            "reshard_after_forward=False. "
            f"train-step error: {str(train_step_error).splitlines()[0]}"
            if str(train_step_error) else repr(train_step_error)
        ),
    )


def _audit_formal_fully_shard(args: argparse.Namespace, mesh, dump_dir: Path) -> AuditResult:
    """Probe formal fully_shard graph capture without making it a required passing baseline."""
    try:
        return _audit_formal_fully_shard_train_step(args, mesh, dump_dir)
    except Exception as train_step_error:
        if dist.get_rank() == 0:
            print(
                "[formal_fully_shard] train-step probe failed; "
                "falling back to forward-only probe."
            )
            print(f"train-step failure: {type(train_step_error).__name__}: {train_step_error}")
        try:
            return _audit_formal_fully_shard_forward_only(
                args,
                mesh,
                dump_dir,
                train_step_error,
            )
        except Exception as forward_error:
            if dist.get_rank() == 0:
                print(
                    "[formal_fully_shard] forward-only fallback also failed with "
                    f"{type(forward_error).__name__}: {forward_error}"
                )
            return _failure_result(
                "formal_fully_shard",
                dump_dir,
                forward_error,
                phase="forward_only_after_train_step_failure",
            )


def _run_target(target: str, args: argparse.Namespace, mesh) -> AuditResult:
    """Run one audit target and convert exceptions to structured diagnostics."""
    dump_dir = args.dump_dir / target
    try:
        if target in ("simplefsdp_functional", "simplefsdp_c10d"):
            return _audit_simplefsdp(target, args, mesh, dump_dir)
        if target == "dtensor_redistribute":
            return _audit_dtensor_redistribute(args, mesh, dump_dir)
        if target == "dtensor_redistribute_warmup_compile":
            return _audit_dtensor_redistribute_warmup_compile(args, mesh, dump_dir)
        if target == "dtensor_redistribute_warmup_lowered_compile":
            return _audit_dtensor_redistribute_warmup_lowered_compile(args, mesh, dump_dir)
        if target == "dtensor_redistribute_warmup_legacy_make_fx":
            return _audit_dtensor_redistribute_warmup_legacy_make_fx(args, mesh, dump_dir)
        if target in (
            "functional_all_gather",
            "functional_reduce_scatter",
            "functional_all_reduce",
            "functional_all_to_all",
        ):
            return _audit_functional_collective(target, args, mesh, dump_dir)
        if target == "formal_fully_shard":
            return _audit_formal_fully_shard(args, mesh, dump_dir)
        raise ValueError(f"Unsupported target: {target}")
    except Exception as error:  # pylint: disable=broad-except
        if dist.get_rank() == 0:
            print(f"[{target}] audit failed with {type(error).__name__}: {error}")
            traceback.print_exc()
        return _failure_result(target, dump_dir, error)


def _print_result(result: AuditResult) -> None:
    """Print one target result as stable JSON plus a short human line."""
    print(json.dumps(asdict(result), sort_keys=True))
    status = "PASS" if result.trace_ok and result.correctness_ok and not result.failure_type else "CHECK"
    print(
        f"{status} target={result.target} phase={result.phase} trace_ok={result.trace_ok} "
        f"correctness_ok={result.correctness_ok} all_gather={result.contains_all_gather} "
        f"reduce_scatter={result.contains_reduce_scatter} all_reduce={result.contains_all_reduce} "
        f"all_to_all={result.contains_all_to_all} "
        f"max_abs={result.max_abs_error:.8e} max_rel={result.max_rel_error:.8e}"
    )


def main() -> None:
    """Run graph communication audit target(s)."""
    args = parse_args()
    rank, _ = init_dist()
    mesh = _build_mesh()
    targets = _TARGETS if args.target == "all" else (args.target,)

    for target in targets:
        result = _run_target(target, args, mesh)
        if rank == 0:
            _print_result(result)
        dist.barrier()


if __name__ == "__main__":
    main()
