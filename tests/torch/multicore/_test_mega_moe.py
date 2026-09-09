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

"""Public MegaMoe experts forward/backward system-test workers."""

from __future__ import annotations

import gc
import json
import os
import statistics
import time
from dataclasses import dataclass, fields
from pathlib import Path

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

# The launcher activates CANN and the multicore payload before importing this
# worker. Framework imports stay out of the ``test_mega_moe.py`` launcher.
# pylint: disable=wrong-import-position
import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.expert_parallel.expert_parallel import ExpertParallel
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.platform.torch.common import GroupedExperts
from hyper_parallel.core.multicore.shmem.lifecycle import acquire_symmetric_memory

# pylint: enable=wrong-import-position

_RTOL = 2e-2
_ATOL = 2e-3


def _move_to_test_device(module: torch.nn.Module) -> torch.nn.Module:
    """Move a runtime Torch module to the worker device and dtype."""
    return module.to(device=DEVICE, dtype=torch.bfloat16)


@dataclass(frozen=True)
class MoeShape:
    """Minimal public-layer test shape."""

    local_num_tokens: int = 128
    hidden_size: int = 512
    intermediate_size: int = 128
    num_experts: int = 4
    top_k: int = 2
    ep_size: int = 2

    @property
    def local_experts(self) -> int:
        """Return expert weights owned by each EP rank."""
        return self.num_experts // self.ep_size


@dataclass(frozen=True)
class _RunResult:
    """Forward and backward tensors compared across the two backends."""

    output: torch.Tensor
    input_grad: torch.Tensor
    route_weight_grad: torch.Tensor
    gate_up_weight_grad: torch.Tensor
    down_weight_grad: torch.Tensor
    gate_up_weight: torch.Tensor
    down_weight: torch.Tensor


def _init_dist() -> tuple[int, torch.device]:
    """Bind the local NPU and initialize the EP process group."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    world_size = dist.get_world_size()
    expected_world_size = int(os.getenv("HP_MEGA_MOE_WORLD_SIZE", "2"))
    if world_size != expected_world_size:
        raise ValueError(
            f"worker requires {expected_world_size} ranks, got {world_size}."
        )
    return dist.get_rank(), torch.device("npu", local_rank)


RANK, DEVICE = _init_dist()


def make_balanced_route(
    shape: MoeShape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the balanced token-major route and its exact local histogram."""
    positions = torch.arange(
        shape.local_num_tokens * shape.top_k,
        dtype=torch.int64,
        device=DEVICE,
    ).view(shape.local_num_tokens, shape.top_k)
    topk_ids = (positions + RANK * shape.top_k).remainder(shape.num_experts)
    base_weights = torch.arange(
        shape.top_k,
        0,
        -1,
        dtype=torch.float32,
        device=DEVICE,
    )
    topk_weights = (
        (base_weights / base_weights.sum())
        .expand(
            shape.local_num_tokens,
            -1,
        )
        .clone()
    )
    routed_slots = shape.local_num_tokens * shape.top_k
    if routed_slots % shape.num_experts:
        raise ValueError(
            "balanced route requires routed slots divisible by num_experts, "
            f"got routed_slots={routed_slots}, num_experts={shape.num_experts}."
        )
    tokens_per_expert = torch.full(
        (shape.num_experts,),
        routed_slots // shape.num_experts,
        dtype=torch.int32,
        device=DEVICE,
    )
    return topk_ids.to(torch.int32), topk_weights, tokens_per_expert


def _make_fixed_route(
    shape: MoeShape,
    pattern: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one deterministic route and the Router-owned histogram."""
    if pattern == "balanced":
        return make_balanced_route(shape)
    positions = torch.arange(
        shape.local_num_tokens * shape.top_k,
        dtype=torch.int64,
        device=DEVICE,
    ).view(shape.local_num_tokens, shape.top_k)
    if pattern == "zero_token_experts":
        topk_ids = positions.remainder(shape.ep_size) * shape.local_experts
    elif pattern == "skew":
        token_rows = torch.arange(
            shape.local_num_tokens,
            dtype=torch.int64,
            device=DEVICE,
        )
        second_expert = torch.where(
            token_rows.remainder(8) == 0,
            torch.full_like(token_rows, shape.local_experts),
            torch.ones_like(token_rows),
        )
        topk_ids = torch.stack((torch.zeros_like(token_rows), second_expert), dim=1)
    elif pattern == "single_destination":
        topk_ids = positions.remainder(shape.local_experts)
    else:
        raise ValueError(f"unsupported route pattern {pattern!r}.")
    topk_weights = torch.full(
        (shape.local_num_tokens, shape.top_k),
        1.0 / shape.top_k,
        dtype=torch.float32,
        device=DEVICE,
    )
    tokens_per_expert = torch.bincount(
        topk_ids.reshape(-1),
        minlength=shape.num_experts,
    ).to(torch.int32)
    return topk_ids.to(torch.int32), topk_weights, tokens_per_expert


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local value of a plain tensor or DTensor."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def new_common_moe(shape: MoeShape) -> GroupedExperts:
    """Build common grouped experts and apply standard expert parallelism."""
    torch.manual_seed(2026)
    torch.npu.manual_seed(2026)
    common_moe = GroupedExperts(
        dim=shape.hidden_size,
        hidden_dim=shape.intermediate_size,
        num_experts=shape.num_experts,
        use_grouped_mm=True,
    )
    common_moe = _move_to_test_device(common_moe)
    ep_mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(shape.ep_size,),
        mesh_dim_names=("ep",),
    )
    ExpertParallel().apply(common_moe, ep_mesh)
    return common_moe


def _copy_common_weights_to_mega(
    common_moe: GroupedExperts,
    mega: MegaMoeExperts,
) -> None:
    """Copy the local common-MoE expert shard into MegaMoe's packed layout."""
    gate_weight = _local_tensor(common_moe.w1)
    up_weight = _local_tensor(common_moe.w3)
    down_weight = _local_tensor(common_moe.w2)
    with torch.no_grad():
        mega.gate_up_weight.copy_(
            torch.cat(
                (gate_weight.transpose(1, 2), up_weight.transpose(1, 2)),
                dim=-1,
            )
        )
        mega.down_weight.copy_(down_weight.transpose(1, 2))


def new_layers(
    shape: MoeShape,
    *,
    expert_capacity_factor: float | None = None,
) -> tuple[MegaMoeExperts, torch.nn.Module]:
    """Construct parameter-aligned MegaMoe and common expert layers."""
    common_moe = new_common_moe(shape)
    mega = MegaMoeExperts(
        local_num_tokens=shape.local_num_tokens,
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        num_experts=shape.num_experts,
        top_k=shape.top_k,
        expert_capacity_factor=expert_capacity_factor,
        ep_size=shape.ep_size,
        ep_group=dist.group.WORLD,
    ).to(device=DEVICE, dtype=torch.bfloat16)
    _copy_common_weights_to_mega(common_moe, mega)
    return mega, common_moe


def make_data(shape: MoeShape) -> tuple[torch.Tensor, torch.Tensor]:
    """Create rank-specific input and upstream gradient tensors."""
    torch.manual_seed(30_000 + RANK)
    torch.npu.manual_seed(30_000 + RANK)
    hidden_states = torch.randn(
        shape.local_num_tokens,
        shape.hidden_size,
        dtype=torch.float32,
        device=DEVICE,
    ).to(torch.bfloat16)
    grad_output = torch.randn(
        shape.local_num_tokens,
        shape.hidden_size,
        dtype=torch.float32,
        device=DEVICE,
    ).to(torch.bfloat16)
    return hidden_states, grad_output


def forward_layer(
    layer: MegaMoeExperts | GroupedExperts,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    tokens_per_expert: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run either backend with the same route and weight expert outputs once."""
    if isinstance(layer, MegaMoeExperts):
        return layer(
            hidden_states, topk_ids, topk_weights,
            tokens_per_expert=tokens_per_expert,
        )
    if tokens_per_expert is None:
        tokens_per_expert = torch.bincount(
            topk_ids.reshape(-1).to(torch.int64), minlength=layer.num_experts,
        )
    routed_input, mapping = torch_npu.npu_moe_token_permute(
        hidden_states.reshape(-1, hidden_states.shape[-1]), topk_ids,
    )
    expert_output = layer(routed_input, tokens_per_expert)
    output = torch_npu.npu_moe_token_unpermute(
        expert_output, mapping, probs=topk_weights.float().contiguous(),
    )
    return output.reshape(hidden_states.shape)


def run_layer(
    layer: MegaMoeExperts | torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    tokens_per_expert: torch.Tensor | None,
    grad_output: torch.Tensor,
) -> _RunResult:
    """Run one layer F+B plus SGD and detach all compared tensors."""
    layer.zero_grad(set_to_none=True)
    hidden = hidden_states.detach().clone().requires_grad_(True)
    route_weights = topk_weights.detach().clone().requires_grad_(True)
    output = forward_layer(
        layer,
        hidden,
        topk_ids,
        route_weights,
        tokens_per_expert=tokens_per_expert,
    )
    output.backward(grad_output)
    optimizer = torch.optim.SGD(layer.parameters(), lr=1e-3)
    optimizer.step()
    torch.npu.synchronize(DEVICE)
    gate_up_weight_grad, down_weight_grad = expert_weight_gradients(layer)
    gate_up_weight, down_weight = expert_weights(layer)
    gradients = {
        "input": hidden.grad,
        "route weight": route_weights.grad,
        "gate/up weight": gate_up_weight_grad,
        "down weight": down_weight_grad,
    }
    missing = [name for name, gradient in gradients.items() if gradient is None]
    if missing:
        raise AssertionError(f"rank={RANK}: missing gradients {missing}.")
    return _RunResult(
        output=output.detach().clone(),
        input_grad=hidden.grad.detach().clone(),
        route_weight_grad=route_weights.grad.detach().clone(),
        gate_up_weight_grad=gate_up_weight_grad.detach().clone(),
        down_weight_grad=down_weight_grad.detach().clone(),
        gate_up_weight=gate_up_weight.detach().clone(),
        down_weight=down_weight.detach().clone(),
    )


def expert_weight_gradients(
    layer: MegaMoeExperts | torch.nn.Module,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return expert gradients in MegaMoe's packed local layout."""
    if isinstance(layer, MegaMoeExperts):
        return layer.gate_up_weight.grad, layer.down_weight.grad
    if not isinstance(layer, GroupedExperts):
        raise TypeError(f"unsupported expert layer type {type(layer).__name__}.")
    gate_grad = layer.w1.grad
    up_grad = layer.w3.grad
    down_grad = layer.w2.grad
    if gate_grad is None or up_grad is None or down_grad is None:
        return None, None
    return (
        torch.cat(
            (
                _local_tensor(gate_grad).transpose(1, 2),
                _local_tensor(up_grad).transpose(1, 2),
            ),
            dim=-1,
        ),
        _local_tensor(down_grad).transpose(1, 2),
    )


def expert_weights(
    layer: MegaMoeExperts | torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return expert weights in MegaMoe's packed local layout."""
    if isinstance(layer, MegaMoeExperts):
        return layer.gate_up_weight, layer.down_weight
    if not isinstance(layer, GroupedExperts):
        raise TypeError(f"unsupported expert layer type {type(layer).__name__}.")
    gate_weight = _local_tensor(layer.w1)
    up_weight = _local_tensor(layer.w3)
    down_weight = _local_tensor(layer.w2)
    return (
        torch.cat(
            (gate_weight.transpose(1, 2), up_weight.transpose(1, 2)),
            dim=-1,
        ),
        down_weight.transpose(1, 2),
    )


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    """Require every compared result to be finite."""
    finite = bool(torch.isfinite(tensor).all().item())
    assert finite, (
        f"rank={RANK}: {name} contains non-finite values, shape={tuple(tensor.shape)}."
    )


def assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    """Compare BF16 common-MoE and MegaMoe results with diagnostic maxima."""
    if torch.allclose(actual, expected, rtol=_RTOL, atol=_ATOL):
        return
    difference = (actual.float() - expected.float()).abs()
    maximum_absolute = float(difference.max().item())
    maximum_relative = float(
        (difference / expected.float().abs().clamp_min(1e-12)).max().item()
    )
    raise AssertionError(
        f"rank={RANK}: {name} differs from common MoE at rtol={_RTOL}, "
        f"atol={_ATOL}, max_abs={maximum_absolute}, max_rel={maximum_relative}."
    )


def _assert_active_expert_gradients(
    shape: MoeShape,
    local_counts: torch.Tensor,
    result: _RunResult,
    backend: str,
) -> None:
    """Require useful dW for every expert in the balanced route."""
    global_counts = local_counts.clone()
    dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
    expert_start = RANK * shape.local_experts
    for local_idx in range(shape.local_experts):
        global_idx = expert_start + local_idx
        count = int(global_counts[global_idx].item())
        assert count > 0, (
            f"rank={RANK}: balanced route did not activate expert {global_idx}."
        )
        for name, gradient in (
            ("gate_up_weight_grad", result.gate_up_weight_grad[local_idx]),
            ("down_weight_grad", result.down_weight_grad[local_idx]),
        ):
            nonzero = int(torch.count_nonzero(gradient).item())
            assert nonzero > 0, (
                f"rank={RANK}: {backend} active expert {global_idx} {name} is all zero, "
                f"route_count={count}."
            )


def assert_results_close(common_result: _RunResult, mega_result: _RunResult) -> None:
    """Compare every forward and backward result from two expert backends."""
    for result_field in fields(_RunResult):
        field_name = result_field.name
        common_tensor = getattr(common_result, field_name)
        mega_tensor = getattr(mega_result, field_name)
        assert mega_tensor.shape == common_tensor.shape, (
            f"rank={RANK}: {field_name} shape mismatch: "
            f"mega={tuple(mega_tensor.shape)}, common={tuple(common_tensor.shape)}."
        )
        assert_finite(f"common {field_name}", common_tensor)
        assert_finite(f"MegaMoe {field_name}", mega_tensor)
        assert_close(field_name, mega_tensor, common_tensor)


def _run_precision_case() -> None:
    """Compare routed-expert F+B and optimizer updates with common MoE."""
    shape = MoeShape()
    hidden_states, grad_output = make_data(shape)
    mega, common_moe = new_layers(shape)
    try:
        for pattern in ("balanced", "zero_token_experts", "skew", "single_destination"):
            topk_ids, topk_weights, tokens_per_expert = _make_fixed_route(
                shape,
                pattern,
            )
            common_result = run_layer(
                common_moe,
                hidden_states,
                topk_ids,
                topk_weights,
                tokens_per_expert,
                grad_output,
            )
            mega_result = run_layer(
                mega,
                hidden_states,
                topk_ids,
                topk_weights,
                tokens_per_expert,
                grad_output,
            )
            assert_results_close(common_result, mega_result)
            if pattern == "balanced":
                _assert_active_expert_gradients(
                    shape,
                    tokens_per_expert,
                    common_result,
                    "common",
                )
                _assert_active_expert_gradients(
                    shape,
                    tokens_per_expert,
                    mega_result,
                    "MegaMoe",
                )
    finally:
        mega.close()

    bounded, bounded_common = new_layers(shape, expert_capacity_factor=1.5)
    topk_ids, topk_weights, tokens_per_expert = make_balanced_route(shape)
    try:
        bounded_common_result = run_layer(
            bounded_common,
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            grad_output,
        )
        bounded_result = run_layer(
            bounded,
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            grad_output,
        )
        assert_results_close(bounded_common_result, bounded_result)
        overflow_ids, overflow_weights, overflow_counts = _make_fixed_route(
            shape,
            "single_destination",
        )
        try:
            bounded(
                hidden_states,
                overflow_ids,
                overflow_weights,
                tokens_per_expert=overflow_counts,
            )
        except RuntimeError as error:
            assert "capacity overflow" in str(error), (
                f"rank={RANK}: bounded overflow reported the wrong error: {error}."
            )
        else:
            raise AssertionError(
                f"rank={RANK}: bounded MegaMoe accepted an overflowing route."
            )
    finally:
        bounded.close()


def test_mega_moe_level0_balanced() -> None:
    """Validate SHMEM lifecycle, then compare routed-expert training results."""
    memory = acquire_symmetric_memory(dist.group.WORLD)
    regular = memory.empty((256,), torch.uint8)
    aligned = memory.aligned_empty((256,), torch.uint8, 512)
    assert aligned.data_ptr() % 512 == 0, (
        f"rank={RANK}: aligned SHMEM pointer is not 512-byte aligned."
    )
    regular.fill_(RANK)
    aligned.fill_(RANK)
    memory.barrier()
    memory.free(regular)
    memory.free(aligned)
    assert regular.untyped_storage().nbytes() == 0, (
        f"rank={RANK}: regular SHMEM tensor remained accessible after free."
    )
    assert aligned.untyped_storage().nbytes() == 0, (
        f"rank={RANK}: aligned SHMEM tensor remained accessible after free."
    )
    memory.close()
    memory.close()
    _run_precision_case()


def performance_shape() -> MoeShape:
    """Return the fixed representative four-card performance shape."""
    return MoeShape(
        local_num_tokens=4096,
        hidden_size=5120,
        intermediate_size=1792,
        num_experts=32,
        top_k=8,
        ep_size=4,
    )


def _new_performance_layer(
    shape: MoeShape,
    backend: str,
) -> MegaMoeExperts | torch.nn.Module:
    """Construct one seeded backend without retaining its comparison peer."""
    if backend == "common":
        return new_common_moe(shape)
    if backend != "mega_moe":
        raise ValueError(
            f"HP_MEGA_MOE_PERF_BACKEND must be common or mega_moe, got {backend!r}."
        )
    return MegaMoeExperts(
        local_num_tokens=shape.local_num_tokens,
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        num_experts=shape.num_experts,
        top_k=shape.top_k,
        expert_capacity_factor=1.5,
        ep_size=shape.ep_size,
        ep_group=dist.group.WORLD,
    ).to(
        device=DEVICE,
        dtype=torch.bfloat16,
    )


def rank_max_latency(elapsed_ms: float) -> float:
    """Return one F+B latency reduced by maximum across all ranks."""
    latency = torch.tensor(elapsed_ms, dtype=torch.float32, device=DEVICE)
    dist.all_reduce(latency, op=dist.ReduceOp.MAX)
    return float(latency.cpu().item())


def _timed_fwd_bwd(
    layer: MegaMoeExperts | torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    validate_gradients: bool = False,
) -> float:
    """Time one synchronized layer forward and backward operation."""
    layer.zero_grad(set_to_none=True)
    hidden = hidden_states.detach().requires_grad_(True)
    route_weights = topk_weights.detach().requires_grad_(True)
    dist.barrier()
    torch.npu.synchronize(DEVICE)
    start = time.perf_counter()
    output = forward_layer(
        layer,
        hidden,
        topk_ids,
        route_weights,
        tokens_per_expert=tokens_per_expert,
    )
    output.backward(grad_output)
    torch.npu.synchronize(DEVICE)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    gate_up_weight_grad, down_weight_grad = expert_weight_gradients(layer)
    gradients = (
        ("input", hidden.grad),
        ("route weight", route_weights.grad),
        ("gate/up weight", gate_up_weight_grad),
        ("down weight", down_weight_grad),
    )
    missing = [name for name, gradient in gradients if gradient is None]
    if missing:
        raise AssertionError(
            f"rank={RANK}: performance F+B missed gradients {missing}."
        )
    if validate_gradients:
        finite = torch.ones((), dtype=torch.int32, device=DEVICE)
        for _, gradient in gradients:
            finite = torch.minimum(
                finite,
                torch.isfinite(gradient).all().to(torch.int32),
            )
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite.cpu().item()):
            raise AssertionError(
                f"rank={RANK}: performance F+B produced non-finite gradients."
            )
    return rank_max_latency(elapsed_ms)


def _performance_result(
    backend: str,
    shape: MoeShape,
    warmup_steps: int,
    latencies: list[float],
) -> dict:
    """Build one routed-expert performance result."""
    return {
        "backend": backend,
        "topology": {
            "world_size": dist.get_world_size(),
            "ep": shape.ep_size,
        },
        "shape": {
            "local_num_tokens": shape.local_num_tokens,
            "hidden_size": shape.hidden_size,
            "intermediate_size": shape.intermediate_size,
            "num_experts": shape.num_experts,
            "top_k": shape.top_k,
            "dtype": "bfloat16",
        },
        "warmup_steps": warmup_steps,
        "measured_steps": len(latencies),
        "validation": {"gradients_present": True, "gradients_finite": True},
        "rank_max_fwd_bwd_ms": {
            "median": statistics.median(latencies),
            "minimum": min(latencies),
            "maximum": max(latencies),
            "samples": latencies,
        },
        "device_healthy": True,
    }


def _benchmark_performance_backend(
    backend: str,
    shape: MoeShape,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    grad_output: torch.Tensor,
    warmup_steps: int,
    measured_steps: int,
) -> dict:
    """Warm up and measure one routed-expert backend."""
    layer = _new_performance_layer(shape, backend)
    try:
        _timed_fwd_bwd(
            layer,
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            grad_output,
            validate_gradients=True,
        )
        latencies = []
        for step in range(warmup_steps + measured_steps):
            latency = _timed_fwd_bwd(
                layer,
                hidden_states,
                topk_ids,
                topk_weights,
                tokens_per_expert,
                grad_output,
            )
            if step >= warmup_steps:
                latencies.append(latency)
        return _performance_result(backend, shape, warmup_steps, latencies)
    finally:
        if isinstance(layer, MegaMoeExperts):
            layer.close()


def test_mega_moe_fwd_bwd_performance() -> None:
    """Measure common then MegaMoe F+B in one distributed A/B launch."""
    warmup_steps = int(os.getenv("HP_MEGA_MOE_PERF_WARMUP", "3"))
    measured_steps = int(os.getenv("HP_MEGA_MOE_PERF_MEASURED", "5"))
    if warmup_steps < 0 or measured_steps <= 0:
        raise ValueError(
            f"invalid performance steps: warmup={warmup_steps}, measured={measured_steps}."
        )

    shape = performance_shape()
    topk_ids, topk_weights, tokens_per_expert = make_balanced_route(shape)
    hidden_states, grad_output = make_data(shape)
    results = {}
    for backend in ("common", "mega_moe"):
        results[backend] = _benchmark_performance_backend(
            backend,
            shape,
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            grad_output,
            warmup_steps,
            measured_steps,
        )
        gc.collect()
        torch.npu.empty_cache()
        dist.barrier()

    common_median = results["common"]["rank_max_fwd_bwd_ms"]["median"]
    mega_median = results["mega_moe"]["rank_max_fwd_bwd_ms"]["median"]
    ratio = mega_median / common_median
    maximum_ratio = float(os.getenv("HP_MEGA_MOE_MAX_PERF_RATIO", "1.10"))
    if ratio > maximum_ratio:
        raise AssertionError(
            "MegaMoe routed-expert performance regressed: "
            f"common_ms={common_median}, mega_ms={mega_median}, "
            f"ratio={ratio:.6f}, maximum_ratio={maximum_ratio:.6f}."
        )
    if RANK == 0:
        result_path = os.getenv("HP_MEGA_MOE_PERF_RESULT")
        if not result_path:
            raise ValueError("HP_MEGA_MOE_PERF_RESULT must name the output JSON path.")
        result = {
            "order": ["common", "mega_moe"],
            "results": results,
            "mega_over_common": ratio,
            "maximum_allowed_ratio": maximum_ratio,
        }
        path = Path(result_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"PERF_RESULT_JSON={path}")
        print(json.dumps(result, sort_keys=True))
    dist.barrier()
