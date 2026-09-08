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
"""Default-capacity and Router-inclusive representative MegaMoe acceptance."""

from __future__ import annotations

import gc
import os
import statistics
import time
from dataclasses import asdict

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import MegaMoeExperts
from tests.torch.expert_parallel import _test_mega_moe as baseline
from tests.torch.expert_parallel._mega_moe_utils import (
    assert_stable_memory,
    collect_memory,
    memory_sample,
    start_shmem_lifetime,
    write_evidence,
)


def _router_weight(shape: baseline.MoeShape) -> torch.nn.Parameter:
    """Use identical trainable Router weights in every A/B repetition."""
    generator = torch.Generator().manual_seed(706)
    weight = torch.randn(shape.num_experts, shape.hidden_size, generator=generator)
    return torch.nn.Parameter((weight / shape.hidden_size**0.5).to(baseline.DEVICE))


def _route(hidden: torch.Tensor, weight: torch.Tensor, top_k: int) -> tuple:
    """Compute Router projection, softmax, Top-K and selected-weight normalization."""
    probabilities = (hidden.float() @ weight.T).softmax(dim=-1)
    values, indices = probabilities.topk(top_k, dim=-1)
    return indices.to(torch.int32), values / values.sum(dim=-1, keepdim=True)


def _run_router_layer(layer: torch.nn.Module, shape: baseline.MoeShape) -> list[torch.Tensor]:
    """Validate a connected Router/expert graph, including Router dW and SGD."""
    hidden, upstream = baseline.make_data(shape)
    hidden.requires_grad_(True)
    router = _router_weight(shape)
    ids, weights = _route(hidden, router, shape.top_k)
    weights.retain_grad()
    output = baseline.forward_layer(layer, hidden, ids, weights)
    output.backward(upstream)
    optimizer = torch.optim.SGD([*layer.parameters(), router], lr=1e-3)
    optimizer.step()
    torch.npu.synchronize()
    compared = [output, hidden.grad, weights.grad, router.grad, router]
    compared.extend(baseline.expert_weight_gradients(layer))
    compared.extend(baseline.expert_weights(layer))
    for index, tensor in enumerate(compared):
        assert tensor is not None, f"rank={baseline.RANK}: missing Router graph tensor {index}."
        baseline.assert_finite(f"Router graph tensor {index}", tensor)
    return [tensor.detach().clone() for tensor in compared]


def _validate_pair(shape: baseline.MoeShape, factor: float | None, scope: str) -> None:
    """Compare real output, every gradient and one update at the measured shape."""
    start_shmem_lifetime()
    os.environ.pop("SYMMETRIC_MEMORY_HEAP_SIZE", None)
    mega, common = baseline.new_layers(shape, expert_capacity_factor=factor)
    try:
        if scope == "router_experts":
            reference = _run_router_layer(common, shape)
            actual = _run_router_layer(mega, shape)
            for index, (expected, observed) in enumerate(zip(reference, actual)):
                baseline.assert_close(f"Router graph tensor {index}", observed, expected)
        else:
            hidden, upstream = baseline.make_data(shape)
            ids, weights, _ = baseline.make_balanced_route(shape)
            reference = baseline.run_layer(common, hidden, ids, weights, None, upstream)
            actual = baseline.run_layer(mega, hidden, ids, weights, None, upstream)
            baseline.assert_results_close(reference, actual)
    finally:
        mega.close()


def _measurement_layer(shape: baseline.MoeShape, backend: str, factor: float | None) -> torch.nn.Module:
    """Retain only one parameter-aligned backend while measuring memory."""
    os.environ.pop("SYMMETRIC_MEMORY_HEAP_SIZE", None)
    if backend == "common":
        return baseline.new_common_moe(shape)
    start_shmem_lifetime()
    mega, common = baseline.new_layers(shape, expert_capacity_factor=factor)
    del common
    gc.collect()
    torch.npu.empty_cache()
    return mega


def _iteration(
    layer: torch.nn.Module,
    router: torch.nn.Parameter,
    shape: baseline.MoeShape,
    inputs: tuple,
    scope: str,
) -> tuple[float, float, int, int]:
    """Time host-to-completion F and F+B, including histogram construction."""
    source, upstream, ids, weights = inputs
    layer.zero_grad(set_to_none=True)
    router.grad = None
    hidden = source.detach().requires_grad_(True)
    selected_weights = weights.detach().requires_grad_(True)
    dist.barrier()
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    start = time.perf_counter()
    if scope == "router_experts":
        ids, selected_weights = _route(hidden, router, shape.top_k)
    output = baseline.forward_layer(layer, hidden, ids, selected_weights)
    torch.npu.synchronize()
    forward_ms = (time.perf_counter() - start) * 1000.0
    output.backward(upstream)
    torch.npu.synchronize()
    forward_backward_ms = (time.perf_counter() - start) * 1000.0
    peak_allocated = torch.npu.max_memory_allocated()
    peak_reserved = torch.npu.max_memory_reserved()
    gradients = [hidden.grad, *baseline.expert_weight_gradients(layer)]
    gradients.append(router.grad if scope == "router_experts" else selected_weights.grad)
    for index, gradient in enumerate(gradients):
        assert gradient is not None, f"rank={baseline.RANK}: missing measured gradient {index}."
        baseline.assert_finite(f"measured gradient {index}", gradient)
    return (
        baseline.rank_max_latency(forward_ms), baseline.rank_max_latency(forward_backward_ms),
        peak_allocated, peak_reserved,
    )


def _measure_backend(
    shape: baseline.MoeShape, backend: str, factor: float | None, scope: str,
) -> dict:
    """Run fixed warmup and stable iterations with an isolated live backend."""
    layer = _measurement_layer(shape, backend, factor)
    router = _router_weight(shape)
    hidden, upstream = baseline.make_data(shape)
    ids, weights, _ = baseline.make_balanced_route(shape)
    inputs = (hidden, upstream, ids, weights)
    latencies, samples = [], []
    try:
        for step in range(8):
            latency = _iteration(layer, router, shape, inputs, scope)
            layer.zero_grad(set_to_none=True)
            router.grad = None
            if step >= 3:
                latencies.append(latency)
                samples.append(collect_memory())
        assert_stable_memory(samples)
        return {
            "backend": backend,
            "effective_capacity_factor": factor if backend == "mega_moe" else None,
            "peak_allocated_bytes": max(latency[2] for latency in latencies),
            "peak_reserved_bytes": max(latency[3] for latency in latencies),
            "stable_memory": samples,
            "rank_max_forward_ms": [latency[0] for latency in latencies],
            "rank_max_fwd_bwd_ms": [latency[1] for latency in latencies],
            "gradients_present_and_finite": True,
        }
    finally:
        if isinstance(layer, MegaMoeExperts):
            layer.close()


def test_mega_moe_default_capacity_acceptance() -> None:
    """Record same-shape default/bounded A/B and Router-inclusive F/F+B costs."""
    shape = baseline.performance_shape()
    comparisons = []
    for factor in (None, 1.5):
        for scope in ("experts", "router_experts"):
            _validate_pair(shape, factor, scope)
            gc.collect()
            torch.npu.empty_cache()
            repetitions = []
            for backend in ("common", "mega_moe", "mega_moe", "common"):
                if baseline.RANK == 0:
                    print(f"Level1 factor={factor} scope={scope} backend={backend}", flush=True)
                repetitions.append(_measure_backend(shape, backend, factor, scope))
                gc.collect()
                torch.npu.empty_cache()
                dist.barrier()
            medians = {
                backend: statistics.median(
                    value for result in repetitions if result["backend"] == backend
                    for value in result["rank_max_fwd_bwd_ms"]
                )
                for backend in ("common", "mega_moe")
            }
            comparisons.append({
                "comparison_capacity_factor": factor, "timing_scope": scope,
                "counts_supplied": False, "histogram_in_timing": True,
                "router_in_timing": scope == "router_experts", "warmup_steps": 3,
                "measured_steps": 5, "order": [result["backend"] for result in repetitions],
                "output_gradient_update_validation": True, "repetitions": repetitions,
                "mega_over_common_fwd_bwd": medians["mega_moe"] / medians["common"],
                "after_close_memory": memory_sample(),
            })
    write_evidence({
        "shape": asdict(shape), "dtype": "bfloat16", "ep_is_whole_world": True,
        "router_projection_dtype": "float32",
        "comparisons": comparisons,
        "timing_note": "Host wall time with a completion sync after F and B; setup and validation excluded.",
    })
