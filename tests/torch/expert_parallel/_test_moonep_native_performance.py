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
"""Native grouped-MoE production-policy paired forward/backward timings.

This worker never mocks placement or profitability and collects no profile.
Every result retains all rank-critical samples and the paired ABBA order.
"""
from __future__ import annotations

import argparse
import copy
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import weakref
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
import torch_npu

from hyper_parallel.core.expert_parallel.moonep import native, precision
from hyper_parallel.distributed.expert_parallel import experts as execution
from tests.torch.expert_parallel._test_moonep_native import NativeExperts


def _source_identity() -> dict[str, str]:
    """Hash the exact production files that determine this paired experiment."""
    root = Path(__file__).resolve().parents[3]
    files = ('hyper_parallel/core/expert_parallel/moonep/native.py',
             'hyper_parallel/core/expert_parallel/moonep/precision.py',
             'hyper_parallel/core/expert_parallel/moonep/placement.py',
             'hyper_parallel/distributed/expert_parallel/experts.py',
             'hyper_parallel/components/functional/npu_grouped_swiglu.py')
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}


def _measure_case(tokens: int, hidden: int, intermediate: int, experts: int, topk: int, hot: bool,
                  warmup: int, samples: int) -> dict:
    """Measure rank-critical complete forward/backward with a fixed ABBA boundary."""
    rank, ranks = dist.get_rank(), dist.get_world_size()
    baseline = nn.Module()
    baseline.experts = NativeExperts(rank, ranks, hidden, intermediate, torch.bfloat16, experts // ranks)
    candidate = copy.deepcopy(baseline)
    execution.bind_local_expert_forward(baseline, ranks, use_grouped_gemm=True)
    execution.bind_local_expert_forward(candidate, ranks, use_grouped_gemm=True)
    generator = torch.Generator().manual_seed(125 + rank)
    source = (torch.randn(1, tokens, hidden, generator=generator) * 0.1).to('npu', torch.bfloat16)
    first = torch.zeros(tokens, dtype=torch.int64) if hot else torch.arange(tokens) % experts
    ids = torch.stack(tuple((first + slot) % experts for slot in range(topk)), dim=-1).to('npu')
    probabilities = torch.full((tokens, topk), 1.0 / topk, device='npu', dtype=torch.bfloat16)

    def iteration(enabled: bool) -> None:
        """Run one complete production forward and backward with the chosen backend flag."""
        module = candidate if enabled else baseline
        module.zero_grad(set_to_none=True)
        states = source.detach().requires_grad_()
        weights = probabilities.detach().requires_grad_()
        output = execution.ep_routed_forward(
            module, states, router_fn=lambda *_: (ids, weights),
            ep_group=dist.group.WORLD, enable_moonep=enabled)
        output.sum().backward()

    # Record the production policy's actual decision outside the timing boundary.
    local_counts = execution._expert_token_counts(ids, experts)
    selection = native.prepare_native_placement(
        source.reshape(-1, hidden).repeat_interleave(topk, dim=0), ids.reshape(-1, 1), local_counts,
        ep_group=dist.group.WORLD, hidden_size=hidden, intermediate_size=intermediate,
        cube_cores=torch.npu.get_device_limit(source.device.index)['cube_core_num'],
        weight_bytes=sum(parameter.numel() * parameter.element_size() for parameter in candidate.parameters()))
    borrowed = selection.invocation is not None
    ownership = {'borrow_calls': 0, 'weight_send_bytes': 0, 'weight_receive_bytes': 0,
                 'gradient_return_send_bytes': 0, 'gradient_return_receive_bytes': 0,
                 'owned_bf16_parameter_view_bytes': 0, 'fp32_scratch_peak_bytes': 0}
    target_refs = []
    original_borrow = native.borrow_parameters
    gradient_value = precision.GradientTarget.gradient

    def observe_borrow(parameters: tuple[torch.Tensor, ...], copies: tuple[tuple[int, ...], ...],
                       group: dist.ProcessGroup, *options: tuple[bool, ...]) -> tuple[
                           tuple[torch.Tensor, ...], tuple[precision.GradientTarget | None, ...]]:
        """Count actual transport and ownership in one untimed production iteration."""
        ownership['borrow_calls'] += 1
        local_experts = parameters[0].shape[0]
        sent = sum(expert >= 0 and expert // local_experts == rank for row in copies for expert in row)
        received = sum(expert >= 0 for expert in copies[rank])
        view_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
        expert_bytes = view_bytes // local_experts
        expert_fp32_bytes = sum(parameter.numel() * 4 for parameter in parameters) // local_experts
        ownership['weight_send_bytes'] += sent * expert_bytes
        ownership['weight_receive_bytes'] += received * expert_bytes
        ownership['gradient_return_send_bytes'] += received * expert_fp32_bytes
        ownership['gradient_return_receive_bytes'] += sent * expert_fp32_bytes
        ownership['owned_bf16_parameter_view_bytes'] += 2 * view_bytes
        views, targets = original_borrow(parameters, copies, group, *options)
        target_refs.extend(weakref.ref(target) for target in targets if target is not None)
        return views, targets

    def observe_gradient(target: precision.GradientTarget, fallback: torch.Tensor) -> torch.Tensor:
        """Count the peak before this invocation consumes its complete set of accumulators."""
        live = sum(reference().buffer.numel() * reference().buffer.element_size()
                   for reference in target_refs if reference() is not None and reference().buffer is not None)
        ownership['fp32_scratch_peak_bytes'] = max(ownership['fp32_scratch_peak_bytes'], live)
        return gradient_value(target, fallback)

    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    allocated_before = torch.npu.memory_allocated()
    with patch.object(native, 'borrow_parameters', side_effect=observe_borrow), \
            patch.object(precision.GradientTarget, 'gradient', observe_gradient):
        iteration(True)
    torch.npu.synchronize()
    ownership['allocator_peak_bytes'] = torch.npu.max_memory_allocated()
    ownership['allocator_peak_delta_bytes'] = ownership['allocator_peak_bytes'] - allocated_before
    ownership['fp32_scratch_live_after_backward_bytes'] = sum(
        reference().buffer.numel() * reference().buffer.element_size()
        for reference in target_refs if reference() is not None and reference().buffer is not None)
    assert ownership['fp32_scratch_live_after_backward_bytes'] == 0, ownership
    assert bool(ownership['borrow_calls']) == borrowed, (ownership, borrowed)
    peers = [None] * ranks
    dist.all_gather_object(peers, ownership)
    blocks = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for round_index in range(4):
            order = (False, True, True, False) if round_index % 2 == 0 else (True, False, False, True)
            for enabled in order:
                for _ in range(warmup):
                    iteration(enabled)
                torch.npu.synchronize()
                times = []
                for _ in range(samples):
                    torch.npu.synchronize()
                    start = time.perf_counter_ns()
                    iteration(enabled)
                    torch.npu.synchronize()
                    times.append((time.perf_counter_ns() - start) / 1e6)
                # Collect peer timings only after this block's measured work completes.
                local = torch.tensor(times, device='npu', dtype=torch.float64)
                gathered = [torch.empty_like(local) for _ in range(ranks)]
                dist.all_gather(gathered, local)
                critical = torch.stack(gathered).cpu().max(dim=0).values.tolist()
                blocks.append({'round': round_index, 'moonep': enabled, 'samples_ms': critical,
                               'median_ms': statistics.median(critical)})
    finally:
        if gc_was_enabled:
            gc.enable()
    baseline_times = [value for block in blocks if not block['moonep'] for value in block['samples_ms']]
    candidate_times = [value for block in blocks if block['moonep'] for value in block['samples_ms']]
    baseline_ms = statistics.median(baseline_times)
    candidate_ms = statistics.median(candidate_times)
    return {'tokens': tokens, 'hidden': hidden, 'intermediate': intermediate, 'hotspot': hot,
            'experts': experts, 'topk': topk, 'ep': ranks,
            'production_policy_borrowed': borrowed, 'baseline_ms': baseline_ms, 'moonep_ms': candidate_ms,
            'ratio': candidate_ms / baseline_ms,
            'acceptance_ratio_limit': 1.03 if not hot or tokens <= 128 else 1.0,
            'accepted': candidate_ms <= baseline_ms * (1.03 if not hot or tokens <= 128 else 1.0),
            'ownership_by_rank': peers,
            'blocks': blocks, 'warmup': warmup, 'samples_per_block': samples}


def main() -> None:
    """Persist every measured specification immediately, including failed acceptance."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--samples', type=int, default=10)
    args = parser.parse_args()
    torch_npu.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(seconds=300))
    result = {'source_sha256': _source_identity(), 'timing': 'wall complete forward+backward, synchronize boundaries',
              'production_policy': True, 'profile': False, 'results': []}
    for size in [(128, 512, 128, 8, 2), (4096, 5120, 1792, 32, 8)]:
        for hot in (False, True):
            result['results'].append(_measure_case(*size, hot, args.warmup, args.samples))
            if dist.get_rank() == 0:
                output = Path(args.output)
                temporary = output.with_suffix('.tmp')
                temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
                temporary.replace(output)
    result['all_accepted'] = all(row['accepted'] for row in result['results'])
    if dist.get_rank() == 0:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
