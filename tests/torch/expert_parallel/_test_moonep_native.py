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
"""Representative native MoonEP device contracts and actual V4.1 execution.

Arithmetic, seeds and tolerances are shared with the original workers. Only
profitability is forced for small correctness tensors; the separate performance
worker always uses the unmodified production policy.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import nullcontext
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint
import torch_npu
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts

from hyper_parallel import fully_shard, init_device_mesh
from hyper_parallel.components.modules.grouped_experts import GroupedExperts
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy, MixedPrecisionPolicy
from hyper_parallel.core.expert_parallel.moonep import native, precision
from hyper_parallel.distributed.expert_parallel import experts as execution
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41TopKRouter
from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import (
    deepseek_v41_ep_compute_fn,
)
from tests.torch.expert_parallel.test_moonep_native import NATIVE_CASES


@pytest.fixture
def _distributed_group() -> Iterator[None]:
    """Own one case's real EP process group and clean it after failures too."""
    torch_npu.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(seconds=180))
    try:
        yield
    finally:
        dist.destroy_process_group()


class NativeExperts(nn.Module):
    """Stacked model-owned SwiGLU weights without a MegaMoE dependency."""

    def __init__(self, rank: int, ranks: int, hidden: int, intermediate: int, dtype: torch.dtype,
                 local_experts: int = 2) -> None:
        """Initialize deterministic model-owned projections for this EP owner."""
        super().__init__()
        self.num_experts = local_experts * ranks
        self.hidden_size = hidden
        self.intermediate_size = intermediate
        generator = torch.Generator().manual_seed(822)
        gate = torch.randn(self.num_experts, 2 * intermediate, hidden, generator=generator) * 0.03
        down = torch.randn(self.num_experts, hidden, intermediate, generator=generator) * 0.03
        owned = slice(local_experts * rank, local_experts * (rank + 1))
        self.gate_up_proj = nn.Parameter(gate[owned].to(dtype=dtype, device='npu'))
        self.down_proj = nn.Parameter(down[owned].to(dtype=dtype, device='npu'))

def _tensor_gradient(parameter: torch.Tensor) -> torch.Tensor:
    """Read the authoritative FSDP gradient, restoring its logical DP extent."""
    gradient = parameter.grad
    if gradient is None:
        raise AssertionError('Native expert gradient is missing')
    to_local = getattr(gradient, 'to_local', None)
    return gradient if to_local is None else to_local()

def _native_oracle(inputs: torch.Tensor, ids: torch.Tensor, probabilities: torch.Tensor,
            weights: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Compute each selected expert independently using the same dtype boundaries."""
    gate_weight, down_weight = weights
    result = torch.zeros_like(inputs)
    for expert in range(gate_weight.shape[0]):
        gate, up = functional.linear(inputs, gate_weight[expert]).chunk(2, dim=-1)
        projected = functional.linear(functional.silu(gate) * up, down_weight[expert])
        coefficient = ((ids == expert) * probabilities).sum(-1, keepdim=True)
        result = result + projected * coefficient.reshape(1, -1, 1)
    return result

def _run_native_correctness(args: argparse.Namespace) -> dict:
    """Compare normal, replayed and delayed native dispatches with an independent oracle."""
    rank = dist.get_rank()
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32
    if args.fsdp:
        mesh = init_device_mesh('npu', (2, dist.get_world_size() // 2), mesh_dim_names=('dp', 'ep'))
        group = mesh['ep'].get_group()
        dp_size = 2
        dp_rank = mesh['dp'].get_local_rank()
    else:
        mesh = None
        group = dist.group.WORLD
        dp_size = 1
        dp_rank = 0
    ep_rank, ep_size = dist.get_rank(group), dist.get_world_size(group)
    module = nn.Module()
    module.experts = NativeExperts(ep_rank, ep_size, 64, 128,
                                   torch.float32 if args.master_float32 else dtype)
    execution.bind_local_expert_forward(module, ep_size, use_grouped_gemm=args.grouped_gemm)
    if args.capacity_factor is not None:
        module.experts.capacity_factor = args.capacity_factor
    gate_parts = [torch.empty_like(module.experts.gate_up_proj) for _ in range(ep_size)]
    down_parts = [torch.empty_like(module.experts.down_proj) for _ in range(ep_size)]
    dist.all_gather(gate_parts, module.experts.gate_up_proj, group=group)
    dist.all_gather(down_parts, module.experts.down_proj, group=group)
    source_gate, source_down = torch.cat(gate_parts), torch.cat(down_parts)
    if args.fsdp:
        options = {'mesh': mesh['dp'], 'mp_policy': MixedPrecisionPolicy(
            param_dtype=dtype, reduce_dtype=torch.float32 if args.master_float32 else None)}
        if args.offload:
            options['offload_policy'] = CPUOffloadPolicy(pin_memory=True)
        fully_shard(module.experts, **options)
    identities = tuple(id(parameter) for parameter in module.parameters())
    borrow_calls = []
    gradient_targets = []
    original_borrow = native.borrow_parameters

    def observe_borrow(weight: tuple[torch.Tensor, ...], copies: tuple[tuple[int, ...], ...],
                       ep_group: dist.ProcessGroup, *extra: tuple[bool, ...]) -> tuple[
                           tuple[torch.Tensor, ...], tuple[precision.GradientTarget | None, ...]]:
        """Count real weight exchanges without replacing the transfer implementation."""
        borrow_calls.append(tuple(tuple(row) for row in copies))
        result, targets = original_borrow(weight, copies, ep_group, *extra)
        assert all(target is None or target.buffer is None for target in targets)
        gradient_targets.extend(weakref.ref(target) for target in targets if target is not None)
        return result, targets

    def make_invocation(length: int, hot: bool, seed: int) -> tuple:
        """Create an independent forward graph and its stage-compatible expert oracle."""
        generator = torch.Generator().manual_seed(seed + rank)
        values = torch.randn(1, length, 64, generator=generator) * 0.1
        inputs = values.to(dtype=dtype, device='npu').requires_grad_()
        expected_inputs = inputs.detach().clone().requires_grad_()
        first = torch.zeros(length, dtype=torch.int64) if hot else torch.arange(length) % (2 * ep_size)
        ids = torch.stack((first, (first + 1) % (2 * ep_size)), dim=-1).to('npu')
        probabilities = torch.full((length, 2), 0.5, dtype=dtype, device='npu', requires_grad=True)
        expected_probabilities = probabilities.detach().clone().requires_grad_()
        gate = source_gate.to(dtype=dtype).detach().clone().requires_grad_()
        down = source_down.to(dtype=dtype).detach().clone().requires_grad_()

        def forward(states: torch.Tensor) -> torch.Tensor:
            """Keep the selected HP dispatch schedule around the real expert module."""
            if args.async_dispatch:
                pending = execution.ep_routed_dispatch(
                    module, states, router_fn=lambda *_: (ids, probabilities), ep_group=group, enable_moonep=True)
                return execution.ep_routed_experts_and_combine(module, pending, group)
            return execution.ep_routed_forward(
                module, states, router_fn=lambda *_: (ids, probabilities), ep_group=group, enable_moonep=True)

        if args.checkpoint == 'none':
            output = forward(inputs)
        else:
            output = checkpoint(forward, inputs, use_reentrant=args.checkpoint == 'reentrant')
        oracle_probabilities = expected_probabilities
        if args.capacity_factor is not None:
            keep, _ = execution.apply_capacity_limit(ids, args.capacity_factor, 2 * ep_size, probabilities.detach())
            oracle_probabilities = expected_probabilities * keep
        expected = _native_oracle(expected_inputs, ids, oracle_probabilities, (gate, down))
        return (output, inputs, probabilities), (expected, expected_inputs, expected_probabilities), (gate, down)

    rtol, atol = (0.02, 0.002) if dtype == torch.bfloat16 else (1e-4, 1e-5)
    results = []

    def pack(tensor: torch.Tensor) -> tuple[torch.device, torch.Tensor]:
        """Retain saved tensors in CPU memory for the offload compatibility case."""
        return tensor.device, tensor.cpu()

    def unpack(saved: tuple[torch.device, torch.Tensor]) -> torch.Tensor:
        """Restore a saved tensor to its original device without retaining graphs."""
        return saved[1].to(saved[0])

    hooks = (torch.autograd.graph.saved_tensors_hooks(pack, unpack)
             if args.saved_cpu_hooks else nullcontext())
    with hooks, patch.object(native, 'placement_is_profitable', return_value=True), \
         patch.object(native, 'borrow_parameters', side_effect=observe_borrow):
        if args.inference_first:
            with torch.inference_mode():
                make_invocation(24 + ep_rank, True, 111)
            torch.npu.synchronize()
        # Alternate routes and shapes, including an entirely empty source.
        for iteration, (hot, length) in enumerate([(True, 32 + ep_rank), (False, 16),
                                                   (True, 0 if ep_rank == 1 else 19), (True, 32 + ep_rank)]):
            module.zero_grad(set_to_none=True)
            before = len(borrow_calls)
            actual, expected, weights = make_invocation(length, hot, 191 + iteration)
            pairs = [(actual, expected, weights)]
            if args.delayed:
                pairs.append(make_invocation(13 + ep_rank, True, 301 + iteration))
            for actual, expected, weights in reversed(pairs):
                torch.testing.assert_close(actual[0], expected[0], rtol=rtol, atol=atol)
                actual[0].sum().backward(retain_graph=args.retain)
                expected[0].sum().backward(retain_graph=args.retain)
                if args.retain:
                    actual[0].sum().backward()
                    expected[0].sum().backward()
                for value, reference in zip(actual[1:], expected[1:]):
                    torch.testing.assert_close(value.grad, reference.grad, rtol=rtol, atol=atol)
            assert all(target() is None or target().buffer is None for target in gradient_targets)
            for name, parameter in module.experts.named_parameters():
                position = 0 if name == 'gate_up_proj' else 1
                expected_gradient = sum(pair[2][position].grad for pair in pairs)
                dist.all_reduce(expected_gradient)
                expected_gradient /= dp_size
                owned = expected_gradient[2 * ep_rank:2 * (ep_rank + 1)]
                actual_gradient = _tensor_gradient(parameter)
                if args.fsdp:
                    owned = owned.chunk(dp_size, dim=0)[dp_rank]
                torch.testing.assert_close(
                    actual_gradient, owned.to(device=actual_gradient.device, dtype=actual_gradient.dtype),
                    rtol=rtol, atol=atol)
            assert tuple(id(parameter) for parameter in module.parameters()) == identities
            if hot:
                assert len(borrow_calls) > before, 'Correctness hotspot must borrow actual expert weights'
            results.append({'hot': hot, 'source_rows': length * 2,
                            'borrow_calls': len(borrow_calls) - before})
    mega_modules = (
        'hyper_parallel.core.multicore.modules.mega_moe',
        'hyper_parallel.core.expert_parallel.moonep.mega_moe',
        'hyper_parallel.core.expert_parallel.moonep.shmem_transfer',
    )
    assert not any(name.startswith(mega_modules) for name in sys.modules)
    return {'rank': rank, 'cases': results, 'forced_profitable_correctness': True, 'args': vars(args)}

def _native_main() -> None:
    """Run one bounded torchrun device worker and persist per-rank evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32')
    parser.add_argument('--checkpoint', choices=('none', 'reentrant', 'nonreentrant'), default='none')
    parser.add_argument('--async-dispatch', action='store_true')
    parser.add_argument('--delayed', action='store_true')
    parser.add_argument('--retain', action='store_true')
    parser.add_argument('--fsdp', action='store_true')
    parser.add_argument('--offload', action='store_true')
    parser.add_argument('--master-float32', action='store_true')
    parser.add_argument('--saved-cpu-hooks', action='store_true')
    parser.add_argument('--inference-first', action='store_true')
    parser.add_argument('--capacity-factor', type=float)
    parser.add_argument('--grouped-gemm', action='store_true')
    args = parser.parse_args()
    result = _run_native_correctness(args)
    torch.npu.synchronize()
    Path(f'{args.output}.rank{dist.get_rank()}.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')


def _make_grouped_experts(args: argparse.Namespace, rank: int, ranks: int):
    """Construct through the actual replacement, then partition its expert axis."""
    source = nn.Module()
    source.num_experts = 2 * ranks
    source.num_local_experts = 2 * ranks
    source.hidden_size = 32
    source.intermediate_size = 48
    source.has_gate = not args.ungated
    source.is_transposed = True
    source.act_fn = functional.silu
    generator = torch.Generator().manual_seed(522)
    width = 48 if args.ungated else 96
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32
    shapes = {('up_proj' if args.ungated else 'gate_up_proj'): (2 * ranks, 32, width),
              'down_proj': (2 * ranks, 48, 32)}
    if args.bias:
        shapes.update(bias1=(2 * ranks, width), bias2=(2 * ranks, 32))
    full = {}
    for name, shape in shapes.items():
        value = torch.randn(shape, generator=generator) * 0.03
        value = value.to(device='npu', dtype=dtype)
        full[name] = value.detach().clone()
        if args.two_dimensional and name not in ('bias1', 'bias2'):
            value = value.flatten(0, 1)
        setattr(source, name, nn.Parameter(value))
    experts = GroupedExperts(module=source)
    for name in shapes:
        value = full[name][2 * rank:2 * (rank + 1)]
        if args.two_dimensional and name not in ('bias1', 'bias2'):
            value = value.flatten(0, 1)
        setattr(experts, name, nn.Parameter(value.clone()))
    experts.num_local_experts = 2
    return experts, full

def _grouped_oracle(inputs, ids, probabilities, parameters, ungated):
    """Use direct per-expert projections with the source activation and bias order."""
    first_name = 'up_proj' if ungated else 'gate_up_proj'
    output = torch.zeros_like(inputs)
    for expert in range(parameters[first_name].shape[0]):
        projected = inputs @ parameters[first_name][expert]
        if 'bias1' in parameters:
            projected = projected + parameters['bias1'][expert]
        if ungated:
            activated = functional.silu(projected)
        else:
            gate, up = projected.chunk(2, -1)
            activated = functional.silu(gate) * up
        projected = activated @ parameters['down_proj'][expert]
        if 'bias2' in parameters:
            projected = projected + parameters['bias2'][expert]
        coefficient = ((ids == expert) * probabilities).sum(-1).reshape(1, -1, 1)
        output = output + projected * coefficient
    return output

def _grouped_main() -> None:
    """Run one four-rank layout test without a MegaMoE executor or SHMEM heap."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--ungated', action='store_true')
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--two-dimensional', action='store_true')
    parser.add_argument('--dtype', default='bfloat16', choices=('float32', 'bfloat16'))
    parser.add_argument('--disable-moon', action='store_true')
    parser.add_argument('--direct-ab', action='store_true')
    parser.add_argument('--optimizer-step', action='store_true')
    parser.add_argument('--freeze', choices=('none', 'base', 'bias', 'all'), default='none')
    parser.add_argument('--empty-receiver', action='store_true')
    args = parser.parse_args()
    rank, ranks = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(42 + rank)
    torch.npu.manual_seed(42 + rank)
    module = nn.Module()
    module.experts, source = _make_grouped_experts(args, rank, ranks)
    execution.bind_local_expert_forward(module, ranks, use_grouped_gemm=True)
    if args.freeze != 'none':
        for name, parameter in module.experts.named_parameters():
            parameter.requires_grad_((args.freeze == 'base' and name in ('bias1', 'bias2'))
                                     or (args.freeze == 'bias' and name not in ('bias1', 'bias2')))
    identities = tuple(id(parameter) for parameter in module.parameters())
    reference_module = copy.deepcopy(module) if args.direct_ab else None
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1, weight_decay=0.1) if args.optimizer_step else None
    reference_optimizer = (torch.optim.SGD(reference_module.parameters(), lr=0.1, weight_decay=0.1)
                           if args.optimizer_step and reference_module is not None else None)
    native_borrow = native.borrow_parameters
    calls = []
    gradient_targets = []

    def observe(weight: tuple[torch.Tensor, ...], copies: tuple[tuple[int, ...], ...],
                group: dist.ProcessGroup, *extra: tuple[bool, ...]) -> tuple[
                    tuple[torch.Tensor, ...], tuple[precision.GradientTarget | None, ...]]:
        """Count actual transfers while preserving native HCCL execution."""
        calls.append(tuple(tuple(row) for row in copies))
        result, targets = native_borrow(weight, copies, group, *extra)
        assert all(target is None or target.buffer is None for target in targets)
        gradient_targets.extend(weakref.ref(target) for target in targets if target is not None)
        return result, targets

    results = []
    with patch.object(native, 'placement_is_profitable', return_value=True), \
         patch.object(native, 'borrow_parameters', side_effect=observe):
        scenarios = [(True, 32 + rank), (False, 16), (True, 0 if rank == 1 else 31)]
        if args.empty_receiver:
            scenarios = [(True, 3 if rank == 0 else 0)]
        for hot, length in scenarios:
            module.zero_grad(set_to_none=True)
            if reference_module is not None:
                reference_module.zero_grad(set_to_none=True)
            dtype = next(module.parameters()).dtype
            inputs = torch.randn(1, length, 32, device='npu', dtype=dtype, requires_grad=True)
            expected_inputs = inputs.detach().clone().requires_grad_()
            first = torch.zeros(length, dtype=torch.int64) if hot else torch.arange(length) % (2 * ranks)
            ids = torch.stack((first, (first + 1) % (2 * ranks)), -1).to('npu')
            probabilities = torch.full((length, 2), 0.5, dtype=dtype, device='npu', requires_grad=True)
            expected_probabilities = probabilities.detach().clone().requires_grad_()
            parameters = {name: value.detach().clone().requires_grad_() for name, value in source.items()}
            before = len(calls)
            output = execution.ep_routed_forward(
                module, inputs, router_fn=lambda *_: (ids, probabilities), ep_group=dist.group.WORLD,
                enable_moonep=not args.disable_moon)
            if reference_module is None:
                expected = _grouped_oracle(expected_inputs, ids, expected_probabilities, parameters, args.ungated)
            else:
                expected = execution.ep_routed_forward(
                    reference_module, expected_inputs, router_fn=lambda *_: (ids, expected_probabilities),
                    ep_group=dist.group.WORLD, enable_moonep=False)
            output.sum().backward()
            expected.sum().backward()
            assert all(target() is None or target().buffer is None for target in gradient_targets)
            if reference_module is not None:
                snapshot = {
                    'actual_output': output.detach().cpu(), 'reference_output': expected.detach().cpu(),
                    'actual_dx': inputs.grad.cpu(), 'reference_dx': expected_inputs.grad.cpu(),
                    'actual_dprobs': probabilities.grad.cpu(), 'reference_dprobs': expected_probabilities.grad.cpu(),
                    'actual_gradients': {name: None if parameter.grad is None else parameter.grad.cpu()
                                         for name, parameter in module.experts.named_parameters()},
                    'reference_gradients': {name: None if parameter.grad is None else parameter.grad.cpu()
                                            for name, parameter in reference_module.experts.named_parameters()},
                    'inputs': inputs.detach().cpu(), 'ids': ids.cpu(),
                    'probabilities': probabilities.detach().cpu(), 'source': source,
                }
                torch.save(snapshot, f'{args.output}.step{len(results)}.rank{rank}.pt')
            tolerance = {'rtol': 0.02, 'atol': 0.002} if dtype == torch.bfloat16 else {'rtol': 1e-4, 'atol': 1e-5}
            for actual, reference in [(output, expected), (inputs.grad, expected_inputs.grad),
                                      (probabilities.grad, expected_probabilities.grad)]:
                torch.testing.assert_close(actual, reference, **tolerance)
            for name, parameter in module.experts.named_parameters():
                if reference_module is None:
                    gradient = parameters[name].grad
                    dist.all_reduce(gradient)
                    owned = gradient[2 * rank:2 * (rank + 1)].reshape(parameter.shape)
                else:
                    owned = reference_module.experts.get_parameter(name).grad
                if owned is None:
                    owned = torch.zeros_like(parameter)
                actual_gradient = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
                try:
                    torch.testing.assert_close(actual_gradient, owned, **tolerance)
                except AssertionError as error:
                    torch.save({'parameter': name, 'actual': actual_gradient.cpu(), 'expected': owned.cpu(),
                                'inputs': inputs.detach().cpu(), 'ids': ids.cpu(),
                                'probabilities': probabilities.detach().cpu(),
                                'source': {key: value.cpu() for key, value in source.items()},
                                'hot': hot, 'rank': rank}, f'{args.output}.failure.rank{rank}.pt')
                    raise AssertionError(f'{name}: {error}') from error
            if optimizer is not None:
                masks = {name: parameter.grad is None for name, parameter in module.experts.named_parameters()}
                expected_masks = {name: parameter.grad is None
                                  for name, parameter in reference_module.experts.named_parameters()}
                before_parameters = {name: parameter.detach().cpu().clone()
                                     for name, parameter in module.experts.named_parameters()}
                optimizer.step()
                reference_optimizer.step()
                snapshot = {'none_masks': masks, 'reference_none_masks': expected_masks,
                            'before': before_parameters,
                            'after': {name: parameter.detach().cpu()
                                      for name, parameter in module.experts.named_parameters()},
                            'reference_after': {name: parameter.detach().cpu()
                                                for name, parameter in reference_module.experts.named_parameters()}}
                torch.save(snapshot, f'{args.output}.optimizer.step{len(results)}.rank{rank}.pt')
                assert masks == expected_masks, (
                    f'Optimizer None/zero gradient semantics differ: {masks} != {expected_masks}')
                for name, parameter in module.experts.named_parameters():
                    expected_parameter = reference_module.experts.get_parameter(name)
                    comparison = {'rtol': 0, 'atol': 0} if expected_masks[name] else tolerance
                    torch.testing.assert_close(parameter, expected_parameter, **comparison)
            assert identities == tuple(id(parameter) for parameter in module.parameters())
            if hot and not args.disable_moon:
                assert len(calls) > before
            results.append({'hot': hot, 'source_rows': 2 * length, 'borrow_calls': len(calls) - before})
    Path(f'{args.output}.rank{rank}.json').write_text(
        json.dumps({'args': vars(args), 'results': results}, indent=2), encoding='utf-8')


class _NativeV41(nn.Module):
    """Keep the model's actual router and source expert Parameter objects."""

    def __init__(self, rank: int, ranks: int, limit: float) -> None:
        """Build source expert shards and preserve the actual learned router."""
        super().__init__()
        config = SimpleNamespace(
            num_local_experts=2 * ranks, hidden_size=64, intermediate_size=128,
            hidden_act='silu', swiglu_limit=limit, num_experts_per_tok=2,
            scoring_func='sigmoid', routed_scaling_factor=1.0,
            router_aux_loss_coef=0.0, router_bias_update_rate=0.0,
        )
        self.is_hash = False
        self.gate = DeepseekV41TopKRouter(config).to('npu')
        self.experts = DeepseekV4Experts(config)
        with torch.no_grad():
            for parameter in self.experts.parameters():
                parameter.normal_(std=0.03)
            self.gate.weight.normal_(std=0.03)
        for name, parameter in list(self.experts.named_parameters()):
            setattr(self.experts, name, nn.Parameter(parameter[2 * rank:2 * rank + 2].detach().clone()))
        self.experts.to(device='npu', dtype=torch.bfloat16)
        self.shared_experts = nn.Linear(64, 64, bias=False).to(device='npu', dtype=torch.bfloat16)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Expose the source model signature consumed by the public compute factory."""
        raise NotImplementedError

def _run_v41_factory() -> None:
    """Compare real public factories under positive and disabled clamp limits.

    Only profitability is forced, so these small precision tensors actually
    exercise the real common allocator, HCCL weights and FP32 owner gradients.
    """
    rank, ranks = dist.get_rank(), dist.get_world_size()
    mesh = init_device_mesh('npu', (ranks,), mesh_dim_names=('ep',))
    torch.manual_seed(144)
    targets = []
    copies = []
    borrow = native.borrow_parameters

    def observe(parameters: tuple[torch.Tensor, ...], plan: tuple[tuple[int, ...], ...],
                group: dist.ProcessGroup, *options: tuple[bool, ...]) -> tuple[
                    tuple[torch.Tensor, ...], tuple[precision.GradientTarget | None, ...]]:
        """Count real borrowing and retain only weak accumulator references."""
        copies.append(plan)
        views, accumulators = borrow(parameters, plan, group, *options)
        targets.extend(weakref.ref(target) for target in accumulators if target is not None)
        return views, accumulators

    for limit in (0.0, 1.5):
        ordinary = _NativeV41(rank, ranks, limit)
        moon = copy.deepcopy(ordinary)
        identities = tuple(id(parameter) for parameter in moon.parameters())
        functions = []
        for module, enabled in ((ordinary, False), (moon, True)):
            functions.append(deepseek_v41_ep_compute_fn(
                module=module, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=mesh,
                use_grouped_gemm=True, enable_moonep=enabled))
        for length in (32 + rank, 0 if rank == 1 else 19):
            for module in (ordinary, moon):
                module.zero_grad(set_to_none=True)
                with torch.no_grad():
                    module.gate.bias.zero_()
                    module.gate.bias[:2].fill_(20)
            generator = torch.Generator().manual_seed(198 + rank)
            reference = (torch.randn(1, length, 64, generator=generator) * 0.1).to('npu', torch.bfloat16)
            inputs = [reference.detach().clone().requires_grad_() for _ in range(2)]
            probabilities = []

            def retain_probabilities(_module: nn.Module, _args: tuple[torch.Tensor, ...],
                                     output: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
                """Keep router probability gradients for the direct comparison."""
                output[1].retain_grad()
                probabilities.append(output[1])

            hooks = [module.gate.register_forward_hook(retain_probabilities) for module in (ordinary, moon)]
            before = len(copies)
            with patch.object(native, 'placement_is_profitable', return_value=True), \
                    patch.object(native, 'borrow_parameters', side_effect=observe):
                outputs = [function(module, states)
                           for function, module, states in zip(functions, (ordinary, moon), inputs)]
                for output in outputs:
                    output.sum().backward()
            for hook in hooks:
                hook.remove()
            for actual, expected in ((outputs[1], outputs[0]), (inputs[1].grad, inputs[0].grad),
                                     (probabilities[1].grad, probabilities[0].grad)):
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
            for name, parameter in moon.named_parameters():
                reference_gradient = ordinary.get_parameter(name).grad
                if reference_gradient is None:
                    assert parameter.grad is None, f'{name}: actual={parameter.grad}, expected=None'
                else:
                    torch.testing.assert_close(parameter.grad, reference_gradient, rtol=0.02, atol=0.002)
            assert len(copies) > before, f'borrow calls={len(copies)}, previous={before}'
            assert identities == tuple(id(parameter) for parameter in moon.parameters())
            assert all(target() is None or target().buffer is None for target in targets)
            assert isinstance(moon.experts, DeepseekV4Experts)


@pytest.mark.parametrize('case_name', tuple(NATIVE_CASES))
@pytest.mark.usefixtures('_distributed_group')
def test_native_moonep_case(case_name: str, tmp_path: Path) -> None:
    """Exercise one representative contract through HP's real native execution."""
    implementation, arguments, chunks = NATIVE_CASES[case_name]
    with patch.dict(os.environ, {'HP_EP_DISPATCH_CHUNKS': str(chunks)}):
        if implementation == 'v41':
            _run_v41_factory()
            return
        argv = [__file__, '--output', str(tmp_path / case_name), *arguments]
        run = _native_main if implementation == 'native' else _grouped_main
        with patch.object(sys, 'argv', argv):
            run()
