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
"""Eight-NPU checks for padded/empty shards and bound optimizer checkpoints."""

import json
import os
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
import torch_npu
from torch import nn

from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor, SkipDTensorDispatch
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard
from hyper_parallel.core.optimizer import get_hyper_optimizer
from hyper_parallel.core.optimizer.muon import compute_muon_slice_scale, zeropower_via_newtonschulz5


def _clone_state(value: Any) -> Any:
    """Snapshot optimizer state while preserving DTensor layouts."""
    if isinstance(value, DTensor):
        return DTensor.from_local_with_layout(value.to_local().clone(), value.layout, shape=tuple(value.shape))
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    return value


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Measure the relative L2 error of a valid local shard."""
    if actual.numel() == 0:
        return 0.0
    return float(((actual.float() - expected.float()).norm() /
                  expected.float().norm().clamp_min(1e-30)).item())


def _run_case(mesh: DeviceMesh, dtype: torch.dtype, shape: tuple[int, int],
              tp_dim: Optional[int], headwise: bool = False) -> None:
    """Compare sharded Muon updates and checkpoint continuation with a full-matrix reference."""
    rank = dist.get_rank()
    dp_rank, tp_rank = mesh.get_coordinate()
    full_initial = torch.arange(shape[0] * shape[1], device='npu', dtype=torch.float32).view(shape) / 100
    if dtype == torch.bfloat16:
        full_initial = full_initial.to(dtype).float()
    source = full_initial
    tp_start, tp_size = 0, shape[tp_dim] if tp_dim is not None else 0
    if tp_dim is not None:
        chunk = (shape[tp_dim] + 1) // 2
        tp_start = min(tp_rank * chunk, shape[tp_dim])
        tp_size = min(chunk, shape[tp_dim] - tp_start)
        source = source.narrow(tp_dim, tp_start, tp_size)
    rows = (source.shape[0] + 3) // 4
    start = min(dp_rank * rows, source.shape[0])
    valid_rows = min(rows, source.shape[0] - start)
    local_shape = (valid_rows, source.shape[1])
    physical = torch.zeros((rows, source.shape[1]), dtype=dtype, device='npu')
    physical[:valid_rows].copy_(source.narrow(0, start, valid_rows))
    placements = ((StridedShard(0, 2), Shard(0)) if tp_dim == 0 else
                  (Shard(0), Shard(1)) if tp_dim == 1 else (Shard(0), Replicate()))
    layout = Layout.from_device_mesh(mesh)
    layout.set_placements(placements)
    layout.placement_to_tensor_map(2)
    layout.set_tensor_meta(shape, (shape[1], 1), dtype)

    def slice_full(full: torch.Tensor) -> torch.Tensor:
        """Select this rank's valid rows and columns from the reference matrix.

        Args:
            full: Global reference matrix.
        """
        if tp_dim is not None:
            full = full.narrow(tp_dim, tp_start, tp_size)
        return full.narrow(0, start, valid_rows)

    def make_optimizer(
        local: torch.Tensor,
    ) -> tuple[nn.Module, Union[ChainedOptimizer, Float16OptimizerWithFloat16Params]]:
        """Bind this rank's physical parameter to its global matrix layout.

        Args:
            local: Initial values of the padded local shard.
        """
        model = nn.Module()
        model.weight = nn.Parameter(local.clone())
        model.weight.model_name = 'weight'
        raw = get_hyper_optimizer(model, [model.weight], [], muon_kwargs={
            'muon_lr': 0.001, 'muon_weight_decay': 0.0, 'muon_momentum': 0.95,
            'muon_ns_steps': 5, 'muon_ns_variant': 'asym5',
        })
        if headwise:
            raw.optimizers_dict['muon'].reshape_fn = lambda name, inp: [inp.view(-1, 2, inp.shape[-1])]
        raw.defer_parameter_binding()
        optimizer = Float16OptimizerWithFloat16Params(raw, model) if dtype == torch.bfloat16 else raw
        specs = {model.weight: (layout, local_shape)}
        optimizer.bind_parameter_layouts(specs)
        return model, optimizer

    model, optimizer = make_optimizer(physical)
    reference = full_initial.clone()
    momentum = torch.zeros_like(reference)
    errors = []
    for step in range(3):
        full_grad = torch.sin(torch.arange(reference.numel(), device='npu').float() + step + 1).view(shape)
        full_grad = full_grad.to(dtype).float()
        direction = full_grad + momentum * 0.95
        momentum.mul_(0.95).add_(full_grad)
        direction.mul_(0.95).add_(full_grad)
        ns_input = direction.to(torch.bfloat16)
        if headwise:
            ns_input = ns_input.view(-1, 2, ns_input.shape[-1])
        update = zeropower_via_newtonschulz5(ns_input, 5)
        update.mul_(compute_muon_slice_scale(update, 0.2))
        reference.add_(update.reshape(shape).float(), alpha=-0.001)
        grad = torch.zeros_like(model.weight)
        grad[:valid_rows].copy_(slice_full(full_grad))
        model.weight.grad = grad
        with SkipDTensorDispatch(no_skip={torch.zeros_like}):
            optimizer.step()
        optimizer.zero_grad()
        param = optimizer.optimizer_param_by_model_param[model.weight]
        error = _relative(param.to_local(), slice_full(reference))
        errors.append(error)
        if error >= 1e-4:
            raise AssertionError(f'padded update failed on rank{rank}: {shape}, {placements}, {error}')
        if bool(torch.count_nonzero(model.weight[valid_rows:])):
            raise AssertionError(f'padding is nonzero on rank{rank}')
    state = _clone_state(optimizer.state_dict())
    restored_model, restored = make_optimizer(model.weight.detach())
    restored.load_state_dict(state)
    for target in (optimizer, restored):
        target.model.weight.grad = torch.zeros_like(target.model.weight)
        with SkipDTensorDispatch(no_skip={torch.zeros_like}):
            target.step()
        target.zero_grad()
    original = optimizer.optimizer_param_by_model_param[model.weight].to_local()
    recovered = restored.optimizer_param_by_model_param[restored_model.weight].to_local()
    if _relative(original, recovered) >= 1e-6:
        raise AssertionError(f'checkpoint continuation differs on rank{rank}')
    if rank == 0:
        print('SHARD_CHECK_PASSED', json.dumps({'dtype': str(dtype), 'shape': shape, 'tp_dim': tp_dim,
                                               'headwise': headwise, 'errors': errors}), flush=True)


def test_graph_optimizer_shard_layouts() -> None:
    """Compare matrix updates and bound-state continuation across eight NPUs."""
    torch_npu.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl')
    mesh = init_device_mesh('npu', (4, 2), mesh_dim_names=('fsdp_shard', 'tp'), rank_list=tuple(range(8)))
    for dtype in (torch.float32, torch.bfloat16):
        for shape, tp_dim, headwise in [((5, 6), None, False), ((6, 5), 1, False),
                                       ((9, 6), 0, False), ((8, 6), None, True)]:
            _run_case(mesh, dtype, shape, tp_dim, headwise)
    dist.barrier()
    dist.destroy_process_group()
