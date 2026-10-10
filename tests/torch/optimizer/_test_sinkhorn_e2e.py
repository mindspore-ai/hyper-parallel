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

"""NPU worker: Sinkhorn and head-wise Muon sharding/checkpoint parity."""

import copy
import os

import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=unused-import  # register the NPU backend
from torch import nn
from torch.distributed.device_mesh import init_device_mesh as init_torch_mesh
from torch.distributed.tensor import DTensor as TorchDTensor, Replicate as TorchReplicate, Shard as TorchShard
from torch.distributed.tensor import distribute_tensor as torch_distribute_tensor

from hyper_parallel import DTensor, SkipDTensorDispatch, distribute_tensor, init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.optimizer import get_hyper_optimizer
from hyper_parallel.core.optimizer.sinkhorn import Sinkhorn
from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor


def _init():
    """Initialize HCCL after selecting the local accelerator."""
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group("hccl")


def _clone_state(value):
    """Snapshot tensors through their supported clone operation, preserving DTensor layouts."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    return copy.deepcopy(value)


def _run_sinkhorn_layout(mesh, placements, shape, distribute=distribute_tensor):
    """Compare several updates and synchronized momentum to a full-matrix reference."""
    torch.manual_seed(19)
    initial = torch.randn(*shape, device="npu")
    reference = nn.Parameter(initial.clone())
    reference_optimizer = Sinkhorn([reference], lr=0.04)
    param = nn.Parameter(distribute(initial.clone(), mesh, placements))
    optimizer = Sinkhorn([param], lr=0.04)
    for iteration in range(4):
        gradient = torch.randn_like(initial)
        gradient[0] = 0
        if shape[0] > 1:
            gradient[1] *= 1e-8
        reference.grad = gradient.clone()
        param.grad = distribute(gradient, mesh, placements)
        local_gradient = param.grad.to_local().clone()
        reference_optimizer.step()
        with SkipDTensorDispatch():
            optimizer.step()
        torch.testing.assert_close(reference.grad, gradient, rtol=0, atol=0)
        torch.testing.assert_close(param.grad.to_local(), local_gradient, rtol=0, atol=0)
        expected_local = distribute(reference.detach().clone(), mesh, placements).to_local()
        torch.testing.assert_close(param.to_local(), expected_local, rtol=3e-5, atol=2e-6)
        optimizer._broadcast_state_fused_for_ckpt()  # pylint: disable=protected-access
        state = optimizer.state[param]["momentum_buffer"]
        if not isinstance(state, (DTensor, TorchDTensor)):
            raise AssertionError(f"Expected DTensor momentum state, got {type(state)}")
        expected_momentum = distribute(
            reference_optimizer.state[reference]["momentum_buffer"].clone(), mesh, placements,
        ).to_local()
        torch.testing.assert_close(state.to_local(), expected_momentum, rtol=1e-5, atol=1e-6)
        if iteration == 1:
            optimizer.load_state_dict(_clone_state(optimizer.state_dict()))
        optimizer.cleanup_synced_state()


def test_sinkhorn_layouts_4p():
    """Rows, columns, two axes, repeated row sharding, replicas and uneven shards."""
    _init()
    for dtype in (torch.float32, torch.bfloat16):
        source = torch.randn(3, 7, device="npu", dtype=dtype).T
        original = source.clone()
        param = nn.Parameter(torch.zeros_like(source))
        param.grad = source
        optimizer = Sinkhorn([param])
        optimizer.step()
        torch.testing.assert_close(source, original, rtol=0, atol=0)
    mesh = init_device_mesh("npu", (2, 2), mesh_dim_names=("first", "second"))
    for placements, shape in (
            ((Shard(0), Replicate()), (7, 6)),
            ((Shard(1), Replicate()), (7, 6)),
            ((Shard(0), Shard(1)), (7, 5)),
            ((Shard(0), Shard(0)), (8, 6)),
            ((Replicate(), Shard(0)), (7, 5)),
            ((Replicate(), Replicate()), (7, 5)),
    ):
        _run_sinkhorn_layout(mesh, placements, shape)
    native_mesh = init_torch_mesh("npu", (2, 2), mesh_dim_names=("row", "column"))
    for placements, shape in (
            ((TorchShard(0), TorchReplicate()), (7, 5)),
            ((TorchShard(0), TorchShard(1)), (7, 5)),
            ((TorchShard(0), TorchShard(1)), (1, 1)),
    ):
        _run_sinkhorn_layout(native_mesh, placements, shape, distribute=torch_distribute_tensor)
    dist.barrier()


def _round_trip_optimizer(model, optimizer):
    """Restore an initialized chain and compare saved states before the next update."""
    checkpoint = _clone_state(optimizer.state_dict())
    states = {name: {param: {key: to_local_if_dtensor(value).clone()
                            for key, value in state.items() if isinstance(value, torch.Tensor)}
                     for param, state in leaf.state.items()}
              for name, leaf in optimizer.optimizers_dict.items()}
    weights = {param: param.to_local().clone() for param in model.parameters()}
    optimizer.load_state_dict(checkpoint)
    for param, expected in weights.items():
        torch.testing.assert_close(param.to_local(), expected, msg=f"load changed {param.model_name}")
    for name, leaf in optimizer.optimizers_dict.items():
        for param, state in states[name].items():
            for key, expected in state.items():
                torch.testing.assert_close(to_local_if_dtensor(leaf.state[param][key]), expected,
                                           msg=f"load changed {name}/{param.model_name}/{key}")
    return optimizer


def test_head_wise_and_chain_checkpoint_4p():
    """Three-family hybrid sharding and checkpoint resume retain independent Q/K heads."""
    _init()
    mesh = init_device_mesh("npu", (2, 2), mesh_dim_names=("replicate", "shard"))
    torch.manual_seed(27)
    model = nn.Module()
    model.q_proj = nn.Linear(8, 12, bias=False).npu()
    model.embed = nn.Embedding(10, 8).npu()
    model.norm = nn.LayerNorm(8).npu()
    reference = copy.deepcopy(model)
    config = {"muon": {"head_wise": True, "head_dim": 4}, "sinkhorn": {}, "adamw": {}}
    reference_optimizer = get_hyper_optimizer(reference, **config)
    # Shard rows through the middle of a head: 6 rows per rank versus head_dim=4.
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            module.register_parameter(name, nn.Parameter(distribute_tensor(
                param.detach().clone(), mesh, (Replicate(), Shard(0)))))
    optimizer = get_hyper_optimizer(model, **config)
    for iteration in range(4):
        for param, ref in zip(model.parameters(), reference.parameters()):
            gradient = torch.randn_like(ref)
            ref.grad = gradient
            param.grad = distribute_tensor(gradient.clone(), mesh, param.placements)
        reference_optimizer.step()
        with SkipDTensorDispatch():
            optimizer.step()
        for param, ref in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(param.full_tensor(), ref, rtol=1e-2, atol=1e-3)
        if iteration == 1:
            optimizer = _round_trip_optimizer(model, optimizer)
    dist.barrier()
