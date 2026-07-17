# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S2.2（2 进程）: Phase A _shard_module_params。"""

import pytest
import torch
import torch.nn as nn

from hyper_models.components.distributed.sharding_applier import (
    _shard_module_params,
)
from hyper_models.components.distributed.sharding_config import (
    CP,
    EP,
    TP,
    PlacementMismatchError,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.components.distributed.conftest import run_dist


def _worker_shard_params(rank, world_size):
    torch.manual_seed(0)
    lin = nn.Linear(16, 16, bias=False)
    full_q = lin.weight.detach().clone()
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    # q_proj 风格 Shard(0)
    _shard_module_params(lin, {"weight": {TP: Shard(0), CP: Replicate()}},
                         mesh, ("tp",))
    assert isinstance(lin.weight, DTensor)
    local = lin.weight.to_local()
    assert local.shape == (16 // world_size, 16)
    # N1: 逐 rank 断言本 rank 的 local 切片 == 全量对应切片
    chunk = 16 // world_size
    torch.testing.assert_close(local, full_q[rank * chunk:(rank + 1) * chunk, :])
    assert lin.weight.requires_grad

    # o_proj 风格 Shard(1)（N2）
    lin2 = nn.Linear(16, 16, bias=False)
    full_o = lin2.weight.detach().clone()
    _shard_module_params(lin2, {"weight": {TP: Shard(1), CP: Replicate()}},
                         mesh, ("tp",))
    local2 = lin2.weight.to_local()
    assert local2.shape == (16, 16 // world_size)
    torch.testing.assert_close(local2, full_o[:, rank * chunk:(rank + 1) * chunk])


def _worker_ep_shard(rank, world_size):
    torch.manual_seed(0)
    holder = nn.Module()
    holder.experts = nn.Module()
    holder.experts.w1 = nn.Parameter(torch.randn(4, 8, 16))
    full = holder.experts.w1.detach().clone()
    mesh = init_device_mesh("cpu", (1, world_size), mesh_dim_names=("tp", "ep"))

    _shard_module_params(
        holder, {"experts.w1": {EP: Shard(0), TP: Shard(0), CP: Replicate()}},
        mesh, ("tp", "ep"))
    local = holder.experts.w1.to_local()
    # N7 变体：EP 沿 expert 维切，TP 沿 dim0（expert 维已被 EP 切后为局部）
    assert local.shape == (4 // world_size, 8, 16)
    chunk_e = 4 // world_size
    torch.testing.assert_close(local, full[rank * chunk_e:(rank + 1) * chunk_e])


def _worker_already_dtensor(rank, world_size):
    torch.manual_seed(0)
    lin = nn.Linear(8, 8, bias=False)
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    spec = {"weight": {TP: Shard(0)}}
    _shard_module_params(lin, spec, mesh, ("tp",))
    # placement 一致 → 幂等跳过
    _shard_module_params(lin, spec, mesh, ("tp",))
    # placement 不一致 → 抛错
    with pytest.raises(PlacementMismatchError):
        _shard_module_params(lin, {"weight": {TP: Shard(1)}}, mesh, ("tp",))


def _worker_meta(rank, world_size):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    with torch.device("meta"):
        lin = nn.Linear(16, 16, bias=False)
    _shard_module_params(lin, {"weight": {TP: Shard(0)}}, mesh, ("tp",))
    assert isinstance(lin.weight, DTensor)
    assert lin.weight.to_local().is_meta
    assert lin.weight.to_local().shape == (16 // world_size, 16)


def test_shard_params_tp2():
    run_dist(2, _worker_shard_params)


def test_ep_shard_2proc():
    run_dist(2, _worker_ep_shard)


def test_already_dtensor_2proc():
    run_dist(2, _worker_already_dtensor)


def test_meta_tensor_2proc():
    run_dist(2, _worker_meta)
