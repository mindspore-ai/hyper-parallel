# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S3.1（2 进程）: flex_cp_allgather 前向全局一致 + backward == 手工 reduce-scatter。"""

import torch
import torch.distributed as dist

from hyper_models.components.distributed.cp_utils import flex_cp_allgather
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from tests.components.distributed.conftest import run_dist


def _worker(rank, world_size):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("cp",))
    torch.manual_seed(0)
    full_k = torch.randn(1, 2, 8, 4)
    full_v = torch.randn(1, 2, 8, 4)
    chunk = 8 // world_size
    slc = slice(rank * chunk, (rank + 1) * chunk)

    k = full_k[:, :, slc].contiguous().requires_grad_(True)
    v = full_v[:, :, slc].contiguous().requires_grad_(True)
    gk, gv = flex_cp_allgather(k, v, 2, mesh)
    # gather 后 K/V 全局一致（cat 顺序 [chunk_rank0, chunk_rank1]）
    torch.testing.assert_close(gk, full_k)
    torch.testing.assert_close(gv, full_v)

    # backward：对 gk 加非均匀权重求和，k.grad == 手工 reduce-scatter
    w = torch.arange(gk.numel(), dtype=torch.float32).reshape(gk.shape)
    (gk * w).sum().backward()
    # 手工期望：grad = w（各 rank 相同）→ all_reduce 后取本 rank chunk
    expect = w * world_size
    torch.testing.assert_close(k.grad, expect[:, :, slc])

    # cp_size=1 直通
    mesh1 = init_device_mesh("cpu", (1, 1), mesh_dim_names=("cp", "tp"))["cp"]
    k2, v2 = flex_cp_allgather(k.detach(), v.detach(), 2, mesh1)
    assert k2.shape[2] == chunk


def test_cp_allgather_2proc():
    run_dist(2, _worker)
