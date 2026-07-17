# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S2.3（2 进程）: RedistOp.execute + _classify_collective 五组合数值。"""

import torch
import torch.distributed as dist

from hyper_models.components.distributed.precompiled_boundary import (
    RedistOp,
    _classify_collective,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from tests.components.distributed.conftest import run_dist


def _classify_cases():
    R, P, S = Replicate(), Partial(), Shard(1)
    assert _classify_collective((S,), (S,)) == "identity"
    assert _classify_collective((S,), (R,)) == "all_gather"
    assert _classify_collective((P,), (S,)) == "reduce_scatter"
    assert _classify_collective((P,), (R,)) == "all_reduce"
    assert _classify_collective((R,), (S,)) == "redistribute"


def _worker(rank, world_size):
    _classify_cases()
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    torch.manual_seed(42)
    full = torch.randn(4, 8)
    chunk = 8 // world_size

    # 1. identity：直通零通信
    op = RedistOp("x", None, mesh, (Replicate(),), (Replicate(),), "identity")
    out = op.execute(full)
    assert out is full
    out_dt = op.execute(full, as_dtensor=True)
    assert isinstance(out_dt, DTensor)

    # 2. Shard→Replicate（all_gather）：本 rank 切片 → 全量
    local = full[:, rank * chunk:(rank + 1) * chunk].contiguous()
    op = RedistOp("x", None, mesh, (Shard(1),), (Replicate(),), "all_gather")
    out = op.execute(local)
    torch.testing.assert_close(out, full)

    # 3. Partial→Shard（reduce_scatter）：各 rank 持有不同的 partial 贡献
    # partial_i = full * (rank+1)；reduce_scatter 后每 rank 拿 sum 的第 rank 块
    partial = full * (rank + 1)
    expect_sum = full * sum(range(1, world_size + 1))
    op = RedistOp("x", None, mesh, (Partial(),), (Shard(1),), "reduce_scatter")
    out = op.execute(partial)
    # N3：逐 rank 断言本 rank 拿到输出第 rank 块且数值 = 各块之和
    torch.testing.assert_close(out, expect_sum[:, rank * chunk:(rank + 1) * chunk])

    # 4. Partial→Replicate（all_reduce）
    op = RedistOp("x", None, mesh, (Partial(),), (Replicate(),), "all_reduce")
    out = op.execute(partial)
    torch.testing.assert_close(out, expect_sum)

    # 5. Replicate→Shard（redistribute / 切片）
    op = RedistOp("x", None, mesh, (Replicate(),), (Shard(1),), "redistribute")
    out = op.execute(full)
    torch.testing.assert_close(out, full[:, rank * chunk:(rank + 1) * chunk])

    # as_dtensor=True 路径：返回 DTensor 且 placement 为 dst
    out_dt = op.execute(full, as_dtensor=True)
    assert isinstance(out_dt, DTensor)
    assert tuple(out_dt.placements) == (Shard(1),)
    torch.testing.assert_close(
        out_dt.to_local(), full[:, rank * chunk:(rank + 1) * chunk])


def test_redist_op_2proc():
    run_dist(2, _worker)
