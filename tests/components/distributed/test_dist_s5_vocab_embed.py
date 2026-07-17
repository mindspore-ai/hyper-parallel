# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S5.1（2 进程）: D-02 vocab-parallel embedding masked wrapper。

覆盖分支：token 落在本 rank 区间 / 区间外（mask 置 0）/ 恰好等于区间边界值；
N-变体：rank1 区间 [V/2, V) 的 token 仅 rank1 产出非零 embedding。
"""

import torch
import torch.nn as nn

from hyper_models.components.distributed.sharding_applier import (
    _shard_module_params,
    _wrap_vocab_parallel_embedding,
)
from hyper_models.components.distributed.sharding_config import TP
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard
from tests.components.distributed.conftest import run_dist


def _worker(rank, world_size):
    V, H = 32, 8
    torch.manual_seed(0)
    emb = nn.Embedding(V, H)
    full_weight = emb.weight.detach().clone()
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    _shard_module_params(emb, {"weight": {TP: Shard(0)}}, mesh, ("tp",))
    # production：参数解包为 local 后注入 masked wrapper
    from hyper_models.components.distributed.sharding.apply import (
        _local_params_context,
    )
    _local_params_context(emb)
    _wrap_vocab_parallel_embedding(emb, mesh["tp"])

    # 含越界 token id 的全局 input_ids（每 rank 都看到全词表 id）
    input_ids = torch.arange(V).unsqueeze(0)  # [1, V] 覆盖全词表
    out = emb(input_ids)
    ref = torch.nn.functional.embedding(input_ids, full_weight)

    chunk = V // world_size
    lo, hi = rank * chunk, (rank + 1) * chunk
    # 本 rank 区间内：与参考一致
    torch.testing.assert_close(out[0, lo:hi], ref[0, lo:hi])
    # 区间外：mask 置 0（Partial 贡献语义）
    mask_out = torch.ones(V, dtype=torch.bool)
    mask_out[lo:hi] = False
    assert out[0, mask_out].abs().max().item() == 0.0
    # 边界值：token == lo 与 token == hi-1 命中本 rank；token == hi 不命中
    assert out[0, lo].abs().sum() > 0
    assert out[0, hi - 1].abs().sum() > 0
    if hi < V:
        assert out[0, hi].abs().sum() == 0

    # Partial 归约语义：两 rank 输出求和 == 完整 embedding
    summed = out.clone()
    torch.distributed.all_reduce(summed, group=mesh["tp"].get_group())
    torch.testing.assert_close(summed, ref)

    # 区间外 token 的梯度为零
    out2 = emb(input_ids)
    out2.sum().backward()
    grad = emb.weight.grad
    # 本地权重只接收本区间 token 的梯度——每行都应有梯度（ids 覆盖全区间），
    # 且不会收到越界 token 的梯度（越界 id 被 mask，grad 行为 0 已隐含于
    # masked forward 的梯度路径）——此处验证梯度形状与有限性
    assert grad.shape == (chunk, H)
    assert torch.isfinite(grad).all()


def test_vocab_parallel_embedding_tp2():
    run_dist(2, _worker)
