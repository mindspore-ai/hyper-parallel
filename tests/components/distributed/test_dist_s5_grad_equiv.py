# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S5.3（2 进程）: M_D.15a 双模式梯度等价（TP-Shard 与 TP-Replicate 两类参数）。"""

import torch

from hyper_models.components.distributed import ShardingPlanner, apply_sharding_plan
from hyper_models.components.distributed.testing.grad_equiv import (
    assert_grad_equivalence,
    run_one_step,
    simulate_tp_replicate_grad_sync,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyLlamaForCausalLM,
    run_dist,
)


def _worker(rank, world_size):
    cfg = TinyConfig()
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    tp_group = mesh["tp"].get_group()
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    labels = torch.randint(0, 32, (2, 8))

    # 单卡参考梯度
    torch.manual_seed(1234)
    ref_model = TinyLlamaForCausalLM(cfg)
    _, ref_grads = run_one_step(ref_model, x, labels, cfg.vocab_size)
    # 分布式侧的 loss 在每个 rank 上对 all_gather 后的完整 logits 重复计算，
    # all_gather 的反向（reduce_scatter）把各 rank 相同的梯度流求和——分布式
    # 梯度 = world_size × 单卡梯度（production/validate 两模式语义一致，
    # 双模式等价性不受影响；真实训练中由 loss_parallel 或 DP 平均吸收该缩放）。
    ref_grads = {k: (v * world_size if v is not None else None)
                 for k, v in ref_grads.items()}

    grads = {}
    for mode in ("production", "validate"):
        torch.manual_seed(1234)
        model = TinyLlamaForCausalLM(cfg)
        plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
        model, _ = apply_sharding_plan(model, plan, mesh,
                                       validate_mode=(mode == "validate"))
        loss, g = run_one_step(model, x, labels, cfg.vocab_size)
        grads[mode] = g

    # 双模式梯度等价（两模式 backward 均为 local autograd，05 §1.0）
    assert_grad_equivalence(grads["production"], grads["validate"], rtol=1e-3)

    # 与单卡参考对拍（逐 rank，覆盖 TP-Shard 与 TP-Replicate 两类）
    chunk_v = cfg.vocab_size // world_size
    chunk_i = cfg.intermediate_size // world_size
    for mode in ("production", "validate"):
        g = grads[mode]
        # TP-Shard 参数：本 rank 梯度 == 参考梯度对应切片（免同步）
        torch.testing.assert_close(
            g["model.embed_tokens.weight"],
            ref_grads["model.embed_tokens.weight"][rank * chunk_v:(rank + 1) * chunk_v],
            rtol=1e-3, atol=1e-5)
        torch.testing.assert_close(
            g["model.layers.0.mlp.gate_proj.weight"],
            ref_grads["model.layers.0.mlp.gate_proj.weight"][rank * chunk_i:(rank + 1) * chunk_i],
            rtol=1e-3, atol=1e-5)
        # N10：TP-Replicate 参数（norm）——Partial 贡献需 tp_grad_info 旁路
        # all-reduce 后与参考一致（两 rank 相等）
        norm_name = "model.layers.0.input_layernorm.weight"
        synced = simulate_tp_replicate_grad_sync(g[norm_name], tp_group)
        torch.testing.assert_close(synced, ref_grads[norm_name],
                                   rtol=1e-3, atol=1e-5)


def test_grad_equiv_tp2():
    run_dist(2, _worker)
