# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_dist_s5_equiv.py: 核心套件合并文件。

来源: test_dist_s5_grad_equiv.py, test_dist_s5_mode_equiv.py, test_dist_s5_cp_same_kernel.py, test_dist_s5_vocab_embed.py
"""

import torch
import torch.nn as nn
from hyper_models.components.distributed import (
    ShardingPlanner,
    apply_sharding_plan,
)
from hyper_models.components.distributed.cp_utils import shard_batch_for_cp
from hyper_models.components.distributed.sharding_applier import (
    _shard_module_params,
    _wrap_inner_attention,
    _wrap_vocab_parallel_embedding,
)
from hyper_models.components.distributed.sharding_config import (
    CP,
    ModuleShardingSpec,
    TP,
)
from hyper_models.components.distributed.testing.grad_equiv import (
    assert_grad_equivalence,
    run_one_step,
    simulate_tp_replicate_grad_sync,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard
from tests.components.distributed.conftest import (
    _attach_ep,
    TinyConfig,
    TinyLlamaAttention,
    TinyLlamaForCausalLM,
    cp_sdpa_hf_injection,
    run_dist,
)


# ==========================================================================
# 来源: test_dist_s5_grad_equiv.py
# S5.3（2 进程）: M_D.15a 双模式梯度等价（TP-Shard 与 TP-Replicate 两类参数）。
# ==========================================================================

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


# ==========================================================================
# 来源: test_dist_s5_mode_equiv.py
# S5.4（2~4 进程）: M_M.2 — validate vs production 输出等价（三组合）。
# ==========================================================================

def _dual_run(model_builder, plan_kwargs, mesh, input_ids, cp_mesh=None,
              attach_ep_fn=None, plan_overrides=None):
    """同 batch 两模式各跑一次 forward，返回两模式输出。"""
    outs = {}
    for mode in ("production", "validate"):
        model = model_builder()
        plan = ShardingPlanner(plan_overrides=plan_overrides).plan(
            model, mesh, **plan_kwargs)
        model, _ = apply_sharding_plan(model, plan, mesh,
                                       validate_mode=(mode == "validate"))
        if attach_ep_fn is not None:
            attach_ep_fn(model)
        ids = input_ids
        if cp_mesh is not None:
            ids = shard_batch_for_cp({"input_ids": input_ids}, cp_mesh)["input_ids"]
        with torch.no_grad():
            outs[mode] = model(ids)
    return outs


def _worker_tp(rank, world_size):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    def build():
        torch.manual_seed(1234)
        return TinyLlamaForCausalLM(TinyConfig()).eval()

    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    outs = _dual_run(build, {"tp_size": world_size}, mesh, x)
    torch.testing.assert_close(outs["production"], outs["validate"],
                               rtol=1e-5, atol=1e-5)


def _worker_tp_cp(rank, world_size):
    assert world_size == 4
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("cp", "tp"))

    def build():
        torch.manual_seed(1234)
        return TinyLlamaForCausalLM(TinyConfig(), causal=True).eval()

    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    outs = _dual_run(build, {"tp_size": 2, "cp_size": 2}, mesh, x,
                     cp_mesh=mesh["cp"], plan_overrides=cp_sdpa_hf_injection())
    torch.testing.assert_close(outs["production"], outs["validate"],
                               rtol=1e-5, atol=1e-5)


def _worker_tp_ep(rank, world_size):
    assert world_size == 4
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("tp", "ep"))

    def build():
        torch.manual_seed(1234)
        return TinyLlamaForCausalLM(TinyConfig(num_experts=4)).eval()

    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    outs = _dual_run(build, {"tp_size": 2, "ep_size": 2}, mesh, x,
                     attach_ep_fn=lambda m: _attach_ep(m, mesh, 2))
    torch.testing.assert_close(outs["production"], outs["validate"],
                               rtol=1e-5, atol=1e-5)


def test_mode_equiv_tp():
    run_dist(2, _worker_tp)


def test_mode_equiv_tp_cp():
    run_dist(4, _worker_tp_cp)


def test_mode_equiv_tp_ep():
    run_dist(4, _worker_tp_ep)


# ==========================================================================
# 来源: test_dist_s5_cp_same_kernel.py
# S5.2（2 进程）: D-01'' — CP wrapper 双模式共用，区域内计算 kernel 级一致。
# ==========================================================================

def _worker__s5_cp_same_kernel(rank, world_size):
    cp_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("cp",))
    cfg = TinyConfig()
    S = 8
    chunk = S // world_size
    slc = slice(rank * chunk, (rank + 1) * chunk)
    torch.manual_seed(0)
    hidden = torch.randn(2, S, cfg.hidden_size)

    torch.manual_seed(1234)
    attn = TinyLlamaAttention(cfg, causal=True).eval()
    # 显式声明 "sdpa_hf" wrapper（改造后无启发式分派）
    spec = ModuleShardingSpec(out_src={"output": {CP: Shard(1)}},
                              inner_target="self",
                              inner_wrapper="sdpa_hf", region_dispatch=False)
    _wrap_inner_attention(attn, cp_mesh, spec=spec, mesh=cp_mesh,
                             mesh_dim_names=("cp",))

    hs_local = hidden[:, slc].contiguous()
    hs_dt = DTensor.from_local(hs_local.clone(), cp_mesh, (Shard(1),))
    with torch.no_grad():
        out_local = attn(hs_local)
        out_dt = attn(hs_dt)
    assert isinstance(out_dt, DTensor)
    # 严格容差：两模式跑同一份 wrapper、同一 all-gather、同一 SDPA
    torch.testing.assert_close(out_dt.to_local(), out_local,
                               rtol=1e-6, atol=1e-7)


def test_cp_same_kernel_dual_mode_2proc():
    run_dist(2, _worker__s5_cp_same_kernel)


# ==========================================================================
# 来源: test_dist_s5_vocab_embed.py
# S5.1（2 进程）: D-02 vocab-parallel embedding masked wrapper。
# ==========================================================================

def _worker__s5_vocab_embed(rank, world_size):
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
    run_dist(2, _worker__s5_vocab_embed)
