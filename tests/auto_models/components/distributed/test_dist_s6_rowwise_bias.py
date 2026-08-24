# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_dist_s6_rowwise_bias.py: D-22 rowwise bias 后置的双模式数值/梯度对拍。

带 bias 的 TinyLlama 变体（TinyConfig(bias=True)：q/k/v/o_proj 与
gate/up/down_proj 全部带 bias，OPT/GPT-NeoX 风格），TP=2：

- production / validate vs 单卡参考输出对拍（SP 开/关）——修复前
  production 输出 = 正确值 + tp_size × rowwise_bias（见
  docs/trainer/d22_rowwise_bias_deferred_design.md §1 的复现签名）；
- 双模式梯度等价 + 与单卡参考梯度对拍（SP 下）：
  colwise bias（q_proj/gate_proj，Shard(0) 随权重）本地分片切片对拍；
  rowwise deferred bias（o_proj/down_proj，保持 Replicate）模拟
  source_shard_info 旁路 all-reduce 后对拍；
- 参数身份不变：apply 后 state_dict 键集合与 apply 前完全一致。
"""

import torch
from hyper_parallel.auto_models.components.distributed import (
    ShardingPlanner,
    apply_sharding_plan,
)
from hyper_parallel.auto_models.components.distributed.testing.grad_equiv import (
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
    cfg = TinyConfig(bias=True)
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    tp_group = mesh["tp"].get_group()
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    labels = torch.randint(0, 32, (2, 8))

    # 单卡参考（输出 + 梯度）
    torch.manual_seed(1234)
    ref_model = TinyLlamaForCausalLM(cfg)
    with torch.no_grad():
        ref_out = ref_model(x)
    _, ref_grads = run_one_step(ref_model, x, labels, cfg.vocab_size)
    # 与 test_dist_s5_equiv 同一缩放约定：各 rank 在 all_gather 后的完整
    # logits 上重复计算 loss，all_gather 反向把相同梯度求和 → 分布式梯度 =
    # world_size × 单卡梯度
    ref_grads = {k: (v * world_size if v is not None else None)
                 for k, v in ref_grads.items()}

    for sp in (True, False):
        outs, grads = {}, {}
        for mode in ("production", "validate"):
            torch.manual_seed(1234)
            model = TinyLlamaForCausalLM(cfg)
            ref_keys = set(model.state_dict())
            plan = ShardingPlanner().plan(
                model, mesh, tp_size=world_size, sequence_parallel=sp)
            # D-22 标记存在性：两个 rowwise 位置都被标记后置
            for layer in ("0", "1"):
                assert plan.modules[
                    f"model.layers.{layer}.self_attn"
                ]._deferred_bias_params == ("o_proj.bias",)
                assert plan.modules[
                    f"model.layers.{layer}.mlp"
                ]._deferred_bias_params == ("down_proj.bias",)
            model, _ = apply_sharding_plan(
                model, plan, mesh, validate_mode=(mode == "validate"))
            # 参数身份不变：state_dict 键集合不受 bias 抑制/后置影响
            assert set(model.state_dict()) == ref_keys
            with torch.no_grad():
                outs[mode] = model(x)
            _, g = run_one_step(model, x, labels, cfg.vocab_size)
            grads[mode] = g

        # 核心对拍：两模式 vs 单卡（修复前 production 误差 = (tp-1)·bias）
        for mode in ("production", "validate"):
            torch.testing.assert_close(outs[mode], ref_out,
                                       rtol=1e-4, atol=1e-5)

        # 双模式梯度等价（两模式 backward 均为 local autograd，05 §1.0）
        assert_grad_equivalence(grads["production"], grads["validate"],
                                rtol=1e-3)

        if not sp:
            # nosp 下每 rank 持全量输出，Replicate 参数的局部梯度已是全量
            # 贡献的冗余拷贝，不经 source_shard_info 旁路——梯度对拍限定 SP 形态
            # （与既有 grad_equiv 套件口径一致）
            continue
        for mode in ("production", "validate"):
            g = grads[mode]
            # colwise bias（Shard(0) 随权重）：本 rank 梯度 == 参考切片
            chunk_h = cfg.hidden_size // world_size
            chunk_i = cfg.intermediate_size // world_size
            torch.testing.assert_close(
                g["model.layers.0.self_attn.q_proj.bias"],
                ref_grads["model.layers.0.self_attn.q_proj.bias"][
                    rank * chunk_h:(rank + 1) * chunk_h],
                rtol=1e-3, atol=1e-5)
            torch.testing.assert_close(
                g["model.layers.0.mlp.gate_proj.bias"],
                ref_grads["model.layers.0.mlp.gate_proj.bias"][
                    rank * chunk_i:(rank + 1) * chunk_i],
                rtol=1e-3, atol=1e-5)
            # rowwise deferred bias（保持 Replicate）：Partial 贡献需
            # source_shard_info 旁路 all-reduce 后与参考一致（两 rank 相等）
            for name in ("model.layers.0.self_attn.o_proj.bias",
                         "model.layers.0.mlp.down_proj.bias"):
                synced = simulate_tp_replicate_grad_sync(g[name], tp_group)
                torch.testing.assert_close(synced, ref_grads[name],
                                           rtol=1e-3, atol=1e-5)


def test_rowwise_bias_dual_mode_tp2():
    run_dist(2, _worker)
