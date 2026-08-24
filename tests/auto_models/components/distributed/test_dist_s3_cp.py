# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_dist_s3_cp.py: 核心套件合并文件。

来源: test_dist_s3_cp_allgather.py, test_dist_s3_cp_hf_wrapper.py, test_dist_s3_cp_qkv_wrapper.py, test_dist_s3_phase_c_cp.py, test_dist_s3_tp_cp_e2e.py
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from hyper_parallel.auto_models.components.distributed import (
    ShardingPlanner,
    apply_sharding_plan,
)
from hyper_parallel.auto_models.components.distributed.cp_utils import (
    flex_cp_allgather,
    shard_batch_for_cp,
)
from hyper_parallel.auto_models.components.distributed.precompiled_boundary import PrecompiledBoundary
from hyper_parallel.auto_models.components.distributed.sharding_applier import _wrap_inner_attention
from hyper_parallel.auto_models.components.distributed.sharding_config import (
    CP,
    ModuleShardingSpec,
    TP,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Shard,
)
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyLlamaAttention,
    TinyLlamaForCausalLM,
    cp_sdpa_hf_injection,
    run_dist,
)


# ==========================================================================
# 来源: test_dist_s3_cp_allgather.py
# S3.1（2 进程）: flex_cp_allgather 前向全局一致 + backward == 手工 reduce-scatter。
# ==========================================================================

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


# ==========================================================================
# 来源: test_dist_s3_cp_hf_wrapper.py
# S3.4（2 进程）: HF 原语拦截 CP wrapper — G4 causal + 拦截还原 + 双模式同源。
# ==========================================================================

def _worker__s3_cp_hf_wrapper(rank, world_size):
    cp_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("cp",))
    full_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("cp",))
    cfg = TinyConfig()
    S = 8
    chunk = S // world_size
    slc = slice(rank * chunk, (rank + 1) * chunk)

    torch.manual_seed(1234)
    ref_attn = TinyLlamaAttention(cfg, causal=True).eval()
    torch.manual_seed(0)
    hidden = torch.randn(2, S, cfg.hidden_size)
    with torch.no_grad():
        ref = ref_attn(hidden)  # 单卡参考（全序列 causal）

    # ── G4 causal 用例：HF 风格 attention CP=2 vs 单卡 ──
    torch.manual_seed(1234)
    attn = TinyLlamaAttention(cfg, causal=True).eval()
    # 显式声明：HF 风格 forward(hidden_states) → "sdpa_hf" 原语拦截 wrapper
    spec = ModuleShardingSpec(
        out_src={"output": {CP: Shard(1), TP: Partial()}},
        inner_target="self", inner_wrapper="sdpa_hf", region_dispatch=False)
    _wrap_inner_attention(attn, cp_mesh, spec=spec, mesh=full_mesh,
                             mesh_dim_names=("cp",))
    orig_sdpa = F.scaled_dot_product_attention
    with torch.no_grad():
        out = attn(hidden[:, slc].contiguous())
    # 拦截后 F.scaled_dot_product_attention 已还原
    assert F.scaled_dot_product_attention is orig_sdpa
    # N4：rank1（Q 全局位置 [S/2, S)）输出 == 单卡参考对应切片
    torch.testing.assert_close(out, ref[:, slc], rtol=1e-5, atol=1e-5)

    # ── 双模式同源断言：DTensor 输入（validate 路径）与 local 输入
    # （production 路径）经同一 wrapper 输出逐元素相等 ──
    torch.manual_seed(1234)
    attn2 = TinyLlamaAttention(cfg, causal=True).eval()
    _wrap_inner_attention(attn2, cp_mesh, spec=spec, mesh=full_mesh,
                             mesh_dim_names=("cp",))
    hs_dt = DTensor.from_local(hidden[:, slc].contiguous(), full_mesh,
                               (Shard(1),))
    with torch.no_grad():
        out_dt = attn2(hs_dt)
        out_local = attn2(hidden[:, slc].contiguous())
    assert isinstance(out_dt, DTensor)
    # validate 出口按声明 out_src 重包装
    assert tuple(out_dt.placements)[0] == Shard(1)
    torch.testing.assert_close(out_dt.to_local(), out_local,
                               rtol=1e-6, atol=1e-6)


def test_cp_hf_wrapper_2proc():
    run_dist(2, _worker__s3_cp_hf_wrapper)


# ==========================================================================
# 来源: test_dist_s3_cp_qkv_wrapper.py
# S3.3（2 进程）: NeMo 风格 (q,k,v) CP wrapper — CP=2 输出 vs 单卡参考。
# ==========================================================================

class ToyNeMoAttention(nn.Module):
    """NeMo 风格：inner_attention 子模块 forward(q,k,v,is_causal=False)。"""

    class Inner(nn.Module):
        def forward(self, q, k, v, is_causal=False, attn_mask=None):
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal, attn_mask=attn_mask)

    def __init__(self):
        super().__init__()
        self.inner_attention = self.Inner()

    def forward(self, q, k, v, is_causal=False, attn_mask=None):
        return self.inner_attention(q, k, v, is_causal=is_causal,
                                    attn_mask=attn_mask)


def _worker__s3_cp_qkv_wrapper(rank, world_size):
    cp_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("cp",))
    torch.manual_seed(0)
    B, N, S, D = 2, 2, 8, 4
    full_q = torch.randn(B, N, S, D)
    full_k = torch.randn(B, N, S, D)
    full_v = torch.randn(B, N, S, D)
    chunk = S // world_size
    slc = slice(rank * chunk, (rank + 1) * chunk)

    for causal in (False, True):
        # 单卡参考：全序列 attention 后取本地 Q chunk 切片。
        # 注意不能用 F.sdpa(q_chunk, full_k, full_v, is_causal=True) 当参考——
        # torch 的 is_causal 在 q_len ≠ kv_len 时按左上角对齐（等价于假设
        # chunk 位于序列开头），对 rank>0 的 chunk 会错误掩码（G4）。
        ref_full = F.scaled_dot_product_attention(
            full_q, full_k, full_v, is_causal=causal)
        ref = ref_full[:, :, slc]

        attn = ToyNeMoAttention()
        # 显式声明：NeMo 风格 (q,k,v) 签名 → "sdpa_qkv" wrapper（D-04 mask
        # 内置）；inner 子模块重包声明（输出布局 == q 布局，layout-preserving）
        spec = ModuleShardingSpec(inner_target="inner_attention",
                                  inner_wrapper="sdpa_qkv",
                                  inner_out_src="first_input", region_dispatch=False)
        _wrap_inner_attention(attn, cp_mesh, spec=spec)
        out = attn(full_q[:, :, slc].contiguous(),
                   full_k[:, :, slc].contiguous(),
                   full_v[:, :, slc].contiguous(),
                   is_causal=causal)
        # N4（causal 时）：rank1 的 Q 全局位置 [S/2, S) 必须 attend 到 [0, 位置]
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_cp_qkv_wrapper_2proc():
    run_dist(2, _worker__s3_cp_qkv_wrapper)


# ==========================================================================
# 来源: test_dist_s3_phase_c_cp.py
# S3.5（2 进程）: _apply_phase_c CP 分支 — 双模式注入同一 wrapper。
# ==========================================================================

def _worker__s3_phase_c_cp(rank, world_size):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("cp",))
    cp_mesh = mesh["cp"]

    torch.manual_seed(1234)
    ref_model = TinyLlamaForCausalLM(TinyConfig(), causal=True).eval()
    torch.manual_seed(7)
    input_ids = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref = ref_model(input_ids)

    chunk = 8 // world_size
    slc = slice(rank * chunk, (rank + 1) * chunk)

    for mode in ("production", "validate"):
        torch.manual_seed(1234)
        model = TinyLlamaForCausalLM(TinyConfig(), causal=True).eval()
        # 保存原始 attention forward 引用，验证注入发生
        orig_fwd = model.model.layers[0].self_attn.forward
        # 显式注入：HF 风格 attention → "sdpa_hf" CP wrapper
        planner = ShardingPlanner(plan_overrides=cp_sdpa_hf_injection())
        plan = planner.plan(model, mesh, tp_size=1, cp_size=world_size)
        model, _ = apply_sharding_plan(model, plan, mesh,
                                       validate_mode=(mode == "validate"))
        attn = model.model.layers[0].self_attn
        # 两模式均注入了 CP wrapper（forward 已被替换）
        assert attn.forward != orig_fwd
        batch = shard_batch_for_cp({"input_ids": input_ids}, cp_mesh)
        with torch.no_grad():
            out = model(batch["input_ids"])
        # D-07：lm_head 输出为本 rank CP chunk 的 logits，逐 rank 对拍
        torch.testing.assert_close(out, ref[:, slc], rtol=1e-5, atol=1e-5)


def _worker_no_cp_no_inject(rank, world_size):
    """cp 轴 size=1 → planner 过滤后无 cp 维，不注入 CP wrapper。"""
    mesh = init_device_mesh("cpu", (1, world_size), mesh_dim_names=("cp", "tp"))
    torch.manual_seed(1234)
    model = TinyLlamaForCausalLM(TinyConfig()).eval()
    orig_fwd = model.model.layers[0].self_attn.forward
    plan = ShardingPlanner().plan(model, mesh, tp_size=world_size, cp_size=1)
    assert "cp" not in plan.mesh_dim_names
    apply_sharding_plan(model, plan, mesh)
    # 只被 production boundary 包装了一层（__wrapped__ 即原始 forward；
    # bound method 每次访问生成新对象，用 == 比较而非 is）。
    assert model.model.layers[0].self_attn.forward.__wrapped__ == orig_fwd


def test_phase_c_cp_dual_mode_2proc():
    run_dist(2, _worker__s3_phase_c_cp)


def test_phase_c_cp_size1_no_inject():
    run_dist(2, _worker_no_cp_no_inject)


# ==========================================================================
# 来源: test_dist_s3_tp_cp_e2e.py
# S3.8（4 进程）: TP=2×CP=2 组合端到端 + R8（boundary 无 CP 维非 identity op）。
# ==========================================================================

def _worker__s3_tp_cp_e2e(rank, world_size):
    assert world_size == 4
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("cp", "tp"))
    cp_mesh = mesh["cp"]

    torch.manual_seed(1234)
    ref_model = TinyLlamaForCausalLM(TinyConfig(), causal=True).eval()
    torch.manual_seed(7)
    input_ids = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref = ref_model(input_ids)

    # R8：所有 boundary 的 in/out plan 无 CP 维非 identity op
    model = TinyLlamaForCausalLM(TinyConfig(), causal=True).eval()
    plan = ShardingPlanner().plan(model, mesh, tp_size=2, cp_size=2)
    assert plan.mesh_dim_names == ("cp", "tp")
    cp_idx = plan.mesh_dim_names.index("cp")
    for fqn, spec in plan.modules.items():
        b = PrecompiledBoundary(spec, mesh, plan.mesh_dim_names)
        for op in list(b.in_plan) + list(b.out_plan):
            assert op.src_placements[cp_idx] == op.dst_placements[cp_idx], (
                f"{fqn} 的 boundary 出现 CP 维非 identity op（R8 违反）")

    # TP×CP production 端到端（causal，覆盖 G4）；CP wrapper 显式注入
    cp_rank = cp_mesh.get_local_rank()
    chunk = 8 // 2  # cp_size=2
    slc = slice(cp_rank * chunk, (cp_rank + 1) * chunk)
    for mode in ("production", "validate"):
        torch.manual_seed(1234)
        model = TinyLlamaForCausalLM(TinyConfig(), causal=True).eval()
        planner = ShardingPlanner(plan_overrides=cp_sdpa_hf_injection())
        plan = planner.plan(model, mesh, tp_size=2, cp_size=2)
        model, _ = apply_sharding_plan(model, plan, mesh,
                                       validate_mode=(mode == "validate"))
        batch = shard_batch_for_cp({"input_ids": input_ids}, cp_mesh)
        with torch.no_grad():
            out = model(batch["input_ids"])
        # D-07：lm_head 不做 CP gather——输出为本 rank CP chunk 的 logits
        # （vocab 全量），逐 rank 与单卡参考对应切片对拍
        torch.testing.assert_close(out, ref[:, slc], rtol=1e-5, atol=1e-5)


def test_tp_cp_e2e_4proc():
    run_dist(4, _worker__s3_tp_cp_e2e)
