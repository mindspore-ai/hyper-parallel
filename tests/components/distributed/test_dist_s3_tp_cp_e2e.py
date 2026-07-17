# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S3.8（4 进程）: TP=2×CP=2 组合端到端 + R8（boundary 无 CP 维非 identity op）。"""

import torch

from hyper_models.components.distributed import (
    ShardingPlanner,
    apply_sharding_plan,
)
from hyper_models.components.distributed.cp_utils import shard_batch_for_cp
from hyper_models.components.distributed.precompiled_boundary import (
    PrecompiledBoundary,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyLlamaForCausalLM,
    run_dist,
)


def _worker(rank, world_size):
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

    # TP×CP production 端到端（causal，覆盖 G4）
    cp_rank = cp_mesh.get_local_rank()
    chunk = 8 // 2  # cp_size=2
    slc = slice(cp_rank * chunk, (cp_rank + 1) * chunk)
    for mode in ("production", "validate"):
        torch.manual_seed(1234)
        model = TinyLlamaForCausalLM(TinyConfig(), causal=True).eval()
        plan = ShardingPlanner().plan(model, mesh, tp_size=2, cp_size=2)
        model, _ = apply_sharding_plan(model, plan, mesh,
                                       validate_mode=(mode == "validate"))
        batch = shard_batch_for_cp({"input_ids": input_ids}, cp_mesh)
        with torch.no_grad():
            out = model(batch["input_ids"])
        # D-07：lm_head 不做 CP gather——输出为本 rank CP chunk 的 logits
        # （vocab 全量），逐 rank 与单卡参考对应切片对拍
        torch.testing.assert_close(out, ref[:, slc], rtol=1e-5, atol=1e-5)


def test_tp_cp_e2e_4proc():
    run_dist(4, _worker)
