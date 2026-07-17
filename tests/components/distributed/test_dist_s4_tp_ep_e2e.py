# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S4.5（4 进程）: TP=2×EP=2 组合端到端（双模式）。"""

import torch

from hyper_models.components.distributed import ShardingPlanner, apply_sharding_plan
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyLlamaForCausalLM,
    run_dist,
)
from tests.components.distributed.test_dist_s4_moe_local_map import _attach_ep


def _worker(rank, world_size):
    assert world_size == 4
    torch.manual_seed(1234)
    ref = TinyLlamaForCausalLM(TinyConfig(num_experts=4)).eval()
    torch.manual_seed(7)
    input_ids = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref_logits = ref(input_ids)

    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("tp", "ep"))
    for mode in ("production", "validate"):
        torch.manual_seed(1234)
        model = TinyLlamaForCausalLM(TinyConfig(num_experts=4)).eval()
        plan = ShardingPlanner().plan(model, mesh, tp_size=2, ep_size=2)
        model, _ = apply_sharding_plan(model, plan, mesh,
                                       validate_mode=(mode == "validate"))
        _attach_ep(model, mesh, 2)
        with torch.no_grad():
            out = model(input_ids)
        # lm_head 全量输出（无 CP），逐 rank 与单卡参考对拍
        torch.testing.assert_close(out, ref_logits, rtol=1e-5, atol=1e-5)


def test_tp_ep_e2e_4proc():
    run_dist(4, _worker)
