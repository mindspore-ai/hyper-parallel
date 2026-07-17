# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S4.4（2 进程）: D-03' — validate 下 MoE 经 local region 缝合，
边界 DTensor 契约保持（out_src 声明式），下游链式校验不断链。"""

import torch

from hyper_models.components.distributed import ShardingPlanner, apply_sharding_plan
from hyper_models.components.distributed.precompiled_boundary import (
    PrecompiledBoundary,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyLlamaForCausalLM,
    run_dist,
)
from tests.components.distributed.test_dist_s4_moe_local_map import _attach_ep


def _worker(rank, world_size):
    torch.manual_seed(1234)
    ref = TinyLlamaForCausalLM(TinyConfig(num_experts=4)).eval()
    torch.manual_seed(7)
    input_ids = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref_logits = ref(input_ids)

    mesh = init_device_mesh("cpu", (1, world_size), mesh_dim_names=("tp", "ep"))
    torch.manual_seed(1234)
    model = TinyLlamaForCausalLM(TinyConfig(num_experts=4)).eval()
    plan = ShardingPlanner().plan(model, mesh, tp_size=1, ep_size=world_size)
    moe_spec = plan.modules["model.layers.0.mlp"]
    assert moe_spec.use_local_map
    model, _ = apply_sharding_plan(model, plan, mesh, validate_mode=True)
    _attach_ep(model, mesh, world_size)

    # 边界缝合观测：包装 redistribute_outputs，记录 MoE 出口 tensor 类型
    seen = []
    orig_redistribute_outputs = PrecompiledBoundary.redistribute_outputs

    def spy(self, outputs, *, as_dtensor_input=False):
        if as_dtensor_input:
            seen.append(isinstance(outputs, DTensor))
        return orig_redistribute_outputs(
            self, outputs, as_dtensor_input=as_dtensor_input)

    PrecompiledBoundary.redistribute_outputs = spy
    try:
        with torch.no_grad():
            out = model(input_ids)
    finally:
        PrecompiledBoundary.redistribute_outputs = orig_redistribute_outputs

    # MoE 边界出口是 DTensor（local region 按声明 out_src 重包装）
    assert any(seen), "validate 下未观测到 DTensor 边界出口"
    # 数值与单卡参考一致（链不断、可对拍）
    torch.testing.assert_close(out, ref_logits, rtol=1e-5, atol=1e-5)

    # 下游 norm 的 in_src 校验不受断链影响：模型完整跑通即证明
    # （in_src 校验在链式传播阶段已完成；validate forward 未抛错）


def test_moe_validate_region_ep2():
    run_dist(2, _worker)
