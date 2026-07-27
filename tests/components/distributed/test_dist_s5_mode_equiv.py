# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S5.4（2~4 进程）: M_M.2 — validate vs production 输出等价（三组合）。"""

import torch

from hyper_models.components.distributed import ShardingPlanner, apply_sharding_plan
from hyper_models.components.distributed.cp_utils import shard_batch_for_cp
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyLlamaForCausalLM,
    run_dist,
)
from tests.components.distributed.test_dist_s4_moe_local_map import _attach_ep


def _dual_run(model_builder, plan_kwargs, mesh, input_ids, cp_mesh=None,
              attach_ep_fn=None):
    """同 batch 两模式各跑一次 forward，返回两模式输出。"""
    outs = {}
    for mode in ("production", "validate"):
        model = model_builder()
        plan = ShardingPlanner().plan(model, mesh, **plan_kwargs)
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
                     cp_mesh=mesh["cp"])
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
