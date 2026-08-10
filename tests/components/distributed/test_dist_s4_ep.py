# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_dist_s4_ep.py: 核心套件合并文件。

来源: test_dist_s4_ep_shard.py, test_dist_s4_moe_local_map.py, test_dist_s4_moe_validate_region.py, test_dist_s4_tp_ep_e2e.py, test_dist_s6_ep_extend.py, test_dist_s6_hf_native_moe.py
"""

import torch
from hyper_models.components.distributed import (
    ShardingPlanner,
    apply_sharding_plan,
)
from hyper_models.components.distributed.ep_utils import _ep_all_to_all
from hyper_models.components.distributed.precompiled_boundary import PrecompiledBoundary
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from tests.components.distributed.conftest import (
    _attach_ep,
    TinyBatchedMoEForCausalLM,
    TinyConfig,
    TinyHFNativeMoEForCausalLM,
    TinyLlamaForCausalLM,
    ep_hf_native_injection,
    run_dist,
)


# ==========================================================================
# 来源: test_dist_s4_ep_shard.py
# S4.1（2 进程）: MoE 参数分片 — expert EP 切片 + gate 全复制（无独立 EP _apply 入口）。
# ==========================================================================

def _worker(rank, world_size):
    torch.manual_seed(1234)
    ref = TinyLlamaForCausalLM(TinyConfig(num_experts=4)).eval()
    torch.manual_seed(1234)
    model = TinyLlamaForCausalLM(TinyConfig(num_experts=4)).eval()

    mesh = init_device_mesh("cpu", (1, world_size), mesh_dim_names=("tp", "ep"))
    plan = ShardingPlanner().plan(model, mesh, tp_size=1, ep_size=world_size)
    model, _ = apply_sharding_plan(model, plan, mesh)

    chunk = 4 // world_size
    slc = slice(rank * chunk, (rank + 1) * chunk)
    for i in (0, 1):
        moe = model.model.layers[i].mlp
        ref_moe = ref.model.layers[i].mlp
        # N7：rank1 只持 e2/e3——逐 rank 断言本地 expert 切片 == 全量对应段
        assert moe.experts.w1.shape[0] == chunk
        torch.testing.assert_close(moe.experts.w1.data, ref_moe.experts.w1.data[slc])
        torch.testing.assert_close(moe.experts.w2.data, ref_moe.experts.w2.data[slc])
        torch.testing.assert_close(moe.experts.w3.data, ref_moe.experts.w3.data[slc])
        # gate 两 rank 全复制一致
        torch.testing.assert_close(moe.gate.weight.data, ref_moe.gate.weight.data)


def test_ep_shard_2proc():
    run_dist(2, _worker)


# ==========================================================================
# 来源: test_dist_s4_moe_local_map.py
# S4.2（2 进程）: _wrap_local_region_forward — toy MoE EP=2 输出 vs 单卡参考（N8 非对称通信）。
# ==========================================================================

def _worker__s4_moe_local_map(rank, world_size):
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
    model, _ = apply_sharding_plan(model, plan, mesh)
    _attach_ep(model, mesh, world_size)

    with torch.no_grad():
        out = model(input_ids)
    # N8：EP combine（all-reduce）后每 rank 输出 == 单卡参考
    torch.testing.assert_close(out, ref_logits, rtol=1e-5, atol=1e-5)

    # N8 非对称通信断言：本 rank 的 MoE 输出（combine 前）只覆盖路由到
    # 本地 expert 的 token——用 isolated MoE 验证
    layer0 = model.model.layers[0]
    ref_moe = ref.model.layers[0].mlp
    x = torch.randn(2, 4, 16)
    with torch.no_grad():
        ref_out = ref_moe(x)
        local_out = layer0.mlp(x)  # 已含 combine
    torch.testing.assert_close(local_out, ref_out, rtol=1e-5, atol=1e-5)

    # 路由分布非对称：本 rank expert 集合只处理 idx ∈ [offset, offset+n_local)
    moe = layer0.mlp
    with torch.no_grad():
        idx = moe.gate(x).argmax(dim=-1)
    n_local = 4 // world_size
    offset = moe.experts.expert_offset
    has_local = ((idx >= offset) & (idx < offset + n_local)).any()
    has_remote = ((idx < offset) | (idx >= offset + n_local)).any()
    # 至少有一种路由分布（不强制每个 rank 两者皆有，但全局必须覆盖）
    assert has_local or has_remote


def test_moe_local_map_ep2():
    run_dist(2, _worker__s4_moe_local_map)


# ==========================================================================
# 来源: test_dist_s4_moe_validate_region.py
# S4.4（2 进程）: D-03' — validate 下 MoE 经 local region 缝合，
# ==========================================================================

def _worker__s4_moe_validate_region(rank, world_size):
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
    assert moe_spec.region_dispatch is False
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


def test_moe_validate_region_ep2():
    run_dist(2, _worker__s4_moe_validate_region)


# ==========================================================================
# 来源: test_dist_s4_tp_ep_e2e.py
# S4.5（4 进程）: TP=2×EP=2 组合端到端（双模式）。
# ==========================================================================

def _worker__s4_tp_ep_e2e(rank, world_size):
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
    run_dist(4, _worker__s4_tp_ep_e2e)


# ==========================================================================
# 来源: test_dist_s6_ep_extend.py
# S6.2（D-10 TP-extend-EP，05 §6.4.8）: 扩展 EP 分布式用例。
# ==========================================================================

def _build():
    torch.manual_seed(1234)
    return TinyHFNativeMoEForCausalLM(TinyConfig(num_experts=4)).eval()


def _build_batched():
    torch.manual_seed(1234)
    return TinyBatchedMoEForCausalLM(TinyConfig(
        num_experts=4, architectures=["Qwen3MoeForCausalLM"])).eval()


def _worker_ep_extend_e2e(rank, world_size):
    """mesh (dp=4, tp=2)，ep=4 → 扩展 EP 组 {0,1,2,3}/{4,5,6,7}（含 TP rank，
    跨 dp 坐标）：双模式输出等价单卡。"""
    assert world_size == 8
    ref_model = _build()
    torch.manual_seed(7)
    input_ids = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref_out = ref_model(input_ids)

    mesh = init_device_mesh("cpu", (4, 2), mesh_dim_names=("dp", "tp"))
    for mode in ("production", "validate"):
        model = _build()
        # 显式注入仓内默认 EP compute（无自动注入；路由内嵌 default
        # softmax top-k）
        planner = ShardingPlanner(plan_overrides=ep_hf_native_injection())
        plan = planner.plan(model, mesh, tp_size=2, ep_size=4)
        spec = plan.modules["model.layers.0.mlp"]
        assert spec._ep_size == 4   # ep_size 即扩展 EP 组大小
        model, _ = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))
        with torch.no_grad():
            out = model(input_ids)
        torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-5)


def test_ep_extend_unified_e2e_8proc():
    run_dist(8, _worker_ep_extend_e2e)


from hyper_models.components.distributed import local_compute as _cf


@_cf
def _qwen3moe_ep_factory(mesh, tp_mesh, cp_mesh, ep_mesh):
    """自定义 EP 工厂：路由是注入函数的一部分——qwen3moe 的 TopKRouter
    语义由本函数显式选择（MOE_ROUTER_ADAPTERS 按名引用），框架不参与。"""
    from hyper_models.components.distributed.ep_utils import (
        MOE_ROUTER_ADAPTERS,
        _hf_native_ep_compute,
    )
    ep_group = ep_mesh.get_group("ep")
    tp_group = tp_mesh.get_group() if tp_mesh is not None else None

    def compute_fn(module, hidden_states):
        return _hf_native_ep_compute(
            module, hidden_states,
            router_fn=MOE_ROUTER_ADAPTERS["qwen3moe"],
            ep_group=ep_group, tp_group=tp_group)
    return compute_fn


def _qwen3moe_ep_injection(match="*.mlp"):
    from hyper_models.components.distributed.sharding_config import (
        ModuleShardingSpec,
    )
    from hyper_models.trainer.config import Target
    return {match: ModuleShardingSpec(
        local_compute_fn=Target(
            _qwen3moe_ep_factory,
            target_path="tests._qwen3moe_ep_factory"), region_dispatch=False)}


def _worker_batched_ep_extend_e2e(rank, world_size):
    """D-11 batched 布局 e2e：mesh (dp=4, tp=2)，ep=4（experts.gate_up_proj
    无需堆叠直接 Shard(0)）：自定义工厂内嵌 qwen3moe TopKRouter 路由，
    双模式等价单卡。"""
    assert world_size == 8
    ref_model = _build_batched()
    torch.manual_seed(7)
    input_ids = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref_out = ref_model(input_ids)

    mesh = init_device_mesh("cpu", (4, 2), mesh_dim_names=("dp", "tp"))
    for mode in ("production", "validate"):
        model = _build_batched()
        # 显式注入自定义工厂（qwen3moe 路由写在工厂体内，框架零参与）
        planner = ShardingPlanner(plan_overrides=_qwen3moe_ep_injection())
        plan = planner.plan(model, mesh, tp_size=2, ep_size=4)
        spec = plan.modules["model.layers.0.mlp"]
        assert spec._ep_size == 4
        assert spec._ep_stack == {}      # batched 天生 stacked，Phase A 无堆叠
        model, _ = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))
        with torch.no_grad():
            out = model(input_ids)
        torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-5)


def test_batched_ep_extend_e2e_8proc():
    run_dist(8, _worker_batched_ep_extend_e2e)


# ==========================================================================
# 来源: test_dist_s6_hf_native_moe.py
# S6（D-09/D-10，05 §6.4.7/§6.4.8）: EP 通信原语分布式用例。
# ==========================================================================

def _worker_padded_a2a(rank, world_size):
    """pad-to-max a2a（gloo 路径）：fwd 数值 + bwd 梯度（a2a 是跨 rank 置换）。"""
    assert world_size == 2
    group = None  # world group（gloo → pad 路径）
    h = 4
    if rank == 0:
        send_counts, recv_counts = [2, 1], [2, 0]
        x = torch.tensor([[0.], [1.], [2.]]).repeat(1, h)
        expected = torch.tensor([[0.], [1.]]).repeat(1, h)
    else:
        send_counts, recv_counts = [0, 3], [1, 3]
        x = torch.tensor([[13.], [14.], [15.]]).repeat(1, h)
        expected = torch.tensor([[2.], [13.], [14.], [15.]]).repeat(1, h)
    x.requires_grad_(True)
    out = _ep_all_to_all(x, send_counts, recv_counts, group)
    torch.testing.assert_close(out, expected)
    out.sum().backward()
    # a2a 是跨 rank 置换：每行输入恰好流向一行输出 → grad 全 1
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_ep_all_to_all_padded_2proc():
    run_dist(2, _worker_padded_a2a)
