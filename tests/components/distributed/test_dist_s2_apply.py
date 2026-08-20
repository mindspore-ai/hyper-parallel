# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_dist_s2_apply.py: 核心套件合并文件。

来源: test_dist_s2_apply_e2e.py, test_dist_s2_shard_params.py, test_dist_s2_local_params.py, test_dist_s2_tied.py, test_dist_s2_validate_fwd.py, test_dist_s2_production_fwd.py, test_dist_s2_redist_op.py, test_dist_s2_head_count.py
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from hyper_models.components.distributed import (
    ShardingPlanner,
    apply_sharding_plan,
)
from hyper_models.components.distributed.precompiled_boundary import (
    RedistOp,
    _classify_collective,
)
from hyper_models.components.distributed.sharding.apply import _local_params_context
from hyper_models.components.distributed.sharding_applier import (
    _shard_module_params,
    detect_tied_weights,
)
from hyper_models.components.distributed.sharding_config import (
    CP,
    EP,
    ModuleShardingSpec,
    PlacementMismatchError,
    TP,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Replicate,
    Shard,
)
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyLlamaAttention,
    TinyLlamaForCausalLM,
    run_dist,
)


# ==========================================================================
# 来源: test_dist_s2_apply_e2e.py
# S2.11（2 进程）: apply_sharding_plan 主入口 — TP=2 双模式端到端。
# ==========================================================================

def _build():
    torch.manual_seed(1234)
    return TinyLlamaForCausalLM(TinyConfig()).eval()


def _worker(rank, world_size):
    ref_model = _build()
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref = ref_model(x)

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    outs = {}
    for mode in ("production", "validate"):
        model = _build()
        plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
        model, tp_grad_info = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))
        # 返回二元组结构
        assert isinstance(model, nn.Module)
        if mode == "production":
            assert tp_grad_info is not None
            # tp_grad_info 覆盖所有 spec 参数
            assert "model.layers.0.self_attn.q_proj.weight" in tp_grad_info
            assert tp_grad_info["model.layers.0.self_attn.q_proj.weight"][0] == Shard(0)
            assert tp_grad_info["model.layers.0.input_layernorm.weight"][0] == Replicate()
        else:
            assert tp_grad_info is None
        with torch.no_grad():
            outs[mode] = model(x)

    # 双模式端到端：production == validate == 单卡参考（逐 rank 断言）
    torch.testing.assert_close(outs["production"], ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(outs["validate"], ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(outs["production"], outs["validate"],
                               rtol=1e-5, atol=1e-5)


def _worker_list_model(rank, world_size):
    """list[nn.Module] 支持（PP 多 part 形式）。"""
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = _build()
    plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
    result, info = apply_sharding_plan([model], plan, mesh)
    assert isinstance(result, list) and len(result) == 1


def test_apply_e2e_dual_mode_tp2():
    run_dist(2, _worker)


def test_apply_list_model_2proc():
    run_dist(2, _worker_list_model)


# ==========================================================================
# 来源: test_dist_s2_shard_params.py
# S2.2（2 进程）: Phase A _shard_module_params。
# ==========================================================================

def _worker_shard_params(rank, world_size):
    torch.manual_seed(0)
    lin = nn.Linear(16, 16, bias=False)
    full_q = lin.weight.detach().clone()
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    # q_proj 风格 Shard(0)
    _shard_module_params(lin, {"weight": {TP: Shard(0), CP: Replicate()}},
                         mesh, ("tp",))
    assert isinstance(lin.weight, DTensor)
    local = lin.weight.to_local()
    assert local.shape == (16 // world_size, 16)
    # N1: 逐 rank 断言本 rank 的 local 切片 == 全量对应切片
    chunk = 16 // world_size
    torch.testing.assert_close(local, full_q[rank * chunk:(rank + 1) * chunk, :])
    assert lin.weight.requires_grad

    # o_proj 风格 Shard(1)（N2）
    lin2 = nn.Linear(16, 16, bias=False)
    full_o = lin2.weight.detach().clone()
    _shard_module_params(lin2, {"weight": {TP: Shard(1), CP: Replicate()}},
                         mesh, ("tp",))
    local2 = lin2.weight.to_local()
    assert local2.shape == (16, 16 // world_size)
    torch.testing.assert_close(local2, full_o[:, rank * chunk:(rank + 1) * chunk])


def _worker_ep_shard(rank, world_size):
    torch.manual_seed(0)
    holder = nn.Module()
    holder.experts = nn.Module()
    holder.experts.w1 = nn.Parameter(torch.randn(4, 8, 16))
    full = holder.experts.w1.detach().clone()
    mesh = init_device_mesh("cpu", (1, world_size), mesh_dim_names=("tp", "ep"))

    _shard_module_params(
        holder, {"experts.w1": {EP: Shard(0), TP: Shard(0), CP: Replicate()}},
        mesh, ("tp", "ep"))
    local = holder.experts.w1.to_local()
    # N7 变体：EP 沿 expert 维切，TP 沿 dim0（expert 维已被 EP 切后为局部）
    assert local.shape == (4 // world_size, 8, 16)
    chunk_e = 4 // world_size
    torch.testing.assert_close(local, full[rank * chunk_e:(rank + 1) * chunk_e])


def _worker_already_dtensor(rank, world_size):
    torch.manual_seed(0)
    lin = nn.Linear(8, 8, bias=False)
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    spec = {"weight": {TP: Shard(0)}}
    _shard_module_params(lin, spec, mesh, ("tp",))
    # placement 一致 → 幂等跳过
    _shard_module_params(lin, spec, mesh, ("tp",))
    # placement 不一致 → 抛错
    with pytest.raises(PlacementMismatchError):
        _shard_module_params(lin, {"weight": {TP: Shard(1)}}, mesh, ("tp",))


def _worker_meta(rank, world_size):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    with torch.device("meta"):
        lin = nn.Linear(16, 16, bias=False)
    _shard_module_params(lin, {"weight": {TP: Shard(0)}}, mesh, ("tp",))
    assert isinstance(lin.weight, DTensor)
    assert lin.weight.to_local().is_meta
    assert lin.weight.to_local().shape == (16 // world_size, 16)


def test_shard_params_tp2():
    run_dist(2, _worker_shard_params)


def test_ep_shard_2proc():
    run_dist(2, _worker_ep_shard)


def test_already_dtensor_2proc():
    run_dist(2, _worker_already_dtensor)


def test_meta_tensor_2proc():
    run_dist(2, _worker_meta)


# ==========================================================================
# 来源: test_dist_s2_local_params.py
# S2.8（2 进程）: _local_params_context 零拷贝解包 + placement 快照。
# ==========================================================================

def _worker__s2_local_params(rank, world_size):
    torch.manual_seed(0)
    lin = nn.Linear(16, 16, bias=False)
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    _shard_module_params(lin, {"weight": {TP: Shard(0)}}, mesh, ("tp",))
    dt_local = lin.weight.to_local()
    dt_ptr = dt_local.data_ptr()

    records = _local_params_context(lin)
    # 解包后为 plain tensor
    assert not isinstance(lin.weight, DTensor)
    assert isinstance(lin.weight, nn.Parameter)
    # 零拷贝：data_ptr 与 DTensor._local_tensor 共享存储
    assert lin.weight.data_ptr() == dt_ptr
    # requires_grad 保留
    assert lin.weight.requires_grad
    # placement 快照
    assert "weight" in records
    assert tuple(records["weight"]) == (Shard(0),)
    # 数值不变
    torch.testing.assert_close(lin.weight.data, dt_local)

    # 无 DTensor 时 no-op
    records2 = _local_params_context(lin)
    assert records2 == {}


def test_local_params_context_2proc():
    run_dist(2, _worker__s2_local_params)


# ==========================================================================
# 来源: test_dist_s2_tied.py
# S2.10（2 进程）: Phase D tied weights（detect/broadcast/replicate）。
# ==========================================================================

def _worker_tied(rank, world_size):
    torch.manual_seed(1234)
    cfg = TinyConfig(tie_word_embeddings=True)
    model = TinyLlamaForCausalLM(cfg)

    tied = detect_tied_weights(model)
    assert tied == [("model.embed_tokens.weight", "lm_head.weight")]

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
    assert plan.tied_pairs == [("model.embed_tokens.weight", "lm_head.weight")]
    model, _ = apply_sharding_plan(model, plan, mesh)

    # N9：rank1 是接收端（src=0）——每个 rank 上两端 local 存储逐元素一致
    emb_w = model.model.embed_tokens.weight
    lm_w = model.lm_head.weight
    torch.testing.assert_close(emb_w.data, lm_w.data)
    # 且与全量切片的本地段一致（tied 内容 == 原始 embed 权重切片）
    torch.manual_seed(1234)
    ref = TinyLlamaForCausalLM(cfg)
    chunk = cfg.vocab_size // world_size
    torch.testing.assert_close(
        emb_w.data,
        ref.model.embed_tokens.weight.data[rank * chunk:(rank + 1) * chunk],
    )


def _worker_not_tied(rank, world_size):
    torch.manual_seed(1234)
    model = TinyLlamaForCausalLM(TinyConfig(tie_word_embeddings=False))
    assert detect_tied_weights(model) == []
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
    assert plan.tied_pairs == []


def test_tied_weights_2proc():
    run_dist(2, _worker_tied)


def test_not_tied_2proc():
    run_dist(2, _worker_not_tied)


# ==========================================================================
# 来源: test_dist_s2_validate_fwd.py
# S2.7（2 进程）: _wrap_validate_forward — 正确 plan 全 pass + 错误声明抛错。
# ==========================================================================

def _build__s2_validate_fwd():
    torch.manual_seed(1234)
    return TinyLlamaForCausalLM(TinyConfig()).eval()


def _worker_pass(rank, world_size):
    ref_model = _build__s2_validate_fwd()
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref = ref_model(x)

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = _build__s2_validate_fwd()
    plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
    model, tp_grad_info = apply_sharding_plan(model, plan, mesh, validate_mode=True)
    assert tp_grad_info is None  # validate 模式无 tp_grad_info
    with torch.no_grad():
        out = model(x)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def _worker_mismatch(rank, world_size):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = _build__s2_validate_fwd()
    plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
    # 故意把 out_src 声明为 Replicate（DTensor 传播实际产出 Partial）
    plan.modules["model.layers.0.self_attn"].out_src = {
        "output": {TP: Replicate()}}
    model, _ = apply_sharding_plan(model, plan, mesh, validate_mode=True)
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    with pytest.raises(PlacementMismatchError) as exc:
        model(x)
    assert exc.value.stage.startswith("out_src")


def _worker_param_mismatch(rank, world_size):
    """参数声明错误（o_proj 应为 Shard(1) 误写 Shard(0)）→ dispatch 层
    layout 校验即拦截（ValueError）——比 out_src 校验更早暴露。"""
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = _build__s2_validate_fwd()
    plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
    plan.modules["model.layers.0.self_attn"].params["o_proj.weight"] = {TP: Shard(0)}
    model, _ = apply_sharding_plan(model, plan, mesh, validate_mode=True)
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    with pytest.raises(ValueError, match="layout"):
        model(x)


def _worker_terminal_out_dst(rank, world_size):
    """terminal 模块（lm_head）的 out_dst 校验函数：声明与实际 DTensor 传播
    placement 不一致 → 抛 PlacementMismatchError(stage="out_dst")。

    注：端到端路径下 boundary.redistribute_outputs 以声明的 out_dst 为目标，
    产出恒等于声明——out_dst 校验是防御性的，此处直接对校验函数构造不一致。
    """
    from hyper_models.components.distributed.sharding_applier import (
        _validate_out_dst,
    )
    from hyper_models.components.distributed.sharding_config import (
        ModuleShardingSpec,
    )
    from hyper_parallel.core.dtensor.dtensor import DTensor

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    spec = ModuleShardingSpec(
        out_dst={"output": {TP: Shard(1)}},
    )
    spec._is_terminal = True
    dt = DTensor.from_local(torch.randn(2, 4), mesh, (Replicate(),))
    with pytest.raises(PlacementMismatchError) as exc:
        _validate_out_dst(dt, spec, ("tp",), "Linear")
    assert exc.value.stage.startswith("out_dst")

    # 一致 → 不抛
    spec2 = ModuleShardingSpec(out_dst={"output": {TP: Replicate()}})
    spec2._is_terminal = True
    _validate_out_dst(dt, spec2, ("tp",), "Linear")


def test_validate_forward_pass_tp2():
    run_dist(2, _worker_pass)


def test_validate_forward_out_src_mismatch_tp2():
    run_dist(2, _worker_mismatch)


def test_validate_forward_param_mismatch_tp2():
    run_dist(2, _worker_param_mismatch)


def test_validate_terminal_out_dst_tp2():
    run_dist(2, _worker_terminal_out_dst)


# ==========================================================================
# 来源: test_dist_s2_production_fwd.py
# S2.6（2 进程）: _wrap_production_forward — tiny_llama attention+mlp TP=2 数值。
# ==========================================================================

def _build__s2_production_fwd():
    torch.manual_seed(1234)
    return TinyLlamaForCausalLM(TinyConfig()).eval()


def _worker__s2_production_fwd(rank, world_size):
    # 单卡参考
    ref_model = _build__s2_production_fwd()
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref = ref_model(x)

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = _build__s2_production_fwd()
    plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
    model, _ = apply_sharding_plan(model, plan, mesh, validate_mode=False)
    with torch.no_grad():
        out = model(x)
    # 逐 rank 断言（输出为完整 logits——lm_head out_dst=Replicate）
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_production_forward_tp2():
    run_dist(2, _worker__s2_production_fwd)


# ==========================================================================
# 来源: test_dist_s2_redist_op.py
# S2.3（2 进程）: RedistOp.execute + _classify_collective 五组合数值。
# ==========================================================================

def _classify_cases():
    R, P, S = Replicate(), Partial(), Shard(1)
    assert _classify_collective((S,), (S,)) == "identity"
    assert _classify_collective((S,), (R,)) == "all_gather"
    assert _classify_collective((P,), (S,)) == "reduce_scatter"
    assert _classify_collective((P,), (R,)) == "all_reduce"
    assert _classify_collective((R,), (S,)) == "redistribute"


def _worker__s2_reop(rank, world_size):
    _classify_cases()
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    torch.manual_seed(42)
    full = torch.randn(4, 8)
    chunk = 8 // world_size

    # 1. identity：直通零通信
    op = RedistOp("x", None, mesh, (Replicate(),), (Replicate(),), "identity")
    out = op.execute(full)
    assert out is full
    out_dt = op.execute(full, as_dtensor=True)
    assert isinstance(out_dt, DTensor)

    # 2. Shard→Replicate（all_gather）：本 rank 切片 → 全量
    local = full[:, rank * chunk:(rank + 1) * chunk].contiguous()
    op = RedistOp("x", None, mesh, (Shard(1),), (Replicate(),), "all_gather")
    out = op.execute(local)
    torch.testing.assert_close(out, full)

    # 3. Partial→Shard（reduce_scatter）：各 rank 持有不同的 partial 贡献
    # partial_i = full * (rank+1)；reduce_scatter 后每 rank 拿 sum 的第 rank 块
    partial = full * (rank + 1)
    expect_sum = full * sum(range(1, world_size + 1))
    op = RedistOp("x", None, mesh, (Partial(),), (Shard(1),), "reduce_scatter")
    out = op.execute(partial)
    # N3：逐 rank 断言本 rank 拿到输出第 rank 块且数值 = 各块之和
    torch.testing.assert_close(out, expect_sum[:, rank * chunk:(rank + 1) * chunk])

    # 4. Partial→Replicate（all_reduce）
    op = RedistOp("x", None, mesh, (Partial(),), (Replicate(),), "all_reduce")
    out = op.execute(partial)
    torch.testing.assert_close(out, expect_sum)

    # 5. Replicate→Shard（redistribute / 切片）
    op = RedistOp("x", None, mesh, (Replicate(),), (Shard(1),), "redistribute")
    out = op.execute(full)
    torch.testing.assert_close(out, full[:, rank * chunk:(rank + 1) * chunk])

    # as_dtensor=True 路径：返回 DTensor 且 placement 为 dst
    out_dt = op.execute(full, as_dtensor=True)
    assert isinstance(out_dt, DTensor)
    assert tuple(out_dt.placements) == (Shard(1),)
    torch.testing.assert_close(
        out_dt.to_local(), full[:, rank * chunk:(rank + 1) * chunk])


def test_redist_op_2proc():
    run_dist(2, _worker__s2_reop)


# ==========================================================================
# 来源: test_dist_s2_head_count.py
# S2.8（2 进程）: D-17 头数改写端到端 —— 非 TP 容错 attention（reshape 显式
# ==========================================================================

_HEADS = TinyConfig().num_attention_heads   # 4


class ExplicitHeadsAttention(TinyLlamaAttention):
    """非 TP 容错写法：reshape 显式使用 self.num_heads（HF 生态常见写法）。"""

    def forward(self, hidden_states, position_ids=None):
        b, s, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(b, s, self.num_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        o = o.transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(o)


def _build__s2_head_count():
    torch.manual_seed(1234)
    model = TinyLlamaForCausalLM(TinyConfig()).eval()
    for layer in model.model.layers:
        attn = ExplicitHeadsAttention(model.config)
        attn.load_state_dict(layer.self_attn.state_dict())
        layer.self_attn = attn
    return model


def _reference():
    ref_model = _build__s2_head_count()
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref = ref_model(x)
    return x, ref


def _local_region_attn_spec():
    """契约与 planner 派生的 attention spec 一致（SP 默认开启），仅多声明
    region_dispatch 使模块走 local-region。"""
    return ModuleShardingSpec(
        params={
            "q_proj.weight": {TP: Shard(0)},
            "k_proj.weight": {TP: Shard(0)},
            "v_proj.weight": {TP: Shard(0)},
            "o_proj.weight": {TP: Shard(1)},
        },
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Replicate()}},
        out_src={TP: Partial()},
        out_dst={TP: Shard(1)},
        region_dispatch=False,
    )


def _worker_production(rank, world_size):
    """production：头数改写为本地值，显式 num_heads reshape 数值对齐。"""
    x, ref = _reference()

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = _build__s2_head_count()
    plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
    model, _ = apply_sharding_plan(model, plan, mesh, validate_mode=False)
    for layer in model.model.layers:
        assert layer.self_attn.num_heads == _HEADS // world_size
        assert layer.self_attn.config.num_attention_heads == _HEADS  # config 不改写
    with torch.no_grad():
        out = model(x)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def _worker_validate_boundary(rank, world_size):
    """validate（boundary）：不改写属性，DTensor 全局逻辑形状自动推导。"""
    x, ref = _reference()

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = _build__s2_head_count()
    plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
    model, _ = apply_sharding_plan(model, plan, mesh, validate_mode=True)
    for layer in model.model.layers:
        # 关键断言：属性保持全局值，未做任何手动/自动改写
        assert layer.self_attn.num_heads == _HEADS
        assert not hasattr(layer.self_attn, "_hp_full_head_counts")
    with torch.no_grad():
        out = model(x)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def _worker_validate_local_region(rank, world_size):
    """validate（local-region）：区域内 local tensor → 改写且数值对齐。"""
    x, ref = _reference()

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = _build__s2_head_count()
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.self_attn": _local_region_attn_spec(),
        "model.layers.1.self_attn": _local_region_attn_spec(),
    })
    plan = planner.plan(model, mesh, tp_size=world_size)
    model, _ = apply_sharding_plan(model, plan, mesh, validate_mode=True)
    for layer in model.model.layers:
        assert layer.self_attn.num_heads == _HEADS // world_size
    with torch.no_grad():
        out = model(x)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_head_count_production_tp2():
    run_dist(2, _worker_production)


def test_head_count_validate_boundary_tp2():
    run_dist(2, _worker_validate_boundary)


def test_head_count_validate_local_region_tp2():
    run_dist(2, _worker_validate_local_region)
