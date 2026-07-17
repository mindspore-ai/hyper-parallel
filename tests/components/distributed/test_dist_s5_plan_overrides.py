# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S5.7（2 进程）: plan_overrides 端到端 —— 自研多输入 attention 双模式等价（05 §3.6.7）。

自研模型场景：attention forward 签名为 (attn_bias, x)，被切张量 x 既非
首个位置参数、也不叫 hidden_states——模板默认 key 经签名绑定 miss、
单 op 位置兜底错绑到下标 0。用 plan_overrides 把契约 key 改为 "x"，
production/validate 双模式输出均须等价单卡参考。
"""

import copy

import torch
import torch.nn as nn

from hyper_models.components.distributed import ShardingPlanner, apply_sharding_plan
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyLlamaAttention,
    TinyLlamaForCausalLM,
    TinyLlamaMLP,
    TinyLlamaModel,
    TinyRMSNorm,
    run_dist,
)


class MultiInputAttention(TinyLlamaAttention):
    """forward(self, attn_bias, x, ...)：x 在位置 1 且名字非 hidden_states。"""

    def forward(self, attn_bias, x, position_ids=None):
        return TinyLlamaAttention.forward(self, x, position_ids)


class MultiInputDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = TinyRMSNorm(config.hidden_size)
        self.self_attn = MultiInputAttention(config)
        self.post_attention_layernorm = TinyRMSNorm(config.hidden_size)
        self.mlp = TinyLlamaMLP(config)

    def forward(self, hidden_states, position_ids=None):
        hidden_states = hidden_states + self.self_attn(
            None, self.input_layernorm(hidden_states), position_ids)
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states))
        return hidden_states


class MultiInputModel(TinyLlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            MultiInputDecoderLayer(config)
            for _ in range(config.num_hidden_layers))


class MultiInputForCausalLM(TinyLlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = MultiInputModel(config)


def _build():
    torch.manual_seed(1234)
    return MultiInputForCausalLM(TinyConfig()).eval()


def _worker(rank, world_size):
    ref_model = _build()
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref = ref_model(x)

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    for mode in ("production", "validate"):
        model = _build()
        # 先推导一次拿模板填充的 spec，把契约 key 改为真实签名参数名后作为
        # override 回注——用户只需关心 key，placement 沿用模板推导结果
        base_plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
        overrides = {}
        for fqn, spec in base_plan.modules.items():
            if fqn.endswith("self_attn"):
                new_spec = copy.deepcopy(spec)
                for attr in ("in_src", "in_dst"):
                    d = getattr(new_spec, attr)
                    d["x"] = d.pop("hidden_states")
                overrides[fqn] = new_spec
        assert len(overrides) == 2

        planner = ShardingPlanner(plan_overrides=overrides)
        plan = planner.plan(model, mesh, tp_size=world_size)
        assert set(plan.modules["model.layers.0.self_attn"].in_src) == {"x"}

        model, _ = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))
        with torch.no_grad():
            out = model(x)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_plan_overrides_multi_input_e2e_tp2():
    run_dist(2, _worker)
