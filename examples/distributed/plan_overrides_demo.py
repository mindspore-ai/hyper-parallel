# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""plan_overrides 全场景示例：一个 YAML 覆盖五种 override 场景。

用法:
    PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/plan_overrides_demo.py

配套 plan_overrides_demo.yaml 覆盖：

| 场景 | YAML 条目 | 验证点（plan 内省断言） |
|---|---|---|
| 1. merge + 注入字段 | `*.self_attn` + local_compute_fn | 注入生效（kernel 计数器），params/契约继承推导 |
| 2. merge + 契约 DSL | `*.mlp` + params `{tp: "shard(N)"}` | DSL 解析为 Placement 对象并字段粒度替换 |
| 3. merge + 显式空 {} | `*.input_layernorm` + `params: {}` | 推导分片被清空（不切参数的纯 I/O 缝合公民） |
| 4. when 激活条件 | `when: cp` + inner_wrapper | cp=1 → 条目跳过（inner_wrapper 不生效；日志可见） |
| 5. insert 完整自声明 | `model.aux`（模板不识别） | 新边界插入 plan，契约驱动真实入口/出口通信 |

模型复用 perf_replacement 的 TinyModel（朴素实现），额外挂一个模板不识
别的自研模块 `model.aux`（insert 场景载体）。全部场景在 TP=2 下与单卡
参考双模式对拍。
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ^ 让 YAML `_target_: perf_kernels.*` 可解析（resolve 发生在 import 时）。

import perf_kernels  # noqa: E402
from perf_replacement import (  # noqa: E402
    TinyModel,
    load_plan_overrides_from_yaml,
)
from hyper_parallel.auto_models.components.distributed import (  # noqa: E402
    ShardingPlanner,
    apply_sharding_plan,
)
from hyper_parallel.auto_models.components.distributed.sharding_config import TP  # noqa: E402
from hyper_parallel.core.dtensor.device_mesh import (  # noqa: E402
    init_device_mesh,
)
from hyper_parallel.core.dtensor.placement_types import (  # noqa: E402
    Replicate,
    Shard,
)


class DemoModel(TinyModel):
    """TinyModel + 一个模板不识别的自研模块 aux（insert 场景载体）。

    aux 位于 layers 之后、final norm 之前——SP 布局下 hidden 在此处为
    Shard(1)，因此 aux 的声明契约会驱动真实通信（入口 gather/出口 scatter）。
    """

    def __init__(self, vocab=64, h=32, n_heads=4, n_layers=2):
        super().__init__(vocab, h, n_heads, n_layers)
        self.model.aux = nn.Linear(h, h, bias=False)

    def forward(self, input_ids):
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        h = self.model.aux(h)
        return self.lm_head(self.model.norm(h))


def check_plan_scenarios(plan):
    """逐场景的 plan 内省断言（apply 前——plan 是声明的物化结果）。"""
    # 场景 1：merge + 注入字段（params/契约继承推导）
    attn = plan.modules["model.layers.0.self_attn"]
    assert attn.local_compute_fn is not None, "场景1：local_compute_fn 未注入"
    assert "q_proj.weight" in attn.params, "场景1：params 应继承推导"
    # 场景 2：merge + 契约 DSL（字段粒度替换为声明值）
    mlp = plan.modules["model.layers.0.mlp"]
    assert mlp.params["gate_proj.weight"][TP] == Shard(0), "场景2：DSL 未生效"
    assert mlp.params["down_proj.weight"][TP] == Shard(1), "场景2：DSL 未生效"
    # 场景 3：merge + 显式空（params 被清空）
    norm = plan.modules["model.layers.0.input_layernorm"]
    assert norm.params == {}, "场景3：显式 {} 应清空推导分片"
    # 场景 4：when: cp 且 cp=1 → 条目被跳过（inner_wrapper 不生效）
    assert attn.inner_wrapper is None, "场景4：when=cp 在 cp=1 下应跳过"
    # 场景 5：insert——新边界进入 plan，契约为声明值
    aux = plan.modules["model.aux"]
    assert aux.params["weight"][TP] == Replicate(), "场景5：insert params"
    assert aux.in_dst["hidden_states"][TP] == Replicate(), "场景5：insert 契约"
    assert aux.out_dst["output"][TP] == Shard(1), "场景5：insert 契约"


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    tp_size = dist.get_world_size()
    mesh = init_device_mesh("cpu", (1, tp_size), mesh_dim_names=("dp", "tp"))

    # 单卡参考（含 aux 模块）
    torch.manual_seed(0)
    ref = DemoModel().eval()
    torch.manual_seed(7)
    input_ids = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        expected = ref(input_ids)

    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "plan_overrides_demo.yaml")
    # cp=1/ep=1：when 过滤的拓扑上下文（场景 4 的条目因此被跳过）
    overrides = load_plan_overrides_from_yaml(yaml_path, cp_size=1, ep_size=1)

    for mode in ("production", "validate"):
        torch.manual_seed(0)
        model = DemoModel().eval()
        planner = ShardingPlanner(plan_overrides=overrides)
        plan = planner.plan(model, mesh, tp_size=tp_size)
        check_plan_scenarios(plan)             # 五场景 plan 内省断言
        model, _ = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))
        with torch.no_grad():
            out = model(input_ids)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        assert perf_kernels.FLASH_CALLS["n"] > 0, "场景1 kernel 未被调用"
        perf_kernels.FLASH_CALLS["n"] = 0
        print(f"[rank{rank}] {mode}: plan_overrides 五场景（merge 注入 / "
              f"契约 DSL / 显式空 / when 跳过 / insert 自声明）全部生效，"
              f"输出对拍单卡参考 ✓")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
