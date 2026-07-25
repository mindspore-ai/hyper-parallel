# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""嵌套 spec（D-14）独立示例：外层 local_map + 内层策略传播校验（validate 孤岛）。

用法:
    PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/nested_local_map.py

场景（05 §13.1 场景 2）：对整个语言模型声明输入/输出切分契约——但整模型
forward 范围太大（embed/残差/glue 代码可能存在无 dispatch 实现的算子），
全量 DTensor 传播走不通；同时仍希望关键模块（attention/mlp/norm/lm_head）
能跑 dispatch 级策略传播校验。

做法（05 §13.4 validate 孤岛）：
- 根 fqn "" 注入外层 spec：use_local_map=True，params={}，仅声明
  input_ids / logits 的 I/O 契约——外层走 local-region 骨架，区域内
  全程 local tensor（glue 代码永远不碰 DTensor）；
- 内层标准边界照常推导；validate 模式下每个内层边界自动成"孤岛"：
  入口 from_local(declared in_src) → DTensor dispatch 传播 →
  assert out_src == declared → 出口 to_local 回落 local 世界；
- 双模式同跑：production（全 local，零 dispatch）与 validate（孤岛校验）
  输出均与单卡参考对拍；
- 关键机制（05 §13.3 不变式 3）：外层 local region 的临时参数解包
  **排除内层边界子树**——孤岛内参数保持 DTensor（dispatch 靠
  __torch_function__），本例用 forward_pre_hook 直接探针验证。
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from hyper_models.components.distributed import (
    ShardingPlanner,
    apply_sharding_plan,
)
from hyper_models.components.distributed.sharding_config import (
    ModuleShardingSpec,
    TP,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


class TinyRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states):
        var = hidden_states.pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(var + self.eps) * self.weight


class TinyAttention(nn.Module):
    def __init__(self, h, n_heads):
        super().__init__()
        self.head_dim = h // n_heads
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)

    def forward(self, hidden_states):
        b, s, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, s, -1, self.head_dim)
        k = self.k_proj(hidden_states).view(b, s, -1, self.head_dim)
        v = self.v_proj(hidden_states).view(b, s, -1, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.o_proj(o.transpose(1, 2).reshape(b, s, -1))


class TinyMLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.gate_proj = nn.Linear(h, 4 * h, bias=False)
        self.up_proj = nn.Linear(h, 4 * h, bias=False)
        self.down_proj = nn.Linear(4 * h, h, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states))


class TinyBlock(nn.Module):
    def __init__(self, h, n_heads):
        super().__init__()
        self.input_layernorm = TinyRMSNorm(h)
        self.self_attn = TinyAttention(h, n_heads)
        self.post_attention_layernorm = TinyRMSNorm(h)
        self.mlp = TinyMLP(h)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class TinyModel(nn.Module):
    def __init__(self, vocab=64, h=32, n_heads=4, n_layers=2):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, h)
        self.model.layers = nn.ModuleList(
            TinyBlock(h, n_heads) for _ in range(n_layers))
        self.model.norm = TinyRMSNorm(h)
        self.lm_head = nn.Linear(h, vocab, bias=False)

    def forward(self, input_ids):
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return self.lm_head(self.model.norm(h))


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()   # 2：mesh (tp=2)

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("tp",))

    # 单卡参考
    torch.manual_seed(0)
    ref = TinyModel().eval()
    torch.manual_seed(7)
    input_ids = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        expected = ref(input_ids)

    # 外层 spec：根 fqn ""（整个 LM），use_local_map=True，params={}，
    # 仅声明输入/输出契约——input_ids 全复制（各 rank 同 batch），
    # logits 经 lm_head（loss_parallel=False）TP 维全复制
    root_spec = ModuleShardingSpec(
        params={},
        use_local_map=True,   # 外层走 local-region 骨架：区域内全 local
        in_src={"input_ids": {TP: Replicate()}},
        in_dst={"input_ids": {TP: Replicate()}},     # identity
        out_src={"output": {TP: Replicate()}},
        out_dst={"output": {TP: Replicate()}},       # identity
    )

    for mode in ("production", "validate"):
        torch.manual_seed(0)
        model = TinyModel().eval()
        plan = ShardingPlanner(plan_overrides={"": root_spec}).plan(
            model, mesh, tp_size=world)
        assert "" in plan.modules                     # 外层根 spec 已插入
        assert "model.layers.0.self_attn" in plan.modules   # 内层边界保留
        model, _ = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))

        if mode == "validate":
            # 不变式 3 探针：外层 local region 内，内层边界参数必须保持
            # DTensor（孤岛 dispatch 依赖 __torch_function__）
            probe = {}

            def _hook(module, args):
                probe["attn_param_is_dtensor"] = isinstance(
                    module.q_proj.weight, DTensor)

            handle = model.model.layers[0].self_attn.register_forward_pre_hook(_hook)
            with torch.no_grad():
                out = model(input_ids)
            handle.remove()
            assert probe.get("attn_param_is_dtensor"), (
                "内层边界参数在外层 local region 内必须保持 DTensor")
            print(f"[rank{rank}] validate: 内层孤岛 dispatch 校验通过 "
                  f"(out_src 断言生效，孤岛内参数保持 DTensor)")
        else:
            with torch.no_grad():
                out = model(input_ids)
            print(f"[rank{rank}] production: 外层 local region 全 local 直通")

        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        print(f"[rank{rank}] {mode}: 嵌套 local_map(外层 LM) + 孤岛(内层) "
              f"TP={world} output matches single-card reference")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
