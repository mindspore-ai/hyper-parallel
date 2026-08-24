# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""自定义模块示例 2：inner_target + inner_wrapper —— 自定义 CP wrapper。

region_dispatch 判断口诀：注入物含通信原语/自定义 kernel/数据依赖分支 → False（黑盒托管）；纯 aten 标准算子 → True（validate 穿透真校验）。拿不准时用 check_dispatchable(fn, example_inputs, mesh) 在开发期探明。

用法:
    PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/custom_inner_wrapper.py

场景：NeMo 风格自研 attention，inner 子模块名是非标准的 `core_attention`，
且想用自己的 CP wrapper（注册为命名方案）。

要点：
- `inner_target="core_attention"`：纯位置——显式指定 inner 子模块
  （与 `inner_wrapper` 成对必填；自动定位启发式已删除，任何目标都须
  显式声明，包装模块自身写 `"self"`）；
- `inner_wrapper="demo_allgather"`：纯行为——str 引用 INNER_WRAPPER_REGISTRY
  里的命名方案（内置四路 "sdpa_qkv"/"sdpa_hf"/"flex_qkv"/"flex_hf"
  之外的第 5 种出口）；也可以直接给 callable 或 Target；
- **统一 override 通道**：plan_overrides 命中推导边界时走 merge——
  空字段（params/契约）继承模板推导结果，只需声明注入字段；glob key
  "*.self_attn" 一条覆盖所有层；
- wrapper 契约：@inner_wrapper fn(target_module, mesh, tp_mesh, cp_mesh,
  ep_mesh) -> None（mesh 家族框架填充），原地替换 target.forward；
- 本例 wrapper 只做 K/V all-gather（复用 flex_cp_allgather），语义与内置
  "sdpa_qkv" 相同；非 causal 简化。**local-only**：函数体内零 DTensor
  代码——validate 的解包/重包由框架双模适配器托管（重包布局用声明的
  inner_out_src="first_input"，即输出 == q 布局），production/validate
  双模式对拍。
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from hyper_parallel.auto_models.components.distributed import (
    ModuleShardingSpec,
    ShardingPlanner,
    apply_sharding_plan,
    inner_wrapper,
)
from hyper_parallel.auto_models.components.distributed.cp_utils import (
    flex_cp_allgather,
    shard_batch_for_cp,
)
from hyper_parallel.auto_models.components.distributed.cp_wrappers import (
    INNER_WRAPPER_REGISTRY,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh


# ── 自定义 CP wrapper（注册为命名方案） ─────────────────────────────────────

@inner_wrapper
def demo_cp_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """在 target_module.forward(q, k, v) 前对 K/V 做 CP all-gather。

    @inner_wrapper 是强制纪律（injection.py）：target_module + mesh 家族
    四个参数为必选上下文，apply 时全部由框架填充（本例只用 cp_mesh），
    用户只管使用。**local-only**：替换后的 forward 只面向 local 张量——
    DTensor 解包/参数临时解包/输出重包全部由框架的双模适配器托管（重包
    用 plan 里显式声明的 inner_out_src，本例 "first_input" = 输出布局
    与 q 一致）；本函数体内零 DTensor 代码。通信组取 cp_mesh.get_group()
    （DeviceMesh 缓存组，不得 new_group）。替换后的 forward 必须能接收
    原 forward 的全部入参（apply 时校验）。
    """
    orig_forward = target_module.forward

    def cp_forward(q, k, v, **kwargs):
        gk, gv = flex_cp_allgather(
            k.contiguous(), v.contiguous(), 2, cp_mesh)  # S_local → S_full
        return orig_forward(q, gk, gv, **kwargs)

    target_module.forward = cp_forward


INNER_WRAPPER_REGISTRY["demo_allgather"] = demo_cp_wrapper


# ── 模型 ────────────────────────────────────────────────────────────────────

class TinyRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states):
        var = hidden_states.pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(var + self.eps) * self.weight


class CoreAttention(nn.Module):
    """NeMo 风格 inner attention：forward(q, k, v) 签名。"""

    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


class MyNeMoAttention(nn.Module):
    """外层 attention：q/k/v/o 投影 + 非标准命名的 inner 子模块。

    inner 子模块目标必须靠 `inner_target` 显式指定（自动定位启发式
    已删除）——这里包装 `core_attention` 子模块。
    """

    def __init__(self, h, n_heads):
        super().__init__()
        self.head_dim = h // n_heads
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)
        self.core_attention = CoreAttention()

    def forward(self, hidden_states):
        b, s, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, s, -1, self.head_dim)
        k = self.k_proj(hidden_states).view(b, s, -1, self.head_dim)
        v = self.v_proj(hidden_states).view(b, s, -1, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = self.core_attention(q, k, v)      # ← CP wrapper 替换此调用目标
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
        self.self_attn = MyNeMoAttention(h, n_heads)
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


def build_overrides(num_layers):
    """attention 边界 spec：只需声明注入字段——merge 语义下 params 与 I/O
    契约从 planner 推导结果继承（统一改造后无需重声明）。

    glob key "*.self_attn" 一条覆盖所有层（fnmatch，* 跨段匹配）。
    """
    return {
        "*.self_attn": ModuleShardingSpec(
            inner_target="core_attention",        # 纯位置：非标准属性名
            inner_wrapper="demo_allgather",       # 纯行为：注册表命名方案
            inner_out_src="first_input",          # 纯布局：输出 == q（inner
                                                  # 子模块重包的显式声明）
            region_dispatch=False,                # wrapper 内含 all-gather
                                                  # 通信 → 不可 dispatch
        ),
    }


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    cp_size = dist.get_world_size()
    mesh = init_device_mesh("cpu", (cp_size,), mesh_dim_names=("cp",))
    cp_mesh = mesh["cp"]

    # 单卡参考（全序列）
    torch.manual_seed(0)
    ref = TinyModel().eval()
    torch.manual_seed(7)
    input_ids = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        expected = ref(input_ids)

    local_ids = shard_batch_for_cp({"input_ids": input_ids}, cp_mesh)["input_ids"]
    for mode in ("production", "validate"):
        torch.manual_seed(0)
        model = TinyModel().eval()
        planner = ShardingPlanner(plan_overrides=build_overrides(num_layers=2))
        plan = planner.plan(model, mesh, cp_size=cp_size)
        model, _ = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))
        # 解析结果回写：可观察性（INFO 日志同样可见）
        spec = plan.modules["model.layers.0.self_attn"]
        assert spec._resolved_inner_wrapper == "demo_allgather"

        with torch.no_grad():
            out_local = model(local_ids)

        gathered = [torch.empty_like(out_local) for _ in range(cp_size)]
        dist.all_gather(gathered, out_local, group=cp_mesh.get_group())
        out = torch.cat(gathered, dim=1)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        print(f"[rank{rank}] {mode}: custom inner_wrapper='demo_allgather' "
              f"(inner_target='core_attention') matches single-card reference")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
