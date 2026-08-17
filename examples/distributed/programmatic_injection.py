# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""编程式注入示例：不依赖 trainer / YAML，只用 ShardingPlanner + apply_sharding_plan。

region_dispatch 判断口诀：注入物含通信原语/自定义 kernel/数据依赖分支 → False（黑盒托管）；纯 aten 标准算子 → True（validate 穿透真校验）。拿不准时用 check_dispatchable(fn, example_inputs, mesh) 在开发期探明。

用法:
    PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/programmatic_injection.py

面向的读者：想把**双模式 DTensor 能力**（ShardingPlanner 推导 +
apply_sharding_plan 应用 + validate/production 双模式）接入**自己的训练
框架**的开发者——不引入 trainer、不写一行 YAML，全部注入都是普通的
Python 对象。

本例在一个 TP=2 的三层玩具模型上，用**一个** plan_overrides dict 演示全部
编程式注入形态：

| 层 | 注入字段 | 形态 |
|---|---|---|
| layers.0.self_attn | inner_wrapper | ① @inner_wrapper 装饰的 callable **直接传函数**（最常用） |
| layers.1.self_attn | inner_wrapper | ② Target 延迟引用（携带数据配置键） |
| layers.2.self_attn | inner_wrapper | ③ 注册表名（可选：按名共享 / YAML str 引用时才需要注册） |
| *.mlp（glob 两层） | local_compute_fn | ④ @local_compute 装饰的区域计算工厂**直接传函数** |
| layers.2.mlp | local_compute_fn | ⑤ @local_compute 工厂 Target（需要配置键/YAML 载体时用；精确 key 覆盖 glob） |

**注册表与 Target 都不是必需的**：注入函数都可以直接传装饰后的函数对象。
`INNER_WRAPPER_REGISTRY` 只在两种场合需要——YAML 里用字符串引用、团队
按名共享方案；`Target` 只在两种场合需要——YAML `_target_` 引用、工厂需要
apply 期上下文（mesh 家族）或携带配置键。

要点：
- 注入函数一律带模板装饰器（@inner_wrapper / @local_compute），
  mesh 家族四参数必选、框架填充、用不用随你（统一接口规范）；
- 替换后的 forward / compute fn 只面向 **local 张量**——DTensor 解包与
  输出重包由框架的双模适配器/local-region 骨架托管，本文件零 DTensor 代码；
- target="self"（整个边界模块）时重包布局来自边界 out_src 声明，无需
  额外配置；只有包装 inner 子模块时才需声明 inner_out_src（见
  custom_inner_wrapper.py）；
- 配置键（如 block_size）只携带数据，按名绑定到工厂形参。
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from hyper_models.components.distributed import (
    ModuleShardingSpec,
    ShardingPlanner,
    apply_sharding_plan,
    inner_wrapper,
    local_compute,
)
from hyper_models.components.distributed.cp_wrappers import INNER_WRAPPER_REGISTRY
from hyper_models.trainer.config import Target
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh

# ── 注入生效证明用的计数器 ──────────────────────────────────────────────────
CALLS = {"flash": 0, "probe": 0, "target_wrap": 0, "swiglu": 0, "block": 0}


# ── 模型：HF 风格 eager attention + 朴素 MLP ────────────────────────────────

class EagerAttention(nn.Module):
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
        o = F.scaled_dot_product_attention(q, k, v)
        return self.o_proj(o.transpose(1, 2).reshape(b, s, -1))


class SlowMLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.gate_proj = nn.Linear(h, 4 * h, bias=False)
        self.up_proj = nn.Linear(h, 4 * h, bias=False)
        self.down_proj = nn.Linear(4 * h, h, bias=False)

    def forward(self, hidden_states):
        g = self.gate_proj(hidden_states)
        return self.down_proj((g * torch.sigmoid(g))
                              * self.up_proj(hidden_states))


class TinyBlock(nn.Module):
    def __init__(self, h, n_heads):
        super().__init__()
        self.self_attn = EagerAttention(h, n_heads)
        self.mlp = SlowMLP(h)

    def forward(self, x):
        return x + self.mlp(self.self_attn(x))


class TinyModel(nn.Module):
    def __init__(self, vocab=64, h=32, n_heads=4, n_layers=3):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, h)
        self.model.layers = nn.ModuleList(
            TinyBlock(h, n_heads) for _ in range(n_layers))
        self.lm_head = nn.Linear(h, vocab, bias=False)

    def forward(self, input_ids):
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return self.lm_head(h)


# ── 注入函数（全部 local-only，零 DTensor 代码）─────────────────────────────

def _flash(module, hidden_states):
    """共享的 flash 数学：一次 F.sdpa 调用（TP 语义不在 kernel 里——
    q/k/v 列切、o_proj 行切都是参数分片的事，kernel 只见本地 shard）。"""
    b, s, _ = hidden_states.shape
    q = module.q_proj(hidden_states).view(b, s, -1, module.head_dim)
    k = module.k_proj(hidden_states).view(b, s, -1, module.head_dim)
    v = module.v_proj(hidden_states).view(b, s, -1, module.head_dim)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    o = F.scaled_dot_product_attention(q, k, v)
    return module.o_proj(o.transpose(1, 2).reshape(b, s, -1))


# ③ 注册表命名方案（可选）：注册后任意 spec/YAML 可按名引用（团队共享）
@inner_wrapper
def demo_flash_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    def fwd(hidden_states, *args, **kwargs):
        CALLS["flash"] += 1
        return _flash(target_module, hidden_states)
    target_module.forward = fwd


INNER_WRAPPER_REGISTRY["demo_flash"] = demo_flash_wrapper


# ② 装饰 callable 直传：织入探针（先记录再调原实现——保留原 forward 引用）
@inner_wrapper
def probe_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    orig_forward = target_module.forward

    def fwd(hidden_states, *args, **kwargs):
        CALLS["probe"] += 1
        return orig_forward(hidden_states, *args, **kwargs)
    target_module.forward = fwd


# ③ Target 引用的 wrapper 工厂：携带数据配置键（block_size）
@inner_wrapper
def configurable_flash_wrapper(target_module, mesh, tp_mesh, cp_mesh,
                               ep_mesh, block_size=64):
    def fwd(hidden_states, *args, **kwargs):
        CALLS["target_wrap"] += 1
        CALLS["block"] = block_size
        return _flash(target_module, hidden_states)
    target_module.forward = fwd


# ④ local_compute_fn 的区域计算工厂：融合 swiglu 替换朴素实现
#    （mesh 家族必选、框架填充，本例不使用——统一接口规范）
@local_compute
def fused_swiglu_compute(mesh, tp_mesh, cp_mesh, ep_mesh):
    def compute_fn(module, hidden_states):
        CALLS["swiglu"] += 1
        return module.down_proj(
            F.silu(module.gate_proj(hidden_states))
            * module.up_proj(hidden_states))
    return compute_fn


# ⑤ 工厂 Target：apply 时 build 一次（mesh 家族框架填充，本例不用）
@local_compute
def counted_swiglu_factory(mesh, tp_mesh, cp_mesh, ep_mesh, block_size=64):
    def compute_fn(module, hidden_states):
        CALLS["block"] = block_size
        return module.down_proj(
            F.silu(module.gate_proj(hidden_states))
            * module.up_proj(hidden_states))
    return compute_fn


# ── 编程式 plan_overrides：一个 dict 覆盖全部形态 ───────────────────────────

def build_overrides():
    # 本例全部注入物都是纯标准算子（F.sdpa/F.silu/线性层，可 dispatch）
    # → region_dispatch=True：validate 下策略传播穿透注入函数、out_src
    # 真校验。若注入物含通信/自定义 kernel（如 CP/EP 场景），必须显式
    # region_dispatch=False（见 cp.py / ep.py 的注释说明）
    return {
        # ① 装饰 callable 直传（最常用形态，不需要注册表）
        "model.layers.0.self_attn": ModuleShardingSpec(
            inner_target="self", inner_wrapper=probe_wrapper,
            region_dispatch=True),
        # ② Target + 数据配置键
        "model.layers.1.self_attn": ModuleShardingSpec(
            inner_target="self",
            region_dispatch=True,
            inner_wrapper=Target(
                configurable_flash_wrapper,
                target_path="examples.programmatic_injection."
                            "configurable_flash_wrapper",
                block_size=128)),
        # ③ 注册表名（按名共享/YAML 引用才需要注册）
        "model.layers.2.self_attn": ModuleShardingSpec(
            inner_target="self", inner_wrapper="demo_flash",
            region_dispatch=True),
        # ④ glob + @local_compute callable（merge：契约继承推导）
        "*.mlp": ModuleShardingSpec(local_compute_fn=fused_swiglu_compute,
                                    region_dispatch=True),
        # ⑤ 精确 key 后写覆盖 glob：layer2 的 mlp 改走工厂 Target
        "model.layers.2.mlp": ModuleShardingSpec(
            region_dispatch=True,
            local_compute_fn=Target(
                counted_swiglu_factory,
                target_path="examples.programmatic_injection."
                            "counted_swiglu_factory",
                block_size=256)),
    }


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    tp_size = dist.get_world_size()
    mesh = init_device_mesh("cpu", (1, tp_size), mesh_dim_names=("dp", "tp"))

    # 单卡参考
    torch.manual_seed(0)
    ref = TinyModel().eval()
    torch.manual_seed(7)
    input_ids = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        expected = ref(input_ids)

    for mode in ("production", "validate"):
        torch.manual_seed(0)
        model = TinyModel().eval()
        # 只需两行：plan + apply——没有 trainer、没有 YAML
        plan = ShardingPlanner(plan_overrides=build_overrides()).plan(
            model, mesh, tp_size=tp_size)
        model, source_shard_info = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))

        with torch.no_grad():
            out = model(input_ids)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

        # 五种注入形态逐一断言生效：attention 每层各一次；glob 的 swiglu
        # 覆盖 layers.0/1 两个 mlp（layer2 被精确 key 的工厂 Target 覆盖）
        assert CALLS["probe"] == 1, "① callable 直传未生效"
        assert CALLS["target_wrap"] == 1 and CALLS["block"] == 256, \
            "②⑤ Target 未生效或配置键未按名绑定"
        assert CALLS["flash"] == 1, "③ 注册表名未生效"
        assert CALLS["swiglu"] == 2, "④ local_compute fn 未生效（应 2 次）"
        # 内省回写：解析结果可查
        spec0 = plan.modules["model.layers.2.self_attn"]
        assert spec0._resolved_inner_wrapper == "demo_flash"
        assert spec0._resolved_inner_target == "self"

        print(f"[rank{rank}] {mode}: 编程式注入五形态（callable/Target/"
              f"注册表名/local_compute/工厂Target）全部生效，输出对拍单卡 ✓")
        for k in CALLS:
            CALLS[k] = 0
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
