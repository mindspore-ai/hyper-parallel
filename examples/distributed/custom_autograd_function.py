# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""自定义 autograd.Function 的边界接入（教程 §10.8）：第三方宿主不可改时
的完整纪律——子类化 + __class__ 实例级替换 + FunctionModule + plan_overrides。

region_dispatch 判断口诀：注入物含通信原语/自定义 kernel/数据依赖分支 → False（黑盒托管）；纯 aten 标准算子 → True（validate 穿透真校验）。拿不准时用 check_dispatchable(fn, example_inputs, mesh) 在开发期探明。

用法:
    PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/custom_autograd_function.py

场景：第三方库（本文件开头模拟 HF modeling 代码，**不可改**）的
`ThirdPartyMixer.forward` 里裸调用自定义 autograd.Function
（`SeqNormFn.apply`）——没有 FQN，边界机制不可见。目标：给这次调用插入
重排布通信（入口 all-gather 拿全序列 → Function 在完整序列上计算 →
出口切回 SP 本地 chunk），且不修改第三方文件。

SeqNormFn 故意选为"全序列统计量"语义（x 减全序列均值）：如果 all-gather
没有发生，均值只在本地 chunk 上算——数值必然与单卡不符。**对拍本身就是
桥接探针**：布局错 = 数值错，无需额外断言。

完整纪律（本例演示顺序）：
0. 版本锁定：子类 forward 是复制的第三方代码，pin 依赖版本（注释说明）；
1. smoke test：原类 vs 子类单卡对拍——依赖升级时让它大声失败；
2. 实例级 `__class__` 替换：构建模型之后、plan() 之前，nn.Module 无
   __slots__，权重零拷贝（本例宿主无参数，性质相同）；
3. FunctionModule 挂载 → FQN 出现 → plan_overrides 声明边界
   （params={} + region_dispatch=False——自定义 Function 不在 DTensor
   dispatch 覆盖范围，双模式都在 local tensor 上执行它）；
4. 双模式对拍单卡参考。
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from hyper_models.components.distributed import (
    FunctionModule,
    ShardingPlanner,
    apply_sharding_plan,
)
from hyper_models.components.distributed.sharding_config import (
    ModuleShardingSpec,
    TP,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard

B, S, V, H, HEADS = 2, 8, 64, 32, 4


# ── 模拟第三方/HF 代码（不可改）────────────────────────────────────────────
class SeqNormFn(torch.autograd.Function):
    """自定义 Function：x - mean(x)，均值为**完整序列**统计量——
    SP 切分下必须看到全序列，否则结果错误。"""

    @staticmethod
    def forward(ctx, x):
        m = x.mean()
        ctx.save_for_backward(m)
        return x - m

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out - grad_out.mean()


class ThirdPartyMixer(nn.Module):
    """第三方模块（不可改）：forward 里裸调用 SeqNormFn.apply。"""

    def forward(self, x):
        x = SeqNormFn.apply(x)      # ← 裸调用：无 FQN，边界机制不可见
        return x * 1.5 + 0.1        # 任意后续计算
# ── 第三方代码结束 ──────────────────────────────────────────────────────────


# ── 你的代码 ────────────────────────────────────────────────────────────────
# 纪律 0：下面的子类 forward 复制自 ThirdPartyMixer（仅一行不同）——
# 它把你的模型钉在当前第三方版本上，请 pin 依赖版本（如 transformers==x.y.z），
# 并用纪律 1 的 smoke test 保证升级时大声失败。
class ThirdPartyMixerPatched(ThirdPartyMixer):
    """仅把 SeqNormFn.apply(x) 换成 self.seqnorm_fn(x)，其余照抄。"""

    def __init__(self):
        super().__init__()
        self.seqnorm_fn = FunctionModule(SeqNormFn)

    def forward(self, x):
        x = self.seqnorm_fn(x)      # 原来：SeqNormFn.apply(x)
        return x * 1.5 + 0.1


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


class TinyBlock(nn.Module):
    def __init__(self, h, n_heads):
        super().__init__()
        self.input_layernorm = TinyRMSNorm(h)
        self.self_attn = TinyAttention(h, n_heads)
        self.post_attention_layernorm = TinyRMSNorm(h)
        self.mixer = ThirdPartyMixer()          # 第三方宿主（无参数）

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mixer(self.post_attention_layernorm(x))


class TinyModel(nn.Module):
    def __init__(self, vocab=V, h=H, n_heads=HEADS):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, h)
        self.model.layers = nn.ModuleList([TinyBlock(h, n_heads)])
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

    torch.manual_seed(7)
    input_ids = torch.randint(0, V, (B, S))
    x_probe = torch.randn(B, S, H)

    # 纪律 1：smoke test——原类 vs 子类单卡对拍（第三方升级时必须保持相等）
    orig = ThirdPartyMixer().eval()
    patched = ThirdPartyMixerPatched().eval()
    torch.testing.assert_close(patched(x_probe), orig(x_probe))

    # 单卡参考
    torch.manual_seed(0)
    ref = TinyModel().eval()
    with torch.no_grad():
        expected = ref(input_ids)

    # 纪律 3 的边界契约：Function 需要完整序列 → 入口 all-gather；
    # 输出各 rank 全量相同（Replicate）→ 出口切回 SP 本地 chunk（无通信）
    spec = ModuleShardingSpec(
        params={},                          # Function 无参数
        region_dispatch=False,               # 自定义 Function 不在 dispatch 覆盖范围
        in_src={"x": {TP: Shard(1)}},       # SP 本地 chunk 到达
        in_dst={"x": {TP: Replicate()}},    # ← 入口 all-gather 全序列
        out_src={"output": {TP: Replicate()}},
        out_dst={"output": {TP: Shard(1)}}, # ← 出口切回本地 chunk
    )

    for mode in ("production", "validate"):
        torch.manual_seed(0)
        model = TinyModel().eval()

        # 纪律 2：实例级 __class__ 替换（plan 之前；nn.Module 无 __slots__，
        # 实例 __dict__ 原封不动、权重零拷贝）
        for layer in model.model.layers:
            layer.mixer.__class__ = ThirdPartyMixerPatched
            layer.mixer.seqnorm_fn = FunctionModule(SeqNormFn)

        # 纪律 3：FQN 已出现 → plan_overrides 声明边界；planner 的 DX guard
        # 会对未声明的 FunctionModule 发出 warning（本例已声明，不触发）
        plan = ShardingPlanner(plan_overrides={
            "model.layers.0.mixer.seqnorm_fn": spec,
        }).plan(model, mesh, tp_size=world)
        assert "model.layers.0.mixer.seqnorm_fn" in plan.modules
        model, _ = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))

        with torch.no_grad():
            out = model(input_ids)
        # 对拍即探针：SeqNormFn 语义为全序列均值——all-gather 若未生效，
        # 均值在本地 chunk 上计算，此处必然 mismatch
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        print(f"[rank{rank}] {mode}: 第三方宿主 __class__ 替换 + "
              f"FunctionModule 边界（入口 all-gather [B,S/{world},H]→[B,S,H]）"
              f"| SeqNorm 全序列语义对拍单卡 ✓")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
