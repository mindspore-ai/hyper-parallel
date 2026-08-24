# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""性能替换端到端示例：YAML 注入自定义高性能 kernel，替换朴素实现。

region_dispatch 判断口诀：注入物含通信原语/自定义 kernel/数据依赖分支 → False（黑盒托管）；纯 aten 标准算子 → True（validate 穿透真校验）。拿不准时用 check_dispatchable(fn, example_inputs, mesh) 在开发期探明。

用法:
    PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/perf_replacement.py

场景：模型里识别出两个低性能实现——
- SlowCausalAttention：HF eager 风格——显式 S×S scores + mask + 两次
  matmul，全程无融合 kernel（对应未走 flash/sdpa 的参考实现）；
- SlowMLP：silu 被分解成 sigmoid+mul 一串小 kernel（对应未融合算子）。

本示例不改模型代码，通过 YAML 的 plan_overrides 把朴素实现替换为
perf_kernels.py 里的高性能实现（F.sdpa / 融合 F.silu），并对比两条通道：

- **local_compute_fn 通道**（perf_replacement.yaml，变体 2）：Target 工厂
  在 apply 时 build 出 compute_fn，在 local-region 骨架内以本地张量执行，
  参数解包/I/O 契约由骨架托管——kernel 零并行逻辑、零双模式负担（推荐）；
- **inner_wrapper 通道**（perf_replacement_inner_wrap.yaml，变体 3）：
  原地替换 target.forward，无骨架托管——wrapper 自负双模式容错
  （validate 下输入是 DTensor），适合需要"织入"（保留原 forward 引用）
  或只替换 inner 子模块的场景。inner-wrap 机制不限于 CP（声明即应用）；
- **merge 语义**：match 命中 planner 推导边界，params 切分与 I/O 契约继承
  推导结果，YAML 只声明注入字段——TP=2 的参数分片（列切/行切）与输出
  缝合由框架负责；
- 验证三件事：替换真正生效（kernel 计数器）、数值与朴素实现对拍一致、
  同条件计时的性能差异（信息性打印）。
"""

import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ^ 让 YAML `_target_: perf_kernels.*` 可解析——真实场景即"用户 kernel 包
#   在 PYTHONPATH 上"。resolve 发生在 import 时，必须先于 YAML 解析。

import perf_kernels  # noqa: E402
from typing import List  # noqa: E402
from hyper_parallel.auto_models.components.distributed import (  # noqa: E402
    ShardingPlanner,
    apply_sharding_plan,
)
from hyper_parallel.auto_models.config.resolver import resolve_component  # noqa: E402
from hyper_parallel.auto_models.trainer.config import (  # noqa: E402
    PlanOverride,
    entries_to_plan_overrides,
)
from hyper_parallel.core.dtensor.device_mesh import (  # noqa: E402
    init_device_mesh,
)


class TinyRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states):
        var = hidden_states.pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(var + self.eps) * self.weight


class SlowCausalAttention(nn.Module):
    """朴素 attention（故意慢）：HF eager 风格——显式构造 S×S scores 矩阵、
    加 mask、softmax、再一次 matmul，全程无融合 kernel（真实场景对应未走
    flash/sdpa kernel 的参考实现）。"""

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
        mask = torch.triu(torch.full((s, s), float("-inf"),
                                     dtype=q.dtype), diagonal=1)
        scores = q @ k.transpose(-2, -1) / self.head_dim ** 0.5
        o = torch.softmax(scores + mask, dim=-1) @ v
        return self.o_proj(o.transpose(1, 2).reshape(b, s, -1))


class SlowMLP(nn.Module):
    """朴素 MLP（故意慢）：silu 分解成 sigmoid+mul 多 kernel，未融合。"""

    def __init__(self, h):
        super().__init__()
        self.gate_proj = nn.Linear(h, 4 * h, bias=False)
        self.up_proj = nn.Linear(h, 4 * h, bias=False)
        self.down_proj = nn.Linear(4 * h, h, bias=False)

    def forward(self, hidden_states):
        g = self.gate_proj(hidden_states)
        # 分解的 silu = g * sigmoid(g)：独立 sigmoid kernel + 两次逐点 mul，
        # 对比融合 F.silu 的一次 kernel（真实场景对应未融合的算子串）。
        sig = torch.sigmoid(g)
        return self.down_proj((g * sig) * self.up_proj(hidden_states))


class TinyBlock(nn.Module):
    def __init__(self, h, n_heads):
        super().__init__()
        self.input_layernorm = TinyRMSNorm(h)
        self.self_attn = SlowCausalAttention(h, n_heads)
        self.post_attention_layernorm = TinyRMSNorm(h)
        self.mlp = SlowMLP(h)

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


def load_plan_overrides_from_yaml(yaml_path, *, cp_size=1, ep_size=1):
    """YAML → List[PlanOverride] → plan_overrides（planner 的唯一 override 接口）。

    与 trainer 内部走的是同一条脱糖路径：resolve_component 按 schema 解析
    （含 when 的 Literal 校验、Target 延迟生成）→ entries_to_plan_overrides
    脱糖（placement 字符串 DSL → Placement 对象、when 声明式过滤、同 match
    逐字段合并）。when 过滤需要并行拓扑（cp_size/ep_size）作上下文。
    """
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    entries = resolve_component(
        raw.get("plan_overrides"),
        expected_type=List[PlanOverride], path="plan_overrides")
    return entries_to_plan_overrides(
        entries, cp_size=cp_size, ep_size=ep_size)


def show_plan_overrides(overrides, rank):
    """打印 YAML 脱糖后的 plan_overrides 实际内容（可观察性）。

    ── 等价的不经 YAML 写法（programmatic 直接构造，效果完全相同）────────
    from hyper_parallel.auto_models.components.distributed import ModuleShardingSpec
    from hyper_parallel.auto_models.trainer.config import Target
    overrides = {
        "*.self_attn": ModuleShardingSpec(
            region_dispatch=True,   # F.sdpa 融合 kernel：纯标准算子可 dispatch
            local_compute_fn=Target(
                perf_kernels.flash_attention_factory,
                target_path="perf_kernels.flash_attention_factory")),
        "*.mlp": ModuleShardingSpec(
            region_dispatch=True,   # F.silu 融合：纯标准算子可 dispatch
            local_compute_fn=Target(
                perf_kernels.fused_swiglu_factory,
                target_path="perf_kernels.fused_swiglu_factory")),
    }
    # inner_wrapper 通道（perf_replacement_inner_wrap.yaml）等价写法：
    # overrides = {
    #     "*.self_attn": ModuleShardingSpec(
    #         inner_target="self",
    #         region_dispatch=True,   # 同上：纯算子替换，validate 穿透真校验
    #         inner_wrapper=Target(
    #             perf_kernels.flash_attention_wrapper,
    #             target_path="perf_kernels.flash_attention_wrapper")),
    # }
    ────────────────────────────────────────────────────────────────────
    """
    if rank != 0:
        return
    print("[rank0] YAML → plan_overrides 脱糖结果：")
    for match, spec in overrides.items():
        fields = []
        if spec.local_compute_fn is not None:
            fields.append(
                "local_compute_fn=Target("
                f"{spec.local_compute_fn._target_path})")
        if spec.inner_target is not None:
            fields.append(f"inner_target={spec.inner_target!r}")
        if spec.inner_wrapper is not None:
            w = spec.inner_wrapper
            fields.append(
                f"inner_wrapper=Target({w._target_path})"
                if hasattr(w, "_target_") else f"inner_wrapper={w!r}")
        print(f'  "{match}" -> ModuleShardingSpec({", ".join(fields)})')


def _reset_counters():
    perf_kernels.FLASH_CALLS["n"] = 0
    perf_kernels.FUSED_CALLS["n"] = 0


def run_variant(mode, plan_overrides, mesh, tp_size, input_ids, expected):
    """同条件跑一个变体（原始/替换），返回每步耗时 ms。对拍单卡朴素参考。"""
    torch.manual_seed(0)
    model = TinyModel().eval()
    planner = ShardingPlanner(plan_overrides=plan_overrides)
    plan = planner.plan(model, mesh, tp_size=tp_size)
    model, _ = apply_sharding_plan(
        model, plan, mesh, validate_mode=(mode == "validate"))
    with torch.no_grad():
        out = model(input_ids)                  # warmup + 正确性
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        start = time.perf_counter()
        for _ in range(20):
            model(input_ids)
    return (time.perf_counter() - start) / 20 * 1e3


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    tp_size = dist.get_world_size()
    mesh = init_device_mesh("cpu", (1, tp_size), mesh_dim_names=("dp", "tp"))

    # 单卡参考：朴素实现（替换后的 kernel 必须在数值上与之等价）
    torch.manual_seed(0)
    ref = TinyModel().eval()
    torch.manual_seed(7)
    input_ids = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        expected = ref(input_ids)

    here = os.path.dirname(os.path.abspath(__file__))
    yaml_local = os.path.join(here, "perf_replacement.yaml")
    yaml_wrap = os.path.join(here, "perf_replacement_inner_wrap.yaml")
    for mode in ("production", "validate"):
        # 变体 1：原始朴素实现（无注入，仅 TP 分片）
        _reset_counters()
        t_slow = run_variant(mode, None, mesh, tp_size, input_ids, expected)
        assert perf_kernels.FLASH_CALLS["n"] == 0, "原始实现不应触碰 fast path"

        # 变体 2：YAML 注入高性能 kernel（local_compute_fn 通道：attn+MLP）
        overrides = load_plan_overrides_from_yaml(yaml_local)
        if mode == "production":
            show_plan_overrides(overrides, rank)
        _reset_counters()
        t_fast = run_variant(mode, overrides, mesh, tp_size,
                             input_ids, expected)
        assert perf_kernels.FLASH_CALLS["n"] > 0, "fast attention 未被调用"
        assert perf_kernels.FUSED_CALLS["n"] > 0, "fused MLP 未被调用"

        # 变体 3：YAML 注入（inner_wrapper 通道：只替换 attention 的 forward）
        overrides_wrap = load_plan_overrides_from_yaml(yaml_wrap)
        if mode == "production":
            show_plan_overrides(overrides_wrap, rank)
        _reset_counters()
        t_wrap = run_variant(mode, overrides_wrap, mesh, tp_size,
                             input_ids, expected)
        assert perf_kernels.FLASH_CALLS["n"] > 0, "inner_wrap 替换未生效"
        assert perf_kernels.FUSED_CALLS["n"] == 0, "本变体不应替换 MLP"

        print(f"[rank{rank}] {mode}: YAML 注入替换生效 "
              f"(local_compute_fn: flash+fused / inner_wrapper: flash)，"
              f"输出对拍朴素参考 ✓")
        if rank == 0:
            print(f"[rank{rank}] {mode}: 同条件计时 slow={t_slow:.2f}ms "
                  f"local_compute_fn={t_fast:.2f}ms inner_wrap(仅 attn)="
                  f"{t_wrap:.2f}ms（信息性，规模小差异有限）")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
