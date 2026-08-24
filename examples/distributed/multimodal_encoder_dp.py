# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""多模态双 mesh 组合独立示例：ViT encoder_dp + LLM dp×tp×ep（桥接边界 all-gather）。

region_dispatch 判断口诀：注入物含通信原语/自定义 kernel/数据依赖分支 → False（黑盒托管）；纯 aten 标准算子 → True（validate 穿透真校验）。拿不准时用 check_dispatchable(fn, example_inputs, mesh) 在开发期探明。

用法:
    PYTHONPATH=. torchrun --nproc_per_node=4 examples/distributed/multimodal_encoder_dp.py

拓扑（4 卡，同一进程组上的两个 mesh 视图）：
- LLM 视图：llm_mesh (dp=2, tp=2)，ep_size=4——D-10 TP-extend-EP：expert
  mesh 从 dense 区域（dp×tp=4 rank）派生 (edp=1, ep=4)，4 专家每 rank 1 个；
- ViT 视图（encoder_dp）：vit_mesh = llm_mesh._flatten("encoder_dp") →
  dense 区域全体 rank 的 1-D 视图（轴名自定义，不与根 mesh 轴名冲突即可）。
  两个用途：训练侧 fully_shard(vision_tower, mesh=vit_mesh) 的 FSDP 权重
  通信域 + 数据管道按 vit_mesh.get_local_rank() 给各 rank 分配图像子集；
  DTensor 侧 ViT 零参数分片（params={}）；
- 桥接（本例核心）：vision_tower 的 **out 边界** 声明
  {TP: Shard(0)} → {TP: Replicate()} —— all-gather 就是一次边界重分布，
  由 precompiled boundary 双模式执行，无需独立 helper / autograd Function。

坐标系约定（05 §3，读本例所有 spec 的前提）：**plan 描述的是单个 dp
切片的布局与通信——plan.mesh_dim_names 恒为 tp/cp/ep 子集，永远不含
dp**。这不是缺失，是分层：dp 维的切分由另外两套文档表达——数据管道
（各 rank 拿到什么数据）和 FSDP（参数/梯度的 dp 域）。若试图在 spec
里声明 DP placement，planner 会 fail-fast 并指向本约定。

因此桥接 spec 的读法是：在单个 dp 切片内，ViT 特征在 tp 组间
Shard(0)（各 rank 编码不同图像）、gather 后 Replicate（组内全量相同）
——没有隐瞒任何 dp 信息，dp=4 的故事由下面三处讲：
- vit_mesh（llm_mesh._flatten("encoder_dp")）：encoder_dp 域本身，
  用于数据分配（enc_rank）与训练侧 fully_shard(vision_tower, mesh=vit_mesh)；
- 数据布局：文本 batch 按 dp=2 切（TP 组内相同）；图像按 encoder_dp=4 切
  （每 rank 编码本 shard 图像子集的 1/tp 份）；
- gather 后 TP 组内两个 rank 持有相同的全量特征，merge glue（pool +
  注入 embedding）在 local 世界运行。

机制要点：
- 桥接 spec 走 **plan_overrides** 既定自定义入口（根 fqn "" 挂在
  vision_tower 自身），apply 用 llm_mesh（gather 的通信域是 LLM 的 tp 组）；
- "tp=1"（ViT **计算** 不做 TP）由 params={} + `derive=False`（关闭
  内层模板推导）表达——I/O 排布描述数据布局，params 描述参数切分，
  两者分开才精确；
- region_dispatch=False：ViT 内部不做 dispatch（validate 下整区 local）；
- LLM 侧与纯 LLM 完全同构：独立 plan/apply（tp=2, ep=4），内层边界照常
  推导，validate 孤岛照常断言；
- 约束：all-gather 是静态形状集合通信 → 各 rank 视觉 token 数必须对齐
  （本例每 rank 恰好 1 张图 × P 个 patch；真实场景需数据管道做视觉 token
  均衡/padding）。

梯度语义（训练场景）：边界通信 autograd 感知——fwd all-gather /
bwd reduce-scatter-sum。TP 组内 LLM 计算是分片的，梯度求和恰好补全每个
rank 自己图像块的完整梯度；ViT 参数梯度再由 dp=4 FSDP 域 all-reduce =
全局 batch 梯度。
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from hyper_parallel.auto_models.components.distributed import (
    ShardingPlanner,
    apply_sharding_plan,
    routed_only_ep_compute_fn,
)
from hyper_parallel.auto_models.components.distributed.sharding_config import (
    ModuleShardingSpec,
    TP,
)
from hyper_parallel.auto_models.trainer.config import Target
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard

B, S, V = 2, 8, 64        # 每个 LLM dp shard 的 batch / 序列长 / 词表
P, PATCH = 4, 6           # 每张图的 patch 数 / patch 维
H, HEADS, INTER, EXPERTS = 32, 4, 16, 4


class TinyRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states):
        var = hidden_states.pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(var + self.eps) * self.weight


class TinyViT(nn.Module):
    """每 rank 只编码自己的图像子集（encoder_dp）；输出该子集特征 [N_local, H]。

    参数永不被 DTensor 分片（bridge spec params={}）——纯 FSDP 公民。
    """

    def __init__(self, patch_dim=PATCH, h=H):
        super().__init__()
        self.patch_proj = nn.Linear(patch_dim, h, bias=False)
        self.norm = TinyRMSNorm(h)

    def forward(self, pixel_values):                 # [N_local, patch_dim]
        return self.norm(self.patch_proj(pixel_values))   # [N_local, H]


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


class TinyExpertMLP(nn.Module):
    def __init__(self, h, inter):
        super().__init__()
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def forward(self, x):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyNativeMoE(nn.Module):
    """HF 原生风格 MoE（同 ep.py/tp_cp_ep.py）：gate + per-expert ModuleList，
    D-09 堆叠 + D-10 TP-extend-EP 参数分片；EP compute 由 injections 显式
    注入（仓内默认工厂 routed_only_ep_compute_fn）。"""

    def __init__(self, h, inter, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(h, num_experts, bias=False)
        self.experts = nn.ModuleList(
            TinyExpertMLP(h, inter) for _ in range(num_experts))

    def forward(self, hidden_states):
        b, s, h = hidden_states.shape
        x = hidden_states.view(-1, h)
        logits = self.gate(hidden_states).view(-1, self.num_experts)
        weights = logits.softmax(-1)
        topk_w, topk_idx = weights.topk(self.top_k, dim=-1)
        topk_w = topk_w / topk_w.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        for e_idx, expert in enumerate(self.experts):
            tok, slot = (topk_idx == e_idx).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            out.index_add_(0, tok, expert(x[tok]) * topk_w[tok, slot].unsqueeze(-1))
        return out.view(b, s, h)


class TinyBlock(nn.Module):
    def __init__(self, h, n_heads, inter, num_experts):
        super().__init__()
        self.input_layernorm = TinyRMSNorm(h)
        self.self_attn = TinyAttention(h, n_heads)
        self.post_attention_layernorm = TinyRMSNorm(h)
        self.mlp = TinyNativeMoE(h, inter, num_experts)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class TinyMoeLLM(nn.Module):
    """与 tp_cp_ep.py 的 TinyMoeModel 同构，仅多一个 img_emb 注入口
    （merge glue：图像特征 pool 后广播加到文本 embedding，local tensor 世界）。"""

    def __init__(self, vocab=V, h=H, n_heads=HEADS, n_layers=2,
                 inter=INTER, num_experts=EXPERTS):
        super().__init__()
        self.config = type("Cfg", (), {
            "architectures": ["TinyMoeForCausalLM"],   # arch → default router adapter
            "num_experts": num_experts,
            "num_experts_per_tok": 2,
        })()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, h)
        self.model.layers = nn.ModuleList(
            TinyBlock(h, n_heads, inter, num_experts) for _ in range(n_layers))
        self.model.norm = TinyRMSNorm(h)
        self.lm_head = nn.Linear(h, vocab, bias=False)

    def forward(self, input_ids, img_emb=None):
        h = self.model.embed_tokens(input_ids)
        if img_emb is not None:            # merge glue：局部 chunk 上广播加，位置无关
            h = h + img_emb.unsqueeze(1)
        for layer in self.model.layers:
            h = layer(h)
        return self.lm_head(self.model.norm(h))


class TinyVLM(nn.Module):
    """多模态容器：vision_tower 与 language_model 是兄弟模块。
    容器 forward 是纯 local glue（无 spec），只调用两个已被包装的子模块。"""

    def __init__(self):
        super().__init__()
        self.vision_tower = TinyViT()
        self.language_model = TinyMoeLLM()

    def forward(self, input_ids, pixel_values):
        feats = self.vision_tower(pixel_values)     # 桥接 out 边界 all-gather → [B*P, H]
        img_emb = feats.view(-1, P, H).mean(1)      # 每图 mean-pool → [B, H]
        return self.language_model(input_ids, img_emb)


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()   # 4：LLM (dp=2, tp=2)，ViT encoder_dp=4，ep=4
    dp_size, tp_size, ep_size = 2, 2, 4
    assert world == dp_size * tp_size

    llm_mesh = init_device_mesh("cpu", (dp_size, tp_size),
                                mesh_dim_names=("dp", "tp"))
    # encoder_dp 域 = dense 区域全体 rank 的 1-D flatten（行主序：dp 外 tp 内，
    # flat 序号 = dp_rank*tp + tp_rank）。用途：① 训练侧 fully_shard(
    # vision_tower, mesh=vit_mesh) 的 FSDP 域；② 数据布局按 enc_rank 分配图像
    vit_mesh = llm_mesh._flatten("encoder_dp")
    dp_rank = llm_mesh["dp"].get_local_rank()
    tp_rank = llm_mesh["tp"].get_local_rank()
    enc_rank = vit_mesh.get_local_rank()
    assert enc_rank == dp_rank * tp_size + tp_rank      # flatten 行主序自检

    # 数据双布局：文本按 dp=2 切（TP 组内相同）；图像按 encoder_dp=4 切——
    # 全批图像按 flatten 序排成 [world, P, patch]，本 rank 编码第 enc_rank 张
    #（本例每 shard B=2 张图 = tp_size，恰好转到本 TP 组的 2 个 rank 上）
    torch.manual_seed(7)
    all_ids = torch.randint(0, V, (dp_size, B, S))       # [dp, B, S]
    all_images = torch.randn(world, P, PATCH)            # [encoder_dp, P, patch]
    input_ids = all_ids[dp_rank]                         # 本 shard 文本 [B, S]
    my_image = all_images[enc_rank]                      # 本 rank 编码的图 [P, patch]
    # 本 shard 全部图 = 本 TP 组各 rank 编码的图按 flatten 序拼接
    shard_images = all_images[dp_rank * tp_size:
                              (dp_rank + 1) * tp_size].reshape(B * P, PATCH)

    # 单卡参考：本 shard 的全部图像 + 全文本
    torch.manual_seed(0)
    ref = TinyVLM().eval()
    with torch.no_grad():
        expected = ref(input_ids, shard_images)
        ref_feats = ref.vision_tower(shard_images)       # [B*P, H]，图按序拼接

    # 桥接 spec（坐标系 = 单 dp 切片，05 §3 约定）：tp 组内特征分片 →
    # gather 后组内全量相同。"tp=1"（ViT 计算不做 TP）由 params={} +
    # 无内层 spec 表达；dp=4 语义见上文 vit_mesh / 数据布局 / FSDP 三处
    bridge = ModuleShardingSpec(
        params={},
        region_dispatch=False,                      # ViT 内部不做 dispatch
        in_src={"pixel_values": {TP: Shard(0)}},    # tp 组间：各 rank 特征分片
        in_dst={"pixel_values": {TP: Shard(0)}},    # identity，入口无通信
        out_src={"output": {TP: Shard(0)}},
        out_dst={"output": {TP: Replicate()}},      # ← tp 组内 all-gather
    )

    for mode in ("production", "validate"):
        torch.manual_seed(0)
        model = TinyVLM().eval()

        # ① ViT 侧：plan_overrides 注入桥接（D-14 根 fqn ""），apply 用
        #    llm_mesh（gather 的通信域是 LLM 的 tp 组）
        # derive=False：ViT 子树不做任何模板推导——encoder_dp 下各 rank
        # 数据不同，内层任何 TP 集合通信都是数学错误；plan 只含显式声明
        # 的桥接 spec（取代 plan.modules 后处理剪除写法）
        vit_plan = ShardingPlanner(plan_overrides={"": bridge},
                                   derive=False).plan(
            model.vision_tower, llm_mesh, tp_size=tp_size)
        assert vit_plan.mesh_dim_names == ("tp",)   # 坐标系约定：恒不含 dp
        assert set(vit_plan.modules) == {""}        # 零推导：只有桥接
        apply_sharding_plan(model.vision_tower, vit_plan, llm_mesh,
                            validate_mode=(mode == "validate"))

        # ② LLM 侧：与纯 LLM 完全同构的独立 plan/apply（dp 轴被 planner 过滤）；
        #    EP compute 显式注入——expert mesh 由框架在 apply 时统一派生
        #   （与专家参数分片共享同一对象），经 ep_mesh 上下文传入工厂
        llm_plan = ShardingPlanner(plan_overrides={
            "*.mlp": ModuleShardingSpec(
                region_dispatch=False,      # EP compute 内含 a2a → 不可 dispatch
                local_compute_fn=Target(
                    routed_only_ep_compute_fn,
                    target_path="hyper_parallel.auto_models.components.distributed."
                                "ep_compute.routed_only_ep_compute_fn"),
            ),
        }).plan(
            model.language_model, llm_mesh, tp_size=tp_size, ep_size=ep_size)
        assert llm_plan.mesh_dim_names == ("tp",)        # dp 轴已剥离
        moe_spec = llm_plan.modules["model.layers.0.mlp"]
        assert moe_spec._ep_size == ep_size   # D-10：expert mesh (edp=1, ep=4)
        assert moe_spec._ep_stack             # per-expert → Phase A 堆叠（D-09）
        model.language_model, _ = apply_sharding_plan(
            model.language_model, llm_plan, llm_mesh,
            validate_mode=(mode == "validate"))

        # ③ 前向 + 桥接探针：gather 后特征必须 == 单卡 ViT 对本 shard 全部图的输出
        captured = {}
        handle = model.vision_tower.register_forward_hook(
            lambda _m, _args, out: captured.__setitem__("feats", out))
        with torch.no_grad():
            out = model(input_ids, my_image)
        handle.remove()

        torch.testing.assert_close(
            captured["feats"], ref_feats, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        print(f"[rank{rank}] {mode}: encoder_dp(dp={world}) ViT + "
              f"LLM(dp={dp_size}×tp={tp_size}×ep={ep_size}) "
              f"| dp_rank={dp_rank} enc_rank={enc_rank} 编码全批第{enc_rank}张图 "
              f"| 桥接 gather [{P},{H}]→[{B * P},{H}] ✓ "
              f"| logits 对拍单卡 ✓")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
