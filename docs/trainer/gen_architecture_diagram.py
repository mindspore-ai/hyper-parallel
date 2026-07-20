# -*- coding: utf-8 -*-
"""生成 Hyper-Parallel 整体架构图（docs/architecture_overview.png）。

依据 docs/detailed_design/ 01-06 + architecture_overview.md。
运行：python3 docs/gen_architecture_diagram.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

C_DONE = "#E2EFDA"   # 已实现：绿
C_PART = "#FFF2CC"   # 部分实现：黄
C_TODO = "#DEEBF7"   # 待实现：蓝灰
C_EDGE = "#4A4A4A"
C_BAND = "#F7F7F7"

fig, ax = plt.subplots(figsize=(17, 12), dpi=180)
ax.set_xlim(0, 100)
ax.set_ylim(0, 76.5)
ax.axis("off")

LX = 13.5          # 层内盒子起始 x（左侧留给层标签）
LW = 71.5          # 层内盒子区宽度


def band(y, h, label, sub):
    ax.add_patch(FancyBboxPatch((1, y), 97.5, h, boxstyle="round,pad=0.15",
                                fc=C_BAND, ec="#BBBBBB", lw=0.8, zorder=1))
    ax.text(2.2, y + h - 1.2, label, fontsize=10.5, fontweight="bold", va="top",
            zorder=3, linespacing=1.35)
    ax.text(2.2, y + 1.0, sub, fontsize=6.8, color="#666666", va="bottom",
            zorder=3, linespacing=1.35)


def box(x, y, w, h, title, lines, status, fs_title=9, fs_body=7.0):
    fc = {"done": C_DONE, "part": C_PART, "todo": C_TODO}[status]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25",
                                fc=fc, ec=C_EDGE, lw=1.1, zorder=4))
    ax.text(x + w / 2, y + h - 1.3, title, fontsize=fs_title, fontweight="bold",
            ha="center", va="top", zorder=5)
    if lines:
        ax.text(x + w / 2, y + h - 3.0, "\n".join(lines), fontsize=fs_body,
                ha="center", va="top", color="#333333", zorder=5, linespacing=1.5)


def arrow(x1, y1, x2, y2, color="#555555", lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, color=color, lw=lw, zorder=6))


def row(y, h, boxes, gap=1.6, fs_title=9):
    """在一层内均匀摆放盒子。boxes: [(title, lines, status), ...]"""
    n = len(boxes)
    w = (LW - gap * (n - 1)) / n
    for i, (t, lines, st) in enumerate(boxes):
        box(LX + i * (w + gap), y, w, h, t, lines, st, fs_title=fs_title)
    return w


# ── 标题与图例 ─────────────────────────────────────────────────────────────
ax.text(50, 75.0, "Hyper-Parallel 整体架构", fontsize=19, fontweight="bold", ha="center")
ax.text(50, 73.0, "依据 docs/detailed_design/ 01–06（2026-07-22 定稿口径）",
        fontsize=10, color="#666666", ha="center")
for x, c, t in [(58, C_DONE, "已实现（282 用例全绿）"), (73.5, C_PART, "部分实现"), (84, C_TODO, "设计定稿·待实现")]:
    ax.add_patch(FancyBboxPatch((x, 70.6), 2.0, 1.5, boxstyle="round,pad=0.1", fc=c, ec=C_EDGE, lw=0.8))
    ax.text(x + 2.5, 71.35, t, fontsize=8.5, va="center")

# ── L1 用户接口 ────────────────────────────────────────────────────────────
band(63.0, 6.2, "L1\n用户接口", "一个 YAML 启动训练")
row(63.6, 5.0, [
    ("train.yaml", ["_target_ 声明式组件装配"], "todo"),
    ("CLI 入口（S12）", ["torchrun + 配置模板"], "todo"),
])

# ── L2 Recipe 编排 ─────────────────────────────────────────────────────────
band(54.8, 6.8, "L2\nRecipe 编排", "03 · 顺序以 01 §4.1 为准")
row(55.3, 5.6, [
    ("FinetuneRecipe（待实现）", ["setup() ④.1–④.14 编排", "run_train_validation_loop ⑤"], "todo"),
    ("BaseRecipe（待实现）", ["__state_tracked 自动状态追踪", "save / load_checkpoint"], "todo"),
    ("StepScheduler（待实现）", ["grad_acc 分组 · ckpt/val 节奏", "SIGTERM 分布式协调"], "todo"),
])

# ── L3 训练组件 ────────────────────────────────────────────────────────────
band(46.6, 6.8, "L3\n训练组件", "typed build / untyped instantiate")
row(47.1, 5.6, [
    ("配置系统 01（待实现）", ["ConfigNode · RecipeConfig", "components/config/node.py"], "todo"),
    ("数据管道 02（待实现）", ["build_dataloader · packing", "sampler · default_collater"], "todo"),
    ("Checkpoint 04（待实现）", ["Checkpointer · DCP 切分", "OptimizerState: list[Optimizer]"], "todo"),
    ("Optim / Loss 03（待实现）", ["list[Optimizer] canonical", "LR 调度 · MaskedCE"], "todo"),
])

# ── L4 模型构建与并行策略 ──────────────────────────────────────────────────
band(36.4, 8.8, "L4\n模型构建与并行", "01 §8.3：①PP→⑤→⑥→⑨")
row(36.9, 7.2, [
    ("HyperAutoModel 01（待实现）", ["from_pretrained", "meta 空壳 → load_base_model", "registry 懒加载"], "todo"),
    ("ShardingPlanner 05（已实现）", ["ParamRole×14 · ARCH_OVERRIDES", "边界推断→模板匹配→链式传播", "MLA 覆盖（D-13）"], "done"),
    ("apply_sharding_plan 05（已实现）", ["Phase A 分片（D-10 分流）", "Phase B handler · Phase C 包装", "→ (model, tp_grad_info)"], "done"),
    ("FSDP2Manager 06（待实现）", ["per-block DP 包裹", "fsdp2_strategy_parallelize", "※ D-12：梯度同步二选一"], "todo"),
], gap=1.4)
_w4 = (LW - 1.4 * 3) / 4
for i in range(3):
    x1 = LX + i * (_w4 + 1.4) + _w4 + 0.1
    arrow(x1, 40.5, x1 + 1.2, 40.5)

# ── L5 分布式核心 ──────────────────────────────────────────────────────────
band(26.2, 8.8, "L5\n分布式核心", "主 mesh 无 EP 轴\nexpert mesh apply 期派生")
row(26.7, 7.2, [
    ("MeshContext 06（待实现）", ["init_device_mesh", "mesh_dim_names + rank_list", "唯一 mesh 构建点"], "todo"),
    ("PrecompiledBoundary（已实现）", ["边界通信预编译", "RedistOp · identity 解包"], "done"),
    ("cp_utils（已实现）", ["shard_batch_for_cp", "flex_cp_allgather", "all-gather K/V（D-01''）"], "done"),
    ("ep_utils（已实现）", ["a2a 后端分派", "MOE_ROUTER_ADAPTERS", "_hf_native_ep_compute"], "done"),
    ("FSDP2 / DTensor（core 已实现）", ["DTENSOR_UNIFIED", "_orig_dtensor_placements", "DeviceMesh.concatenate"], "done"),
], gap=1.3, fs_title=7.8)

# ── L6 运行时 ──────────────────────────────────────────────────────────────
band(18.6, 6.2, "L6\n运行时", "NCCL·HCCL / gloo")
row(19.1, 5.0, [
    ("PyTorch（torch.distributed.fsdp / init_device_mesh）", [], "todo"),
])

# ── 层间箭头 ───────────────────────────────────────────────────────────────
cx = LX + LW / 2
arrow(cx - 12, 62.9, cx - 12, 61.9)   # L1 → L2
arrow(cx + 12, 62.9, cx + 12, 61.9)
arrow(cx - 18, 54.7, cx - 18, 53.7)   # L2 → L3
arrow(cx, 54.7, cx, 53.7)
arrow(cx + 18, 54.7, cx + 18, 53.7)
arrow(cx - 20, 46.5, cx - 20, 45.5)   # L3 → L4
arrow(cx + 8, 46.5, cx + 8, 45.5)
arrow(cx - 16, 36.3, cx - 16, 35.3)   # L4 → L5
arrow(cx + 4, 36.3, cx + 4, 35.3)
arrow(cx + 24, 36.3, cx + 24, 35.3)
arrow(cx, 26.1, cx, 25.1)             # L5 → L6

# ── 底部注解条 ─────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((1, 1.5), 97.5, 15.5, boxstyle="round,pad=0.2",
                            fc="#FFFFFF", ec="#999999", lw=1.0, zorder=2))
ax.text(3, 15.4, "关键设计", fontsize=11, fontweight="bold", va="top", zorder=3)
col1 = ("双模式（05 核心）\n"
        "· 生产模式：_local_params_context 零拷贝解包\n"
        "  + PrecompiledBoundary 预编译边界（性能优先）\n"
        "· 校验模式：纯 DTensor __torch_dispatch__\n"
        "  （stock PyTorch 可跑，正确性优先）\n"
        "· 等价性验证：testing/grad_equiv.py")
col2 = ("已定稿决策\n"
        "· CP（D-01''）：all-gather K/V，ring 已否决\n"
        "· EP（D-10）：TP-extend-EP，expert 仅 {EP: Shard(0)}，\n"
        "  派生 (edp, ep) expert mesh，Megatron ETP=1 同构\n"
        "· optimizer：list[Optimizer]（03 canonical）\n"
        "· RecipeConfig canonical：01 §3.3")
col3 = ("待决点（D-12）\n"
        "TP-Replicate 参数梯度同步二选一：\n"
        "a) 调时序走 DTENSOR_UNIFIED\n"
        "b) fully_shard(tp_grad_info=...) 消费端\n"
        "（实现 06 parallelize 时决策）\n"
        "\n"
        "进度：已实现 25 / 部分 8 / 未实现 86（110 人·日）")
ax.text(3, 13.0, col1, fontsize=8.2, va="top", zorder=3, linespacing=1.55)
ax.text(38, 13.0, col2, fontsize=8.2, va="top", zorder=3, linespacing=1.55)
ax.text(71, 13.0, col3, fontsize=8.2, va="top", zorder=3, linespacing=1.55)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture_overview.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved:", out)
