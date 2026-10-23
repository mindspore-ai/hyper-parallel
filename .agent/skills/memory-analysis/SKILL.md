---
name: memory-analysis
description: >
  NPU memory analysis for training: record allocation history, dump snapshots
  at chosen moments (including mid-backward), and break device memory down
  into parameters / gradients / optimizer state / activations with each large
  block attributed to the call site that allocated it, plus fragmentation,
  cross-step growth for leak triage, and a check on whether the snapshot is
  the peak at all. Use for 显存分析 / 显存拆解 / 显存峰值 / 显存碎片 /
  显存泄露 / OOM 定位 / memory breakdown. Rule-based, cross-checked against
  the allocator's own invariants, and reproducible; emits a visual HTML
  report.
---

# Memory Analysis

Record allocation history, dump a snapshot at the moment that matters, and
break device memory down into what is holding it and which line allocated it.
**Rule-based and reproducible** — the same snapshot gives the same breakdown.

**This file is the index.** The snapshot guide and breakdown dimensions load
on demand.

> 对齐状态：字段、帧形态与恒等式均在 **torch_npu 2.9.0.post2 + CANN 9.0.0 /
> Ascend 910** 上实采核对，分类规则对着一次已知构成的运行逐项对账（见
> `references/breakdown-dimensions.md` § 实测对账）。规则版本 v2。

## Subcommands

```bash
python .agent/skills/memory-analysis/scripts/<tool>.py <options>
```

| Intent | Tool | Purpose |
|---|---|---|
| 采集 | `collect_snapshot.py --run-dir <d> --launch "<cmd>" [--steps 1-3]` | 驱动被测命令并校验每份 dump 都带计数器旁车 |
| 训练脚本内挂钩 | `mem_hook.start()` / `step()` / `dump(tag)` | 进程内记录与落盘，**任意时刻可 dump** |
| 拆解 + HTML | `breakdown.py --snapshot <f> [--compare <f2>] [--json r.json]` | 解析快照、分类、归属、校验（退出码 0=全通过 / 2=有不通过 / 1=文件缺失）|

训练脚本接入：

```python
import sys
sys.path.insert(0, ".agent/skills/memory-analysis/scripts")
import mem_hook

mem_hook.start()                      # 开始记录；读 collect_snapshot 导出的 MEM_* 变量
for step, batch in enumerate(loader):
    out = model(batch)
    out.register_hook(lambda g: mem_hook.dump("midbwd") and None)  # 想看激活就在此处
    loss(out).backward(); opt.step(); opt.zero_grad()
    mem_hook.step()                   # 落在配置窗口内的 step 自动 dump
mem_hook.stop()
```

## References

| When | Read |
|---|---|
| 快照字段、帧形态、落盘坑、实测恒等式 | [references/snapshot-format.md](references/snapshot-format.md) |
| 总量三层、类别判定规则与实测对账、归属规则、校验项 | [references/breakdown-dimensions.md](references/breakdown-dimensions.md) |

## Report

`templates/memory-breakdown.html` — 六节：总量与碎片（含与峰值的差距）、
按类别、最大活跃块及其调用点、跨快照增长、一致性校验、结论。

## What it covers

- **总量与碎片**：Reserved / Allocated / Active 三层与碎片量。
- **按类别**：参数 / 梯度 / 优化器状态 / 激活 / 其他用户张量 / unclassified，
  判据见 references，已对着已知构成的运行逐项对账。
- **块级归属**：最大活跃块各自归到**分配它的那行自有代码**（梯度无 Python 帧，
  归到 `AccumulateGrad`）。
- **跨 step 增长**：`--compare` 比对两份快照，区分泄露与一次性大分配。
- **是否峰值**：把快照活跃量与 `max_memory_allocated` 对比并报差距——
  步后快照里激活为 0，实测比峰值低 30 MiB，不标出来就会被误读。

### 尚未实现（不要当成已有能力）

- **Module / Layer 粒度汇总**：当前归属到调用点（文件:行），未按 module 名聚合。
- **重计算收益估算**：需要 module 粒度激活量，依赖上一项。
- **峰值时刻构成**：当前只标注快照与峰值的差距；回放 `device_traces` 还原峰值
  构成的口径 `[待对齐]`。

## Hard rules

- **快照不是峰值。** 报告必须同时给出 `max_memory_allocated`；差距超 2% 即校验
  不通过。要看激活必须在 backward 期间 dump。
- **`frames[0]` 不是归属。** 记录栈最内层是快照机制自身，取第 0 帧会让每个块都
  归到同一处；按 references 的规则扫帧。
- **梯度按 native 帧的 `AccumulateGrad` 判定**——它没有任何 Python 帧，按 Python
  帧分类会整类漏掉。
- **Default steps 1-3** 且至少两份以便比对；单份快照分不清泄露与一次性峰值。
- **校验不通过的数字不采信。** 四条恒等式（逐段与全局、与进程计数器）是解析
  正确的正面证据。
- **Reproducible evidence.** 拆解带被采集运行的命令、commit 与环境
  （与 `autoresearch` / `profiling` / `perf-playbook` 共享该纪律）。
- **换省显存手段必须同时量化性能。** 本 skill 只定位占用来源；手段选择是
  `perf-playbook` 的事，且显存降了但性能掉了不算收益。
- A classification needing an internal 口径 stays `[待对齐]`; an unparsed field
  stays `[待解析]`. Never fill a number that was not computed.
