---
name: memory-analysis
description: >
  NPU memory analysis for training: collect a memory snapshot, break device
  memory down into static vs dynamic across the forward / backward /
  optimizer phases, locate the peak and its tensor-level composition, roll
  activation memory up by module, estimate the recompute saving, and flag
  fragmentation / leak / peak problems. Use for 显存分析 / 显存拆解 /
  显存峰值 / 显存碎片 / 显存泄露 / OOM 定位 / memory breakdown. Rule-based
  and reproducible; emits a visual HTML report.
---

# Memory Analysis

Collect a memory snapshot, break device memory down, locate the peak and what
composes it, and flag fragmentation / leak / peak problems. **Rule-based and
reproducible** — the same snapshot gives the same breakdown.

**This file is the index.** The snapshot guide and breakdown dimensions load
on demand.

> 对齐状态：采集基于公开的 NPU 内存快照（`torch_npu` memory snapshot /
> `torch.npu.memory._snapshot` 等价接口）；静/动态与阶段归类规则在
> `references/breakdown-dimensions.md`，内部特有口径标 `[待对齐]`。

## Subcommands

```bash
python .agent/skills/memory-analysis/scripts/<tool>.py <options>
```

| Intent | Tool | Purpose |
|---|---|---|
| 采集快照 | `collect_snapshot.py --run-dir <d> --launch "<cmd>"` | drive the run with snapshot recording (default steps 1-3) and gather it |
| 拆解 + HTML | `breakdown.py --snapshot <f> --out report.html` | parse the snapshot, break memory down, locate the peak, emit HTML |

## References

| When | Read |
|---|---|
| 内存快照内容与字段含义 | [references/snapshot-format.md](references/snapshot-format.md) |
| 静/动态、阶段、激活按 module、重计算收益 的拆解规则 | [references/breakdown-dimensions.md](references/breakdown-dimensions.md) |

## Report

`templates/memory-breakdown.html` — static vs dynamic memory across
forward / backward / optimizer, the peak point with its tensor-level
composition, the per-module activation rollup, and the estimated
fine-grained recompute saving.

## What it covers

- **静态 vs 动态**：参数 / 优化器状态 / 常驻 buffer（静态）vs 激活 / 临时
  张量（动态），按前向/反向/优化器阶段分。
- **峰值定位**：峰值出现的位置（阶段 / 调用点）与其显存构成，拆到 Tensor
  粒度（哪些张量在峰值时刻常驻）。
- **激活按 module**：每个 module 的激活显存占用，据此估算细粒度重算能拿到
  的显存收益。
- **碎片 / 泄露 / 峰值**：碎片分析（已分配 vs 保留 vs 空洞）、泄露定位
  （跨 step 不释放的增长）、峰值分析。

## Hard rules

- **Default steps 1-3.** Snapshot the first few steps unless a specific phase
  is targeted; a single-step snapshot misses the steady-state peak.
- **Rule-based and reproducible.** Static/dynamic and phase classification is
  by documented rule; the same snapshot gives the same breakdown. State the
  rule version.
- **Peak is the first question.** Locate the peak and its composition before
  proposing a fix; a leak and a one-off peak have different fixes.
- **Reproducible evidence.** The breakdown carries the command, commit and
  environment of the snapshotted run (shared with `autoresearch`,
  `profiling`, `perf-playbook`).
- A classification needing an internal口径 stays `[待对齐]`; an unparsed
  field stays `[待解析]`. Never fill a number that was not computed. Memory
  手段 selection (recompute / swap / offload) is `perf-playbook`'s job.
