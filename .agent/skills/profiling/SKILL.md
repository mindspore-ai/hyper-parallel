---
name: profiling
description: >
  Collect and break down training performance on Ascend: drive the profiler
  (MindStudio msprof / torch_npu profiler), parse the output file set, and
  break the step down by compute (Cube/Vector/FA), communication
  (TP/DP/EP/CP/PP, uncovered), free, and optimizer-update, at layer / module
  / operator views, into a visual HTML report. Use for profiling / 性能拆解 /
  性能瓶颈分析 / profiling 分析 / perf breakdown. The breakdown is
  rule-based and reproducible; the perf-playbook skill turns it into tuning
  手段.
---

# Profiling

Collect a profiling trace, parse its output file set, and break the step down
into where the time goes, so the `perf-playbook` skill can pick the tuning
手段. The breakdown is **rule-based and reproducible** — the same trace
produces the same breakdown — not a judgement call.

**This file is the index.** The output-file guide and breakdown dimensions
load on demand.

> 对齐状态：采集基于公开的 MindStudio **msprof** / `torch_npu.profiler`；
> 拆解的分类规则（哪些算子归 Cube/Vector/FA、哪些通信算子归哪个并行轴）在
> `references/breakdown-dimensions.md` 中给出默认规则，随实践补充；任何内部
> 特有的归类口径标 `[待对齐]`。

## Subcommands

```bash
python .agent/skills/profiling/scripts/<tool>.py <options>
```

| Intent | Tool | Purpose |
|---|---|---|
| 采集 profiling | `collect_prof.py --run-dir <d> --launch "<cmd>"` | drive the run under the profiler and gather the output set |
| 拆解 + HTML | `breakdown.py --prof-dir <d> --out report.html` | parse the trace, break the step down, emit the HTML report |

## References

| When | Read |
|---|---|
| profiling 输出文件清单 + 各文件内容与表头含义 | [references/output-files.md](references/output-files.md) |
| 拆解维度：计算/通信/Free/优化器 的分类规则 + Layer/Module/算子视角 | [references/breakdown-dimensions.md](references/breakdown-dimensions.md) |

## Report

`templates/perf-breakdown.html` — the visual breakdown: step time split by
compute / communication (uncovered) / free / optimizer with each part's
share, plus the layer / module / operator rollups. `breakdown.py` fills it
from the parsed trace.

## Hard rules

- **Breakdown before手段.** This skill names the bottleneck; it does not
  pick the fix — that is `perf-playbook`. Do not jump to a手段 from a raw
  trace.
- **Rule-based and reproducible.** The classification (operator → compute
  class, communication → parallel axis) is by documented rule; the same
  trace must produce the same breakdown. State the rule version used.
- **Collect correctly first.** A profiling run with warmup not skipped, or a
  single un-repeated step, is not a reliable trace; skip warmup and profile
  steady-state steps (default a few steps after warmup).
- **Reproducible evidence.** The breakdown carries the command, commit and
  environment of the profiled run; a breakdown with no provenance is not
  evidence (shared with `autoresearch`, `perf-playbook`).
- A classification needing an internal口径 stays `[待对齐]`; an unparsed
  field stays `[待解析]`. Never fill a share that was not computed.
