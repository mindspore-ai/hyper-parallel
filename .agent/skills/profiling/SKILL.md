---
name: profiling
description: >
  Collect and break down training performance on Ascend: drive the profiler
  (torch_npu.profiler, or MindStudio msprof), parse the output file set, and
  break the step down into computing / uncovered communication / free with the
  optimizer share, plus a compute-class split (Cube/Vector/AI_CPU) and
  per-link communication bandwidth, into an HTML report. Use for profiling /
  性能拆解 / 性能瓶颈分析 / profiling 分析 / perf breakdown. The breakdown is
  rule-based, cross-checked against the profiler's own invariants, and
  reproducible; the perf-playbook skill turns it into tuning 手段.
---

# Profiling

Collect a profiling trace, parse its output file set, and break the step down
into where the time goes, so the `perf-playbook` skill can pick the tuning
手段. The breakdown is **rule-based and reproducible** — the same trace
produces the same breakdown — not a judgement call.

**This file is the index.** The output-file guide and breakdown dimensions
load on demand.

> 对齐状态：文件名、表头与加和关系均在 **CANN 9.0.0 + torch_npu 2.9.0.post2 /
> Ascend 910** 上实采核对；分类规则见
> `references/breakdown-dimensions.md`（规则版本 v2）。仍需内部口径的项标
> `[待对齐]`（当前：FA 二级细分、多通信组的轴 join、Module/Layer 视角）。

## Subcommands

```bash
python .agent/skills/profiling/scripts/<tool>.py <options>
```

| Intent | Tool | Purpose |
|---|---|---|
| 采集 | `collect_prof.py --run-dir <d> --launch "<cmd>" [--warmup 3 --active 3]` | 驱动被测命令并校验输出完整（`analyse.done` + `step_trace_time.csv`）|
| 训练脚本内挂钩 | `prof_hook.profile_steps()` | 训练进程内的 profiler 上下文，读 `collect_prof` 导出的 `PROF_*` |
| 拆解 + HTML | `breakdown.py --prof-dir <d> --out report.html [--json rec.json]` | 解析 trace、拆解步时、出 HTML（退出码 0=校验全通过 / 2=有校验不通过 / 1=没找到输出）|

`--prof-dir` 可传 `ASCEND_PROFILER_OUTPUT`、`*_ascend_pt` 目录、或装着多个
rank 目录的父目录（多卡逐 rank 解析）。

训练脚本接入：

```python
import sys
sys.path.insert(0, ".agent/skills/profiling/scripts")
from prof_hook import profile_steps

with profile_steps() as prof:
    for batch in loader:
        train_step(batch)
        prof.step()          # 必须每步调用，否则 warmup/active 不推进
```

## Capture inside HyperParallel

Within this repository prefer the Trainer's native hook for capture:
`--profiling.enabled=true --profiling.start_step=N --profiling.end_step=M
--profiling.trace_dir=DIR` (ProfilingCallback steps the profiler once per
optimizer step on one rank). `collect_prof.py --mode inrun` requires the
training script itself to call `prof_hook.profile_steps`, so it fits
standalone scripts, not the stock Trainer entry.

Invariant semantics on real workloads: the optimizer-interval count is an
integer multiple of the captured steps (the house Muon optimizer runs
muon+adamw, two intervals per step); the Stage identity tolerance scales
with the stage time; and communication.json vs step_trace is a reported
ratio, not an equality — overlap and multi-rank aggregation separate the
two measures (1.77x observed on an 8-rank FSDP run with both parses
correct).

## References

| When | Read |
|---|---|
| profiling 输出文件清单 + 各文件内容与表头含义（含实测恒等式与易错点）| [references/output-files.md](references/output-files.md) |
| 拆解维度：加和关系、Core Type 分类、通信读法、一致性校验 | [references/breakdown-dimensions.md](references/breakdown-dimensions.md) |

## Report

`templates/perf-breakdown.html` — 六节：步时拆解、计算维度、AI_CPU 回退热点、
通信维度（含链路带宽）、一致性校验、瓶颈结论。`breakdown.py` 从 trace 填充，
并把采集 provenance（CANN/torch_npu 版本、schedule、并行组）写在页首。

## Hard rules

- **Breakdown before 手段.** This skill names the bottleneck; it does not pick
  the fix — that is `perf-playbook`. Do not jump to a 手段 from a raw trace.
- **Rule-based and reproducible.** The classification is by documented rule and
  the same trace must produce the same breakdown. State the rule version used.
- **Collect correctly first.** Warmup must be skipped and at least two
  steady-state steps profiled; `Level1` or above, or there is no
  `step_trace_time.csv` and so no step breakdown. These are machine-checked —
  `breakdown.py` reports them as consistency checks and exits 2 on failure.
- **校验不通过的数字不采信.** The report's consistency section states which
  invariants held (`Stage` identity with a per-step floor plus stage-scaled
  tolerance, the total-vs-uncovered communication bound, schedule, optimizer
  intervals as an integer multiple of the steps; the cross-file communication
  ratio is reported, not gated). A breakdown with a failed check is a parse
  problem, not a finding.
- **Invariant or tolerance changes need two operating points.** Calibrate any
  check-semantics change on at least two step-time regimes before shipping it
  (field basis: a ~2.6s step on a 9B run and a ~0.51s step on a 0.36B run —
  the benign `Stage` identity drift is a per-step constant of tens of µs, so
  a purely relative tolerance fails fast-stepping models and a tight absolute
  one fails slow ones). A change validated on one model scale is single-sided.
- **Reproducible evidence.** The breakdown carries the command, commit and
  environment of the profiled run; a breakdown with no provenance is not
  evidence (shared with `autoresearch`, `perf-playbook`).
- 优化器是 `Computing` 的子集，**不是第四个加和项**；加和维度只有计算 /
  未掩盖通信 / Free 三项。
- AI_CPU 类别里先剔掉 HCCL 的 `*AicpuKernel` 再谈「回退」——它在 AI CPU 上跑，
  但属于通信。
- A classification needing an internal 口径 stays `[待对齐]`; an unparsed field
  stays `[待解析]`. Never fill a share that was not computed.
