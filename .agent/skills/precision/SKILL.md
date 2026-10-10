---
name: precision
description: >
  Precision comparison and problem localization for model bring-up on Ascend:
  collect dump logs (MindStudio msprobe), compare two or more dumps to find
  the first module / operator / line where precision diverges, and localize
  common precision problems (nan, loss spike, non-convergence). Use for
  精度比对 / 精度对齐 / 精度问题定位 / dump 比对 / nan / loss spike /
  不收敛 / precision debug. Produces a precision report from the template;
  the acceptance 口径 is the precision-acceptance rule.
---

# Precision

Collect precision evidence, compare it against a baseline to find the first
divergence, and localize precision problems. The acceptance bar is
`rules/precision-acceptance.md` (1000-step error band plus grad_norm,
first-step, determinism and resume checks); this skill produces the evidence
that rule judges.

**This file is the index.** The flow and triage references load on demand.

> 对齐状态：采集/比对基于公开的 MindStudio **msprobe** 工具链；与某内部
> 精度对齐实现（precision_align）对齐的专有判据与打点策略，标记为
> `[待对齐]`，待内部逻辑提供后补实。本 skill 当前可用于标准 msprobe 流程。

## Subcommands

```bash
python .agent/skills/precision/scripts/<tool>.py <options>
```

| Intent | Tool | Purpose |
|---|---|---|
| 采集 dump | `collect_dump.py --run-dir <d> --launch "<cmd>"` | drive the run under an msprobe dump config and gather the dump |
| 比对两份/多份 | `diff_dump.py --candidate <d1> --baseline <d2>` | run msprobe compare, report the first diverging API / module |
| 定位精度问题 | 见 `references/triage-nan-spike-diverge.md` | nan / spike / 不收敛 的定位流程 |

## References

| When | Read |
|---|---|
| 采集→比对→首个异常 的标准流程（msprobe）| [references/align-flow.md](references/align-flow.md) |
| nan / loss spike / 不收敛 的定位思路 | [references/triage-nan-spike-diverge.md](references/triage-nan-spike-diverge.md) |
| 验收口径（1000 step 等）| `rules/precision-acceptance.md` |

## Report

`templates/precision-report.md` — the precision analysis report: what was
compared, the first diverging module / op / line, the error numbers against
the stated baseline, and the next-step fix advice. Every number carries its
command, commit and environment.

## Hard rules

- **Baseline is explicit and stated.** A precision result without the
  baseline it was compared to (same model / config / parallel / data order,
  one axis differing) is not a result. Record both sides' commit, config and
  environment.
- **Evidence is reproducible.** Every number carries command + commit +
  environment; a bare number is not evidence (shared with `autoresearch`,
  `gate-doctor`, `feature-doc`, `precision-acceptance`).
- **First divergence, not last symptom.** Report the earliest module / op /
  line where candidate and baseline diverge beyond the band; a downstream nan
  is a symptom, the first divergence is the cause.
- **Acceptance 口径 comes from the rule**, not an ad-hoc bar; link to
  `precision-acceptance` and state which criterion (abs / rel) was used.
- **bf16 ties are known noise.** Divergence only on tie positions whose
  reference-score gap is below one bf16 ulp is within acceptance; state the
  tie count and worst gap rather than claiming bitwise identity.
- An unmeasured result stays `[待测]`; a step needing the internal align
  logic stays `[待对齐]`. Never fill either with a plausible number.
