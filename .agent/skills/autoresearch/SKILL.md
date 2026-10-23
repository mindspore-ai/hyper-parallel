---
name: autoresearch
description: >
  Bounded autonomous-experiment loop for kernel / graph optimization:
  benchmark + correctness gate + keep/discard/crash decision + append-only
  bookkeeping + commit/restore git protocol, with four rules enforced in
  code rather than left to agent discipline. Use for 自动实验/性能优化循环/
  autoresearch/算子调优迭代, or any "measure a pending edit against the best
  recorded result and keep it only if it wins and the gate passes" workflow.
  The agent owns the creative half (pick an idea, make the edit); this skill
  owns the deterministic half.
---

# Autoresearch

A bounded autonomous-experiment loop. One iteration = one optimization
experiment: edit the single declared target file, benchmark it, pass the
correctness gate, record the outcome, and keep / discard / crash. Runs until
the agent stops. Design follows torchtitan's `graph_trainer/autoresearch`,
promoting its deterministic protocol from prose into code.

**This file is the index + hard rules.** The per-run operating manual is
scaffolded into each run directory as `manual.md` (fill its `[SETUP]`).

## Four rules enforced in code

These come from real incidents, not agent memory:

1. **Benchmarks only measure committed code.** Remote benchmarks sync by
   commit; an uncommitted edit silently measures the previous state, so
   `iterate` commits the experiment first.
2. **Positive-evidence markers are required** (e.g. `engaged=1`). Their
   absence is a `crash`, never a quiet pass — a fallback run and an
   optimized run are indistinguishable by "no warning".
3. **A failed gate voids the metric**, however good it looks.
4. **Failures stay in history.** `discard`/`crash` restore the target file
   with a new forward commit, never by rewriting history; the best-so-far
   state is always HEAD.

Two manual-level prohibitions keep the agent from leaking its own answers
(scaffolded into `manual.md`): do not inspect git history (`results.tsv` is
the only record of prior experiments), and do not read the gate script's
source (run it, read only its assertion output).

## Subcommands

```bash
python .agent/skills/autoresearch/scripts/autoresearch.py <subcommand> [options]
```

| Intent | Command | Purpose |
|---|---|---|
| 搭建一个 run 目录 | `init --run-dir <d> --target <f> --benchmark-cmd "<c>"` | scaffold run.json + living docs |
| 记录基线 | `baseline --run-dir <d>` | benchmark the current commit as the first keep |
| 测一次待定改动 | `iterate --run-dir <d> --description "<what>"` | commit→gate→benchmark→decide→record→keep/restore |
| 看进度 | `status --run-dir <d>` | summarize the run so far |

`iterate` expects the experimental edit to sit **uncommitted** in the run's
target files, with nothing else dirty outside the run directory.

## run.json contract

- `benchmark_cmd` / `gate_cmd`: **argv lists** (shell constructs live inside
  the invoked script, never in config).
- `metric_pattern`: regex with a `(?P<metric>...)` group; the last match wins.
- `metric_lower_is_better`, `noise_fraction`: direction and the discard band.
- `noise_confirm_reruns` (default 0): when the first sample lands inside the
  noise band, rerun N times and decide on the median.
- `aux_patterns`: `{name: regex-with-(?P<value>...)}`; captured values ride
  along in the result row's description.
- `bench_must_match` / `gate_must_match`: literal markers; absence = crash.

## The experiment loop (agent side)

1. Re-read `ideas.md`, `learnings.md`, `experiment_log.md`; pick the next idea.
2. Edit the target file(s) — keep the tree dirty only there.
3. `iterate --run-dir <d> --description "..."` — the tool owns the rest.
4. Append an `experiment_log.md` entry; update `ideas.md` / `learnings.md`.
Out of ideas → profile, study the problem, try another angle. Never stop
until the operator stops you.

## Hard rules (must not violate)

- Never edit `config.py` / `loop.py` (the engine) to make a run pass.
- Benchmarks and gates are argv lists — no `shell=True`, no inline pipes.
- A within-noise result is `discard`, not a forced keep.
- Keep the engine dependency-free (stdlib only) and pylint-clean.

## Field record

Two field runs drove the DeepSeek-V4.1 fused Lightning-Indexer adapter: the
first took a four-probe metric 697→123ms over three experiments; the second
(master baseline, six probes including long sequences) took 4617→261ms and
caught a wrong bf16-scoring optimization at the gate (crash, logged). Both
runs' product changes were delivered through the fused-indexer PR; the run
directories (`autoresearch/`, `autoresearch2/`) carry the full logs.

## Unit tests

`tests/ut/tools/test_autoresearch_loop.py` drives baseline / keep / discard /
crash / dirty-tree / noise-confirm / aux-capture against a temporary git
repository (CPU, no accelerator). It puts the engine directory on `sys.path`
the same way the entry script does.
