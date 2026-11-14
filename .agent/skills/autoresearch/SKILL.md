---
name: autoresearch
description: >
  Bounded autonomous-experiment loop for kernel / graph optimization:
  benchmark + correctness gate + keep/discard/crash decision + append-only
  bookkeeping + commit/restore git protocol, with four rules enforced in
  code rather than left to agent discipline. Use for automated experiments,
  performance optimization loops, iterative operator tuning, or any "measure a pending edit against the best
  recorded result and keep it only if it wins and the gate passes" workflow.
  The agent owns the creative half (pick an idea, make the edit); this skill
  owns the deterministic half.
---

# Autoresearch

A bounded autonomous-experiment loop. One iteration = one optimization
experiment: edit the declared target file(s), pass the correctness gate,
benchmark the candidate, record the outcome, and keep / discard / crash.
The Agent drives iterations within the declared budget. Design follows torchtitan's `graph_trainer/autoresearch`,
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

Two manual-level prohibitions protect blind runs (scaffolded into
`manual.md`). Use the current run's artifacts for prior experiments, and
consume the gate script through its assertion output.

## Subcommands

```bash
python .agent/skills/autoresearch/scripts/autoresearch.py <subcommand> [options]
```

| Intent | Command | Purpose |
|---|---|---|
| Initialize a run directory | `init --run-dir <d> --target <f> --benchmark-cmd "<c>"` | scaffold run.json + living docs |
| Record a baseline | `baseline --run-dir <d>` | benchmark the current commit as the first keep |
| Evaluate a pending change | `iterate --run-dir <d> --description "<what>"` | commit→gate→benchmark→decide→record→keep/restore |
| Inspect progress | `status --run-dir <d>` | summarize the run so far |

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
- `training_protocol`: optional workload ID and full-window step counts.
  Use it with a separate `gate_cmd` and the
  [structured training receipt](../perf-playbook/references/training-rsi.md#structured-training-receipts).

## The experiment loop (agent side)

1. Read `status`, `ideas.md`, `learnings.md`, and `experiment_log.md`.
   Review any `retry_ready` candidates before selecting an unrelated idea.
2. Edit the target file(s) — keep the tree dirty only there.
3. `iterate --run-dir <d> --description "..."` — the tool owns the rest.
4. Append an `experiment_log.md` entry; update `ideas.md` / `learnings.md`.
Continue within the declared budget. Record which targets passed, which
remain open, and why the loop stopped.

## Candidate receipts and retries

Each `iterate` appends a receipt to `candidates.jsonl` alongside the existing
metric history in `results.tsv`. It stores the parent and candidate commits,
decision stage and result, gate status, required markers, and configured
auxiliary captures. Configure an auxiliary capture for a concise failure
reason, engagement result, or evidence identifier when the runner emits one.
Training runs using the structured contract also record the baseline and every
gate or benchmark invocation, including failed noise-confirmation attempts.

- Tag changes with repeatable `--changes <scope>` flags. Use stable scope
  names within a run, such as `optimizer-residency` or `operator-path`.
- A candidate with `--revisit-on <scope>` becomes `retry_ready` when a later
  **kept** candidate changes that scope. Both `iterate` and `status` show
  eligible commits. Record the condition and the intended reapplication
  in the experiment log.
- Reapply the idea to the current kept configuration from the run's own
  notes or patch. Run it as a new candidate with `--retry-of <commit>` and
  the appropriate gate. This consumes that retry entry. A subsequent retry
  can declare its own trigger. Use the run's normal resource admission
  and budget for every retry.

For training runs, follow the [training RSI protocol](../perf-playbook/references/training-rsi.md).
Use [MFU Riser](../mfu-riser/SKILL.md) when the task needs architecture FLOPs
audits, linked capacity series, or screening followed by performance acceptance.

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
runs informed the fused-indexer integration. The original run artifacts were
stored in external `autoresearch/` and `autoresearch2/` directories; these
numbers are historical context, not benchmark results reproduced by this
checkout. The unit tests below exercise the experiment protocol locally.

## Unit tests

`tests/ut/tools/test_autoresearch_loop.py` drives baseline / keep / discard /
crash / dirty-tree / noise-confirm / aux-capture against a temporary git
repository (CPU, no accelerator). It puts the engine directory on `sys.path`
the same way the entry script does.
