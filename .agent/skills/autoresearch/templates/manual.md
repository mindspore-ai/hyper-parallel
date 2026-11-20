# Autoresearch operating manual

This manual guides the Agent. In each iteration, implement one candidate within the
declared target files, run the gate and benchmark, and record `keep`, `discard`, or
`crash`. Continue within the run's budget.

## [SETUP] Run configuration

- **Optimization objective:** <FILL: what to optimize>
- **Primary metric:** <FILL: benchmark_cmd output field and optimization direction>
- **Starting point:** <FILL: current implementation and known performance>
- **Scaffolding level:** <FILL: readable references and access to the network or history>
- **Budget and stopping conditions:** <FILL: time or trial count, improvement over baseline, and independent performance target if applicable>
- **Measurement windows:** <FILL: correctness and performance windows, topology, probes, routing, and metric convention>
- **Change classes and gates:** <FILL: evidence for equivalent implementations, numerical tolerances, or training-policy changes>
- **Retry tags:** <FILL: stable configuration scopes used in this run, such as optimizer-residency>

## Editable files

Use `target_files` in this directory's `run.json`. The tool refuses to benchmark
uncommitted changes outside that list.

**Do not modify:** <FILL: reference implementation, gate scripts, trainer, or other
protected files whose changes would alter the evaluation>.
Analysis tools may be added under this run directory's `scripts/`.

## Read scope

**May read:** <FILL: target files, read-only reference implementation, current run files, and self-generated profiles>
**Do not read:** <FILL: unrelated subsystems; record a reason in experiment_log before expanding scope>

## History and gate-source restrictions

These requirements cannot be relaxed by `[SETUP]`.

- **Do not inspect Git history.** `git log`, `git log -p`, `git show <commit>`,
  `git diff <commit>`, and `git blame` are prohibited. Prior commit titles and diffstats
  can reveal optimizations that this run is intended to rediscover independently.
  Working-tree `git status` and `git diff` without commit references are allowed.
  Read this run's commits and failure evidence from `results.tsv`, `candidates.jsonl`,
  and its own experiment logs.
- **Do not read gate-script source.** Execute the gate and use assertion messages
  in its output to diagnose failures. Symbols in gate source can reveal previous
  optimization choices.
- Treat material with uncertain read authorization as out of scope.

## Benchmark and correctness gate

`run.json` defines `benchmark_cmd` and `gate_cmd`. Missing positive markers from
`bench_must_match` or `gate_must_match`, such as `engaged=1`, result in `crash`.
Absence of warnings does not establish that an optimization executed. For a first
measurement inside the noise band, configured automatic repeats use the median before
deciding (`noise_confirm_reruns`). Do not override this with a single reading.

## Records

The tool appends `results.tsv` and `candidates.jsonl` and manages candidate commits and
restoration. After every iteration, append `experiment_log.md` and update `ideas.md`
and `learnings.md`. Preserve a reusable change description or a patch produced within
this run for candidates that may be retried.

## Experiment loop

Within the declared budget:

1. Run `status`, check `retry_ready` candidates first, then choose the next experiment
   using `ideas.md`, `learnings.md`, and `experiment_log.md`.
2. Implement the change in `target_files`, leaving only those files modified. When
   context is limited, implementation and execution may be delegated to a subagent.
   Pass this manual's restrictions and commands verbatim; the main loop receives the
   result rather than rewriting the rules for the delegate.
3. Run `autoresearch.py iterate --run-dir <this-directory> --description "<short-description>"`
   to commit, validate, and benchmark. Use `--changes` for changed scopes and
   `--revisit-on` for dependency scopes. For a retry, reapply the change to the current
   best configuration and pass `--retry-of <commit>`.
4. Append the returned result to `experiment_log.md` and update ideas and learnings.

**Crash:** Identify the first failing stage and distinguish training, validation, and
resource failures. Record the fix, retry conditions, and evidence paths.

**Stop:** Record whether the target was met, the remaining gap, and the stopping reason.

## Environment requirements

- <FILL: occupancy checks and resource-sharing rules; do not interfere with other jobs; release resources after completion>
- <FILL: known environment constraints, such as timeouts, caches, and concurrent compilation>
