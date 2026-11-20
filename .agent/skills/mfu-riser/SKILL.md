---
name: mfu-riser
description: >
  Drive a bounded, evidence-led training MFU optimization loop using parallel
  strategy priors, Ascend profiling, and verified end-to-end experiments. Use to
  reach a declared training MFU target through equivalent implementations,
  communication, memory, optimizer, or parallel-layout improvements and
  authorized workload tuning.
---

# MFU Riser

Turn model and hardware priors, measured operator evidence, and official API references
into falsifiable optimization proposals. Execute, measure, validate, and update the best
candidate within the user's MFU target and budget. **The Agent is the planner. The
Python controller executes trials and records results; it contains no additional LLM
and does not generate proposals.**

Use the [training RSI protocol](../perf-playbook/references/training-rsi.md) to freeze
the objective, model semantics, routing, and acceptance criteria. Set
`performance_policy` for architecture-aware FLOPs auditing, tiered validation,
and broader candidate scopes. Fusion-only sessions can use the original v1 contract.

## Entry requirements and fixed measurement protocol

Obtain the reproducible initial state, dataset, launch command, device set, complete
training-step count, target, and budget. An initial state can be a checkpoint or a
declared random initialization. Reuse information already available in the task context.
If no target is specified, analyze and measure the baseline before proposing a justified
target. Build the configuration using [contracts.md](references/contracts.md); do not
substitute guessed values for measurements.

- Keep model work, inputs, initialization, routing, optimizer, devices, peak and timing
  boundaries fixed within a comparison series. A permitted sequence or batch change
  starts a linked series with its own measured baseline.
- Audit the MFU numerator against active model work. Report end-to-end MFU, HFU with
  its evidence coverage, and effective tokens/s/device. Search ranking uses MFU.
- Use a short screening window when performance is the priority. It still requires
  implementation correctness and positive path engagement. Use the full gate and at
  least 20 steady-state steps in the acceptance series, followed by independent
  confirmation on the same source and workload.
- Treat fixed routing as a declared performance workload. Preserve the model
  architecture, logical Top-K and token semantics in that workload. Changes to
  training policy use the acceptance criteria for that policy.

## Iteration workflow

1. **Establish priors.** Use [parallel-strategy-analyzer](../parallel-strategy-analyzer/SKILL.md)
   with the actual architecture and topology to estimate feasible layouts, memory, and
   cost. Save the structured result as `prior.json`. An existing stable layout may serve
   as the baseline. Priors are not measurements.
2. **Establish the baseline and adapter.** Follow [strategy-validator](../strategy-validator/SKILL.md)
   to connect the existing training entry point and reproducible initial state to the controller's
   five argv stages. Fix numerical tolerances, validators, workload, and budget at setup.
   See [acceptance.md](references/acceptance.md).
3. **Collect runtime evidence.** Use [profiling](../profiling/SKILL.md) to capture at least
   two steps after warmup. `scripts/collect_state.py` calls the existing parser and extracts
   operator shapes, types, counts, and durations. Use [memory-analysis](../memory-analysis/SKILL.md)
   for memory pressure and [moe-profiler](../moe-profiler/SKILL.md) for MoE-routing changes.
   Mark unavailable data as unknown.
4. **Form hypotheses.** Use [planner.md](references/planner.md) to rank proposals by
   expected savings on the critical path. When `status` returns `plan_from_critical_path`,
   size the gap to the target and investigate the largest exposed costs before
   micro-optimizing. Check HyperParallel implementations and version-matched
   [Ascend operators](../perf-playbook/references/ascend-fusion/index.md) against measured
   shapes and backward support.
5. **Implement and execute.** Implement the candidate in an isolated directory or behind
   an explicit feature switch. Run `trial`: numerical/compatibility `precheck` →
   unprofiled `benchmark` → separate `profile` → complete `validate` → job-specific
   `cleanup`. Record exact commands and environments for the first three stages.
6. **Accept and learn.** Retain only correct, engaged, complete results that exceed the
   noise threshold. Record failures, invalid trials, and local-only gains in history.
   Compare the final trace with predicted intermediate signals to explain the outcome.
   Add reviewed, generalizable findings to perf-playbook cases; keep speculation as
   unverified hypotheses.
7. **Continue the loop.** Read `status` and feed the best candidate's evidence and new
   negative results back into the planner. Repeat steps 4–6. A screening result at the
   target leads to a linked acceptance series. A passing acceptance result leads to
   independent `confirm`. On exhausting the trial or time budget, report the best
   valid candidate. Every trial, including failures and confirmations, consumes budget;
   other skill defaults do not permit an unbounded loop.

For runtime failures or material performance defects, use the
[failure repair procedure](../perf-playbook/references/failure-repair.md).
Preserve the failed trial and require original-workload regression evidence before
accepting the affected path. A diagnostic fallback or smaller input does not close it.

## Execution entry points

Run from the repository root. The Agent creates `run.json` and `plan.json` according to
the contract; their paths may be outside the repository.

```bash
python .agent/skills/mfu-riser/scripts/mfu_riser.py init --run-dir /path/to/new-run --config /path/to/run.json
python .agent/skills/mfu-riser/scripts/mfu_riser.py trial --run-dir /path/to/new-run --plan /path/to/plan.json
python .agent/skills/mfu-riser/scripts/mfu_riser.py status --run-dir /path/to/new-run
python .agent/skills/mfu-riser/scripts/mfu_riser.py report --run-dir /path/to/new-run
```

`trial` executes one proposal. **The Agent using this skill drives the complete loop**;
the user does not need to issue a new instruction for each iteration. Continue while
the target is unmet and budget remains. `scripts/riser_contract.py` enforces budget,
acceptance, and decision rules. The [linked-series contract](references/contracts.md#linked-series)
describes how to change an authorized capacity setting or replay a screened winner.

`scripts/audit_rank_steps.py` reuses `steps_rankN.jsonl` and `meter_steps_rankN.jsonl`
records to audit all-rank inputs, scalar trajectories, and independently derived work,
then export measurement samples. It lists uncovered checks explicitly and does not
replace module-gradient or fusion-engagement acceptance.

`recover` performs trial-specific cleanup and records an interrupted trial as `crash`
before allowing another experiment. Failed cleanup leaves the trial pending. The best
result points to workspace, configuration, and source snapshots; it does not overwrite
user source or deploy automatically. Without Git, use complete declared source hashes
and snapshots. Every remote rank must confirm that it imported and measured the same
source.

## Outputs

Artifacts include `REPORT.md`, `controller.json`, append-only `history.jsonl`, and each
trial's `context.json`, `source.tar.gz`, stage logs, `measurement.json`, profiling JSON
and HTML, raw acceptance evidence, and `result.json`. Reports distinguish local operator
speedups, end-to-end gains, unmet targets, and pending numerical acceptance.

The controller does not include a cluster-specific SSH or torchrun launcher. Reuse the
project launcher and implement the contract adapter on first integration. Missing
adapter support must not be reported as completed NPU optimization.
