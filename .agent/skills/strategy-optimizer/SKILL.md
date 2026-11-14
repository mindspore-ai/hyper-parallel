---
name: strategy-optimizer
description: >
  Coordinate parallel-strategy analysis and measured validation in a bounded
  search. Use for selecting a parallel layout for a new model or hardware
  configuration, comparing candidate layouts, or refining a measured baseline.
  Returns the best eligible measured candidate within the search budget.
---

# Strategy Optimizer

Combine parallel-strategy analysis (PSA), real-device benchmarks, and
failure-driven refinement. The result is the best eligible candidate tested
within the configured rounds; the search does not prove a global optimum.

Use [parallel-strategy-analyzer](../parallel-strategy-analyzer/SKILL.md) for
analysis alone, or [strategy-validator](../strategy-validator/SKILL.md) to
measure an existing candidate set.

## Responsibilities and scope

- **Agent:** Supply the actual architecture, topology, runnable training
  configuration, measured baseline, objective, and resource allocation.
  Interpret evidence and decide whether further investigation is needed.
- **Search adapter:** Call `hyper_parallel.auto_parallel.config_adapter`,
  check layout compatibility, and construct a small candidate neighborhood.
- **Validation runner:** Execute candidates and extract throughput, step
  duration, and peak memory.
- **Controller:** Evaluate complete measurements, refine supported
  constraints, record results, and export the selected configuration.

The current executable scheduler supports **one node**. Its PSA
`num_nodes` must be 1 and `cards_per_node` must match `--devices`.
This is a limit of this runner, not a statement about all HyperParallel
parallel-strategy tools. The runner exports configuration files; it does
not deploy training jobs into production.

## Prerequisites

Provide:

1. A runnable Trainer YAML containing `model._target_` and `optimizer`,
   with its model assets and data accessible.
2. An auto-parallel search YAML describing the actual model, hardware,
   search space, and constraints.
3. A measured baseline throughput and its parallel layout. Keep useful
   model work, data, precision, optimizer, and timing convention comparable.
4. An allocated device set, target throughput multiplier, optional peak
   memory limit, and bounded validation duration.

There is no universal MFU target for a model family. Derive targets from
the measured baseline and the declared
[utilization convention](../perf-playbook/references/mfu-targets.md).
A scalar throughput target alone does not establish numerical correctness;
use the workload's declared acceptance protocol before retaining a changed
training implementation.

## Search workflow

```text
Runnable baseline + search configuration + measured objective
                            |
                            v
                 PSA candidate + valid neighbors
                            |
                            v
              Validate layouts and generate Trainer YAML
                            |
                            v
                  Execute candidate benchmarks
                            |
                            v
                Check completion, metrics, memory
                            |
               +------------+-------------+
               |                          |
           Target met                Target unmet
               |                          |
               v                          v
        Export best result     Refine supported constraints
                                          |
                             Remaining rounds and candidates?
                                          |
                                  Return to PSA or stop
```

Each round considers at most three candidates and skips strategies already
measured in this run. The default limit is three rounds, and the CLI
accepts one to three. Stop when an eligible candidate reaches the target,
the round limit is reached, or no unmeasured candidate remains.

### Candidate generation

With `--psa-config`, the adapter validates the search YAML and invokes
`config_adapter.search_strategies`. It starts with the selected PSA result,
then considers valid neighbors such as removing CP, increasing EP, or
removing TP. These neighbors are proposals, not measured improvements.

The adapter checks the Trainer's layout domains:

```text
world_size = DP × TP × CP × PP
stage_size = world_size / PP
EP divides stage_size
dp_shard divides DP × CP
edp_shard divides stage_size / EP
```

EP is not an additional independent factor in the world-size product.
A full-sharding refinement changes the selected strategy, so the adapter
removes stale PSA scores and memory estimates for that modified candidate.
See [PSA cost-model interpretation](references/psa-cost-model.md) for
analytical estimates, calibration, and model-family limits.

### Measured selection

A candidate is eligible only when execution succeeds and throughput, peak
memory, and step duration are finite and positive. When specified, peak
memory must also satisfy `--max-memory-gb`. The controller selects the
eligible candidate with the highest measured throughput.

The target is:

```text
candidate throughput >= baseline throughput × target improvement
```

An ineligible or failed run cannot become the selected configuration.
The output distinguishes target completion from a partial result.

### Refinement and diagnostic limits

The current controller implements these refinement rules:

| Observed result | Next-round constraint |
|---|---|
| Error text contains OOM or out-of-memory | Request full sharding and full activation checkpointing |
| Error text contains communication or timeout | Reduce the maximum TP and CP degrees |
| Eligible result has peak-memory data | Record observed peak memory |

These are search heuristics. A timeout does not by itself identify a
communication root cause, and an analytical cost underestimate does not
prove a communication bottleneck. Use
[cluster-debug](../cluster-debug/SKILL.md) for distributed failures and
[profiling](../profiling/SKILL.md) for phase and operator attribution.

For pipeline bubbles, inspect stage imbalance, P2P waits, microbatch count,
and schedule behavior before proposing a new layout. For unstable timing,
check device occupancy, input readiness, host synchronization, and repeated
measurements. The current refinement function does not automatically
change precision, global batch, or the PP schedule to address these cases.

## Execution

Run from the repository root. Replace the paths and variables below with
the current workload's verified values.

```bash
python .agent/skills/strategy-optimizer/scripts/run_optimization.py \
  --model "$MODEL_NAME" \
  --devices 8 \
  --baseline-throughput "$BASELINE_TOKENS_PER_S" \
  --baseline-config "TP=2,CP=1,EP=2,PP=1,DP=4" \
  --psa-config /path/to/search.yaml \
  --config-template /path/to/runnable-training.yaml \
  --output-dir /path/to/new-optimization-run \
  --target-improvement 1.2 \
  --max-iterations 3 \
  --validation-steps 50 \
  --timeout 1800
```

The layout shown consumes eight devices; it is an example, not a
recommended configuration for every model. `--timeout` limits each
validation invocation. It is not a whole-search wall-clock budget.
Use an empty output directory to avoid mixing evidence from different runs.

Use `--plan-only` to generate candidates without running validation.
Without `--psa-config`, this mode uses a heuristic neighborhood and marks
the result `PLANNED`. Without a runnable template, generated YAML files are
planning descriptors, not executable training configurations.

## Outputs and interpretation

The run writes:

- `optimization_result.json`: baseline, per-round candidates, constraints,
  validation results, and `final_result`.
- `optimization_report.md`: final status and result summary.
- `configs/iteration_*/`: candidate configurations.
- `validation/iteration_*/`: validation evidence.
- `production_config.yaml`, when an eligible result exists: the selected
  configuration exported for review. The filename does not imply deployment.

The serialized fields `optimal_candidate`, `optimal_metrics`, and
`optimal_strategy` refer to the best eligible measured result in this run.

| Final status | Meaning |
|---|---|
| `SUCCESS` | An eligible measured candidate reached the target |
| `PARTIAL` | An eligible candidate exists, but the target was not reached |
| `FAILED` | No eligible measured candidate was produced |
| `PLANNED` | Candidate generation only; no validation measurements |

Report the remaining target gap, failures, and measurement uncertainty.
Preserve configuration, source revision, raw logs, and metric conventions
for reproduction. Do not report planned estimates or synthetic examples
as measured model performance.
