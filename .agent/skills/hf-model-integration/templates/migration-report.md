# <Model> Migration Report

> Record the command, source commit, and environment for every measurement.
> Mark unavailable measurements [not measured]; do not substitute estimates.
> Numerical acceptance follows `.agent/rules/precision-acceptance.md`.

| Field | Value |
|---|---|
| Model | [FILL: Hugging Face repository and revision] |
| Target branch / PR | [FILL] |
| Owner | [FILL] |
| Status | In progress / Awaiting acceptance / Accepted |

## 1. Migration scope

- **Architecture**: [FILL: layer count, attention/MLP structure, specialized modules].
- **Reuse and new implementation**: [FILL: existing optimized components,
  new components, and the reason for each choice].
- **Excluded components**: [FILL: scope and rationale].

## 2. Adaptation inventory

| Adaptation | Approach | Status |
|---|---|---|
| Deferred initialization / meta device | [FILL] | [FILL] |
| Optimized module replacement | [FILL] | [FILL] |
| TP integration | [FILL] | [FILL] |
| CP integration | [FILL] | [FILL] |
| EP integration | [FILL] | [FILL] |
| Checkpoint conversion | [FILL: weight-name mapping and sharding] | [FILL] |
| Data/input contract | [FILL] | [FILL] |

## 3. Artifacts

| Artifact | Path | Description |
|---|---|---|
| Model adapter | [FILL] | |
| Checkpoint conversion script | [FILL] | Hugging Face weights to framework format |
| Training YAML | [FILL] | Runnable recipe |
| Launch scripts | [FILL] | Single-node and multi-node |
| Tests | [FILL] | Unit and system tests |

## 4. Validation

### 4.1 Functionality

| Check | Command | Commit | Environment | Result |
|---|---|---|---|---|
| Unit tests | [FILL] | [FILL] | [FILL] | [FILL] |
| Single-device forward/backward | [FILL] | [FILL] | [FILL] | [FILL] |
| Distributed TP/CP/EP | [FILL] | [FILL] | [FILL] | [FILL] |

### 4.2 Numerical correctness

State the baseline and actual comparison window. Follow the numerical
acceptance rule, normally over 1,000 steps.

- **Baseline**: [FILL].
- Mean absolute loss error [not measured], threshold 0.02.
- Mean relative loss error [not measured], threshold 2%.
- **Gradient norm**: [not measured].
- **First step**: [not measured].
- **Determinism**: [not measured].
- **Checkpoint resume**: [not measured].

### 4.3 Performance

- **Time breakdown**: [not measured: compute, exposed communication, device
  idle time, and the optimizer subset; see `profiling`].
- **Peak device memory**: [not measured; see `memory-analysis`].

## 5. Constraints and known issues

- **Supported scope**: [FILL].
- **Unsupported cases / follow-up work**: [FILL].

## 6. Change log

| Date | Author | Change |
|---|---|---|
| [FILL] | [FILL] | Initial draft |
