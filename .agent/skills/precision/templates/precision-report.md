# <Model / Feature> Numerical Validation Report

> Include the command, source commit, and environment for each measurement.
> Declare the baseline. Mark unavailable measurements [not measured] and
> unavailable internal criteria [pending validation]. Use the
> [precision-acceptance rule](../../../rules/precision-acceptance.md) for selecting
> and recording the task's acceptance criteria before execution.

| Field | Value |
|---|---|
| Model / feature | [FILL] |
| Tested axis | [FILL: the single difference between candidate and baseline] |
| Candidate | [FILL: implementation/device, commit, configuration, environment] |
| Baseline | [FILL: implementation/device, commit, configuration, environment] |
| Node | [FILL: same-node, back-to-back A/B] |

## 1. Conclusion

- **Acceptance status**: [FILL: verdict under the selected primary and secondary criteria; missing evidence].
- **First divergence**: [FILL: module/operator/source line and execution order, or no failing comparison].

## 2. Capture

- **Tool and configuration**: msprobe, level=[FILL], step=[FILL].
- **Command**: `python .agent/skills/precision/scripts/collect_dump.py ...`,
  launched separately for candidate and baseline.
- **Dump paths**: candidate [FILL], baseline [FILL].
- **Internal instrumentation differences, if any**: [pending validation].

## 3. Comparison

- **Command**: `python .agent/skills/precision/scripts/diff_dump.py --candidate <d1> --baseline <d2>`.
- **First failing item**: [FILL: name, execution order, cosine, maximum relative error].
- **Per-tensor criterion**: bitwise by default; any explicit cosine or
  relative-error tolerance and its justification [FILL].
- **BF16 ties**: tie count [FILL], worst reference-score gap [FILL].
  Apply the below-one-ULP boundary only to qualifying tied selections.

## 4. Acceptance criteria

Follow the [criterion-selection process](../../../rules/precision-acceptance.md).
Use the user's criteria when specified; otherwise record the default rule
(normally 1,000 steps, model-scale adjustment where applicable, mean absolute
loss error <= 0.02 OR mean relative loss error <= 2%). Trajectory criteria are
separate from per-tensor comparison; do not copy defaults over custom criteria.

- **Source**: [FILL: user instruction/decision reference, or default rule and why applicable].
- **Planned / actual paired window**: [FILL: step range, count, warm-start exclusions, scale adjustment].
- **Metrics and aggregation**: [FILL: definitions, absolute/relative error calculation].
- **Acceptance logic**: [FILL: AND/OR and any user-specified secondary checks].
- **Tool gate vs task criterion**: [FILL: differences and evidence-based verdict, if applicable].

| Check | Result | Criterion |
|---|---|---|
| Mean absolute loss error | [not measured] | [FILL: selected threshold; default 0.02] |
| Mean relative loss error | [not measured] | [FILL: selected threshold; default 2%] |
| Gradient-norm trajectory | [not measured] | [FILL: selected band and aggregation] |
| First step | [not measured] | [FILL: selected criterion] |
| Determinism with the same seed | [not measured] | [FILL: selected criterion] |
| Checkpoint resume | [not measured] | [FILL: selected criterion] |

## 5. Proposed fix

- **Root-cause assessment**: [FILL: based on the first divergence].
- **Proposed correction**: [FILL].
- **Regression method**: [FILL: repeat identical inputs and verify recovery].

## 6. Evidence index

- Commands, commits, and environments: [FILL].
- Comparison tables, dumps, and logs: [FILL].
