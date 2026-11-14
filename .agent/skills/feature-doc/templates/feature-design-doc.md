# <Feature / Model> Design Document

> Complete every [FILL] entry. Explain any inapplicable section instead of
> leaving it empty. Mark unavailable measurements [not measured] with a
> reason. A new document fills the complete template; an update changes
> only the affected sections and adds a dated change-log entry.

| Field | Value |
|---|---|
| Feature/model | [FILL] |
| Owner | [FILL] |
| Related PR / issue | [FILL] |
| Target branch | [FILL] |
| Status | Draft / In review / Accepted |

## 1. Feature/model description

- **Background**: [FILL: problem and source requirements].
- **Goal and scope**: [FILL: supported outcomes and explicit exclusions].
- **Public interface**: [FILL: configuration fields, entry points, and YAML
  interface; include names, types, defaults, and value semantics].

## 2. Migration and adaptation plan

- **Overall approach**: [FILL: source implementation and adaptation strategy].
- **Key adaptations**: [FILL: deferred initialization, optimized module
  replacements, TP/CP/EP integration, checkpoint conversion, data and
  operator integration, as applicable].
- **Reuse and new implementation**: [FILL: what is reused or rewritten and why].

## 3. Code-change scope

Identify the file, module/function, and nature of each change so a reviewer
can locate the implementation.

| File | Changed component | Change description |
|---|---|---|
| [FILL] | [FILL] | [FILL] |

## 4. Usage constraints

- **Supported inputs and environments**: [FILL: shapes, dtypes, parallel
  degrees, sequence lengths, hardware].
- **Unsupported cases**: [FILL: each case and whether the limitation is
  algorithmic or an implementation gap].

## 5. Validation report

> Every measurement requires a command, source commit, and environment.
> Missing evidence that the intended path ran cannot count as a pass.

### 5.1 Functionality

| Check | Command | Commit | Environment | Result |
|---|---|---|---|---|
| Unit tests | [FILL] | [FILL] | [FILL] | [FILL] |
| System/smoke tests | [FILL] | [FILL] | [FILL] | [FILL] |

### 5.2 Numerical correctness

Follow `.agent/rules/precision-acceptance.md`, normally over 1,000 steps.
State the baseline and the actual comparison window.

- **Baseline**: [FILL: reference implementation, commit, and configuration].
- **Primary criterion**: mean absolute loss error [not measured], at most
  0.02, **or** mean relative loss error [not measured], at most 2%.
- **Gradient norm**: [not measured].
- **First step**: [not measured].
- **Determinism**: [not measured].
- **Checkpoint resume**: [not measured].
- **Command / commit / environment**: [FILL].

### 5.3 Performance

- **Breakdown**: [not measured: compute by execution category and relevant
  fused operators, exposed TP/DP/EP/CP/PP communication, device idle time,
  and the optimizer subset].
- **Primary metric**: [not measured: same-node, back-to-back A/B with both
  configurations recorded].
- **Command / commit / environment**: [FILL].

## 6. Dependencies and compatibility

- **Dependencies**: [FILL: upstream branch, operator package, framework
  versions, and companion features].
- **Compatibility constraints**: [FILL: supported versions and interactions
  with other features].

## 7. Change log

| Date | Author | Change |
|---|---|---|
| [FILL] | [FILL] | Initial draft |
