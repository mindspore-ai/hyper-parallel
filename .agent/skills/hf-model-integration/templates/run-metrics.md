# Run Metrics (fill one column per bring-up run)

Fill this at post-run review, from the run ledger and logs — estimates are
acceptable if marked `(est.)`, qualitative facts are not. The point of the
table is the trend across runs, so keep the definitions below stable; change
them only with a ledger entry explaining why and a recomputed history.

## Definitions

- **Operator decision inputs**: messages that changed control flow, counted
  by category — selection/criteria, resource, transfer-channel,
  direction/rule. Progress nudges and acknowledgements do not count.
  Split task-domain (bring-up decisions) from governance-domain (cluster
  operations rulings) when both occur.
- **Autonomous recovery loops**: fault signal → diagnosis → action →
  verification closed with no human in the loop; one count per fault.
- **Gate first-pass rate**: the seven-gate basis is S0 spec card, S1 layout
  contract, S2 forward parity, S2 training smoke, S3 equivalence matrix,
  S4 profile baseline (all consistency checks), S5 long-run. First
  submission passing = first-pass.
- **Tool defects exposed → fixed**: toolchain (skill/script) defects this
  run surfaced, and how many were repaired within the run (with tests).
- **Deposits**: PRs touched / issues filed / doctrine clauses added.
- **Device time**: node-hours (8-die node), from step time × steps plus
  startup; mark `(est.)` until recomputed from logs.

## Table skeleton

| Metric | run #N <model> |
|---|---|
| Architecture traits | |
| Route (zero-registration / provider) | |
| Model-specific code at run end (lines) | |
| Stages covered | |
| Operator decision inputs (total; by category; task vs governance) | |
| Autonomous recovery loops (task / governance) | |
| Gate first-pass rate (x/7) | |
| Device time (node·h, est.) | |
| Wall-clock | |
| Tool defects exposed → fixed | |
| Deposits (PRs / issues / clauses) | |
| Reproducibility audit (clean-room result, if run) | |
