# Clean-Room Reproducibility Protocol

A memory-free audit of the bring-up workflow: a fresh agent, given only this
repository and a model directory, must reach the CPU-reachable gates by
following the written assets alone. It answers two questions at once — *can
someone else's agent reproduce the workflow?* and *which written assets are
wrong or missing?* — and doubles as the asset-vs-memory ablation in any
longitudinal evaluation (the clean agent is the "no inherited context" arm).

## Setup rules

1. **Fresh agent, zero inherited state**: no session memory, no chat history,
   no operator coaching beyond the task statement below. Its working
   directory is outside any existing run's directories.
2. **Inputs**: a checkout of this repository and a local HF model directory
   (config + index + tokenizer; weight shards optional — without them the
   audit covers the meta/name gates plus the weights-free parity variant of
   checklist §6.2, and records the first weights gate as its stop). The
   Python environment must satisfy the repository's `requirements.txt`;
   when an import is still blocked on an offline host, the blocked gate is
   recorded as an **environment stop** with its exit evidence — an
   environment stop is not a workflow failure.
3. **Scope**: S0 through the CPU-reachable part of S2 (survey, recipe,
   layout pin, same-process parity and a CPU smoke if feasible). No cluster
   access, no accelerator — the protocol is deliberately zero-device.
   The kit ships no dedicated CPU smoke path today: if the auditor builds
   one, it is declared as an invention and filed as a gap, not absorbed.
4. **Task statement given to the agent**: "Bring up <model> in this
   repository by following its own agent assets. Stop at the first gate that
   needs an accelerator. Record every gap where the written assets were
   wrong, missing, or contradictory."

## Required outputs

- A staged report: per stage, what the assets instructed, what was done,
  the gate evidence (commands + exit codes), and the artifact paths.
- A **gap list**, harshly itemized: every point where an asset was wrong
  (bug), missing (undocumented step the agent had to invent), or
  contradictory (two assets disagree). Each gap names the file it indicts.
- A reproduction command register: every claim re-runnable as written.
- One row for the run-metrics table (see `templates/run-metrics.md`).

## Grading

- **Reproducibility verdict**: pass if every attempted gate was reached by
  following written assets, with inventions listed as gaps rather than
  silently absorbed.
- Gaps feed the normal post-run review: each one is fixed in the indicted
  asset (code + test where executable) or explicitly rejected with a reason
  in the run ledger. A gap fixed without a test is not closed.

## Releasing a run record

Run records carry host names, account names, node addresses and co-tenant
identities, so publishing one is a rewrite, not a copy. Substitute the
placeholder forms this skill already uses, **then rescan the output** — the
check costs one command and a single surviving address is a leak. Two
mechanical traps, both observed: documents from different runs often share
a basename (`s0_spec.md`), so stage them under their source paths or one
silently overwrites another and the count looks right; and the rescan must
run against the staged copies, not the sources, or it proves nothing.
Rehearse the procedure on the real documents before the submission
deadline: a rewrite that has only been described has not been tested.

## Field record

First execution (2026-10, a 360M tied-embedding model): S0–S2 CPU gates all
reached; 16 gaps filed, including a workflow-level handoff contradiction, a
false-positive layout gate on tied weights, and stage-table references to
not-yet-merged skills — all three repaired in the indicted assets and
re-verified by the next on-device run.
