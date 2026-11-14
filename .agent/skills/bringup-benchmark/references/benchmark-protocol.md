# Benchmark Protocol

The ruler for bring-up assets: what a trial is, how it is scored, how the
resolution is established, and when an A/B difference may be called real.
The companion overview and command lines live in [../SKILL.md](../SKILL.md).

## 1. What is measured

A benchmark run binds to one **committed asset ref** (never a dirty tree).
`bench.py snapshot` materializes it as a detached worktree; both layers
read the assets from that snapshot, so two runs compare asset versions,
not whatever the operator happens to have open.

## 2. Deterministic layer

- Cells: every zero-device check from the integration kit × every
  validation-set model (`catalog/validation.yaml`). Checks run from
  `config.json` + the safetensors index; weight shards are optional, so
  catalog entries are kilobytes.
- A cell is `ok` when the observed verdict equals the catalog's expected
  verdict for that category. Expected verdicts encode *correct detector
  behavior*, not "everything passes": the tied-embeddings row expects the
  tied-save detector to fire.
- Exit-code mapping: 0 → `pass`, 2 → `fail`, 1 → `skipped` (missing
  inputs; investigate, never count as ok), anything else → `error`.
- Resolution: exactly one cell — any flip is detectable and exact.
  The layer's blind spots are the categories the validation set does not
  contain; every scorecard lists `covered_categories`, and the stage
  report must carry the complement as "what this ruler cannot see".

## 3. Agent layer

- Trial = one clean-room run per the integration skill's
  `references/cleanroom-protocol.md`, on the snapshot ref, producing the
  machine-readable pair (see that protocol's "Machine-readable outputs"):
  - `trial_meta.json` (`bringup-bench/trial-meta/v1`): model, assets ref,
    agent model, stages, per-gate verdicts from
    {pass, fail, env_stop, blocked, not_reached}, invention count. Gates
    take either shape: canonical list of {name, verdict, evidence}, or the
    compact {name: verdict} mapping (the scalar score never reads
    evidence; the prose report carries it).
  - `gaps.jsonl`: one JSON object per gap with
    {id, category ∈ {wrong, missing, contradictory}, stage, indicted_file,
    summary, evidence}. The indicted file is mandatory — a gap nobody can
    act on is not a gap.
- Score: primary scalar = total gap count (lower is better); gates passed,
  env stops and inventions are reported alongside, never folded into the
  scalar. Environment stops are not workflow failures and never count
  against the assets.
- Fixed factors: the agent model and trial scope are recorded in
  `trial_meta.json` and must be identical across every trial that feeds
  one calibration or one A/B. Changing either invalidates the resolution.

## 4. Calibration (resolution)

Repeat N ≥ 5 trials with everything fixed (same ref, model, scope, agent
model). `bench.py calibrate` outputs:

- spread = max − min of gap totals, sd = sample standard deviation,
- **minimum detectable gap difference (MDD)
  = max(1, spread, ceil(2·sd))** — deliberately conservative; with small N
  an optimistic resolution would launder noise into "improvements".

The MDD is printed at the head of every A/B report. Re-calibrate whenever
the benchmark itself, the trial scope, or the agent model changes.

## 5. A/B exit rule (three-valued)

`bench.py compare` on two trial groups:

- |mean(B) − mean(A)| ≤ MDD → **indistinguishable** (a result, not a
  failure; it is the honest answer for small asset changes),
- otherwise **better** / **worse** by sign.

An asset change may be merged on "better" only; "indistinguishable" means
the change rides along with a later, larger batch or gets a bigger test.
Every "better" must be traceable: name the asset commit(s) and, for the
deterministic layer, the exact flipped cells.

## 6. Self-checks (gate into service)

The benchmark may be used for decisions only after, on the current ref:

1. **A/A identity (deterministic)**: two suite runs of the same ref are
   byte-identical in verdicts (`diff` reports `identical: true`).
2. **A/A indistinguishability (agent)**: splitting the calibration repeats
   into two arms yields `indistinguishable`.
3. **Injection detection**: for each patch under `references/injections/`,
   a snapshot with the damage applied either flips at least one cell
   (`detected: true`) or the miss is recorded as a named blind spot.
   Measured outcomes on the 2026-11-03 baseline: tied-alias damage →
   detected (tied-embeddings row); fused-experts damage → detected (the
   per-expert-MoE row exercises the fused↔per-expert reconciliation,
   because newer transformers builds grouped experts while the release
   stores them per expert); rename-retry damage → blind by design on the
   validation set (the renamed category is sealed in the test set and
   covered by its periodic audit).

Re-run the self-checks after any benchmark change and record the three
artifacts next to the scorecards.

## 7. Catalog governance and the sealed test set

- The validation set is fixed by the operator; every change needs a ledger
  entry with the reason and the re-baselined scorecard.
- The TEST set (held-out families and held-out variants of seen families)
  lives outside the repository where benchmarked agents cannot read it.
  It is consulted only in periodic audits, never in merge decisions, so
  the validation set cannot be gamed by overfitting assets to it.
- New categories enter through the test set first and move to the
  validation set after one audit cycle.

## 8. Known limitations (state them, do not hide them)

- Agent-layer isolation is procedural (fresh session, pinned snapshot,
  outside-run working directory); it is not a sandbox proof.
- Gap counting measures documentation-and-tooling quality through one
  task family; a zero-gap trial on a seen category does not promise an
  unseen family (that is exactly what the sealed test set samples).
- The deterministic layer sees only what its catalog covers; its blind
  spots are enumerated per §2 and shrink only by adding categories.
