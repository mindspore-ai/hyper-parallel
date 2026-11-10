# Bring-up Asset Benchmark (the agent-level ruler)

Measures the *written bring-up assets* — the integration skill, checklists,
and the generic kit — not models, not the framework. One command answers
"did this asset change make the workflow better, worse, or is the
difference below what we can measure?", with evidence files for every claim.

**Boundary (non-negotiable)**: this skill is part of the adjudication
layer. It measures the evolvable asset set, so it must never be edited by
an improvement proposal, and asset snapshots under measurement never
include local modifications — only committed refs. Changes to the
benchmark itself go through operator review, and after any such change the
calibration (resolution) must be re-established before new A/B verdicts.

## Two layers

| Layer | Unit | Variance | Resolution |
|---|---|---|---|
| Deterministic | every zero-device check × catalog model, from a snapshot ref | none (A/A must be identical) | one cell; blind spots = uncovered categories, listed in every report |
| Agent | one clean-room trial (see the integration skill's `references/cleanroom-protocol.md`), scored from `trial_meta.json` + `gaps.jsonl` | sampling | minimum detectable gap difference from repeat runs (`calibrate`) |

`checks.expect` in the catalog is the checker's *correct verdict* for that
category — a tied release must trip the tied-save detector, so its expected
verdict is `fail` and observing `fail` scores `ok`.

## Commands

```bash
B=.agent/skills/bringup-benchmark

# 1. Materialize the asset version under measurement (committed ref only)
python $B/scripts/bench.py snapshot --repo-root . --ref <sha> --dest /tmp/assets-A

# 2. Deterministic suite (site paths live OUTSIDE the repo; no default)
python $B/scripts/bench.py suite --assets-root /tmp/assets-A \
    --catalog $B/catalog/validation.yaml --paths <local-paths.yaml> \
    --out out/suite_A.json

# 3. Score a clean-room trial directory
python $B/scripts/bench.py score --trial-dir <trial-dir> --out out/trial_1.json

# 4. Calibration: repeat trials on ONE ref -> resolution (MDD)
python $B/scripts/bench.py calibrate --scores out/trial_*.json --out out/resolution.json

# 5. Three-valued A/B, gated by the resolution
python $B/scripts/bench.py compare --arm-a out/A_*.json --arm-b out/B_*.json \
    --resolution out/resolution.json --out out/ab.json

# 6. Self-checks: A/A identity, and detection of injected asset damage
python $B/scripts/bench.py diff --card-a out/suite_A.json --card-b out/suite_A2.json --out out/aa.json
python $B/scripts/bench.py snapshot --repo-root . --ref <sha> --dest /tmp/assets-inj \
    --inject $B/references/injections/drop-tied-alias-exclusion.patch
python $B/scripts/bench.py diff --card-a out/suite_A.json --card-b out/suite_inj.json \
    --injection drop-tied-alias-exclusion --out out/inj.json
```

```bash
# 7. Second-operator support: quarantine an in-repo model family from a
#    snapshot (agent arm must not see the drawn model's own implementation;
#    generic assets stay; manifest discloses moves + residual mentions)
python $B/scripts/bench.py mask --snapshot /tmp/assets-B --family <name> \
    --quarantine /tmp/quarantine-B --out out/mask_manifest.json
```

Full protocol — trial contract, scoring definitions, the MDD formula, the
A/B exit rule, self-check acceptance, catalog governance and the sealed
test set: [references/benchmark-protocol.md](references/benchmark-protocol.md).

## Hard rules

- No A/B conclusion without a current `resolution.json`; differences at or
  below the MDD are reported as **indistinguishable**, never rounded up.
- An undetected injection is a *blind-spot finding*, not a pass — it goes
  into the report's "what this ruler cannot see" section.
- Site paths (model directories) come from a local paths file or
  `BRINGUP_BENCH_PATHS`; they are never committed and have no default.
- The TEST-set model list is sealed outside the repository; benchmarked
  agents must not be able to read it. Validation-set changes require a
  ledger entry.
