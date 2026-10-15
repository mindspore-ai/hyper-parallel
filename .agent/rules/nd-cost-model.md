---
name: nd-cost-model
description: "The ND cost model (auto_parallel): what a change to it must prove, and how a measured figure may be used"
paths:
  - hyper_parallel/auto_parallel/**
  - tests/ut/auto_parallel/**
---

# ND Cost Model

ND is the device-free cost model under `hyper_parallel/auto_parallel/`: `sapp_nd/`, the layer census and the
config adapter. It ranks parallel strategies by peak memory and a relative score. These rules bind every change
to it and every figure quoted about it.

## Before reading or changing it

- State the commit each claim was checked at, claim by claim, not once for a batch. ND branches move under a
  session, and a stale branch can be thousands of lines behind: read there, live code looks missing.
- Confirm which tree Python imports, in the same process as the run that matters
  (`python -c "import hyper_parallel; print(hyper_parallel.__file__)"`). An editable install pointing at another
  worktree prices with old code and says nothing.

## A change to an estimate

- One register item, one commit. Say what the change should move before running anything, then what it moved.
- Run the golden harness before and after (`nd_golden/`, kept beside the checkout, not in this repository):
  `golden_fx` 23157 numbers, `parsers_fx` 430, `ms_mtp_fx` 44 and `f1_fx` 162, 23793 in all, compared bit for
  bit by `diff_golden`. Check the count before trusting a clean diff: `golden.py` is superseded and exits 0 on
  20517 numbers.
- Account for every moved number in the commit message: how many, on which sources, which way, and whether a
  search reorders or changes its best. An unexplained move blocks the commit; no move is a result to state too.
- A new test fails on the parent commit and passes on the change.
- Lint the changed files against the parent SHA, never `HEAD`: a stale base export reports old messages as new.
- Working branches take new commits: no amend, no force-push. Squash only when a PR is opened, where
  AGENTS.md's one commit per PR applies.
- A config field nothing parses reads as 0 (`CostModelConfig.__getattr__`, reported under the ranking;
  `run_nd --strict` refuses it). A stated field can look priced and not be: check that it reaches the estimate.

## Using a measured figure (register B12)

- A term's LEVEL is validated against the profiler's own columns (Computing, Communication, Free), which exist
  only under the profiler; its RANKING against the trainer's own step time (`step_trainer`). Never mix the two
  inside one quantity: the trainer's step less the profiled parts gives a negative idle.
- Pair a profiled and an unprofiled measurement only within one round. Across rounds the profiled side alone
  moves about 4.6% on one strategy, more than the profiler cost being measured.
- Re-derive every figure from saved output before quoting it, and name the file and the derivation. A number
  that no file reproduces is not quoted.
- Check that a number agrees with the one everyone else quotes. A mismatch is a thing to check, not a reason
  to leave it out.
- Audit the caveats as hard as the claims: a provenance error survives inside a caveat, which reads as the
  careful part.

## What ND is, by decision

- A relative scorer (register A5): no device rate or peak prices a part, and the ratios fitted on a measured
  round own the scale; a host link's `sustained_tflops` only turns an offload copy's seconds into the score's
  units where no fitted ratio does. Do not add an absolute rate without revisiting that decision.
- A model of allocated memory, where a run dies on reserved: "ND says it fits" is not a bound.
