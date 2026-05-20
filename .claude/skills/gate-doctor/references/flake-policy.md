# Flake Policy

How gate-doctor classifies and reacts to test failures during the
autofix close-the-gate loop.

## Contents

- [TL;DR](#tldr)
- [Per-failure decision tree](#per-failure-decision-tree)
- [The 3-question triage (PR-INDUCED vs UNRELATED)](#the-3-question-triage-pr-induced-vs-unrelated)
- [State the agent must track across the loop](#state-the-agent-must-track-across-the-loop)
- [Sticky-flake PR comment template](#sticky-flake-pr-comment-template)
- [Path C — PR-author-authorized temp comment-out](#path-c--pr-author-authorized-temp-comment-out)
- [Why no silent `@pytest.mark.skip` on upstream tests](#why-no-silent-pytestmarkskip-on-upstream-tests)
- [Counter-example accounting (worked)](#counter-example-accounting-worked)

## TL;DR

| Pattern across consecutive builds | Action |
|---|---|
| Any failure judged PR-INDUCED | Patch the file. Local pylint 10.00/10. `autogit commit`. `/retest`. |
| All UNRELATED, **same** test fails ≥2 builds, PR-author authorized | **Path C (temp comment-out).** Apply a `gate-doctor: temp-skip` marker on the failing test in the PR branch, commit, `/retest`. Remember the entry in conversation memory so the final report can flag it for revert. See "Path C" below. |
| All UNRELATED, **same** test fails ≥2 builds, NOT PR-author authorized | **Path B (sticky flake comment).** Post a PR comment with triage evidence, escalate to maintainer. Do NOT silently `@skip` upstream tests. |
| All UNRELATED, **different** tests each build | **Path A (random flake).** Just `/retest`. No patch, no comment. Cap at 5 in a row, then escalate. |

The split between "patch", "retest only", "temp-skip", and "comment + escalate"
is the whole point of this policy. Patching for a flake makes the codebase
worse; retesting through a real PR bug masks the bug; silently disabling
upstream tests is content-integrity tampering; refusing to comment-out *any*
confirmed-unrelated flake leaves the PR author unable to validate their own
changes. The 3-question triage is the gate; whether the PR author has
authorized temp-skip determines which of Path B vs Path C fires when the
triage says UNRELATED + sticky.

## Per-failure decision tree

For every FAILURE build, run the 3-question triage on each failure
entry, bucket each as PR-INDUCED or UNRELATED, then route the
**build as a whole** through this tree:

```
(a) Any failure PR-INDUCED?
    → Yes: fix the production code per
      [root-cause-fix.md](root-cause-fix.md).
        · Trivial (Check_* linter + recipe + ≤5 LoC + single file):
          patch directly, commit, /retest. Increment patch_attempts.
        · Deep (Smoke/UT/ST/Ascend, assertion/shape/dtype/dispatch,
          multi-file, or no canonical recipe): stop, set
          terminal_state = "needs-human-guidance", emit final
          report, ask the user for direction. Do not patch blind.
      UNRELATED entries in the same build are noise this round —
      defer their classification to the next FAILURE.

(b) All failures UNRELATED, **same** test ≥2 consecutive builds
    (sticky)?
    → Path-C-authorized (PR author on own fork, hasn't opted out):
      apply the temp-skip marker (§ Path C below), commit, /retest.
      Remember the entry for the final report's REVERT-BEFORE-MERGE
      section.
    → Not authorized: Path B. Post a maintainer-facing PR comment
      with triage evidence. Set terminal_state =
      "upstream-sticky-flake" and exit.

(c) All failures UNRELATED, **different** tests each build (random)?
    → Path A. /retest only. No patch, no comment. Increment
      flake_retest_attempts; cap at 5; on cap set terminal_state =
      "flake-ceiling".

(d) Mixed PR-INDUCED + UNRELATED in the same build?
    → PR-INDUCED dominates: take branch (a). The UNRELATED entries
      roll over to the next build's classification — if they recur
      sticky, route to (b) on that build.
```

This tree is the canonical authority. The Path A/B/C narrative
sections below explain the *why* for each route; the
[3-question triage](#the-3-question-triage-pr-induced-vs-unrelated)
section below explains how to bucket an individual failure.

## The 3-question triage (PR-INDUCED vs UNRELATED)

For each failure entry in the diagnose JSON, answer all three. Any
"yes" pushes the verdict toward PR-INDUCED; only when all three are
"no" with evidence is the failure UNRELATED.

### Q1 — Did the PR change anything that the failing test transitively depends on at build / load time?

Check the PR diff for:

- `.gitmodules`, any submodule pin bump, third-party version change.
- Build files: `build.sh`, `CMakeLists.txt`, top-level `setup.py`, any
  `*.cmake` under the touched directory tree.
- Environment / runtime config: `ASCEND_*`, `MS_DEV_*`, `GLOG_*`,
  conda or pip pinning, container image references.
- Test scaffolding the failing test loads transitively:
  `tests/mark_utils.py`, fixtures under the test's `conftest.py` chain.

If yes → likely PR-INDUCED even when the failing test file isn't
touched.

### Q2 — Does the failing test reach the PR's edited code?

Grep the failing test file and the modules it imports for symbols
defined in the PR diff. A failing `test_swiglu_*` for a PR that only
touches `tests/st/networks/llm_parallel_feature/` shows no overlap —
that's a real "no". A failing `test_with_stream.py` for a PR that
adds a new ops entry in `mindspore/ops/api_def/` may overlap through
the generated dispatch table — that's a "maybe yes".

When the test reaches PR code via an indirect import path (codegen,
plugin registration), trust the import chain over the test name.

### Q3 — Did the same test pass on a recent master build?

Use `gate_doctor.py diagnose` against a recent master CI build (or
another PR built against the same master tip) and confirm the test
passed there.

- Passed on master → genuinely flaky-or-PR-induced. Combined with
  Q1=no and Q2=no, this is the only path to "UNRELATED".
- Failed on master too → master HEAD is broken. Comment on the PR
  saying so; this is sticky-flake territory and must NOT be silently
  skipped in the PR.

If you can't get a Q3 datapoint (no recent comparable build), bias
toward PR-INDUCED for safety. The cost of a wasted /retest is much
smaller than the cost of merging a real regression masked as "flake".

## State the agent must track across the loop

Per-loop counters and observations:

- `patch_attempts`: how many fix commits the agent has pushed in this
  loop. Cap at 2.
- `flake_retest`: how many consecutive UNRELATED-failure /retests the
  agent has issued without any new fix commit. Cap at 5.
- `failed_tests_history`: ordered list of test-name sets, one entry per
  completed build in this loop. Used to detect "same test ≥2 builds"
  vs "different test each time".
- `pr_comments_posted`: how many sticky-flake comments the agent has
  already posted on the PR (avoid duplicate noise).

A "sticky flake" is detected when the *intersection* of the last two
entries in `failed_tests_history` is non-empty AND all members of
that intersection are UNRELATED.

A "random flake regime" is detected when the *intersection* of the
last two entries is empty AND all failures are UNRELATED.

## Sticky-flake PR comment template

When the agent detects a sticky upstream flake, post one comment per
sticky cluster (deduplicate against prior posts). Use this template:

```
@<maintainer-or-blank> gate-doctor reports a sticky upstream flake
on this PR:

- Test: `<test::nodeid>`
- Failing in builds: #<N1> (<url1>), #<N2> (<url2>)
- 3-question triage: Q1=no (PR touches <files>, none reach this test),
  Q2=no (test does not import PR symbols), Q3=<yes-passed-on-master |
  no-master-also-broken | unknown>.
- Failure signature: <one-line excerpt or assertion>

This appears unrelated to the PR's changes. Please advise whether to
(a) skip the test on master, (b) wait for an upstream fix, or
(c) flag a real regression. gate-doctor will keep /retest-ing on a
back-off until the PR carries `ci-pipeline-passed`.
```

Post via a direct API call with the `GITCODE_TOKEN` env var (the
same shape the existing `retest` / `check-pr` subcommands use under
the hood):

```bash
curl -sX POST \
  -H "PRIVATE-TOKEN: $GITCODE_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"body\": \"$BODY\"}" \
  "https://gitcode.com/api/v5/repos/<owner>/<repo>/pulls/<N>/comments"
```

## Path C — PR-author-authorized temp comment-out

This is the diagnostic-skip path. It is the **only** sanctioned way
gate-doctor disables a test, and it fires only when **all** of these
hold simultaneously:

1. The 3-question triage verdicts the failure as **UNRELATED** to
   the PR (Q1, Q2, Q3 all say "no" with evidence, or Q3 says
   "passed on master").
2. The failing test set in the current build **intersects** the
   failing set in the previous build — i.e. it is sticky, not
   random. (`failed_tests_history[-1] ∩ failed_tests_history[-2]
   ≠ ∅`.)
3. The PR author has authorized temp-skip in the invocation. The
   default is **authorized** when:
   - The caller is operating on the PR author's own fork branch
     (i.e. `origin` is the author's fork and the branch matches the
     PR head branch), AND
   - The user has not said "don't comment-out anything" earlier in
     this session.

If any of (1)-(3) is not met, fall through to Path B.

### What "temp comment-out" actually means

It is **not** a `@pytest.mark.skip` decorator and **not** a
`@pytest.mark.skipif(...)` either. Decorators look permanent — a
reviewer reading the diff sees a normal skip annotation and may not
realise it must come out before merge.

The temp comment-out is a deliberate **block comment** that disables
the test body in a way that is impossible to miss on diff. The agent
applies it via Read + Edit on the test file, producing this shape:

```python
# >>> gate-doctor: temp-skip for UNRELATED sticky flake — REVERT BEFORE MERGE
# Reason: <triage, e.g. "Q1=no Q2=no Q3=passed-on-master">
# Failed in builds: #<N1>, #<N2>
# Marker-id: gd-tmpskip-<pr>-<random4>
# def test_pynative_swiglu_x():
#     a = build_inputs()
#     b = forward(a)
#     assert b.shape == (4, 8)
# <<< gate-doctor: end temp-skip
```

Properties (the agent MUST follow all of these):

- **Whole function commented out**, not just the body, so the diff
  is unmistakable.
- The `REVERT BEFORE MERGE` phrase is the literal trigger that
  reviewers and the final report scan for. Don't paraphrase it.
- The `Marker-id` is a fresh `gd-tmpskip-<pr>-<random4>` per
  application. Before editing, grep the file for any existing
  `gate-doctor: temp-skip` block that already mentions this test —
  if one exists, leave it alone (don't stack duplicates).
- A separate fix commit is produced via autogit:
  `fix(<area>): temp-skip <test-name> as unrelated flake (gd-tmpskip-…)`.
  The commit message is greppable for later cleanup.
- Class methods (`TestX::test_y`) and parametrized IDs (`test_y[param-1]`)
  are not supported at this granularity. Comment out the whole
  function; if a specific parameter is flaky, ask the user.

### What the agent remembers across the loop

Every temp-skip is recorded in conversation memory as
`(test_nodeid, file:line, marker_id, fail_builds, triage, commit_sha)`.
The final report renders this list under
"Temporary comment-outs (REVERT BEFORE MERGE)" with one bullet per
test. If the list is non-empty on a successful green, the report's
"What you need to do next" section forces the author to revert
before merge.

### Author obligations after merge

The temp-skip is **diagnostic**. The author has three accepted
outcomes:

1. **Best**: revert the temp-skip commits before merge, re-`autofix`,
   confirm the gate still passes on its own (the flake may have
   self-resolved on master), then merge.
2. **Acceptable**: revert the temp-skips, re-`autofix`, and if the
   same flake reappears, switch to Path B (comment on the PR,
   escalate). Merge only with maintainer sign-off.
3. **Forbidden**: merge while the temp-skip commits are still in
   the branch. This ships a permanent disabled test under the
   guise of a diagnostic skip — the exact failure mode this policy
   exists to prevent.

The skill enforces (1) and (2) by surfacing the unrevert list at
the top of every final report. It can't enforce (3) directly —
but reviewers can grep any PR diff for `gd-tmpskip-` or
`gate-doctor: temp-skip` markers and reject merges that still
carry them.

### Counter-example accounting (Path C)

| Build | Result | Failing tests | Triage | Action |
|---|---|---|---|---|
| #400 | FAILURE | `test_a` (PR-INDUCED), `test_b` (UNRELATED) | Q3=no for both | Patch for test_a. patch_attempts=1. /retest |
| #401 | FAILURE | `test_b` (UNRELATED) | Q1=no, Q2=no, Q3=passed-on-master | sticky (test_b ∈ #400 ∩ #401), Path C → temp-skip test_b, commit, /retest |
| #402 | FAILURE | `test_c` (UNRELATED) | Q1=no, Q2=no, Q3=passed-on-master | random-vs-temp-skipped — not sticky against test_b. flake_retest=1. /retest |
| #403 | SUCCESS | — | — | ci-pipeline-passed → DONE. Final report flags test_b for revert. |

Note that test_a (PR-INDUCED) was fixed in production code in
build #400, never temp-skipped. Path C never disables PR-INDUCED
tests — those go through the root-cause-fix rule
([root-cause-fix.md](root-cause-fix.md)) without exception.

## Why no silent `@pytest.mark.skip` on upstream tests

The skill's earlier iteration sometimes pushed `@pytest.mark.skip`
decorators onto upstream test files when the agent judged a failure
unrelated. That practice was flagged by the auto-mode classifier as
content-integrity tampering, and rightly so:

- The PR author doesn't own those tests. Pushing a skip ships a code
  change to shared upstream files in a PR whose stated purpose is
  unrelated.
- Skipping a real flake masks a real signal — if master is broken,
  the right outcome is for master to be fixed, not for each PR to
  paper over it.
- The skip survives merge. A "temporary" skip becomes a permanent
  liability the moment reviewers don't catch it.

A PR-level comment is the integrity-preserving move: it surfaces the
evidence to humans (maintainer, PR author) and they decide whether to
land a real fix, a real skip in a separate PR, or to wait. The
gate-doctor stays available to /retest on a back-off in the meantime.

The only time the agent should push a code skip is via **Path C**
(see above) — the PR-author-authorized temp comment-out, with the
`gate-doctor: temp-skip` marker, the ledger entry, and the
revert-before-merge clause baked into the final report. Even then,
prefer a separate "test-only" PR that the maintainer can review on
its own merits, and never apply Path C without the 3-question triage
verdicting UNRELATED with evidence.

## Counter-example accounting (worked)

### Build sequence A: random flake regime

| Build | Result | Failing tests (all UNRELATED) | Action |
|---|---|---|---|
| #100 | FAILURE | `test_foo` | flake_retest=1, /retest |
| #101 | FAILURE | `test_bar` | flake_retest=2, /retest |
| #102 | FAILURE | `test_baz` | flake_retest=3, /retest |
| #103 | SUCCESS | — | check `ci-pipeline-passed` label, DONE |

`failed_tests_history` had empty pairwise intersections → random
regime. No patches, no comments.

### Build sequence B: sticky flake

| Build | Result | Failing tests (all UNRELATED) | Action |
|---|---|---|---|
| #200 | FAILURE | `test_swiglu_x` | flake_retest=1, /retest |
| #201 | FAILURE | `test_swiglu_x` | sticky detected → comment on PR, escalate |

`test_swiglu_x` ∈ both → sticky. Single PR comment, hand off.

### Build sequence C: PR-induced, ignore the flake noise

| Build | Result | Failing tests | Action |
|---|---|---|---|
| #300 | FAILURE | `test_my_new_op` (PR-INDUCED), `test_unrelated_flake` (UNRELATED) | patch the file for `test_my_new_op`. flake noise is ignored this round. patch_attempts=1, /retest |
| #301 | SUCCESS | — | DONE |

The presence of any PR-INDUCED failure takes precedence over flake
classification — fix the real bug, /retest, and only worry about the
flake if it persists into the next failing build.
