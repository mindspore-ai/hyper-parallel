# Root-Cause Fix Policy

How gate-doctor patches a PR-INDUCED failure: where the fix is allowed
to land, what's forbidden, when to stop and ask for human guidance.

## Contents

- [TL;DR](#tldr)
- [The hard rule: fix production for PR-INDUCED failures](#the-hard-rule-fix-production-for-pr-induced-failures)
- [The two PR-INDUCED exceptions](#the-two-pr-induced-exceptions)
- [Complexity classification: trivial vs deep](#complexity-classification-trivial-vs-deep)
- [Locating the production cause](#locating-the-production-cause)
- [After the patch](#after-the-patch)
- [Counter-example: do not do this](#counter-example-do-not-do-this)

## TL;DR

| The failure is | Verdict | Action |
|---|---|---|
| Trivial, recipe in [common-failures.md](common-failures.md) | PR-INDUCED | Patch the file the linter named; local pylint 10.00/10; `autogit commit`; `/retest`. |
| Smoke/UT assertion, shape/dispatch/numerical, multi-file, or no canonical recipe | PR-INDUCED | **Stop before patching.** Set `terminal_state = "needs-human-guidance"`, emit the final report with the diagnose JSON, and ask the user to describe the production fix or hand over the specific production file to edit. |
| Test contract genuinely changed by PR (rare, needs user authorization) | PR-INDUCED | Update the test in a follow-up commit per the user's confirmation. |
| Confirmed unrelated, sticky across ≥2 builds, PR-author authorized | UNRELATED | Path C temp comment-out on the test in the PR branch, with revert-before-merge marker. See [flake-policy.md](flake-policy.md) § "Path C". |
| Confirmed unrelated, sticky, NOT PR-author authorized | UNRELATED | Path B PR comment + escalation. |
| Unrelated, different test each build (random) | UNRELATED | Just `/retest`; cap at 5. |

The gate fails because either the PR broke something (PR-INDUCED →
fix the breakage in production code, with human guidance for deep
ones) or upstream is flaky (UNRELATED → retest, comment-and-escalate,
or — only with PR-author authorization — diagnostic temp-skip on the
test). Anything that makes a PR-INDUCED test pass without fixing the
breakage is a regression masquerading as a green gate.

## The hard rule: fix production for PR-INDUCED failures

The rule below applies when the 3-question triage from
[flake-policy.md](flake-policy.md) verdicts a failure as
**PR-INDUCED**. The handling for UNRELATED failures (random,
sticky-with-author-authorization, sticky-without-authorization) is
covered separately in flake-policy.md — do **not** apply Path C
(temp comment-out) to a PR-INDUCED failure under any circumstance,
even if the test is tedious to fix.

When the failure is PR-INDUCED, the fix **must** land in production
code — the code the PR touched, or the production code the PR's
changes broke. You **must not**, even when the diff looks small or
the deadline is tight:

- Add `@pytest.mark.skip`, `@pytest.mark.skipif`,
  `@pytest.mark.xfail`, `@pytest.mark.skipfile`, conditional skips
  inside a test function, or any other decorator/branch that
  prevents the assertion from running.
- Drop parametrize entries that fail (e.g. shrinking
  `parametrize('shape', [s1, s2, s3])` to `[s1]` because `s2` fails).
- Loosen tolerance (`rtol`, `atol`, `eps`, comparison margin) to make
  a numerical assertion pass. The current tolerance is the contract
  the test enforces.
- Replace `assert x == y` with `print(x, y)`, `try: assert …; except`,
  or any other softening.
- Replace a real comparison with `assert x is not None` /
  `assert True`.
- Mock the failing op's output (e.g. patch `ms.ops.foo` to return a
  fixed tensor inside the test).
- Move the test to a directory that isn't picked up by the gate.
- Hide the test under a `@arg_mark(level_mark='level…')` change that
  removes it from the gate's level0 selection.

Why: the test is the contract the PR is claiming to satisfy. If a
PR-INDUCED change makes a test fail, the PR is shipping a contract
violation. Disabling the test ships the violation. The reviewer can't
catch it on diff alone because the assertion still exists in the
code — it just doesn't run.

## The two PR-INDUCED exceptions

There are exactly two cases where touching a test is acceptable as
part of the autofix loop. Both are narrow; if you find yourself
arguing the case for a third exception, escalate to the caller.

### Exception 1 — PR intentionally changes a contract

The PR *intentionally* changes a contract that the test encodes, and
the PR description (or a maintainer comment on the PR) already
declares this contract change.

Example: a PR changes an op's return shape from `(B, S, H)` to
`(B, S, H, 1)` for downstream compatibility. The shape test was
asserting `(B, S, H)`. The test must be updated to match the new
shape.

Even here, the right move is usually:

1. The PR description has already explained the contract change and
   listed the tests that need updating.
2. The caller has explicitly authorized "you may update test X to
   match the new contract in commit Y".
3. The test update lands in a separate commit with a message like
   `test(...): align with new <op> contract from <commit>`.

Without all three, escalate to the caller. Don't decide unilaterally.

### Exception 2 — UNRELATED-and-sticky temp comment-out (Path C)

Strictly speaking this is **not** an exception to the PR-INDUCED
rule — by construction Path C never fires for a PR-INDUCED
verdict. It is listed here because the surface action (an edit
in a test file) looks the same to a casual reader, and the
distinction matters.

Path C requires:

1. **3-question triage verdicts UNRELATED** (Q1=no, Q2=no, Q3 says
   the test passes on a recent master build, or no Q3 datapoint
   plus Q1+Q2 both clearly no with cited evidence).
2. **Sticky across ≥2 consecutive builds** in the same loop.
3. **PR-author authorization** (default-on when running on the
   author's own fork branch; opt-out via `--no-temp-skip`).
4. **The marker syntax + ledger entry + revert-before-merge
   clause in the final report** as specified in
   [flake-policy.md §Path C](flake-policy.md).

If any of (1)-(4) is missing, this is **not** Path C, and the
underlying failure must take one of the other paths (production
fix for PR-INDUCED, /retest for random flake, PR comment for
sticky-flake-without-authorization).

The distinguishing question to ask before applying any test edit
is: "Is the failing test pointing at code the PR touches (or
broke), or is it flaky on master regardless of this PR?" If the
answer is "PR's code", you're in PR-INDUCED land — never touch
the test. If the answer is "flaky on master", you may be in Path
C — verify all four bullets above before editing.

## Complexity classification: trivial vs deep

After diagnose tags a failure as PR-INDUCED, classify it as **trivial**
or **deep**. Trivial fixes get patched immediately; deep fixes stop and
ask for human guidance before editing.

### Trivial (patch immediately)

A trivial fix is one where:

- The failure comes from a `Check_*` linter stage, AND
- A canonical fix recipe exists in
  [common-failures.md](common-failures.md), AND
- The fix is a single-file edit, ≤5 lines, mechanically derivable
  from the linter message and the recipe.

Examples:

- Check_Pylint W1510 → add `check=False` to the `subprocess.run`
  call at the named line.
- Check_Pylint W0612 → remove the unused variable assignment.
- Check_DT_Design missing fields → add `Feature:`, `Description:`,
  `Expectation:` lines to the docstring.
- Check_Codespell → swap the typo for the suggested word.
- Check_Tab → convert tabs to spaces in the named line range.
- Check_Markdownlint MDxxx → apply the recipe for that rule code.

### Deep (stop and ask for human guidance)

A deep fix is one where any of these are true:

- The failure comes from a `Smoke_*` test stage (`Smoke_Ascend`,
  `Smoke_CPU`, `Smoke_GPU`, `ut`, `st`, etc.).
- The failure is an assertion / numerical / shape / dtype / dispatch
  error that requires understanding the op's semantics.
- The fix touches multiple files (e.g. `op_def/yaml/*.yaml` plus
  `ops/infer/ops_func_impl/*.cc` plus a Python wrapper).
- No canonical recipe exists in `common-failures.md`.
- The diagnose log surfaces a stack trace into C++ kernels or
  generated dispatch code.

For any deep fix, **stop the autofix loop before editing**. Set
`terminal_state = "needs-human-guidance"`, emit the final report,
and surface this message to the user:

```text
[needs-human-guidance] Reason: PR-INDUCED failure in <test-nodeid>,
stage=<Smoke_Ascend|...>, signature=<one-line excerpt>.

Production cause is likely in <best-guess directory under
mindspore/...> based on triage Q2.

This is in the ops / kernel / dispatch surface and a blind patch
risks shipping a shallow fix that masks the real bug. I'd like
your guidance before editing.

Please either:
  (a) Describe the production fix you want applied, and I'll edit
      the file and run /retest.
  (b) Point me at the specific production file(s) you want me to
      edit, and I'll propose a diff before committing.
  (c) Mark this PR for manual maintainer review.

Diagnose JSON, full pipeline URL, and the patch_attempts /
flake_retest counters are attached below.
```

Why we stop instead of trying a patch: ops / kernel / dispatch
failures usually need more context than the Jenkins log gives —
the op's semantics, related YAML, upstream callers. A shallow
patch can look like it works (the test passes) while actually
masking a deeper bug. Human guidance before the edit is cheap
insurance.

After you provide the guidance, re-run `autofix <pr>` — the loop
picks up from a fresh diagnose, applies the directed fix, and
continues the close-the-gate cycle.

## Locating the production cause

When the user hands over guidance on a deep fix, the agent still
needs to find the right file to edit. Use this in order:

1. **Diagnose's raw_log_excerpt** — the assertion line and surrounding
   stack frames usually name the immediate caller.
2. **PR diff** — `git diff upstream/master...HEAD` lists every file
   the PR touched. The production cause is almost always in this set,
   especially if the diff introduces a new op, changes a signature,
   or modifies a kernel.
3. **Symbol grep** — grep the failing test for ops / functions it
   exercises, then grep PR diff for those symbols.
4. **Infer/dispatch trace** — for shape/dtype errors, follow the
   `ops/op_def/yaml/<op>_op.yaml` → `ops/infer/ops_func_impl/<op>.cc`
   chain; for dispatch errors, follow `ops/api_def/<op>.yaml`.

If the production cause is **not** in the PR diff (e.g. failing test
points to a file the PR doesn't touch), revisit the 3-question
triage — the failure may not actually be PR-INDUCED. A common
miss-call: the PR changes a YAML auto-generation input, and the
resulting generated C++ file (which the PR's diff does *not* show)
is broken downstream.

## After the patch

Before committing any production-code fix:

1. **Local pylint** the touched `.py` files with the MindSpore rcfile;
   must hit 10.00/10.
2. **Local build sanity** for C++ touches: `bash build.sh -e cpu -j8`
   at minimum, ideally the same backend as the failing smoke stage.
3. **Local repro of the failing test** if possible:
   `pytest <test-nodeid> -v`. If the test now passes, that's the
   strongest signal the fix is real.
4. **Diff the patch one more time** to confirm: no test files
   modified, no decorators added, no tolerances loosened, no
   parametrize entries removed.

Then `autogit commit -m "fix(<area>): <one-line root cause>"` and
/retest.

## Counter-example: do not do this

A PR adds a new MoE op `expert_bias_basic`. The gate's Smoke_Ascend
stage runs `test_pynative_expert_bias_basic` which asserts the op's
output equals a reference value. The test fails with
`AssertionError: 0.1234 != 0.1245, atol=1e-3`.

The wrong (forbidden) move:

```python
# Don't:
-np.testing.assert_allclose(out, ref, atol=1e-3)
+np.testing.assert_allclose(out, ref, atol=1e-2)  # loosened
```

The right move: read the new op's kernel, find the numerical bug
(e.g. wrong scaling factor), fix it in the kernel C++ source, rebuild,
re-run locally, then commit the kernel fix. The test stays untouched.
The reviewer can see exactly what the PR did wrong.

If after 2 deep-fix attempts the kernel fix still doesn't land the
test under the original tolerance, escalate to the caller. The
correct outcome at that point is human review of the op's numerical
contract, not a tolerance bump.
