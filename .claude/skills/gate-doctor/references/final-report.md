# Final Report

Every `autofix <pr>` invocation **must** emit one final report when
the loop terminates — success or escalation. The report is the
contract the gate-doctor makes with the PR author: a single
auditable summary of what was tried, what was changed, and what
(if anything) the author still has to do before merge.

The agent renders this from in-conversation memory (counters,
temp-skipped tests, fix commits). No separate persistence file is
required.

## Template

```markdown
# gate-doctor final report — <owner>/<repo>#<N>

- Status: **<terminal_state>**
- Last pipeline: #<N> (<SUCCESS|FAILURE|ABORTED>) — <jenkins-url>
- Wall time: <Hh Mm>

## Timeline (key events only — not every poll)

- <ts> /check-pr posted
- <ts> pr-check-pass label observed
- <ts> /retest posted (build #N)
- <ts> build #N FAILURE on <test(s)> — <triage>
- <ts> patched <file> — pylint 10.00/10 — commit <sha>
- <ts> temp-skip applied on <test-nodeid> → commit <sha>, /retest posted
- <ts> terminal: <terminal_state>

## Fixes applied this run

(empty section if none)

- **<sha>** `fix(<area>): <subject>` — addresses <Check_X rule R>
  in `<file>:<line>`.

## Temporary comment-outs (REVERT BEFORE MERGE)

(empty section if none — but if present, surface it at the top of
the chat message too, not just in the report body)

> The following test cases were temporarily commented out as
> confirmed UNRELATED to this PR. Each carries a `gate-doctor:
> temp-skip` marker on the line directly above. Revert before merge.

- **<test-nodeid>** at `<file>:<line>`
  - failed in builds: #<N1>, #<N2>
  - triage: Q1=<…>, Q2=<…>, Q3=<…>
  - applied in commit: <sha>
  - revert command: `git revert <sha>` (or `git show <sha> -- <file>
    | git apply -R` for partial revert)

## Unresolved failures

(empty when terminal_state = ci-pipeline-passed; populated otherwise)

- **<stage>** `<rule>` at `<file>:<line>` — `<message>`
  - triage verdict: <PR-INDUCED|UNRELATED>
  - why escalated: <one line>

## Counters

- Patch attempts: <N>/2
- Random-flake retests: <N>/5
- Sticky-flake PR comments posted: <N>
- Temp comment-outs applied: <N>

## What you (the PR author) need to do next

<one short paragraph tailored to the terminal state — see below>
```

## Per-terminal-state next-step language

Pick the matching paragraph for the report's last section:

- **ci-pipeline-passed**:
  *(If any temp comment-outs above)* Revert each temp-skip commit
  before merge — `git revert <sha>`. The gate has to re-pass without
  the comment-outs for the merge to ship the real contract.
  *(Otherwise)* The PR is mergeable.

- **patch-ceiling**: Two patches did not move the gate green. Read
  the "Unresolved failures" section; manual review of the remaining
  failure is required.

- **flake-ceiling**: 5 consecutive UNRELATED-different-test builds.
  Master HEAD may be unstable. Either wait for the next master green,
  rebase onto a known-good master commit, or escalate to a maintainer.

- **needs-human-guidance**: A PR-INDUCED failure is in
  Smoke/UT/ST/Ascend/ops (or multi-file, or has no canonical
  recipe). The skill stopped before patching to avoid shipping a
  shallow fix that masks the real bug. Either describe the
  production fix you want applied, or point gate-doctor at the
  specific production file to edit and re-run `autofix`.

- **upstream-sticky-flake**: gate-doctor has posted a maintainer-facing
  PR comment with triage evidence. Wait for maintainer action, then
  re-launch autofix once the master fix lands.

- **scope-creep-refused**: Fix would require editing files outside the
  PR diff. Decide manually: widen the PR diff, open a prerequisite PR
  first, or escalate to a maintainer.

- **user-interrupt**: You stopped the loop. Re-run `autofix <pr>` to
  resume.

## Field-by-field rules

- **Timeline**: key events only (state transitions, fixes, temp-skips,
  terminal exit) — not every adaptive-poll tick. The user audits
  this against PR comments; timestamps must match.
- **Fixes applied**: every commit SHA the agent pushed in this run,
  in order. If the user wants to revert one, they need the SHA, not
  a paraphrase.
- **Temporary comment-outs**: the single most important non-success
  surface. If non-empty, also surface as a top-level message in chat
  — the author must not miss it.
- **Counters**: include even when zero; auditors read these to
  confirm no ceiling was silently bypassed.

## Worked example (success with one temp-skip)

```markdown
# gate-doctor final report — mindspore/hyper-parallel#651

- Status: **ci-pipeline-passed**
- Last pipeline: #2453 (SUCCESS) — https://build.mindspore.cn/.../2453/
- Wall time: 1h 22m

## Timeline

- 14:02 /check-pr posted
- 14:08 pr-check-pass label observed
- 14:08 /retest posted (build #2451)
- 14:23 build #2451 FAILURE on test_pynative_swiglu_x (UNRELATED, sticky vs #2450)
- 14:38 temp-skip applied on test_pynative_swiglu_x → commit abc1234, /retest posted
- 15:03 build #2452 FAILURE on Check_Pylint W1510 in train.py:118 (PR-INDUCED, trivial)
- 15:04 patched train.py:118 — pylint 10.00/10 — commit def5678, /retest posted
- 15:24 build #2453 SUCCESS — ci-pipeline-passed label observed
- 15:24 terminal: ci-pipeline-passed

## Fixes applied this run

- **def5678** `fix(train): satisfy Check_Pylint (W1510)` — added
  `check=False` to the `subprocess.run` call in `train.py:118`.

## Temporary comment-outs (REVERT BEFORE MERGE)

> Revert before merge. One test was temp-skipped as a confirmed
> unrelated flake.

- **tests/st/pynative/test_swiglu.py::test_pynative_swiglu_x** at
  `tests/st/pynative/test_swiglu.py:88`
  - failed in builds: #2450, #2451
  - triage: Q1=no, Q2=no, Q3=passed-on-master
  - applied in commit: abc1234
  - revert command: `git revert abc1234`

## Unresolved failures

(none — gate is green)

## Counters

- Patch attempts: 1/2
- Random-flake retests: 0/5
- Sticky-flake PR comments posted: 0
- Temp comment-outs applied: 1

## What you (the PR author) need to do next

Revert commit `abc1234` (the temp-skip on test_pynative_swiglu_x)
before merge: `git revert abc1234`. Re-run `autofix 651` after revert
— if the gate still passes, the unrelated flake has cleared on master;
if it sticks, escalate to the maintainer for a master-side fix.
```
