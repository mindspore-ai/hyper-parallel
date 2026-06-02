# Closed-Loop Guarantee

How gate-doctor stays alive on a PR from the first comment to
`ci-pipeline-passed` (or a documented escalation) without leaning on
the user to babysit each `/retest`.

Deliberately simple: no separate ledger file, no extra script
machinery. The agent tracks loop state in conversation memory, polls
via `ScheduleWakeup`, and renders the final report from what it
remembers at terminal exit.

## TL;DR

| Rule | Mechanism |
|---|---|
| Never return to the user with the gate still red | Agent discipline — only exit on one of the documented terminal states below. |
| Sleep between Jenkins polls without burning the prompt cache | `ScheduleWakeup` with adaptive `delaySeconds` (60s → 270s → 1200s). Never raw `Bash sleep` for > 60s. |
| Resume after a session restart | Re-run `gate_doctor.py diagnose <pr>`; pipeline state + PR labels are the source of truth. The agent reconstructs progress from PR state, not a private file. |
| Final summary | Always emitted on terminal exit (see [final-report.md](final-report.md)). |

## Terminal states — the only legitimate exits

Stop the loop and emit the final report when ONE of these is true:

| Terminal | Trigger |
|---|---|
| `ci-pipeline-passed` | PR carries BOTH `pr-check-pass` AND `ci-pipeline-passed` labels after a SUCCESS build. **The only success exit.** Build SUCCESS alone is not enough; a single label alone is not enough; the agent must keep adaptive-polling labels until both are observed. |
| `patch-ceiling` | 2 fix commits in a row that did not move the gate green. |
| `flake-ceiling` | 5 consecutive UNRELATED-different-test builds without going green. |
| `needs-human-guidance` | A PR-INDUCED failure is in Smoke/UT/ST/Ascend/ops, multi-file, assertion/shape/dtype/dispatch/numerical, or has no canonical recipe. The skill stops before patching and surfaces the diagnose JSON so the user can describe the production fix or hand over the specific file to edit. |
| `upstream-sticky-flake` | UNRELATED sticky failure with Path C **not** authorized; PR comment posted, awaiting maintainer. |
| `scope-creep-refused` | Fix would require editing files outside the PR diff without authorization. |
| `user-interrupt` | User explicitly cancelled. |

Anything else — "the build is still running, let me return to the
user", "I'll check back in a bit", "let me wait a few minutes" — is
**not** a terminal state. Schedule a wake-up and continue.

## What the agent must track across the loop

Just two things, both kept in conversation memory:

1. **Failed-test sets per build** — to distinguish random flakes
   (different tests each build) from sticky flakes (same test ≥ 2
   builds). Just an in-memory list of sets keyed by build number.
2. **Temp comment-outs applied** — for each Path-C action: test
   nodeid, file path + line, marker_id, fail builds, commit SHA.
   This is the input to the "REVERT BEFORE MERGE" section of the
   final report.

Plus the obvious counters:

- `patch_attempts` (cap 2)
- `flake_retest_attempts` (cap 5)
- `pr_comments_posted`

If the conversation gets restarted, re-derive what you can from PR
state (`diagnose <pr>` for last build, PR labels for terminal status,
`git log` for previously-committed fixes) and proceed. No file-based
ledger is required — the PR itself is the canonical record.

## Adaptive cadence with ScheduleWakeup

When waiting for a state transition (pr-check-pass label,
ci-pipeline-passed label, build moving from BUILDING to terminal),
the agent should schedule its own next wake-up. The intervals
**grow** as the wait stretches out:

| Wait phase | delaySeconds (clamped to [60, 3600]) |
|---|---|
| First check on a fresh wait | 60 |
| Still pending after 1 min | 270 |
| Still pending after ~5 min | 1200 |
| Still pending after ~25 min | 1800 |
| Builds known to take > 30 min | 3600 |

Why the back-off grows:

- **Fast failures surface fast.** /check-pr verdicts, queue
  rejections, lint-only stages usually fail within the first 1–5
  minutes. A short first poll (60s) catches them early so the agent
  can patch and re-retest without burning a long wait.
- **If the build has survived the first 5 minutes, it's already in
  the heavy stages** (Smoke / UT / ST / Ascend). Those run for
  20–60 minutes, and polling them every minute just costs prompt
  cache without changing anything. So the cadence stretches to 20
  min and then 30 min.
- **Upstream Jenkins infra is not free.** Aggressive polling on a
  shared Jenkins instance is rude; longer intervals at the back
  end respect that.

Reset to 60s whenever the loop moves to a new "wait for X" phase
(e.g. after `/check-pr` succeeds, the loop re-enters this schedule
to wait for the `/retest` build; after build SUCCESS, the loop
re-enters waiting for both labels). The reason field should be
specific: `"poll Jenkins #2451 (Smoke_Ascend, 12m elapsed)"`, not
just "waiting".

Do **not** use Bash `sleep` for waits longer than 60s — long leading
sleeps are blocked by the harness, and stacking short sleeps wastes
the prompt cache.

## Don't exit early on a single SUCCESS

The single most common premature-exit bug: build comes back SUCCESS,
agent reports "gate is green!" and returns to the user, but the PR
doesn't actually carry `ci-pipeline-passed` yet (or carries only
`pr-check-pass`). The user then has to manually re-trigger the
gate-doctor to confirm, defeating the closed-loop guarantee.

Remember the gate is two **sequential** phases:

1. `/check-pr` ⇒ `pr-check-pass` label (phase 1).
2. `/retest` after phase 1 ⇒ pipeline runs ⇒ on SUCCESS,
   `ci-pipeline-passed` label is added on top (phase 2). The two
   labels can land minutes apart; the bot aggregates them serially.

The correct behavior on a SUCCESS build is:

1. Confirm the build is **not** a bypass (`is_bypass_build: false`,
   duration ≥ 60s). A bypass build is a phase-2 fake green and must
   be ignored.
2. Re-read the PR labels.
3. If BOTH `pr-check-pass` AND `ci-pipeline-passed` are present →
   terminal success, emit final report, exit.
4. If only `pr-check-pass` is present → the bot is still
   aggregating the phase-2 verdict. ScheduleWakeup with
   delaySeconds=60, re-check on the growing cadence. Do not exit
   on a single re-check.
5. If `pr-check-pass` has dropped off (a new commit invalidated
   phase 1 mid-flight), restart the loop at phase 1: post
   `/check-pr`, wait for `pr-check-pass`, then `/retest` to
   re-acquire phase 2.

Random flake handling is unchanged — Path A
([flake-policy.md](flake-policy.md) § Path A) keeps `/retest`-ing on
the growing cadence until either the gate passes or the
`flake_retest_attempts` ceiling fires.

## Why this discipline exists

Earlier iterations of gate-doctor would post `/retest`, report the
new pipeline number, and exit. The user then had to invoke the
skill again to check progress, again to post the next `/retest`,
again to read the failure log. That's exactly the babysitting we
should be automating away.

The cost of the discipline is the agent has to remember a few
counters and a list of temp-skipped tests across its wake-ups. The
benefit is the user types one command per PR gate run, not 8–12.
