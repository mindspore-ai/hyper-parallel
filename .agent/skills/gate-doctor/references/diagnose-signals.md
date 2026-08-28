# Diagnose Signals & Dual-Phase Gate

How to read `gate_doctor.py diagnose` output and decide the next
action. Loop cadence / terminal states → [closed-loop.md](closed-loop.md).
Per-failure routing → [flake-policy.md](flake-policy.md).

## Finding the latest *real* failure

The PR comment stream is noisy: every `/retest` and `/check-pr` adds
comments, plus bot "trigger blocked" notices. Across a long session
you can see 30+ comments; naive "newest comment wins" is wrong.

Thread **three independent latest-signals**:

1. **Latest real pipeline result** — query the **Jenkins job API
   directly** (not the PR comment table). Pick the newest build whose
   description matches `Pull Request #<N>`, and **skip ABORTED**.
   `/retest` spam aborts intermediate builds, but the bot's build-result
   comment can lag and still show ABORTED as "latest". Jenkins API is
   monotonic by build number; the comment table is not.
2. **Latest /check-pr verdict** — walk PR comments newest-first, or
   read the `pr-check-pass` label. This is **independent of the
   pipeline**; SUCCESS pipeline can co-exist with a failing /check-pr.
3. **Bypass-build detection** — a Jenkins build that returns SUCCESS
   in under 60s with zero parsed failures is a *bypass build*: someone
   posted `/retest` while `pr-check-pass` was missing (~17s, proves
   nothing). The script tags these `is_bypass_build: true`; treat as
   "not a real result".

`gate_doctor.py diagnose` returns these as explicit fields
(`pipeline_number` / `pipeline_result` / `is_bypass_build` /
`check_pr_failure` / `gate_trigger_blocked`) plus aggregate `gate_ok`.
**Trust those — never invent a build number from a comment URL.**

## Two failure surfaces, in order

The GitCode gate has **two sequential phases**, not two parallel ones:

```
Phase 1 — description gate
    post /check-pr   ────►   pr-check-pass label appears
                              (PR description + self-checklist
                               + 设计 + linked issue all valid)

Phase 2 — pipeline gate (only valid while pr-check-pass is fresh)
    post /retest     ────►   Jenkins pipeline runs
    pipeline SUCCESS ────►   ci-pipeline-passed label appears
    (all Check_* + Smoke + UT + ST stages green)

Mergeable when BOTH labels are on the PR.
```

Why the order matters:

- `/retest` before `pr-check-pass` triggers a bypass build
  (`is_bypass_build: true`) — looks green, proves nothing.
- A new commit can invalidate `pr-check-pass` mid-loop; go back to
  phase 1, `/check-pr` again, then `/retest`.
- The two labels can land minutes apart; keep polling until both appear.

### Diagnose → next action

| `check_pr_failure` | `pipeline_result` | Phase | Next action |
|---|---|---|---|
| set | (any) | Phase 1 failing | Patch PR description via autogit; post `/check-pr`; restart. Do **not** `/retest` until `pr-check-pass` is current. |
| null | `null` (no build yet) | Between phases | Phase 1 may have just passed. Post `/retest` once, then poll. |
| null | `BUILDING` | Phase 2 mid-flight | ScheduleWakeup; re-poll on growing cadence ([closed-loop.md](closed-loop.md)). |
| null | `FAILURE` | Phase 2 failing | Route per [flake-policy.md](flake-policy.md) § Per-failure decision tree. |
| null | `SUCCESS` + `is_bypass_build: true` | Phase 2 fake green | Not a real result. Continue adaptive-polling. |
| null | `SUCCESS` + `is_bypass_build: false`, only `pr-check-pass` | Phase 2 done, bot aggregating | Don't exit. Poll until `ci-pipeline-passed` also appears. |
| null | `SUCCESS` + `is_bypass_build: false`, both labels | Both phases green | Terminal: `ci-pipeline-passed`. Emit final report. |
| null | `SUCCESS`, `pr-check-pass` dropped | Phase 1 invalidated | Restart phase 1: post `/check-pr` again. |

Always post `/check-pr` *before* every `/retest`. Terminal success is
**BOTH labels**, not just build SUCCESS.
