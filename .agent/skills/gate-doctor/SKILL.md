---
name: gate-doctor
description: "Drive a red MindSpore-family GitCode PR gate to green end-to-end — read state, post /check-pr → /retest, classify every failure (root-cause fix in production code for PR-INDUCED, /retest only for random flakes, REVERT-BEFORE-MERGE temp comment-out for confirmed unrelated sticky flakes), then loop until both pr-check-pass AND ci-pipeline-passed labels appear. Emits one structured final report on exit. Trigger on: 触发门禁 / 重跑门禁 / 门禁失败 / PR 流水线挂了 / 修一下流水线 / autofix 门禁 / 把 PR 修绿 / 一直跑到通过 / 看下 #N 现在情况 / ut/st/Ascend 用例失败 / Smoke_Ascend 挂了 / 门禁随机挂 / gate is failing / CI is flaky / random retest pass / /retest / /check-pr, or any prompt naming a MindSpore GitCode PR alongside a request to investigate or fix CI. Default PR repo is mindspore/hyper-parallel; pass full URL or `mindspore/mindspore#N` for cross-repo."
---

# Gate Doctor — 全自动门禁修复 Skill

End-to-end PR pipeline diagnose + autofix for MindSpore-family GitCode
repositories. The skill teaches the AI agent how to:

1. Read a PR's gate-comment stream and locate the **latest** failing pipeline.
2. Pull the failing Jenkins build's console log.
3. Classify which `Check_*` stage failed and **why** (with file:line evidence).
4. Edit the offending source to satisfy the rule.
5. Re-run pylint locally with the MindSpore rcfile to confirm the fix.
6. Commit + push via the `autogit` skill (must already be configured).
7. Re-trigger the gate via `/retest` (or `/check-pr` when the description gate
   was the failing check).

## Tool Requirements

- `GITCODE_TOKEN` env var (same one autogit uses) with PR-comment write
  permission.
- `git` with `origin` (fork) + `upstream` (main repo) remotes already
  configured.
- `python3.8+` with `pylint==3.3.7` available locally for pre-push pylint
  parity with CI (`pip install pylint==3.3.7 -i https://pypi.tuna.tsinghua.edu.cn/simple`).
- A working `autogit` skill in the same `.agent/skills/` tree — `gate-doctor`
  shells out to `autogit commit` to land patches.

## When to Activate

| User says | Subcommand | Purpose |
|---|---|---|
| "看下 #N 现在情况" / "PR 状态" / 用户只给一个 PR 链接没有动词 / "看下门禁报错" / "PR 流水线挂了" / "诊断 #N" | `diagnose <pr>` | Read-only snapshot. Parses latest non-ABORTED Jenkins build + PR labels, emits JSON failure report. |
| "触发门禁" / "重跑门禁" / "/retest" | `retest <pr>` | Post `/retest` on the PR. |
| "提 check-pr" / "重新校验描述" / "/check-pr" | `check-pr <pr>` | Post `/check-pr` to re-trigger the description / self-checklist validator. |
| "自动修门禁" / "autofix #N" / "把 PR 修绿" / "一直跑到通过" | `autofix <pr>` | Closed-loop autonomous job — drives the full /check-pr → /retest → triage → fix loop. |

For full parameter details: `python3 {skill_dir}/scripts/gate_doctor.py <cmd> --help`.

**PR reference resolution**:

- Bare number (`647`) or `#647` → defaults to `mindspore/hyper-parallel`
  (the repo this skill lives in).
- Full GitCode URL (`https://gitcode.com/mindspore/mindspore/pull/92567`)
  → uses the owner/repo from the URL.
- Short form `mindspore/mindspore#92567` → same as above, terser.
- For repos *not* in the built-in Jenkins-job mapping
  (`mindspore/mindspore` and `mindspore/hyper-parallel`), pass
  `--jenkins-job=<name>` so the script knows where to query.

If the user invokes a trigger phrase without a PR identifier:
- Check the current branch via `git rev-parse --abbrev-ref HEAD`. If it
  matches a known PR head branch on origin, resolve the PR number from the
  GitCode API for the repo this branch belongs to.
- Otherwise ask the user for the PR number / URL once.

## Invocation

Run from anywhere (script resolves paths via `git rev-parse --show-toplevel`):

```bash
python3 {skill_dir}/scripts/gate_doctor.py <subcommand> <pr_ref> [options]
```

Temp-skip itself is just a normal file edit with the canonical
marker syntax (see [references/flake-policy.md](references/flake-policy.md)
§ Path C). No separate subcommand; the agent applies it via Read +
Edit + autogit so reviewers can audit the diff like any other commit.

## Workflows

### First-touch on a PR link

```
user: https://gitcode.com/mindspore/hyper-parallel/pull/651 看下现在情况
agent: Running gate-doctor `diagnose` …
        → latest pipeline #2451 BUILDING (Smoke_Ascend), 8m elapsed
        → /check-pr verdict: pass (label pr-check-pass fresh)
        → no failed stages on the previous build
        要我直接 `autofix 651` 一直跟到通过吗?
```

Always start with `diagnose` when the user hands you only a PR link.
Never post /check-pr or /retest blindly — the diagnose output
dictates the next action.

### Quick gate retrigger (default repo = hyper-parallel)

```
user: 触发一下 #647 的门禁
agent: Running gate-doctor `retest` …
        → POST /comments {"body": "/retest"} on mindspore/hyper-parallel#647
        → pipeline #2451 is running.
```

### Cross-repo retrigger (full URL form)

```
user: 触发一下 https://gitcode.com/mindspore/mindspore/pull/92567 的门禁
agent: Running gate-doctor `retest` …
        → POST /comments {"body": "/retest"} on mindspore/mindspore#92567
        → pipeline #3912 is running.
```

### Full autofix closed loop (canonical)

```
user: 修一下 #647 的流水线
agent: Running gate-doctor autofix …
        [12:00:00] /check-pr posted on #647
        [12:01:00] pr-check-pass label still pending, waiting 5m
        [12:06:00] pr-check-pass label present
        [12:06:01] /retest posted, pipeline #2451 enqueued
        [12:07:01] build #2451 BUILDING (1m wait)
        [12:12:01] build #2451 BUILDING (5m wait)
        [12:22:01] build #2451 FAILURE — diagnosing
        [12:22:30] Check_Pylint W1510 subprocess-run-check in foo.py:12
                   3-question triage: Q1 NO, Q2 YES, Q3 NO → PR-induced
                   patching foo.py — added check=False
        [12:22:45] pylint --rcfile=ms_pylintrc → 10.00/10
        [12:22:50] autogit commit "fix(foo): satisfy Check_Pylint" → SHA abc1234
        [12:22:55] /retest posted, pipeline #2452 enqueued
        [12:23:55] build #2452 BUILDING (1m wait)
        … (back-off continues 1m → 5m → 10m → 10m …)
        [12:55:00] build #2452 SUCCESS, ci-pipeline-passed label present
        autofix complete: PR #647 is green at pipeline #2452.
```

Full per-rule rationale lives in the **Hard rules** section below.
The key shape: /check-pr precedes every /retest; cadence grows
(60s → 270s → 1200s → 1800s); failures route through the 3-question
triage; terminal success requires **both** `pr-check-pass` and
`ci-pipeline-passed` labels on the PR.

Step-by-step per-subcommand playbook (commit-message picks, pylint
flags, recovery from stuck gate):
[references/workflow.md](references/workflow.md).

## Closed-loop guarantee

`autofix` is **persistent by contract**. Once invoked it must not
return to the user until the gate is green or a terminal state
fires. Practical mechanics:

- **ScheduleWakeup** drives the wait phases (60s → 270s → 1200s →
  1800s, clamped to [60, 3600]). The reason field is always
  concrete: `"poll Jenkins #2451 (Smoke_Ascend, 12m elapsed)"`.
- **Never use raw `sleep`** for waits > 60s — long leading sleeps
  are blocked by the harness and stacking short sleeps wastes the
  prompt cache.
- **Terminal exit only on one of seven states** —
  ci-pipeline-passed, patch-ceiling, flake-ceiling,
  needs-human-guidance, upstream-sticky-flake, scope-creep-refused,
  user-interrupt. "I'll check back later" is **not** a terminal
  state and may not be used to return early.
- **Resume after a session restart** — re-run `diagnose <pr>`; PR
  labels + Jenkins state are the source of truth. The agent
  reconstructs counters / temp-skip history from PR state, not a
  private file.

Full discipline + the seven terminal-state table live in
[references/closed-loop.md](references/closed-loop.md).

## Final report contract

Every `autofix` invocation must emit one structured final report on
terminal exit — success or escalation. The report is the
author-facing summary of the entire run:

- Timeline (key state transitions, fixes, temp-skips, and the
  terminal exit — not every adaptive-poll tick).
- Fixes applied (every commit SHA + the failure it addressed).
- Temporary comment-outs (one bullet per Path-C entry, with the
  revert command). This section is the load-bearing surface — if
  non-empty on a successful green, the report's "What you need to
  do next" mandates revert before merge.
- Unresolved failures (with triage verdicts) when the terminal
  state isn't ci-pipeline-passed.
- Counters (patch attempts, random-flake retests, sticky comments,
  temp comment-outs).
- Next-step instructions tailored to the terminal state.

The full template + per-terminal-state next-step list lives in
[references/final-report.md](references/final-report.md). The
report is rendered from in-conversation memory and printed to chat
as the agent's last message.

## Sticky-flake routing: Path B vs Path C

When the 3-question triage verdicts an UNRELATED-and-sticky
failure, there are two valid paths. The skill picks based on
authorization:

- **Path C (temp comment-out)** — fires when the user is the PR
  author, running on their own fork branch, and has not asked the
  agent to skip this path. Apply the `gate-doctor: temp-skip`
  marker on the failing test in the PR branch with a self-generated
  marker-id; commit via autogit; the final report flags the entry
  for revert-before-merge. Full policy in
  [references/flake-policy.md](references/flake-policy.md) § Path C.
- **Path B (PR comment + escalate)** — fires when temp-skip is
  not authorized (cross-repo PR, or the user has said "don't
  comment-out anything" earlier in the session). Post a
  maintainer-facing PR comment with triage evidence; the loop
  terminates (`upstream-sticky-flake`).

Default is Path C when running on the author's own branch.

The catalogue of common gate failures + fix recipes
(`Check_DT_Design` / `Check_Pylint` / `Check_Cpplint` / `Check_Codespell` /
`Check_Tab` / `Check_Utf8` / `Check_Markdownlint` / `Check_Cmakelint` /
`Check_Lizard` / `Check_ShellCheck` / `Check_ClangFormat` / `Check_Linklint` /
`Check_Cppcheck` / `Check_Notebooklint` / `Check_Rstlint` / `Check_Scanoss`,
plus `/check-pr` failures) lives in
[references/common-failures.md](references/common-failures.md). **Always
consult that file before making edits** — the right fix is usually one of the
recipes there.

## Finding the latest *real* failure

The PR page is the noisy part. Every `/retest` adds a new build-result
comment, every `/check-pr` adds a new verdict comment, and the
GitCode bot also posts intermediate "trigger blocked" notices.
Across a long session you can easily see 30+ comments, and naive
"newest comment wins" reads the wrong thing.

The skill threads three independent latest-signals, each with its own
source-of-truth path:

1. **Latest real pipeline result** — query the **Jenkins job API
   directly** (not the PR comment table). Pick the newest build whose
   description matches `Pull Request #<N>`, and **skip ABORTED**.
   *Why:* /retest spam aborts intermediate builds, but the bot's
   build-result comment can lag minutes behind and still show the
   ABORTED entry as "latest". The Jenkins API is monotonic by build
   number, the comment table isn't.
2. **Latest /check-pr verdict** — walk PR comments newest-first,
   match `当前/check-pr未通过` or the pr-check-pass label. This is
   **independent of the pipeline**; a SUCCESS pipeline can co-exist
   with a failing /check-pr and vice versa.
3. **Bypass-build detection** — a Jenkins build that returns SUCCESS
   in under 60s with zero parsed failures is a *bypass build*: it
   ran because someone posted `/retest` while `pr-check-pass` was
   missing, finished in ~17s, and proves nothing. The script tags
   these `is_bypass_build: true`; treat them as "not a real result".

`gate_doctor.py diagnose` returns these as three explicit fields
(`pipeline_number` / `pipeline_result` / `is_bypass_build` /
`check_pr_failure` / `gate_trigger_blocked`) plus an aggregate
`gate_ok`. Trust those — never invent a build number from a comment
URL by hand.

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

- `/retest` posted before `pr-check-pass` triggers a ~17-second
  **bypass build**: it returns SUCCESS without running anything, and
  the bot's "Build Result" comment makes the PR *look* green even
  though `ci-pipeline-passed` is missing. The skill detects these
  (`is_bypass_build: true`) and treats them as "not a real result".
- A new commit invalidates `pr-check-pass` mid-loop. If that happens
  *during* a pipeline run, the build can complete but
  `ci-pipeline-passed` will never land — the loop must go back to
  phase 1, post `/check-pr` again, then re-`/retest`.
- The two labels can land minutes apart even on a clean run; the
  bot aggregates them serially.

A given `diagnose` call returns one of these shapes, and each shape
maps to a specific next action:

| `check_pr_failure` | `pipeline_result` | Phase | Next action |
|---|---|---|---|
| set | (any) | Phase 1 failing | Patch the PR description via autogit; post `/check-pr`; restart the iteration. Do **not** post `/retest` until `pr-check-pass` is current. |
| null | `null` (no build yet) | Between phases | Phase 1 may have just passed. Post `/retest` once, then poll. |
| null | `BUILDING` | Phase 2 mid-flight | ScheduleWakeup; re-poll on the growing cadence. |
| null | `FAILURE` | Phase 2 failing | Route per [flake-policy.md § Per-failure decision tree](references/flake-policy.md): production fix / Path C / Path B / random retest. |
| null | `SUCCESS` + `is_bypass_build: true` | Phase 2 fake green | Bypass — not a real result. Continue adaptive-polling; the real phase-2 build is still pending or never started. |
| null | `SUCCESS` + `is_bypass_build: false`, only `pr-check-pass` label | Phase 2 done, bot still aggregating | Don't exit. Adaptive-poll labels until `ci-pipeline-passed` also appears. |
| null | `SUCCESS` + `is_bypass_build: false`, both labels | Both phases green | Terminal: `ci-pipeline-passed`. Emit final report. |
| null | `SUCCESS`, `pr-check-pass` has dropped | Phase 1 invalidated mid-flight | Restart at phase 1: post `/check-pr` again. |

This is why the autofix loop posts `/check-pr` *before* every
`/retest`, and why the terminal condition is "BOTH labels", not just
"build SUCCESS".

## Hard rules

The per-failure routing flowchart (Path A random / Path B comment /
Path C temp-skip / production fix) lives in
[references/flake-policy.md](references/flake-policy.md) §
"Per-failure decision tree". The rules below govern the loop shape
and the edit-surface boundary.

**Loop shape**

- **Source of truth is the Jenkins-direct latest build.**
  `gate_doctor.py diagnose` queries the Jenkins job API for the
  newest non-ABORTED build. **Why:** the bot's "build result
  table" comment lags minutes behind, and `/retest` spam produces
  ABORTED builds that masquerade as the latest in the comment
  stream.
- **Never invent a Jenkins URL or pipeline number** — only use
  what `diagnose` returns. **Why:** a hallucinated build number
  points at someone else's build; the patch / triage decision
  becomes random.
- **`/check-pr` is the entry to every `/retest`.** Never post
  `/retest` unless the PR carries a fresh `pr-check-pass` label.
  If the label is missing or stale, post `/check-pr` first, patch
  the description on failure, restart the iteration. **Why:**
  posting `/retest` without `pr-check-pass` triggers a ~17-second
  bypass build that proves nothing.
- **Adaptive polling cadence (grows):** 60s → 270s → 1200s → 1800s,
  via ScheduleWakeup. Reset to 60s on every new "wait for X"
  phase. Never raw `sleep` > 60s. Full rationale + table:
  [references/closed-loop.md](references/closed-loop.md).
- **Terminal success = `pr-check-pass` THEN `ci-pipeline-passed`,
  both observed.** The two labels are a sequence, not a pair:
  phase 1 (`pr-check-pass` via `/check-pr`) gates phase 2
  (`ci-pipeline-passed` via `/retest`-driven pipeline SUCCESS).
  Build SUCCESS alone is not enough; only `pr-check-pass` is not
  enough; only `ci-pipeline-passed` cannot legitimately appear
  without phase 1 anyway. **Why:** the bot aggregates the two
  labels serially and they often land minutes apart; exiting at
  build SUCCESS leaves the PR unmergeable and forces the user to
  re-invoke. See "Two failure surfaces, in order" above.
- **Closed-loop discipline.** Once `autofix` is invoked it does
  not return until one of the seven terminal states fires.
  "I'll check back later" is not a terminal state. **Why:**
  babysitting is exactly the work this skill exists to remove.
- **Patch ceiling: 2.** If 2 fix commits in a row don't move the
  gate green, stop and surface (`patch-ceiling`). **Why:** if the
  third patch hasn't landed it, `diagnose` is probably wrong
  about the root cause, and more autonomous edits dig the hole
  deeper.

**Edit-surface boundary**

- **Never edit a file without first running `diagnose`.**
  **Why:** blind patching is the dominant source of "fix attempt
  broke something else" regressions.
- **Never `git push --force`.** Append a new fix commit and
  `/retest`. **Why:** force-push on a gate-running branch races
  with the runner and can produce nonsense results.
- **Always re-run local pylint with the MindSpore rcfile after
  editing.** **Why:** CI uses the same rcfile + pylint 3.3.7;
  hitting 10.00/10 locally is the cheapest way to know the next
  push won't re-fail Check_Pylint.
- **Out-of-diff failures are NOT automatically unrelated.** Run
  the 3-question triage before declaring "upstream flake".
  **Why:** submodule pin bumps, env-var changes, and conftest
  edits in the PR can break tests elsewhere in the tree.
- **PR-INDUCED failures must be root-cause-fixed in production
  code.** No `@pytest.mark.skip` / `xfail`, no tolerance
  loosening, no dropping parametrize entries, no assertion
  softening, no `arg_mark` level changes. **Why:** the test is
  the contract the PR claims to satisfy — disabling it ships the
  violation while the assertion still lexically exists, which a
  reviewer cannot catch on diff alone. Full list +
  the one narrow exception:
  [references/root-cause-fix.md](references/root-cause-fix.md).
- **Trivial fix → patch directly. Deep fix → surface for human
  guidance before editing.** Trivial = a `Check_*` linter with a
  recipe in [common-failures.md](references/common-failures.md),
  single file, ≤5 LoC. Deep = Smoke / UT / ST / Ascend / ops,
  assertion / shape / dtype / dispatch / numerical, multi-file,
  or no canonical recipe. Deep fix sets `terminal_state =
  "needs-human-guidance"`, emits the final report with the
  diagnose JSON attached, and asks the user to either describe
  the production fix or hand the agent the specific file to
  edit. **Why:** ops kernel / dispatch fixes usually need more
  context than the failure log gives (op semantics, related YAML,
  upstream callers); a blind patch ships a masking fix the gate
  may accidentally accept.
- **Final report always.** Every terminal exit emits the
  template in [final-report.md](references/final-report.md).
  Required even on escalation. **Why:** the report is the audit
  trail; missing it leaves no record of what was tried.

## Safety Guarantees

- **Diagnose is read-only.** No file edits, no API writes.
- **`retest` / `check-pr` post a single comment.** No code or branch changes.
- **`autofix`** never edits files outside the PR's own diff, never modifies
  `upstream/master`, and never touches `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md`
  unless the failure is explicitly about the template itself.
- **Patch ceiling:** `autofix` performs at most **2** edit-and-commit
  iterations per invocation. The third iteration must be the user's call.

## Important Restrictions

- No AI-assistant attribution trailers (e.g. `Co-Authored-By: …`) in commit messages. **Why:** the
  project's `autogit` commit-msg hook bounces commits that carry
  third-party-tool attribution; the commit gets rejected at push
  time.
- Don't touch `mindformers` / `metadef` / other submodules unless
  the gate failure is inside the submodule's tracked files (rare).
  **Why:** submodules are maintainer-owned; pushing a fix into a
  submodule from a wrapper-repo PR bypasses the submodule
  maintainer's review and creates a divergent pin nobody asked for.
- If `.agent/rules/code-style.md` exists in the active repo, load
  it before every patch decision and use it as a tie-breaker
  against pylint rules. **Why:** project conventions occasionally
  override the upstream rcfile (e.g. lazy-import patterns in
  platform backends); the tie-breaker prevents pylint-perfect but
  convention-wrong patches.

## Error Quick Fix

| Error | Fix |
|---|---|
| `GITCODE_TOKEN not found` | `export GITCODE_TOKEN=<token>` |
| `Cannot reach build.mindspore.cn` | network unavailable from this host — escalate to user |
| `No failure comments on PR` | gate may have not run yet — `retest` and re-`diagnose` |
| `Same failure recurred after autofix` | stop and ask the user; the failure may need manual analysis |
| `autogit commit refuses (uncommitted submodule artefact)` | clean the submodule first: `git -C <submodule> clean -fdx` then retry |

## References

- **Workflow detail** — [references/workflow.md](references/workflow.md)
  (step-by-step playbook, 3-question triage example, local pylint
  invocation)
- **Closed-loop guarantee** — [references/closed-loop.md](references/closed-loop.md)
  (closed-loop discipline, ScheduleWakeup cadence + rationale,
  in-memory state tracking, seven terminal states, dual-label
  exit condition)
- **Final report contract** — [references/final-report.md](references/final-report.md)
  (template, per-terminal-state next-step instructions, worked
  example)
- **Failure recipes** — [references/common-failures.md](references/common-failures.md)
- **Flake policy** — [references/flake-policy.md](references/flake-policy.md)
  (3-question triage, Path A random / Path B comment-and-escalate /
  Path C author-authorized temp comment-out, PR-comment template)
- **Root-cause fix policy** — [references/root-cause-fix.md](references/root-cause-fix.md)
  (PR-INDUCED production-not-test rule, two exceptions, trivial vs
  deep complexity split, when to stop and ask for human guidance)
