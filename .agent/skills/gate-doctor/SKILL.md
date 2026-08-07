---
name: gate-doctor
description: >
  Drive a red MindSpore-family GitCode PR gate to green: diagnose,
  /check-pr → /retest, triage failures, patch or escalate, loop until
  both pr-check-pass and ci-pipeline-passed. Use for 门禁/CI/retest/
  check-pr/autofix/把 PR 修绿, or any MindSpore GitCode PR gate request.
  Default repo mindspore/hyper-parallel; pass full URL or owner/repo#N
  for cross-repo. Details load from references/ on demand.
---

# Gate Doctor

End-to-end PR gate diagnose + autofix for MindSpore-family GitCode repos.
**This file is the index + hard red lines.** Load the matching
`references/*.md` before acting on that topic — do not improvise policy.

## Tool Requirements

- `GITCODE_TOKEN` (PR-comment write), same as autogit
- `git` with `origin` (fork) + `upstream` (main)
- `python3.8+` with `pylint==3.3.7` (CI parity)
- Working `autogit` skill (autofix commits via it)

## Subcommands

| Intent | Command | Purpose |
|---|---|---|
| 看情况 / 诊断 / bare PR link | `diagnose <pr>` | Read-only Jenkins + labels → JSON |
| 触发门禁 / `/retest` | `retest <pr>` | Post `/retest` |
| 校验描述 / `/check-pr` | `check-pr <pr>` | Post `/check-pr` |
| 修绿 / autofix / 一直跑到通过 | `autofix <pr>` | Closed loop until terminal |

```bash
python3 {skill_dir}/scripts/gate_doctor.py <subcommand> <pr_ref> [options]
```

PR refs: `#647` → `mindspore/hyper-parallel`; full URL or `owner/repo#N`
for others; unknown Jenkins mapping needs `--jenkins-job=<name>`.
No PR id → resolve from current branch via GitCode API, else ask once.

Announce at start: `Running gate-doctor `<subcommand>` …`

## Load-by-phase navigation

| When | Read first |
|---|---|
| Any subcommand mechanics, stuck gate, pylint invoke | [references/workflow.md](references/workflow.md) |
| Interpreting diagnose JSON / bypass / dual labels | [references/diagnose-signals.md](references/diagnose-signals.md) |
| autofix persistence, ScheduleWakeup, 7 terminal states | [references/closed-loop.md](references/closed-loop.md) |
| Failure routing Path A/B/C, 3-question triage, temp-skip | [references/flake-policy.md](references/flake-policy.md) |
| PR-INDUCED production fix vs deep→ask-human | [references/root-cause-fix.md](references/root-cause-fix.md) |
| Before editing for a `Check_*` / `/check-pr` failure | [references/common-failures.md](references/common-failures.md) |
| Every terminal exit (success or escalate) | [references/final-report.md](references/final-report.md) |

**Canonical autofix shape:** `/check-pr` before every `/retest` → triage
via flake-policy → fix via common-failures / root-cause-fix → both
labels for green. Always start a bare PR link with `diagnose`.

## Hard rules (must not violate)

**Loop**

- Trust only `diagnose` Jenkins fields — never invent build numbers.
- Never `/retest` without fresh `pr-check-pass` (else bypass build).
- Poll via ScheduleWakeup 60s → 270s → 1200s → 1800s; no raw `sleep` > 60s.
- Terminal success = **both** `pr-check-pass` and `ci-pipeline-passed`.
- `autofix` returns only on a documented terminal state (see closed-loop).
- Patch ceiling: **2** fix commits per invocation, then `patch-ceiling`.

**Edits**

- No file edits before `diagnose`.
- Never `git push --force`.
- After `.py` edits, local pylint + MindSpore rcfile must be 10.00/10.
- Out-of-diff failures need 3-question triage — not auto-UNRELATED.
- PR-INDUCED → root-cause fix in **production** code (no skip/xfail/
  soften asserts / drop params / change `arg_mark` level) — see
  root-cause-fix.
- Trivial `Check_*` (recipe, ≤5 LoC, one file) → patch; deep Smoke/UT/ST/
  ops → `needs-human-guidance` + final report.
- Every terminal exit emits the final-report template.

## Safety & restrictions

- `diagnose` read-only; `retest`/`check-pr` only post one comment.
- `autofix` stays inside the PR diff; never edit `upstream/master`;
  touch PR template only if that is the failing check.
- No AI attribution trailers in commits (`autogit` hook rejects them).
- Do not touch submodules (`mindformers` / `metadef` / …) unless the
  failure is inside tracked submodule files.
- If `.agent/rules/code-style.md` exists, load it before patching
  (tie-break vs pylint).

## Error quick fix

| Error | Fix |
|---|---|
| `GITCODE_TOKEN not found` | `export GITCODE_TOKEN=<token>` |
| `Cannot reach build.mindspore.cn` | Escalate — host has no network path |
| `No failure comments on PR` | `retest` then re-`diagnose` |
| Same failure after autofix | Stop; ask user (may need manual root cause) |
| autogit refuses (submodule dirty) | `git -C <submodule> clean -fdx` then retry |
