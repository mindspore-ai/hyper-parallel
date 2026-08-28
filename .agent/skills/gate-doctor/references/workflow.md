# Gate Doctor — Workflow Playbook

Per-subcommand mechanics. The loop shape, cadence, terminal states,
and triage flowchart live elsewhere:

- Diagnose JSON / bypass builds / dual-phase next-action table →
  [diagnose-signals.md](diagnose-signals.md)
- Loop discipline + cadence + terminal states →
  [closed-loop.md](closed-loop.md)
- 3-question triage + Path A/B/C routing → [flake-policy.md](flake-policy.md)
- PR-INDUCED production-fix rule + complexity split →
  [root-cause-fix.md](root-cause-fix.md)
- Final-report template → [final-report.md](final-report.md)

This file covers the cheap mechanical bits: the preconditions every
subcommand needs, the actual `gate_doctor.py` invocation per
subcommand, and how to unstick a gate that won't run.

## Common preconditions

Before any subcommand runs:

1. `GITCODE_TOKEN` is set (`echo ${GITCODE_TOKEN:+set}` prints "set").
2. Current working dir is inside a git repo whose `origin` is the user's
   GitCode fork and `upstream` is the upstream repo.
3. Local branch matches the PR head branch — if not, switch:
   ```bash
   git checkout <pr_head_branch>
   git fetch origin '+refs/heads/<pr_head_branch>:refs/remotes/origin/<pr_head_branch>'
   ```
   The explicit refspec (`+refs/heads/...`) is **required**: a plain
   `git fetch origin <branch>` only updates FETCH_HEAD, leaving the
   remote-tracking ref stale. (autogit has the same gotcha.)

## `retest`

```
user: 触发门禁 #647            # default = hyper-parallel
user: 触发门禁 https://gitcode.com/mindspore/mindspore/pull/92567   # ms via URL
user: 触发门禁 mindspore/mindspore#92567                            # ms short form
```

Steps:

1. `python3 gate_doctor.py retest <ref>` → POSTs `/retest` as a PR comment
   on the repo resolved from `<ref>`. A bare number lands on
   `mindspore/hyper-parallel`; full URL / `owner/repo#N` overrides.
2. Wait ~30s; re-call `diagnose <ref>` and check whether a new pipeline
   number appears in the comments (`micro-compass: The pipeline #NNNN is
   running...`).
3. Report the new pipeline number + link to the user.

Verify: latest comment author is the user; comment body is exactly `/retest`.

## `check-pr`

Same shape as `retest`, but the comment body is `/check-pr`. Use this when
the previous failure was on description / self-checklist rather than the
build pipeline.

## `diagnose` (read-only)

```
user: 看下 #92567 流水线为啥挂了
```

Steps:

1. Pull comments → walk newest-first → find the freshest of these signals:
   - A "Build Result" table line containing `FAILURE`.
   - A `/check-pr` failure block (matches `当前/check-pr未通过`).
2. If pipeline failure → extract Jenkins build URL → fetch `consoleText` →
   parse failure lines per [common-failures.md](common-failures.md)
   patterns.
3. Emit JSON to stdout:
   ```
   {
     "pipeline_number": 3903,
     "pipeline_url": "<jenkins-build-url>",
     "failed_stages": ["Check_Pylint", "Check_DT_Design"],
     "failures": [
       {"stage": "Check_DT_Design", "rule": "DT_Design",
        "file": "", "line": null, "function": "test_pynative_mhc_basic",
        "message": "missing docstring fields: Description:, Expectation:"},
       {"stage": "Check_Pylint", "rule": "W1510",
        "file": "tests/.../test_pynative_hp.py", "line": 118,
        "function": null,
        "message": "'subprocess.run' used without explicitly defining the value for 'check'."}
     ],
     "raw_log_excerpt": "...last ~200 ERROR/FAILURE lines from Jenkins..."
   }
   ```
4. **Show the JSON to the user**. Don't patch yet.

Verify: `pipeline_number` must be non-null OR `check_pr_failure` set.
Otherwise the script returns exit 1 and the LLM should fall back to
`retest`.

## `autofix` per-iteration mechanics

The autofix loop's shape, cadence, and terminal states are in
[closed-loop.md](closed-loop.md); the per-failure routing is in
[flake-policy.md](flake-policy.md); the PR-INDUCED production-fix
rule is in [root-cause-fix.md](root-cause-fix.md). What follows is
only the cheap mechanical bits of one iteration.

### Per-iteration steps (trivial PR-INDUCED fix)

1. **Diagnose:** `gate_doctor.py diagnose <pr>`. Read the JSON.
2. **Recipe lookup:** for each unique stage in `failed_stages`, find
   the recipe in [common-failures.md](common-failures.md). Build an
   ordered patch list of `(file, line, action)`.
3. **Patch:** apply each edit with `Read` + `Edit`. Never rewrite
   code unrelated to the failure.
4. **Local pylint verify:** if any `.py` file changed, run
   ```bash
   curl -fsSL https://tools.mindspore.cn/tools/check/pylint/rules/pylintrc \
        -o /tmp/ms_pylintrc
   pylint --rcfile=/tmp/ms_pylintrc <changed files...>
   ```
   The score must be **10.00/10**. (CI uses the same rcfile + pylint
   3.3.7; install locally with `pip install pylint==3.3.7`.)
5. **Commit + push:**
   ```bash
   python3 .agent/skills/autogit/scripts/autogit.py commit \
       -m "fix(<pkg>): satisfy <Check_X> (<rule>)"
   ```
   The commit message must cite the failing stage and rule so the
   next `diagnose` can confirm it was addressed.
6. **Retest:** `gate_doctor.py retest <pr>`.
7. **Wait + re-diagnose** via ScheduleWakeup on the growing cadence
   in [closed-loop.md](closed-loop.md).

### Anti-patterns

- ❌ Calling `autogit pr --squash` mid-loop. Squash only at the end after
  a green run, when the maintainer asks.
- ❌ Editing files mentioned only in `raw_log_excerpt` but not in
  `failures[]`. The excerpt is for human sanity; the parsed `failures`
  list is the source of truth.
- ❌ Posting `/retest` before pushing the fix commit. The gate reruns
  with the same broken code and fails again.
- ❌ Suppressing pylint locally (`# pylint: disable=...`) when
  [common-failures.md](common-failures.md) recipe asks for a real fix.
  Disable comments are only acceptable for documented platform-specific
  reasons.

## Recovery from a stuck gate

If the gate page shows a pipeline label like `stat/checking` for >30 min and
no Jenkins log appears:

1. `gate_doctor.py retest <pr>` to nudge it.
2. If still stuck, ask the user to escalate via the MindSpore-Assistant
   label or to ping a maintainer. Do not edit code blindly.
