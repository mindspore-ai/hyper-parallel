# Autogit — Pipelines & Interactive Enforcement

Moved from SKILL.md entry. Command flags: [commands.md](commands.md).
AI step playbook: [orchestration.md](orchestration.md).

## Lint ownership: pre-commit git hook

Lint (pylint + markdownlint) runs via the project's **pre-commit git hook**
(`bash scripts/pre-commit/install.sh`). Autogit does not duplicate it. Missing
hook → non-blocking reminder on `autogit commit`.

Reject AI attribution on every commit:

```bash
cp .agent/hooks/commit-msg .git/hooks/commit-msg && chmod +x .git/hooks/commit-msg
```

## Pipeline A — `autogit commit` (per-commit, fast)

```
preview commit-msg → AGENTS catalog check → git commit (pre-commit hook lint) → push
```

No UT/ST gates here — keep commits cheap; regression at PR time.

## Pipeline B — `autogit pr` (PR-time, regression)

```
UT(scope?) → ST(scope?) → PR title+body preview → create PR
```

- **UT / ST gates:** `skip` / `changed` / `full`. Defaults when user hits Enter: UT=`changed`, ST=`skip`.
- `changed` = test files in this PR's diff; `full` = whole suite (`tests/ut` or torch+mindspore ST).
- ST missing dirs / launcher / single-card → hard error (re-run with `--st skip`).
- `pr --to N` (append): UT only; `changed` inspects local working tree before branch ops.
- Content preview last — written content must match what passed gates.

## Interactive enforcement (script-level)

Both classes use `sys.stdin.isatty()`.

### Class 1 — Gate choices

- **tty:** ask UT then ST one-at-a-time (`c`/`f`/`n`/Enter).
- **non-tty (agent):** `AutoGitError` listing undecided gates + exact flags. Agent asks user in chat, re-invokes once with all flags.

Flags: `--ut {skip,changed,full}`, `--st {skip,changed,full}`.

### Class 2 — Content previews

`commit` / `pr` / `update` refuse content the user has not seen.

- **tty:** bordered preview; Enter/`y` accept, `e` abort to edit, `c` cancel.
- **non-tty:** error naming the flag (`-m` / `--title` / `--body`) after user approval.

**Never** pass `-m`/`--title`/`--body` the user has not explicitly approved.
