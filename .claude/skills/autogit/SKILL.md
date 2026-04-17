---
name: autogit
description: GitCode fork workflow automation. Use this skill whenever the user wants to commit code, push, create or append to a Pull Request, view PR status, squash commits, regenerate a PR description, or run lint checks against a GitCode `origin` (fork) + `upstream` repository. Supports both Chinese and English natural-language triggers (e.g. "帮我提交", "create PR", "看下 PR 状态") and slash-command shortcuts (`/commit`, `/create-pr`, etc.). The full trigger → subcommand mapping lives in the "When to Activate" section.
---

# AutoGit

Safe, incremental Git workflow for GitCode fork repos. Origin (fork) +
upstream (main repo) pattern. Principle: **never overwrite without explicit
request; always back up before danger**.

Before generating, modifying, reviewing, or committing code, always load and
follow `.claude/rules/code-style.md`.

Announce at start of every invocation:
> Running AutoGit `<command>` …

## When to Activate

- "帮我提交"、"提交代码"、"push" → `commit`
- "创建 PR"、"create PR"、"提个 PR" → `pr`
- "看下 PR 状态"、"PR status" → `status`
- "压缩 commit"、"squash" → `squash`
- "跑下 lint"、"check" → `check`
- `/commit`, `/create-pr`, `/test`, `/code-review`, `/gen-commit-msg`

## Tool Requirements

- **Python 3.8+** with `yaml` package
- **git** with `origin` (fork) and `upstream` (main repo) remotes configured
- **GITCODE_TOKEN** env var or git credential store for API access
- **pylint**, **markdownlint** (auto-installed on first `check` if missing)

## Execution

Run from project root:

```bash
python3 {skill_dir}/scripts/autogit.py <command> [options]
```

`{skill_dir}` is the directory containing this SKILL.md.

## Quick Reference

| Command | Purpose                                                                 | Example |
|---------|-------------------------------------------------------------------------|---------|
| `commit` | Stage, preview commit msg, commit, push (lint runs via repo's pre-commit hook) | `commit -m "feat: add X"` |
| `check` | Run lint checks independently (pylint + markdownlint); does **not** commit | `check` |
| `test` | **Test stage**: pytest tests/ut (full)                                  | `test` |
| `pr` | Ask UT/ST one-at-a-time (tri-state), then create PR to upstream | `pr --reviewer zhangsan` |
| `pr --ut {skip,changed,full}` | UT gate scope (default: ask; ask-default: `changed`) | `pr --ut changed` |
| `pr --st {skip,changed,full}` | ST gate scope (default: ask; ask-default: `skip`)   | `pr --st full` |
| `pr --analyze-only` | Output structured JSON analysis for LLM PR description (forces all gates off) | `pr --analyze-only` |
| `pr --title --body` | Create PR with explicit title/body (from LLM preview) | `pr --title "feat: X" --body "..."` |
| `pr --squash` | Squash commits into one before creating PR | `pr --squash` |
| `pr --to #N` | Ask UT (no ST), then append commits to existing PR                     | `pr --to #160 --amend --ut changed` |
| `status #N` | Show PR status (read-only)                                              | `status #160` |
| `update #N` | Regenerate PR description — previews title/body before writing (same gate as `commit` / `pr`) | `update #160` |
| `update #N --title --body` | Update PR with LLM-approved title/body (skips preview) | `update #160 --title "..." --body "..."` |
| `squash #N` | Squash PR commits into one                                              | `squash #160 -m "msg"` |

For full parameter details run `python3 {skill_dir}/scripts/autogit.py <command> --help`.

## Core Workflows

### Standard Development (recommended)

```text
git checkout -b feat/my-feature
  edit → /autogit commit -m "feat: add A"
  edit → /autogit commit -m "feat: add B"
  done → /autogit pr
review → /autogit pr --to #N          # address feedback
merge  → /autogit squash #N           # optional cleanup
```

### Hotfix

```text
git checkout -b fix/urgent-bug
  fix  → /autogit commit -m "fix: urgent"
  done → /autogit pr
```

Always work on a feature branch. If you commit on master, AutoGit auto-creates a `pr/<timestamp>` branch to protect master.

## Pipelines

### Lint ownership: pre-commit git hook

Lint (pylint + markdownlint) is run by the project's **pre-commit git hook**, installed via `bash scripts/pre-commit/install.sh`. The hook fires on every `git commit` automatically — autogit does not duplicate it. If the hook is not installed, `autogit commit` prints a one-line install reminder before committing (non-blocking).

### Pipeline A — `autogit commit` (per-commit, fast)

```
preview commit-msg → git commit (pre-commit hook auto-runs lint) → push
```

The only interactive step is the commit-message preview. UT/ST are intentionally not gated here — commits should be cheap; heavy regression checks belong to PR time.

### Pipeline B — `autogit pr` (PR-time, regression)

```
UT(scope?) → ST(scope?) → PR title+body preview → create PR
```

- **UT gate** — three states: `skip` / `changed` / `full`.
  - `changed` (default) — only test files appearing in this PR's diff (fast; covers exactly what the PR introduces).
  - `full` — entire `tests/ut` suite (regression coverage).
  - If `changed` finds no UT test files, the gate prints a notice and skips.
- **ST gate** — same three states. Default `skip` because ST is heavy and most envs are single-card.
  - `changed` — only ST test files in the PR diff.
  - `full` — `tests/torch/st` + `tests/mindspore/st`.
  - Missing dirs / launcher / single-card surface as a hard error so the user can re-run with `--st skip`.
- **Content preview last** — what gets written must match what just passed the gates.

`pr --to N` (append) only asks UT (no ST); for the append flow `changed` scope inspects the local working tree (staged + unstaged) since the gate runs before any branch operations.

## Interactive Enforcement (script-level)

Each command refuses to proceed without an explicit user choice. Two enforcement classes, both backed by `sys.stdin.isatty()` detection.

### Class 1 — Gate choices (skip / changed / full)

`autogit pr` resolves two PR-time gates (UT, ST). Each gate is tri-state. `autogit commit` has **no** quality gates — lint runs via the pre-commit hook, UT/ST belong to PR time.

- **tty (terminal user)** — gates asked **one at a time** (`一步一步`). Each prompt accepts `c` (changed) / `f` (full) / `n` (skip) / Enter (default). After one finishes the next is asked.
- **non-tty (Claude agent)** — one `AutoGitError` listing every undecided gate with the exact flag the AI must pass. AI asks the user one-at-a-time in chat, then re-invokes with all flags in a single call.

Flags: `--ut {skip,changed,full}`, `--st {skip,changed,full}`. Absence = ask. Explicit value = honour as-is.

Default policy when user just hits Enter: UT=`changed`, ST=`skip`.

### Class 2 — Content previews (commit msg / PR title / PR body)

`autogit commit`, `autogit pr`, and `autogit update` refuse to write content the user has not seen.

- **tty** — render auto-generated content in a bordered block, accept `Enter`/`y` (use as-is), `e` (abort, edit in chat with user), `c` (cancel).
- **non-tty** — error with the exact flag the AI must pass after user approval. Examples:

  ```
  commit message required (non-interactive context).

  This is a single-line text. Generate it from the current diff/commits,
  show it to the user for approval, then re-invoke with:
    -m "<approved message>"

  NEVER pass content the user has not explicitly seen and approved.
  ```

Why this matters: AI may generate plausible-looking content and silently pass it via `-m` / `--title` / `--body`. The script cannot read the AI's mind, but it can refuse to act without explicit content. **The non-tty error is the only thing that forces "show user first" to actually happen.**

## AI Orchestration

The user types one intent. AI calls **one** autogit subcommand, then handles
per-step gate questions in chat. Full step-by-step playbook + ANTI-PATTERNS
in [references/orchestration.md](references/orchestration.md).

**Quick rules:**

- Always **show the user** the generated commit message / PR title / PR body
  before re-invoking with `-m / --title / --body`.
- Ask UT / ST one at a time; never pass `--ut skip --st skip` because
  "the script asks too much" — only skip if the user explicitly chose to.
- On stage failure: read the error, fix in code, re-run the same stage; if
  the same root cause repeats, stop and ask the user.

## Safety Guarantees

- **No silent overwrites** — push conflicts prompt user, never force-push implicitly.
- **Backup before danger** — squash/rebase create `backup/<timestamp>` branches.
- **Uncommitted changes block PR** — must commit first.
- **Rebase failures auto-abort** — restores original state on conflict.
- **Stash on branch switch** — auto-stash before switching, restore after.
- **Lint via pre-commit hook** — `autogit commit` does not lint; the project's pre-commit git hook (installed via `bash scripts/pre-commit/install.sh`) runs pylint+markdownlint on every `git commit`. Missing hook surfaces as a non-blocking warning.
- **Dedicated lint stage** — `autogit check` runs lint independently when you want a no-commit precheck.
- **Dedicated test stage** — `autogit test` runs `pytest tests/ut` (full).
- **PR test gates** — `autogit pr` asks UT and ST one at a time, each accepts `skip` / `changed` / `full`. Defaults: UT=`changed`, ST=`skip`. Pass `--ut/--st` explicitly to skip the prompt.

## Red Flags

STOP and ask the user if you are about to:

- Force-push to `master`, `main`, or any shared branch.
- Delete a remote branch that is not your own PR branch.
- Run `squash` or `--amend` on a PR you did not create.
- Skip UT/ST gates without explicit user request (`--ut skip` / `--st skip`).
- Modify upstream remote or change its URL.

## Important Restrictions (Must Follow)

DO NOT add "Co-Authored-By: Claude ..." in commit messages.
If `.claude/rules/code-style.md` conflicts with a user request, explain the conflict and provide a compliant alternative instead of emitting non-compliant code.

## Error Quick Fix

| Error | Fix |
|-------|-----|
| Token not found | `export GITCODE_TOKEN=<token>` |
| No upstream remote | `git remote add upstream <URL>` |
| Uncommitted changes | Run `/autogit commit` first |
| Push rejected | `git pull --rebase origin <branch>` |
| Rebase conflict | Use `--no-rebase` or resolve manually |
| Cherry-pick failed | Use a feature branch instead |

## References

- **Command details**: [references/commands.md](references/commands.md) — full parameter docs, execution flows, setup instructions.
- **AI orchestration & anti-patterns**: [references/orchestration.md](references/orchestration.md) — step-by-step playbook for `commit` / `pr` / `pr --to`, failure self-loop, PR description rules, ANTI-PATTERNS list.
- **End-to-end examples**: [references/examples.md](references/examples.md) — 8 real-world scenarios with expected output.
