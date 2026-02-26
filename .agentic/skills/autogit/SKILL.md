---
name: autogit
description: >
  GitCode fork workflow automation (commit, PR, status, squash).
  Use when committing and pushing code to GitCode, creating or appending to PRs,
  checking PR status, squashing commits, or running pre-commit code checks.
  Supports origin (fork) + upstream (main repo) pattern.
---

# AutoGit

## Overview

Safe, incremental Git workflow for GitCode fork repos.
Principle: **never overwrite without explicit request; always back up before danger**.

Announce at start of every invocation:
> Running AutoGit `<command>` …

## Execution

Run from project root:

```bash
python3 {skill_dir}/scripts/autogit.py <command> [options]
```

`{skill_dir}` is the directory containing this SKILL.md.

## Quick Reference

| Command | Purpose | Example |
|---------|---------|---------|
| `commit` | Stage, lint-check, commit, push to origin | `commit -m "feat: add X"` |
| `commit --no-check` | Commit without lint checks | `commit --no-check` |
| `check` | Run lint checks only (no commit) | `check` |
| `pr` | Create PR to upstream | `pr --reviewer zhangsan` |
| `pr --to #N` | Append commits to existing PR | `pr --to #160 --amend` |
| `status #N` | Show PR status (read-only) | `status #160` |
| `update #N` | Regenerate PR description | `update #160` |
| `squash #N` | Squash PR commits into one | `squash #160 -m "msg"` |

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

## Safety Guarantees

- **No silent overwrites** — push conflicts prompt user, never force-push implicitly.
- **Backup before danger** — squash/rebase create `backup/<timestamp>` branches.
- **Uncommitted changes block PR** — must commit first.
- **Rebase failures auto-abort** — restores original state on conflict.
- **Stash on branch switch** — auto-stash before switching, restore after.
- **Lint gate on commit** — default pre-commit checks (pylint, lizard, dt_design, codespell, markdownlint); skip with `--no-check`.

## Red Flags

STOP and ask the user if you are about to:

- Force-push to `master`, `main`, or any shared branch.
- Delete a remote branch that is not your own PR branch.
- Run `squash` or `--amend` on a PR you did not create.
- Skip lint checks without explicit user request (`--no-check`).
- Modify upstream remote or change its URL.

### Rationalizations to Reject

- "It will be fine, I'll just force-push quickly" — NO. Confirm with user.
- "Nobody else is using this branch" — You don't know that. Ask first.
- "Lint checks are too slow, I'll skip them" — Only skip if user says `--no-check`.

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
- **End-to-end examples**: [references/examples.md](references/examples.md) — 7 real-world scenarios with expected output.
