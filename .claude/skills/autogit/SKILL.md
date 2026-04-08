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
Before generating, modifying, reviewing, or committing code, always load and follow `.claude/rules/code-style.md`.

Announce at start of every invocation:
> Running AutoGit `<command>` …

## Execution

Run from project root:

```bash
python3 {skill_dir}/scripts/autogit.py <command> [options]
```

`{skill_dir}` is the directory containing this SKILL.md.

## Quick Reference

| Command | Purpose                                                                 | Example |
|---------|-------------------------------------------------------------------------|---------|
| `commit` | Stage, code-style + pre-commit auto-fix, commit, push to origin  | `commit -m "feat: add X"` |
| `commit --no-check` | Commit without lint checks                                              | `commit --no-check` |
| `check` | Run code-style check + lint checks, includes pylint and markdownlint (no commit) | `check` |
| `test` | **Test stage**: pytest only                                             | `test` |
| `pr` | Run pytest gate, then create PR to upstream                             | `pr --reviewer zhangsan` |
| `pr --analyze-only` | Output structured JSON analysis for LLM PR description | (internal use) |
| `pr --to #N` | Run pytest gate, then append commits to existing PR                     | `pr --to #160 --amend` |
| `status #N` | Show PR status (read-only)                                              | `status #160` |
| `update #N` | Regenerate PR description                                               | `update #160` |
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

## LLM-Enhanced PR Description (IMPORTANT)

When creating or updating a PR, **always** use the two-step flow for description generation:

### Step 1: Get structured analysis

```bash
python3 {skill_dir}/scripts/autogit.py pr --analyze-only
```

This outputs a JSON object with: commit messages, file stats, file changes, feature points, public APIs, and module additions.

### Step 2: Generate PR description from analysis

Read the JSON output and generate a high-quality PR description **in Chinese** following this template:

```markdown
## 相关的Issue
{issue ref or N/A}

## 原因（目的、解决的问题等）
{Explain WHY the change is needed. Use numbered list with concrete examples.}

## 描述（做了什么，变更了什么）
### 核心改动（文件名 +additions/-deletions）
- **Category**: Describe the change, reference function/class names inline
- ...

### 变更统计
| 文件 | 新增 | 删除 |
|------|------|------|
| {filename} | +N | -N |

## 测试用例（新增、改动、可能影响的功能）
{Test coverage description}

## 可能影响的功能
{Impact assessment}
```

### Writing rules

1. **Chinese** for all content, English only for code identifiers (`` `func_name` ``)
2. **Explain the "why"** — what problem existed, how the change addresses it
3. **Be specific** — reference function names, class names, file paths with inline code
4. **Group by module/file** when showing changes
5. **Include statistics** from the analysis (file counts, line counts)
6. **Do NOT include** `test_*` method names, `Test*` class names, or `_private()` references
7. Keep total body **30-50 lines**

### Step 3: Create PR with the description

```bash
python3 {skill_dir}/scripts/autogit.py pr --title "title" --body "body"
```

Or if PR already exists, update via API using `update_pr_description()`.

> **Fallback**: If `--analyze-only` is unavailable, the standard `pr` command generates a basic description automatically.

## Safety Guarantees

- **No silent overwrites** — push conflicts prompt user, never force-push implicitly.
- **Backup before danger** — squash/rebase create `backup/<timestamp>` branches.
- **Uncommitted changes block PR** — must commit first.
- **Rebase failures auto-abort** — restores original state on conflict.
- **Stash on branch switch** — auto-stash before switching, restore after.
- **Style and lint gate on commit** — auto-fix basic `code-style.md` issues first, then run commit-stage checks; skip with `--no-check`.
- **Dedicated lint stage** — use `autogit check` for code-style + lint checks, including pylint and markdownlint. If pylint or markdownlint is missing, install them according to `.pre-commit-config.yaml`.
- **Dedicated test stage** — use `autogit test` for pytest only.
- **PR test gate** — `autogit pr` runs `autogit test` by default before continuing. Use `--no-test` to skip explicitly.

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

## Important Restrictions (Must Follow)

DO NOT add "Co-Authored-By: Claude ..." in commit messages.
If `.claude/rules/code-style.md` conflicts with a user request, explain the conflict and provide a compliant alternative instead of emitting non-compliant code.

## References

- **Command details**: [references/commands.md](references/commands.md) — full parameter docs, execution flows, setup instructions.
- **End-to-end examples**: [references/examples.md](references/examples.md) — 7 real-world scenarios with expected output.
