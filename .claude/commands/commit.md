# /commit — Stage, Check, Commit, Push

Delegates to the `autogit` skill's `commit` command.
The workflow must load `.claude/rules/code-style.md` first and must not commit code that still violates it.

## Usage

```bash
/commit                     # auto-generate message
/commit -m "feat: add X"    # explicit message
/commit --no-check          # skip lint checks
```

## What It Does

Invokes `autogit commit` with any provided arguments. See `.claude/skills/autogit/SKILL.md` for full workflow details.

The workflow covers: load `code-style.md` -> stage changes -> auto-fix style issues -> generate/confirm commit message -> commit -> auto-fix pre-commit issues -> push to origin.
