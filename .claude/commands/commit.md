# /commit — Stage, Check, Commit, Push

Delegates to the `autogit` skill's `commit` command.

## Usage

```bash
/commit                     # auto-generate message
/commit -m "feat: add X"    # explicit message
/commit --no-check          # skip lint checks
```

## What It Does

Invokes `autogit commit` with any provided arguments. See `.claude/skills/autogit/SKILL.md` for full workflow details.

The workflow covers: stage changes -> lint check -> generate/confirm commit message -> commit -> push to origin.
