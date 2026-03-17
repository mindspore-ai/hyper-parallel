# /test — Run Test Stage (pylint + lints + pytest)

Delegates to the `autogit` skill's `test` command.

## Usage

```bash
/test
```

## What It Does

Runs the **test stage** (not commit stage):

1. **Lint checks including pylint** on changed files (or all `hyper_parallel/` and `tests/` if no changes)
2. **pytest** on `tests/`

Pylint is executed only in this test stage; the commit stage runs other lints (lizard, docstring, codespell, etc.) but not pylint.

## See Also

- `.claude/skills/autogit/SKILL.md` — full autogit workflow
- `autogit check` — lint only (includes pylint), no pytest
- `autogit commit` — stage, commit-stage lints (no pylint), commit, push
