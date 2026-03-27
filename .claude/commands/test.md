# /test — Run Test Stage (pytest)

Delegates to the `autogit` skill's `test` command.
The workflow should be used for repository test execution after lint checks are already handled separately.

## Usage

```bash
/test
```

## What It Does

Runs the **test stage**:

1. **pytest** on `tests/ut`

Lint checks are handled separately through `autogit check`, `pre-commit`, or other repository workflows.

## See Also

- `.claude/skills/autogit/SKILL.md` — full autogit workflow
- `autogit check` — lint only, includes pylint and markdownlint
- `autogit commit` — stage, commit-stage lints, commit, push
