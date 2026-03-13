# /create-pr — Create Pull Request to Upstream

Delegates to the `autogit` skill's `pr` command.

## Usage

```bash
/create-pr                      # create new PR to upstream/master
/create-pr --to #N              # append commits to existing PR #N
/create-pr --squash             # squash commits before creating PR
/create-pr --reviewer zhangsan  # assign reviewer
```

## What It Does

Invokes `autogit pr` with any provided arguments. See `.claude/skills/autogit/SKILL.md` for full workflow details.

The workflow covers: verify prerequisites -> sync with upstream -> push to origin -> generate PR content -> create PR via GitCode API.
