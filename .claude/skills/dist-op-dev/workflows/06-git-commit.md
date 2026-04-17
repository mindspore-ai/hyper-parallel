# Workflow 6: Git Commit and PR Creation

## Goal

Create a feature branch, commit all changes via autogit, push to origin, and create a PR to upstream if needed.

## Input

- **All modified code**: Python implementation, YAML registration, UT, and ST files
- **Operator name**: Used for branch naming and commit messages

## Output

- Feature branch pushed to origin
- PR created to upstream (if needed)

---

## Step 1: Create Feature Branch

```bash
git checkout master && git pull --rebase origin master
git checkout -b feat/{OpName}-distributed-support
```

## Step 2: Commit

```bash
/autogit commit -m "feat(shard): add {OpName} operator distributed support"
```

Commit message body template for distributed operators:

```
feat(shard): add {OpName} operator distributed support

- Add {ClassName} class in {file_name}.py
- Register {OpName} in {yaml_file}.yaml
- Add UT cases for {OpName}
- Add ST cases for {OpName} (8-card)
```

## Step 3: Create PR

```bash
/autogit pr
```

## Step 4: Verify

- [ ] All files included (implementation, YAML, UT, ST)
- [ ] Branch pushed to origin
- [ ] PR created against `upstream/master`

---

For commit conventions, lint checks, PR description format, and troubleshooting, see the **autogit** skill (`.claude/skills/autogit/SKILL.md`).
