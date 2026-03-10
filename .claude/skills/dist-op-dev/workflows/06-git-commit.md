# Workflow 6: Git Commit and PR Creation

## Goal

Create feature branch, call **autogit SKILL** to complete code staging, lint check, commit, push, and create PR if needed.

## Input

- **All Modified Code**: Python implementation, YAML registration, UT/ST tests
- **Operator Name**: Operator name from Workflow 1

## Output

- **Feature Branch**: `feat/{OpName}-distributed-support`
- **Git Commit**: Independent commit complying with standards
- **Pushed to Remote**: Code pushed to origin
- **PR Created (optional)**: If needed, PR created to upstream

---

## Step 1: Create Feature Branch

> **Important**: Always develop on feature branch. Avoid committing directly on master/main branch.

### Check Current Branch

```bash
# View current branch
git branch --show-current
```

**If currently on master/main branch**, must create new branch:

```bash
# Create and switch to feature branch
git checkout -b feat/{OpName}-distributed-support
```

### Branch Naming Convention

| Branch Type | Naming Format | Example |
|-------------|---------------|---------|
| **New Feature** | `feat/{OpName}-distributed-support` | `feat/MatMul-distributed-support` |
| **Bug Fix** | `fix/{OpName}-{issue}` | `fix/Add-broadcast-error` |
| **Optimization** | `optimize/{OpName}-{desc}` | `optimize/MatMul-cache` |

**Naming Rules**:
- Use MindSpore naming for operators (PascalCase)
- Multiple words connected with hyphens
- Description should be concise and clear

---

## Step 2: Pre-commit Verification

### 2.1 Confirm Modified Content

```bash
# View modification status
git status
```

**Confirm Complete Modification Content**:
- ✅ Python implementation file (`hyper_parallel/core/shard/ops/parallel_*.py`)
- ✅ YAML registration file (`hyper_parallel/core/shard/ops/yaml/*.yaml`)
- ✅ UT test file (`tests/mindspore/ut/parallel_ops_infer/*.py`, if needed)
- ✅ ST test file (`tests/mindspore/st/shard/**`, `tests/torch/shard/ops/*.py`)

### 2.2 Run Test Verification

```bash
# Run UT tests
pytest tests/mindspore/ut/parallel_ops_infer/ -v

# Run ST tests (8-card environment)
pytest tests/mindspore/st/shard/test_ops_*.py -v
```

**Test Pass Criteria**:
- All UT tests pass
- All ST tests pass (8-card environment)

---

## Step 3: Call autogit commit

### 3.1 Commit Message Template

Generate commit message using the following template based on operator name and development content:

```
feat(shard): add {OpName} operator distributed support

- Add {ClassName} class in {file_name}.py
- Register {OpName} in {yaml_file}.yaml
- Add UT cases covering DP/MP and broadcast scenarios
- Add ST cases for 8-card verification
```

**Template Variable Description**:

| Variable | Description | Example |
|----------|-------------|---------|
| `{OpName}` | Operator name | `Greater`, `MatMul` |
| `{ClassName}` | Distributed operator class name | `GreaterDistributedOp`, `MatMulDistributedOp` |
| `{file_name}` | Python file name | `parallel_elementwise`, `parallel_matmul` |
| `{yaml_file}` | YAML file name | `element_wise_ops_with_shape`, `matmul_ops` |

### 3.2 Execute Commit

**Standard commit (with lint check):**

```bash
/autogit commit -m "feat(shard): add Greater operator distributed support

- Add GreaterDistributedOp class in parallel_elementwise.py
- Register Greater in element_wise_ops_with_shape.yaml
- Add UT cases covering DP/MP and broadcast scenarios
- Add ST cases for 8-card verification"
```

**autogit commit execution flow:**
```
Check changes → git add -A → lint check(pylint/lizard/codespell/markdownlint)
  → git commit → git push origin <branch>
```

**Skip lint check (emergency only):**

```bash
/autogit commit --no-check -m "feat(shard): add Greater operator distributed support"
```

> ⚠️ Only use `--no-check` in emergency situations, normal development should keep lint check enabled.

### 3.3 Run lint check only (optional)

If you only want to check code standards without committing:

```bash
/autogit check
```

---

## Step 4: Confirm Push Status

### 4.1 Check Remote Status

autogit commit has automatically executed `git push origin <branch>`, confirm push status:

```bash
# View current branch
git branch --show-current

# View remote branch status
git branch -vv

# View recent commits
git log -1 --oneline
```

**Success Indicators**:
- Current branch shows `[origin/<branch>]` indicating remote branch is tracked
- No error messages

---

## Step 5: Create Pull Request (optional)

If you need to create a PR to merge into upstream main repository, call autogit pr:

### Standard PR Creation

```bash
/autogit pr
```

**autogit pr execution flow:**
```
Check environment(token, remotes, uncommitted changes)
  → Determine branch type(feature branch / master)
  → Push to origin(prompt on conflict, no force)
  → Analyze diff, auto-generate PR description
  → Call API to create PR, output URL
```

### Specify Target Branch

```bash
/autogit pr --base develop
```

### Specify Reviewer

```bash
/autogit pr --reviewer zhangsan,lisi
```

### Create PR with Squash

```bash
/autogit pr --squash
```

---

## Step 6: Failure Handling

### 6.1 Lint Check Failed

**Common Errors and Fixes**:

| Error Type | Error Example | Fix Method |
|------------|---------------|------------|
| **Undefined variable** | `E0602: Undefined variable 'layout'` | Check variable name and imports |
| **Missing docstring** | `C0116: Missing function docstring` | Add function docstring |
| **Function too long** | `function too long (111 > 100)` | Split into multiple helper methods |
| **CCN too high** | `CCN too high (23 > 19)` | Reduce nesting level, use early return |

**Handling Steps**:
1. Locate problem code based on error message
2. Fix the issue
3. Re-execute `/autogit commit`

### 6.2 Push Conflict

**Error Message**: `Push rejected` or `Updates were rejected`

**Handling Steps**:
```bash
# Pull remote updates
git pull --rebase origin <branch>

# If there are conflicts, resolve them then
git rebase --continue

# Push again
git push origin <branch>
```

### 6.3 Need to Modify Commit Message

If commit is created but needs message modification:

```bash
# Modify the most recent commit message
git commit --amend -m "feat(shard): add Greater operator distributed support

- Add GreaterDistributedOp class in parallel_elementwise.py
- Register Greater in element_wise_ops_with_shape.yaml"

# Force push (for already pushed commits)
git push origin <branch> --force-with-lease
```

---

## Step 7: Follow-up Operations (optional)

### Append Commits to Existing PR

If PR already exists and needs to append new commits:

```bash
# Append new commit (rebase by default)
/autogit pr --to #160

# Merge to last commit
/autogit pr --to #160 --amend

# Skip rebase and append directly
/autogit pr --to #160 --no-rebase
```

### View PR Status

```bash
/autogit status #160
```

Shows: status (open/merged/closed/draft), author, branch, timestamp, statistics, reviewers, URL.

### Update PR Description

```bash
/autogit update #160
```

Re-analyze diff and update PR description.

### Squash PR Commits

```bash
/autogit squash #160 -m "feat(shard): add Greater operator support"
```

Squash multiple commits in PR into one.

---

## Commit Checklist

Before calling autogit commit, ensure:

- [ ] **Branch Correct**
  - [ ] Currently on feature branch (not master/main)
  - [ ] Branch naming follows convention

- [ ] **Code Completeness**
  - [ ] Python implementation file completed
  - [ ] YAML registration file modified
  - [ ] UT test file written (if needed)
  - [ ] ST test file written

- [ ] **Tests Passed**
  - [ ] All UT tests passed
  - [ ] All ST tests passed (8-card environment)

- [ ] **Commit Message**
  - [ ] Type correct (feat/fix/refactor, etc.)
  - [ ] Scope appropriate (shard/layout/dtensor, etc.)
  - [ ] Subject includes operator name
  - [ ] Body lists main changes

---

## autogit Safety Guarantees

autogit provides the following safety guarantees:

| Safety Measure | Description |
|----------------|-------------|
| **No silent overwrites** | Push conflicts prompt user, never force-push implicitly |
| **Branch protection** | Auto-creates new branch when committing on master/main |
| **Lint gate** | Runs lint check by default, ensures code quality |
| **Uncommitted changes block PR** | Must commit first before creating PR |
| **Rebase failure auto-abort** | Restore original state on conflict |
| **Branch switch auto-stash** | Auto-stash before switching, restore after |

---

## Completion Markers

When the following conditions are met, **Workflow 6 is complete**:

```
✅ Feature branch created (feat/{OpName}-distributed-support)
✅ Called /autogit commit to complete commit
✅ Lint check passed (or user explicitly used --no-check)
✅ Commit message follows standard and includes operator name
✅ Code pushed to origin
✅ PR created (if needed)
```

---

## Overall Flow Review

After completing Workflow 6, HyperParallel distributed operator development process is complete:

```
✅ Step 1: Operator Analysis → Analysis report generated
✅ Step 2: Python Implementation → Implementation class created
✅ Step 3: YAML Registration → Operator registered
✅ Step 4: Unit Testing → UT tests passed
✅ Step 5: Integration Testing → ST tests passed
✅ Step 6: Git Commit & PR → Commit created, pushed, PR created (if needed)
```

**Congratulations! Distributed operator development complete.** 🎉

---

## Follow-up Work

If optimization or extension is needed:

1. **Performance Optimization**
   - Optimize `infer_layout` cache hit rate
   - Reduce communication overhead
   - Improve parallel efficiency

2. **Feature Extension**
   - Support more parallel strategies (CP, EP, etc.)
   - Support new broadcast scenarios
   - Add Partial state support

3. **Test Improvement**
   - Add more edge case tests
   - Improve code coverage
   - Add performance tests

---

## autogit Common Issues

| Error | Fix Method |
|-------|------------|
| Token not found | `export GITCODE_TOKEN=<token>` |
| No upstream remote | `git remote add upstream <URL>` |
| Uncommitted changes | Run `/autogit commit` first |
| Push rejected | `git pull --rebase origin <branch>` |
| Rebase conflict | Use `--no-rebase` or resolve manually |
