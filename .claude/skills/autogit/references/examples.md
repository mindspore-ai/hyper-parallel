# AutoGit End-to-End Examples

Real-world scenarios showing complete workflows from start to finish.

---

## Scenario 1: Standard Feature Development

> Goal: Develop a new distributed operator, submit PR, address review feedback, squash before merge.

```bash
# 1. Create feature branch
git checkout -b feat/repeat-interleave

# 2. Develop iteratively — commit after each meaningful chunk
#    (autogit runs lint checks before each commit)
/autogit commit -m "feat(dist_op): add RepeatInterleave forward"
/autogit commit -m "feat(dist_op): add RepeatInterleave backward"
/autogit commit -m "test: add RepeatInterleave unit tests"

# 3. Create PR — auto-generates title and description from diff
/autogit pr --reviewer zhangsan,lisi

# Output:
#   ✅ PR 创建成功!
#   🔗 https://gitcode.com/org/repo/pull/205
#   分支: feat/repeat-interleave
#   Commits: 3

# 4. Reviewer requests changes — fix and append
/autogit pr --to #205
# or for a small typo fix:
/autogit pr --to #205 --amend

# 5. Before merge — squash 4 commits into 1 clean commit
/autogit squash #205

# 6. Check final state
/autogit status #205
```

**Timeline**:

```text
Day 1:  commit → commit → commit → pr          (3 commits, PR #205 created)
Day 2:  (review feedback) → fix → pr --to #205 (4 commits)
Day 3:  squash #205                             (1 commit, ready to merge)
```

---

## Scenario 2: Hotfix on Production Bug

> Goal: Fix a critical bug quickly, skip lint in emergency, create PR immediately.

```bash
# 1. Branch from latest master
git checkout master
git pull upstream master
git checkout -b fix/tensor-shape-crash

# 2. Fix the bug — skip lint for speed (emergency only!)
/autogit commit -m "fix: handle empty tensor shape in reshape op" --no-check

# 3. Create PR immediately
/autogit pr --base master --reviewer zhangsan

# 4. After PR is merged, run lint retroactively
git checkout master
git pull upstream master
/autogit check
```

---

## Scenario 3: Accidental Commit on Master

> Goal: You committed on master by mistake. AutoGit handles it safely.

```bash
# Oops — committed directly on master
/autogit commit -m "feat: add new util function"

# Create PR — autogit detects master, auto-creates pr/<timestamp> branch
/autogit pr

# Output:
#   ⚠️  当前在受保护分支 'master'，将创建新的 PR 分支
#   🌿 创建新分支: pr/20260206_143022
#   📦 备份: backup/20260206_143022
#   🍒 Cherry-pick 1 个 commits...
#   ✅ PR 创建成功!
#   ↩️  已切回 master
```

Master stays clean. Your commit lives on `pr/20260206_143022`.

---

## Scenario 4: Multi-round Review with Conflict Avoidance

> Goal: Reviewer asks for 3 rounds of changes. Upstream master moves fast, causing rebase conflicts.

```bash
# Round 1: Normal append (rebase works fine)
/autogit pr --to #180

# Round 2: Rebase conflicts — use --no-rebase to skip
/autogit pr --to #180 --no-rebase

# Round 3: Small fix — amend into previous commit
/autogit pr --to #180 --amend

# Final: Squash everything before merge
/autogit squash #180 -m "feat(fully_shard): add init support"
```

**Decision tree for --to mode**:

```text
Need to append to PR?
  ├── Normal change     → /autogit pr --to #N           (default, rebase)
  ├── Rebase conflicts  → /autogit pr --to #N --no-rebase
  └── Tiny fix / typo   → /autogit pr --to #N --amend
```

---

## Scenario 5: Update PR Description After Code Changes

> Goal: PR was created early, code changed significantly. Regenerate description.

```bash
# Append more commits
/autogit pr --to #195
/autogit pr --to #195

# Description is now stale — regenerate it
/autogit update #195

# Output:
#   📊 分析代码变更...
#   ✨ 生成 PR 描述...
#   ✅ PR #195 描述已更新!
```

---

## Scenario 6: Check Before Commit

> Goal: Run lint checks without committing, to preview issues first.

```bash
# Make changes, then check only
/autogit check

# Output (example — some checks fail):
#   🔍 运行代码检查...
#   [pylint]
#   my_module.py:42:0: E0602: Undefined variable 'foo'
#   ❌ 检查未通过

# Fix the issue, then commit
/autogit commit -m "fix: remove undefined variable reference"
#   🔍 运行代码检查...
#   ✅ 所有检查通过
#   ✅ 已创建 commit: a1b2c3d4
```

---

## Scenario 7: View PR Status from URL

> Goal: Check a colleague's PR status using URL or number.

```bash
# By number (uses upstream from git remotes)
/autogit status #160

# By full URL
/autogit status https://gitcode.com/org/repo/pull/160

# Output:
#   ╔══════════════════════════════════════════════════════════════════╗
#   ║  PR #160: feat: add new distributed operator
#   ╚══════════════════════════════════════════════════════════════════╝
#   状态: 🟢 开放中
#   作者: zhangsan
#   分支: feat/new-op → master
#   创建: 2026-01-15  更新: 2026-02-01
#   统计: +120 -30 | 5 文件 | 3 commits
#   审核人: lisi, wangwu
#   🔗 https://gitcode.com/org/repo/pull/160
```
