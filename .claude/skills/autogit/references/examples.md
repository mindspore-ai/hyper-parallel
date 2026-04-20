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

> Goal: Fix a critical bug quickly, create PR immediately.

```bash
# 1. Branch from latest master
git checkout master
git pull upstream master
git checkout -b fix/tensor-shape-crash

# 2. Fix the bug (lint runs via pre-commit hook automatically)
/autogit commit -m "fix: handle empty tensor shape in reshape op"

# 3. Create PR immediately (UT/ST defaults: changed/skip; user accepts via prompt)
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

---

## Scenario 8: AI Agent — End-to-end commit → PR with tri-state UT/ST

> Goal: Show the full chat-driven flow when Claude Code (or any agent)
> drives `autogit` from "implement a feature" to "PR open and described".
> Demonstrates: per-step UT/ST prompts, content previews, non-tty errors as
> teaching messages.

```text
USER: 我新加了 core/foo.py + tests/torch/ut/test_foo.py，帮我提交并开 PR

AGENT: I'll commit then open the PR. First the commit.
       Generating commit message from `git diff --cached` …

       ┌────────────────── Proposed commit message ──────────────────┐
       │ feat(core): add foo distributed op                          │
       │                                                              │
       │ Adds Foo op covering case X; UT in tests/torch/ut/test_foo  │
       └──────────────────────────────────────────────────────────────┘
       OK / edit / cancel?

USER: OK

AGENT: $ autogit commit -m "feat(core): add foo distributed op …"
       [WARN] pre-commit hook not installed — this commit will skip lint.
              Install: bash scripts/pre-commit/install.sh
       Created commit a1b2c3d4
       Pushed to origin/feat/foo

       Now opening PR. Running gates one at a time.

       UT? c=changed (本 PR 涉及的 UT 测试) / f=full (全量 tests/ut) / n=skip
           default c

USER: c

AGENT: ST? c=changed / f=full / n=skip — default n (需多卡环境)

USER: n

AGENT: $ autogit pr --analyze-only --ut skip --st skip   # for content gen
       [analysis JSON returned]

       Generating PR title + body from diff + .gitcode template …

       ┌─────────────────────── PR title ────────────────────────────┐
       │ feat(core): 新增 Foo 分布式算子（覆盖场景 X）               │
       └──────────────────────────────────────────────────────────────┘

       ┌─────────────────────── PR body (excerpt) ───────────────────┐
       │ /kind feature                                                │
       │                                                              │
       │ **What does this PR do**                                     │
       │ 新增 Foo 算子 …                                              │
       │ ...                                                          │
       └──────────────────────────────────────────────────────────────┘
       OK / edit / cancel?

USER: OK

AGENT: $ autogit pr --ut changed --st skip \
            --title "..." --body "..."
       Updating remote info...
       UT gate (changed, 1 files):
         tests/torch/ut/test_foo.py
       ........  [100%]
       1 passed in 4.12s
       UT gate passed.
       Submitting 1 commits to org/hyper-parallel
       Creating PR...
       PR created successfully!
          https://gitcode.com/org/hyper-parallel/pull/567
```

**Key things this scenario shows:**

- AI **never** passes `-m / --title / --body` without first showing the user.
- AI asks **one gate at a time** (UT then ST), not a single bundled menu.
- `--ut changed` runs only the test files in the PR diff (here 1 file),
  not the whole `tests/ut` suite.
- ST default is `skip` because most envs are single-card; user can opt up to
  `c` or `f` when running on 8-card.
- The pre-commit hook warning is non-blocking — the commit still goes
  through, the user is told how to install the hook.

**What the AI must NOT do:**

- Self-generate a commit message and pass `-m "..."` without preview.
- Catch the non-tty `AutoGitError` listing UT/ST flags and silently retry
  with `--ut skip --st skip`.
- Chain `commit` → `pr` in one turn without surfacing both the commit
  message and the PR title/body in chat.
