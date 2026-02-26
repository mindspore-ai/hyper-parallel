# AutoGit Command Reference

Complete parameter docs, execution flows, and setup instructions.

---

## 1. commit — Stage, Check, Commit, Push

```bash
autogit commit                      # auto-generate commit message (with lint checks)
autogit commit -m "feat: new thing" # specify commit message
autogit commit --no-check           # skip lint checks (emergency only)
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-m`, `--message` | Commit message | Auto-generated from file names |
| `--no-check` | Skip pre-commit lint checks | Off (checks run) |

### Execution Flow

```text
Check for changes → git add -A → lint checks (pylint/lizard/codespell/markdownlint)
  → git commit → git push origin <branch>
```

### Lint Checks

- Runs pylint, lizard, codespell, markdownlint by default before commit.
- Fails: unstages all changes, prompts for fix.
- Tool not installed: warns but does not block (graceful degradation).
- Use `--no-check` to skip all checks (emergency only).
- Project-level `filter_pylint.txt` supported for known-issue filtering.

### Commit Message Convention

| Prefix | Purpose |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `refactor:` | Refactoring |
| `docs:` | Documentation |
| `test:` | Tests |
| `chore:` | Build/tooling |

---

## 2. check — Run Lint Checks Only

```bash
autogit check    # check all uncommitted changes, do not commit
```

No parameters. Stages files temporarily, runs all checks, then unstages.

### Check Tools & Thresholds

| Tool | Threshold | Target Files |
|------|-----------|-------------|
| pylint | max-line-length=120, design/similarities disabled | `*.py` |
| lizard | CCN≤19, NLOC≤100 | `*.py`, `*.c/cpp/h` |
| cpplint | Google C++ style | `*.c`, `*.cc`, `*.cpp`, `*.h`, `*.hpp` |
| clang-format | Formatting (dry-run, no modify) | `*.c`, `*.cc`, `*.cpp`, `*.h`, `*.hpp` |
| cmakelint | CMake style | `CMakeLists.txt`, `*.cmake` |
| shellcheck | Shell script analysis | `*.sh`, `*.bash` |
| codespell | Spelling check | All text files |
| markdownlint-cli2 | Standard rules | `*.md` |

All tools degrade gracefully — if not installed, warns but does not block.

---

## 3. pr — Create Pull Request

```bash
autogit pr                        # create PR (keep multiple commits)
autogit pr --squash               # create PR (squash into 1 commit)
autogit pr --base develop         # target branch
autogit pr --reviewer zhangsan    # assign reviewer
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--base <branch>` | PR target branch | upstream default branch (master) |
| `--reviewer <users>` | Reviewers (comma-separated) | None |
| `--squash` | Squash all commits into one | Off |

### Smart Branch Strategy

| Current Branch | Behavior | Reason |
|----------------|----------|--------|
| Feature branch | Use current branch directly | Recommended, simple |
| master/main | Auto-create `pr/<timestamp>` branch | Protect main branch |

### Execution Flow

```text
1. Check env (token, remotes, uncommitted changes)
   ⚠️ Refuse if uncommitted changes exist — commit first
2. Determine branch type
   • feature branch → use directly
   • master/main → create new branch + cherry-pick
3. Push to origin (prompt on conflict, never force)
4. Analyze diff, auto-generate PR description
5. Call API to create PR, output URL
```

### Auto-generated PR Description

100% based on code diff. Includes:

- Feature domain and purpose inference
- Added/modified/removed classes and methods
- File change statistics
- Affected functionality analysis

---

## 4. pr --to — Append to Existing PR

```bash
autogit pr --to #160              # append new commit (default: rebase first)
autogit pr --to #160 --amend     # amend into last commit
autogit pr --to #160 --no-rebase # skip rebase, append directly
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--to <#N>` | PR number to append to | Required |
| `--amend` | Merge into last commit instead of new | Off |
| `--no-rebase` | Skip rebase before appending | Off (rebase on) |
| `-m`, `--message` | Commit message for new commit | Auto-generated |

### Three Modes Compared

| Mode | Command | Use Case | Rewrites History? |
|------|---------|----------|-------------------|
| Default | `--to #N` | Normal feature addition | Yes (rebase) |
| Amend | `--to #N --amend` | Small fix, typo | Yes |
| No-rebase | `--to #N --no-rebase` | Avoid conflicts | No |

### Safety

- Auto-stash local changes before branch switch.
- Rebase failure auto-aborts and restores.
- Returns to original branch after completion.

---

## 5. status — View PR Status

```bash
autogit status #160
autogit status https://gitcode.com/org/repo/pull/160
```

Read-only. Displays: state (open/merged/closed/draft), author, branches, timestamps, stats (+/-/files/commits), reviewers, and URL.

---

## 6. update — Regenerate PR Description

```bash
autogit update #160
```

Re-analyzes PR diff and regenerates description via API. No code or commit changes. Preserves original title if it looks better than auto-generated one.

---

## 7. squash — Squash PR Commits

```bash
autogit squash #160              # squash all commits
autogit squash #160 -m "message" # specify squash commit message
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `pr_ref` | PR number or URL | Required |
| `-m`, `--message` | Squashed commit message | First commit's message |

### Safety

- Only executes if PR has > 1 commit.
- Rebases to upstream/target first (avoids unrelated file changes).
- Force-pushes after squash — user should be aware.
- Uses first commit's message as default.

---

## Setup & Prerequisites

### Fork Workflow

```bash
# Check remotes
git remote -v

# Expected:
# origin    git@gitcode.com:<your-username>/<repo>.git  ← your fork (writable)
# upstream  git@gitcode.com:<org>/<repo>.git            ← main repo (read-only)

# Add upstream if missing
git remote add upstream git@gitcode.com:<org>/<repo>.git
```

### Token

```bash
# Linux/macOS
export GITCODE_TOKEN=<your-token>

# Windows PowerShell
$env:GITCODE_TOKEN="<your-token>"
```

Get token: <https://gitcode.com/setting/token-classic>

---

## Platform Support

| Platform | Status |
|----------|--------|
| GitCode | Supported |
| GitHub | Planned |
| GitLab | Planned |
