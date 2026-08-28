# AutoGit Command Reference

Complete parameter docs, execution flows, and setup instructions.

---

## 1. commit — Stage, Preview Message, Commit, Push

```bash
autogit commit                      # generate + preview commit message, then commit
autogit commit -m "feat: new thing" # specify commit message (skips preview)
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-m`, `--message` | Commit message | Auto-generated; previewed on tty, required on non-tty |

### Execution Flow

```text
Check for changes → stage (filter cosmetic-only edits) → warn if pre-commit
  hook missing → AGENTS catalog check → preview commit message (tty) /
  require -m (non-tty) → git commit (pre-commit hook auto-runs lint) →
  git push origin <branch>
```

### Lint Ownership

- Lint (pylint + markdownlint) is owned by the project's **pre-commit git hook**, installed via `bash scripts/pre-commit/install.sh`. The hook fires on every `git commit` automatically — autogit does not duplicate it.
- If the hook is not installed, `autogit commit` prints a one-line install reminder and continues (non-blocking).
- **AGENTS catalog** (`check_agents_catalog.py`) runs in `autogit commit` and `autogit check` (blocking). Ensures `AGENTS.md` Skills/Agents tables match `.agent/skills` and `.agent/agents` on disk.
- For an explicit no-commit precheck, use `autogit check` (see §2).

### Commit Message Convention

| Prefix | Purpose |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `refactor:` | Refactoring |
| `docs:` | Documentation |
| `test:` | Tests |
| `chore:` | Build/tooling |

**No AI-assistant/IDE attribution or third-party references** — Commit messages should only describe business-side changes. Must not contain attribution trailers (`Made-with: <tool>`, `Co-authored-by: <AI assistant>`) or third-party AI tool/service names. autogit rejects such messages. To also catch manual/editor-UI commits, install the commit-msg hook:

```bash
cp .agent/hooks/commit-msg .git/hooks/commit-msg && chmod +x .git/hooks/commit-msg
```

---

## 2. check — Run Lint Checks Only

```bash
autogit check    # check all uncommitted changes, do not commit
```

No parameters. Stages files temporarily, runs all checks, then unstages.
The first step is always `.agent/rules/code-style.md` validation plus auto-fix for mechanical issues.
If `pylint` or `markdownlint-cli2` is missing, AutoGit installs the dependency declared in `.pre-commit-config.yaml` before continuing.

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
| agents-catalog | `AGENTS.md` Skills/Agents vs `.agent/` disk | Always (full tree) |

AutoGit auto-installs missing `pylint` and `markdownlint-cli2` according to `.pre-commit-config.yaml`. Other tools still degrade gracefully if unavailable.

---

## 3. test — Run pytest Only

```bash
autogit test
```

Runs `pytest tests/ut -v` only. Lint checks are intentionally excluded from the test stage.

---

## 4. pr — Create Pull Request

```bash
autogit pr                        # create PR (keep multiple commits)
autogit pr --squash               # create PR (squash into 1 commit)
autogit pr --base develop         # target branch
autogit pr --reviewer zhangsan    # assign reviewer
autogit pr --ut skip --st skip    # skip both gates (only if user opted in)
autogit pr --ut full              # run full UT suite instead of changed-scope
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--base <branch>` | PR target branch | upstream default branch (master) |
| `--reviewer <users>` | Reviewers (comma-separated) | None |
| `--squash` | Squash all commits into one | Off |
| `--ut {skip,changed,full}` | UT gate: skip / changed (PR-diff scope) / full | Ask on tty (Enter→`changed`); error on non-tty |
| `--st {skip,changed,full}` | ST gate: skip / changed (PR-diff scope) / full | Ask on tty (Enter→`skip`); error on non-tty |

### Smart Branch Strategy

| Current Branch | Behavior | Reason |
|----------------|----------|--------|
| Feature branch | Use current branch directly | Recommended, simple |
| master/main | Auto-create `pr/<timestamp>` branch | Protect main branch |

### Execution Flow

```text
1. Check env (token, remotes, uncommitted changes)
   ⚠ Refuse if uncommitted changes exist — commit first
2. Resolve UT/ST gates (ask one-at-a-time on tty; error on non-tty if undecided)
3. Run UT gate (changed: only PR-diff test files; full: tests/ut)
4. Run ST gate when scope != skip (changed: PR-diff ST files; full: tests/{torch,mindspore}/st)
5. Determine branch type
   • feature branch → use directly
   • master/main → create new branch + cherry-pick
6. Push to origin (prompt on conflict, never force)
7. Analyze diff, preview/accept PR title+body
8. Call API to create PR, output URL
```

### Auto-generated PR Description

100% based on code diff. Includes:

- Feature domain and purpose inference
- Added/modified/removed classes and methods
- File change statistics
- Affected functionality analysis

---

## 5. pr --to — Append to Existing PR

```bash
autogit pr --to #160              # append new commit (default: rebase first)
autogit pr --to #160 --amend     # amend into last commit
autogit pr --to #160 --no-rebase # skip rebase, append directly
autogit pr --to #160 --ut skip   # skip UT gate (only if user opted in)
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--to <#N>` | PR number to append to | Required |
| `--amend` | Merge into last commit instead of new | Off |
| `--no-rebase` | Skip rebase before appending | Off (rebase on) |
| `--ut {skip,changed,full}` | UT gate: changed scope reads working-tree (staged + unstaged); ST is not asked for append | Ask on tty (Enter→`changed`); error on non-tty |
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

## 6. status — View PR Status

```bash
autogit status #160
autogit status https://gitcode.com/org/repo/pull/160
```

Read-only. Displays: state (open/merged/closed/draft), author, branches, timestamps, stats (+/-/files/commits), reviewers, and URL.

---

## 7. update — Regenerate PR Description

```bash
autogit update #160
```

Re-analyzes PR diff and regenerates description via API. No code or commit changes. Preserves original title if it looks better than auto-generated one.

---

## 8. squash — Squash PR Commits

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
