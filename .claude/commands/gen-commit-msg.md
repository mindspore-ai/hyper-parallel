# /gen-commit-msg — Generate Commit Message

Generate a well-formatted commit message based on staged changes, without running the
full `autogit commit` workflow.

## Usage

```bash
/gen-commit-msg                # auto-generate from staged changes
/gen-commit-msg --scope shard  # force a specific scope
```

## What It Does

Analyzes staged changes and generates a Conventional Commits message. Unlike `/commit`,
this does NOT stage, lint-check, commit, or push — it only produces the message text.

## Workflow

### Step 1: Analyze Changes

```bash
git diff --cached --name-only
git diff --cached --stat
git diff --cached
git log --oneline -5
```

### Step 2: Determine Type

| Type | When to Use |
| ---- | ----------- |
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `refactor` | Code change without feature/fix |
| `docs` | Documentation only |
| `test` | Adding or fixing tests |
| `chore` | Build, deps, config changes |

### Step 3: Infer Scope

Infer scope from changed file paths:

| Changed Path | Scope |
| ------------ | ----- |
| `hyper_parallel/core/dtensor/` | `dtensor` |
| `hyper_parallel/core/shard/` | `shard` |
| `hyper_parallel/core/shard/ops/` | `ops` |
| `hyper_parallel/core/fully_shard/` | `fsdp` (includes shared `hsdp_*.py` / HSDP scheduler state) |
| `hyper_parallel/core/pipeline_parallel/` | `pipeline` |
| `hyper_parallel/core/activation_checkpoint/` | `activation` |
| `hyper_parallel/core/checkpoint/` | `checkpoint` |
| `hyper_parallel/platform/torch/` | `torch` |
| `hyper_parallel/platform/mindspore/` | `mindspore` |
| `hyper_parallel/platform/` (base) | `platform` |
| `hyper_parallel/collectives/` | `collectives` |
| `tests/` | `test` |
| `docs/` | `docs` |
| `.claude/` | `ci` |
| Multiple areas | omit scope or use broader term |

### Step 4: Generate Message

**Format:**

```
<type>(<scope>): <subject>

<body>
```

**Rules:**

- Subject: imperative mood, ~50-72 chars, no period
- Body: explain "why" not "what", wrap at 72 chars
- Do NOT include AI/IDE attribution (no `Made-with:`, `Co-authored-by:` trailers)
- Follow `.claude/rules/code-style.md` commit convention

### Step 5: Preview and Confirm

Show preview:

```
─────────────────────────────────────
feat(ops): add parallel conv3d operator

Add layout derivation and dispatch for conv3d with
shard support on batch and channel dimensions.
─────────────────────────────────────
```

Ask user to confirm. If confirmed, provide the ready-to-use command:

```bash
git commit -m "$(cat <<'EOF'
<message>
EOF
)"
```

## Examples

**Single op addition:**

```
feat(ops): add parallel unbind operator

Support unbind with automatic layout derivation for
sharded tensors along any dimension.
```

**Cross-platform fix:**

```
fix(platform): align reduce_scatter return type across backends

MindSpore path was returning raw tensor while torch path
returned DTensor. Normalize both to return DTensor with
correct layout.
```

**FSDP change:**

```
refactor(fsdp): simplify parameter unsharding lifecycle

Consolidate pre-forward and pre-backward unsharding into
a shared helper to reduce code duplication between torch
and mindspore paths.
```

## See Also

- `/commit` — Full stage, lint-check, commit, push workflow via autogit
- `.claude/rules/code-style.md` — Commit convention rules
