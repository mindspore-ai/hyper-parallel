---
name: code-review
description: Review HyperParallel code changes for distributed correctness, stream synchronization, memory safety, cross-platform consistency, and code quality. Use when reviewing PRs, code changes, or when the user mentions "review", "code review", or "check this".
---

# HyperParallel Code Review Skill

Review HyperParallel code changes focusing on what CI cannot check: distributed system correctness, stream synchronization safety, memory lifecycle, cross-platform consistency, and code quality.

## Usage Modes

### No Argument

If the user invokes `/code-review` with no arguments, ask what to review:

> What would you like me to review?
> - A PR number or URL (e.g., `/code-review #160`)
> - A local branch (e.g., `/code-review branch`)

### Local Branch Mode

Review changes in the current branch against `upstream/master`:

```
/code-review branch
/code-review branch detailed
```

```bash
git branch --show-current
git diff upstream/master...HEAD --name-only
git diff upstream/master...HEAD
git diff --stat upstream/master...HEAD
git log upstream/master..HEAD --oneline
```

### PR Number Mode

The user provides a PR number or URL:

```
/code-review #160
/code-review https://gitcode.com/mindspore/hyper-parallel/pulls/160
```

Fetch PR data via GitCode API (requires `GITCODE_TOKEN`), or fall back to local branch diff if token unavailable.

### Detailed Mode

Append `detailed` for line-by-line specific comments:

```
/code-review branch detailed
/code-review #160 detailed
```

## Review Workflow

### Step 1: Fetch Changes

Get the full diff, changed file list, and commit history:

```bash
# For local branch
git diff upstream/master...HEAD
git log upstream/master..HEAD --oneline

# For PR number (if API available)
# Fetch PR metadata and diff via GitCode API
```

### Step 2: Classify Changes

Group changed files by risk level:

| Risk | File Patterns | Reason |
|------|--------------|--------|
| CRITICAL | `**/fully_shard/**`, `**/hsdp/**`, stream/sync code | Memory stomping, stale data |
| CRITICAL | Async collectives, `non_blocking`, cross-stream code | Stream sync bugs |
| HIGH | `core/dtensor.py`, `core/layout.py`, `core/tensor_redistribution.py` | DTensor correctness |
| HIGH | `platform/torch/**`, `platform/mindspore/**` | Cross-platform consistency |
| HIGH | `core/pipeline_parallel/**` | Pipeline deadlocks, buffer leaks |
| MEDIUM | `core/shard/ops/**` | Op dispatch, YAML registration |
| MEDIUM | `core/activation_checkpoint/**` | Activation swap lifecycle |
| LOW | `tests/**` | Test additions |
| LOW | `examples/**`, `*.md`, config files | Docs, config |

### Step 3: Deep Review

Perform thorough analysis using the review checklist. See [review-checklist.md](review-checklist.md) for detailed criteria covering:
- Distributed system correctness
- Stream synchronization safety
- Memory lifecycle management
- Cross-platform consistency
- Code quality and design
- Testing adequacy

### Step 4: Check Distributed Correctness

Evaluate distributed system implications. See [distributed-guidelines.md](distributed-guidelines.md) for:
- Stream synchronization rules and common violations
- Memory management patterns and leak prevention
- Cross-platform compatibility requirements
- DTensor invariants and common pitfalls

### Step 5: Formulate Review

Structure feedback with actionable suggestions organized by category.

## Review Areas

| Area | Focus | Reference |
|------|-------|-----------|
| Stream Sync | Async collectives, non_blocking, cross-stream deps | [distributed-guidelines.md](distributed-guidelines.md) |
| Memory Safety | Storage lifecycle, buffer reuse, grad cleanup | [distributed-guidelines.md](distributed-guidelines.md) |
| DTensor | Layout, placement, redistribution, op dispatch | [review-checklist.md](review-checklist.md) |
| Cross-Platform | torch/mindspore parity, platform abstraction | [review-checklist.md](review-checklist.md) |
| Code Quality | Patterns, complexity, conventions | [review-checklist.md](review-checklist.md) |
| Testing | Coverage, distributed test patterns | [review-checklist.md](review-checklist.md) |

## Output Format

```markdown
## PR Review: #<number>
<!-- Or for local branch reviews: -->
## Branch Review: <branch-name> (vs upstream/master)

### Summary
Brief overall assessment (1-2 sentences).

### Distributed Correctness
[Stream sync issues, memory safety concerns, or "No distributed correctness concerns"]

### Cross-Platform Consistency
[Missing platform parity, abstraction violations, or "No cross-platform concerns"]

### DTensor / Op Dispatch
[Layout issues, placement bugs, YAML registration gaps, or "No DTensor concerns"]

### Code Quality
[Design issues, convention violations, or "No concerns"]

### Testing
- [ ] Tests exist for new functionality
- [ ] Distributed tests use torchrun_case / msrun_case correctly
- [ ] Edge cases covered (empty tensors, single rank, partial state)
[Additional testing feedback]

### Recommendation
**Approve** / **Request Changes** / **Needs Discussion**

[Brief justification]
```

### Specific Comments (Detailed Review Only)

Only include this section if the user requests "detailed" review. Do not repeat observations from other sections.

```markdown
### Specific Comments
- `core/layout.py:42` - is_partial() called without parentheses — this is a method, not a property
- `platform/torch/fully_shard/param.py:100-105` - Missing resize_(0) after consuming all-gather output
```

## Key Principles

1. **No repetition** — Each observation appears in exactly one section
2. **Distributed-first** — Prioritize stream sync, memory safety, and correctness over style
3. **Be specific** — Reference file paths and line numbers
4. **Be actionable** — Provide concrete fixes, not vague concerns
5. **Cross-platform awareness** — Always check if the other backend needs a matching change
6. **Assume competence** — The author knows distributed systems; explain only non-obvious context

---

*Code Quality, Thread Safety & Concurrency, and Performance sections in the review checklist are adapted from [PyTorch's PR review skill](https://github.com/pytorch/pytorch/tree/main/.claude/skills/pr-review), with HyperParallel-specific distributed concurrency and cross-platform extensions.*
