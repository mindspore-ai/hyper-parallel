---
name: code-review
description: >
  Review HyperParallel changes for distributed correctness, stream sync,
  memory safety, cross-platform parity, and code quality. Use for /code-review,
  PR review, or when the user asks to review/check changes.
---

# HyperParallel Code Review Skill

**Source of truth for full review workflow.** Agents `code-reviewer` proxies here;
`simple-code-reviewer` is a separate fast checklist only.

Before reviewing: load `.agent/rules/code-style.md` (blocking) and
`.agent/rules/distributed.md` for hard distributed rules.

## Modes

| Invocation | Behavior |
|---|---|
| `/code-review` (no args) | Ask: PR number/URL vs local `branch` |
| `/code-review branch` [`detailed`] | Diff vs `upstream/master` |
| `/code-review #N` or GitCode URL [`detailed`] | PR via API (`GITCODE_TOKEN`) or local fallback |

```bash
git branch --show-current
git diff upstream/master...HEAD --name-only
git diff upstream/master...HEAD
git log upstream/master..HEAD --oneline
```

## Workflow (load details on demand)

1. **Fetch** diff + file list + commits (branch or PR).
2. **Classify** risk — table in [review-checklist.md](review-checklist.md) / patterns below.
3. **Deep review** CRITICAL/HIGH using [review-checklist.md](review-checklist.md).
4. **Distributed** implications via [distributed-guidelines.md](distributed-guidelines.md).
5. **Pylint (PR stage):**
   ```bash
   python3 .agent/skills/autogit/scripts/autogit.py pylint-review
   ```
   Paste into Code Quality. Suppress via `.jenkins/check/config/filter_pylint.txt`, not inline `# pylint: disable=`.
6. **Formulate** output (template below). `code-style` violations need corrected snippets.

### Risk hints

| Risk | Patterns |
|------|----------|
| CRITICAL | `**/fully_shard/**`, async collectives, `non_blocking`, cross-stream |
| HIGH | `core/dtensor/**`, `platform/torch|mindspore/**`, `core/pipeline_parallel/**` |
| MEDIUM | `core/shard/ops/**`, `core/activation_checkpoint/**` |
| LOW | `tests/**`, docs, examples |

## Output skeleton

```markdown
## PR Review: #<n>   OR   ## Branch Review: <branch> (vs upstream/master)
### Summary
### Distributed Correctness
### Cross-Platform Consistency
### DTensor / Op Dispatch
### Code Quality   # include pylint; style fixes are mandatory
### Testing
### Recommendation
**Approve** / **Request Changes** / **Needs Discussion**
```

`detailed` only: add `### Specific Comments` with `path:line` items (no duplication of section findings).

## Principles

1. No repetition across sections  
2. Distributed-first (sync/memory before style)  
3. Specific paths + actionable fixes  
4. Always check other backend when touching `platform/`  
5. Style is blocking  

## References

- [review-checklist.md](review-checklist.md)
- [distributed-guidelines.md](distributed-guidelines.md)
- `.agent/rules/distributed.md`, `.agent/rules/code-style.md`
