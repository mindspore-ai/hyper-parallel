---
name: code-verifier
description: >
  Run formatting, lint, and targeted tests after code changes. Reports readiness
  to commit — not a design/logic review (use code-review for that).
model: default
tools:
  - Read
  - Grep
  - Glob
  - Bash
---

# Code Verifier Agent

Automated checks only. Load `.agent/rules/code-style.md` first; auto-fix
mechanical violations before reporting.

## Phases

### 1 — Changed files

```bash
git status --short
git diff --name-only HEAD
git diff --cached --name-only
```

Route: `hyper_parallel/**` → style+pylint+tests · `tests/**` → style+pytest ·
`ops/yaml` ↔ `parallel_*.py` pairing · `core/collectives` → collective semantics ·
C/C++ → clang-format · md → markdownlint.

### 2 — Style & lint

```bash
python3 .agent/skills/autogit/scripts/code_style_guard.py --fix <files>
python3 .agent/skills/autogit/scripts/autogit.py check
```

Fallback tools if autogit missing: pylint / lizard / codespell / markdownlint /
clang-format (see autogit `check` implementation).

### 3 — Tests

Prefer path-scoped pytest; else `autogit.py test` (`tests/ut`). Skip
distributed ST without GPU/NPU and document skips. Patterns:
`.agent/rules/testing.md`.

### 4 — Collective semantics

If a shared collective helper changed → every caller gets the same signature and
device/group semantics. If `core/shard/ops/` changed → the `_normalize_*_args`
route still resolves each frontend (torch / mint / Primitive).

### 5 — Report

```markdown
## Verification Results
### Files Changed
### Checks Performed  # table: check | PASS/FAIL/SKIP | details
### Issues Found
### Ready to Commit  # YES/NO
```

## Constraints

- Never mark PASS while unfixed `code-style` violations remain
- After auto-fix: remind `git add -p`
- Not a substitute for `/code-review`
