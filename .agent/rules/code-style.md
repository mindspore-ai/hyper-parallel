---
name: code-style
description: Global coding style and conventions for HyperParallel
---

# Code Style

## Purpose

Use these rules as the default coding style and convention set for HyperParallel. Apply them across the repository unless a more specific rule file overrides them.

## File And Formatting Rules

- All `.py` files must start with the Apache 2.0 license header at lines 1-16.
- **Copyright year**: for a **new file**, write `Copyright <current_year>` (the calendar year at creation — read the date from the environment). For an **existing file** whose header year differs from the current year, the header becomes `Copyright <existing_year>-<current_year>` (e.g. `2025-2026`). `autogit` extends this range automatically on commit; do not hand-edit to a past year on a new file.
- Keep Python lines to about 120 characters, following PEP 8 where practical.
- Keep C++ lines to 120 characters, following the project `.clang-format` and Google style expectations.
- Keep module-level function and class definitions separated by two blank lines.
- Keep nested definitions separated by one blank line where required by Python style rules.
- End files with a final newline.
- Start inline comments with `#`.
- Prefer one statement per line.
- Do not leave vague `TODO` comments in committed code. If a `TODO` is necessary, make it specific and actionable.
- **Inline comments (why only):** Prefer concise motivation for non-obvious constraints (~2–4 lines). Do not restate the next line of code. Do not leave job ids, commit hashes, one-off benchmark numbers, or machine-local paths. Upstream/issue/PR links are fine. Public APIs still use Google-style docstrings for contracts (`Args`/`Returns`/`Note`); inline `#` is for *why*, not a second docstring.

## Naming, Typing, And Documentation

- Use `PascalCase` for classes.
- Use `snake_case` for functions and variables.
- Keep method names in `snake_case`.
- Use `_leading_underscore` for private names.
- Use `UPPER_CASE` for module-level constants.
- Do not introduce custom names that both start and end with double underscores.
- Avoid ambiguous single-letter names such as `l`, `I`, and `o`.
- Require type hints on all public function signatures.
- Use Google-style docstrings with these sections when applicable: `Args:`, `Returns:`, `Raises:`, `Example:`, `Note:`.
- Public functions and methods should have docstrings.
- Keep docstring indentation consistent with standard Python formatting.

## Error Handling And Imports

- Order imports clearly: standard library, third-party, then first-party.
- Keep import placement consistent with module export structure such as `__all__`.

### Import placement (non-platform vs platform backends)

- **Default (most of the repo):** Put runtime `import` / `from … import` at **module top** (after the license header and any module docstring). Do **not** put imports inside functions, methods, or nested class bodies except the narrow exceptions under “Other exceptions” below. Applies to e.g. `core/`, `collectives/`, `tests/`, and **platform-agnostic** files such as `platform/platform.py`.
- **Platform backend implementations** (`hyper_parallel/platform/torch/**`, `hyper_parallel/platform/mindspore/**`, and any `*/platform/{torch,mindspore}/**` sub-package): Use **lazy imports inside methods** (lazy import / lazy init) for `torch`, `mindspore`, and their submodules as needed. This avoids pulling in the wrong framework at module import time, reduces import-order/cycle issues, and keeps the other backend unloadable when not in use. Add `# pylint: disable=C0415` on those lines.
- **Backend-conditional code in a platform-agnostic file** (e.g. `core/`): import-time backend imports are the violation (see `docs/rl-architecture.md` §2.1); use a **lazy import inside the function/method** so it only runs on the branch that needs it. I.e. `import torch` at module top of a `core/` file is a C9002 bug; `import torch` inside the `if platform_type == PlatformType.PYTORCH:` branch is the sanctioned pattern.
- **Other exceptions** (outside platform backends; each should include a brief comment explaining why):
  - Import-time circular dependency that cannot be fixed by restructuring.
  - Optional dependencies that may be missing at runtime.
  - Type-only symbols: prefer `from typing import TYPE_CHECKING` and an `if TYPE_CHECKING:` block at module scope instead of importing inside methods.
- Outside `platform/torch/` and `platform/mindspore/`, do not use local imports for convenience; do not blanket-suppress `C0415` unless the case matches an exception above.
- Validate inputs at boundaries.
- Raise `ValueError` with descriptive messages for invalid values.
- Do not rely on `assert` for runtime input validation or business logic checks.
- Handle important return values and exceptions explicitly instead of silently ignoring them.
- When reading dictionaries, prefer `get()` when absence is acceptable; otherwise catch or surface `KeyError` intentionally.

### Example: Dictionary Access

Correct:

```python
timeout = config.get("timeout", 30)

try:
    rank = config["rank"]
except KeyError as exc:
    raise ValueError("config must include 'rank'") from exc
```

Incorrect:

```python
timeout = config["timeout"]
rank = config["rank"]
```

## Design Rules

- Prefer composition over inheritance where possible.
- Methods that do not use instance state should be converted to `@staticmethod` or `@classmethod`.
- Avoid redundant or dead code.
- Avoid excessive local variables, argument counts, boolean clauses, and cyclomatic complexity. Refactor large functions before they become hard to review or test.
- Prefer the logging framework over `print`, `sys.stdout.write`, or `sys.stderr.write` in production code.
- Define instance attributes in `__init__` unless there is a deliberate and well-documented reason not to.
- Avoid direct access to another class's protected members unless no stable public API exists and the coupling is explicitly justified.
- Platform-agnostic code must go through the `get_platform()` abstraction.
- Do not import `torch` or `mindspore` directly in platform-agnostic code.
- Avoid GPU-CPU synchronization in hot paths. In training loops, avoid patterns such as `.item()`, `.numpy()`, and `print(tensor)`.
- Prefer `os.path` or `pathlib` helpers over manual string concatenation for filesystem paths.
- Do not mutate `sys.path` with patterns such as `insert(0, ...)` unless there is no alternative and the reason is documented.
- Keep lambda expressions to simple one-line cases only.
- Use `functools.wraps` when implementing decorators.
- Avoid assignment expressions unless they clearly improve readability.
- Keep comprehensions and conditional expressions readable. Do not pack too many clauses or multi-line logic into a single expression.
- Avoid unused imports and unused loop variables.

### Example: `@staticmethod` / `@classmethod`

Correct:

```python
class MeshUtils:
    @staticmethod
    def normalize_rank(rank: int) -> int:
        return max(rank, 0)
```

Incorrect:

```python
class MeshUtils:
    def normalize_rank(self, rank: int) -> int:
        return max(rank, 0)
```

## Command Execution And Security

- When invoking subprocesses, prefer `shell=False` and pass commands as argument lists.
- Avoid depending on ambient `PATH` resolution for critical executables when a stable explicit path is required by the environment.

### Example: Safe Subprocess Invocation

Correct:

```python
subprocess.run(["/usr/bin/git", "status"], check=True, shell=False)
```

Incorrect:

```python
subprocess.run("git status", shell=True, check=True)
```

## Platform Reference Convention

- If a file needs platform APIs, define `platform = get_platform()` once at module scope.
- Do not store platform on instances such as `self.platform`. This creates ambiguity between module-level and instance-level references and can hide bugs.
- When a class method needs platform access, reference the module-level `platform` variable directly.
- When copying code between methods, verify every `platform.*` call uses the correct API variant for the context, especially `differentiable_*` vs non-differentiable APIs.

### Example: Platform-Agnostic Code Must Not Import Backends Directly

Correct:

```python
from hyper_parallel.platform import get_platform

platform = get_platform()
result = platform.all_reduce(tensor)
```

Incorrect:

```python
import torch.distributed as dist

result = dist.all_reduce(tensor)
```

## Fix Over Evade (pylint / UT / ST)

When a check fails (pylint finding, UT failure, ST failure), **default to a positive fix**. Before reaching for a suppression, a skip, or an environment workaround:

1. Search the repo for similar code / tests / ops and see how the same situation is handled there — follow the established pattern rather than inventing a new one.
2. Fix the root cause (refactor the code, install the correct backend version, correct the test expectation).
3. Only evade when evasion is genuinely the only path (e.g. an inline `# pylint: disable=C0415` on a backend lazy import is the precedent-established pattern; a test needs a real hardware gate like multi-card NPU). In that case **say so explicitly** in the PR description — do not silently disable, `try/except`, `@skipif`, or swap environment variables.

## Commit Convention

Use Conventional Commits: `<type>: <description>` (optional scope: `<type>(<scope>): <description>`).

Allowed types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`.

- **Subject:** imperative mood, **~80 characters**, no trailing period. (Code line length is ~120; do not confuse the two.)
- Body (optional): explain *why*, wrap reasonably; keep business-side only.
- Do not include AI-assistant or IDE attribution trailers such as `Made-with: <tool>` or `Co-authored-by: <AI assistant>`.
- Do not include third-party AI tool/service names in commit messages.
- Enforced by `autogit` and the optional git hook `.agent/hooks/commit-msg` (install into `.git/hooks/commit-msg`). Canonical detail also summarized in `AGENTS.md` § Git Workflow.
