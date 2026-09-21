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

### Import placement

- **Default:** Put runtime `import` / `from … import` at **module top** (after the license header and any module docstring). Do **not** put imports inside functions, methods, or nested class bodies except the narrow exceptions under “Other exceptions” below. Applies to e.g. `core/`, `collectives/`, `tests/`.
- **Multicore** (`hyper_parallel/core/multicore/**`): use direct module-level Torch imports.
  The component root may export its business APIs; the HyperParallel root must not export them.
  Native-library initialization may still be deferred until an operation needs the activated payload.
- **DFunction** (`hyper_parallel/core/shard/dfunction.py`): import Torch at module scope
  and inherit directly from `torch.autograd.Function`.
- **Pipeline** (`hyper_parallel/core/pipeline_parallel/**`): import Torch APIs
  directly at module scope; keep stage execution, micro-batches, and P2P inside the core component.
- **Other exceptions** (each should include a brief comment explaining why):
  - Import-time circular dependency that cannot be fixed by restructuring.
  - Optional dependencies that may be missing at runtime — e.g. `torch_npu` and Omni custom
    operators, which a plain `import hyper_parallel` must not pull in.
  - Type-only symbols: prefer `from typing import TYPE_CHECKING` and an `if TYPE_CHECKING:` block at module scope instead of importing inside methods.
- Do not use local imports for convenience; do not blanket-suppress `C0415` unless the case matches an exception above.
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
- The `platform/` abstraction and the MindSpore backend are gone: use native Torch APIs directly.
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

## Collective Reference Convention

- Import `torch.distributed` once at module scope and call `dist.*` directly, or use the shared
  helpers in `hyper_parallel/core/context_parallel/utils.py` and
  `hyper_parallel/core/dtensor/_utils.py` when the shape is fixed.
- When copying code between methods, check which collective variant the context needs —
  `differentiable_*` in forward/backward paths, the eager ones elsewhere.
- `create_group()` takes a rank list and returns a raw process group; helpers that expect a
  `group_info` wrapper need `.group` unwrapped, and vice versa.

### Example: All-Reduce Inside an Autograd Path

Correct:

```python
from hyper_parallel.core.dtensor._utils import differentiable_all_reduce

result = differentiable_all_reduce(tensor, "sum", group)
```

Outside autograd, the eager call is fine:

```python
import torch.distributed as dist

dist.all_reduce(tensor, group=group)
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
