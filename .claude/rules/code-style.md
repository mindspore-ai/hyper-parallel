---
description: Global coding style and conventions for HyperParallel
---

# Code Style

## Purpose

Use these rules as the default coding style and convention set for HyperParallel. Apply them across the repository unless a more specific rule file overrides them.

## File And Formatting Rules

- All `.py` files must start with the Apache 2.0 license header at lines 1-16.
- Keep Python lines to about 120 characters, following PEP 8 where practical.
- Keep C++ lines to 120 characters, following the project `.clang-format` and Google style expectations.
- Keep module-level function and class definitions separated by two blank lines.
- Keep nested definitions separated by one blank line where required by Python style rules.
- End files with a final newline.
- Start inline comments with `#`.
- Prefer one statement per line.
- Do not leave vague `TODO` comments in committed code. If a `TODO` is necessary, make it specific and actionable.

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
- When using lazy imports inside methods, add `# pylint: disable=C0415`.
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

## Commit Convention

Use the format `<type>: <description>`.

Allowed types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`.

Commit messages should describe business-side changes only.

- Do not include AI or IDE attribution trailers such as `Made-with: Cursor` or `Co-authored-by: Claude`.
- Do not include third-party tool names in commit messages, including Cursor, Copilot, ChatGPT, Claude, Gemini, and similar tools.
- `autogit` and the optional `commit-msg` hook enforce this rule.
