---
description: Global coding style and conventions for HyperParallel
---

## Code Style

- **License header**: All `.py` files start with the Apache 2.0 header (lines 1-16)
- **Line length**: Python ~120 chars (PEP 8); C++ 120 chars (Google style, `.clang-format`)
- **Naming**: Classes `PascalCase`, functions/vars `snake_case`, private `_leading_underscore`
- **Docstrings**: Google-style with `Args:`, `Returns:`, `Raises:`, `Example:`, `Note:` sections
- **Type hints**: Required on all public function signatures
- **Errors**: Raise `ValueError` with descriptive messages; validate at boundaries
- **Imports**: Lazy imports inside methods use `# pylint: disable=C0415`

## Design Patterns

- Composition over inheritance where possible
- Platform-agnostic code must use `get_platform()` abstraction, never import torch/mindspore directly
- Avoid GPU-CPU synchronization in hot paths (no `.item()`, `.numpy()`, `print(tensor)` in training loops)

## Platform Reference Convention

- Use **module-level** `platform = get_platform()` at the top of each file that needs platform APIs
- **Never** store platform as an instance attribute (`self.platform`) — this creates ambiguity between module-level and instance-level references and hides bugs
- When a method in a class needs platform, reference the module-level `platform` variable directly
- When copying code between methods, verify that all `platform.*` calls use the correct API variant (`differentiable_*` vs non-differentiable) for the context

## Commit Convention

`<type>: <description>` — types: `feat` / `fix` / `refactor` / `docs` / `test` / `chore`

**No AI/IDE attribution or third-party references** — Commit messages should only describe business-side changes. Do not include attribution trailers (`Made-with: Cursor`, `Co-authored-by: Claude`) or third-party tool names (Cursor, Copilot, ChatGPT, Claude, Gemini, etc.). Disable Cursor Settings > Agent > Attribution. autogit and the optional `commit-msg` hook enforce this.
