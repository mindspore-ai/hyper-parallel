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

## Commit Convention

`<type>: <description>` — types: `feat` / `fix` / `refactor` / `docs` / `test` / `chore`
