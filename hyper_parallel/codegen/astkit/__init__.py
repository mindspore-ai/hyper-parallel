# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""AST patch toolkit for generated forward lowering.

The lowerer finds each boundary class's ``forward`` and re-implements it as an
explicit call to the codegen runtime (``hyper_redistribute`` plus an inner
wrapper or local compute function).
This package owns that structural patch:

- :mod:`index` — parse the source once and answer "where is class X's
  forward" / "where do imports end".
- :mod:`edits` — apply positional text edits (replace a body, insert after
  imports, append a footer).

Both layers are pure text over source; nothing here imports torch or the
training stack, so a generation-time host can patch a modeling file without
loading the model.
"""
from hyper_parallel.codegen.astkit.edits import (
    TextEdit,
    apply_edits,
    append_module_footer,
    insert_after_imports,
    replace_function_body,
)
from hyper_parallel.codegen.astkit.index import SourceIndex, build_source_index

__all__ = [
    "SourceIndex",
    "TextEdit",
    "apply_edits",
    "append_module_footer",
    "build_source_index",
    "insert_after_imports",
    "replace_function_body",
]
