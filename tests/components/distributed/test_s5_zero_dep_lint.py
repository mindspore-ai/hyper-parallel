# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S5.5: 零依赖 lint（components/distributed 不得 import recipes/_transformers/
models/datasets）。"""

import ast
from pathlib import Path

import hyper_models.components.distributed as dist_pkg

FORBIDDEN = ("recipes", "_transformers", "hyper_parallel.models",
             "datasets", "trainer")


def _imports_of(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module


def test_zero_dependency_lint():
    pkg_dir = Path(dist_pkg.__file__).parent
    checked = 0
    for py in pkg_dir.rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        checked += 1
        for mod in _imports_of(py):
            for bad in FORBIDDEN:
                assert bad not in mod, f"{py} 违规依赖 {mod}（含 {bad}）"
    assert checked >= 8, f"仅检查了 {checked} 个文件，lint 覆盖不足"
