# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Identity-preserving ``module_replacement`` fixture for contract tests.

This mirrors the pattern established in
``tests/hyper_parallel/auto_models/components/model_transform/test_replacement.py`` so the
structure-preserving checks in ``components.model_transform.replacement`` pass:

- ``ReplacementMLP`` reuses the *same* ``nn.Linear`` object as ``SourceMLP``, so
  parameter identity (``proj.weight``/``proj.bias``) is preserved;
- the ``@module_replacement`` factory takes only ``{module, module_fqn, context}``
  and no ``**kwargs``;
- codegen emit receives the YAML Target path explicitly; tests that drive the
  raw ``module_replacement`` decorator directly pass that path beside the spec
  when a ``factory`` record is needed.
"""

from __future__ import annotations

import importlib

import torch
from torch import nn

from hyper_parallel.models.replacement import module_replacement


def resolve_dotted(path: str) -> object:
    """Resolve ``"pkg.mod.attr"`` — the re-import the runtime performs on records."""
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"not a dotted path: {path!r}")
    return getattr(importlib.import_module(module_name), attr)


class SourceMLP(nn.Module):
    """A tiny MLP with a single ``proj`` submodule we will replace."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)


class ReplacementMLP(nn.Module):
    """Structure-preserving replacement: reuses ``source.proj`` unchanged."""

    def __init__(self, source: SourceMLP) -> None:
        super().__init__()
        self.proj = source.proj


class MetaModel(nn.Module):
    """Compile-time meta-model: the ``model.0`` topology the manager compiles against."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(SourceMLP())


@module_replacement
def _replace_mlp(*, module: SourceMLP, module_fqn: str, context: dict) -> ReplacementMLP:
    """``@module_replacement`` factory for ``module_replacement`` records."""
    del module_fqn, context
    return ReplacementMLP(module)


def replacement_factory(*, module, module_fqn, context):
    """Plain alias of ``_replace_mlp`` usable as a YAML ``target`` callable."""
    return _replace_mlp(module=module, module_fqn=module_fqn, context=context)


def build_source() -> SourceMLP:
    """A fresh ``SourceMLP`` with known weight/bias values."""
    mlp = SourceMLP()
    with torch.no_grad():
        mlp.proj.weight.fill_(0.5)
        mlp.proj.bias.fill_(0.25)
    return mlp
