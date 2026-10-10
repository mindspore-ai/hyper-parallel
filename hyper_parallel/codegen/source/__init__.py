# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Source resolution for codegen bundles.

Locate the real ``modeling_*.py`` (installed transformers or checkpoint remote
code) that a generated modeling file is produced from.
"""
from hyper_parallel.codegen.source.compat import sanitize_source_compat
from hyper_parallel.codegen.source.resolver import (
    ResolvedSource,
    import_module_file,
    resolve_hf_source,
    resolve_model_source,
    resolve_remote_code_file,
    resolve_remote_siblings,
    resolve_source_identity,
)

__all__ = [
    "ResolvedSource",
    "import_module_file",
    "resolve_hf_source",
    "resolve_model_source",
    "resolve_remote_code_file",
    "resolve_remote_siblings",
    "resolve_source_identity",
    "sanitize_source_compat",
]
