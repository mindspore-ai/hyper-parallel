# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Codegen spec — the stable projection of a training YAML into codegen inputs.

``project_codegen_spec`` (spec.project) turns a resolved ``TrainerConfig`` into
a ``CodegenSpec``; the spec is what the signature hashes over and what the
plan/emit layers consume.  The spec must only carry fields that actually affect
the generated modeling artifact, so that unrelated YAML fields cannot force a
regeneration.
"""
from hyper_parallel.codegen.spec.project import project_codegen_spec
from hyper_parallel.codegen.spec.types import (
    CodegenSpec,
    ParallelDimsSpec,
    SourceSpec,
    TargetSpec,
    TransformSpec,
)

__all__ = [
    "CodegenSpec",
    "ParallelDimsSpec",
    "SourceSpec",
    "TargetSpec",
    "TransformSpec",
    "project_codegen_spec",
]
