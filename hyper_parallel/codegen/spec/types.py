# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""CodegenSpec — the complete input projection for one codegen run.

The spec is the single stable description of what a generated modeling file
depends on.  It is produced by :func:`project_codegen_spec` (spec.project),
hashed by ``signature_from_spec``, and consumed by the plan/emit layers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ParallelDimsSpec:
    """Parallel topology that affects the generated artifact."""

    dp_size: int = 1
    tp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1
    pp_size: int = 1
    sequence_parallel: bool = False
    loss_parallel: bool = False


@dataclass
class SourceSpec:
    """The original modeling source file the output is generated from."""

    model_name_or_path: str
    architecture: Optional[str] = None
    model_type: Optional[str] = None
    modeling_file: Optional[str] = None
    sha256: Optional[str] = None
    transformers_version: Optional[str] = None
    trust_remote_code: bool = False


@dataclass
class TargetSpec:
    """Codegen output target derived from the YAML path.

    ``artifact_dir`` is derived from the YAML path
    (``<yaml_dir>/generated/``) by the artifact layout.
    """

    enabled: bool = False
    backend: Optional[str] = None
    artifact_dir: str = ""
    output_modeling_name: Optional[str] = None


@dataclass
class TransformSpec:
    """Model-rewriting config that would affect the generated artifact.

    TODO: Populate these fields when operator, weight, and checkpoint
    transforms are supported.
    """

    operator_mappings: dict[str, Any] = field(default_factory=dict)
    weights_mapping: dict[str, Any] = field(default_factory=dict)
    checkpoint_transforms: dict[str, Any] = field(default_factory=dict)
    extra_codegen_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodegenSpec:
    """Complete input projection for one codegen run."""

    source: SourceSpec
    parallel_dims: ParallelDimsSpec
    overrides: list[dict[str, Any]] = field(default_factory=list)
    transform_specs: TransformSpec = field(default_factory=TransformSpec)
    target: TargetSpec = field(default_factory=TargetSpec)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict (for signatures and meta)."""
        return {
            "source": asdict(self.source),
            "parallel_dims": asdict(self.parallel_dims),
            "overrides": list(self.overrides),
            "transform_specs": asdict(self.transform_specs),
            "target": asdict(self.target),
        }


__all__ = [
    "CodegenSpec",
    "ParallelDimsSpec",
    "SourceSpec",
    "TargetSpec",
    "TransformSpec",
]
