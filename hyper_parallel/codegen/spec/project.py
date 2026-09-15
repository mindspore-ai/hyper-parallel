# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Project a resolved ``TrainerConfig`` into a ``CodegenSpec``.

The projection is deliberately narrow: it only extracts the fields that can
change the generated modeling artifact, so unrelated YAML edits do not
invalidate a bundle signature.  It is NOT a second user-facing config source —
everything here derives from the training YAML.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from hyper_parallel.codegen.artifact import (
    ArtifactLayout,
    resolve_artifact_layout,
)
from hyper_parallel.codegen.spec.types import (
    CodegenSpec,
    ParallelDimsSpec,
    SourceSpec,
    TargetSpec,
    TransformSpec,
)

# Parallel dims that the generated artifact depends on.  Kept as one list so
# the spec projection and the signature cannot drift apart.
_PARALLEL_DIM_KEYS = (
    "dp_size",
    "tp_size",
    "cp_size",
    "ep_size",
    "pp_size",
    "sequence_parallel",
    "loss_parallel",
)


def project_codegen_spec(
    config: Any,
    yaml_path: str | None = None,
    *,
    layout: ArtifactLayout | None = None,
) -> CodegenSpec:
    """Project a resolved TrainerConfig into the codegen input spec.

    ``config`` may be a ``TrainerConfig`` (or any object exposing the same
    attributes) so the manager can call this without importing trainer.
    """
    if layout is None:
        if yaml_path is None:
            raise ValueError("project_codegen_spec requires yaml_path or layout")
        layout = resolve_artifact_layout(yaml_path, model_name=_model_name(config))
    source_spec = project_source_spec(config)
    return CodegenSpec(
        source=source_spec,
        parallel_dims=project_parallel_dims(config),
        overrides=project_overrides(config),
        transform_specs=project_transform_specs(config),
        target=TargetSpec(
            enabled=getattr(config, "codegen", False),
            backend=getattr(config, "modeling_backend", None),
            artifact_dir=layout.artifact_dir,
            output_modeling_name=os.path.basename(layout.modeling_path),
        ),
    )


def project_source_spec(config: Any) -> SourceSpec:
    """Project the model identity from the model target / config.

    The modeling source file is resolved later by ``source.resolver``; here
    we only record the identity the source resolver will key on.  The model
    Target holds its id under the key the caller actually used — most YAMLs and
    ``HyperAutoModel.from_pretrained`` use ``pretrained_model_name_or_path`` —
    so read that first and fall back to ``model_name_or_path``.
    """
    model = getattr(config, "model", None)
    model_name_or_path = getattr(model, "pretrained_model_name_or_path", None)
    if model_name_or_path is None:
        model_name_or_path = getattr(model, "model_name_or_path", None)
    if model_name_or_path is None:
        # Target configured without a positional model id — the HF path will
        # derive it from from_pretrained's first arg at load time.
        model_name_or_path = ""
    return SourceSpec(
        model_name_or_path=model_name_or_path,
        architecture=None,
        model_type=None,
        modeling_file=None,
        sha256=None,
        transformers_version=None,
        trust_remote_code=False,
    )


def project_parallel_dims(config: Any) -> ParallelDimsSpec:
    """Project ``accelerator.tp_size/cp_size/ep_size/pp_size`` etc."""
    accelerator = getattr(config, "accelerator", None)
    if accelerator is None:
        return ParallelDimsSpec()
    kwargs: dict[str, Any] = {}
    for key in _PARALLEL_DIM_KEYS:
        if key == "dp_size":
            kwargs[key] = _project_dp_size(accelerator)
        else:
            kwargs[key] = getattr(accelerator, key, ParallelDimsSpec().__dataclass_fields__[key].default)
    return ParallelDimsSpec(**kwargs)


def project_overrides(config: Any) -> list[dict[str, Any]]:
    """Project ``plan_overrides`` to stable canonical dicts.

    The canonicalization lives here: keep the declarative fields, resolve
    ``Target`` values to their ``to_dict()`` form, and drop ``None`` fields so
    the canonical form is stable under YAML default omissions.
    """
    entries = getattr(config, "plan_overrides", None) or []
    return [_canonicalize_override(entry) for entry in entries]


def project_transform_specs(config: Any) -> TransformSpec:
    """Project codegen-affecting model-rewrite config.

    TODO: Populate transform specs when operator, weight, and checkpoint
    transforms are supported.
    """
    model = getattr(config, "model", None)
    config_overrides = getattr(model, "config_overrides", None)
    extra = {}
    if config_overrides:
        extra["config_overrides"] = _canonicalize_value(config_overrides)
    return TransformSpec(extra_codegen_options=extra)


def normalize_model_id(path_or_name: str) -> str:
    """Normalize a model id (path or HF name) into a safe file/module token."""
    name = os.path.basename(str(path_or_name).rstrip("/\\"))
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def _model_name(config: Any) -> Optional[str]:
    """Best-effort model name for artifact file naming."""
    model = getattr(config, "model", None)
    path = getattr(model, "_target_", None)
    if isinstance(path, str):
        name = path.rsplit(".", 1)[-1]
        if name.startswith(("AutoModel", "Pretrained", "from_")):
            return None
        return name
    return getattr(config, "model_name", None)


def _project_dp_size(accelerator: Any) -> int:
    """Derive runtime DP size from ``WORLD_SIZE`` and non-DP topology."""
    explicit = getattr(accelerator, "dp_size", None)
    if explicit:
        return max(1, int(explicit))
    world_size = int(os.environ.get("WORLD_SIZE") or os.environ.get("OMPI_COMM_WORLD_SIZE") or 1)
    tp_size = max(1, int(getattr(accelerator, "tp_size", 1)))
    cp_size = max(1, int(getattr(accelerator, "cp_size", 1)))
    pp_size = max(1, int(getattr(accelerator, "pp_size", 1)))
    non_dp_size = tp_size * cp_size * pp_size
    if world_size % non_dp_size != 0:
        raise ValueError(
            f"world_size {world_size} is not divisible by non-DP size {non_dp_size}"
        )
    return max(1, world_size // non_dp_size)


def _canonicalize_override(entry: Any) -> dict[str, Any]:
    """Stable dict form of one PlanOverride entry.

    Resolves Target values instead of stashing raw ``repr()``.
    ``replace_module`` is projected so changes invalidate the signature.
    """
    result: dict[str, Any] = {"match": entry.match}
    for attr in (
        "when",
        "module_type",
        "exact_type",
        "inner_target",
        "region_dispatch",
        "tp_divide_attrs",
    ):
        value = getattr(entry, attr, None)
        if value is not None:
            result[attr] = value
    for attr in ("inner_wrapper", "replace_module", "local_compute_fn"):
        value = getattr(entry, attr, None)
        if value is not None:
            result[attr] = _canonicalize_value(value)
    for attr in ("params", "in_src", "in_dst", "out_src", "out_dst"):
        value = getattr(entry, attr, None)
        if value is not None:
            result[attr] = _canonicalize_value(value)
    return result


def _canonicalize_value(value: Any) -> Any:
    """Resolve Target objects to dict form; recurse into containers."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _canonicalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item) for item in value]
    return value


__all__ = [
    "normalize_model_id",
    "project_codegen_spec",
    "project_overrides",
    "project_parallel_dims",
    "project_source_spec",
    "project_transform_specs",
]
