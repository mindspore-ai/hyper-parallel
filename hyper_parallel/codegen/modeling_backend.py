# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Three-state modeling backend resolution: ``hf`` / ``custom`` / ``gen``.

``HF`` is wired through the trainer, while ``GEN`` loads a generated artifact.
``CUSTOM`` falls back to HF when no matching implementation is registered.

This module deliberately imports nothing from ``transformers`` at module
scope: ``hf_config`` is duck-typed on ``architectures``, which keeps the
resolver importable (and testable) in environments without torch installed.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ModelingBackend(str, Enum):
    """Which implementation builds the model.

    ``str``-mixin so a raw ``"gen"`` from YAML compares equal to
    ``ModelingBackend.GEN`` without callers having to convert first.
    """

    HF = "hf"
    CUSTOM = "custom"
    GEN = "gen"


def is_custom_model_available(hf_config: Any) -> bool:
    """Whether ``MODEL_ARCH_MAPPING`` resolves this architecture to a custom class.

    Imported lazily: ``registry`` pulls in ``transformers``, and this module is
    meant to stay importable without it.
    """
    architectures = getattr(hf_config, "architectures", None) or []
    if not architectures:
        return False
    try:
        from hyper_parallel.models.registry import _resolve_custom_model_cls
    except ImportError:  # pragma: no cover - no transformers in this env
        return False
    return _resolve_custom_model_cls(architectures[0]) is not None


def resolve_modeling_backend(
    hf_config: Any,
    *,
    force_hf: bool = False,
    codegen_enabled: bool = False,
) -> ModelingBackend:
    """Resolve the backend from ``force_hf`` and the ``codegen`` switch.

    Precedence, highest first:

    1. ``force_hf=True`` — an escape hatch, so it wins over everything.
    2. ``codegen=True`` implies ``GEN`` — turning codegen on is what makes the
       artifact authoritative; a second selector would just be a way to get
       them out of sync.
    3. A custom implementation registered for this architecture.
    4. ``HF``.

    Args:
        hf_config: The ``PretrainedConfig``; only ``architectures`` is read.
        force_hf: Force the HF-native path regardless of everything else.
        codegen_enabled: The ``TrainerConfig.codegen`` switch.
    """
    if force_hf:
        resolved = ModelingBackend.HF
    elif codegen_enabled:
        resolved = ModelingBackend.GEN
    elif is_custom_model_available(hf_config):
        resolved = ModelingBackend.CUSTOM
    else:
        resolved = ModelingBackend.HF

    if force_hf and codegen_enabled:
        logger.warning(
            "codegen is enabled but force_hf=True, so the generated modeling "
            "file will not be used; drop force_hf to build from the artifact",
        )
    return resolved


__all__ = [
    "ModelingBackend",
    "is_custom_model_available",
    "resolve_modeling_backend",
]
