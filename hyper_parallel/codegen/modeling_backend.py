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
from typing import Any, Optional

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


def warn_backend_override(
    resolved: ModelingBackend,
    *,
    requested: Optional[str] = None,
    force_hf: bool = False,
    codegen_enabled: bool = False,
) -> None:
    """Log the combinations where one input silently overrode another.

    These are configuration mistakes that would otherwise show up much later as
    "why is my generated file not being used" — say it at resolve time.
    """
    if force_hf and requested is not None and requested != ModelingBackend.HF:
        logger.warning(
            "modeling_backend=%r was overridden to 'hf' by force_hf=True", requested,
        )
    if force_hf and codegen_enabled:
        logger.warning(
            "codegen is enabled but force_hf=True, so the generated modeling "
            "file will not be used; drop force_hf to build from the artifact",
        )
    if codegen_enabled and resolved is not ModelingBackend.GEN and not force_hf:
        logger.warning(
            "codegen is enabled but modeling_backend resolved to %r; the "
            "generated artifact will be built but not loaded",
            resolved.value,
        )


def resolve_modeling_backend(
    hf_config: Any,
    *,
    modeling_backend: Optional[str] = None,
    force_hf: bool = False,
    codegen_enabled: bool = False,
) -> ModelingBackend:
    """Resolve the backend from the explicit setting, ``force_hf``, and ``codegen``.

    Precedence, highest first:

    1. ``force_hf=True`` — an escape hatch, so it wins over everything.
    2. An explicit ``modeling_backend`` from the YAML.
    3. ``codegen=True`` implies ``GEN`` — turning codegen on is what makes the
       artifact authoritative; requiring a second field would just be a way to
       get them out of sync.
    4. A custom implementation registered for this architecture.
    5. ``HF``.

    Args:
        hf_config: The ``PretrainedConfig``; only ``architectures`` is read.
        modeling_backend: Explicit ``"hf"``/``"custom"``/``"gen"``, or None.
        force_hf: Force the HF-native path regardless of everything else.
        codegen_enabled: The ``TrainerConfig.codegen`` switch.
    """
    requested = modeling_backend

    if force_hf:
        resolved = ModelingBackend.HF
    elif modeling_backend is not None:
        try:
            resolved = ModelingBackend(modeling_backend)
        except ValueError as exc:
            valid = ", ".join(repr(member.value) for member in ModelingBackend)
            raise ValueError(
                f"unknown modeling_backend {modeling_backend!r}; expected one of {valid}"
            ) from exc
    elif codegen_enabled:
        resolved = ModelingBackend.GEN
    elif is_custom_model_available(hf_config):
        resolved = ModelingBackend.CUSTOM
    else:
        resolved = ModelingBackend.HF

    warn_backend_override(
        resolved,
        requested=requested,
        force_hf=force_hf,
        codegen_enabled=codegen_enabled,
    )
    return resolved


__all__ = [
    "ModelingBackend",
    "is_custom_model_available",
    "resolve_modeling_backend",
    "warn_backend_override",
]
