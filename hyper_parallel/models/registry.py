# Copyright 2025-2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""registry: model discovery — architecture → model entry / adapter provider.

Owns ``MODEL_ARCH_MAPPING`` (HF architecture → custom model class, moved
here from ``_transformers/registry.py`` in M1) and the ``ModelAdapterSpec``
registry. This module is a Models-shared mechanism (adjust doc §7.2):

- no Trainer/Data/Optimizer configuration is stored here;
- ``recipes/train.yaml`` is never parsed here;
- CP/EP schemes are never guessed from class names;
- HF config resolution stays in ``_transformers`` (config resolver);
- native models may register a model class directly, without an HF
  ``PretrainedConfig``.

Importing this module must not import Trainer/Data or any concrete model
package. Family registrations are discovered automatically by directory
convention: ``models/<family>/adapter/registration.py`` existing (and
self-registering on import) is all a new family needs — this module is
never edited when a family is added. Discovery itself is a filesystem scan
(no imports); the family's registration module is imported lazily on first
lookup.
"""

import importlib
import logging
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from hyper_parallel.models.adapter_spec import ModelAdapterSpec

logger = logging.getLogger(__name__)

# OrderedDict: arch_name → (module_path, class_name)
# Lazy-loaded — only imported on first access.
MODEL_ARCH_MAPPING = OrderedDict([])


@lru_cache(maxsize=128)
def _resolve_custom_model_cls(arch_name: str) -> Optional[type]:
    """Lazy-load model class from MODEL_ARCH_MAPPING.

    Returns None → fall back to HF native.
    """
    entry = MODEL_ARCH_MAPPING.get(arch_name)
    if entry is None:
        return None
    module_path, class_name = entry[0], entry[1]
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.warning(
            "Failed to load custom model %s from %s: %s. Falling back to HF native.",
            class_name, module_path, e,
        )
        return None


# ────────────────────────────────────────────────────────────────────────────
# ModelAdapterSpec registry (model_type → spec); families self-register from
# their adapter/registration.py, discovered automatically by directory
# convention (models/<family>/adapter/registration.py).
# ────────────────────────────────────────────────────────────────────────────
MODEL_ADAPTER_REGISTRY = {}

# normalized lookup key (see _normalize_family_key) → spec.model_type
_FAMILY_ALIASES: Dict[str, str] = {}

# Cross-family aliases: normalized lookup key → family directory whose
# registration module also registers this family's spec. Needed only when one
# family shares another family's adapter (DeepSeek-V2 reuses DeepSeek-V3's MLA
# sharding rules); new families with their own directory never appear here.
_FAMILY_DIR_ALIASES = {
    "deepseekv2": "deepseek_v3",
}

# normalized family directory name → registration module path; built lazily.
_DISCOVERED_PROVIDERS: Optional[Dict[str, str]] = None

_ARCH_SUFFIXES = (
    "forcausallm", "forconditionalgeneration",
    "forsequenceclassification", "forimagetexttotext",
)


def _normalize_family_key(name: str) -> str:
    """Canonical lookup key: lowercase, separators removed, HF task suffix
    stripped — so the architectures spelling (``deepseekv3``), the model_type
    spelling (``deepseek_v3``) and the class spelling
    (``DeepseekV3ForCausalLM``) all resolve to the same family."""
    key = name.lower().replace("_", "").replace("-", "")
    for suffix in _ARCH_SUFFIXES:
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    return key


def _discover_family_providers() -> Dict[str, str]:
    """Scan ``models/*/adapter/registration.py`` once (filesystem only — no
    family code is imported)."""
    global _DISCOVERED_PROVIDERS
    if _DISCOVERED_PROVIDERS is None:
        providers = {}
        models_dir = Path(__file__).resolve().parent
        for child in sorted(models_dir.iterdir()):
            if (
                    child.is_dir()
                    and child.name.isidentifier()
                    and not child.name.startswith("_")
                    and (child / "adapter" / "registration.py").is_file()
            ):
                providers[_normalize_family_key(child.name)] = (
                    f"{__package__}.{child.name}.adapter.registration"
                )
        _DISCOVERED_PROVIDERS = providers
    return _DISCOVERED_PROVIDERS


def register_model_adapter(spec: ModelAdapterSpec) -> None:
    """Register one family's adapter spec (idempotent; conflicts fail fast)."""
    existing = MODEL_ADAPTER_REGISTRY.get(spec.model_type)
    if existing is not None and existing != spec:
        raise ValueError(
            f"conflicting adapter spec for model_type {spec.model_type!r}: "
            f"{existing!r} vs {spec!r}"
        )
    MODEL_ADAPTER_REGISTRY[spec.model_type] = spec
    for alias in (
            _normalize_family_key(spec.model_type),
            _normalize_family_key(spec.architecture),
    ):
        _FAMILY_ALIASES.setdefault(alias, spec.model_type)


def get_model_adapter(model_type: str) -> Optional[ModelAdapterSpec]:
    """Return the family's adapter spec, lazy-importing its registration.

    Accepts any spelling — model_type (``deepseek_v3``), HF architecture
    (``DeepseekV3ForCausalLM``) or the canonical arch name used by the
    planner (``deepseekv3``).
    """
    spec = MODEL_ADAPTER_REGISTRY.get(model_type)
    if spec is None:
        canonical = _FAMILY_ALIASES.get(_normalize_family_key(model_type))
        if canonical is not None:
            spec = MODEL_ADAPTER_REGISTRY.get(canonical)
    if spec is None:
        key = _normalize_family_key(model_type)
        providers = _discover_family_providers()
        family_dir = _FAMILY_DIR_ALIASES.get(key)
        module_path = providers.get(
            key if family_dir is None else _normalize_family_key(family_dir)
        )
        if module_path is not None:
            # Importing the registration module self-registers the family.
            importlib.import_module(module_path)
            spec = MODEL_ADAPTER_REGISTRY.get(model_type)
        if spec is None:
            canonical = _FAMILY_ALIASES.get(key)
            if canonical is not None:
                spec = MODEL_ADAPTER_REGISTRY.get(canonical)
    return spec
