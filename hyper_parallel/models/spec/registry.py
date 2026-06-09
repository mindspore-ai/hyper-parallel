# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Global ModelSpec registry.

Models register themselves via ``register_spec`` at import time.
The trainer looks up a spec by ``cfg.model.name`` via ``get_spec``.
"""
from hyper_parallel.models.spec.model_spec import ModelSpec

_SPEC_REGISTRY: dict[str, ModelSpec] = {}


def register_spec(name: str, spec: ModelSpec) -> None:
    """Register a ModelSpec under the given name.

    Args:
        name: Unique name (must match YAML ``model.name``).
        spec: The ModelSpec instance.

    Raises:
        ValueError: If the name is already registered.
    """
    if name in _SPEC_REGISTRY:
        raise ValueError(
            f"ModelSpec '{name}' is already registered. "
            f"Existing: {_SPEC_REGISTRY[name]}, new: {spec}"
        )
    _SPEC_REGISTRY[name] = spec


def get_spec(name: str) -> ModelSpec:
    """Look up a registered ModelSpec by name.

    Args:
        name: The model name from YAML ``model.name``.

    Returns:
        The registered ModelSpec.

    Raises:
        ValueError: If the name is not found. Lists available names.
    """
    if name not in _SPEC_REGISTRY:
        available = sorted(_SPEC_REGISTRY.keys())
        raise ValueError(
            f"ModelSpec '{name}' not found. Available: {available}. "
            f"Make sure the model's __init__.py is imported before training."
        )
    return _SPEC_REGISTRY[name]
