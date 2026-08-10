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
"""Resolve one logical model name into training and rollout configuration."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelRegistration:
    name: str
    hyper_model_name: str
    weights_path: str
    tokenizer_path: str


class ModelRegistry:
    """Store logical model registrations shared by training and rollout."""

    def __init__(self) -> None:
        """Initialize an empty model registry."""
        self._models: dict[str, ModelRegistration] = {}

    def register(self, model: ModelRegistration) -> ModelRegistration:
        """Register a model or return its identical existing registration."""
        if not model.name:
            raise ValueError("model.registry_name must be non-empty")
        existing = self._models.get(model.name)
        if existing is not None and existing != model:
            raise ValueError(f"Conflicting model registration: {model.name}")
        self._models[model.name] = model
        return model

    def get(self, name: str) -> ModelRegistration:
        """Return a model registration by its logical name."""
        try:
            return self._models[name]
        except KeyError as error:
            raise ValueError(
                f"Unknown model registration '{name}'; available={sorted(self._models)}"
            ) from error

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered model names in deterministic order."""
        return tuple(sorted(self._models))


MODEL_REGISTRY = ModelRegistry()


def register_configured_model(config: Mapping[str, Any]) -> ModelRegistration:
    """Register the YAML model once and reuse it for both engine families."""
    registry_name = config.get("registry_name")
    if not isinstance(registry_name, str) or not registry_name:
        raise ValueError("model.registry_name must be a non-empty string")
    return MODEL_REGISTRY.register(
        ModelRegistration(
            name=registry_name,
            hyper_model_name=str(config["name"]),
            weights_path=str(config["weights_path"]),
            tokenizer_path=str(config["tokenizer_path"]),
        )
    )
