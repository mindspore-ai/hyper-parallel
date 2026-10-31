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
"""Generic construction provider for upstream Diffusers DiT models."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from types import MethodType
from typing import Any

from diffusers.configuration_utils import FrozenDict
from huggingface_hub import snapshot_download
from torch import nn


class _TrainerCompatibleFrozenDict(FrozenDict):
    """Preserve Diffusers config behavior and expose Trainer's config API."""

    def to_dict(self) -> dict[str, Any]:
        """Return a mutable copy for Trainer-side serialization."""
        return dict(self)


def normalize_diffusers_config(
    model_class: type[nn.Module],
    config: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Filter a config to arguments accepted by a Diffusers model constructor."""
    if isinstance(config, Mapping):
        config_mapping = dict(config)
    else:
        to_dict = getattr(config, "to_dict", None)
        if not callable(to_dict):
            raise TypeError("Diffusers model config must be a mapping or expose to_dict()")
        config_mapping = dict(to_dict())
    config_mapping = {
        name: value
        for name, value in config_mapping.items()
        if not name.startswith("_")
    }

    parameters = inspect.signature(model_class.__init__).parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_kwargs:
        return config_mapping
    return {
        name: value
        for name, value in config_mapping.items()
        if name != "self" and name in parameters
    }


def _finalize_preserved_nonpersistent_buffers(model: nn.Module) -> None:
    """Mark config-derived buffers preserved across materialization initialized."""
    for module in model.modules():
        initialized_buffer = False
        for name in module._non_persistent_buffers_set:  # pylint: disable=protected-access
            buffer = module._buffers.get(name)  # pylint: disable=protected-access
            if buffer is None:
                continue
            buffer._is_hf_initialized = True  # pylint: disable=protected-access
            initialized_buffer = True
        if initialized_buffer:
            module._is_hf_initialized = True  # pylint: disable=protected-access


def _adapt_diffusers_config(model: nn.Module) -> None:
    """Expose a Transformers-compatible config serialization method."""
    config = getattr(model, "config", None)
    if callable(getattr(config, "to_dict", None)):
        return
    if not isinstance(config, Mapping) or not hasattr(model, "_internal_dict"):
        raise TypeError(
            "Diffusers DiT models must expose a mapping config through _internal_dict"
        )
    model._internal_dict = _TrainerCompatibleFrozenDict(config)  # pylint: disable=protected-access


def build_diffusers_model(
    model_class: type[nn.Module],
    config: Mapping[str, Any] | Any,
    *,
    for_pretrained_loading: bool,
) -> nn.Module:
    """Construct an upstream Diffusers model with HyperParallel compatibility."""
    normalized_config = normalize_diffusers_config(model_class, config)
    model = model_class(**normalized_config)
    # Capture the upstream signature before distributed/compile wrappers can
    # obscure it. DiTTrainer uses this metadata only for keyword filtering.
    model._hp_forward_signature = inspect.signature(  # pylint: disable=protected-access
        inspect.unwrap(model.forward)
    )
    _adapt_diffusers_config(model)
    if for_pretrained_loading and not callable(getattr(model, "initialize_weights", None)):
        setattr(
            model,
            "initialize_weights",
            MethodType(_finalize_preserved_nonpersistent_buffers, model),
        )
    return model


def resolve_diffusers_component_path(
    pretrained_model_name_or_path: str | None,
    component_subfolder: str | None,
    *,
    local_files_only: bool = False,
) -> str | None:
    """Resolve a component directory from a local Diffusers tree or Hub repo."""
    if pretrained_model_name_or_path is None:
        return None

    path = Path(pretrained_model_name_or_path).expanduser()
    if path.is_dir() and (path / "config.json").is_file():
        return str(path)
    if path.is_dir() and component_subfolder:
        component_path = path / component_subfolder
        if component_path.is_dir():
            return str(component_path)

    if not component_subfolder:
        return pretrained_model_name_or_path

    normalized = pretrained_model_name_or_path.rstrip("/").replace("\\", "/")
    suffix = f"/{component_subfolder}"
    repo_id = normalized[: -len(suffix)] if normalized.endswith(suffix) else normalized
    if "/" not in repo_id or Path(repo_id).exists():
        return pretrained_model_name_or_path

    snapshot_path = Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[
                f"{component_subfolder}/config.json",
                f"{component_subfolder}/*.safetensors",
                f"{component_subfolder}/*.safetensors.index.json",
            ],
            local_files_only=local_files_only,
        )
    )
    return str(snapshot_path / component_subfolder)


def _resolve_config_source(
    config_path: str | None,
    component_path: str | None,
) -> str:
    """Select an explicit config source or the resolved component directory."""
    if config_path is not None:
        return config_path
    if component_path is None:
        raise ValueError(
            "Diffusers DiT models require config_path when pretrained_model_name_or_path is None"
        )
    return component_path


class DiffusersDiTModelProvider:
    """Default provider used through the existing ``ModelAdapterSpec.dit`` slot."""

    model_class: type[nn.Module] | None = None

    @classmethod
    def _model_class(cls) -> type[nn.Module]:
        if cls.model_class is None:
            raise TypeError(f"{cls.__name__} must define model_class")
        return cls.model_class

    @classmethod
    def load_config(cls, config_source: str, **overrides: Any) -> dict[str, Any]:
        """Load a standard Diffusers config and apply constructor overrides."""
        path = Path(config_source).expanduser()
        if path.is_file():
            with path.open("r", encoding="utf-8") as config_file:
                config = json.load(config_file)
        else:
            model_class = cls._model_class()
            load_config = getattr(model_class, "load_config", None)
            if not callable(load_config):
                raise TypeError(
                    f"{model_class.__name__} must expose load_config() or use a config JSON file"
                )
            config = load_config(config_source)
        config = dict(config)
        config.update(overrides)
        return config

    @classmethod
    def validate_pretrained_options(cls, *, task: str | None, **kwargs: Any) -> None:
        """Validate family-specific load options before resolving model files."""
        del task, kwargs

    @classmethod
    def resolve_pretrained(
        cls,
        pretrained_model_name_or_path: str | None,
        *,
        transformer_subfolder: str | None,
        config_path: str | None,
        task: str | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], str | None]:
        """Resolve config and checkpoint directory for an upstream DiT model."""
        cls.validate_pretrained_options(task=task, **kwargs)
        local_files_only = bool(kwargs.pop("local_files_only", False))
        component_path = resolve_diffusers_component_path(
            pretrained_model_name_or_path,
            transformer_subfolder,
            local_files_only=local_files_only,
        )
        config_source = _resolve_config_source(config_path, component_path)
        config = cls.load_config(config_source, **kwargs)
        return normalize_diffusers_config(cls._model_class(), config), component_path

    @classmethod
    def build_model(cls, config: Mapping[str, Any] | Any) -> nn.Module:
        """Build a model for deferred pretrained checkpoint loading."""
        return build_diffusers_model(
            cls._model_class(),
            config,
            for_pretrained_loading=True,
        )

    @classmethod
    def build_from_config(cls, config: Mapping[str, Any] | Any, **kwargs: Any) -> nn.Module:
        """Build a model whose state will be initialized rather than loaded."""
        del kwargs
        return build_diffusers_model(
            cls._model_class(),
            config,
            for_pretrained_loading=False,
        )


__all__ = [
    "DiffusersDiTModelProvider",
    "build_diffusers_model",
    "normalize_diffusers_config",
    "resolve_diffusers_component_path",
]
