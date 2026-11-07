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
"""Typed dataset components, DataLoader components, and Loader runtime options."""

import inspect
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Optional, Union

from hyper_parallel.trainer.config.target import Target, _serialize_config_value


_SAMPLER_OPTIONS = {"sampler_type", "data_rearrange_map", "data_sharding", "use_background_prefetcher"}


@dataclass
class ModelAssetsConfig:
    """Tokenizer and chat-template configuration for text datasets."""

    chat_template: Optional[Union[str, Target[Any]]] = None
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    tokenizer: Optional[Target[Any]] = None


@dataclass
class DatasetConfig:
    """Dataset target with its model assets and sample transform."""

    target: Target[Any]
    model_assets: Union[ModelAssetsConfig, Target[Any]] = field(default_factory=ModelAssetsConfig)
    data_transform: Optional[Target[Any]] = None
    training_mode: Optional[Literal["sft", "pretrain"]] = None

    def build(self, **runtime_kwargs: Any) -> Any:
        """Build the Dataset target with runtime Trainer arguments."""
        dataset = self.target.build(**runtime_kwargs)
        return dataset

    def __getattr__(self, name: str) -> Any:
        """Expose configured Dataset options through the wrapped target."""
        return getattr(self.target, name)

    def to_dict(self) -> dict[str, Any]:
        """Serialize Dataset components in their compact nested YAML shape."""
        config = self.target.to_dict()
        config["model_assets"] = _serialize_config_value(self.model_assets)
        config["data_transform"] = _serialize_config_value(self.data_transform)
        if self.training_mode is not None:
            config["training_mode"] = self.training_mode
        return config


@dataclass
class LoaderConfig:
    """Sampling, dynamic batching, and source-loading options.

    The transform's ``max_seq_len`` and training micro batch size determine the
    packing and padding limits. ``drop_last`` applies to the Mapping sampler's
    global source blocks, rather than the packed token sequence.
    """

    sampler_type: Literal["single", "cyclic"] = "single"
    repeat: bool = True
    drop_last: bool = True
    min_buffered_samples: int = 200
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: Optional[int] = None
    data_rearrange_map: Optional[Any] = None
    data_sharding: bool = False
    use_background_prefetcher: bool = False
    save_by_idx: Optional[bool] = None

    def __post_init__(self) -> None:
        """Reject invalid limits before workers or distributed groups are used."""
        if self.sampler_type not in ("single", "cyclic"):
            raise ValueError("sampler_type must be 'single' or 'cyclic'")
        for name in ("repeat", "drop_last", "pin_memory", "data_sharding", "use_background_prefetcher", "save_by_idx"):
            value = getattr(self, name)
            if name == "save_by_idx" and value is None:
                continue
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("min_buffered_samples", "num_workers", "prefetch_factor"):
            value = getattr(self, name)
            if value is None and name == "prefetch_factor":
                continue
            minimum = 0 if name == "num_workers" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass
class DataLoaderConfig:
    """DataLoader implementation, typed options, and batch assembly components."""

    target: Target[Any]
    collate_fn: Optional[Target[Any]] = None
    get_batch: Optional[Target[Any]] = None
    loader_config: Optional[LoaderConfig] = None

    def __post_init__(self) -> None:
        """Retain options from programmatically constructed legacy targets."""
        if self.loader_config is None:
            options = {
                item.name: getattr(self.target, item.name)
                for item in fields(LoaderConfig) if hasattr(self.target, item.name)
            }
            self.loader_config = LoaderConfig(**options)
        supported = self.supported_loader_options
        unsupported = [
            item.name for item in fields(LoaderConfig)
            if item.name not in supported and getattr(self.loader_config, item.name) != item.default
        ]
        if unsupported:
            raise ValueError(f"selected Loader does not support options: {unsupported}")

    def build(self, **runtime_kwargs: Any) -> Any:
        """Apply Loader options; runtime split policy and topology take precedence."""
        sampler_only = {"data_rearrange_map", "data_sharding", "use_background_prefetcher"}
        options = {
            item.name: getattr(self.loader_config, item.name)
            for item in fields(LoaderConfig) if item.name not in sampler_only
        }
        dataloader = self.target.build(**{**options, **runtime_kwargs})
        return dataloader

    @property
    def supported_loader_options(self) -> set[str]:
        """Return options handled by this target or the framework's sampler."""
        option_names = {item.name for item in fields(LoaderConfig)}
        parameters = inspect.signature(self.target.callable).parameters
        if any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
            return option_names
        return option_names & (set(parameters) | _SAMPLER_OPTIONS)

    def __getattr__(self, name: str) -> Any:
        """Keep legacy option reads working while storing options in one place."""
        loader_config = object.__getattribute__(self, "loader_config")
        if loader_config is not None and name in {item.name for item in fields(LoaderConfig)}:
            return getattr(loader_config, name)
        return getattr(self.target, name)

    def to_dict(self) -> dict[str, Any]:
        """Serialize components in their compact nested YAML shape."""
        option_names = {item.name for item in fields(LoaderConfig)}
        config = {name: value for name, value in self.target.to_dict().items() if name not in option_names}
        config["collate_fn"] = _serialize_config_value(self.collate_fn)
        config["get_batch"] = _serialize_config_value(self.get_batch)
        loader_options = _serialize_config_value(self.loader_config)
        supported = self.supported_loader_options
        config["loader_config"] = {name: value for name, value in loader_options.items() if name in supported}
        return config
