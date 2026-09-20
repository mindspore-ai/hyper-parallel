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
"""Model-assets, dataset and dataloader configuration sections.

Split from ``auto_models/trainer/config.py`` in stage 7 (05 §15.2.5);
class names, fields and defaults are unchanged.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

from hyper_parallel.trainer.config.target import Target, _serialize_config_value


@dataclass
class ModelAssetsConfig:
    """Tokenizer and chat-template configuration for text datasets."""

    chat_template: Optional[Union[str, Target[Any]]] = None
    tokenizer: Optional[Target[Any]] = None


@dataclass
class DatasetConfig:
    """Dataset target with its model assets and sample transform."""

    target: Target[Any]
    model_assets: Union[ModelAssetsConfig, Target[Any]] = field(default_factory=ModelAssetsConfig)
    data_transform: Optional[Target[Any]] = None

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
        return config


@dataclass
class DataLoaderConfig:
    """DataLoader target and its text-batch assembly components."""

    target: Target[Any]
    batch_adapter: Optional[Target[Any]] = None
    collate_fn: Optional[Target[Any]] = None
    get_batch: Optional[Target[Any]] = None
    sampler_type: Literal["single", "cyclic"] = "single"
    data_rearrange_map: Any = None
    data_sharding: bool = False
    use_background_prefetcher: bool = False

    def build(self, **runtime_kwargs: Any) -> Any:
        """Build the DataLoader target with runtime Dataset arguments."""
        return self.target.build(**runtime_kwargs)

    def __getattr__(self, name: str) -> Any:
        """Expose configured DataLoader options through the wrapped target."""
        return getattr(self.target, name)

    def to_dict(self) -> dict[str, Any]:
        """Serialize components in their compact nested YAML shape."""
        config = self.target.to_dict()
        config["batch_adapter"] = _serialize_config_value(self.batch_adapter)
        config["collate_fn"] = _serialize_config_value(self.collate_fn)
        config["get_batch"] = _serialize_config_value(self.get_batch)
        config["sampler_type"] = self.sampler_type
        config["data_rearrange_map"] = _serialize_config_value(self.data_rearrange_map)
        config["data_sharding"] = self.data_sharding
        config["use_background_prefetcher"] = self.use_background_prefetcher
        return config
