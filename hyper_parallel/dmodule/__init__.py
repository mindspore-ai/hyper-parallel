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
"""Declarative distributed modules for training (M1, ``dmodule``).

Public surface for building models from nested configs, applying
:class:`~hyper_parallel.dmodule.sharding.ShardingConfig`, and registering
models for :class:`~hyper_parallel.trainer.base.BaseTrainer`.

Typical flow::

    from hyper_parallel.dmodule import BaseModel, Module, ShardingConfig
    from hyper_parallel.dmodule.types import MeshAxisName

    class Linear(Module):
        class Config(Module.Config):
            ...

    class MyModel(BaseModel):
        class Config(BaseModel.Config):
            layers: list[Linear.Config] = ...

    cfg = MyModel.Config(...)
    cfg.build().init_states()
    cfg.build().parallelize(tp_mesh)
"""

from hyper_parallel.config import Configurable
from hyper_parallel.dmodule.model import BaseModel, ModelConfigConverter
from hyper_parallel.dmodule.model_spec import ModelSpec
from hyper_parallel.dmodule.module import Module
from hyper_parallel.dmodule.sharding import (
    LocalMapConfig,
    ShardingConfig,
    resolve_placements,
)
from hyper_parallel.dmodule.types import MeshAxisName, NamedPlacement

__all__ = [
    "BaseModel",
    "Configurable",
    "LocalMapConfig",
    "MeshAxisName",
    "ModelConfigConverter",
    "ModelSpec",
    "Module",
    "NamedPlacement",
    "ShardingConfig",
    "resolve_placements",
]
