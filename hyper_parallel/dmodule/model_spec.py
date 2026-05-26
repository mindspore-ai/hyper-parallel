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
"""Per-model registration bundle for the training stack.

Maps ``model.name`` in trainer YAML to construction and parallel hooks.
See :mod:`hyper_parallel.models.spec.registry` for ``register_spec`` / ``get_spec``.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Type

from hyper_parallel.dmodule.model import BaseModel


@dataclass
class ModelSpec:
    """Per-model registration bundle.

    Either ``build_model_fn`` (factory function) or ``model``
    (:class:`~hyper_parallel.dmodule.model.BaseModel.Config`) must be set,
    but not both. Optional hooks customize parallelize, grad clip, pipeline,
    and checkpoint key mapping.

    Args:
        name: Unique model id (matches trainer config ``model.name``).
        build_model_fn: ``(trainer_cfg) -> Module`` factory used by the trainer
            today in :mod:`hyper_parallel.models.spec`.
        model: Declarative :class:`BaseModel.Config` tree; built via
            ``spec.model.build()`` when the trainer supports this path.
        parallelize_fn: ``(model, mesh, trainer_cfg) -> model`` custom
            parallelization (required for current trainer).
        clip_grad_fn: Optional custom gradient clipping.
        pipelining_fn: Optional pipeline-parallel setup.
        state_dict_adapter: Optional checkpoint key translation class.

    Example:
        Declarative config::

            model_cfg = MyModel.Config(num_layers=12)
            spec = ModelSpec(
                name="my_model",
                model=model_cfg,
                parallelize_fn=parallelize_my_model,
            )

        Factory function (current trainer default)::

            def build_my_model(trainer_cfg):
                return MyModel(MyModel.Config(hidden=trainer_cfg.hidden_size))

            spec = ModelSpec(
                name="my_model",
                build_model_fn=build_my_model,
                parallelize_fn=parallelize_my_model,
            )

        Register before training::

            from hyper_parallel.models.spec import register_spec

            register_spec("my_model", spec)
    """

    name: str
    build_model_fn: Optional[Callable] = None
    model: Optional[BaseModel.Config] = None
    parallelize_fn: Optional[Callable] = None
    clip_grad_fn: Optional[Callable] = None
    pipelining_fn: Optional[Callable] = None
    state_dict_adapter: Optional[Type] = None

    def __post_init__(self) -> None:
        """Validate that exactly one model constructor is configured.

        Raises:
            ValueError: If both or neither of ``build_model_fn`` and ``model``
                are set.
        """
        if self.build_model_fn is None and self.model is None:
            raise ValueError(
                f"ModelSpec '{self.name}' requires build_model_fn or model config"
            )
        if self.build_model_fn is not None and self.model is not None:
            raise ValueError(
                f"ModelSpec '{self.name}' must not set both build_model_fn and model"
            )


__all__ = ["ModelSpec"]
