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
"""Whole-model protocol: :class:`BaseModel` and :class:`ModelConfigConverter`."""

from abc import abstractmethod
from dataclasses import dataclass

from hyper_parallel.config import Configurable
from hyper_parallel.dmodule.module import Module


class ModelConfigConverter(Configurable):
    """Transform a model config tree before :meth:`BaseModel.Config.build`.

    Subclasses implement :meth:`convert` to walk or rewrite nested
    :class:`~hyper_parallel.config.configurable.Configurable.Config` nodes
    (typically via :meth:`~hyper_parallel.config.configurable.Configurable.Config.traverse`).

    Example:
        Run ``convert`` on the model config, then ``build`` the model::

            model_cfg = MyModel.Config(num_layers=4)
            MyConverter.Config(enabled=True).build().convert(model_cfg)
            model = model_cfg.build()

        Minimal ``convert`` implementation::

            class MyConverter(ModelConfigConverter):
                @dataclass(kw_only=True, slots=True)
                class Config(ModelConfigConverter.Config):
                    enabled: bool = True

                def convert(self, model_config: Configurable.Config) -> None:
                    for _fqn, cfg, parent, attr in model_config.traverse(Module.Config):
                        new_cfg = cfg  # replace with another Config type as needed
                        setattr(parent, attr, new_cfg)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Converter hyperparameters (subclass adds fields such as ``rank``)."""

    @abstractmethod
    def convert(self, model_config: Configurable.Config) -> None:
        """Rewrite *model_config* in place.

        Args:
            model_config: Root :class:`BaseModel.Config` (or compatible tree)
                to modify before ``build()``.

        Example::

            LoRAConverter.Config(rank=16).build().convert(llama_cfg)
        """
        raise NotImplementedError


class BaseModel(Module):
    """Base class for declarative whole-model components.

    Subclasses define ``Config(BaseModel.Config)`` and assemble child
    :class:`~hyper_parallel.dmodule.module.Module` layers. Register with
    :class:`~hyper_parallel.dmodule.model_spec.ModelSpec` via ``model=...Config``.

    Example::

        @dataclass(kw_only=True, slots=True)
        class MyModelConfig(BaseModel.Config):
            hidden: int = 128

            def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
                return sum(p.numel() for p in model.parameters()), 0

        class MyModel(BaseModel):
            def __init__(self, config: MyModelConfig):
                super().__init__()
                self.config = config

        cfg = MyModelConfig(hidden=256)
        cfg.update_from_config(trainer_config=trainer_cfg)
        model = cfg.build()
        model.init_states()
        model.verify_module_protocol()
    """

    def init_weights(self, **kwargs) -> None:
        """Initialize parameters and buffers (alias for :meth:`init_states`).

        Args:
            **kwargs: Forwarded to :meth:`init_states` (for example
                ``buffer_device`` for RoPE-style buffers).

        Example::

            model.init_weights(buffer_device=torch.device("npu", 0))
        """
        buffer_device = kwargs.get("buffer_device")
        self.init_states(buffer_device=buffer_device)

    def verify_module_protocol(self) -> None:
        """Ensure every submodule is a :class:`~hyper_parallel.dmodule.module.Module`.

        Raises:
            RuntimeError: If any ``named_modules()`` entry is not a
                :class:`Module` instance.

        Example::

            model = cfg.build()
            model.verify_module_protocol()  # before parallelize / training
        """
        failures: list[tuple[str, str]] = []
        for fqn, mod in self.named_modules():
            if not isinstance(mod, Module):
                failures.append((fqn, type(mod).__name__))
        if failures:
            details = ", ".join(f"'{fqn}' ({cls})" for fqn, cls in failures)
            raise RuntimeError(
                f"The following modules do not satisfy the Module protocol: {details}"
            )

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Base config for whole models (extends :class:`Module.Config`)."""

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            """Inject trainer-derived fields into this config before ``build``.

            Override in model configs to copy parallelism flags, sharding specs,
            or other values from the trainer configuration.

            Args:
                trainer_config: Trainer or job configuration object.
                **kwargs: Optional extra inputs for specific models.

            Example::

                cfg = LlamaModel.Config(num_layers=32)
                cfg.update_from_config(trainer_config=trainer_cfg)
                model = cfg.build()
            """
            del trainer_config, kwargs

        def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
            """Return parameter count and estimated FLOPs for logging.

            Args:
                model: Built model instance.
                seq_len: Sequence length used for FLOPs estimation.

            Returns:
                ``(num_params, num_flops)``.

            Raises:
                NotImplementedError: If the concrete model config does not
                    override this method.

            Example::

                nparams, nflops = cfg.get_nparams_and_flops(model, seq_len=4096)
            """
            del model, seq_len
            raise NotImplementedError(
                f"{type(self).__name__} must implement get_nparams_and_flops"
            )


__all__ = ["BaseModel", "ModelConfigConverter"]
