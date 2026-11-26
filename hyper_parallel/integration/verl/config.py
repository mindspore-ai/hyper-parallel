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
"""VeRL configuration for HyperParallel."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf
from verl.trainer.config import CheckpointConfig
from verl.workers.config.actor import ActorConfig
from verl.workers.config.critic import CriticConfig
from verl.workers.config.engine import EngineConfig
from verl.workers.config.optimizer import OptimizerConfig

from hyper_parallel.trainer.config.resolver import resolve_config
from hyper_parallel.trainer.config.trainer import TrainerConfig

if TYPE_CHECKING:
    from verl.workers.config.engine import EngineRouterReplayConfig


logger = logging.getLogger(__name__)


@dataclass
class HyperParallelCheckpointConfig(CheckpointConfig):
    """Limit VeRL checkpoints to synchronous full-state saves and loads."""

    if TYPE_CHECKING:
        strict: bool
        save_lora_only: bool

    def __post_init__(self) -> None:
        full_state = {"model", "optimizer", "extra"}
        if set(self.save_contents or ()) != full_state or set(self.load_contents or ()) != full_state:
            raise NotImplementedError(
                "VeRL checkpoint save_contents and load_contents must include model, optimizer and extra"
            )

        if (self.async_save, self.strict, self.save_lora_only) != (False, True, False):
            raise NotImplementedError("VeRL checkpoint requires async_save=False, strict=True and save_lora_only=False")


@dataclass
class HyperParallelActorConfig(ActorConfig):
    """VeRL policy settings for the HyperParallel backend."""

    strategy: str = "hyper_parallel"
    calculate_sum_pi_squared: bool = False
    use_fused_kernels: bool = False
    use_prefix_grouper: bool = False
    use_torch_compile: bool = False
    checkpoint: HyperParallelCheckpointConfig = field(default_factory=HyperParallelCheckpointConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.strategy != "hyper_parallel":
            raise ValueError("HyperParallelActorConfig requires strategy=hyper_parallel")
        for name in ("calculate_sum_pi_squared", "use_fused_kernels", "use_prefix_grouper", "use_torch_compile"):
            if getattr(self, name):
                raise NotImplementedError(f"HyperParallelActorConfig does not support {name}=True")


@dataclass
class HyperParallelCriticConfig(CriticConfig):
    """Reject critic use before VeRL creates a value-model worker."""

    strategy: str = "hyper_parallel"

    def __post_init__(self) -> None:
        """Fail during config instantiation while critic support is unavailable."""
        raise NotImplementedError("HyperParallel does not support VeRL critic yet")


@dataclass
class HyperParallelEngineConfig(EngineConfig):
    """HyperParallel settings, independent of torch FSDP's configuration."""

    # Engine initialization can disable VeRL optimizer transfers when native FSDP offload is enabled.
    _mutable_fields = EngineConfig._mutable_fields | {"optimizer_offload"}

    if TYPE_CHECKING:
        router_replay: EngineRouterReplayConfig

    strategy: str = "hyper_parallel"
    # Match FSDP2 model precision defaults: FP32 for training, BF16 for inference.
    dtype: str | None = None
    recipe_path: str | None = None
    trainer_config: dict[str, Any] | TrainerConfig = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.strategy != "hyper_parallel":
            raise ValueError("HyperParallelEngineConfig requires strategy=hyper_parallel")
        if self.router_replay.mode != "disabled":
            raise NotImplementedError("The HyperParallel policy adapter does not support router replay")
        if self.dtype not in (None, "bfloat16", "float16", "float32"):
            raise ValueError("HyperParallel engine dtype must be bfloat16, float16, float32 or None")

        recipe = OmegaConf.load(self.recipe_path) if self.recipe_path else OmegaConf.create()
        overrides = (self.trainer_config.to_dict()
                     if isinstance(self.trainer_config, TrainerConfig) else self.trainer_config or {})
        # Replace changed builders completely so their old arguments do not leak.
        for name, options in overrides.items():
            if not isinstance(options, Mapping) or "_target_" not in options:
                continue
            if options["_target_"] != OmegaConf.select(recipe, f"{name}._target_"):
                OmegaConf.update(recipe, name, options, merge=False)
        recipe = OmegaConf.merge(recipe, overrides)
        config = resolve_config(OmegaConf.to_container(recipe, resolve=True))
        if self.recipe_path:
            config._yaml_path = str(Path(self.recipe_path).resolve())  # pylint: disable=protected-access
        # BaseConfig freezes fields after construction.
        object.__setattr__(self, "trainer_config", config)


def _apply_verl_model_config(config: TrainerConfig, model_config: Any) -> None:
    """Apply the checkpoint path and overrides from ``actor_rollout_ref.model``."""
    config.model = config.model.replace(pretrained_model_name_or_path=model_config.path)
    logger.info("HyperParallel VeRL adapter uses VeRL model path: %s", model_config.path)

    flat_overrides = dict(getattr(model_config, "override_config", None) or {})
    nested_overrides = flat_overrides.pop("model_config", None) or {}
    # Keep launcher overrides such as the debug layer count when both forms are used.
    model_overrides = dict(OmegaConf.merge(nested_overrides, flat_overrides))
    # These loader arguments are provided separately by the native model builder.
    for name in ("path", "attn_implementation", "torch_dtype"):
        model_overrides.pop(name, None)
    if model_overrides:
        native_overrides = getattr(config.model, "config_overrides", None) or {}
        merged = OmegaConf.merge(native_overrides, model_overrides)
        config.model = config.model.replace(config_overrides=OmegaConf.to_container(merged, resolve=True))
        logger.info("HyperParallel VeRL adapter applies VeRL model overrides for %s", sorted(model_overrides))


def _apply_verl_engine_config(config: TrainerConfig, engine_config: HyperParallelEngineConfig) -> None:
    """Apply dtype and reproducibility from ``actor_rollout_ref.{actor,ref}.engine``."""
    dtype = engine_config.dtype
    if dtype is None:
        dtype = "bfloat16" if engine_config.forward_only else "float32"

    config.model = config.model.replace(torch_dtype=dtype)
    logger.info(
        "HyperParallel VeRL adapter sets model.torch_dtype=%s (forward_only=%s)", dtype, engine_config.forward_only,
    )

    config.model_init_dtype = dtype
    logger.info("HyperParallel VeRL adapter sets model_init_dtype=%s", dtype)

    config.training.seed = engine_config.seed
    logger.info("HyperParallel VeRL adapter sets training.seed=%s", engine_config.seed)

    config.training.enable_full_determinism = engine_config.full_determinism
    logger.info("HyperParallel VeRL adapter sets training.enable_full_determinism=%s", engine_config.full_determinism)


def _apply_verl_trainer_config(config: TrainerConfig, optimizer_config: OptimizerConfig | None) -> None:
    """Apply trainer settings resolved by VeRL before it creates the workers."""
    # VeRL derives the Actor's scheduler length from trainer.total_training_steps and
    # passes it to the engine through actor_rollout_ref.actor.optim.
    total_training_steps = getattr(optimizer_config, "total_training_steps", None)
    if total_training_steps is not None and total_training_steps > 0:
        config.training.train_iters = total_training_steps
        config.training.lr_scheduler_iters = total_training_steps
        logger.info(
            "HyperParallel VeRL adapter sets training and scheduler steps to %s from VeRL", total_training_steps,
        )


def validate_and_resolve_hyper_parallel_config(
    engine_config: HyperParallelEngineConfig,
    model_config: Any,
    *,
    optimizer_config: OptimizerConfig | None = None,
) -> TrainerConfig:
    """Validate the bound engine settings and adapt the native HyperParallel config."""
    # VeRL overwrites these flags from the model config after EngineConfig.__post_init__.
    if engine_config.use_fused_kernels:
        raise NotImplementedError("The HyperParallel policy adapter does not support use_fused_kernels")
    if not engine_config.use_remove_padding:
        raise NotImplementedError("The HyperParallel policy adapter requires use_remove_padding=True")

    config = engine_config.trainer_config
    _apply_verl_model_config(config, model_config)
    _apply_verl_engine_config(config, engine_config)
    _apply_verl_trainer_config(config, optimizer_config)

    # The VeRL worker does not run the native Trainer's diagnostics lifecycle.
    config.model_integration.mode = "off"
    logger.info("HyperParallel VeRL adapter sets model_integration.mode=off")
    config.debug.check_fsdp_runtime = False
    logger.info("HyperParallel VeRL adapter sets debug.check_fsdp_runtime=False")
    config.dataloader.get_batch = config.dataloader.get_batch.replace(reset_position_ids=True)
    logger.info("HyperParallel VeRL adapter sets reset_position_ids=True for independent packed samples")
    return config
