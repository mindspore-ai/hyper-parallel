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
"""Stable bridge helpers from graph mode to the eager Trainer/AutoModel stack."""

from dataclasses import replace
from typing import Any, Optional

from hyper_parallel.models.build_options import CompileConfig
from hyper_parallel.trainer.config import Target, TrainerConfig

from .parallel_config import PassConfig


# def _wrap_dataloader_for_graph_mode(dataloader_config: Any) -> Any:
#     """Enable graph-stable padding for dynamic online text batches."""
#     if dataloader_config is None:
#         return None

#     target = getattr(dataloader_config, "target", None)
#     if target is None:
#         return dataloader_config

#     if getattr(target, "_target_path", None) != "hyper_parallel.data.batching.DynamicBatchDataLoader":
#         return dataloader_config

#     return replace(
#         dataloader_config,
#         target=target.replace(pad_to_token_budget=True),
#     )


def clone_config_for_graph_mode(config: TrainerConfig) -> TrainerConfig:
    """Clone a TrainerConfig and disable eager per-layer compile.

    Phase-1 graph integration reuses AutoModel for model preparation, but the
    execution step belongs to ``hyper_parallel.compile``. The eager
    ``distributed.compile.apply_compile`` path must therefore be disabled to
    avoid compiling decoder layers twice.
    """
    return replace(
        config,
        model=wrap_model_target_for_graph_mode(config.model),
        # dataloader=_wrap_dataloader_for_graph_mode(config.dataloader),
        compile=CompileConfig(enabled=False),
    )


def build_model_for_graph_mode(
    *,
    model_target: Target[Any],
    distributed_setup: Any = None,
    **kwargs: Any,
) -> Any:
    """Build a model for graph mode while skipping eager FSDP2 wrapping.

    Graph mode still wants AutoModel's mesh construction, TP/SP sharding plan,
    and other preparation steps. The eager FSDP2 runtime wrap, however, must be
    disabled so the graph pass pipeline owns FSDP behavior.
    """
    if distributed_setup is not None:
        distributed_setup = replace(distributed_setup, strategy_config=None)
    return model_target.build(distributed_setup=distributed_setup, **kwargs)


def wrap_model_target_for_graph_mode(model_target: Target[Any]) -> Target[Any]:
    """Wrap the configured model target with graph-mode runtime adjustments."""
    return Target(
        _target_=build_model_for_graph_mode,
        target_path="hyper_parallel.compile.dependency_bridge.build_model_for_graph_mode",
        model_target=model_target,
    )


def build_pass_config_from_trainer_config(
    config: TrainerConfig,
    *,
    fsdp_enabled: Optional[bool] = None,
    enable_overlap: bool = False,
) -> PassConfig:
    """Project Trainer topology intent onto graph-mode pass config.

    The first integration milestone keeps graph mode focused on model
    preparation reuse plus single-process execution, so FSDP graph passes stay
    opt-in and disabled by default.
    """
    accelerator = config.accelerator
    if fsdp_enabled is None:
        fsdp_enabled = (
            config.fsdp_config.dp_shard_size > 1
            or config.fsdp_config.edp_shard_size > 1
        )
    fsdp_degree = config.fsdp_config.dp_shard_size if fsdp_enabled else None
    return PassConfig(
        enable_overlap=enable_overlap,
        fsdp_enabled=fsdp_enabled,
        fsdp_degree=fsdp_degree,
        tp_size=accelerator.tp_size,
        sequence_parallel=accelerator.sequence_parallel,
        loss_parallel=accelerator.loss_parallel,
    )


def loss_to_metrics(loss: Any) -> dict[str, Any]:
    """Normalize graph loss output to a callback-friendly metrics mapping."""
    if isinstance(loss, dict):
        return {
            str(name): value.detach() if hasattr(value, "detach") else value
            for name, value in loss.items()
        }
    return {"graph_loss": loss.detach() if hasattr(loss, "detach") else loss}
