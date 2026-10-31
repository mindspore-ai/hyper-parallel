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
"""DiT trainer assembled from the shared AutoModels BaseTrainer stages."""

from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Any, Dict

import torch  # pylint: disable=forbidden-backend-import
import torch.nn.functional as F  # pylint: disable=forbidden-backend-import

from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.core.utils import clip_grad_norm_
from hyper_parallel.data.batching import calculate_num_micro_batches
from hyper_parallel.data.dit import DiTBatch
from hyper_parallel.models._transformers import resolve_dit_provider
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.config import TrainerConfig
from hyper_parallel.trainer.runtime.device import synchronize
from hyper_parallel.trainer.runtime.logging import create_logger
from hyper_parallel.trainer.runtime.memory import empty_cache, print_device_mem_info

logger = create_logger(__name__)


def _is_rank0() -> bool:
    return (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )


def _rank0_info(message: str, *args: Any) -> None:
    if _is_rank0():
        logger.info(message, *args)


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return value.item()
    return float(value)


def _move_nested_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_nested_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_nested_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_nested_to_device(item, device) for item in value)
    return value


def _count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for parameter in module.parameters():
        numel = parameter.numel()
        total += numel
        if parameter.requires_grad:
            trainable += numel
    return total, trainable


def _unwrap_single_sample(value: Any, field_name: str) -> Any:
    """Unwrap collator-preserved sample lists for an upstream DiT model."""
    if not isinstance(value, list):
        return value
    if len(value) != 1:
        raise ValueError(
            "Diffusers DiT training currently requires micro_batch_size=1; "
            f"field {field_name!r} contains {len(value)} samples"
        )
    return value[0]


def _select_forward_inputs(
    model: torch.nn.Module,
    micro_batch: dict[str, Any],
) -> dict[str, Any]:
    """Project a conditioned batch onto the upstream model's forward inputs."""
    signature = getattr(model, "_hp_forward_signature", None)
    if signature is None:
        signature = inspect.signature(inspect.unwrap(model.forward))
    accepted_names = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        accepted_names = micro_batch.keys()
    return {
        name: _unwrap_single_sample(value, name)
        for name, value in micro_batch.items()
        if name in accepted_names
    }


def _has_condition_inputs(condition_model: torch.nn.Module, micro_batch: dict[str, Any]) -> bool:
    """Return whether a batch provides the condition model's required inputs."""
    signature = inspect.signature(inspect.unwrap(condition_model.get_condition))
    required_names = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return required_names.issubset(micro_batch)


class DiTTrainer:
    """Compose DiT full-finetuning runtime from explicit BaseTrainer stages."""

    base: BaseTrainer

    def __init__(self, config: TrainerConfig) -> None:
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.config = config

        self.base._setup()
        self.base._build_model()

        self._build_model_assets()
        self._build_data_transform()
        self.base._build_dataset()
        self._build_collate_fn()
        self.base._build_dataloader()
        self._build_get_batch()
        self.base._compute_train_iters()

        self.base._build_optimizer()
        self.base._build_lr_scheduler()
        self.base._build_training_context()
        self.base._init_callbacks()

    def _build_model_assets(self) -> None:
        config = self.base.config
        provider = resolve_dit_provider(
            getattr(config.model, "model_provider", None),
            getattr(config.model, "model_type", None),
        )
        build_condition = getattr(provider, "build_condition_model", None)
        if callable(build_condition):
            self.base.condition_model = build_condition(
                model_target=config.model,
                device=self.base.device,
                dp_rank=self.base.mesh.dp_rank,
                seed=self.base.config.training.seed or self.base.default_seed,
            )
        else:
            self.base.condition_model = None
        model_total, model_trainable = _count_parameters(self.base.model)
        condition_total, condition_trainable = (
            _count_parameters(self.base.condition_model)
            if self.base.condition_model is not None
            else (0, 0)
        )
        _rank0_info(
            "DiT trainability: task=%s, transformer_path=%s, condition_path=%s, "
            "transformer_local_trainable=%d/%d, condition_trainable=%d/%d",
            getattr(config.model, "task", None),
            getattr(config.model, "pretrained_model_name_or_path", None),
            getattr(config.model, "condition_model_name_or_path", None),
            model_trainable,
            model_total,
            condition_trainable,
            condition_total,
        )

    def _build_data_transform(self) -> None:
        dataset_config = self.base.config.dataset
        if dataset_config is None:
            raise ValueError("dataset must define a build target")
        if dataset_config.data_transform is None:
            self.base.data_transform = None
            return
        self.base.data_transform = dataset_config.data_transform.build()

    def _build_collate_fn(self) -> None:
        dataloader_config = self.base.config.dataloader
        if dataloader_config is None or dataloader_config.collate_fn is None:
            raise ValueError("dataloader.collate_fn must define a build target")
        training_config = self.base.config.training
        self.base.num_micro_batches = calculate_num_micro_batches(
            global_batch_size=training_config.global_batch_size,
            micro_batch_size=training_config.micro_batch_size,
            dp_world_size=self.base.mesh.dp_size,
        )
        self.base.collate_fn = dataloader_config.collate_fn.build()

    def _build_get_batch(self) -> None:
        config = self.base.config
        get_batch_builder = config.dataloader.get_batch.build if config.dataloader.get_batch else DiTBatch
        self.base.get_batch = get_batch_builder(
            mesh_context=self.base.mesh,
            device=self.base.device,
            pp_shared_data=bool(getattr(config.dataloader, "pp_shared_data", False)),
        )

    def preforward(self, micro_batch: dict[str, Any]) -> dict[str, Any]:
        return {key: _move_nested_to_device(value, self.base.device) for key, value in micro_batch.items()}

    def postforward(
        self,
        outputs: Any,
        micro_batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the flow-matching objective outside the upstream model."""
        if "training_target" not in micro_batch:
            raise ValueError("Conditioned DiT batch missing required loss field: training_target")
        prediction = getattr(outputs, "sample", None)
        if not isinstance(prediction, torch.Tensor):
            raise ValueError("Diffusers DiT model output must contain a tensor in 'sample'")
        training_target = _unwrap_single_sample(
            micro_batch["training_target"],
            "training_target",
        )
        if not isinstance(training_target, torch.Tensor):
            raise ValueError("DiT training_target must be a tensor")

        mse_loss = F.mse_loss(
            prediction.float(),
            training_target.float(),
            reduction="none",
        )
        mse_loss = mse_loss.reshape(mse_loss.shape[0], -1).mean(dim=1).mean()
        scaled_loss = mse_loss / self.base.num_micro_batches
        return scaled_loss, {"mse_loss": scaled_loss}

    def forward_backward_step(self, micro_batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        micro_batch = self.preforward(micro_batch)
        condition_model = getattr(self.base, "condition_model", None)
        if condition_model is not None and _has_condition_inputs(condition_model, micro_batch):
            with torch.no_grad():
                micro_batch = condition_model.get_condition(**micro_batch)
                micro_batch = condition_model.process_condition(**micro_batch)

        model_fwd_context = (
            self.base.model_fwd_context()
            if callable(self.base.model_fwd_context)
            else self.base.model_fwd_context
        )
        with model_fwd_context:
            model_inputs = _select_forward_inputs(self.base.model, micro_batch)
            outputs = self.base.model(**model_inputs)
        loss, loss_dict = self.postforward(outputs, micro_batch)
        del outputs
        if self.base.config.training.empty_cache_before_backward:
            empty_cache()
        with self.base.model_bwd_context:
            loss.backward()
        del micro_batch
        return loss, loss_dict

    def train_step(self, data_iterator: Any) -> Dict[str, float]:
        config = self.base.config
        num_micro_steps = self.base.num_micro_batches
        micro_batches = [self.base.get_batch(data_iterator)]
        for _ in range(1, num_micro_steps):
            micro_batches.append(self.base.get_batch(data_iterator))

        self.base.on_step_begin(micro_batches=micro_batches)
        synchronize()

        total_loss = 0.0
        total_loss_dict = defaultdict(float)
        for micro_step, micro_batch in enumerate(micro_batches):
            self.base.model_reshard(micro_step, num_micro_steps)
            self.base.configure_fsdp_gradient_sync(micro_step, num_micro_steps)
            loss, loss_dict = self.forward_backward_step(micro_batch)
            total_loss += _as_float(loss)
            for loss_name, loss_value in loss_dict.items():
                total_loss_dict[loss_name] += _as_float(loss_value)

        grad_norm = clip_grad_norm_(self.base.model, config.training.max_grad_norm)
        optimizers = self.base.optimizer if isinstance(self.base.optimizer, list) else [self.base.optimizer]
        for optimizer in optimizers:
            with SkipDTensorDispatch():
                optimizer.step()
            optimizer.zero_grad()

        schedulers = (
            self.base.lr_scheduler
            if isinstance(self.base.lr_scheduler, list)
            else ([self.base.lr_scheduler] if self.base.lr_scheduler is not None else [])
        )
        for scheduler in schedulers:
            scheduler.step()

        grad_norm_value = _as_float(grad_norm)
        # Checkpoint and logging callbacks must observe the number of completed
        # optimizer updates; otherwise step-based saves start at step 0 and
        # fire one update later than their configured cadence.
        self.base.state.global_step += 1
        self.base.on_step_end(loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm_value)
        return {"loss": total_loss, "grad_norm": grad_norm_value}

    def train(self) -> None:
        config = self.base.config
        self.base.on_train_begin()
        logger.info(
            "Rank%s Start DiT training. Global step: %s. Train iters: %s. Start epoch: %s. Train epochs: %s.",
            self.base.local_rank,
            self.base.state.global_step,
            self.base.train_iters,
            self.base.state.epoch,
            self.base.train_epochs,
        )

        for epoch in range(self.base.state.epoch, self.base.train_epochs):
            train_dataloader = self.base.train_dataloader
            if hasattr(train_dataloader, "set_epoch"):
                train_dataloader.set_epoch(epoch)

            self.base.on_epoch_begin()
            data_iterator = iter(train_dataloader)
            start_step = self.base.state.global_step - epoch * self.base.train_steps
            train_steps = min(self.base.train_steps, self.base.train_iters - epoch * self.base.train_steps)
            for _ in range(start_step, train_steps):
                try:
                    self.train_step(data_iterator)
                except StopIteration:
                    logger.info("epoch:%s Dataloader finished with drop_last %s", epoch, config.dataloader.drop_last)
                    break

            self.base.on_epoch_end()
            self.base.state.epoch = epoch + 1
            print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
            if self.base.state.global_step >= self.base.train_iters:
                break

        self.base.on_train_end()
        synchronize()
        self.base.destroy_distributed()


__all__ = ["DiTTrainer"]
