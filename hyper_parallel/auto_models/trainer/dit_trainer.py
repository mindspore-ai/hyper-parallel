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

from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict

import torch

from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.auto_models.components.datasets import DiTBatch, calculate_num_micro_batches
from hyper_parallel.auto_models.components.models.wan import build_wan_condition_model
from hyper_parallel.auto_models.components.utils import helper
from hyper_parallel.auto_models.components.utils.device import synchronize  # pylint: disable=syntax-error
from hyper_parallel.auto_models.trainer.base import BaseTrainer
from hyper_parallel.auto_models.trainer.config import Target, TrainerConfig
from hyper_parallel.core.utils import clip_grad_norm_

logger = helper.create_logger(__name__)


def _is_rank0() -> bool:
    return (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )


def _rank0_info(message: str, *args: Any) -> None:
    if _is_rank0():
        logger.info(message, *args)


def _rank0_warning(message: str, *args: Any) -> None:
    if _is_rank0():
        logger.warning(message, *args)


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


def _derive_condition_model_path(model_path: Any) -> str | None:
    if not model_path:
        return None
    suffix = "/transformer"
    normalized = str(model_path).replace("\\", "/")
    if normalized.endswith(suffix):
        return str(model_path)[: -len(suffix)]
    return str(model_path)


def _resolve_condition_model_path(model_target: Target[Any]) -> str | None:
    condition_path = getattr(model_target, "condition_model_name_or_path", None)
    model_path = getattr(model_target, "pretrained_model_name_or_path", None)
    derived_path = _derive_condition_model_path(model_path)
    if condition_path and str(condition_path).lower() != "auto":
        if derived_path is not None and str(condition_path) != str(derived_path):
            _rank0_warning(
                "Wan condition_model_name_or_path=%s differs from pretrained_model_name_or_path base=%s. "
                "Use condition_model_name_or_path=auto if the condition path should follow the transformer path.",
                condition_path,
                derived_path,
            )
        return str(condition_path)
    return derived_path


def _count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for parameter in module.parameters():
        numel = parameter.numel()
        total += numel
        if parameter.requires_grad:
            trainable += numel
    return total, trainable


class DiTTrainer:
    """Compose DiT full-finetuning runtime from explicit BaseTrainer stages."""

    base: BaseTrainer

    def __init__(self, config: TrainerConfig) -> None:
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.config = config

        self.base._setup()
        self.base._build_model()
        self.base._build_loss()

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
        task = getattr(config.model, "task", "t2v")
        condition_path = _resolve_condition_model_path(config.model)
        if condition_path is None:
            raise ValueError("Wan DiT training requires model.condition_model_name_or_path")

        condition_kwargs = dict(getattr(config.model, "condition_model", {}) or {})
        condition_kwargs.setdefault("transformer_dtype", getattr(config.model, "torch_dtype", "bfloat16"))
        condition_kwargs.setdefault("local_files_only", bool(getattr(config.model, "local_files_only", False)))
        self.base.condition_model = build_wan_condition_model(
            base_model_path=condition_path,
            task=task,
            device=self.base.device,
            dp_rank=self.base.mesh.dp_rank,
            seed=self.base.config.training.seed or self.base.default_seed,
            **condition_kwargs,
        )
        self.base.tokenizer = getattr(self.base.condition_model, "tokenizer", None)
        self.base.processor = getattr(self.base.condition_model, "image_processor", None)
        self.base.chat_template = None
        self.base.model_assets = [self.base.model_config]
        model_total, model_trainable = _count_parameters(self.base.model)
        condition_total, condition_trainable = _count_parameters(self.base.condition_model)
        _rank0_info(
            "Wan DiT trainability: task=%s, transformer_path=%s, condition_path=%s, "
            "transformer_local_trainable=%d/%d, condition_trainable=%d/%d",
            task,
            getattr(config.model, "pretrained_model_name_or_path", None),
            condition_path,
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

    def _condition_micro_batch(self, micro_batch: dict[str, Any]) -> dict[str, Any]:
        if "hidden_states" in micro_batch and "training_target" in micro_batch:
            return micro_batch
        condition_model = getattr(self.base, "condition_model", None)
        if condition_model is None:
            raise ValueError("DiT online training requires a condition_model")
        required = ("inputs", "videos")
        missing = [field for field in required if field not in micro_batch]
        if missing:
            raise ValueError(f"Wan raw DiT batch missing required fields: {missing}")
        with torch.no_grad():
            conditions = condition_model.get_condition(
                inputs=micro_batch["inputs"],
                videos=micro_batch["videos"],
                images=micro_batch.get("images"),
            )
            return condition_model.process_condition(**conditions)

    def postforward(self, outputs: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_dict = getattr(outputs, "loss", None)
        if not isinstance(loss_dict, dict) or not loss_dict:
            raise ValueError("Wan DiT model must return a non-empty loss dict")
        scaled_loss_dict = {
            name: loss_value / self.base.num_micro_batches
            for name, loss_value in loss_dict.items()
        }
        loss = torch.stack(list(scaled_loss_dict.values())).sum()
        return loss, scaled_loss_dict

    def forward_backward_step(self, micro_batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        channel_loss_callback = getattr(self.base, "channel_loss_callback", None)
        micro_step_context = (
            channel_loss_callback.micro_step_context(self.base.state, micro_batch)
            if channel_loss_callback is not None
            else nullcontext()
        )
        with micro_step_context:
            micro_batch = self.preforward(micro_batch)
            micro_batch = self._condition_micro_batch(micro_batch)
            micro_batch = self.preforward(micro_batch)

            model_fwd_context = (
                self.base.model_fwd_context()
                if callable(self.base.model_fwd_context)
                else self.base.model_fwd_context
            )
            with model_fwd_context:
                outputs = self.base.model(**micro_batch)
            loss, loss_dict = self.postforward(outputs)
            del outputs
            if self.base.config.training.empty_cache_before_backward:
                helper.empty_cache()
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
        self.base.on_step_end(loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm_value)
        self.base.state.global_step += 1
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
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
            if self.base.state.global_step >= self.base.train_iters:
                break

        self.base.on_train_end()
        synchronize()
        self.base.destroy_distributed()


__all__ = ["DiTTrainer"]
