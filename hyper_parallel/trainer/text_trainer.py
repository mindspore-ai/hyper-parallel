# Copyright 2025-2026 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Text Trainer assembled from the shared BaseTrainer stages."""

__all__ = ["TextTrainer"]

from collections import defaultdict
from typing import Any, Dict

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.batching import calculate_num_micro_batches
from hyper_parallel.data.text import build_chat_template
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from hyper_parallel.trainer.runtime.logging import create_logger
from hyper_parallel.trainer.runtime.memory import print_device_mem_info
from hyper_parallel.trainer.runtime.memory_profiler import memory_profiler
from hyper_parallel.trainer.runtime.device import synchronize
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.config import TrainerConfig

logger = create_logger(__name__)


class TextTrainer:
    """Compose the text training runtime from explicit BaseTrainer stages."""

    base: BaseTrainer

    def __init__(self, config: TrainerConfig) -> None:
        """Build the text Trainer in the same explicit order as VeOmni."""
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.config = config

        self.base._setup()
        memory_profiler.reset(
            self.base.config.memory,
            global_rank=self.base.global_rank,
            tp_rank=self.base.mesh.tp_rank,
            dp_rank=self.base.mesh.dp_rank,
        )
        self.base._build_model()
        self.base._build_loss()

        # datasets
        self._build_model_assets()
        self._build_data_transform()
        self.base._build_dataset()

        # dataloader
        self._build_collate_fn()
        self.base._build_dataloader()

        # get_batch
        self._build_get_batch()
        self.base.attach_model_integration_data_pipeline()
        self.base._compute_train_iters()

        self.base._build_optimizer()
        self.base.model_integration.attach_optimizer(self.base.optimizer)
        self.base._build_lr_scheduler()
        self.base._build_training_context()
        self.base._build_graph_compiler()
        self.base._init_callbacks()

    def _build_model_assets(self) -> None:
        """Build tokenizer-backed assets for text training."""
        config: TrainerConfig = self.base.config
        if config.dataset is None:
            raise ValueError("dataset must define a build target")
        assets_config = config.dataset.model_assets
        tokenizer_target = assets_config.tokenizer
        if tokenizer_target is None:
            self.base.tokenizer = None
        elif getattr(tokenizer_target, "pretrained_model_name_or_path", None):
            self.base.tokenizer = tokenizer_target.build()
        else:
            self.base.tokenizer = tokenizer_target.build(
                pretrained_model_name_or_path=config.model.pretrained_model_name_or_path,
            )

        self.base.chat_template = None
        if assets_config.chat_template is not None:
            if self.base.tokenizer is None:
                raise ValueError("dataset.model_assets.tokenizer is required for conversation data")
            if isinstance(assets_config.chat_template, str):
                self.base.chat_template = build_chat_template(
                    assets_config.chat_template,
                    self.base.tokenizer,
                )
            else:
                self.base.chat_template = assets_config.chat_template.build(
                    tokenizer=self.base.tokenizer,
                )

        self.base.model_assets = [self.base.model_config]
        if self.base.tokenizer is not None:
            self.base.model_assets.append(self.base.tokenizer)
        if self.base.chat_template is not None:
            self.base.model_assets.append(self.base.chat_template)

    def _build_data_transform(self) -> None:
        """Build the configured text sample transform."""
        dataset_config = self.base.config.dataset
        if dataset_config is None:
            raise ValueError("dataset must define a build target")
        if dataset_config.data_transform is None:
            self.base.data_transform = None
            return
        self.base.data_transform = dataset_config.data_transform.build(
            tokenizer=self.base.tokenizer,
            chat_template=self.base.chat_template,
        )

    def _build_collate_fn(self) -> None:
        """Build the text collator and gradient-accumulation batch count."""
        dataloader_config = self.base.config.dataloader
        if dataloader_config is None or dataloader_config.collate_fn is None:
            raise ValueError("dataloader.collate_fn must define a build target")
        training_config = self.base.config.training
        self.base.num_micro_batches = calculate_num_micro_batches(
            global_batch_size=training_config.global_batch_size,
            micro_batch_size=training_config.micro_batch_size,
            dp_world_size=self.base.mesh.dp_size,
        )
        self.base.collate_fn = dataloader_config.collate_fn.build(
            mesh_context=self.base.mesh,
        )

    def _build_get_batch(self) -> None:
        """Build the DataLoader-to-LLM batch runtime."""
        config = self.base.config
        if config.dataloader.get_batch is None:
            raise ValueError("dataloader.get_batch must define a batching runtime target")
        get_batch = config.dataloader.get_batch.build(
            mesh_context=self.base.mesh,
            device=self.base.device,
            tokenizer=self.base.tokenizer,
            data_config=getattr(config.dataset, "data_config", {}),
            pp_shared_data=bool(getattr(config.dataloader, "pp_shared_data", False)),
        )
        self.base.get_batch = get_batch

    @property
    def distributed_setup(self) -> Any:
        """Return the shared distributed setup."""
        return self.base.distributed_setup

    @property
    def mesh(self) -> Any:
        """Return the shared mesh context."""
        return self.base.mesh

    @property
    def dp_cp_mesh(self) -> Any:
        """Return the shared data/context-parallel mesh."""
        return self.base.dp_cp_mesh

    def on_train_begin(self) -> None:
        """Dispatch the training-begin lifecycle hook."""
        self.base.on_train_begin()

    def on_train_end(self) -> None:
        """Dispatch the training-end lifecycle hook."""
        self.base.on_train_end()

    def on_epoch_begin(self) -> None:
        """Dispatch the epoch-begin lifecycle hook."""
        self.base.on_epoch_begin()

    def on_epoch_end(self) -> None:
        """Dispatch the epoch-end lifecycle hook."""
        self.base.on_epoch_end()

    def on_step_begin(self) -> None:
        """Dispatch the step-begin lifecycle hook."""
        self.base.on_step_begin()

    def on_micro_step_begin(self, micro_batch: dict[str, Any]) -> None:
        """Dispatch the micro-step-begin lifecycle hook.

        Args:
            micro_batch: Prepared inputs and lightweight metric metadata.
        """
        self.base.on_micro_step_begin(micro_batch)

    def on_step_end(
            self,
            loss: Any = None,
            loss_dict: Any = None,
            grad_norm: Any = None,
    ) -> None:
        """Dispatch the step-end lifecycle hook.

        Args:
            loss: Aggregated loss for the optimizer step.
            loss_dict: Named loss values for the optimizer step.
            grad_norm: Gradient norm measured before the optimizer update.
        """
        self.base.on_step_end(
            loss=loss,
            loss_dict=loss_dict,
            grad_norm=grad_norm,
        )

    def forward_backward_step(
            self,
            data_iterator: Any,
            num_micro_steps: int,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Fetch and execute one forward-backward micro-step.

        Args:
            data_iterator: Iterator providing one raw FB batch per call.
            num_micro_steps: Number of FB steps in the current optimizer step.

        Returns:
            Loss tensor and named loss tensors for this FB step.
        """
        model_inputs, loss_inputs = self.base.get_batch(data_iterator)
        self.base.current_token_counts = count_loss_token(loss_inputs)
        self.base.step_token_counts = {
            name: token_count * num_micro_steps
            for name, token_count in self.base.current_token_counts.items()
        }
        metric_inputs = {
            **model_inputs,
            **loss_inputs,
            "token_count": self.base.current_token_counts["foundation_tokens"],
        }
        self.on_micro_step_begin(metric_inputs)
        del metric_inputs
        loss, loss_dict = self.base.forward_backward_step(model_inputs, loss_inputs)

        return loss, loss_dict

    def train_step(self, data_iterator: Any) -> Dict[str, float]:
        """Execute one text training step.

        Args:
            data_iterator: Iterator providing the step's micro-batches.

        Returns:
            Aggregated loss and gradient norm for the completed step.
        """
        memory_profiler.step()
        num_micro_steps = self.base.num_micro_batches

        self.on_step_begin()
        self.base.model_integration.begin_step(self.base.state.global_step + 1)

        # A pipeline model owns the complete forward/backward schedule.  The
        # text trainer's normal path invokes ``model(**inputs)`` once per
        # micro-batch, which is invalid for AutoPipeline (its forward is an
        # intentional guard).  Materialize one DP-local pipeline batch through
        # the existing batch adapter, then let the schedule split it into PP
        # micro-batches and transport activations between stages.
        if (
            hasattr(self.base.model, "step")
            and callable(self.base.model.step)
            and getattr(self.base.model, "info", None) is not None
        ):
            return self._pipeline_train_step(data_iterator, num_micro_steps)

        synchronize()

        total_loss = 0.0
        total_loss_dict = defaultdict(int)

        for micro_step in range(num_micro_steps):
            self.base.model_reshard(micro_step, num_micro_steps)
            self.base.configure_fsdp_gradient_sync(
                micro_step,
                num_micro_steps,
            )
            self.base.begin_fsdp_runtime_diagnostics(micro_step)
            loss, loss_dict = self.forward_backward_step(
                data_iterator,
                num_micro_steps,
            )

            total_loss += loss.item()
            for loss_name, loss_value in loss_dict.items():
                total_loss_dict[loss_name] += loss_value.item()

        grad_norm = self.base.prepare_optimizer_step()
        self.base.step_optimizers_and_schedulers()

        # Checkpoint and logging callbacks observe the number of completed
        # optimizer updates.
        self.base.state.global_step += 1
        grad_norm_value = float(grad_norm)
        self.base._end_model_integration_step(  # pylint: disable=protected-access
            {"loss": total_loss, "grad_norm": grad_norm_value}
        )
        self.on_step_end(
            loss=total_loss,
            loss_dict=total_loss_dict,
            grad_norm=grad_norm_value,
        )

        return {
            "loss": total_loss,
            "grad_norm": grad_norm_value,
        }

    @staticmethod
    def _concat_pipeline_inputs(inputs: list[dict[str, Any]]) -> dict[str, Any]:
        """Concatenate tensor-valued batch fields along the sample dimension."""
        if not inputs:
            return {}
        result = {}
        for key in inputs[0]:
            values = [item[key] for item in inputs if key in item]
            if values and all(isinstance(value, torch.Tensor) for value in values):
                result[key] = torch.cat(values, dim=0)
            elif values:
                # Auxiliary metadata is normally identical across the local
                # micro-batches; retain the first value for schedule kwargs.
                result[key] = values[0]
        return result

    def _pipeline_train_step(self, data_iterator: Any, num_micro_steps: int) -> Dict[str, float]:
        """Run one optimizer update through AutoPipeline."""
        model_inputs = []
        loss_inputs = []
        for _ in range(num_micro_steps):
            model_batch, loss_batch = self.base.get_batch(data_iterator)
            model_batch = self.base.preforward(model_batch)
            loss_batch = self.base.preforward(loss_batch)
            self.base.model_integration.record_batch(model_batch, loss_batch)
            token_counts = count_loss_token(loss_batch)
            self.on_micro_step_begin({
                **model_batch, **loss_batch,
                "token_count": token_counts["foundation_tokens"],
            })
            model_inputs.append(dict(model_batch))
            loss_inputs.append(dict(loss_batch))

        batch = self._concat_pipeline_inputs(model_inputs)
        losses_batch = self._concat_pipeline_inputs(loss_inputs)
        input_ids = batch.pop("input_ids", None)
        labels = losses_batch.get("labels", batch.pop("labels", None))
        self.base.step_token_counts = count_loss_token(losses_batch)
        optimizers = self.base.optimizer if isinstance(self.base.optimizer, list) else [self.base.optimizer]
        for optimizer in optimizers:
            optimizer.zero_grad()
        losses = self.base.model.step(
            input_ids if self.base.model.info.has_first_stage else None,
            target=labels if self.base.model.info.has_last_stage else None,
            **batch,
        )
        loss = sum(losses) if losses else torch.zeros((), device=self.base.device)
        # Only the last PP stage produces the schedule loss.  Broadcast the
        # scalar across each PP line for consistent Trainer metrics (the loss
        # graph has already been consumed by schedule.backward at this point).
        metric_loss = loss.detach().clone()
        if torch.distributed.is_initialized() and self.base.model.info.stages:
            # Use the world group rather than a stage-local handle: every
            # rank participates here, while only the last-stage replica(s)
            # contribute a non-zero value.  Averaging by the number of
            # replicas of the last stage yields one consistent scalar for
            # rank-0 logging regardless of TP/CP/DP layout.
            pp_size = self.base.model.pp_mesh.size()
            torch.distributed.all_reduce(metric_loss)
            metric_loss = metric_loss / (torch.distributed.get_world_size() / pp_size)
        grad_norm = self.base.prepare_optimizer_step()
        self.base.step_optimizers_and_schedulers()

        self.base.state.global_step += 1
        grad_norm_value = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        loss_value = metric_loss.item()
        self.base._end_model_integration_step(  # pylint: disable=protected-access
            {"loss": loss_value, "grad_norm": grad_norm_value}
        )
        self.on_step_end(
            loss=loss_value,
            loss_dict={"loss": loss.detach()},
            grad_norm=grad_norm_value,
        )
        return {"loss": loss_value, "grad_norm": grad_norm_value}

    def train(self) -> None:
        """Run the configured global optimizer steps by Dataset epoch."""
        try:
            config = self.base.config
            self.on_train_begin()
            logger.info(
                "Rank%s Start training. Global step: %s. Train iters: %s. Start epoch: %s. Train epochs: %s.",
                self.base.local_rank,
                self.base.state.global_step,
                self.base.train_iters,
                self.base.state.epoch,
                self.base.train_epochs,
            )

            # Checkpoint resume restores state.global_step, state.epoch, and the DataLoader cursor.
            for epoch in range(self.base.state.epoch, self.base.train_epochs):
                train_dataloader = self.base.train_dataloader
                if hasattr(train_dataloader, "set_epoch"):
                    train_dataloader.set_epoch(epoch)

                self.base.state.epoch = epoch
                self.on_epoch_begin()
                data_iterator = iter(train_dataloader) if train_dataloader is not None else None
                start_step = self.base.state.global_step - epoch * self.base.train_steps
                train_steps = min(self.base.train_steps, self.base.train_iters - epoch * self.base.train_steps)
                for _ in range(start_step, train_steps):
                    try:
                        self.train_step(data_iterator)
                    except StopIteration:
                        logger.info(
                            "epoch:%s Dataloader finished with drop_last %s", epoch, config.dataloader.drop_last
                        )
                        break

                self.on_epoch_end()
                self.base.state.epoch = epoch + 1
                print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
                if self.base.state.global_step >= self.base.train_iters:
                    break

            memory_profiler.stop()
            self.on_train_end()

        except BaseException:
            try:
                memory_profiler.stop()
            # Snapshot export must preserve the training error even if interrupted.
            except BaseException:  # pylint: disable=broad-exception-caught
                logger.exception("Failed to save memory snapshot after training error.")
            finally:
                self.base.abort_profilers()
            raise

        synchronize()
        self.base.destroy_distributed()
