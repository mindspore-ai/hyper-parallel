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
"""VLM Trainer assembled from the shared BaseTrainer stages."""

__all__ = ["VLMTrainer"]

from collections import defaultdict
from typing import Any, Dict

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.batching import calculate_num_micro_batches
from hyper_parallel.data.batching.get_batch import OmniParallelBatch
from hyper_parallel.data.omni import OmniDataTransform
from hyper_parallel.distributed.encoder_parallel import build_encoder_parallel_runtime
from hyper_parallel.distributed.encoder_parallel.transport import EncoderParallelBatchTransport
from hyper_parallel.models.registry import resolve_model_adapter
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from hyper_parallel.trainer.runtime.logging import create_logger
from hyper_parallel.trainer.runtime.memory import print_device_mem_info
from hyper_parallel.trainer.runtime.memory_profiler import memory_profiler
from hyper_parallel.trainer.runtime.device import synchronize  # pylint: disable=syntax-error
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.config import TrainerConfig

logger = create_logger(__name__)


class VLMTrainer:
    """Compose the VLM training runtime from explicit BaseTrainer stages."""

    base: BaseTrainer

    def __init__(self, config: TrainerConfig) -> None:
        """Build the VLM Trainer in the same explicit order as VeOmni."""
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
        self.base._init_callbacks()

    def _build_model_assets(self) -> None:
        """Build and publish the processor consumed by the Omni transform."""
        config: TrainerConfig = self.base.config
        if config.dataset is None:
            raise ValueError("dataset must define a build target")

        # The model-owned processor is built through the Omni ``model_assets``
        # target; forward the model's ``trust_remote_code`` flag so a natively
        # registered architecture such as the Kimi-K2.6 family (``kimi_k26``)
        # stays on the local implementation unless the config opts into remote
        # code.
        processor = config.dataset.model_assets.build(
            trust_remote_code=bool(
                getattr(config.model, "trust_remote_code", True)
            ),
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("dataset.model_assets must build a processor with a tokenizer")

        self.base.processor = processor
        self.base.tokenizer = tokenizer
        self.base.chat_template = getattr(processor, "chat_template", None)
        self.base.model_assets = [self.base.model_config, processor]

    def _build_data_transform(self) -> None:
        """Build the Omni encoding lifecycle from the prebuilt processor."""
        dataset_config = self.base.config.dataset
        if dataset_config is None:
            raise ValueError("dataset must define a build target")

        if dataset_config.data_transform is None:
            raise ValueError("dataset.data_transform must define an Omni transform build target")

        if dataset_config.training_mode is None:
            raise ValueError("dataset.training_mode must be explicitly set to 'sft' or 'pretrain' for VLM training")

        data_transform = dataset_config.data_transform.build(
            processor=self.base.processor,
            training_mode=dataset_config.training_mode,
        )
        if not isinstance(data_transform, OmniDataTransform):
            raise TypeError("dataset.data_transform must build an OmniDataTransform")

        self.base.data_transform = data_transform

    def _build_collate_fn(self) -> None:
        """Build the VLM collator and gradient-accumulation batch count."""
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
        if getattr(config.dataloader.get_batch, "encoder_parallel", None) is not None:
            raise ValueError("encoder_parallel belongs under accelerator, not dataloader.get_batch")
        encoder_runtime = build_encoder_parallel_runtime(
            config=config.accelerator.encoder_parallel,
            mesh_context=self.base.mesh,
            model=self.base.model,
            model_adapter_spec=resolve_model_adapter(self.base.model),
            execution_options={
                "micro_batch_size": int(config.training.micro_batch_size),
                "compile_enabled": bool(config.compile.enabled),
                "validate_placement": bool(getattr(config.model, "validate_placement", False)),
            },
        )
        get_batch = config.dataloader.get_batch.build(
            mesh_context=self.base.mesh,
            device=self.base.device,
            data_config=getattr(config.dataset, "data_config", {}),
            pp_shared_data=bool(getattr(config.dataloader, "pp_shared_data", False)),
        )
        if encoder_runtime is not None:
            if not isinstance(get_batch, OmniParallelBatch):
                raise ValueError("encoder parallel requires OmniParallelBatch")
            get_batch.batch_transport = EncoderParallelBatchTransport(runtime=encoder_runtime)
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
            loss: Aggregate step loss.
            loss_dict: Named step losses.
            grad_norm: Measured gradient norm.
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
            data_iterator: Iterator providing one multimodal batch per call.
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
        self.on_micro_step_begin(model_inputs)
        loss, loss_dict = self.base.forward_backward_step(model_inputs, loss_inputs)

        return loss, loss_dict

    def train_step(self, data_iterator: Any) -> Dict[str, float]:
        """Execute one VLM training step.

        Args:
            data_iterator: Iterator providing multimodal micro-batches.

        Returns:
            Aggregated loss and gradient norm.
        """
        memory_profiler.step()
        num_micro_steps = self.base.num_micro_batches

        self.on_step_begin()
        self.base.model_integration.begin_step(self.base.state.global_step + 1)
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

        # Checkpoint and logging callbacks observe completed optimizer updates.
        self.base.state.global_step += 1
        grad_norm_value = float(grad_norm)
        # The base trainer owns this integration hook and the VLM subclass only
        # mirrors the base step order, so reaching the private method is intended.
        self.base._end_model_integration_step(  # pylint: disable=protected-access
            {"loss": total_loss, "grad_norm": grad_norm_value}
        )
        self.on_step_end(
            loss=total_loss,
            loss_dict=total_loss_dict,
            grad_norm=grad_norm_value,
        )

        step_metrics = {
            "loss": total_loss,
            "grad_norm": grad_norm_value,
        }
        return step_metrics

    def train(self) -> None:
        """Run the VLM training loop."""
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
