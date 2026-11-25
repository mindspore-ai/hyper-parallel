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
from dataclasses import replace
from itertools import count
from typing import Any, Dict, Iterator, Optional

import torch

from hyper_parallel.data.batching import calculate_num_micro_batches
from hyper_parallel.data.text import build_chat_template
from hyper_parallel.components.checkpoint.model_restore import restore_model_weights
from hyper_parallel.predict import PredictionSummary
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from hyper_parallel.trainer.runtime.logging import create_logger
from hyper_parallel.trainer.runtime.memory import print_device_mem_info
from hyper_parallel.trainer.runtime.memory_profiler import memory_profiler
from hyper_parallel.trainer.runtime.device import synchronize
from hyper_parallel.trainer.runtime.worker_recovery import (
    recovery_constructor,
    recovery_epoch_initialized,
    recovery_train_loop,
    recovery_train_step,
)
from hyper_parallel.trainer.runtime.run_mode import (
    BuildStep, get_build_steps, validate_run_config, validate_run_capabilities,
)
from hyper_parallel.trainer.runtime.prediction import run_prediction
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.data.distributed import DistributedDataLoader
from hyper_parallel.trainer.config import TrainerConfig

logger = create_logger(__name__)


def _target_signature(target: Any) -> Any:
    """Reduce a configured target (or plain value) to a comparable structure.

    Targets compare by their normalized ``_target_`` path and arguments, never
    by object identity: two YAML sections spelling the same tokenizer produce
    distinct ``Target`` objects that must not read as a conflict.
    """
    if target is None or isinstance(target, str):
        return target
    to_dict = getattr(target, "to_dict", None)
    return to_dict() if callable(to_dict) else repr(target)


class TextTrainer:
    """Compose the text training runtime from explicit BaseTrainer stages."""

    base: BaseTrainer

    def __init__(self, config: TrainerConfig) -> None:
        """Validate standalone configuration before starting runtime services."""
        if getattr(config, "run_mode", "train") != "train":
            validate_run_config(config)
        self._initialize(config)

    @recovery_constructor
    def _initialize(self, config: TrainerConfig) -> None:
        """Assemble the components the configured run mode actually executes.

        Training-only steps (optimizer, scheduler, training context, graph
        compiler, callbacks) are skipped for standalone tasks: they cost
        startup time and, for an optimizer with fp32 main parameters, device
        memory comparable to the weights. The plan comes from
        :func:`get_build_steps` so the skip list is declared and testable
        rather than an accident of a shorter constructor.
        """
        run_mode = getattr(config, "run_mode", "train")
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.config = config

        self.base._setup()
        if run_mode != "train":
            validate_run_capabilities(config, self.base.mesh)
        if run_mode == "predict":
            # Independent decode lanes must never inherit training-time FSDP collectives.
            self.base.distributed_setup = replace(self.base.distributed_setup, strategy_config=None)
        self._reset_memory_profiler()
        self.base._build_model()

        # The plan is resolved after the profiler reset and model build so those
        # first stages keep depending on nothing but the sections they read.
        steps = get_build_steps(run_mode)
        if BuildStep.LOSS in steps:
            self.base._build_loss()

        if BuildStep.MODEL_RESTORE in steps:
            self.base.restore_outcome = restore_model_weights(
                self.base.model,
                config,
                self.base.mesh,
            )

        self._build_model_assets()
        self._build_data_pipeline(steps)

        if BuildStep.OPTIMIZER in steps:
            self.base._build_optimizer()
            self.base.model_integration.attach_optimizer(self.base.optimizer)
        if BuildStep.LR_SCHEDULER in steps:
            self.base._build_lr_scheduler()
        if BuildStep.TRAINING_CONTEXT in steps:
            self.base._build_training_context()
        if BuildStep.GRAPH_COMPILER in steps:
            self.base._build_graph_compiler()
        if BuildStep.CALLBACKS in steps:
            self.base._init_callbacks()

    def _build_data_pipeline(self, steps: tuple[str, ...]) -> None:
        """Assemble only the data stages declared by the selected task plan."""
        if BuildStep.DATA_TRANSFORM in steps:
            self._build_data_transform()
        if BuildStep.DATASET in steps:
            self.base._build_dataset()
        if BuildStep.COLLATE in steps:
            self._build_collate_fn()
        if BuildStep.DATALOADER in steps:
            self.base._build_dataloader()
        if BuildStep.GET_BATCH in steps:
            self._build_get_batch()
            self.base.attach_model_integration_data_pipeline()
        if BuildStep.TRAIN_ITERS in steps:
            self.base._compute_train_iters()

    def _reset_memory_profiler(self) -> None:
        """Start allocator snapshots with this run's identities."""
        memory_profiler.reset(
            self.base.config.memory,
            global_rank=self.base.global_rank,
            tp_rank=self.base.mesh.tp_rank,
            dp_rank=self.base.mesh.dp_rank,
        )

    def run(self) -> None:
        """Execute the run mode this Trainer was configured for.

        Raises:
            ValueError: An unknown task mode reached the dispatcher.
        """
        run_mode = self.base.config.run_mode
        if run_mode == "train":
            self.train()
            return
        if run_mode == "predict":
            summary = self.predict()
            logger.info("Prediction report: %s", summary.to_dict())
            return
        raise ValueError(f"TextTrainer.run() does not support run_mode={run_mode!r}")

    def predict(self) -> PredictionSummary:
        """Generate the configured input file and return its committed summary."""
        if self.base.config.run_mode != "predict":
            raise ValueError("TextTrainer.predict() requires run_mode='predict'")
        return run_prediction(self.base)

    def _build_model_assets(self) -> None:
        """Build tokenizer-backed assets from the top-level or dataset config.

        Standalone prediction can configure assets without a dataset,
        so the top-level ``model_assets`` section takes precedence and the
        dataset section is only a compatibility fallback.
        """
        config: TrainerConfig = self.base.config
        assets_config = self._resolve_assets_config()
        if assets_config is None:
            # Unconfigured assets are legal: pre-tokenized Indexed data needs no
            # tokenizer, and training has always allowed that.
            self.base.tokenizer = None
            self.base.chat_template = None
            self.base.model_assets = [self.base.model_config]
            return
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
                    chat_template_kwargs=assets_config.chat_template_kwargs,
                    log_first_rendered_template=config.debug.log_first_rendered_template,
                )
            else:
                if assets_config.chat_template_kwargs or config.debug.log_first_rendered_template:
                    raise ValueError(
                        "dataset.model_assets chat-template options require chat_template='tokenizer'"
                    )
                self.base.chat_template = assets_config.chat_template.build(
                    tokenizer=self.base.tokenizer,
                )

        self.base.model_assets = [self.base.model_config]
        if self.base.tokenizer is not None:
            self.base.model_assets.append(self.base.tokenizer)
        if self.base.chat_template is not None:
            self.base.model_assets.append(self.base.chat_template)

    @staticmethod
    def _is_configured(assets: Any) -> bool:
        """Whether an assets section carries an explicit tokenizer or template.

        ``DatasetConfig.model_assets`` defaults to an empty ``ModelAssetsConfig``,
        so presence alone says nothing: only a populated section is a user choice.
        """
        if assets is None:
            return False
        if getattr(assets, "tokenizer", None) is not None:
            return True
        return getattr(assets, "chat_template", None) is not None

    @staticmethod
    def _same_assets(left: Any, right: Any) -> bool:
        """Compare two configured assets sections by their resolved target paths.

        Targets are compared by ``_target_`` path and arguments rather than
        object identity, so two YAML sections that spell the same tokenizer do
        not conflict.
        """

        def signature(assets: Any) -> Any:
            """Reduce one assets section to a comparable plain structure.

            Args:
                assets: Assets configuration exposing ``to_dict`` or tokenizer/template targets.

            Returns:
                Serialized configuration or normalized tokenizer and template target signatures.
            """
            if hasattr(assets, "to_dict"):
                return assets.to_dict()
            return {
                "tokenizer": _target_signature(getattr(assets, "tokenizer", None)),
                "chat_template": _target_signature(getattr(assets, "chat_template", None)),
            }

        return signature(left) == signature(right)

    def _resolve_assets_config(self) -> Any:
        """Return the effective model-assets section, or ``None`` if unconfigured.

        Explicit top-level assets win; a dataset section is the compatibility
        fallback. Declaring both with different values is a configuration
        error rather than a silent pick.

        Raises:
            ValueError: Both sections are configured and disagree.
        """
        config: TrainerConfig = self.base.config
        dataset_config = self.base.config.dataset
        # ``model_assets`` is new; a legacy config object without it simply has no
        # top-level assets, so the dataset section stays the only source.
        top_assets = getattr(config, "model_assets", None)
        top_level = top_assets if self._is_configured(top_assets) else None
        from_dataset = getattr(dataset_config, "model_assets", None) if dataset_config is not None else None
        from_dataset = from_dataset if self._is_configured(from_dataset) else None
        if top_level is not None and from_dataset is not None and not self._same_assets(top_level, from_dataset):
            raise ValueError(
                "model_assets is configured both at the top level and under "
                "dataset.model_assets with different values; configure one"
            )
        return top_level if top_level is not None else from_dataset

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
            dataloader_context=self.base.dataloader_context,
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

    def on_micro_step_end(self, loss_dict: dict[str, torch.Tensor]) -> None:
        """Dispatch the micro-step-end lifecycle hook.

        Args:
            loss_dict: Named losses and metrics from the completed forward-backward step.
        """
        self.base.on_micro_step_end(loss_dict)

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
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Fetch and execute one forward-backward micro-step.

        Args:
            data_iterator: Iterator providing one raw FB batch per call.

        Returns:
            Loss tensor and named loss tensors for this FB step.
        """
        model_inputs, loss_inputs = self.base.get_batch(data_iterator)
        self.base.current_token_counts = count_loss_token(loss_inputs)
        metric_inputs = {
            **model_inputs,
            **loss_inputs,
            "token_count": self.base.current_token_counts["foundation_tokens"],
        }
        self.on_micro_step_begin(metric_inputs)
        del metric_inputs
        loss, loss_dict = self.base.forward_backward_step(model_inputs, loss_inputs)
        self.on_micro_step_end(loss_dict)

        return loss, loss_dict

    @recovery_train_step
    def train_step(self, data_iterator: Any) -> Dict[str, float]:
        """Execute one text training step.

        Args:
            data_iterator: Iterator providing the step's micro-batches.

        Returns:
            Aggregated loss and gradient norm for the completed step.
        """
        memory_profiler.step()
        num_micro_steps = self.base.num_micro_batches
        self.base.begin_loss_step()

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
            )

            total_loss += loss.item()
            for loss_name, loss_value in loss_dict.items():
                total_loss_dict[loss_name] += loss_value.item()

        total_loss, grad_norm = self.base.prepare_optimizer_step(total_loss, total_loss_dict)
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

    @recovery_epoch_initialized
    def _create_data_iterator(self) -> Optional[Iterator[Any]]:
        """Construct the native iterator before consuming this epoch's data."""
        loader = self.base.train_dataloader
        return iter(loader) if loader is not None else None

    def _train_epoch(self, epoch: int, balanced_loader: bool) -> bool:
        """Consume complete optimizer steps and report whether the source epoch ended."""
        data_iterator = self._create_data_iterator()
        start_step = self.base.state.global_step - epoch * self.base.train_steps
        train_steps = min(self.base.train_steps, self.base.train_iters - epoch * self.base.train_steps)
        if balanced_loader:
            start_step = 0
            train_steps = self.base.train_iters - self.base.state.global_step
        for _ in range(start_step, train_steps):
            try:
                self.train_step(data_iterator)
            except StopIteration:
                if balanced_loader:
                    optimizers = self.base.optimizer
                    for optimizer in optimizers if isinstance(optimizers, list) else [optimizers]:
                        optimizer.zero_grad()
                logger.info(
                    "epoch:%s Dataloader finished with drop_last %s", epoch, self.base.config.dataloader.drop_last
                )
                return True
        return False

    @recovery_train_loop
    def train(self) -> None:
        """Run the configured global optimizer steps by Dataset epoch."""
        try:
            self.on_train_begin()
            logger.info(
                "Rank%s Start training. Global step: %s. Train iters: %s. Start epoch: %s. Train epochs: %s.",
                self.base.local_rank,
                self.base.state.global_step,
                self.base.train_iters,
                self.base.state.epoch,
                self.base.train_epochs,
            )

            train_dataloader = self.base.train_dataloader
            balanced_loader = isinstance(train_dataloader, DistributedDataLoader)
            start_epoch = self.base.state.epoch
            epochs = count(start_epoch) if balanced_loader else range(start_epoch, self.base.train_epochs)
            for epoch in epochs:
                if self.base.state.global_step >= self.base.train_iters:
                    break
                # The first collective iterator may hold a restored checkpoint cursor.
                if hasattr(train_dataloader, "set_epoch") and (not balanced_loader or epoch != start_epoch):
                    train_dataloader.set_epoch(epoch)

                self.base.state.epoch = epoch
                self.on_epoch_begin()
                epoch_start_step = self.base.state.global_step
                epoch_exhausted = self._train_epoch(epoch, balanced_loader)

                if balanced_loader and epoch != start_epoch and self.base.state.global_step == epoch_start_step:
                    raise ValueError("Indexed source epoch cannot provide one complete optimizer step")

                self.on_epoch_end()
                self.base.state.epoch = epoch + 1 if not balanced_loader or epoch_exhausted else epoch
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

        if balanced_loader:
            try:
                train_dataloader.wait_for_prefetch()
            finally:
                train_dataloader.close()

        synchronize()
        self.base.destroy_distributed()
