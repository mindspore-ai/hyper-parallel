# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict
from typing import Any, Dict

from hyper_parallel import SkipDTensorDispatch, hsdp_sync_stream
from hyper_parallel.core.utils import clip_grad_norm_
from hyper_models.components.datasets.llm import build_chat_template
from hyper_models.components.loss.loss_utils import count_loss_token
from hyper_models.components.utils import helper
from hyper_models.components.utils.device import synchronize
from hyper_models.trainer.base import BaseTrainer, HyperIter
from hyper_models.trainer.config import TrainerConfig

logger = helper.create_logger(__name__)


class TextTrainer:
    """Compose the text training runtime from explicit BaseTrainer stages."""

    base: BaseTrainer

    def __init__(self, config: TrainerConfig) -> None:
        """Build the text Trainer in the same explicit order as VeOmni."""
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
        self.base._compute_train_steps()

        self.base._build_optimizer()
        self.base._build_lr_scheduler()
        self.base._build_training_context()
        self.base._init_callbacks()

    def _build_model_assets(self) -> None:
        """Build tokenizer-backed assets for text training."""
        config: TrainerConfig = self.base.config
        assets_config = config.model_assets
        tokenizer_target = assets_config.tokenizer
        if tokenizer_target is None:
            self.base.tokenizer = None
        elif hasattr(tokenizer_target, "pretrained_model_name_or_path"):
            self.base.tokenizer = tokenizer_target.build()
        else:
            self.base.tokenizer = tokenizer_target.build(
                pretrained_model_name_or_path=config.model.pretrained_model_name_or_path,
            )

        self.base.chat_template = None
        if assets_config.datasets_type == "conversation":
            if self.base.tokenizer is None:
                raise ValueError("model_assets.tokenizer is required for conversation data")
            if assets_config.chat_template is None:
                raise ValueError("model_assets.chat_template is required for conversation data")
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
        if self.base.config.data_transform is None:
            self.base.data_transform = None
            return
        self.base.data_transform = self.base.config.data_transform.build(
            tokenizer=self.base.tokenizer,
            chat_template=self.base.chat_template,
        )

    def _build_collate_fn(self) -> None:
        """Build the text collator and gradient-accumulation batch count."""
        from hyper_models.components.datasets.build_collate_fn import calculate_num_micro_batches

        if self.base.config.collate_fn is None:
            raise ValueError("collate_fn must define a build target")
        training_config = self.base.config.training
        self.base.num_micro_batches = calculate_num_micro_batches(
            global_batch_size=training_config.global_batch_size,
            micro_batch_size=training_config.micro_batch_size,
            dp_world_size=self.base.mesh.dp_size,
        )
        self.base.collate_fn = self.base.config.collate_fn.build()

    def _build_get_batch(self) -> None:
        """Build the DataLoader-to-LLM batch adapter."""
        from hyper_models.components.datasets.llm.get_batch import build_llm_get_batch
        from hyper_models.components.datasets.parallel import create_batch_parallel_context

        config = self.base.config
        pp_shared_data = bool(getattr(config.dataloader, "pp_shared_data", False))
        data_config = getattr(config.dataset, "data_config", {})
        tokenizer = getattr(self.base, "tokenizer", None)
        eod_token_id = getattr(
            tokenizer,
            "eod",
            getattr(tokenizer, "eos_token_id", 0),
        )
        parallel_context = create_batch_parallel_context(
            self.base.mesh,
            pp_shared_data=pp_shared_data,
        )
        get_batch_builder = (
            build_llm_get_batch
            if config.get_batch is None
            else config.get_batch.build
        )
        self.base.get_batch = get_batch_builder(
            parallel_context=parallel_context,
            device=self.base.device,
            eod_token_id=eod_token_id,
            reset_position_ids=bool(data_config.get("reset_position_ids", False)),
            reset_attention_mask=bool(data_config.get("reset_attention_mask", False)),
            eod_mask_loss=bool(data_config.get("eod_mask_loss", False)),
            create_attention_mask=bool(
                data_config.get("create_attention_mask_in_dataloader", True)
            ),
        )

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

    def on_step_begin(self, micro_batches: Any = None) -> None:
        """Dispatch the step-begin lifecycle hook."""
        self.base.on_step_begin(micro_batches=micro_batches)

    def on_step_end(
            self,
            loss: Any = None,
            loss_dict: Any = None,
            grad_norm: Any = None,
    ) -> None:
        """Dispatch the step-end lifecycle hook."""
        self.base.on_step_end(
            loss=loss,
            loss_dict=loss_dict,
            grad_norm=grad_norm,
        )

    def train_step(
            self,
            data_iterator: Any,
    ) -> Dict[str, float]:
        """Execute one text training step."""
        config = self.base.config
        first_micro_batch = self.base.get_batch(data_iterator)
        self.base.state.global_step += 1

        self.on_step_begin(micro_batches=None)
        synchronize()

        total_loss = 0.0
        total_loss_dict = defaultdict(int)
        num_micro_steps = self.base.num_micro_batches

        for micro_step in range(num_micro_steps):
            micro_batch = (
                first_micro_batch
                if micro_step == 0
                else self.base.get_batch(data_iterator)
            )
            self.base.model_reshard(micro_step, num_micro_steps)
            self.base._configure_fsdp_gradient_sync(
                micro_step,
                num_micro_steps,
            )
            self.base.micro_batch_token_len = count_loss_token(
                micro_batch.loss_count_inputs()
            )
            self.base.micro_batches_token_len = {
                name: token_count * num_micro_steps
                for name, token_count in self.base.micro_batch_token_len.items()
            }
            loss, loss_dict = self.base.forward_backward_step(micro_batch)

            total_loss += loss.item()
            for loss_name, loss_value in loss_dict.items():
                total_loss_dict[loss_name] += loss_value.item()

        hsdp_sync_stream()
        grad_norm = clip_grad_norm_(
            self.base.model,
            config.training.max_grad_norm,
        )

        optimizers = (
            self.base.optimizer
            if isinstance(self.base.optimizer, list)
            else [self.base.optimizer]
        )
        for optimizer in optimizers:
            with SkipDTensorDispatch():
                optimizer.step()
            optimizer.zero_grad()

        schedulers = (
            self.base.lr_scheduler
            if isinstance(self.base.lr_scheduler, list)
            else (
                [self.base.lr_scheduler]
                if self.base.lr_scheduler is not None
                else []
            )
        )
        for scheduler in schedulers:
            scheduler.step()

        grad_norm_value = float(grad_norm)
        self.on_step_end(
            loss=total_loss,
            loss_dict=total_loss_dict,
            grad_norm=grad_norm_value,
        )
        return {
            "loss": total_loss,
            "grad_norm": grad_norm_value,
        }

    def train(self) -> None:
        """Run the text training loop."""
        config = self.base.config
        self.on_train_begin()
        logger.info(
            "Rank%s Start training. Start step: %s. Train steps: %s. "
            "Start epoch: %s. Train epochs: %s.",
            self.base.local_rank,
            self.base.start_step,
            self.base.train_steps,
            self.base.start_epoch,
            config.training.num_train_epochs,
        )

        for epoch in range(
                self.base.start_epoch,
                config.training.num_train_epochs,
        ):
            if hasattr(self.base.train_dataloader, "set_epoch"):
                self.base.train_dataloader.set_epoch(epoch)
            self.base.state.epoch = epoch
            self.on_epoch_begin()

            self.base.data_iterator = HyperIter(
                self.base.train_dataloader,
                use_background_prefetcher=(
                    config.dataloader.use_background_prefetcher
                ),
            )

            for _ in range(self.base.start_step, self.base.train_steps):
                if self.base.state.global_step >= self.base.train_steps:
                    break
                try:
                    self.train_step(self.base.data_iterator)
                except StopIteration:
                    logger.info(
                        "epoch:%s Dataloader finished with drop_last %s",
                        epoch,
                        config.dataloader.drop_last,
                    )
                    break

            self.on_epoch_end()
            self.base.start_step = 0
            helper.print_device_mem_info(
                f"VRAM usage after epoch {epoch + 1}"
            )

            if config.dataloader.use_background_prefetcher:
                self.base.data_iterator.stop()
            if self.base.state.global_step >= self.base.train_steps:
                break

        self.on_train_end()

        if (hasattr(self.base, "data_iterator") and config.dataloader.use_background_prefetcher):
            self.base.data_iterator.stop()

        synchronize()
        self.base.destroy_distributed()


__all__ = ["TextTrainer"]
