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
"""Expose HyperParallel execution components to the VeRL engine."""

from __future__ import annotations

from typing import Any

import torch

from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.data.parallel import create_dataloader_parallel_context
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.config import TrainerConfig
from hyper_parallel.trainer.runtime.device import get_device_type, get_torch_device
from hyper_parallel.trainer.runtime.distributed import get_global_rank_safe, get_local_rank_safe, get_world_size_safe
from hyper_parallel.trainer.runtime.logging import enable_third_party_logging, setup_logging
from hyper_parallel.trainer.runtime.random import enable_high_precision_for_bf16, set_seed


class VerlTrainerAdapter(BaseTrainer):
    """Use native trainer components with caller-owned batches and objectives."""

    # VeRL supplies the batches and objective that BaseTrainer would otherwise build.
    # pylint: disable-next=super-init-not-called
    def __init__(
        self,
        config: TrainerConfig,
        *,
        distributed_setup: Any,
        tokenizer: Any,
        forward_only: bool = False,
    ) -> None:
        """Select native construction stages without building an SFT data pipeline."""
        self.config = config
        self.distributed_setup = distributed_setup
        self.tokenizer = tokenizer
        self._setup()
        self._build_model()
        self.model_assets = [self.model_config]
        if self.tokenizer is not None:
            self.model_assets.append(self.tokenizer)
        # VeRL already supplies each model rank with its role's batch.
        self.dataloader_context = create_dataloader_parallel_context(self.mesh_context, tp_source_rank=None)
        self._build_collate_fn()
        self._build_get_batch()
        if forward_only:
            self.model.requires_grad_(False)
            self.model.eval()
        else:
            self._build_optimizer()
            if self.config.lr_scheduler is not None:
                self._build_lr_scheduler()
        self._build_training_context()

    @property
    def mesh(self) -> Any:
        """Expose mesh_context to inherited BaseTrainer methods."""
        return self.mesh_context

    def _setup(self) -> None:
        """Reuse VeRL's distributed environment and initialize the local runtime."""
        config = self.config
        self.mesh_context = self.distributed_setup.mesh_context
        self.device_mesh = self.mesh_context.device_mesh
        self.train_iters = config.training.train_iters
        setup_logging()
        self.local_rank = get_local_rank_safe()
        self.global_rank = get_global_rank_safe()
        self.world_size = get_world_size_safe()
        self.device = torch.device(get_device_type(), self.local_rank)
        get_torch_device().set_device(self.device)
        if config.training.seed is None:
            config.training.seed = self.default_seed
        set_seed(config.training.seed, config.training.enable_full_determinism)
        enable_high_precision_for_bf16()
        if self.local_rank == 0:
            enable_third_party_logging()
        if config.training.enable_full_determinism:
            torch.use_deterministic_algorithms(True, warn_only=False)

    def _build_collate_fn(self) -> None:
        """Build the recipe collator for caller-owned micro-batches."""
        self.collate_fn = self.config.dataloader.collate_fn.build(mesh_context=self.mesh_context)

    def _build_get_batch(self) -> None:
        """Build native input processing without creating a DataLoader."""
        loader = self.config.dataloader
        self.get_batch = loader.get_batch.build(
            mesh_context=self.mesh_context,
            dataloader_context=self.dataloader_context,
            device=self.device,
            tokenizer=self.tokenizer,
            data_config=getattr(self.config.dataset, "data_config", {}),
            pp_shared_data=bool(getattr(loader, "pp_shared_data", False)),
        )

    def optimizer_zero_grad(self) -> None:
        """Release optimizer gradients when this role owns an optimizer."""
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def forward_step(
        self,
        token_sequences: Any,
        *,
        global_step: int,
        micro_step: int,
        num_micro_steps: int,
        forward_only: bool,
    ) -> tuple[Any, torch.Tensor]:
        """Prepare one micro-batch and return model outputs with shifted labels."""
        samples = [
            {
                "input_ids": tokens,
                "labels": torch.nn.functional.pad(tokens[1:], (0, 1), value=-100),
            }
            for tokens in token_sequences.unbind()
        ]
        model_inputs, loss_inputs = self.get_batch(iter([self.collate_fn(samples)]))
        if not forward_only:
            self.configure_fsdp_gradient_sync(micro_step, num_micro_steps)
        if self.fsdp_runtime_diagnostics is not None:
            self.fsdp_runtime_diagnostics.begin_micro_step(global_step, micro_step)
        model_inputs = self.preforward(model_inputs)
        loss_inputs = self.preforward(loss_inputs)
        model_inputs.pop("labels", None)
        model_inputs.pop("shift_labels", None)
        context = self.model_fwd_context() if callable(self.model_fwd_context) else self.model_fwd_context
        with context:
            outputs = self.model(**model_inputs, use_cache=False)
        return outputs, loss_inputs["labels"]

    def optimizer_step(self) -> float:
        """Clip and update once at VeRL's optimizer boundary, without rescaling loss."""
        grad_norm = self.clip_grad_norm()
        optimizers = self.optimizer if isinstance(self.optimizer, list) else [self.optimizer]
        for optimizer in optimizers:
            with SkipDTensorDispatch(no_skip={torch.zeros_like}):
                optimizer.step()
        return float(grad_norm)

    def lr_scheduler_step(self) -> float:
        """Advance schedulers only when VeRL requests its scheduler stage."""
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return self.optimizer.param_groups[0]["lr"]
