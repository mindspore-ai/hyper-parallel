# Copyright 2025-2026 Bytedance Ltd. and/or its affiliates
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

"""
Base Trainer class for distributed training.

This module provides the BaseTrainer class which serves as the foundation
for all trainer implementations. Subclasses can override specific methods
to customize training behavior.

Features:
    - Callback system for extensible training hooks
    - Distributed training support
    - Gradient accumulation
    - Checkpointing
"""

import json
import logging
import os
import queue
import threading
from abc import ABC
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin
from transformers.modeling_outputs import ModelOutput

from hyper_parallel import HSDPModule, SkipDTensorDispatch, hsdp_sync_stream
from hyper_parallel.core.utils import clip_grad_norm_
from .config import TrainerConfig, save_configs
from ..components.data.chat_template import ChatTemplate
from ..components.data.data_collator import calculate_num_micro_batches
from ..components.distributed.init_utils import get_local_rank_safe, get_global_rank_safe, get_world_size_safe
from ..components.distributed.infrastructure import (
    create_distributed_setup_from_config,
    destroy_process_group,
    setup_logging,
    initialize_distributed,
)
from ..components.loss.loss_utils import count_loss_token, mean_global_loss
from ..components.utils import helper
from ..components.utils.device import synchronize, get_torch_device, get_device_type

from .callbacks import TempLogCallback, TrainerState


logger = logging.getLogger(__name__)


class BackgroundPrefetcher:
    """
    Prefetches batches from a dataloader in a background thread to overlap data loading
    with GPU computation. Synchronizes dataloader state for correct checkpointing.
    """

    def __init__(self, dataloader, maxsize=1):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.queue = queue.Queue(maxsize=maxsize)
        self.stop_event = threading.Event()
        self.original_state_dict = getattr(dataloader, "state_dict", None)
        self.current_state = None
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True
        self.thread.start()

    def _worker(self):
        try:
            while not self.stop_event.is_set():
                try:
                    item = next(self.iterator)
                except StopIteration:
                    self.queue.put((StopIteration, None))
                    break

                # Ensure we capture the state so that subsequent dataloader advances
                # don't mutate the captured state in-place. The underlying dataloader's
                # state_dict() should handle deepcopying if necessary.
                state = self.original_state_dict() if self.original_state_dict else None
                self.queue.put((item, state))
        except Exception as e:
            self.queue.put((e, None))

    def __iter__(self):
        return self

    def __next__(self):
        res = self.queue.get()
        if isinstance(res, tuple) and len(res) == 2:
            item, state = res
            if item is StopIteration:
                raise StopIteration
            if isinstance(item, Exception):
                raise item
            self.current_state = state
            return item
        else:
            if res is StopIteration:
                raise StopIteration
            if isinstance(res, Exception):
                raise res
            return res

    def state_dict(self):
        if self.current_state is not None:
            return self.current_state
        if self.original_state_dict:
            return self.original_state_dict()
        return {}

    def stop(self, timeout: float = 5.0):
        self.stop_event.set()
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except queue.Empty:
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                logger.warning("BackgroundPrefetcher worker thread did not terminate within timeout.")


class HyperIter:
    """
    A unified iterator wrapper that handles both standard iteration and background prefetching.
    """

    def __init__(self, dataloader, use_background_prefetcher: bool = False, maxsize: int = 1):
        self.dataloader = dataloader
        self.use_background_prefetcher = use_background_prefetcher
        if use_background_prefetcher:
            self.iterator = BackgroundPrefetcher(dataloader, maxsize=maxsize)
        else:
            self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator)

    def stop(self, timeout: float = 5.0):
        if self.use_background_prefetcher and hasattr(self.iterator, "stop"):
            self.iterator.stop(timeout=timeout)

    def state_dict(self):
        if self.use_background_prefetcher and hasattr(self.iterator, "state_dict"):
            return self.iterator.state_dict()
        if hasattr(self.dataloader, "state_dict"):
            return self.dataloader.state_dict()
        return {}


class BaseTrainer(Stateful, ABC):
    """
    Base trainer class for distributed model training.

    This class provides the core training infrastructure including:
    - Distributed initialization and parallelism setup
    - Model, optimizer, and scheduler initialization
    - Training step execution with gradient accumulation
    - Checkpointing and fault tolerance
    - Metrics logging

    Subclasses can override the following methods to customize behavior:
    - `_build_distributed_setup()`: Connect AutoModel DistributedSetup/MeshContext
    - `_build_model()`: Connect AutoModel from_pretrained/parallelize
    - `_build_model_assets()`: Build tokenizer/processor assets
    - `_build_data_transform()`: Build modality-specific sample transforms
    - `forward_backward_step()`: Customize forward/backward logic
    - `train_step()`: Customize training step execution
    - `train()`: Train the model

    Callback Hooks:
        The trainer calls callback methods at various stages:
        - evaluate_callback: evaluation callback
        - trace_callback: tracing callback (meter, wandb, tqdm, profile)
        - checkpoint_callback: checkpointing callback
    """

    # Core configs
    config: TrainerConfig
    device: torch.device

    # AutoModel distributed setup.
    # These objects have one owner: ``../components/distributed/config.py`` and
    # ``../components/distributed/mesh.py``. Trainer code consumes them but must
    # not create a second DeviceMesh or a VeOmni ParallelState.
    dist_info: Any
    distributed_setup: Any
    mesh_context: Any
    device_mesh: Any
    moe_mesh: Any

    # Data
    train_dataset: Dataset
    # collate_fn: DataCollator
    # train_dataloader: DistributedDataloader
    collate_fn: Any
    train_dataloader: Any

    # Model
    model: PreTrainedModel = None
    model_parts: List[torch.nn.Module] | None = None
    hsdp_model_parts: List[torch.nn.Module] = []
    model_config: PretrainedConfig = PretrainedConfig()
    tokenizer: PreTrainedTokenizerBase = None
    processor: ProcessorMixin = None
    chat_template: ChatTemplate | None = None
    model_assets: List[Any] = []

    # Training components
    optimizer: Optimizer = None
    lr_scheduler: LRScheduler = None

    # Training context
    model_fwd_context: Any
    model_bwd_context: Any

    # # Runtime metrics, controlled by trace_callback
    # environ_meter: helper.EnvironMeter  # see in trace_callback.EnvironMeterCallback
    # step_env_metrics: Dict[str, Any]  # mfu, flops, tokens, etc
    # step_train_metrics: Dict[str, Any]  # loss, grad_norm, lr, etc

    # # Checkpointer
    # checkpointer: CheckpointerBase  # see in checkpoint_callback.CheckpointerCallback

    # Callback system
    state: TrainerState

    # Training states
    train_steps: int = 0  # total training steps
    start_epoch: int = 0  # start epoch
    start_step: int = 0  # start step

    # default seed if deterministic is enabled but no seed is provided
    default_seed: int = 42

    def __init__(self, config: TrainerConfig):
        """
        Initialize the trainer.

        Args:
            args: Global Arguments
                Should have attributes: model, data, train
                model: ModelArguments
                data: DataArguments
                train: TrainingArguments
        """

        self.config: TrainerConfig = config

        # General setup owns distributed initialization as its first stage.
        # ``_build_distributed_setup`` remains a reserved backend hook; the
        # trainer only defines its contract here.
        self._setup()

        # Reserved atomic model-build interface.
        #
        # Owner:
        #   - ``../_transformers/auto_model.py::HyperAutoModel*.from_pretrained``
        #   - ``../_transformers/infrastructure.py`` for materialization,
        #     checkpoint loading, PEFT/quantization, and parallelize.
        #
        # Contract:
        #   - consume the ``DistributedSetup`` created above;
        #   - set ``model`` and the resolved checkpoint ``model_config``;
        #   - return a materialized, weight-loaded, already-parallelized model;
        #   - never call a second trainer-side ``build_parallelize_model``.
        self._build_model()

        # Trainer-owned components are constructed only after AutoModel has
        # finalized parameter identity, sharding, freezing, and tied weights.
        # ``_build_model_assets`` loads the tokenizer/processor together with
        # their serialized chat template. The dataset/data-transform layer
        # consumes that template; BaseTrainer neither defines nor interprets
        # model-specific chat protocols.
        self._build_model_assets()
        self._build_data_transform()
        self._build_dataset()
        self._build_collate_fn()
        self._build_dataloader()
        self._compute_train_steps()
        self._build_optimizer()
        self._build_lr_scheduler()
        self._build_training_context()
        self._init_callbacks()

    def _setup(self):
        # log args
        setup_logging()

        # Initialize distributed process group
        backend = getattr(self.config.training, "backend", "nccl")
        initialize_distributed(backend=backend)

        self.local_rank = get_local_rank_safe()
        self.global_rank = get_global_rank_safe()
        self.world_size = get_world_size_safe()

        if self.global_rank == 0:
            logger.info(
                "Trainer config:\n%s",
                json.dumps(self.config.to_dict(), indent=2),
            )

        # init distributed environment
        device_str = f"{get_device_type()}:{self.local_rank}"
        get_torch_device().set_device(device_str)
        self.device = torch.device(device_str)

        logger.info("Process rank: %s, world size: %s", self.global_rank, self.world_size)

        # Register distributed_setup
        self.distributed_setup = create_distributed_setup_from_config(self.config)
        self.mesh = self.distributed_setup.mesh_context
        self.device_mesh = self.mesh.device_mesh
        self.dp_cp_mesh = self.mesh.dp_cp_mesh

        # Set random seed
        helper.set_seed(self.config.training.seed, self.config.training.enable_full_determinism)

        # Enable high precision for bf16
        helper.enable_high_precision_for_bf16()

        # Enable third party logging
        if self.local_rank == 0:
            helper.enable_third_party_logging()

        # Save arguments
        if self.global_rank == 0:
            save_configs(self.config, self.config.checkpoint.checkpoint_dir)

        # # Gradient checkpointing debug
        # set_checkpoint_debug_enabled(self.config.gradient_checkpointing.debug)

    def _build_model(self) -> None:
        """Build the model and derive Trainer-owned runtime state."""
        self.peft_config = self.config.peft
        self.model = self.config.model.build(
            distributed_setup=self.distributed_setup,
            peft_config=self.peft_config,
        )
        self.model_config = self.model.config
        model_parts = getattr(self.model, "parts", None)
        self.model_parts = list(model_parts) if model_parts is not None else [self.model]
        self.hsdp_model_parts = [
            model_part
            for model_part in self.model_parts
            if isinstance(model_part, HSDPModule)
        ]

    def _build_model_assets(self):
        """Build model assets for the temporary dummy-data training path."""
        self.tokenizer = (
            self.config.tokenizer.build()
            if self.config.tokenizer is not None
            else None
        )
        self.chat_template = None
        self.model_assets = [self.model_config]
        if self.tokenizer is not None:
            self.model_assets.append(self.tokenizer)

    def _build_data_transform(self) -> None:
        """Build the configured transform from model assets."""
        if self.config.data_transform is None:
            self.data_transform = None
            return
        self.data_transform = self.config.data_transform.build(
            tokenizer=self.tokenizer,
            processor=self.processor,
            chat_template=self.chat_template,
            model_config=self.model_config,
        )

    def _build_dataset(self):
        """Build dataset-owned runtime state through the configured target."""
        if self.config.dataset is None:
            raise ValueError("config.dataset must define a build target")
        self.train_dataset = self.config.dataset.build(
            transform=self.data_transform,
        )

    def _build_collate_fn(self):
        """Build the collator through the configured target."""
        if self.config.collate_fn is None:
            raise ValueError("config.collate_fn must define a build target")
        training_config = self.config.training
        num_micro_batches = calculate_num_micro_batches(
            global_batch_size=training_config.global_batch_size,
            micro_batch_size=training_config.micro_batch_size,
            dp_world_size=self.mesh.dp_size,
        )
        self.collate_fn = self.config.collate_fn.build(
            num_micro_batch=num_micro_batches,
        )

    def _build_dataloader(self):
        """Build the training dataloader through the configured target."""
        if self.config.dataloader is None:
            raise ValueError("config.dataloader must define a build target")
        local_step_batch_size = (
            self.config.training.global_batch_size // self.mesh.dp_size
        )
        seed = self.config.training.seed
        if seed is None:
            seed = self.default_seed
        self.train_dataloader = self.config.dataloader.build(
            dataset=self.train_dataset,
            collate_fn=self.collate_fn,
            batch_size=local_step_batch_size,
            dp_world_size=self.mesh.dp_size,
            dp_rank=self.mesh.dp_rank,
            seed=seed,
        )

    def _compute_train_steps(self) -> None:
        """Resolve the total number of optimizer steps."""
        training_config = self.config.training
        max_steps = training_config.max_steps
        if max_steps is not None:
            if (
                isinstance(max_steps, bool)
                or not isinstance(max_steps, int)
                or max_steps <= 0
            ):
                raise ValueError("training.max_steps must be a positive integer")
            self.train_steps = max_steps
            return

        try:
            steps_per_epoch = len(self.train_dataloader)
        except TypeError as exc:
            raise ValueError(
                "training.max_steps must be configured when the dataloader "
                "does not have a finite length"
            ) from exc

        if steps_per_epoch <= 0:
            raise ValueError("train_dataloader must provide at least one step per epoch")
        if (
            isinstance(training_config.num_train_epochs, bool)
            or not isinstance(training_config.num_train_epochs, int)
            or training_config.num_train_epochs <= 0
        ):
            raise ValueError("training.num_train_epochs must be a positive integer")

        self.train_steps = steps_per_epoch * training_config.num_train_epochs

    def _build_optimizer(self):
        config: TrainerConfig = self.config
        self.optimizer = config.optimizer.build(
            model=self.model,
            device_mesh=self.device_mesh,
            is_peft=self.peft_config is not None,
        )

    def _build_lr_scheduler(self):
        config: TrainerConfig = self.config
        self.lr_scheduler = (
            config.lr_scheduler.build(
                optimizer=self.optimizer,
                train_steps=self.train_steps,
            )
            if config.lr_scheduler is not None
            else None
        )

    def _build_training_context(self):
        """Build training context for distributed training."""
        self.model_fwd_context, self.model_bwd_context = nullcontext(), nullcontext()

    def _init_callbacks(self):
        """Initialize callbacks."""
        self.temp_log_callback = TempLogCallback(self)
        self._callbacks = [
            self.temp_log_callback,
        ]
        self.state = TrainerState()

    def on_train_begin(self):
        for callback in self._callbacks:
            callback.on_train_begin(self.state)

    def on_train_end(self):
        for callback in self._callbacks:
            callback.on_train_end(self.state)

    def on_epoch_begin(self):
        for callback in self._callbacks:
            callback.on_epoch_begin(self.state)

    def on_epoch_end(self):
        for callback in self._callbacks:
            callback.on_epoch_end(self.state)

    def on_step_begin(self, micro_batches=None, **kwargs):
        for callback in self._callbacks:
            callback.on_step_begin(self.state, micro_batches=micro_batches, **kwargs)

    def on_step_end(self, loss=None, loss_dict=None, grad_norm=None):
        for callback in self._callbacks:
            callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)

    def preforward(self, micro_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Preprocess micro batches before forward pass.

        Tensors are moved to ``self.device`` non-blockingly. Nested dicts
        (e.g. ``multimodal_metadata`` emitted by ``PackingCollator``) are
        recursed so inner tensor values land on the device too; Python ints
        / lists / etc. pass through unchanged.
        """

        def _to_device(v: Any) -> Any:
            if isinstance(v, torch.Tensor):
                return v.to(self.device, non_blocking=True)
            if isinstance(v, dict):
                return {k: _to_device(vv) for k, vv in v.items()}
            return v

        # chunk_mbs_config = getattr(self.config.training, "chunk_mbs_config", None)
        # self._chunk_mbs_ranges = build_chunk_mbs_ranges(micro_batch, chunk_mbs_config)
        micro_batch = {k: _to_device(v) for k, v in micro_batch.items()} # type: ignore
        # if getattr(self, "LOG_SAMPLE", True):
        #     helper.print_example(example=micro_batch, rank=self.config.training.local_rank)
        #     self.LOG_SAMPLE = False
        return micro_batch

    def postforward(
        self, outputs: ModelOutput, micro_batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Postprocess model outputs after forward pass."""
        loss_dict: Dict[str, torch.Tensor] = mean_global_loss(
            outputs.loss,
            self.micro_batch_token_len,
            self.micro_batches_token_len,
            device_mesh=self.mesh,
        )
        loss = torch.stack(list(loss_dict.values())).sum()
        return loss, loss_dict

    def forward_backward_step(
        self, micro_batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        channel_loss_callback = getattr(self, "channel_loss_callback", None)
        micro_step_context = (
            channel_loss_callback.micro_step_context(self.state, micro_batch)
            if channel_loss_callback is not None
            else nullcontext()
        )
        with micro_step_context:
            micro_batch = self.preforward(micro_batch)
            if channel_loss_callback is not None:
                channel_loss_callback.strip_model_inputs(micro_batch)

            with self.model_fwd_context:
                outputs: ModelOutput = self.model(**micro_batch, use_cache=False)

            # with use_parallel_state("base"):
            loss, loss_dict = self.postforward(outputs, micro_batch)

            # Backward pass
            with self.model_bwd_context:
                loss.backward()

            del micro_batch
            return loss, loss_dict

    def model_reshard(self, micro_step: int, num_micro_steps: int):
        """Reshard model after backward pass."""
        config: TrainerConfig = self.config
        if (
            config.fsdp_config.reshard_after_backward is False
            and num_micro_steps > 1
        ):
            if micro_step == 0:
                for model_part in self.hsdp_model_parts:
                    model_part.set_reshard_after_backward(False)
            elif micro_step == num_micro_steps - 1:
                for model_part in self.hsdp_model_parts:
                    model_part.set_reshard_after_backward(True)

    def _configure_fsdp_gradient_sync(self, micro_step: int, num_micro_steps: int):
        config: TrainerConfig = self.config
        if (
            config.fsdp_config.dp_shard_size > 1
            and num_micro_steps > 1
        ):
            is_last_micro_batch = micro_step == num_micro_steps - 1
            is_hsdp = self.mesh.dp_replicate_size > 1
            for model_part in self.hsdp_model_parts:
                model_part.set_requires_gradient_sync(is_last_micro_batch)
                model_part.set_is_last_backward(is_last_micro_batch)
                if is_hsdp:
                    model_part.set_requires_all_reduce(is_last_micro_batch)

    def train_step(
        self,
        data_iterator: Any,
    ) -> Dict[str, float]:
        config = self.config

        micro_batches: List[Dict[str, Any]] = next(data_iterator)
        self.state.global_step += 1

        self.on_step_begin(micro_batches=micro_batches)

        # Forward and backward for each micro batch
        synchronize()

        total_loss = 0.0
        total_loss_dict = defaultdict(int)

        # token num for fixed_ce_loss in postforward
        self.micro_batches_token_len = count_loss_token(micro_batches)
        num_micro_steps = len(micro_batches)
        # forward and backward pass with gradient_accumulationsteps
        for micro_step, micro_batch in enumerate(micro_batches):
            self.model_reshard(micro_step, num_micro_steps)
            self._configure_fsdp_gradient_sync(micro_step, num_micro_steps)
            loss: torch.Tensor
            loss_dict: Dict[str, torch.Tensor]
            # token num for fixed_ce_loss in postforward
            self.micro_batch_token_len = count_loss_token(micro_batch)
            loss, loss_dict = self.forward_backward_step(micro_batch)

            total_loss += loss.item()
            for k, v in loss_dict.items():
                total_loss_dict[k] += v.item()

        # Wait for hsdp gradient handle to be completed
        hsdp_sync_stream()

        # Gradient clipping (reads FSDP/EP groups from current ParallelState)
        grad_norm = clip_grad_norm_(
            self.model,
            config.training.max_grad_norm,
        )

        # Optimizer and scheduler step
        optimizers = self.optimizer if isinstance(self.optimizer, list) else [self.optimizer]
        for optimizer in optimizers:
            with SkipDTensorDispatch():
                optimizer.step()
            optimizer.zero_grad()

        schedulers = (
            self.lr_scheduler
            if isinstance(self.lr_scheduler, list)
            else ([self.lr_scheduler] if self.lr_scheduler is not None else [])
        )
        for scheduler in schedulers:
            scheduler.step()

        grad_norm_value = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        self.on_step_end(loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm_value)
        return {
            "loss": total_loss,
            "grad_norm": grad_norm_value,
        }

    def destroy_distributed(self):
        if not dist.is_available() or not dist.is_initialized():
            return

        helper.empty_cache()
        dist.barrier()

        synchronize()
        destroy_process_group()

    def train(self):
        config: TrainerConfig = self.config
        self.on_train_begin()
        logger.info(
            f"Rank{self.local_rank} Start training. "
            f"Start step: {self.start_step}. "
            f"Train steps: {self.train_steps}. "
            f"Start epoch: {self.start_epoch}. "
            f"Train epochs: {config.training.num_train_epochs}."
        )

        for epoch in range(self.start_epoch, config.training.num_train_epochs):
            if hasattr(self.train_dataloader, "set_epoch"):
                self.train_dataloader.set_epoch(epoch)
            self.state.epoch = epoch

            self.on_epoch_begin()

            # Create a batch generator
            self.data_iterator = HyperIter(
                self.train_dataloader, use_background_prefetcher=config.dataloader.use_background_prefetcher
            )

            for _ in range(self.start_step, self.train_steps):
                if self.state.global_step >= self.train_steps:
                    break
                try:
                    self.train_step(self.data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {config.dataloader.drop_last}")
                    break

            self.on_epoch_end()

            self.start_step = 0

            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

            if config.dataloader.use_background_prefetcher:
                self.data_iterator.stop()

            if self.state.global_step >= self.train_steps:
                break

        self.on_train_end()

        if "data_iterator" in locals() and config.dataloader.use_background_prefetcher:
            self.data_iterator.stop()

        synchronize()

        self.destroy_distributed()
