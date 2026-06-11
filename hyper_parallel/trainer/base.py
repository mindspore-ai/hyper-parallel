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
"""BaseTrainer — composable training skeleton with 13 overridable ``_build_*`` steps.

Design references:
-  BaseTrainer (``): composition pattern, 13-step init
- hyper ``fsdp_demo.py``: FSDP wrapping via ``model.layers`` iteration
- hyper st tests: TP → CP → AC → FSDP composition order

Subclasses (LLMTrainer, DiTTrainer, VLMTrainer) use the composition pattern:
instantiate ``BaseTrainer`` and call ``_build_*`` methods selectively, overriding
or skipping steps as needed.
"""
import json
import logging
import math
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
from torch.utils.data import Dataset, DistributedSampler

from hyper_parallel import (
    get_platform,
    init_empty_weights,
    init_process_group,
    destroy_process_group,
    hsdp_sync_stream,
    SkipDTensorDispatch,
    HSDPModule,
)
from hyper_parallel.core.distributed_checkpoint import load as dcp_load
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.hsdp_utils import GroupInfo
from hyper_parallel.core.utils import clip_grad_norm_
from hyper_parallel.models.spec.registry import get_spec
from hyper_parallel.trainer.parallel_dims import ParallelDims
from hyper_parallel.trainer.utils.loss import count_loss_token, mean_global_loss
from hyper_parallel.trainer.callbacks.base import (
    LoggingCallback,
    CheckpointCallback,
    SafetensorsExportCallback,
    EvalCallback,
    ProfilerCallback,
    WandbCallback,
    ProgressCallback,
    MoEMonitorCallback,
    GradientHealthCallback,
    GCCallback,
    TensorBoardCallback,
    MemoryMonitorCallback,
)

if TYPE_CHECKING:
    # Type-only imports — never executed at runtime, so the platform-agnostic
    # rule ("no torch/mindspore in trainer code") is preserved. Same pattern
    # as
    from torch import nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader
    from hyper_parallel.core.dtensor.device_mesh import DeviceMesh

platform = get_platform()
logger = logging.getLogger(__name__)


class TrainerState:
    """Mutable training state shared across callbacks.

    Attributes:
        global_step: Current training step (update count).
        epoch: Current epoch index.
        max_steps: Total number of training steps.
    """

    def __init__(self, max_steps: int = 0):
        self.global_step: int = 0
        self.epoch: int = 0
        self.max_steps: int = max_steps
        self.log_history: list = []


class BaseTrainer:
    """Composable training skeleton.

    Provides 13 ``_build_*`` methods that subclasses can call, override, or skip.
    The default ``_build_parallelized_model`` applies TP → CP → AC → FSDP by
    iterating ``model.layers`` — matching hyper's own ``fsdp_demo.py`` style.

    Args:
        args: Training configuration (typically parsed from YAML).
    """

    # PEP 526 annotations — populated by ``_build_*``; ``None`` until built.
    model: Optional["nn.Module"] = None
    optimizer: Optional["Optimizer"] = None
    lr_scheduler: Optional["LRScheduler"] = None
    train_dataloader: Optional["DataLoader"] = None
    mesh: Optional["DeviceMesh"] = None

    def __init__(self, args):
        # Only early-bound fields live here; the rest is built via
        # ``_build_*`` methods invoked by the subclass.
        self.args = args
        self.spec = get_spec(args.model.name)
        self.state = TrainerState(max_steps=getattr(args.train, 'max_steps', 0))

    # ------------------------------------------------------------------
    # 13 overridable _build_* methods
    # ------------------------------------------------------------------

    def _setup(self):
        """Step 1: Initialize distributed environment, device mesh, and seed.

        Calls hyper's own ``init_process_group`` and ``init_device_mesh``.
        Mesh shape is derived from ``args.parallel`` (dp, tp, cp, pp, ep).
        """
        backend = getattr(self.args.train, 'comm_backend', None)
        init_process_group(backend=backend)

        local_rank = getattr(self.args.train, 'local_rank', 0)
        device_type = platform.device_type()  # "npu" or "cuda"
        # Use platform.device(idx) — backend-agnostic.
        self.device = platform.device(local_rank)
        device_handle = platform.get_device_handle(device_type)
        device_handle.set_device(local_rank)

        # Build & validate parallel dims in one place (fail-fast).

        self.parallel_dims = ParallelDims.from_config(
            self.args.train.accelerator, world_size=platform.get_world_size(),
        )
        logger.info_rank0("ParallelDims: %s", self.parallel_dims.summary())
        self.mesh = self.parallel_dims.build_mesh(platform.device_type())

        # Build DP group_info for trainer-level all_reduce (loss/token sync).
        # Uses hyper's GroupInfo + mesh.get_group (platform-agnostic).

        dp_group = self._get_combined_dp_group()
        dp_size = self.parallel_dims.dp_size
        self._dp_group_info = GroupInfo(
            group_name="trainer_dp", group=dp_group, rank_size=dp_size,
        )

        debug_cfg = getattr(self.args.train, 'debug', None)
        seed = getattr(self.args.train, 'seed', 42)
        platform.manual_seed(seed)
        # Seed device-side RNG explicitly: ``platform.manual_seed`` only
        # covers CPU.
        try:
            handle = platform.get_device_handle(device_type)
            if hasattr(handle, "manual_seed_all"):
                handle.manual_seed_all(seed)
            elif hasattr(handle, "manual_seed"):
                handle.manual_seed(seed)
        except Exception as exc:  # pylint: disable=W0718
            logger.warning("Device-side seed init skipped: %s", exc)

        if debug_cfg is not None and getattr(debug_cfg, 'deterministic', False):
            warn_only = getattr(debug_cfg, 'deterministic_warn_only', False)
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info_rank0("Deterministic algorithms enabled (warn_only=%s)", warn_only)

        logger.info_rank0(
            "Setup complete: rank=%d, world_size=%d, mesh=%s",
            platform.get_rank(), platform.get_world_size(),
            self.mesh.mesh_dim_names,
        )
        logger.info_rank0(
            "Config: data.type=%s, model.name=%s, model.num_hidden_layers=%s, "
            "init_device=%s, max_steps=%d, global_bs=%d",
            getattr(self.args.data, 'type', '?'),
            getattr(self.args.model, 'name', '?'),
            getattr(self.args.model, 'num_hidden_layers', '?'),
            getattr(self.args.train, 'init_device', '?'),
            self.state.max_steps,
            getattr(self.args.train, 'global_batch_size', '?'),
        )

    def _build_model(self):
        """Step 2: Construct model via ``spec.build_model_fn``.

        The model is a plain ``nn.Module`` at this point — not yet parallelized.
        When ``args.runtime.init_device == "meta"``, the model is constructed on
        the meta device (no memory allocated) and real weights are loaded after
        FSDP sharding via ``_load_weights_after_parallel``.
        """
        init_device = getattr(self.args.train, 'init_device', 'meta')
        # Meta-device init: each rank materialises only its own shard
        # post-FSDP — pre-trained weights via DCP, otherwise random init.
        if init_device == "meta":

            with init_empty_weights():
                self.model = self.spec.build_model_fn(self.args)
            logger.info_rank0(
                "Model built on meta device (no memory allocated): %s",
                type(self.model).__name__,
            )
        else:
            self.model = self.spec.build_model_fn(self.args)
            logger.info_rank0("Model built on %s: %s", init_device, type(self.model).__name__)

        # Cross-check parallel degrees against the actual model hyperparams
        # (heads%tp, kv_heads%tp, num_experts%ep, seq_len%(cp*tp)).
        # Fails fast here instead of crashing inside parallelize_module.
        seq_len = getattr(self.args.data, 'max_seq_len', None)
        self.parallel_dims.validate_against_model(self.model, seq_len=seq_len)

    def _freeze_model(self):
        """Step 3: Freeze specified modules (optional)."""
        freeze_modules = getattr(self.args.model, 'freeze_modules', None)
        if not freeze_modules:
            return
        for name, param in self.model.named_parameters():
            if any(pattern in name for pattern in freeze_modules):
                param.requires_grad_(False)

    def _build_model_assets(self):
        """Step 4: Build tokenizer, processor, chat_template.

        Default: no-op. LLMTrainer overrides to build tokenizer + chat_template.
        VLMTrainer overrides to build processor.
        """
        self.tokenizer = None
        self.processor = None

    def _build_data_transform(self):
        """Step 5: Build data preprocessing transform.

        Default: identity transform. LLMTrainer overrides for tokenization.
        """
        self.data_transform = None

    def _build_dataset(self):
        """Step 6: Build training dataset.

        Supports:
        - ``data.type = "dummy"``: random tokens for validation
        - ``data.type = "hf_datasets"``: HuggingFace datasets
        - ``data.type = "megatron_indexed"``: Megatron .bin/.idx format

        Subclass can override for custom dataset logic.
        """

        data_type = getattr(self.args.data, 'type', 'dummy')
        seq_len = getattr(self.args.data, 'max_seq_len', 2048)

        if data_type == "dummy":
            vocab_size = getattr(self.model, 'config', None)
            vocab_size = vocab_size.vocab_size if vocab_size else 32000
            total_samples = self.state.max_steps * getattr(self.args.train, 'global_batch_size', 8)

            class DummyDataset(Dataset):
                """Deterministic random token dataset for FSDP validation.

                Each sample's content is fixed by its index (seeded by
                ``base_seed + idx``), so the same index always produces the
                same tokens regardless of access order or DP configuration.
                """
                def __init__(self, num_samples, seq_length, vocab, base_seed=42):
                    self.num_samples = num_samples
                    self.seq_length = seq_length
                    self.vocab = vocab
                    self.base_seed = base_seed

                def __len__(self):
                    return self.num_samples

                def __getitem__(self, idx):
                    g = torch.Generator().manual_seed(self.base_seed + idx)
                    input_ids = torch.randint(
                        0, self.vocab, (self.seq_length,), generator=g,
                    )
                    return {"input_ids": input_ids, "labels": input_ids.clone()}

            self.train_dataset = DummyDataset(total_samples, seq_len, vocab_size)
            logger.info_rank0("Dummy dataset created: %d samples, seq_len=%d", total_samples, seq_len)

        elif data_type == "hf_datasets":
            from datasets import load_dataset  # pylint: disable=C0415  # optional dep
            train_path = self.args.data.train_path
            streaming = getattr(self.args.data, 'streaming', False)
            if streaming:
                # ``DistributedSampler`` requires ``__len__``, which
                # ``IterableDataset`` lacks; reject loudly until a
                # sampler-less streaming path is wired.
                raise NotImplementedError(
                    "data.streaming=True not yet wired. The current "
                    "_build_dataloader uses DistributedSampler which requires "
                    "len(dataset); IterableDataset has no __len__. "
                    "Use data.streaming=False (map-style) for now, or "
                    "subclass _build_dataset + _build_dataloader to emit an "
                    "IterableDataset that self-shards via dp_rank/dp_size."
                )
            ds = load_dataset(train_path, split="train", streaming=False)
            if self.data_transform:
                ds = ds.map(
                    self.data_transform, remove_columns=ds.column_names,
                )
            self.train_dataset = ds
            logger.info_rank0("HF dataset loaded: %s (map-style)", train_path)

        else:
            raise ValueError(f"Unknown data type: {data_type}. Supported: dummy, hf_datasets")

    def _build_collate_fn(self):
        """Step 7: Build data collator.

        Default: pads input_ids and labels to max length in the batch.
        """

        def _default_collate(batch):
            """Simple padding collator."""
            max_len = max(item["input_ids"].size(0) for item in batch)
            input_ids_list = []
            labels_list = []
            for item in batch:
                pad_len = max_len - item["input_ids"].size(0)
                input_ids_list.append(
                    torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=0)
                )
                labels_list.append(
                    torch.nn.functional.pad(item["labels"], (0, pad_len), value=-100)
                )
            return {
                "input_ids": torch.stack(input_ids_list),
                "labels": torch.stack(labels_list),
            }

        self.collate_fn = _default_collate

    def _build_dataloader(self):
        """Step 8: Build distributed stateful dataloader.

        Uses ``torchdata.stateful_dataloader.StatefulDataLoader`` so that
        iterator position is checkpointable — enabling exact resume after
        restart (matching ).

        Each ``next()`` call yields a list of micro-batches (for gradient
        accumulation).
        """
        from torchdata.stateful_dataloader import StatefulDataLoader  # pylint: disable=C0415  # optional dep

        micro_bs = getattr(self.args.train, 'micro_batch_size', 1)

        # Sampler uses DP rank/size — TP/CP/PP/EP peers share data.
        dp_size = self.parallel_dims.dp_size
        non_dp = self.parallel_dims.non_dp_size
        global_rank = platform.get_rank()
        dp_rank = global_rank // non_dp if non_dp > 1 else global_rank

        shuffle = getattr(self.args.data, "shuffle", True)
        sampler_seed = getattr(self.args.train, 'seed', 0)

        self.sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=shuffle,
            seed=sampler_seed,
            drop_last=True,
        )

        # StatefulDataLoader supports state_dict() / load_state_dict()
        # for checkpoint resume (torchdata API, used by  + ).
        num_workers = getattr(self.args.data, 'num_workers', 0)
        prefetch_factor = getattr(self.args.data, 'prefetch_factor', None)
        pin_memory = getattr(self.args.data, 'pin_memory', True)
        loader_kwargs = {
            "batch_size": micro_bs,
            "sampler": self.sampler,
            "collate_fn": self.collate_fn,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": True,
        }
        # prefetch_factor is only accepted when num_workers > 0
        if num_workers > 0 and prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        self.train_dataloader = StatefulDataLoader(
            self.train_dataset, **loader_kwargs,
        )

        # Use dp_size (not world_size) — TP/CP/PP ranks share data, not split it.
        self._grad_accum = max(
            getattr(self.args.train, 'global_batch_size', micro_bs) // (micro_bs * dp_size),
            1,
        )

        logger.info_rank0(
            "Dataloader built: micro_bs=%d, grad_accum=%d, dataset_size=%d",
            micro_bs, self._grad_accum, len(self.train_dataset),
        )

    def _build_parallelized_model(self):
        """Step 9: Apply parallel strategies to the model.

        Each model owns its full parallelize pipeline in
        ``models/<name>/parallelize.py`` (convention) and
        registers it via ``ModelSpec.parallelize_fn``. There is no shared
        "default" template — model-specific TP/EP/CP/AC/FSDP/Prefetch
        composition lives next to the model that needs it.
        """
        if self.spec.parallelize_fn is None:
            raise ValueError(
                f"Model '{self.spec.name}' has no ``parallelize_fn`` registered "
                f"on its ModelSpec. Each model must own its parallelize "
                f"pipeline in models/<name>/parallelize.py."
            )
        self.model = self.spec.parallelize_fn(self.model, self.mesh, self.args)
        self._post_parallelize()

    def _post_parallelize(self):
        """Common steps after parallelization (materialize weights + train mode).

        Order when ``init_device == "meta"`` and ``weights_path`` is set:

        1. Run ``_materialize_and_init_shards`` first — this calls
           ``model.to_empty(device=...)`` + kaiming / zero init for every
           parameter. That is the **baseline** state so no param stays on
           meta (which would trip ``HSDPState._validate_no_meta_params``).
        2. Then ``_load_weights`` copies the upstream checkpoint on top.
           Every key that matches overwrites the random init; anything
           missing in the checkpoint stays with its kaiming / zero init.

        This pattern handles partial checkpoints cleanly — e.g. hyper's
        current Qwen3-VL-MoE model lacks ``q_norm`` / ``k_norm`` /
        ``pos_embed`` / ``deepstack_merger_list``; those stay random
        while all other weights come from the pretrained checkpoint.
        """
        init_device = getattr(self.args.train, 'init_device', 'meta')
        weights_path = getattr(self.args.model, 'weights_path', None)
        if init_device == "meta":
            # Always materialize first (random init baseline) so no param
            # stays on meta — then overlay the checkpoint.
            self._materialize_and_init_shards()
            if weights_path:
                self._load_weights(weights_path)
        elif weights_path:
            self._load_weights(weights_path)
        # Mixed-precision storage policy: trainable params keep an fp32
        # master; frozen params keep their loaded low-precision dtype so the
        # forward stays uniform-bf16 within their FSDP unit.
        self._maybe_downcast_frozen_params()
        self._maybe_upcast_trainable_params()
        self.model.train()

    def _maybe_downcast_frozen_params(self) -> None:
        """Maybe downcast frozen params (internal)."""
        freeze_modules = getattr(self.args.model, 'freeze_modules', None)
        if not freeze_modules:
            return
        mp_cfg = getattr(self.args.train, 'mixed_precision', None)
        if mp_cfg is None or not getattr(mp_cfg, 'enabled', False):
            return

        target_dtype = {
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
            'float16': torch.float16,
            'fp16': torch.float16,
        }.get(getattr(mp_cfg, 'param_dtype', 'bfloat16'))
        if target_dtype is None:
            return
        n_cast = 0
        for name, param in self.model.named_parameters():
            if not any(pat in name for pat in freeze_modules):
                continue
            if param.requires_grad:
                continue
            local = param.data
            if hasattr(local, 'to_local'):
                local = local.to_local()
            if local.dtype == target_dtype:
                continue
            new_local = local.to(target_dtype)
            # DTensor: rebuild the global view via from_local with same placements.
            if hasattr(param.data, 'to_local'):

                if isinstance(param.data, DTensor):
                    param.data = DTensor.from_local(
                        new_local,
                        device_mesh=param.data.device_mesh,
                        placements=param.data.placements,
                    )
                else:
                    param.data = new_local
            else:
                param.data = new_local
            n_cast += 1
        logger.info_rank0(
            "Post-load: cast %d frozen params to %s",
            n_cast, target_dtype,
        )

    def _maybe_upcast_trainable_params(self) -> None:
        """Upcast trainable params to ``float32``.

        Implements the standard mixed-precision pattern (fp32 master weight
        + low-precision forward): trainable params loaded from a bf16
        checkpoint would otherwise stay in bf16, and Adam moment estimates
        accumulating at bf16's 7-bit mantissa diverge noticeably after only
        a handful of optimizer steps.

        Runs AFTER ``_maybe_downcast_frozen_params`` so frozen params keep
        their bf16 storage and only trainable params get the fp32 master.
        """
        mp_cfg = getattr(self.args.train, 'mixed_precision', None)
        if mp_cfg is None or not getattr(mp_cfg, 'enabled', False):
            return

        n_cast = 0
        for _, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            local = param.data
            if hasattr(local, 'to_local'):
                local = local.to_local()
            if local.dtype == torch.float32:
                continue
            new_local = local.to(torch.float32)
            if hasattr(param.data, 'to_local'):

                if isinstance(param.data, DTensor):
                    param.data = DTensor.from_local(
                        new_local,
                        device_mesh=param.data.device_mesh,
                        placements=param.data.placements,
                    )
                else:
                    param.data = new_local
            else:
                param.data = new_local
            n_cast += 1
        logger.info_rank0(
            "Post-load: upcast %d trainable params to float32", n_cast,
        )

    def _build_optimizer(self):
        """Step 10: Build optimizer. Must be called AFTER ``_build_parallelized_model``.

        After FSDP, parameters are DTensor shards — optimizer operates on local shards.
        Optimizer must be created after ``fully_shard``.
        """
        lr = getattr(self.args.train.optimizer, 'lr', 1e-4)
        weight_decay = getattr(self.args.train.optimizer, 'weight_decay', 0.01)

        # bias / LayerNorm / RMSNorm go to no-decay; grouping matters even
        # at wd=0 — foreach Adam reduction order differs per group on NPU.
        decay_keywords = ("bias", "layernorm", "norm", "rmsnorm")

        def _is_no_decay(name: str) -> bool:
            lname = name.lower()
            return any(kw in lname for kw in decay_keywords)

        decay_params = []
        no_decay_params = []
        seen_ids = set()
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # Dedup tied params (same nn.Parameter shared across modules).
            if id(p) in seen_ids:
                continue
            seen_ids.add(id(p))
            if _is_no_decay(n):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        adam_eps = getattr(self.args.train.optimizer, 'eps', 1e-8)
        adam_betas = getattr(self.args.train.optimizer, 'betas', (0.9, 0.999))
        adam_foreach = getattr(self.args.train.optimizer, 'foreach', None)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=adam_betas,
            eps=adam_eps,
            foreach=adam_foreach,
        )
        logger.info_rank0(
            "Optimizer: AdamW lr=%.2e wd=%.3g  decay_params=%d  no_decay_params=%d",
            lr, weight_decay, len(decay_params), len(no_decay_params),
        )

    def _build_lr_scheduler(self):
        """Step 11: Build learning rate scheduler.

        Supports cosine decay with warmup. Falls back to constant LR if
        warmup_ratio is 0 and decay_style is 'constant'.
        """

        total_steps = self.state.max_steps
        warmup_ratio = getattr(self.args.train.optimizer, 'lr_warmup_ratio', 0.0)
        # ``ceil`` matches the standard warmup convention so a fractional
        # ``warmup_ratio * max_steps`` rounds up to the next full step.
        warmup_steps = math.ceil(total_steps * warmup_ratio)
        decay_style = getattr(self.args.train.optimizer, 'lr_decay_style', 'cosine')
        lr_min = getattr(self.args.train.optimizer, 'lr_min', 0.0)
        lr_max = getattr(self.args.train.optimizer, 'lr', 1e-4)

        def _lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if decay_style == 'constant':
                return 1.0
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_ratio = lr_min / lr_max if lr_max > 0 else 0.0
            return min_ratio + (1.0 - min_ratio) * cosine_decay

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)
        logger.info_rank0(
            "LR scheduler: %s, warmup_steps=%d/%d, lr=%.2e→%.2e",
            decay_style, warmup_steps, total_steps, lr_max, lr_min,
        )

    def _build_training_context(self):
        """Step 12: Build forward/backward context managers.

        Delegates AMP context to ``_make_amp_context`` — the single place that
        knows about backend-specific autocast. MindSpore subclass should
        override ``_make_amp_context`` only.
        """


        mp_cfg = getattr(self.args.train, 'mixed_precision', None)
        # FSDP2 mp_policy already runs forward/backward in low precision;
        # stacking ``torch.autocast`` on top would re-promote LayerNorm /
        # softmax / log_softmax to fp32 and break the uniform-bf16 forward.
        self.model_fwd_context = nullcontext()
        self.grad_scaler = None
        if mp_cfg and getattr(mp_cfg, 'enabled', False):
            param_dtype_str = getattr(mp_cfg, 'param_dtype', 'bfloat16')
            logger.info_rank0(
                "Mixed precision via FSDP2 mp_policy: dtype=%s on %s "
                "(autocast disabled — pure low-precision forward under mp_policy)",
                param_dtype_str, platform.device_type(),
            )

        self.model_bwd_context = nullcontext()

    def _make_amp_context(self, param_dtype_str: str):
        """Build the AMP forward context. Backend-specific.

        This is the SOLE method in the trainer that touches the backend AMP
        API directly. Override this single method to add MindSpore support
        (``ms.amp.auto_mixed_precision``). Default uses ``torch.autocast``.
        """
        dtype = getattr(torch, param_dtype_str, torch.bfloat16)
        return torch.autocast(platform.device_type(), dtype=dtype)

    def _init_callbacks(self):
        """Step 13: Initialize callbacks (explicit mode).

        Each callback is a named field — engineer sees all callbacks and their
        order in ``on_step_end`` at a glance. Add/remove/reorder = change one line.
        """
        self.logging_callback = LoggingCallback(self)
        self.checkpoint_callback = CheckpointCallback(self)
        self.hf_export_callback = SafetensorsExportCallback(self)
        self.eval_callback = EvalCallback(self)
        self.profiler_callback = ProfilerCallback(self)
        self.wandb_callback = WandbCallback(self)
        self.tensorboard_callback = TensorBoardCallback(self)
        self.progress_callback = ProgressCallback(self)
        self.moe_monitor_callback = MoEMonitorCallback(self)
        # Health + operability (no-ops unless enabled in cfg.train.debug / .memory_monitor).
        self.gradient_health_callback = GradientHealthCallback(self)
        self.memory_monitor_callback = MemoryMonitorCallback(self)
        self.gc_callback = GCCallback(self)
        # ``user_callbacks`` lets external code append extra Callback instances
        # (e.g. domain-specific monitors) without editing this method. They get
        # the same lifecycle dispatch as built-ins.
        self.user_callbacks: list = []
        logger.info_rank0(
            "Callbacks initialized: logging, checkpoint, hf_export, eval, "
            "profiler, wandb, tensorboard, progress, moe_monitor, "
            "gradient_health, memory_monitor, gc"
        )

    # ------------------------------------------------------------------
    # Public API: external callback registration
    # ------------------------------------------------------------------

    def add_callback(self, callback) -> None:
        """Register an extra ``Callback`` to receive every lifecycle event.

        Use this to plug domain-specific monitors (custom metric sinks,
        in-house experiment trackers, RL reward loggers) without editing
        the trainer. Built-in callbacks always run first; user callbacks
        run in registration order so a later user callback can read state
        the earlier ones updated.
        """
        self.user_callbacks.append(callback)
        logger.info_rank0(
            "User callback registered: %s", type(callback).__name__,
        )

    # ------------------------------------------------------------------
    # Callback dispatch (explicit mode)
    # ------------------------------------------------------------------

    def _builtin_callbacks(self) -> list:
        """Return built-in callbacks in fixed dispatch order.

        Centralised so every dispatcher iterates the same list — adding a
        callback only needs an entry here plus a named field in
        ``_init_callbacks`` (no per-event copy/paste).
        """
        return [
            self.logging_callback,
            self.eval_callback,
            self.profiler_callback,
            self.wandb_callback,
            self.tensorboard_callback,
            self.progress_callback,
            self.checkpoint_callback,
            self.hf_export_callback,
            self.moe_monitor_callback,
            self.gradient_health_callback,
            self.memory_monitor_callback,
            self.gc_callback,
        ]

    def _all_callbacks(self) -> list:
        """Built-in callbacks followed by user-registered ones."""
        return self._builtin_callbacks() + list(self.user_callbacks)

    def on_init_end(self):
        """Dispatch one-shot ``on_init_end`` after every ``_build_*`` ran.

        Fired by the subclass at the end of its own ``__init__`` (see
        ``LLMTrainer.__init__``); ``BaseTrainer.train()`` does NOT call it
        because BaseTrainer instances are sometimes wrapped (composition
        pattern) and the wrapper owns the init lifecycle.
        """
        for cb in self._all_callbacks():
            cb.on_init_end(self.state)

    def on_train_begin(self):
        """Dispatch on_train_begin to all callbacks."""
        # Memory monitor first so it captures the truly-initial peak.
        self.memory_monitor_callback.on_train_begin(self.state)
        self.moe_monitor_callback.on_train_begin(self.state)
        self.profiler_callback.on_train_begin(self.state)
        self.wandb_callback.on_train_begin(self.state)
        self.tensorboard_callback.on_train_begin(self.state)
        self.progress_callback.on_train_begin(self.state)
        # Checkpoint runs LAST so resume sees an already-armed TB writer
        # (it'll record the load event via dispatch_load_event).
        self.checkpoint_callback.on_train_begin(self.state)
        for cb in self.user_callbacks:
            cb.on_train_begin(self.state)

    def on_train_end(self):
        """Dispatch on_train_end to all callbacks."""
        self.checkpoint_callback.on_train_end(self.state)
        self.hf_export_callback.on_train_end(self.state)
        self.progress_callback.on_train_end(self.state)
        self.tensorboard_callback.on_train_end(self.state)
        self.wandb_callback.on_train_end(self.state)
        self.profiler_callback.on_train_end(self.state)
        for cb in self.user_callbacks:
            cb.on_train_end(self.state)

    def on_step_begin(self):
        """Dispatch on_step_begin to all callbacks."""
        self.logging_callback.on_step_begin(self.state)
        for cb in self.user_callbacks:
            cb.on_step_begin(self.state)

    def on_step_end(self, loss=None, grad_norm=None):
        """Dispatch on_step_end to all callbacks (built-ins + user)."""
        for cb in self._all_callbacks():
            cb.on_step_end(self.state, loss=loss, grad_norm=grad_norm)

    def on_substep_end(self):
        """Dispatch on_substep_end (after each micro-batch forward/backward)."""
        self.moe_monitor_callback.on_substep_end(self.state)
        for cb in self.user_callbacks:
            cb.on_substep_end(self.state)

    def on_pre_optimizer_step(self, grad_norm=None):
        """Dispatch on_pre_optimizer_step (after grad clip, before optimizer.step)."""
        # Health check runs FIRST so a NaN aborts before the logger misleads.
        self.gradient_health_callback.on_pre_optimizer_step(
            self.state, grad_norm=grad_norm,
        )
        self.logging_callback.on_pre_optimizer_step(self.state, grad_norm=grad_norm)
        self.wandb_callback.on_pre_optimizer_step(self.state, grad_norm=grad_norm)
        self.tensorboard_callback.on_pre_optimizer_step(self.state, grad_norm=grad_norm)
        for cb in self.user_callbacks:
            cb.on_pre_optimizer_step(self.state, grad_norm=grad_norm)

    def on_epoch_begin(self):
        """Dispatch on_epoch_begin."""
        for cb in self._all_callbacks():
            cb.on_epoch_begin(self.state)

    def on_epoch_end(self):
        """Dispatch on_epoch_end."""
        for cb in self._all_callbacks():
            cb.on_epoch_end(self.state)

    # ------------------------------------------------------------------
    # Event fan-out (LoggingCallback / CheckpointCallback emit these)
    # ------------------------------------------------------------------

    def dispatch_log_event(self, metrics: dict) -> None:
        """Forward a metrics record to every callback's ``on_log``.

        ``LoggingCallback`` calls this so TensorBoard / W&B / external sinks
        log the SAME numbers — single source of truth, no duplicate work.
        """
        for cb in self._all_callbacks():
            cb.on_log(self.state, metrics=metrics)

    def dispatch_save_event(self, checkpoint_dir: str) -> None:
        """Forward a ckpt-save event to every callback's ``on_save``."""
        for cb in self._all_callbacks():
            cb.on_save(self.state, checkpoint_dir=checkpoint_dir)

    def dispatch_load_event(self, checkpoint_dir: str) -> None:
        """Forward a ckpt-load event to every callback's ``on_load``."""
        for cb in self._all_callbacks():
            cb.on_load(self.state, checkpoint_dir=checkpoint_dir)

    def dispatch_evaluate_event(self, metrics: dict = None) -> None:
        """Forward an eval-pass-complete event to every callback's ``on_evaluate``."""
        for cb in self._all_callbacks():
            cb.on_evaluate(self.state, metrics=metrics)

    # ------------------------------------------------------------------
    # Training core
    # ------------------------------------------------------------------

    def forward_backward_step(
        self,
        micro_batch: Dict[str, Any],
        micro_batch_tokens: int,
        global_tokens: int,
        num_micro: int = 1,
    ):
        """Run forward + backward for one micro-batch.

        Uses  global token normalisation: each micro-batch's
        loss is scaled by ``micro_tokens / global_tokens`` so that every token
        across all ranks and all micro-batches contributes equally to the
        gradient, regardless of DP size or grad_accum.

        Args:
            micro_batch: Dict of input tensors.
            micro_batch_tokens: Non-padding token count for this micro-batch.
            global_tokens: Total non-padding tokens across **all** ranks and
                           **all** micro-batches (computed via all-reduce).

        Returns:
            Tuple of (raw_loss_scalar, micro_batch_tokens) for logging.
        """
        # Move tensors to device. Multimodal batches may contain nested
        # lists/tuples/dicts, so keep this recursive instead of assuming a
        # flat LM-only batch.
        def _to_device(value):
            if hasattr(value, "to"):
                return value.to(self.device, non_blocking=True)
            if isinstance(value, dict):
                return {k: _to_device(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_to_device(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_to_device(v) for v in value)
            return value

        micro_batch = {k: _to_device(v) for k, v in micro_batch.items()}

        # Forward (with training context for activation offload)
        with self.model_fwd_context:
            outputs = self.model(**micro_batch, use_cache=False)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        # TP scenario: loss may be Partial DTensor — reduce before backward
        if hasattr(loss, 'is_partial') and loss.is_partial():
            loss = loss.reduce_partial()

        # Keep raw loss value for logging before scaling
        raw_loss = loss.detach()

        # token_weighted: ``CE_mean * (micro_tokens / global_tokens) * dp_size``
        # — DP-size and grad_accum invariant after FSDP's ``AVG`` reduction.
        dp_size = self.parallel_dims.dp_size
        agg = getattr(self.args.train.optimizer, 'loss_aggregation', 'token_weighted')
        if agg == 'rank_average':
            # Per-rank CE mean; FSDP2 ``ReduceOp.AVG`` handles cross-rank
            # averaging. Dividing by ``num_micro`` averages grad-accum so
            # 1-card grad_accum=N tracks the FSDP path's effective grad.
            scaled_loss = loss / num_micro if num_micro > 1 else loss
        else:
            scaled_loss = mean_global_loss(
                loss, micro_batch_tokens, global_tokens, dp_size,
            )

        # Backward (with training context)
        with self.model_bwd_context:
            scaled_loss.backward()

        return raw_loss, micro_batch_tokens

    def train_step(self, data_iterator):
        """Execute one training step with gradient accumulation.

        Precision-aligned across different DP configurations by:
        1. All-reducing global token count before loss scaling ()
        2. Syncing gradients only on the last micro-batch ()
        3. All-reducing loss weighted by token count for reporting

        Args:
            data_iterator: Iterator yielding lists of micro-batch dicts.
        """
        # Pull data first; ``StopIteration`` propagates without bumping the
        # step counter so checkpoint dirs / log indices match the steps that
        # actually trained.
        micro_batches = next(data_iterator)
        self.state.global_step += 1
        num_micro = len(micro_batches)

        # ---- Phase 1: count global tokens ( style) ----
        # All-reduce BEFORE forward so every rank uses the same denominator.

        token_counts = [count_loss_token(mb) for mb in micro_batches]
        local_tokens = sum(token_counts)
        if local_tokens == 0:
            local_tokens = 1
        global_tokens = local_tokens
        if platform.get_world_size() > 1:
            gt = platform.full((1,), local_tokens).to(self.device)
            platform.all_reduce(gt, self._dp_group_info)
            global_tokens = max(int(gt.item()), 1)
        # Expose for callbacks (e.g. LoggingCallback throughput).
        self._last_global_tokens = global_tokens

        # ---- Phase 2: forward / backward with per-micro-batch sync control ----
        total_loss_sum = 0.0   # weighted: loss * micro_tokens
        total_loss_arith_sum = 0.0   # arithmetic sum across micro-batches
        total_tokens_local = 0
        for i, mb in enumerate(micro_batches):
            is_last = i == num_micro - 1

            # Gradient sync: only on the last micro-batch ( pattern).
            # Before last: accumulate gradients locally, no communication.
            if isinstance(self.model, HSDPModule):
                self.model.set_requires_gradient_sync(is_last)
                self.model.set_is_last_backward(is_last)

            # FSDP reshard optimization for gradient accumulation
            self._maybe_toggle_reshard(i, num_micro)

            raw_loss, mb_tokens = self.forward_backward_step(
                mb, token_counts[i], global_tokens, num_micro=num_micro,
            )
            total_loss_sum += raw_loss.item() * mb_tokens
            total_loss_arith_sum += raw_loss.item()
            total_tokens_local += mb_tokens
            self.on_substep_end()

        # Wait for async gradient reduce
        #
        hsdp_sync_stream()

        # Gradient clipping — DTensor-aware, returns plain Tensor
        max_grad_norm = getattr(self.args.train.optimizer, 'max_grad_norm', 1.0)
        clip_fn = self.spec.clip_grad_fn or clip_grad_norm_
        grad_norm = clip_fn(self.model.parameters(), max_grad_norm)
        self.on_pre_optimizer_step(grad_norm=grad_norm.item())

        # Optimizer step — must be inside SkipDTensorDispatch
        with SkipDTensorDispatch():
            self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.optimizer.zero_grad()

        # ---- Phase 3: all-reduce loss for reporting ----
        agg = getattr(self.args.train.optimizer, 'loss_aggregation', 'token_weighted')
        if agg == 'token_weighted':
            # Global token-weighted: sum local ce_sum across DP, divide by
            # global token count. Equivalent to mean-loss-per-token under
            # post-DP gradient mean.
            if platform.get_world_size() > 1:
                ls = platform.full((1,), total_loss_sum).to(self.device)
                platform.all_reduce(ls, self._dp_group_info)
                avg_loss = ls.item() / max(global_tokens, 1)
            else:
                avg_loss = total_loss_sum / max(total_tokens_local, 1)
        else:
            # rank_average: each rank averages its micro-batches, then
            # ``all_reduce`` averages across DP ranks. Divisor is the DP
            # group size (NOT global world_size); under TP/EP/PP/CP the
            # all-reduce only sums DP ranks, so dividing by world_size
            # would systematically under-report the loss.
            local_mean = total_loss_arith_sum / max(num_micro, 1)
            dp_size = self._dp_group_info.rank_size
            if dp_size > 1:
                ls = platform.full((1,), local_mean).to(self.device)
                platform.all_reduce(ls, self._dp_group_info)
                avg_loss = ls.item() / dp_size
            else:
                avg_loss = local_mean

        return {"loss": avg_loss, "grad_norm": grad_norm.item()}

    def train(self):
        """Main training loop: epoch → step → micro-batch.

        Dispatches callbacks at each lifecycle point (explicit mode).
        on_train_begin is called first — CheckpointCallback uses it to restore
        state.global_step from a saved checkpoint, so the loop below will
        correctly skip already-completed steps.
        """
        logger.info_rank0(
            "Training starts: max_steps=%d, epochs=%d",
            self.state.max_steps,
            getattr(self.args.train, 'num_train_epochs', 1),
        )
        # on_train_begin runs checkpoint resume — state.global_step may be
        # updated to the resumed step before the loop starts.
        self.on_train_begin()
        num_epochs = getattr(self.args.train, 'num_train_epochs', 1)

        if self.state.global_step > 0:
            logger.info_rank0(
                "Resuming training from step %d", self.state.global_step,
            )

        for epoch in range(num_epochs):
            if self.state.global_step >= self.state.max_steps:
                break
            self.state.epoch = epoch
            if hasattr(self, 'sampler'):
                self.sampler.set_epoch(epoch)
            self.on_epoch_begin()

            # Build micro-batch iterator from the stateful dataloader.
            # StatefulDataLoader tracks iterator position internally,
            # so after resume it skips already-consumed batches.
            data_iterator = self._make_micro_batch_iterator()

            # Drive the loop on the live ``global_step`` so total training
            # never exceeds ``max_steps`` regardless of ``num_train_epochs``
            # or resume offset.
            while self.state.global_step < self.state.max_steps:
                self.on_step_begin()
                try:
                    metrics = self.train_step(data_iterator)
                except StopIteration:
                    logger.info_rank0("Epoch %d: dataloader exhausted", epoch)
                    break

                self.on_step_end(
                    loss=metrics["loss"],
                    grad_norm=metrics["grad_norm"],
                )

            self.on_epoch_end()

        self.on_train_end()
        destroy_process_group()
        logger.info_rank0("Training completed")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_micro_batch_iterator(self):
        """Yield lists of micro-batches from the stateful dataloader.

        Groups ``self._grad_accum`` consecutive batches into a list for
        gradient accumulation. The underlying ``StatefulDataLoader`` tracks
        iteration position, so checkpoint/resume skips consumed batches.
        """
        batch_buffer = []
        for batch in self.train_dataloader:
            batch_buffer.append(batch)
            if len(batch_buffer) >= self._grad_accum:
                yield batch_buffer
                batch_buffer = []
        if batch_buffer:
            yield batch_buffer

    def _get_layers(self) -> list:
        """Return the repeating layers for FSDP/AC wrapping.

        Default: ``model.layers`` (standard transformer convention).
        Override in subclass for models with different structure.
        """
        if hasattr(self.model, 'layers'):
            return list(self.model.layers)
        raise ValueError(
            f"Model {type(self.model).__name__} has no .layers attribute. "
            f"Either add self.layers to the model, or override _get_layers() "
            f"in the Trainer subclass."
        )

    def _get_combined_dp_group(self):
        """Return the combined data-parallel ProcessGroup for trainer all-reduce.

        Prefers the ``"loss"`` flatten alias registered by
        ``ParallelDims.build_mesh`` (folds CP into the DP group when CP is
        active so token-count denominators include CP-sharded contributions).
        Falls back to ``"dp"``, then to the legacy ``dp_shard`` /
        ``dp_replicate`` axes for callers that built a custom mesh.
        """
        for name in ("loss", "dp", "dp_shard", "dp_replicate"):
            try:
                return self.mesh.get_group(name)
            except (KeyError, ValueError):
                continue
        return self.mesh.get_group()

    def _build_fsdp_kwargs(self) -> dict:
        """Build kwargs for ``fully_shard`` calls (dense parameters).

        For expert parameters when EP > 1, use ``_build_expert_fsdp_kwargs``.
        """
        for name in ("dp_shard", "dp", "dp_replicate"):
            try:
                dp_mesh = self.mesh[name]
                break
            except (KeyError, TypeError):
                continue
        else:
            dp_mesh = self.mesh
        kwargs = {"mesh": dp_mesh}

        reshard = getattr(self.args.train.accelerator, 'reshard_after_forward', True)
        kwargs["reshard_after_forward"] = reshard

        return kwargs

    def _build_expert_fsdp_kwargs(self) -> dict:
        """Build kwargs for ``fully_shard`` calls on expert parameters.

        When EP > 1, expert parameters are sharded across the EP group
        with a separate mesh dimension. Falls back to dense FSDP kwargs
        if EP is not enabled.
        """
        if not self.parallel_dims.ep_enabled:
            return self._build_fsdp_kwargs()

        try:
            ep_mesh = self.mesh["ep"]
        except (KeyError, TypeError):
            logger.warning("EP=%d but no 'ep' dimension in mesh, falling back to dp mesh",
                           self.parallel_dims.ep)
            return self._build_fsdp_kwargs()

        kwargs = {"mesh": ep_mesh}
        reshard = getattr(self.args.train.accelerator, 'reshard_after_forward', True)
        kwargs["reshard_after_forward"] = reshard
        return kwargs

    def _materialize_and_init_shards(self) -> None:
        """Materialize meta-device parameters/buffers to real device in-place.

        After ``fully_shard`` on a meta-device model, each rank's parameters
        are meta DTensor shards **and FSDP2 holds internal views into those
        meta storages** (flat_param / unsharded buffer). Replacing the
        ``DTensor._local_tensor`` attribute leaves FSDP's internal views
        pointing at the old meta storage, so the first forward's all-gather
        still hits meta → ``c10d::_allgather_base_`` raises.

        PyTorch's ``nn.Module.to_empty(device=...)`` is the FSDP2-safe path:
        it walks every parameter/buffer (including DTensor shards) and
        **allocates real device storage in-place via ``torch.empty_like``**,
        preserving every existing view. After ``to_empty``, storage is
        uninitialised — we init on the local shard with kaiming_uniform for
        weights, zero for biases / 1-D / buffers.

        Reference: PyTorch FSDP2 meta-init docs;
         ``trainer.py`` post-``fully_shard`` init pattern.
        """
        device_type = platform.device_type()
        # Step 1: meta → real storage, in-place (FSDP-views preserved).
        self.model.to_empty(device=device_type)
        self._materialize_replicate_params(device_type)
        # Step 2: init the local shard of every param (and zero every buffer).
        param_count = self._init_local_shards()
        # Re-derive buffers wiped by ``to_empty`` (e.g. ``inv_freq``);
        # without this RoPE silently returns identity rotation.
        for module in self.model.modules():
            if hasattr(module, "reset_inv_freq"):
                module.reset_inv_freq()
        # Re-tie weights — ``to_empty`` gives every nn.Parameter fresh
        # storage so ``__init__``-time ties are broken. Must happen before
        # ``lazy_init`` re-wraps params as DTensor (non-leaf), which would
        # cause ``register_parameter`` to reject the assignment.
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()
        # ``to_empty`` strips DTensor; ``lazy_init`` re-wraps shards before
        # ``_load_weights`` / optimizer step see the params (the forward
        # pre-hook does the same later, but the loader needs DTensor first).
        reset_count = self._lazy_init_hsdp_modules()
        logger.info_rank0(
            "Meta → real on %s: to_empty + kaiming/zero init on %d params; "
            "FSDP lazy_init re-wrapped %d modules back to DTensor",
            device_type, param_count, reset_count,
        )

    def _iter_hsdp_states(self):
        """Yield the HSDP state attached to every HSDP-wrapped submodule."""
        for module in self.model.modules():
            if not isinstance(module, HSDPModule):
                continue
            scheduler = getattr(module, 'hsdp_scheduler', None)
            state = getattr(scheduler, 'hsdp_state', None) if scheduler else None
            if state is None:
                continue
            yield state

    def _materialize_replicate_params(self, device_type: str) -> None:
        """``to_empty`` skips ``replicate_params`` whose ``_local_tensor`` lives
        outside ``module._parameters`` — materialize them manually.
        """
        for state in self._iter_hsdp_states():
            for hsdp_param in getattr(state, 'replicate_params', []) or []:
                local = getattr(hsdp_param.sharded_param, "_local_tensor", None)
                if local is not None and local.is_meta:
                    new_local = torch.empty_like(local, device=device_type)
                    hsdp_param.sharded_param._local_tensor = new_local  # pylint: disable=W0212

    def _init_local_shards(self) -> int:
        """Init local shard of every param (kaiming for >=2D, zero else); zero buffers."""
        param_count = 0
        with torch.no_grad():
            for _, param in self.model.named_parameters():
                local = param._local_tensor if hasattr(param, '_local_tensor') else param  # pylint: disable=W0212
                if local.is_meta:
                    continue
                if local.dim() >= 2:
                    torch.nn.init.kaiming_uniform_(local)
                else:
                    torch.nn.init.zeros_(local)
                param_count += 1
            for _, buf in self.model.named_buffers():
                if buf is not None:
                    buf.zero_()
        return param_count

    def _lazy_init_hsdp_modules(self) -> int:
        """Re-wrap HSDP shards into DTensor so loader / optimizer see them."""
        reset_count = 0
        for state in self._iter_hsdp_states():
            if hasattr(state, 'lazy_init'):
                state.lazy_init()
                reset_count += 1
        return reset_count

    def _load_weights(self, weights_path: str) -> None:
        """Load pre-trained weights from ``weights_path`` into the (possibly sharded) model.

        Uses hyper's distributed checkpoint ``load`` API so that each rank only
        reads the shard it owns.  Falls back to a plain ``torch.load`` + partial
        ``load_state_dict`` for single-file checkpoints (e.g. safetensors).

        Args:
            weights_path: Path to a directory containing a distributed checkpoint,
                          or a single ``.pt`` / ``.bin`` file.
        """
        logger.info_rank0("Loading weights from %s", weights_path)
        try:
            if os.path.isdir(weights_path):
                hf_index = os.path.join(weights_path, "model.safetensors.index.json")
                # Delegate model-specific renaming / expert-splitting to
                # the per-spec ``state_dict_adapter``.
                adapter_cls = getattr(self.spec, "state_dict_adapter", None)
                if os.path.isfile(hf_index) and adapter_cls is not None:
                    self._load_hf_safetensors(weights_path, adapter_cls)
                else:
                    self._load_hyper_dcp(weights_path)
            else:
                self._load_single_file(weights_path)
            logger.info_rank0("Weights loaded from %s", weights_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load weights from {weights_path}: {exc}. "
                "weights_path was provided so silent random-init fallback is unsafe — "
                "uniform-logits loss would corrupt downstream training metrics."
            ) from exc

    def _load_hf_safetensors(self, weights_path: str, adapter_cls) -> None:
        """Load HF safetensors via spec's ``state_dict_adapter``; drop shape mismatches."""
        # Cast loaded params down to the checkpoint's advertised dtype so the
        # fp32 master matches what forward consumes.
        load_dtype = self._resolve_hf_load_dtype(weights_path)
        adapter = adapter_cls()
        hf_sd = adapter.load_hf_state_dict(
            weights_path, self.model.config, dtype=load_dtype,
        )
        valid_sd, dropped, missing, unexpected = self._validate_hf_state_dict(hf_sd)
        if dropped:
            logger.warning(
                "Dropped %d keys due to shape mismatch (first 5: %s)",
                len(dropped), dropped[:5],
            )
        # Derive missing/unexpected ourselves — ``HSDPModule.load_state_dict``
        # returns ``None``.
        self.model.load_state_dict(valid_sd, strict=False)
        model_name = getattr(self.args.model, "name", "")
        logger.info_rank0(
            "HF (%s) load: %d tensors into hyper model",
            model_name, len(valid_sd),
        )
        if missing:
            logger.warning(
                "Missing (randomly initialised): %d keys, e.g. %s ...",
                len(missing), missing[:5],
            )
        if unexpected:
            logger.warning(
                "Unexpected (ignored): %d keys, e.g. %s ...",
                len(unexpected), unexpected[:5],
            )

    def _resolve_hf_load_dtype(self, weights_path: str):
        """Resolve the dtype to cast loaded HF tensors to (matches checkpoint config)."""
        dtype_map = {
            'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16,
            'float16': torch.float16, 'fp16': torch.float16,
            'float32': torch.float32, 'fp32': torch.float32,
        }
        cfg_dtype = (
            getattr(self.model.config, 'dtype', None)
            or getattr(self.model.config, 'torch_dtype', None)
        )
        if cfg_dtype is None:
            cfg_json = os.path.join(weights_path, 'config.json')
            if os.path.isfile(cfg_json):
                try:
                    with open(cfg_json, 'r', encoding='utf-8') as f:
                        cfg = json.load(f)
                    cfg_dtype = cfg.get('dtype') or cfg.get('torch_dtype')
                except (OSError, json.JSONDecodeError):
                    cfg_dtype = None
        if isinstance(cfg_dtype, str):
            return dtype_map.get(cfg_dtype)
        if isinstance(cfg_dtype, torch.dtype):
            return cfg_dtype
        return None

    def _validate_hf_state_dict(self, hf_sd: dict):
        """Strip wrapper segments and drop tensors whose shape differs from the model.

        Pre-validate shapes: ``load_state_dict`` aborts on the first mismatch
        and leaves later keys un-loaded.

        Returns:
            ``(valid_sd, dropped, missing, unexpected)``.
        """
        # Strip wrapper segments (e.g. ``_checkpoint_wrapped_module``) so
        # loader keys match ``named_parameters`` paths.
        wrapper_segments = ("._checkpoint_wrapped_module",)

        def _strip(k: str) -> str:
            for s in wrapper_segments:
                k = k.replace(s, "")
            return k
        logical_to_real = {}
        real_to_param = {}
        for name, param in self.model.named_parameters():
            logical_to_real[_strip(name)] = name
            real_to_param[name] = param
        valid_sd: dict = {}
        dropped: list = []
        for hf_name, hf_tensor in hf_sd.items():
            real_name = logical_to_real.get(hf_name)
            if real_name is None:
                continue
            tgt = tuple(real_to_param[real_name].shape)
            src = tuple(hf_tensor.shape)
            if src == tgt:
                valid_sd[real_name] = hf_tensor
            else:
                dropped.append((real_name, src, tgt))
        param_names = set(real_to_param.keys())
        loaded_names = set(valid_sd.keys())
        missing = sorted(param_names - loaded_names)
        unexpected = sorted(loaded_names - param_names)
        return valid_sd, dropped, missing, unexpected

    def _load_hyper_dcp(self, weights_path: str) -> None:
        """Load weights from hyper's own DCP checkpoint format."""
        model_sd = self.model.state_dict()
        dcp_load(model_sd, checkpoint_id=weights_path, use_collectives=False)
        self.model.load_state_dict(model_sd)

    def _load_single_file(self, weights_path: str) -> None:
        """Load weights from a single ``.pt`` / ``.safetensors`` / ``.bin`` file."""
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            logger.warning("Missing keys when loading weights: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys when loading weights: %s", unexpected)

    def _maybe_toggle_reshard(self, micro_step: int, num_micro_steps: int):
        """Toggle FSDP reshard_after_backward for gradient accumulation optimization.

        During gradient accumulation, skip resharding between micro-steps to avoid
        redundant all-gather. Only reshard after the last micro-step.
        """
        if not isinstance(self.model, HSDPModule) or num_micro_steps <= 1:
            return
        if micro_step == 0:
            self.model.set_reshard_after_backward(False)
        elif micro_step == num_micro_steps - 1:
            self.model.set_reshard_after_backward(True)
