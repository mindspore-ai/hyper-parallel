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

Design notes:
- Composition over inheritance: a trainer holds a ``BaseTrainer`` and calls its
  13 ``_build_*`` steps in order, overriding or skipping steps as needed.
- FSDP/AC wrapping iterates ``model.layers`` when the model exposes decoder layers.
- Parallel composition order is TP → CP → AC → FSDP.

Subclasses (LLMTrainer, VLMTrainer, ...) follow this pattern: instantiate a
``BaseTrainer`` and drive its ``_build_*`` methods selectively.
"""
import json
import logging
import math
import os
import random
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DistributedSampler

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
# ``_resolve_local_tensor`` is the canonical shard resolver used by
# ``HSDPModule.load_state_dict``; reused (rather than duplicated) to load a
# checkpoint into a model that holds DTensor params but is not itself an
# ``HSDPModule`` (pipeline parallelism composed with per-module FSDP).
from hyper_parallel.core.fully_shard.api import _resolve_local_tensor
from hyper_parallel.core.fully_shard.hsdp_utils import GroupInfo
from hyper_parallel.core.utils import clip_grad_norm_
from hyper_parallel.data import build_dataset
from hyper_parallel.models.spec.registry import get_spec
from hyper_parallel.trainer.config import get_vision_parallel_config
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
    TrainingStateMonitorCallback,
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
        self.substep_info: Dict[str, Any] = {}


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
    # Pipeline-parallel state — set by ``_build_pipelined_model`` when ``pp>1``.
    pp_enabled: bool = False
    pp_schedule: Optional[Any] = None
    pp_micro_batch_num: int = 1
    pp_has_first_stage: bool = False
    pp_has_last_stage: bool = False
    _pp_tie_embeddings: bool = False
    _pp_stage_fsdp_sharded: bool = False

    def __init__(self, args):
        # Only early-bound fields live here; the rest is built via
        # ``_build_*`` methods invoked by the subclass.
        self.args = args
        self.spec = get_spec(args.model.name)
        self.state = TrainerState(max_steps=args.train.max_steps)
        self._pp_stage_modules: list["nn.Module"] = []
        self._pp_tp_loss_repeats = 1

    # ------------------------------------------------------------------
    # 13 overridable _build_* methods
    # ------------------------------------------------------------------

    @property
    def _deterministic(self) -> bool:
        return bool(self.args.train.debug.deterministic)

    def _apply_pre_init_deterministic_env(self):
        """Pin HCCL / PYTHONHASHSEED before ``init_process_group`` boots the backend."""
        if not self._deterministic:
            return
        seed = self.args.train.seed
        os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "1")
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        os.environ.setdefault("FLASH_ATTENTION_DETERMINISTIC", "1")
        os.environ.setdefault("HCCL_DETERMINISTIC", "true")
        os.environ.setdefault("PYTHONHASHSEED", str(seed))

    def _parallel_dim_size(self, name: str) -> int:
        """Return a configured parallel dimension size."""
        return int(getattr(self.parallel_dims, name, 1) or 1)

    def _cp_size(self) -> int:
        """Return configured context-parallel size."""
        return self._parallel_dim_size("cp")

    def _share_samples_across_dp(self) -> bool:
        """Return whether the visual validation path reuses samples across DP."""
        return get_vision_parallel_config(self.args.model).get(
            "share_samples_across_dp", False,
        )

    def _setup(self):
        """Step 1: Initialize distributed environment, device mesh, and seed.

        Calls hyper's own ``init_process_group`` and ``init_device_mesh``.
        Mesh shape is derived from ``args.parallel`` (dp, tp, cp, pp, ep).
        """
        self._apply_pre_init_deterministic_env()
        backend = self.args.train.comm_backend
        init_process_group(backend=backend)

        local_rank = self.args.train.local_rank
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
        # Mixed precision lives in FSDP2's MixedPrecisionPolicy, so a
        # low-precision run needs a dp_shard axis (size-1 is enough) for the
        # FSDP wrap to exist — see ``build_mesh``'s force_dp_shard contract.
        mp_cfg = self.args.train.mixed_precision
        needs_mp_wrap = bool(
            mp_cfg.enabled
            and mp_cfg.param_dtype not in ('float32', 'fp32')
        )
        # PP stages carry the dtype policy only through a per-stage FSDP wrap,
        # which exists only for pure dp_shard sharding (no HSDP, see
        # ``_resolve_fsdp_mesh``) — reject every PP composition that would
        # silently run full-precision instead.
        if (needs_mp_wrap and self.parallel_dims.pp > 1
                and (self.parallel_dims.dp_shard == 1
                     or self.parallel_dims.dp_replicate > 1)
                and self._cp_size() == 1):
            raise ValueError(
                "mixed_precision with a low-precision param_dtype under PP "
                "needs an FSDP-wrappable data-parallel axis: the dtype policy "
                "lives on the per-stage FSDP wrap, which neither pure PP nor "
                "PP+HSDP provides. Use dp_shard>=2 with dp_replicate=1, or "
                "set param_dtype=float32."
            )
        self.mesh = self.parallel_dims.build_mesh(
            platform.device_type(), force_dp_shard=needs_mp_wrap,
        )

        # Build DP group_info for trainer-level all_reduce (loss/token sync).
        # Uses hyper's GroupInfo + mesh.get_group (platform-agnostic).

        dp_group = self._get_combined_dp_group()
        dp_size = self.parallel_dims.dp_size
        self._dp_group_info = GroupInfo(
            group_name="trainer_dp", group=dp_group, rank_size=dp_size,
        )

        seed = self.args.train.seed
        platform.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        # ``platform.manual_seed`` only covers CPU; seed the device RNG too.
        try:
            handle = platform.get_device_handle(device_type)
            if hasattr(handle, "manual_seed_all"):
                handle.manual_seed_all(seed)
            elif hasattr(handle, "manual_seed"):
                handle.manual_seed(seed)
        except Exception as exc:  # pylint: disable=W0718
            logger.warning("Device-side seed init skipped: %s", exc)

        if self._deterministic:
            warn_only = self.args.train.debug.deterministic_warn_only
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # TF32 affects CUDA only; the attribute may be missing on older torch.
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            except AttributeError:
                pass
            logger.info_rank0("Deterministic algorithms enabled (warn_only=%s)", warn_only)

        logger.info_rank0(
            "Setup complete: rank=%d, world_size=%d, mesh=%s",
            platform.get_rank(), platform.get_world_size(),
            self.mesh.mesh_dim_names,
        )
        logger.info_rank0(
            "Config: data.type=%s, model.name=%s, model.num_hidden_layers=%s, "
            "init_device=%s, max_steps=%d, global_bs=%d",
            self.args.data.type,
            self.args.model.name,
            self.args.model.num_hidden_layers,
            self.args.train.init_device,
            self.state.max_steps,
            self.args.train.global_batch_size,
        )

    def _build_model(self):
        """Step 2: Construct model via ``spec.build_model_fn``.

        The model is a plain ``nn.Module`` at this point — not yet parallelized.
        When ``args.runtime.init_device == "meta"``, the model is constructed on
        the meta device (no memory allocated) and real weights are loaded after
        FSDP sharding via ``_load_weights_after_parallel``.
        """
        init_device = self.args.train.init_device
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
        seq_len = self.args.data.max_seq_len
        self.parallel_dims.validate_against_model(self.model, seq_len=seq_len)

    def _freeze_model(self):
        """Step 3: Freeze specified modules (optional)."""
        freeze_modules = self.args.model.freeze_modules
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
        """Step 6: Build training dataset via the data-type registry.

        Dispatches on ``args.data.type`` against
        :data:`hyper_parallel.data.DATASET_REGISTRY`. Built-in formats:
        ``dummy``, ``hf_datasets``, ``json_file``, ``preset_pt``,
        ``vl_dummy``, ``megatron``. Plug in a custom format by importing
        a module that calls ``@DATASET_REGISTRY.register(...)``.

        Subclasses can override to populate ``self.train_dataset``
        differently before this method runs (or skip it entirely).
        """
        if getattr(self, "train_dataset", None) is not None:
            return
        if self.args.data.streaming:
            # ``DistributedSampler`` requires ``__len__``; an iterable path
            # would need a sampler-less dataloader. Reject loudly until that
            # path is wired so users see a clear error instead of a
            # ``TypeError: object of type ... has no len()``.
            raise NotImplementedError(
                "data.streaming=True is not yet wired. The default "
                "_build_dataloader uses DistributedSampler which requires "
                "len(dataset); subclass _build_dataset + _build_dataloader "
                "to emit an IterableDataset that self-shards via dp_rank/dp_size."
            )
        data_type = self.args.data.type
        self.train_dataset = build_dataset(
            data_type,
            base=self,
            args=self.args,
            tokenizer=getattr(self, "tokenizer", None),
            data_transform=getattr(self, "data_transform", None),
        )

    def _build_collate_fn(self):
        """Step 7: Build data collator.

        Default: pads input_ids and labels to max length in the batch.
        SequenceParallel TP and context parallel both slice the sequence
        dim, so variable-length batches additionally pad up to a multiple
        of ``cp * tp`` — the trailing pad carries label ``-100``, which the
        CE masks out, so the padding is mathematically inert.
        """
        seq_divisor = self.parallel_dims.seq_divisor

        def _default_collate(batch):
            """Simple padding collator."""
            max_len = max(item["input_ids"].size(0) for item in batch)
            if seq_divisor > 1 and max_len % seq_divisor:
                max_len += seq_divisor - max_len % seq_divisor
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
            out = {
                "input_ids": torch.stack(input_ids_list),
                "labels": torch.stack(labels_list),
            }
            if "num_items_in_batch" in batch[0]:
                out["num_items_in_batch"] = sum(
                    int(item["num_items_in_batch"]) for item in batch
                )
            if "attention_mask" in batch[0]:
                masks = []
                for item in batch:
                    pad_len = max_len - item["attention_mask"].size(0)
                    masks.append(torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0))
                out["attention_mask"] = torch.stack(masks)
            if "position_ids" in batch[0]:
                positions = []
                for item in batch:
                    pos = item["position_ids"]
                    pad_len = max_len - pos.shape[-1]
                    positions.append(torch.nn.functional.pad(pos, (0, pad_len), value=0))
                if positions[0].dim() == 1:
                    out["position_ids"] = torch.stack(positions)
                else:
                    out["position_ids"] = torch.stack(positions).transpose(0, 1).contiguous()
            return out

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

        micro_bs = self.args.train.micro_batch_size

        # Sampler uses DP rank/size — TP/CP/PP/EP peers share data.
        dp_size = self.parallel_dims.dp_size
        non_dp = self.parallel_dims.non_dp_size
        global_rank = platform.get_rank()
        try:
            dp_rank = self.mesh["dp"].get_local_rank()
        except (KeyError, ValueError, RuntimeError):
            dp_rank = global_rank // non_dp if non_dp > 1 else global_rank

        shuffle = self.args.data.shuffle
        sampler_seed = self.args.train.seed
        self.sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=1 if self._share_samples_across_dp() else dp_size,
            rank=0 if self._share_samples_across_dp() else dp_rank,
            shuffle=shuffle,
            seed=sampler_seed,
            drop_last=True,
        )

        # StatefulDataLoader supports state_dict() / load_state_dict()
        # for checkpoint resume (torchdata API, used by  + ).
        num_workers = self.args.data.num_workers
        prefetch_factor = self.args.data.prefetch_factor
        pin_memory = self.args.data.pin_memory

        # Spawned-worker RNG is not bit-stable across 1c↔Nc; force num_workers=0
        # in deterministic mode.
        if self._deterministic and num_workers > 0:
            logger.warning(
                "debug.deterministic=True forces data.num_workers from %d → 0",
                num_workers,
            )
            num_workers = 0

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
        if self._deterministic:
            # Pin loader RNG to the trainer seed so shuffle order is stable.
            gen = torch.Generator()
            gen.manual_seed(int(self.args.train.seed))
            loader_kwargs["generator"] = gen
        self.train_dataloader = StatefulDataLoader(
            self.train_dataset, **loader_kwargs,
        )

        # Use dp_size (not world_size) — TP/CP/PP ranks share data, not split it.
        self._grad_accum = max(
            self.args.train.global_batch_size // (
                micro_bs * (1 if self._share_samples_across_dp() else dp_size)
            ),
            1,
        )

        if self._share_samples_across_dp():
            logger.warning_rank0(
                "vision_parallel.share_samples_across_dp=true. Use this only for "
                "validation/self-consistency checks; normal training should keep "
                "distinct samples across DP ranks."
            )
        logger.info_rank0(
            "Dataloader built: micro_bs=%d, grad_accum=%d, dataset_size=%d, "
            "share_samples_across_dp=%s",
            micro_bs, self._grad_accum, len(self.train_dataset),
            str(self._share_samples_across_dp()),
        )

    def _build_parallelized_model(self):
        """Step 9: Apply parallel strategies to the model.

        Each model owns its full parallelize pipeline in
        ``models/<name>/parallelize.py`` (convention) and
        registers it via ``ModelSpec.parallelize_fn``. There is no shared
        "default" template — model-specific TP/EP/CP/AC/FSDP/Prefetch
        composition lives next to the model that needs it.
        """
        if self.parallel_dims.pp_enabled:
            self._build_pipelined_model()
            return
        if self.spec.parallelize_fn is None:
            raise ValueError(
                f"Model '{self.spec.name}' has no ``parallelize_fn`` registered "
                f"on its ModelSpec. Each model must own its parallelize "
                f"pipeline in models/<name>/parallelize.py."
            )
        self.model = self.spec.parallelize_fn(self.model, self.mesh, self.args)
        self._post_parallelize()

    def _validate_pp_model_parallel_grad_clipping(self, dims) -> None:
        """Reject PP model-parallel clipping until DTensor norms are placement-aware."""
        max_grad_norm = float(self.args.train.optimizer.max_grad_norm)
        if max_grad_norm > 0 and (dims.tp > 1 or dims.ep > 1):
            raise NotImplementedError(
                "Trainer PP with TP or EP requires max_grad_norm=0: the current "
                "pipeline gradient norm does not yet deduplicate replicated "
                "DTensor placements while reducing TP/EP shards."
            )

    def _set_pp_stage_modules(self, stages: list[Any]) -> None:
        """Expose local stage modules and validate their data-parallel representation."""
        if len(stages) == 1:
            self.model = stages[0].submodule
        else:
            self.model = torch.nn.ModuleList([stage.submodule for stage in stages])
        self._pp_stage_fsdp_sharded = any(
            isinstance(module, HSDPModule)
            for module in self.model.modules()
        )
        has_plain_dtensor = any(isinstance(param, DTensor) for param in self.model.parameters())
        if self._pp_fsdp_composed and not self._pp_stage_fsdp_sharded and has_plain_dtensor:
            raise NotImplementedError(
                "Trainer PP data-parallel fallback cannot synchronize DTensor "
                "stage parameters across the combined DP group. Use dp_shard "
                "with dp_replicate=1 for PP+TP/EP, or disable TP/EP when using "
                "PP with dp_replicate>1."
            )

    def _validate_pp_runtime_options(self, dims) -> int:
        """Validate PP loss, batch, checkpointing, and export options."""
        # The PP loss/grad is normalized to the global token mean. This is
        # equivalent to ``rank_average`` when every row has the same number of
        # valid labels; the runtime validates that case before scheduling.
        agg = self.args.train.optimizer.loss_aggregation
        if agg not in ('token_weighted', 'rank_average'):
            raise NotImplementedError(
                f"Trainer PP supports loss_aggregation='token_weighted' or "
                f"'rank_average' with uniform valid-token rows only (got {agg!r})."
            )

        # The schedule sees the effective batch after the dataloader floors the
        # configured global batch, so validate the effective size here.
        micro_num = int(self.args.train.accelerator.pp_micro_batch_num)
        if micro_num < 1:
            raise ValueError(f"pp_micro_batch_num ({micro_num}) must be >= 1.")
        global_bs = self.args.train.global_batch_size
        micro_bs = int(self.args.train.micro_batch_size)
        grad_accum = max(int(global_bs) // (micro_bs * dims.dp_size), 1)
        effective_bs = grad_accum * micro_bs
        if effective_bs % micro_num != 0:
            raise ValueError(
                f"effective PP batch ({effective_bs} = grad_accum*"
                f"micro_batch_size, floored from global_batch_size={global_bs}) "
                f"must be divisible by pp_micro_batch_num ({micro_num}); "
                f"adjust global_batch_size / micro_batch_size / pp_micro_batch_num."
            )

        # The PP path bypasses ``parallelize_fn`` and replaces the full model
        # with a stage fragment, so AC and HF-weight export are not yet wired.
        ac_mode = self.args.train.gradient_checkpointing.activation_checkpoint
        if ac_mode not in ("off", "none", None, False, ""):
            raise NotImplementedError(
                f"activation_checkpoint={ac_mode!r} is not yet wired for the "
                f"trainer PP path; set gradient_checkpointing.activation_checkpoint "
                f"to 'none' for pp>1."
            )
        if self.args.train.checkpoint.save_hf_weights:
            raise NotImplementedError(
                "checkpoint.save_hf_weights is not yet supported under the "
                "trainer PP path (each rank holds only a stage fragment); set "
                "save_hf_weights=false for pp>1."
            )
        return micro_num

    def _build_pipelined_model(self) -> None:
        """Pipeline-parallel build path (``pp > 1``).

        Unlike the ``parallelize_fn`` path, the model is **first** materialized
        and weight-loaded as the *full* network (``_post_parallelize`` is FSDP-
        agnostic — ``to_empty`` + ``load_state_dict(strict=False)`` work on an
        unwrapped module), then handed to ``spec.pipelining_fn`` which slices it
        into this rank's :class:`Qwen3_5StageModule` and returns the
        ``ScheduleGPipe`` + stages. ``self.model`` is then re-pointed at the
        stage module so the optimizer / grad-clip built next see only this
        rank's stage parameters.

        The trainer supports PP alone and the model-provided FSDP/TP/EP
        compositions validated below. Unsupported domains such as PP+CP,
        model-parallel clipping without a placement-aware norm, and plain-DP
        fallback over DTensor stage parameters fail before training starts.
        """
        if self.spec.pipelining_fn is None:
            raise ValueError(
                f"Model '{self.spec.name}' has parallel.pp>1 but no "
                f"``pipelining_fn`` registered on its ModelSpec. Register the "
                f"model's pipeline splitter (e.g. ``pipeline_<name>_for_trainer``)."
            )
        dims = self.parallel_dims
        # PP composed with FSDP (dp_shard / dp_replicate): each stage's children
        # are wrapped as FSDP units (load-before-shard) and the 1F1B schedule
        # defers grad reduction to the final micro-batch backward — every micro
        # accumulates the unsharded grad locally, then the explicit
        # FSDP_REDUCE_GRAD step reduces once (see the torch pipeline stage's
        # per-micro grad-sync defer + ``PipelineStage.execute_reduce_grad``).
        # EP shards experts within each layer (intra-stage). TP / CP shard the
        # token sequence; the pipeline carries the sequence-sharded hidden states
        # across stages (lm_head re-gathers for a full-sequence loss).
        if dims.cp > 1:
            raise NotImplementedError(
                "Trainer pipeline parallelism supports PP alone, PP+FSDP, "
                f"PP+EP+FSDP, or PP+TP+FSDP (got cp={dims.cp}). Composing PP with "
                "CP is not yet wired."
            )
        self._validate_pp_model_parallel_grad_clipping(dims)
        self._pp_fsdp_composed = dims.dp_shard > 1 or dims.dp_replicate > 1
        micro_num = self._validate_pp_runtime_options(dims)
        # Capture the tie flag while ``self.model`` is still the full model — the
        # PP grad-clip dedups the tied embed / lm_head, which otherwise lives on
        # two stages (stage 0's ``embed_tokens`` + the last stage's ``lm_head``).
        self._pp_tie_embeddings = bool(
            getattr(self.model.config, "tie_word_embeddings", False)
        )
        init_device = self.args.train.init_device
        if self._pp_fsdp_composed:
            if init_device != "meta":
                raise NotImplementedError(
                    "Trainer PP+FSDP currently requires init_device='meta' "
                    f"(got {init_device!r}): each stage's FSDP units are sharded "
                    "on the meta device, then materialized + weight-loaded as "
                    "shards — the same meta path as non-PP FSDP."
                )
            # Wrap-on-meta then materialize: ``pipelining_fn`` splits the meta
            # model and ``fully_shard``-wraps the stage's children, producing
            # correctly-sized meta shards. ``_post_parallelize`` then runs while
            # ``self.model`` is still the full model, so ``_load_weights`` maps
            # the checkpoint by the full-model parameter names (the stage shares
            # those exact param objects, so its shards receive the weights too).
            # Doing it the other way round (materialize full → ``fully_shard`` a
            # real param) leaves the loaded full tensor in place and trips FSDP's
            # sharded-size check at the first forward.
            self.pp_schedule, stages = self.spec.pipelining_fn(
                self.model, self.mesh, self.args,
            )
            self._pp_stage_modules = [stage.submodule for stage in stages]
            self._post_parallelize()
            # The stage was built while the model was still on meta (so
            # ``fully_shard`` could create meta shards), which left
            # ``stage.device`` on meta. ``_post_parallelize`` materialized the
            # params to the real device; point the stage there too so its P2P
            # activation buffers — allocated lazily on ``stage.device`` — land
            # on the compute device instead of meta.
            for stage in stages:
                stage.device = self.device
                # The stage's init-time shared-parameter broadcast was skipped on
                # meta; now that the shards are materialized + weight-loaded, sync
                # the tied embed / lm_head ends so both stages start identical.
                stage._sync_shared_parameters()  # pylint: disable=protected-access
        else:
            # PP alone: materialize + load the full model, then split (no FSDP
            # wrap). The full model must be on the trainer device before the
            # split so a CPU ``init_device`` doesn't leave stages on CPU while
            # ``_pp_train_step`` moves batches to ``self.device``.
            self._post_parallelize()
            self.model = self.model.to(self.device)
            self.pp_schedule, stages = self.spec.pipelining_fn(
                self.model, self.mesh, self.args,
            )
            self._pp_stage_modules = [stage.submodule for stage in stages]
        self._pp_tp_loss_repeats = max(int(getattr(self.model, "hp_loss_tp_scale_size", 1)), 1)
        pp_mesh = self.mesh["pp"]
        pp_rank = pp_mesh.get_local_rank()
        self.pp_enabled = True
        self.pp_micro_batch_num = micro_num
        self.pp_has_first_stage = pp_rank == 0
        self.pp_has_last_stage = pp_rank == pp_mesh.size() - 1
        # Pipeline group for broadcasting the last stage's loss to every rank.
        self._pp_group_info = GroupInfo(
            group_name="trainer_pp", group=pp_mesh.get_group(),
            rank_size=pp_mesh.size(),
        )
        # First stage's global rank — the broadcast source for single-reader
        # data loading in ``_pp_train_step`` (constant, so resolve it once).
        self._pp_src_rank = platform.get_global_rank(pp_mesh.get_group(), 0)
        # Re-point ``self.model`` at this rank's stage(s) so the optimizer and
        # gradient clipping operate on the stage parameters only. Under VPP a
        # rank owns several non-contiguous chunks; expose all their submodules
        # (a ModuleList) so every chunk's params are optimized / clipped.
        self._set_pp_stage_modules(stages)
        logger.info_rank0(
            "Pipeline build: pp_size=%d, this rank is stage %d (first=%s, last=%s)",
            pp_mesh.size(), pp_rank, self.pp_has_first_stage, self.pp_has_last_stage,
        )

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

        This pattern handles partial checkpoints cleanly: any parameter the
        checkpoint does not supply (e.g. a reduced-layer run where the loader
        filters out higher layers' keys) keeps its kaiming / zero init, while
        every key the checkpoint does provide overwrites it. The full Qwen3-VL-
        MoE checkpoint supplies every module the model defines — ``q_norm`` /
        ``k_norm`` (per text layer), the vision ``pos_embed`` and
        ``deepstack_merger_list`` included — so a complete load leaves nothing
        random.
        """
        init_device = self.args.train.init_device
        weights_path = self.args.model.weights_path
        if init_device == "meta":
            # Always materialize first (random init baseline) so no param
            # stays on meta — then overlay the checkpoint.
            self._materialize_and_init_shards()
            if weights_path:
                self._load_weights(weights_path)
        elif weights_path:
            self._load_weights(weights_path)
        # Mixed-precision storage policy: respect the configured param_dtype
        # for both trainable and frozen params so optimizer state follows the
        # same precision contract the forward advertises.
        self._maybe_downcast_frozen_params()
        self._maybe_cast_trainable_params()
        self.model.train()

    def _maybe_downcast_frozen_params(self) -> None:
        """Maybe downcast frozen params (internal)."""
        freeze_modules = self.args.model.freeze_modules
        if not freeze_modules:
            return
        mp_cfg = self.args.train.mixed_precision
        if not mp_cfg.enabled:
            return

        target_dtype = {
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
            'float16': torch.float16,
            'fp16': torch.float16,
        }.get(mp_cfg.param_dtype)
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

    def _maybe_cast_trainable_params(self) -> None:
        """Cast trainable params to the configured mixed-precision storage dtype."""
        mp_cfg = self.args.train.mixed_precision
        if not mp_cfg.enabled:
            return

        dtype_map = {
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
            'float16': torch.float16,
            'fp16': torch.float16,
            'float32': torch.float32,
            'fp32': torch.float32,
        }
        target_dtype = dtype_map.get(mp_cfg.param_dtype)
        if target_dtype is None:
            return
        target_reduce_dtype = dtype_map.get(mp_cfg.reduce_dtype)

        def _get_param_local_tensor(param: platform.Parameter) -> platform.Tensor:
            data = param.data
            if isinstance(data, DTensor):
                return data.to_local()
            return data

        def _set_param_local_tensor(param: platform.Parameter, local: platform.Tensor) -> None:
            data = param.data
            if isinstance(data, DTensor):
                param.data = DTensor.from_local(
                    local,
                    device_mesh=data.device_mesh,
                    placements=data.placements,
                    shape=tuple(data.shape),
                )
            else:
                param.data = local

        def _cast_param_data(param: platform.Parameter) -> bool:
            if not param.requires_grad:
                return False
            local = _get_param_local_tensor(param)
            if local.dtype == target_dtype:
                return False
            new_local = local.to(target_dtype)
            _set_param_local_tensor(param, new_local)
            return True

        n_cast = 0
        seen_param_ids = set()
        for _, param in self.model.named_parameters():
            seen_param_ids.add(id(param))
            if _cast_param_data(param):
                n_cast += 1
        def _refresh_hsdp_dtype(hsdp_param) -> None:
            hsdp_param.orig_dtype = target_dtype
            hsdp_param.param_dtype = None
            hsdp_param.reduce_dtype = (
                None if target_reduce_dtype == target_dtype else target_reduce_dtype
            )
            hsdp_param.unsharded_param_buffers = []
            hsdp_param.reset_sharded_param()
            if hasattr(hsdp_param, "_unsharded_param"):
                delattr(hsdp_param, "_unsharded_param")

        def _refresh_hsdp_state_dtype(state) -> None:
            if state.param_group is None:
                return
            state.param_group.reset_iter_state()
            state.param_group.all_gather_buckets = []

        for state in self._iter_hsdp_states():
            buckets = (
                getattr(state, 'replicate_params', []) or [],
                getattr(state, 'hsdp_params', []) or [],
            )
            for bucket in buckets:
                for hsdp_param in bucket:
                    param = getattr(hsdp_param, 'sharded_param', None)
                    if param is None:
                        continue
                    if id(param) not in seen_param_ids and _cast_param_data(param):
                        n_cast += 1
                    seen_param_ids.add(id(param))
                    _refresh_hsdp_dtype(hsdp_param)
            _refresh_hsdp_state_dtype(state)
        logger.info_rank0(
            "Post-load: cast %d trainable params to %s", n_cast, target_dtype,
        )

    def _build_optimizer(self):
        """Step 10: Build optimizer. Must be called AFTER ``_build_parallelized_model``.

        After FSDP, parameters are DTensor shards — optimizer operates on local shards.
        Optimizer must be created after ``fully_shard``.
        """
        lr = self.args.train.optimizer.lr
        weight_decay = self.args.train.optimizer.weight_decay

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
        adam_eps = self.args.train.optimizer.eps
        adam_betas = self.args.train.optimizer.betas
        adam_foreach = self.args.train.optimizer.foreach
        # ``None`` intentionally follows PyTorch/HF ``adamw_torch`` defaults.
        # Deterministic mode controls algorithm selection globally; it should not
        # silently change the optimizer kernel unless the YAML asks for it.
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
        warmup_ratio = self.args.train.optimizer.lr_warmup_ratio
        # ``ceil`` matches the standard warmup convention so a fractional
        # ``warmup_ratio * max_steps`` rounds up to the next full step.
        warmup_steps = math.ceil(total_steps * warmup_ratio)
        decay_style = self.args.train.optimizer.lr_decay_style
        lr_min = self.args.train.optimizer.lr_min
        lr_max = self.args.train.optimizer.lr

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

        Mixed precision is realised entirely through FSDP2
        ``MixedPrecisionPolicy`` (param_dtype / reduce_dtype / output_dtype).
        No autocast context is entered — the model's own ``.float()`` /
        ``.to(weight.dtype)`` cast points handle the fp32 residual stream.
        """
        mp_cfg = self.args.train.mixed_precision
        self.model_fwd_context = nullcontext()
        self.model_bwd_context = nullcontext()
        self.grad_scaler = None
        if mp_cfg.enabled:
            logger.info_rank0(
                "Mixed precision via FSDP2 mp_policy: param=%s reduce=%s on %s",
                mp_cfg.param_dtype,
                mp_cfg.reduce_dtype,
                platform.device_type(),
            )

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
        self.training_state_monitor_callback = TrainingStateMonitorCallback(self)
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
            "training_state_monitor, " 
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
            self.training_state_monitor_callback,
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
        self.training_state_monitor_callback.on_train_begin(self.state)
        self.profiler_callback.on_train_begin(self.state)
        self.wandb_callback.on_train_begin(self.state)
        self.tensorboard_callback.on_train_begin(self.state)
        # Checkpoint runs after log writers are armed and before progress so
        # resumed ``global_step`` is reflected in the tqdm initial position.
        self.checkpoint_callback.on_train_begin(self.state)
        self.progress_callback.on_train_begin(self.state)
        for cb in self.user_callbacks:
            cb.on_train_begin(self.state)

    def on_train_end(self):
        """Dispatch on_train_end to all callbacks."""
        self.checkpoint_callback.on_train_end(self.state)
        self.hf_export_callback.on_train_end(self.state)
        self.progress_callback.on_train_end(self.state)
        self.training_state_monitor_callback.on_train_end(self.state)
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
        self.training_state_monitor_callback.on_step_end(
            self.state, loss=loss, grad_norm=grad_norm,
        )
        for cb in self._all_callbacks():
            if cb is self.training_state_monitor_callback:
                continue
            cb.on_step_end(self.state, loss=loss, grad_norm=grad_norm)

    def on_substep_end(self):
        """Dispatch on_substep_end (after each micro-batch forward/backward)."""
        self.moe_monitor_callback.on_substep_end(self.state)
        self.training_state_monitor_callback.on_substep_end(self.state)
        for cb in self.user_callbacks:
            cb.on_substep_end(self.state)

    def on_pre_optimizer_step(self, grad_norm=None):
        """Dispatch on_pre_optimizer_step (after grad clip, before optimizer.step)."""
        # Health check runs FIRST so a NaN aborts before the logger misleads.
        self.training_state_monitor_callback.on_pre_optimizer_step(
            self.state, grad_norm=grad_norm,
        )
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

    def _move_value_to_device(self, value):
        """Move nested tensor-like values to this trainer's device."""
        if hasattr(value, "to"):
            return value.to(self.device, non_blocking=True)
        if isinstance(value, dict):
            return {k: self._move_value_to_device(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._move_value_to_device(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._move_value_to_device(v) for v in value)
        return value

    def _prepare_forward_batch(self, micro_batch):
        """Move a micro-batch to device and extract CP-shifted labels."""
        micro_batch = {
            key: self._move_value_to_device(value)
            for key, value in micro_batch.items()
        }
        labels_are_shifted = bool(micro_batch.pop("_hp_labels_are_shifted", False))
        shifted_labels = micro_batch.pop("labels", None) if labels_are_shifted else None
        if labels_are_shifted and shifted_labels is None:
            raise ValueError("CP-shifted loss marker is set but labels are missing.")
        return micro_batch, labels_are_shifted, shifted_labels

    def _compute_micro_loss(
        self,
        outputs,
        labels_are_shifted: bool,
        shifted_labels,
        micro_batch_tokens: int,
    ):
        """Return mean loss and summed loss for one micro-batch."""
        if not labels_are_shifted:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            return loss, loss.detach() * max(micro_batch_tokens, 1)

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        target_device = logits.device if hasattr(logits, "device") else self.device
        shifted_labels = shifted_labels.to(target_device, non_blocking=True)
        loss_sum = torch.nn.functional.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            shifted_labels.contiguous().view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        return loss_sum / max(micro_batch_tokens, 1), loss_sum

    def _scale_loss_for_backward(
        self,
        loss,
        loss_sum,
        labels_are_shifted: bool,
        micro_batch_tokens: int,
        global_tokens: int,
        num_micro: int,
    ):
        """Scale one micro-batch loss according to trainer loss aggregation."""
        dp_size = self.parallel_dims.dp_size
        agg = self.args.train.optimizer.loss_aggregation
        cp_size = self._cp_size()
        cp_rank_average = agg == "rank_average" and cp_size > 1
        if agg == 'rank_average' and not cp_rank_average:
            scaled_loss = loss / num_micro if num_micro > 1 else loss
            rank_average_loss_scale_size = getattr(
                self.model,
                "hp_rank_average_loss_scale_size",
                1,
            )
            if rank_average_loss_scale_size != 1:
                scaled_loss = scaled_loss / rank_average_loss_scale_size
            return scaled_loss

        loss_scale_size = getattr(self.model, "hp_token_loss_scale_size", dp_size)
        if labels_are_shifted:
            scaled_loss = loss_sum / max(global_tokens, 1) * loss_scale_size
        else:
            scaled_loss = mean_global_loss(
                loss, micro_batch_tokens, global_tokens, loss_scale_size,
            )
        tp_loss_scale_size = getattr(
            self.model,
            "hp_loss_tp_scale_size",
            max(1, self._parallel_dim_size("tp")),
        )
        if tp_loss_scale_size != 1:
            scaled_loss = scaled_loss / tp_loss_scale_size
        ep_loss_scale_size = getattr(self.model, "hp_loss_ep_scale_size", 1)
        if ep_loss_scale_size != 1:
            scaled_loss = scaled_loss / ep_loss_scale_size
        return scaled_loss

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
        micro_batch, labels_are_shifted, shifted_labels = self._prepare_forward_batch(micro_batch)

        # Forward (with training context for activation offload)
        with self.model_fwd_context:
            outputs = self.model(**micro_batch, use_cache=False)
        loss, loss_sum = self._compute_micro_loss(
            outputs, labels_are_shifted, shifted_labels, micro_batch_tokens,
        )

        # TP scenario: loss may be Partial DTensor — reduce before backward
        if hasattr(loss, 'is_partial') and loss.is_partial():
            loss = loss.reduce_partial()

        # Keep raw loss value for logging before scaling
        raw_loss = loss.detach()

        scaled_loss = self._scale_loss_for_backward(
            loss,
            loss_sum,
            labels_are_shifted,
            micro_batch_tokens,
            global_tokens,
            num_micro,
        )

        # Backward (with training context)
        with self.model_bwd_context:
            scaled_loss.backward()

        return raw_loss, micro_batch_tokens

    def _shard_micro_batches_for_cp(self, micro_batches):
        """Slice each micro-batch's sequence onto this context-parallel rank.

        Under CP the model forward consumes only this rank's sequence slice (the
        Ulysses all-to-all / sequence-gather reconstruct the full sequence inside
        attention). The next-token shift is performed here on the **full**
        sequence before slicing so the cross-rank boundary target is preserved.
        The model remains HF-like: it receives explicit global ``position_ids``
        and no CP-only forward arguments. The trainer computes cross-entropy
        from the model logits for these pre-shifted local targets, and the
        per-rank token counts aggregate back to the single-card loss across the
        ``cp`` group (folded into the trainer's loss / FSDP reduction). No-op
        when ``cp<=1``.

        Args:
            micro_batches: List of per-micro-batch dicts from the data iterator.

        Returns:
            The CP-sharded micro-batch list (or the input unchanged when ``cp<=1``).
        """
        cp_size = self._cp_size()
        if cp_size <= 1:
            return micro_batches
        cp_rank = self.mesh["cp"].get_local_rank()
        sharded = []
        for micro_batch in micro_batches:
            input_ids = micro_batch["input_ids"]
            seq_len = input_ids.shape[1]
            if seq_len % cp_size != 0:
                raise ValueError(
                    f"sequence length ({seq_len}) must be divisible by cp ({cp_size})."
                )
            shard = seq_len // cp_size
            start = cp_rank * shard
            seq_slice = slice(start, start + shard)
            local = dict(micro_batch)
            local["input_ids"] = input_ids[:, seq_slice].contiguous()
            position_ids = micro_batch.get("position_ids")
            if position_ids is not None:
                if position_ids.dim() == 2:
                    local["position_ids"] = position_ids[:, seq_slice].contiguous()
                else:
                    pos_slice = [slice(None)] * position_ids.dim()
                    pos_slice[-1] = seq_slice
                    local["position_ids"] = position_ids[tuple(pos_slice)].contiguous()
            else:
                has_multimodal_positions = any(
                    micro_batch.get(name) is not None
                    for name in (
                        "pixel_values", "image_grid_thw", "pixel_values_videos",
                        "video_grid_thw", "mm_token_type_ids",
                    )
                )
                if not has_multimodal_positions:
                    local["position_ids"] = torch.arange(
                        start, start + shard, device=input_ids.device, dtype=torch.long,
                    ).view(1, -1).expand(input_ids.shape[0], -1)
            mm_token_type_ids = micro_batch.get("mm_token_type_ids")
            if mm_token_type_ids is not None:
                mm_slice = [slice(None)] * mm_token_type_ids.dim()
                mm_slice[-1] = seq_slice
                local["mm_token_type_ids"] = mm_token_type_ids[
                    tuple(mm_slice)
                ].contiguous()
            labels = micro_batch.get("labels")
            if labels is not None:
                shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
                local["labels"] = shifted[:, seq_slice].contiguous()
                local["_hp_labels_are_shifted"] = True
            attn = micro_batch.get("attention_mask")
            if attn is not None and hasattr(attn, "dim") and attn.dim() == 2:
                local["attention_mask"] = attn[:, seq_slice].contiguous()
            sharded.append(local)
        return sharded

    def _collect_global_tokens(self, token_counts):
        """Count valid loss tokens and all-reduce across the data-parallel group."""
        local_tokens = sum(token_counts) or 1
        global_tokens = local_tokens
        if platform.get_world_size() > 1 and self._dp_group_info.group is not None:
            token_tensor = platform.full((1,), local_tokens).to(self.device)
            platform.all_reduce(token_tensor, self._dp_group_info)
            global_tokens = max(int(token_tensor.item()), 1)
        self._last_global_tokens = global_tokens
        return local_tokens, global_tokens

    def _run_micro_batches(self, micro_batches, token_counts, global_tokens):
        """Run forward/backward over accumulated micro-batches."""
        num_micro = len(micro_batches)
        total_loss_sum = 0.0
        total_loss_arith_sum = 0.0
        total_tokens_local = 0
        for index, micro_batch in enumerate(micro_batches):
            is_last = index == num_micro - 1
            if isinstance(self.model, HSDPModule):
                self.model.set_requires_gradient_sync(is_last)
                self.model.set_is_last_backward(is_last)
            self._maybe_toggle_reshard(index, num_micro)

            raw_loss, micro_tokens = self.forward_backward_step(
                micro_batch,
                token_counts[index],
                global_tokens,
                num_micro=num_micro,
            )
            loss_value = raw_loss.item()
            total_loss_sum += loss_value * micro_tokens
            total_loss_arith_sum += loss_value
            total_tokens_local += micro_tokens
            self.state.substep_info = {
                "raw_loss": loss_value,
                "micro_tokens": micro_tokens,
            }
            self.on_substep_end()
        return total_loss_sum, total_loss_arith_sum, total_tokens_local

    def _run_post_fsdp_grad_reduce(self) -> None:
        """Run an optional model-provided reducer after FSDP gradients drain."""
        post_fsdp_grad_reduce = getattr(self.model, "hp_post_fsdp_grad_reduce", None)
        if post_fsdp_grad_reduce is not None:
            post_fsdp_grad_reduce()

    def _non_pp_clip_grad_norm(self, max_grad_norm: float):
        """Clip non-pipeline gradients using the configured clipping function."""
        clip_fn = self.spec.clip_grad_fn or clip_grad_norm_
        return clip_fn(self.model.parameters(), max_grad_norm)

    def _optimizer_step_after_backward(self, clip_fn):
        """Clip gradients if enabled, run optimizer/scheduler, and clear grads."""
        max_grad_norm = float(self.args.train.optimizer.max_grad_norm)
        grad_norm = clip_fn(max_grad_norm) if max_grad_norm > 0.0 else None
        grad_norm_value = None if grad_norm is None else grad_norm.item()
        self.on_pre_optimizer_step(grad_norm=grad_norm_value)

        with SkipDTensorDispatch():
            self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return grad_norm_value

    def _aggregate_non_pp_loss(
        self,
        total_loss_sum: float,
        total_loss_arith_sum: float,
        total_tokens_local: int,
        global_tokens: int,
        num_micro: int,
    ) -> float:
        """Aggregate the reported non-pipeline loss across DP ranks."""
        agg = self.args.train.optimizer.loss_aggregation
        cp_size = self._cp_size()
        if agg == "token_weighted" or (agg == "rank_average" and cp_size > 1):
            if platform.get_world_size() > 1 and self._dp_group_info.group is not None:
                loss_tensor = platform.full((1,), total_loss_sum).to(self.device)
                platform.all_reduce(loss_tensor, self._dp_group_info)
                return loss_tensor.item() / max(global_tokens, 1)
            return total_loss_sum / max(total_tokens_local, 1)

        local_mean = total_loss_arith_sum / max(num_micro, 1)
        dp_size = self._dp_group_info.rank_size
        if dp_size <= 1:
            return local_mean
        loss_tensor = platform.full((1,), local_mean).to(self.device)
        platform.all_reduce(loss_tensor, self._dp_group_info)
        return loss_tensor.item() / dp_size

    def _average_model_parallel_metric(self, avg_loss: float) -> float:
        """Average replicated loss metrics over model-parallel EP when needed."""
        tp_size = self._parallel_dim_size("tp")
        ep_size = self._parallel_dim_size("ep")
        if tp_size > 1 and ep_size > 1:
            return avg_loss
        if ep_size <= 1:
            return avg_loss
        try:
            ep_group = self.mesh.get_group("ep")
        except (KeyError, ValueError):
            return avg_loss
        metric = platform.full((1,), avg_loss).to(self.device)
        ep_group_info = GroupInfo(
            group_name="trainer_ep_metric",
            group=ep_group,
            rank_size=ep_size,
        )
        platform.all_reduce(metric, ep_group_info)
        return metric.item() / ep_size

    def train_step(self, data_iterator):
        """Execute one training step with gradient accumulation.

        Consistent across different DP configurations by:
        1. All-reducing global token count before loss scaling ()
        2. Syncing gradients only on the last micro-batch ()
        3. All-reducing loss weighted by token count for reporting

        Args:
            data_iterator: Iterator yielding lists of micro-batch dicts.
        """
        if self.pp_enabled:
            return self._pp_train_step(data_iterator)
        micro_batches = next(data_iterator)
        prepare_batch_fn = getattr(self.spec, "prepare_batch_fn", None)
        if prepare_batch_fn is not None:
            micro_batches = [
                prepare_batch_fn(batch, self.model)
                for batch in micro_batches
            ]
        micro_batches = self._shard_micro_batches_for_cp(micro_batches)
        self.state.global_step += 1
        num_micro = len(micro_batches)

        token_counts = [count_loss_token(mb) for mb in micro_batches]
        _, global_tokens = self._collect_global_tokens(token_counts)
        total_loss_sum, total_loss_arith_sum, total_tokens_local = self._run_micro_batches(
            micro_batches,
            token_counts,
            global_tokens,
        )

        # Wait for async gradient reduce
        #
        hsdp_sync_stream()
        self._run_post_fsdp_grad_reduce()
        grad_norm_value = self._optimizer_step_after_backward(self._non_pp_clip_grad_norm)
        avg_loss = self._aggregate_non_pp_loss(
            total_loss_sum,
            total_loss_arith_sum,
            total_tokens_local,
            global_tokens,
            num_micro,
        )
        avg_loss = self._average_model_parallel_metric(avg_loss)

        return {"loss": avg_loss, "grad_norm": grad_norm_value}

    @staticmethod
    def _pp_concat_micro_batches(micro_batches):
        """Concatenate grad-accum micro-batches into one global batch (dim 0).

        Under PP the schedule owns micro-batching, so the trainer rebuilds the
        global batch from the grad-accum group and lets ``ScheduleGPipe``
        re-split it into ``pp_micro_batch_num`` chunks.

        The pipeline runs a single fused ``sum``-CE backward over the whole
        batch, which reproduces the trainer's ``token_weighted`` single-card
        gradient **only when every micro-batch shares the same sequence length**
        (then ``sum-CE / valid_tokens`` is the common token-mean). Micro-batches
        of differing shape are therefore rejected with a clear error — pad to a
        fixed ``max_seq_len`` so the grad-accum group is uniform, or size the
        batch so ``grad_accum == 1``. Non-tensor values are taken from the first
        micro-batch.
        """
        if len(micro_batches) == 1:
            return dict(micro_batches[0])
        merged = {}
        for key in micro_batches[0].keys():
            values = [mb[key] for mb in micro_batches]
            first = values[0]
            if not hasattr(first, "dim"):
                merged[key] = first
                continue
            if any(value.shape[1:] != first.shape[1:] for value in values):
                raise NotImplementedError(
                    f"PP gradient accumulation requires uniform-shape "
                    f"micro-batches; '{key}' varies across the group (shapes "
                    f"{[tuple(value.shape) for value in values]}). Pad to a fixed "
                    f"max_seq_len, or size the batch so grad_accum == 1."
                )
            merged[key] = torch.cat(values, dim=0)
        return merged

    def _pp_clip_grad_norm(self, max_grad_norm: float):
        """Clip gradients by the **global** norm across all pipeline stages.

        Each stage holds a disjoint parameter slab, so the single-card total
        norm is recovered by summing the per-stage squared norms and all-reducing
        over the pipeline group. The shared coefficient is then applied on every
        stage — essential for the tied embed / lm_head, whose stage-0 and
        last-stage copies must receive the *same* scaling to stay bit-identical
        after the optimizer step (a per-stage coefficient would desync them).

        The tied copy is counted once: the last stage skips its ``lm_head.weight``
        duplicate from the norm sum (it equals stage 0's ``embed_tokens.weight``)
        but is still scaled, so the global norm matches the single-card norm.

        Args:
            max_grad_norm: Clip threshold; the effective coefficient is
                ``min(1, max_grad_norm / total_norm)``.

        Returns:
            The global gradient norm (a scalar tensor) for logging.
        """
        params = [p for p in self.model.parameters() if p.grad is not None]
        skip = None
        if self._pp_tie_embeddings and self.pp_has_last_stage:
            # The last global stage's submodule owns the tied ``lm_head``. Under
            # VPP ``self.model`` is a ModuleList of this rank's chunks, only one
            # of which (the last stage) carries ``lm_head`` — find it there.
            head_owner = self.model
            if isinstance(head_owner, torch.nn.ModuleList):
                head_owner = next(
                    (s for s in head_owner if hasattr(s, "lm_head")), None)
            if head_owner is not None and hasattr(head_owner, "lm_head"):
                skip = head_owner.lm_head.weight
        local_sq = torch.zeros((), device=self.device, dtype=torch.float32)
        for param in params:
            if param is skip:
                continue
            grad = param.grad.detach()
            # Under PP+FSDP the grad is a sharded DTensor; reduce on the local
            # shard so the cross-stage all-reduce stays a plain-tensor collective.
            if hasattr(grad, "to_local"):
                grad = grad.to_local()
            local_sq = local_sq + grad.float().pow(2).sum()
        platform.all_reduce(local_sq, self._pp_group_info)
        # Under PP+FSDP the grads are dp-sharded, so also sum the per-dp-shard
        # squared norms across the dp group to get the true global grad norm.
        if getattr(self, "_pp_stage_fsdp_sharded", False):
            platform.all_reduce(local_sq, self._dp_group_info)
        total_norm = local_sq.sqrt()
        clip_coef = (max_grad_norm / (total_norm + 1e-6)).clamp(max=1.0)
        for param in params:
            param.grad.mul_(clip_coef.to(param.grad.dtype))
        return total_norm

    def _pp_load_first_stage_batch(self, data_iterator):
        """Load and prepare the global PP batch on the first stage only."""
        batch = None
        targets = None
        stop = 0
        if not self.pp_has_first_stage:
            return batch, targets, stop
        try:
            micro_batches = next(data_iterator)
            batch = self._pp_concat_micro_batches(micro_batches)
            batch = {
                key: (value.to(self.device, non_blocking=True) if hasattr(value, "to") else value)
                for key, value in batch.items()
            }
            if batch["input_ids"].shape[0] % self.pp_micro_batch_num != 0:
                stop = 1
            else:
                labels = batch["labels"]
                targets = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:].to(torch.int64)
        except StopIteration:
            stop = 1
        return batch, targets, stop

    def _pp_broadcast_control(self, batch, targets, stop: int):
        """Broadcast stop/shape metadata across the pipeline group."""
        ctrl = platform.full((4,), 0, dtype=torch.int64).to(self.device)
        if stop:
            ctrl[0] = 1
        elif self.pp_has_first_stage:
            ctrl[1] = int(targets.shape[0])
            ctrl[2] = int(targets.shape[1])
            ctrl[3] = 1 if batch.get("attention_mask") is not None else 0
        platform.broadcast(ctrl, self._pp_src_rank, self._pp_group_info.group)
        return ctrl.tolist()

    def _pp_broadcast_2d_int64(self, src_tensor, rows: int, seq: int):
        """Broadcast one 2-D int64 tensor from the first pipeline stage."""
        tensor = (
            src_tensor.to(torch.int64).contiguous()
            if self.pp_has_first_stage
            else platform.full((rows, seq), 0, dtype=torch.int64).to(self.device)
        )
        platform.broadcast(tensor, self._pp_src_rank, self._pp_group_info.group)
        return tensor

    def _pp_prepare_broadcast_inputs(self, batch, targets, stop: int):
        """Broadcast targets and optional all-stage masks for one PP step."""
        stop, rows, seq, has_attn = self._pp_broadcast_control(batch, targets, stop)
        if stop:
            raise StopIteration

        targets = self._pp_broadcast_2d_int64(targets, rows, seq)
        attention_mask = None
        if has_attn:
            source_mask = batch["attention_mask"] if self.pp_has_first_stage else None
            attention_mask = self._pp_broadcast_2d_int64(source_mask, rows, seq)
        return targets, attention_mask, has_attn

    def _pp_count_valid_tokens(self, targets) -> int:
        """Count valid shifted targets and sum across DP for PP+FSDP."""
        n_valid = max(int((targets != -100).sum().item()), 1)
        if getattr(self, "_pp_fsdp_composed", False):
            token_tensor = platform.full((1,), n_valid).to(self.device)
            platform.all_reduce(token_tensor, self._dp_group_info)
            n_valid = max(int(token_tensor.item()), 1)
        self._last_global_tokens = n_valid
        return n_valid

    def _pp_validate_rank_average_targets(self, targets) -> None:
        """Validate the PP token-mean path also represents rank-average loss."""
        agg = self.args.train.optimizer.loss_aggregation
        if agg != "rank_average":
            return
        row_tokens = (targets != -100).sum(dim=1)
        if row_tokens.numel() <= 1:
            return
        if int(row_tokens.min().item()) == int(row_tokens.max().item()):
            return
        raise NotImplementedError(
            "Trainer PP with loss_aggregation='rank_average' requires uniform "
            "valid-token counts per row so the fused token-mean loss matches "
            "the single-card rank-average gradient."
        )

    def _pp_normalize_grads(self, n_valid: int) -> None:
        """Normalize fully reduced pipeline gradients to the global token mean.

        Core pipeline schedules retain unit backward sensitivity for standalone
        callers. After PP/FSDP/shared/TP/EP/DP reductions, multiplying the final
        averaged gradients by ``dp_size / (n_valid * tp_loss_repeats)`` yields
        the same global token mean before clipping and the optimizer step. The
        TP divisor removes duplicate backward sensitivity when the last stage
        materializes a replicated loss as a local tensor.
        """
        dp_size = max(int(self.parallel_dims.dp_size), 1)
        denominator = max(n_valid * self._pp_tp_loss_repeats, 1)
        grad_scale = dp_size / denominator
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            if grad is None:
                continue
            local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
            local_grad.mul_(grad_scale)

    def _pp_run_schedule(self, batch, targets, attention_mask, has_attn):
        """Run the configured PP schedule with the broadcast inputs."""
        run_kwargs = {"targets": targets}
        kwargs_batch_dim = getattr(self.pp_schedule, "_kwargs_batch_dim", {}) or {}
        if self.pp_has_first_stage:
            for key in kwargs_batch_dim:
                if key != "targets" and key in batch:
                    run_kwargs[key] = batch[key]
            return self.pp_schedule.run(batch["input_ids"], **run_kwargs)
        if has_attn and "attention_mask" in kwargs_batch_dim:
            run_kwargs["attention_mask"] = attention_mask
        return self.pp_schedule.run(**run_kwargs)

    def _pp_post_schedule_grad_reduce(self) -> None:
        """Run optional post-FSDP reducers on local pipeline stage modules."""
        stage_modules = list(self.model) if isinstance(self.model, torch.nn.ModuleList) else [self.model]
        for stage_module in stage_modules:
            stage_tp_reduce = getattr(stage_module, "hp_post_fsdp_grad_reduce", None)
            if stage_tp_reduce is not None:
                stage_tp_reduce()

    def _pp_average_plain_dp_grads(self) -> None:
        """Average plain replicated grads for PP+DP without per-stage FSDP shards."""
        if not getattr(self, "_pp_fsdp_composed", False):
            return
        dp_size = max(int(self.parallel_dims.dp_size), 1)
        if dp_size <= 1 or getattr(self, "_pp_stage_fsdp_sharded", False):
            return
        for param in self.model.parameters():
            if param.grad is not None:
                platform.all_reduce(param.grad, self._dp_group_info)
                param.grad.div_(dp_size)

    def _pp_reduce_reported_loss(self, outputs, n_valid: int) -> float:
        """Reduce last-stage sum-CE into a reported token-mean PP loss."""
        local_sum_ce = 0.0
        if self.pp_has_last_stage:
            local_sum_ce = sum(out.detach().float() for out in outputs).item()
        sum_ce_t = platform.full((1,), local_sum_ce).to(self.device)
        if getattr(self, "_pp_fsdp_composed", False):
            platform.all_reduce(sum_ce_t, self._dp_group_info)
        loss_t = sum_ce_t / n_valid
        platform.all_reduce(loss_t, self._pp_group_info)
        return loss_t.item()

    def _pp_train_step(self, data_iterator):
        """Pipeline-parallel training step (``pp > 1``).

        Only the first stage reads the dataloader; the last stage's ``targets``
        and the all-stage ``attention_mask`` are broadcast across the pipeline
        group so non-first stages never load (and, for VL, never decode) the
        identical batch. Heavy vision inputs stay on stage 0.

        ``ScheduleGPipe`` owns micro-batching and the forward/backward, so the
        trainer feeds it the **full** global batch (the grad-accum micro-batches
        concatenated). Only the last stage produces the per-micro-batch sum-CE;
        it is normalised to mean-CE and all-reduced across the pipeline group so
        every rank — including the rank-0 logger, which is the *first* stage —
        reports the same loss matching the single-card token-mean baseline.
        Gradient clipping uses the **global** cross-stage norm
        (:meth:`_pp_clip_grad_norm`) so every stage scales by the same
        coefficient — required so the tied embed / lm_head copies stay in sync.
        """
        batch, targets, stop = self._pp_load_first_stage_batch(data_iterator)
        targets, attention_mask, has_attn = self._pp_prepare_broadcast_inputs(batch, targets, stop)
        self.state.global_step += 1
        self._pp_validate_rank_average_targets(targets)
        n_valid = self._pp_count_valid_tokens(targets)
        outputs = self._pp_run_schedule(batch, targets, attention_mask, has_attn)
        self._pp_post_schedule_grad_reduce()
        self._pp_average_plain_dp_grads()
        self._pp_normalize_grads(n_valid)
        grad_norm_value = self._optimizer_step_after_backward(self._pp_clip_grad_norm)
        return {"loss": self._pp_reduce_reported_loss(outputs, n_valid), "grad_norm": grad_norm_value}

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
            self.args.train.num_train_epochs,
        )
        # on_train_begin runs checkpoint resume — state.global_step may be
        # updated to the resumed step before the loop starts.
        self.on_train_begin()
        num_epochs = self.args.train.num_train_epochs

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

        Default: ``model.layers`` when the model exposes decoder layers.
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
        # No data-parallel axis: pure TP still needs the 1-D group because its
        # SequenceParallel ranks hold different token shards. Pure EP peers see
        # the same tokens and must not be folded into the token/loss denominator.
        if self.mesh.mesh_dim_names == ("ep",):
            return None
        # Other 1-D meshes (pure TP; pure CP normally has a ``loss`` alias)
        # return their own group. Multi-dim meshes with no DP/loss axis return
        # ``None``.
        try:
            return self.mesh.get_group()
        except (ValueError, RuntimeError):
            return None

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

        reshard = self.args.train.accelerator.reshard_after_forward
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
        reshard = self.args.train.accelerator.reshard_after_forward
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

        This is the meta-init path used after ``fully_shard`` has installed
        FSDP views.
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
        # cause ``register_parameter`` to reject the assignment. Skipped under
        # PP: the tied embed / lm_head live on different stages, kept consistent
        # by the pipeline ``SharedParameterInfo`` (init broadcast + grad
        # all-reduce); a model-level tie would alias them into one object and
        # orphan the captured shared parameter (its grad would stay ``None``).
        if hasattr(self.model, "tie_weights") and int(self.parallel_dims.pp) <= 1:
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
        seen = set()
        roots = [self.model, *getattr(self, "_pp_stage_modules", [])]
        for root in roots:
            if root is None:
                continue
            for module in root.modules():
                if not isinstance(module, HSDPModule):
                    continue
                scheduler = getattr(module, 'hsdp_scheduler', None)
                state = getattr(scheduler, 'hsdp_state', None) if scheduler else None
                if state is None or id(state) in seen:
                    continue
                seen.add(id(state))
                yield state

    def _materialize_replicate_params(self, device_type: str) -> None:
        """Materialize meta ``_local_tensor`` storage that ``to_empty`` cannot reach.

        Walks ``replicate_params`` (explicit no-shard buckets, e.g. ``(1, H)``
        shapes) and, for single-card FSDP, ``hsdp_params`` — the flat-buffer
        rebase in ``_init_flat_param_buffer`` is skipped at
        ``shard_world_size == 1``, leaving those params on meta and tripping
        ``_validate_no_meta_params`` in ``lazy_init``. The two buckets are
        disjoint by construction (see ``state.py`` ``_init_hsdp_params``).
        """
        for state in self._iter_hsdp_states():
            buckets = (
                getattr(state, 'replicate_params', []) or [],
                getattr(state, 'hsdp_params', []) or [],
            )
            for bucket in buckets:
                for hsdp_param in bucket:
                    local = getattr(hsdp_param.sharded_param, "_local_tensor", None)
                    if local is not None and local.is_meta:
                        new_local = torch.empty_like(local, device=device_type)
                        hsdp_param.sharded_param._local_tensor = new_local  # pylint: disable=W0212
                        hsdp_param._sharded_param_data = new_local.view(-1)  # pylint: disable=W0212

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

    def _load_validated_state_dict(self, valid_sd: Dict[str, Any]) -> None:
        """Copy a validated plain-tensor state_dict into ``self.model``.

        Routes by model shape:

        * ``HSDPModule`` root (non-PP FSDP) — delegate to its shard-aware
          ``load_state_dict``, which distributes plain tensors onto local shards.
        * plain root with no DTensor params (no FSDP, or PP alone) — use the
          default ``load_state_dict`` (plain ``copy_``).
        * plain root that *holds* DTensor params (pipeline parallelism composed
          with per-module FSDP) — copy per-parameter, distributing each plain
          tensor onto its local shard. The default ``load_state_dict`` would
          recurse into the DTensor child and hit the unregistered DTensor
          ``copy_`` ("Operator copy_ does not contain parallel layout infer
          func").

        Args:
            valid_sd: Fully-qualified name → plain tensor, already shape-checked.
        """
        if isinstance(self.model, HSDPModule):
            self.model.load_state_dict(valid_sd, strict=False)
            return
        if not any(isinstance(p, DTensor) for _, p in self.model.named_parameters()):
            self.model.load_state_dict(valid_sd, strict=False)
            return
        targets: Dict[str, Any] = dict(self.model.named_parameters())
        targets.update(dict(self.model.named_buffers()))
        with platform.no_grad():
            for key, val in valid_sd.items():
                target = targets.get(key)
                if target is None:
                    continue
                if isinstance(target, DTensor):
                    val = _resolve_local_tensor(key, val, target)
                platform.load_into_param(target, val)

    def _load_hf_safetensors(self, weights_path: str, adapter_cls) -> None:
        """Load checkpoint safetensors via spec's ``state_dict_adapter``; drop shape mismatches."""
        # Cast loaded params down to the checkpoint's advertised dtype so the
        # fp32 master matches what forward consumes.
        load_dtype = self._resolve_hf_load_dtype(weights_path)
        adapter = adapter_cls()
        hf_sd = adapter.load_hf_state_dict(
            weights_path, self.model.config, dtype=load_dtype,
        )
        # Apply model-provided TP load transforms: slice the full checkpoint
        # weight onto this rank's shard for parameters the parallelize plan
        # sliced manually as plain (non-DTensor) tensors — e.g. Qwen3.5 GatedDeltaNet
        # ``conv1d`` / ``dt_bias`` / ``A_log`` under TP. The model is built on
        # meta and sliced before load, so without this the size-mismatched full
        # weight would be dropped (the shard then trains from random init).
        transform_fn = getattr(self.spec, "tp_load_transform_fn", None)
        if transform_fn is not None:
            for key, fn in transform_fn(self.model, self.mesh, self.args).items():
                if key in hf_sd:
                    hf_sd[key] = fn(hf_sd[key])
        valid_sd, dropped, missing, unexpected = self._validate_hf_state_dict(hf_sd)
        if dropped:
            logger.warning(
                "Dropped %d keys due to shape mismatch (first 5: %s)",
                len(dropped), dropped[:5],
            )
        # Derive missing/unexpected ourselves — ``HSDPModule.load_state_dict``
        # returns ``None``.
        self._load_validated_state_dict(valid_sd)
        model_name = self.args.model.name
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
        """Resolve the dtype to cast loaded checkpoint tensors to."""
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
        # Strip activation-checkpoint wrapper segments so loader keys match
        # ``named_parameters`` paths. The root module's parameter walk bypasses
        # each wrapper's own name-stripping override, so the segment leaks into
        # the FQN here. Covers the torch-native checkpoint_wrapper
        # (``_checkpoint_wrapped_module``), the hyper torch activation wrapper
        # (``_swap_wrapped_module``), and the hyper MindSpore activation wrapper
        # (``_ckpt_wrapped_module``); stripping an absent segment is a no-op.
        wrapper_segments = (
            "._checkpoint_wrapped_module",
            "._swap_wrapped_module",
            "._ckpt_wrapped_module",
        )
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
