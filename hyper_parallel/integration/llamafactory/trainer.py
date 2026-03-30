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
"""HyperParallel trainer backend for LlamaFactory."""
import json
import logging
import os
import types
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from torch import nn
from transformers import Seq2SeqTrainer

from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.core.fully_shard.api import HSDPModule, hsdp_sync_stream
from hyper_parallel.core.utils import clip_grad_norm_ as hp_clip_grad_norm_
from hyper_parallel.integration.llamafactory.utils import fsdp2_prepare_model
from hyper_parallel.platform import get_platform

logger = logging.getLogger(__name__)

_VALID_DTYPES = {"float32", "float16", "bfloat16", "fp32", "fp16", "bf16"}
_HSDP_MODEL_NAME = "hsdp_model"
_HSDP_OPTIMIZER_NAME = "optimizer"


@dataclass
class HyperParallelArguments:
    """Minimal HyperParallel configuration needed by the trainer backend."""

    tp_size: int = 1
    device_type: str = "auto"
    param_dtype: Optional[str] = None
    reduce_dtype: Optional[str] = None
    reshard_after_forward: Optional[bool] = None

    def validate(self) -> None:
        """Validate supported argument values."""
        if self.tp_size != 1:
            raise ValueError(
                "Current trainer backend only supports replacing FSDP/fully_shard. "
                f"Expected tp_size=1, got {self.tp_size}."
            )
        if self.param_dtype is not None and self.param_dtype not in _VALID_DTYPES:
            raise ValueError(
                f"param_dtype must be one of {sorted(_VALID_DTYPES)}, got {self.param_dtype!r}."
            )
        if self.reduce_dtype is not None and self.reduce_dtype not in _VALID_DTYPES:
            raise ValueError(
                f"reduce_dtype must be one of {sorted(_VALID_DTYPES)}, got {self.reduce_dtype!r}."
            )
        if self.device_type not in {"auto", "npu", "cuda", "cpu"}:
            raise ValueError(
                f"device_type must be one of ['auto', 'cpu', 'cuda', 'npu'], got {self.device_type!r}."
            )
        if self.reshard_after_forward is not None and not isinstance(self.reshard_after_forward, bool):
            raise ValueError(
                "reshard_after_forward must be a bool when provided, "
                f"got {type(self.reshard_after_forward).__name__}."
            )

    @classmethod
    def from_dict(cls, config: dict) -> "HyperParallelArguments":
        """Build arguments from a plain dict."""
        known_fields = set(cls.__dataclass_fields__)  # pylint: disable=no-member
        hp_args = cls(**{key: value for key, value in config.items() if key in known_fields})
        hp_args.validate()
        return hp_args

    @classmethod
    def from_finetuning_args(cls, finetuning_args) -> "HyperParallelArguments":
        """Extract HyperParallel arguments from LlamaFactory finetuning args."""
        raw = getattr(finetuning_args, "hyper_parallel_args", None)
        if raw is None:
            hp_args = cls()
            hp_args.validate()
            return hp_args
        if isinstance(raw, str):
            with open(raw, "r", encoding="utf-8") as file:
                raw = json.load(file)
        if not isinstance(raw, dict):
            raise ValueError(
                "finetuning_args.hyper_parallel_args must be a dict or JSON file path, "
                f"got {type(raw).__name__}."
            )
        return cls.from_dict(raw)


def _localize_optimizer_state(optim_sd: dict) -> dict:
    """Convert DTensors in optimizer state dict to local CPU tensors for serialization.

    Args:
        optim_sd: Optimizer state dict from ``optimizer.state_dict()``.

    Returns:
        A new state dict with the same structure but all DTensor / Tensor values
        replaced by their local (shard) equivalents on CPU.
    """
    from hyper_parallel.core.dtensor.dtensor import DTensor as _DTensor  # pylint: disable=C0415

    new_state = {}
    for param_idx, state in optim_sd.get("state", {}).items():
        local_state = {}
        for key, val in state.items():
            if isinstance(val, _DTensor):
                local_state[key] = val.to_local().detach().cpu()
            elif isinstance(val, torch.Tensor):
                local_state[key] = val.detach().cpu()
            else:
                local_state[key] = val
        new_state[param_idx] = local_state
    return {"state": new_state, "param_groups": optim_sd.get("param_groups", [])}


def _load_local_optimizer_state(optimizer, saved_sd: dict) -> None:
    """Copy saved local optimizer state into the optimizer's current (possibly DTensor-backed) state.

    Args:
        optimizer: The optimizer whose state to restore.
        saved_sd: State dict saved by ``_localize_optimizer_state`` (local CPU tensors).
    """
    from hyper_parallel.core.dtensor.dtensor import DTensor as _DTensor  # pylint: disable=C0415

    # Build param index → param object mapping
    param_by_idx: dict[int, torch.nn.Parameter] = {}
    idx = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            param_by_idx[idx] = p
            idx += 1

    for param_idx, saved_state in saved_sd.get("state", {}).items():
        param_idx = int(param_idx) if isinstance(param_idx, str) else param_idx
        param = param_by_idx.get(param_idx)
        if param is None or param not in optimizer.state:
            continue
        current_state = optimizer.state[param]
        for key, saved_val in saved_state.items():
            current_val = current_state.get(key)
            if current_val is None:
                # New state entry (e.g. step counter added later)
                if isinstance(saved_val, torch.Tensor):
                    device = param.to_local().device if isinstance(param, _DTensor) else param.device
                    current_state[key] = saved_val.to(device)
                else:
                    current_state[key] = saved_val
            elif isinstance(current_val, _DTensor):
                local = current_val.to_local()
                local.copy_(saved_val.to(local.device))
            elif isinstance(current_val, torch.Tensor):
                current_val.copy_(saved_val.to(current_val.device))
            else:
                current_state[key] = saved_val

    # Restore hyper-parameters (lr, betas, etc.)
    for saved_group, current_group in zip(saved_sd.get("param_groups", []), optimizer.param_groups):
        for key, val in saved_group.items():
            if key != "params":
                current_group[key] = val


def _wrap_optimizer_step_with_skip_dtensor_dispatch(optimizer) -> None:
    """Wrap optimizer.step so DTensor dispatch is skipped during parameter updates."""
    if getattr(optimizer, "_hp_step_wrapped", False):
        return

    original_step = optimizer.step

    def _hp_step(bound_optimizer, *args, **kwargs):
        del bound_optimizer
        with SkipDTensorDispatch():
            return original_step(*args, **kwargs)

    optimizer.step = types.MethodType(_hp_step, optimizer)
    setattr(optimizer, "_hp_step_wrapped", True)


def _export_to_hf_format(model: nn.Module, tokenizer, save_dir: str):
    """Gather full state dict via HyperParallel and save in HuggingFace-compatible format.

    Uses HyperParallel's own ``get_model_state_dict(full_state_dict=True, cpu_offload=True)``
    which calls ``DTensor.full_tensor()`` (all-gather) for each sharded parameter.
    Rank 0 gets the full gathered weights on CPU; other ranks get an empty dict.
    """
    from hyper_parallel.core.fully_shard.api import (  # pylint: disable=C0415
        get_model_state_dict as hp_get_model_state_dict,
    )
    from torch.distributed.checkpoint.state_dict import StateDictOptions  # pylint: disable=C0415

    export_dir = Path(save_dir)
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state_dict = hp_get_model_state_dict(model, options=options)
    state_dict = _normalize_hf_export_state_dict(state_dict)

    if get_platform().get_rank() == 0:
        export_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(export_dir), state_dict=state_dict)
        else:
            torch.save(state_dict, export_dir / "pytorch_model.bin")

        if tokenizer is not None:
            tokenizer.save_pretrained(str(export_dir))

    if get_platform().get_world_size() > 1:
        torch.distributed.barrier()


def _normalize_hf_export_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize gathered tensors to match the baseline HF/LlamaFactory export.

    HyperParallel mixed precision can leave the live parameters in reduced
    precision, which halves the on-disk checkpoint size compared with the
    baseline FSDP2 export. The baseline path saves full-precision weights, so
    cast floating tensors back to fp32 before forwarding them to HF save logic.

    Shared/tied tensors are cast once and then reused to preserve aliasing.
    """
    normalized: dict[str, Any] = {}
    cast_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            normalized[key] = value
            continue

        if value.dtype == torch.float32:
            normalized[key] = value
            continue

        storage = value.untyped_storage()
        cache_key = (
            storage.data_ptr(),
            value.storage_offset(),
            tuple(value.size()),
            tuple(value.stride()),
            value.device.type,
            str(value.dtype),
        )
        casted = cast_cache.get(cache_key)
        if casted is None:
            casted = value.to(dtype=torch.float32)
            cast_cache[cache_key] = casted
        normalized[key] = casted

    return normalized


class HyperParallelTrainer(Seq2SeqTrainer):
    """Trainer backend that swaps FSDP2 prepare for HyperParallel fully_shard."""

    def __init__(
        self,
        hp_args: HyperParallelArguments,
        finetuning_args=None,
        processor=None,
        ref_model: Optional[nn.Module] = None,
        **kwargs,
    ):
        kwargs["processing_class"] = kwargs.pop("tokenizer", kwargs.get("processing_class", None))
        gen_kwargs = kwargs.pop("gen_kwargs", None)
        self._hp_args = hp_args
        self.finetuning_args = finetuning_args
        super().__init__(**kwargs)
        if not getattr(self.accelerator, "is_fsdp2", False):
            raise ValueError("HyperParallel trainer requires Accelerate FSDP2 mode to be enabled.")
        if gen_kwargs is not None:
            self._gen_kwargs = gen_kwargs
        self.ref_model = ref_model

        if processor is not None:
            self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model, self._hp_args)
        self._orig_accelerator_clip_grad_norm = self.accelerator.clip_grad_norm_
        self._orig_fsdp2_prepare_model = None
        self._accelerator_patches_active = False

    def _activate_accelerator_patches(self) -> None:
        """Activate temporary Accelerate patches for HyperParallel training."""
        if self._accelerator_patches_active:
            return

        import accelerate.accelerator as acc_module  # pylint: disable=C0415

        hp_args = self._hp_args

        self._orig_fsdp2_prepare_model = acc_module.fsdp2_prepare_model

        def _hp_fsdp2_prepare_model(accelerator, model):
            return fsdp2_prepare_model(accelerator, model, hp_args)

        acc_module.fsdp2_prepare_model = _hp_fsdp2_prepare_model

        def _hp_clip_grad_norm(accelerator, parameters, max_norm, norm_type=2):
            if getattr(accelerator, "is_fsdp2", False):
                accelerator.unscale_gradients()
                parameter_list = list(parameters)
                parameter_ids = {id(param) for param in parameter_list}
                for model in accelerator._models:  # pylint: disable=protected-access
                    if not isinstance(model, HSDPModule):
                        continue
                    model_param_ids = {id(param) for param in model.parameters()}
                    if parameter_ids and parameter_ids.issubset(model_param_ids):
                        return hp_clip_grad_norm_(parameter_list, max_norm, norm_type=norm_type)
            return self._orig_accelerator_clip_grad_norm(parameters, max_norm, norm_type=norm_type)

        self.accelerator.clip_grad_norm_ = types.MethodType(_hp_clip_grad_norm, self.accelerator)
        self._accelerator_patches_active = True

    def _restore_accelerator_patches(self) -> None:
        """Restore Accelerate patches to avoid cross-trainer contamination."""
        if not self._accelerator_patches_active:
            return

        import accelerate.accelerator as acc_module  # pylint: disable=C0415

        if self._orig_fsdp2_prepare_model is not None:
            acc_module.fsdp2_prepare_model = self._orig_fsdp2_prepare_model
        self.accelerator.clip_grad_norm_ = self._orig_accelerator_clip_grad_norm
        self._accelerator_patches_active = False

    def _wrap_model(self, model: nn.Module, training: bool = True, dataloader=None) -> nn.Module:
        """Let Accelerate own FSDP2/HSDP wrapping so optimizer remapping stays correct."""
        del dataloader
        if isinstance(model, HSDPModule):
            return model
        if training and getattr(self.accelerator, "is_fsdp2", False):
            # Trainer usually wraps here, but FSDP2 must be prepared by Accelerate.
            return model
        return super()._wrap_model(model, training=training)

    def _get_train_sampler(self, *args, **kwargs):
        """Respect disable_shuffling when provided by the caller."""
        if getattr(self.finetuning_args, "disable_shuffling", False):
            return torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)

    def compute_loss(self, model, inputs, *args, **kwargs):
        """Support ASFT-style loss when a reference model is configured."""
        if getattr(self.finetuning_args, "use_asft_loss", False) and self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_outputs.logits
            outputs = model(**inputs)
            return self.compute_loss_func(outputs, inputs["labels"], ref_logits)
        return super().compute_loss(model, inputs, *args, **kwargs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Remove the prompt span from generated tokens during generation-based eval."""
        if self.args.predict_with_generate:
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            **gen_kwargs,
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(self, dataset, predict_results, skip_special_tokens: bool = True) -> None:
        """Save generation results to `generated_predictions.jsonl`."""
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info("Saving prediction results to %s", output_prediction_file)

        labels = np.where(
            predict_results.label_ids != getattr(self.data_collator, "label_pad_token_id", -100),
            predict_results.label_ids,
            self.processing_class.pad_token_id,
        )
        preds = np.where(
            predict_results.predictions != getattr(self.data_collator, "label_pad_token_id", -100),
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for index, pred in enumerate(preds):
            pad_len = np.nonzero(pred != self.processing_class.pad_token_id)[0]
            if len(pad_len):
                preds[index] = np.concatenate((pred[pad_len[0] :], pred[: pad_len[0]]), axis=-1)

        input_ids_column = dataset["input_ids"]
        try:
            input_ids_list = input_ids_column.to_pylist()
        except AttributeError:
            input_ids_list = list(input_ids_column)

        decoded_inputs = self.processing_class.batch_decode(input_ids_list, skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as file:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                file.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    def _move_model_to_device(self, model: nn.Module, device: Optional[torch.device] = None):
        """Skip redundant device moves for HSDP-wrapped models."""
        if isinstance(model, HSDPModule):
            return model
        if device is None:
            return model
        return model.to(device)

    def train(self, *args, **kwargs):
        """Activate HP-specific Accelerate patches only during training."""
        self._activate_accelerator_patches()
        try:
            return super().train(*args, **kwargs)
        finally:
            self._restore_accelerator_patches()

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """Keep Accelerate training flow and only add HSDP sync hooks."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        sync_gradients = getattr(self.accelerator, "sync_gradients", True)
        if isinstance(model, HSDPModule):
            model.set_is_last_backward(sync_gradients)
            model.set_requires_gradient_sync(sync_gradients)

        compute_loss_context_manager = getattr(self, "compute_loss_context_manager", nullcontext)
        with compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if not getattr(self, "model_accepts_loss_kwargs", False) and getattr(self, "compute_loss_func", None) is None:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        if isinstance(model, HSDPModule) and sync_gradients:
            hsdp_sync_stream()

        return loss.detach()

    def create_optimizer(self):
        """Create optimizer and wrap step with SkipDTensorDispatch."""
        optimizer = super().create_optimizer()
        _wrap_optimizer_step_with_skip_dtensor_dispatch(optimizer)
        return optimizer

    # ---- Checkpoint save/load via HyperParallel native APIs ----

    def _save_optimizer_and_scheduler(self, output_dir: str) -> None:
        """Save model/optimizer shards per-rank and scheduler via torch.save.

        - Model: saved via HyperParallel's ``hp_save(use_collectives=False)`` so each
          rank writes its own shard independently (no collective communication).
        - Optimizer: DTensor state values are converted to local CPU tensors and
          saved per-rank via ``torch.save``.
        - Scheduler: standard ``torch.save`` (same as Trainer default).
        """
        from hyper_parallel.core.checkpoint.api import save as hp_save  # pylint: disable=C0415

        os.makedirs(output_dir, exist_ok=True)
        rank = get_platform().get_rank()

        # Model shards (for checkpoint resuming, separate from save_model HF export)
        model_dir = os.path.join(output_dir, f"{_HSDP_MODEL_NAME}_0")
        os.makedirs(model_dir, exist_ok=True)
        logger.info("Saving HSDP model shards to %s (rank %d)", model_dir, rank)
        model_sd = self.model.state_dict()
        hp_save(model_sd, checkpoint_id=model_dir, use_collectives=False)

        # Optimizer shards (per-rank, local tensors)
        if self.optimizer is not None:
            optim_file = os.path.join(output_dir, f"{_HSDP_OPTIMIZER_NAME}_rank{rank}.pt")
            logger.info("Saving optimizer shard to %s", optim_file)
            local_optim_sd = _localize_optimizer_state(self.optimizer.state_dict())
            torch.save(local_optim_sd, optim_file)

        # Scheduler (standard torch.save, same as Trainer default)
        if self.args.should_save and self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model: Optional[nn.Module] = None) -> None:
        """Load model from HSDP sharded checkpoint saved by ``hp_save``."""
        from hyper_parallel.core.checkpoint.api import load as hp_load  # pylint: disable=C0415

        target = model if model is not None else self.model
        model_dir = os.path.join(resume_from_checkpoint, f"{_HSDP_MODEL_NAME}_0")

        if not os.path.isdir(model_dir):
            # Fallback to standard Trainer load (HF weights / FSDP checkpoint)
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        logger.info("Loading HSDP model shards from %s", model_dir)
        state_dict = target.state_dict()
        hp_load(state_dict, checkpoint_id=model_dir, use_collectives=False)
        # hp_load modifies DTensor local storage in-place via the planner;
        # call load_state_dict to ensure consistency with HSDP internal bookkeeping.
        target.load_state_dict(state_dict)

        # Remember checkpoint dir for optimizer loading in _load_optimizer_and_scheduler
        self._pending_hsdp_checkpoint = resume_from_checkpoint
        return None

    def _load_optimizer_and_scheduler(self, checkpoint: Optional[str] = None) -> None:
        """Load optimizer/scheduler from per-rank checkpoint files."""
        ckpt_dir = getattr(self, "_pending_hsdp_checkpoint", None) or checkpoint
        if ckpt_dir is None:
            return

        rank = get_platform().get_rank()
        optim_file = os.path.join(ckpt_dir, f"{_HSDP_OPTIMIZER_NAME}_rank{rank}.pt")

        if os.path.isfile(optim_file) and self.optimizer is not None:
            logger.info("Loading optimizer shard from %s", optim_file)
            saved_sd = torch.load(optim_file, map_location="cpu", weights_only=True)
            _load_local_optimizer_state(self.optimizer, saved_sd)

        # Scheduler
        scheduler_file = os.path.join(ckpt_dir, "scheduler.pt")
        if os.path.isfile(scheduler_file) and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(torch.load(scheduler_file, map_location="cpu", weights_only=True))

    def save_model(  # pylint: disable=invalid-name
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        """Save model weights.

        Match the baseline LlamaFactory behavior by exporting HF-format weights
        both for intermediate checkpoints and the final output directory.
        HSDP-native shards for resume are still handled separately by
        ``_save_optimizer_and_scheduler``.
        """
        save_dir = output_dir or self.args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        _export_to_hf_format(self.model, getattr(self, "processing_class", None), save_dir)


__all__ = ["HyperParallelArguments", "HyperParallelTrainer"]
