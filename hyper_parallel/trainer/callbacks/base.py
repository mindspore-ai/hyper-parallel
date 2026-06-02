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
"""Callback base class and built-in callbacks.

dispatched explicitly in ``on_step_end`` etc. Engineer sees all callbacks and
order at a glance.

``checkpoint_callback.py`` (242 lines) + ``trace_callback.py`` (231 lines).
"""
import copy
import gc
import json
import logging
import math
import os
import threading
import time
from typing import TYPE_CHECKING, Optional

import torch

from hyper_parallel import get_platform
from hyper_parallel.core.distributed_checkpoint import load as dcp_load, save as dcp_save
from hyper_parallel.core.distributed_checkpoint.offline_transform import (
    save_state_dict_as_huggingface_format,
)
from hyper_parallel.core.fully_shard.api import get_model_state_dict

platform = get_platform()

if TYPE_CHECKING:
    from hyper_parallel.trainer.base import BaseTrainer, TrainerState

logger = logging.getLogger(__name__)

class Callback:
    """Base class for all trainer callbacks.

    Each callback holds a reference to the trainer for accessing model,
    optimizer, state, and config. Subclass and override the hooks you need.

    Args:
        trainer: The BaseTrainer instance.
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        self.trainer = trainer

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_init_end(self, state: "TrainerState", **kwargs) -> None:
        """Called once at the end of ``BaseTrainer.__init__`` / subclass init.

        At this point every ``_build_*`` has run — model is parallelised,
        optimizer/scheduler/dataloader are built, callbacks are constructed.
        Use this for one-shot setup that must see the FINAL trainer state
        (e.g. logging the parameter count, opening a TensorBoard writer
        keyed by run_id, validating user config against the built model).
        """

    def on_train_begin(self, state: "TrainerState", **kwargs) -> None:
        """Called at the start of ``train()`` (before any optimizer.step).

        ``CheckpointCallback`` runs resume here, so when this hook fires
        ``state.global_step`` may already be > 0 if a checkpoint was loaded.
        """

    def on_train_end(self, state: "TrainerState", **kwargs) -> None:
        """Called at the end of training (before ``destroy_process_group``).

        Final checkpoints, profiler stops, W&B finish, etc. happen here.
        """

    def on_epoch_begin(self, state: "TrainerState", **kwargs) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(self, state: "TrainerState", **kwargs) -> None:
        """Called at the end of each epoch."""

    def on_step_begin(self, state: "TrainerState", **kwargs) -> None:
        """Called at the start of each training step (before fwd of mb 0)."""

    def on_step_end(self, state: "TrainerState", *, loss: float = None,
                    grad_norm: float = None, **kwargs) -> None:
        """Called at the end of each training step (after optimizer.step)."""

    def on_substep_end(self, state: "TrainerState", **kwargs) -> None:
        """Called after each micro-batch fwd+bwd (gradient accumulation)."""

    def on_pre_optimizer_step(self, state: "TrainerState", *,
                              grad_norm: float = None, **kwargs) -> None:
        """Called after grad clip, before ``optimizer.step``.

        ``grad_norm`` here is the post-clip scalar produced by hyper's
        DTensor-aware clipper — use it to detect NaN/Inf or to log the
        effective clip ratio.
        """

    def on_log(self, state: "TrainerState", *, metrics: dict, **kwargs) -> None:
        """Called when ``LoggingCallback`` emits a structured metrics record.

        Reuse this hook in TensorBoard / W&B / external metric sinks so
        every logging backend sees the SAME record. Avoids three callbacks
        each computing throughput / lr independently.

        Args:
            metrics: Dict containing at minimum ``step``, ``loss``,
                ``grad_norm``, ``lr``, ``step_time``; throughput fields
                (``tokens_per_sec``, ``tflops``, ``mfu``) are present iff
                ``logging.report_throughput`` is on.
        """

    def on_save(self, state: "TrainerState", *, checkpoint_dir: str,
                **kwargs) -> None:
        """Called immediately after ``CheckpointCallback`` finishes a save.

        Use to upload to remote storage, register the ckpt with an
        experiment tracker, or trigger downstream eval jobs. ``checkpoint_dir``
        is the on-disk path containing model shards + optimizer/scheduler/RNG/
        dataloader/extra_state.
        """

    def on_load(self, state: "TrainerState", *, checkpoint_dir: str,
                **kwargs) -> None:
        """Called immediately after ``CheckpointCallback`` finishes a resume.

        Use to verify the resumed step matches expectations, log the
        restore event, or seed downstream callbacks with the resumed state.
        """

    def on_evaluate(self, state: "TrainerState", *, metrics: dict = None,
                    **kwargs) -> None:
        """Called when an evaluation pass completes.

        Currently triggered as a stub from ``EvalCallback``; once a real
        eval loop lands the callback will pass back the eval ``metrics``
        dict for sinks (TensorBoard / W&B) to log.
        """

class LoggingCallback(Callback):
    """Log training metrics: loss, grad_norm, lr, throughput.

    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        log_cfg = getattr(trainer.args, 'logging', None)
        self.log_steps = getattr(log_cfg, 'log_steps', 10) if log_cfg else 10
        self.report_global_loss = (
            getattr(log_cfg, 'report_global_loss', False) if log_cfg else False
        )
        self.report_throughput = (
            getattr(log_cfg, 'report_throughput', True) if log_cfg else True
        )
        self.model_flops_per_token = (
            getattr(log_cfg, 'model_flops_per_token', None) if log_cfg else None
        )
        self.peak_tflops = (
            getattr(log_cfg, 'peak_tflops', None) if log_cfg else None
        )
        # Estimate per-step tokens as upper bound (batch × seq_len).  Real
        # token count is available per step via ``last_global_tokens`` that
        # ``BaseTrainer.train_step`` stashes onto the trainer.
        gbs = getattr(trainer.args.train, 'global_batch_size', 1)
        seq_len = getattr(trainer.args.data, 'max_seq_len', 1)
        self._tokens_per_step_est = int(gbs) * int(seq_len)
        self._step_start_time = 0.0

    def on_step_begin(self, state: "TrainerState", **kwargs) -> None:
        self._step_start_time = time.time()

    def on_step_end(self, state: "TrainerState", *, loss: float = None,
                    grad_norm: float = None, **kwargs) -> None:
        if state.global_step % self.log_steps != 0:
            return

        elapsed = max(time.time() - self._step_start_time, 1e-9)
        lr = 0.0
        if self.trainer.lr_scheduler is not None:
            lr = self.trainer.lr_scheduler.get_last_lr()[0]

        metrics = {
            "step": state.global_step,
            # 8-decimal precision keeps fp32 sub-bf16 differences visible
            # in the log for sanity comparisons across runs.
            "loss": f"{loss:.8f}" if loss is not None else "N/A",
            "grad_norm": (
                f"{grad_norm:.8f}" if grad_norm is not None else "N/A"
            ),
            "lr": f"{lr:.2e}",
            "step_time": f"{elapsed:.2f}s",
        }

        tokens_per_sec = None
        if self.report_throughput:
            # Prefer real per-step token count stashed by train_step.
            tokens = getattr(self.trainer, '_last_global_tokens',
                             self._tokens_per_step_est)
            tokens_per_sec = tokens / elapsed
            metrics["tokens_per_sec"] = f"{tokens_per_sec:,.0f}"

            if self.model_flops_per_token and self.peak_tflops:
                # Observed TFLOPS = tokens/sec × flops/token / 1e12.
                # MFU = observed / (peak × world_size).
                world = max(platform.get_world_size(), 1)
                observed_tflops = (
                    tokens_per_sec * self.model_flops_per_token / 1e12
                )
                mfu = observed_tflops / (self.peak_tflops * world)
                metrics["tflops"] = f"{observed_tflops:.1f}"
                metrics["mfu"] = f"{mfu * 100:.1f}%"

        # Include aux_loss from MoEMonitorCallback when available.
        moe_cb = getattr(self.trainer, 'moe_monitor_callback', None)
        aux_loss = getattr(moe_cb, 'last_mean_aux_loss', None) if moe_cb is not None else None
        if aux_loss is not None:
            metrics["aux_loss"] = f"{aux_loss:.6f}"

        logger.info_rank0(" | ".join(f"{k}={v}" for k, v in metrics.items()))

        record = {
            "step": state.global_step,
            "loss": loss,
            "grad_norm": grad_norm,
            "lr": lr,
            "step_time": elapsed,
            "tokens_per_sec": tokens_per_sec,
            "aux_loss": aux_loss,
        }
        state.log_history.append(record)

        # Fan-out to other log-event listeners (TB / W&B / sinks).
        dispatch = getattr(self.trainer, "dispatch_log_event", None)
        if dispatch is not None:
            dispatch(record)

class CheckpointCallback(Callback):
    """Save distributed checkpoints and handle resume.

    Uses hyper's own DCP ``save`` / ``load`` APIs.
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        ckpt_cfg = getattr(trainer.args, 'checkpoint', None)
        self.save_steps = getattr(ckpt_cfg, 'save_steps', 0) if ckpt_cfg else 0
        self.output_dir = (
            getattr(ckpt_cfg, 'output_dir', 'outputs') if ckpt_cfg else 'outputs'
        )
        self.load_path = (
            getattr(ckpt_cfg, 'load_path', None) if ckpt_cfg else None
        )
        self.save_async = (
            getattr(ckpt_cfg, 'save_async', False) if ckpt_cfg else False
        )
        self._last_saved_step = -1
        self._save_thread = None   # async save worker

    def on_train_begin(self, state: "TrainerState", **kwargs) -> None:
        """Resume from checkpoint: model + optimizer + lr_scheduler + step + RNG.

        RFC DoD: "Save → resume → 续训 loss 一致（含 dataloader + RNG 恢复）"
        """
        if not self.load_path:
            return
        try:
            # pylint: disable=C0415
            # Non-model artifacts (optimizer/scheduler/RNG) are plain dicts —
            # use torch.save/load, matching the save side.

            if not os.path.isdir(self.load_path):
                logger.warning("Checkpoint path not found: %s", self.load_path)
                return

            # 1. Restore model via hyper DCP
            model_sd = self.trainer.model.state_dict()
            dcp_load(model_sd, checkpoint_id=self.load_path, use_collectives=False)
            self.trainer.model.load_state_dict(model_sd)
            logger.info("Model restored from %s", self.load_path)

            # 2. Restore extra state (step, epoch)
            extra_path = os.path.join(self.load_path, "extra_state.json")
            if os.path.isfile(extra_path):
                with open(extra_path, encoding="utf-8") as f:
                    extra = json.load(f)
                state.global_step = extra.get("global_step", 0)
                state.epoch = extra.get("epoch", 0)
                logger.info("Resumed at step=%d, epoch=%d",
                            state.global_step, state.epoch)

            # 3. Restore optimizer
            optim_path = os.path.join(self.load_path, f"optimizer_rank{platform.get_rank()}.pt")
            if os.path.isfile(optim_path) and self.trainer.optimizer:
                optim_sd = torch.load(optim_path, map_location="cpu", weights_only=True)
                self.trainer.optimizer.load_state_dict(optim_sd)
                logger.info("Optimizer restored")

            # 4. Restore LR scheduler
            sched_path = os.path.join(self.load_path, "scheduler.pt")
            if os.path.isfile(sched_path) and self.trainer.lr_scheduler:
                sched_sd = torch.load(sched_path, map_location="cpu", weights_only=True)
                self.trainer.lr_scheduler.load_state_dict(sched_sd)
                logger.info("LR scheduler restored")

            # 5. Restore RNG state
            rng_path = os.path.join(self.load_path, f"rng_rank{platform.get_rank()}.pt")
            if os.path.isfile(rng_path):
                rng_state = torch.load(rng_path, map_location="cpu", weights_only=True)
                platform.set_rng_state(rng_state)
                logger.info("RNG state restored")

            # 6. Restore dataloader position (StatefulDataLoader)
            dl_path = os.path.join(self.load_path, f"dataloader_rank{platform.get_rank()}.pt")
            if os.path.isfile(dl_path) and hasattr(self.trainer, 'train_dataloader'):
                dl_state = torch.load(dl_path, map_location="cpu", weights_only=False)
                self.trainer.train_dataloader.load_state_dict(dl_state)
                logger.info("Dataloader state restored")

            # Fan-out the load event so other callbacks (TensorBoard /
            # W&B / external trackers) can record the resume.
            dispatch = getattr(self.trainer, "dispatch_load_event", None)
            if dispatch is not None:
                dispatch(self.load_path)

        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Failed to load checkpoint from %s: %s", self.load_path, exc)

    def on_step_end(self, state: "TrainerState", *, loss: float = None,
                    grad_norm: float = None, **kwargs) -> None:
        if self.save_steps <= 0:
            return
        if state.global_step % self.save_steps != 0:
            return
        if state.global_step == self._last_saved_step:
            return
        self._dispatch_save(state)

    def on_train_end(self, state: "TrainerState", **kwargs) -> None:
        """Save final checkpoint (synchronously, to guarantee completion)."""
        # Wait for any outstanding async save first so the two don't race on
        # the same directory / state-dict iterator.
        self._join_pending()
        if self.save_steps > 0 and state.global_step != self._last_saved_step:
            # Final save always sync — the process is about to exit.
            self._save(state)

    # --- async plumbing -------------------------------------------------
    def _dispatch_save(self, state: "TrainerState") -> None:
        """Route to sync or async save based on ``save_async`` flag."""
        if not self.save_async:
            self._save(state)
            return
        # Wait for previous save to finish before starting a new one; saving
        # twice concurrently would double RAM and race the filesystem.
        self._join_pending()
        # pylint: disable=C0415
        # Snapshot state fields so the worker doesn't see later mutations.
        snap_step = state.global_step
        snap_epoch = state.epoch
        state_snapshot = copy.copy(state)
        state_snapshot.global_step = snap_step
        state_snapshot.epoch = snap_epoch
        self._save_thread = threading.Thread(
            target=self._save,
            args=(state_snapshot,),
            name=f"ckpt-save-step{snap_step}",
            daemon=True,
        )
        self._save_thread.start()
        logger.info_rank0(
            "Checkpoint save for step %d dispatched async (thread=%s)",
            snap_step, self._save_thread.name,
        )

    def _join_pending(self) -> None:
        """Block until any running async save finishes."""
        t = self._save_thread
        if t is not None and t.is_alive():
            logger.info_rank0(
                "Waiting for prior async ckpt save (%s)...", t.name,
            )
            t.join()
        self._save_thread = None

    def _save(self, state: "TrainerState") -> None:
        """Save complete training state: model + optimizer + scheduler + step + RNG.

        RFC DoD: "Save → resume → 续训 loss 一致（含 dataloader + RNG 恢复）"
        """
        # Optimizer/scheduler/RNG state dicts are plain Python dicts, not
        # nn.Module — platform.save_checkpoint expects Module (safetensors).
        # Use torch.save/load for these non-model artifacts.
        save_dir = os.path.join(self.output_dir, f"step_{state.global_step}")
        os.makedirs(save_dir, exist_ok=True)
        rank = platform.get_rank()

        try:
            # 1. Model — via hyper DCP (each rank saves its own shards)
            model_sd = self.trainer.model.state_dict()
            dcp_save(model_sd, checkpoint_id=save_dir, use_collectives=False)

            # 2. Optimizer — per-rank
            if self.trainer.optimizer:
                optim_path = os.path.join(save_dir, f"optimizer_rank{rank}.pt")
                torch.save(self.trainer.optimizer.state_dict(), optim_path)

            # 3. LR scheduler
            if self.trainer.lr_scheduler and rank == 0:
                sched_path = os.path.join(save_dir, "scheduler.pt")
                torch.save(self.trainer.lr_scheduler.state_dict(), sched_path)

            # 4. Extra state: global_step, epoch
            if rank == 0:
                extra = {
                    "global_step": state.global_step,
                    "epoch": state.epoch,
                }
                extra_path = os.path.join(save_dir, "extra_state.json")
                with open(extra_path, "w", encoding="utf-8") as f:
                    json.dump(extra, f)

            # 5. RNG state — per-rank via platform API
            rng_state = platform.get_rng_state()
            rng_path = os.path.join(save_dir, f"rng_rank{rank}.pt")
            torch.save(rng_state, rng_path)

            # 6. Dataloader position — per-rank (StatefulDataLoader)
            if hasattr(self.trainer, 'train_dataloader') and hasattr(
                self.trainer.train_dataloader, 'state_dict'
            ):
                dl_path = os.path.join(save_dir, f"dataloader_rank{rank}.pt")
                torch.save(self.trainer.train_dataloader.state_dict(), dl_path)

            self._last_saved_step = state.global_step
            logger.info_rank0("Checkpoint saved to %s", save_dir)

            # Fan-out the save event so other callbacks (W&B artifact
            # upload, remote-storage sync, downstream eval triggers) can
            # observe the new checkpoint without coupling to ckpt internals.
            dispatch = getattr(self.trainer, "dispatch_save_event", None)
            if dispatch is not None:
                dispatch(save_dir)

        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Failed to save checkpoint: %s", exc)

        # HF format export is handled by SafetensorsExportCallback (separate concern).

class SafetensorsExportCallback(Callback):
    """Export model weights in HuggingFace safetensor format.

    Separated from CheckpointCallback per RFC Section 5.2.
    Uses ``get_model_state_dict`` with ``full_state_dict=True`` to gather
    all FSDP shards into a full state dict before saving.

    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        ckpt_cfg = getattr(trainer.args, 'checkpoint', None)
        self.enabled = getattr(ckpt_cfg, 'save_hf_weights', False) if ckpt_cfg else False
        self.save_steps = getattr(ckpt_cfg, 'save_steps', 0) if ckpt_cfg else 0
        self.output_dir = getattr(ckpt_cfg, 'output_dir', 'outputs') if ckpt_cfg else 'outputs'
        self._last_saved_step = -1

    def on_step_end(self, state: "TrainerState", *, loss: Optional[float] = None,
                    grad_norm: Optional[float] = None, **kwargs) -> None:
        if not self.enabled or self.save_steps <= 0:
            return
        if state.global_step % self.save_steps != 0:
            return
        if state.global_step == self._last_saved_step:
            return
        self._export(state)

    def on_train_end(self, state: "TrainerState", **kwargs) -> None:
        if self.enabled and self.save_steps > 0 and state.global_step != self._last_saved_step:
            self._export(state)

    def _export(self, state: "TrainerState") -> None:
        """Gather full state dict from FSDP shards and save in HF format.

        Routes through ``spec.state_dict_adapter().save_hf_state_dict`` when
        the model's ``ModelSpec`` provides one, so per-model HF tensor
        renaming and per-expert packing live in the model package, not in
        this generic callback. Falls back to the legacy
        ``save_state_dict_as_huggingface_format`` path when the spec has no
        adapter (keeps ad-hoc / template models working).
        """
        # pylint: disable=C0415

        rank = platform.get_rank()
        save_dir = os.path.join(self.output_dir, f"step_{state.global_step}", "hf_ckpt")

        try:
            # ``StateDictOptions`` is a torch-backend type; hyper does not yet
            # provide a wrapper, so the trainer reaches into torch directly.
            # pylint: disable=C0415
            from torch.distributed.checkpoint.state_dict import StateDictOptions
            # full_state_dict=True gathers all FSDP shards; cpu_offload avoids OOM
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            full_sd = get_model_state_dict(self.trainer.model, options=options)

            if rank == 0:
                os.makedirs(save_dir, exist_ok=True)

                # Prefer the model-specific save adapter (closes the load/save
                # loop via the ModelSpec contract). When absent, fall back to
                # the generic offline-transform path.
                spec = getattr(self.trainer, "spec", None)
                adapter_cls = getattr(spec, "state_dict_adapter", None) if spec else None
                save_fn = (
                    getattr(adapter_cls(), "save_hf_state_dict", None)
                    if adapter_cls is not None else None
                )
                if save_fn is not None:
                    hf_sd = save_fn(full_sd, self.trainer.model.config)
                    from safetensors.torch import save_file  # pylint: disable=C0415
                    save_file(hf_sd, os.path.join(save_dir, "model.safetensors"))
                    logger.info(
                        "HF checkpoint saved via %s.save_hf_state_dict to %s",
                        adapter_cls.__name__, save_dir,
                    )
                else:
                    save_state_dict_as_huggingface_format(full_sd, save_dir)
                    logger.info(
                        "HF checkpoint saved (no adapter on spec) to %s", save_dir,
                    )

            self._last_saved_step = state.global_step

        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning_rank0("Failed to save HF checkpoint: %s", exc)

class EvalCallback(Callback):
    """Evaluation callback stub.

    Full evaluation is not yet implemented. This stub logs a warning whenever
    an evaluation trigger is received so the absence of eval is visible in
    training logs rather than silently skipped.
    """

    def on_step_end(self, state: "TrainerState", *, loss: Optional[float] = None,
                    grad_norm: Optional[float] = None, **kwargs) -> None:
        eval_cfg = getattr(self.trainer.args, 'eval', None)
        eval_steps = getattr(eval_cfg, 'eval_steps', 0) if eval_cfg else 0
        if eval_steps > 0 and state.global_step % eval_steps == 0:
            if platform.get_rank() == 0:
                logger.warning(
                    "EvalCallback: evaluation not implemented (step=%d)", state.global_step
                )

class ProfilerCallback(Callback):
    """Training profiler callback — STUB (not verified).

    Hook reserved for ``torch.profiler.profile`` integration. Not yet
    verified against the trainer; if you enable ``args.profiler.enabled``
    we emit a one-time warning so the absence of profiling traces is
    visible. To implement: wire ``torch.profiler.profile`` start/step/stop
    in ``on_train_begin`` / ``on_step_end`` / ``on_train_end``.
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        prof_cfg = getattr(trainer.args, 'profiler', None)
        if getattr(prof_cfg, 'enabled', False) and platform.get_rank() == 0:
            logger.warning(
                "ProfilerCallback: enabled=True but the implementation is "
                "a stub — torch profiler is NOT started. Implement before "
                "relying on traces."
            )

class WandbCallback(Callback):
    """Weights & Biases logging callback — STUB (not verified).

    Hook reserved for W&B integration. Not yet verified; if you enable
    ``args.wandb.enabled`` we emit a one-time warning so missing W&B logs
    are visible. To implement: wire ``wandb.init`` / ``wandb.log`` /
    ``wandb.finish`` in ``on_train_begin`` / ``on_step_end`` /
    ``on_train_end`` and verify against a real W&B run.
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        wandb_cfg = getattr(trainer.args, 'wandb', None)
        if getattr(wandb_cfg, 'enabled', False) and platform.get_rank() == 0:
            logger.warning(
                "WandbCallback: enabled=True but the implementation is a "
                "stub — nothing is sent to W&B. Implement before relying on "
                "W&B dashboards."
            )

class ProgressCallback(Callback):
    """tqdm progress bar callback (rank 0 only).

    Displays a progress bar over training steps with live loss and grad_norm
    metrics.  Requires ``tqdm``; degrades gracefully if not installed.
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        self._pbar = None

    def on_train_begin(self, state: "TrainerState", **kwargs) -> None:
        if platform.get_rank() != 0:
            return
        try:
            # pylint: disable=C0415
            from tqdm import tqdm  # pylint: disable=C0415  # optional dep
            self._pbar = tqdm(
                total=state.max_steps,
                initial=state.global_step,
                desc="Training",
                unit="step",
                dynamic_ncols=True,
            )
        except ImportError:
            logger.warning("ProgressCallback: 'tqdm' not installed — progress bar disabled")

    def on_step_end(self, state: "TrainerState", *, loss: Optional[float] = None,
                    grad_norm: Optional[float] = None, **kwargs) -> None:
        if self._pbar is None:
            return
        postfix = {}
        if loss is not None:
            postfix["loss"] = f"{loss:.4f}"
        if grad_norm is not None:
            postfix["gnorm"] = f"{grad_norm:.4f}"
        self._pbar.set_postfix(postfix)
        self._pbar.update(1)

    def on_train_end(self, state: "TrainerState", **kwargs) -> None:
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

class MoEMonitorCallback(Callback):
    """Mixture-of-Experts load-balancing monitor.

    Delegates to :class:`~hyper_parallel.core.moe_utils.MoEMonitorCallback`
    for expert bias updates and aux_loss aggregation.  Exposes
    ``last_mean_aux_loss`` so that :class:`LoggingCallback` can include it
    in the main training loss log line.

    Config: ``cfg.train.moe_monitor.*`` (see :class:`MoEMonitorConfig`).
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        """Initialize MoEMonitorCallback from trainer config."""
        super().__init__(trainer)
        moe_cfg = getattr(trainer.args, 'moe_monitor', None)
        self.enabled = getattr(moe_cfg, 'enabled', False) if moe_cfg else False
        self._impl = None

        if self.enabled:
            from hyper_parallel.core.moe_utils import (  # pylint: disable=C0415
                MoEMonitorCallback as _CoreMoEMonitorCallback,
            )
            from hyper_parallel.core.fully_shard.hsdp_utils import (  # pylint: disable=C0415
                GroupInfo,
            )
            lr = getattr(moe_cfg, 'lr', 1e-3)
            num_recomputations = getattr(moe_cfg, 'num_recomputations', 1)

            # Resolve DP/TP/CP groups from trainer's device mesh.
            dp_group = getattr(self.trainer, '_dp_group_info', None)
            tp_group = None
            cp_group = None
            mesh = getattr(self.trainer, 'mesh', None)
            if mesh is not None:
                for name, attr_name in [("tp", "tp_group"), ("cp", "cp_group")]:
                    try:
                        raw_group = mesh.get_group(name)
                        group_info = GroupInfo(
                            group_name=name, group=raw_group,
                            rank_size=raw_group.size(),
                        )
                        if attr_name == "tp_group":
                            tp_group = group_info
                        else:
                            cp_group = group_info
                    except (KeyError, ValueError, AttributeError):
                        pass

            self._impl = _CoreMoEMonitorCallback(
                model=self.trainer.model,
                lr=lr,
                dp_group=dp_group,
                tp_group=tp_group,
                cp_group=cp_group,
                num_recomputations=num_recomputations,
            )

    @property
    def last_mean_aux_loss(self) -> Optional[float]:
        """Mean aux_loss across MoE layers from the last ``on_step_end``."""
        if self._impl is not None:
            return self._impl.last_mean_aux_loss
        return None

    def on_train_begin(self, state: "TrainerState", **kwargs) -> None:
        """Log one-time confirmation when MoE monitoring is enabled."""
        if self.enabled and platform.get_rank() == 0:
            logger.info("MoEMonitorCallback: MoE expert-load monitoring enabled")

    def on_step_end(self, state: "TrainerState", *, loss: float = None,
                    grad_norm: float = None, **kwargs) -> None:
        """Delegate expert bias update to core MoEMonitorCallback."""
        if self._impl is not None:
            self._impl.on_step_end()

    def on_substep_end(self, state: "TrainerState", **kwargs) -> None:
        """No-op; expert bias updates happen in on_step_end."""

class GradientHealthCallback(Callback):
    """Detect NaN / Inf grad_norm and raise / warn.

    Hooks ``on_pre_optimizer_step`` — which fires after ``clip_grad_norm_``
    and before ``optimizer.step()``. ``grad_norm`` at that point is a plain
    scalar produced by hyper's DTensor-aware clipper. If it's not finite, the
    optimizer.step() would silently corrupt weights with NaN; we want to
    surface it immediately.

    Config: ``cfg.train.debug.check_nan_inf``.
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        debug_cfg = getattr(trainer.args, 'debug', None)
        self.enabled = (
            getattr(debug_cfg, 'check_nan_inf', False) if debug_cfg else False
        )

    def on_pre_optimizer_step(self, state: "TrainerState", *,
                              grad_norm: Optional[float] = None,
                              **kwargs) -> None:
        if not self.enabled or grad_norm is None:
            return
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            # Always log on every rank — divergence may be rank-local.
            logger.error(
                "GradientHealthCallback: grad_norm=%s at step %d "
                "(NaN/Inf). Optimizer.step would corrupt weights.",
                grad_norm, state.global_step,
            )
            # Raise on rank 0 only; other ranks will be torn down by NCCL.
            if platform.get_rank() == 0:
                raise RuntimeError(
                    f"Non-finite grad_norm={grad_norm} at "
                    f"step {state.global_step}. "
                    "Disable cfg.train.debug.check_nan_inf to skip this guard."
                )

class GCCallback(Callback):
    """Explicit garbage-collection scheduler.

    Python's cyclic GC can stall large training jobs when it decides to run;
    forcing a collection every N steps — outside the compute hot path —
    keeps pauses predictable.).

    Config: ``cfg.train.debug.gc_steps`` (``0`` disables).
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        debug_cfg = getattr(trainer.args, 'debug', None)
        self.gc_steps = (
            getattr(debug_cfg, 'gc_steps', 0) if debug_cfg else 0
        )
        if self.gc_steps > 0:
            # Disable the automatic generational collector; we'll drive it.
            gc.disable()
            logger.info("GCCallback: Python gc.collect every %d steps "
                        "(auto GC disabled)", self.gc_steps)

    def on_step_end(self, state: "TrainerState", *,
                    loss: Optional[float] = None,
                    grad_norm: Optional[float] = None, **kwargs) -> None:
        if self.gc_steps <= 0:
            return
        if state.global_step % self.gc_steps != 0:
            return
        gc.collect()

class TensorBoardCallback(Callback):
    """TensorBoard scalar writer — STUB (not verified).

    Hook reserved for ``torch.utils.tensorboard.SummaryWriter`` integration.
    Not yet verified; if you enable ``args.tensorboard.enabled`` we emit
    a one-time warning so missing TB scalars are visible. To implement:
    open SummaryWriter in ``on_train_begin``, write scalars in ``on_log``,
    close in ``on_train_end``.
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        tb_cfg = getattr(trainer.args, 'tensorboard', None)
        if getattr(tb_cfg, 'enabled', False) and platform.get_rank() == 0:
            logger.warning(
                "TensorBoardCallback: enabled=True but the implementation "
                "is a stub — nothing is written to TensorBoard. Implement "
                "before relying on TB scalars."
            )

class MemoryMonitorCallback(Callback):
    """Peak / current device memory monitor — STUB (not verified).

    Hook reserved for ``platform.get_device_handle().memory_allocated`` /
    ``max_memory_allocated`` polling. Not yet verified; if you enable
    ``args.memory_monitor.enabled`` we emit a one-time warning so missing
    memory logs are visible. To implement: poll the device handle in
    ``on_step_end`` gated by ``log_steps`` and log
    ``cur=...GB peak=...GB``.
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        cfg = getattr(trainer.args, 'memory_monitor', None)
        if getattr(cfg, 'enabled', False) and platform.get_rank() == 0:
            logger.warning(
                "MemoryMonitorCallback: enabled=True but the implementation "
                "is a stub — no memory stats are emitted. Implement before "
                "relying on these logs."
            )
