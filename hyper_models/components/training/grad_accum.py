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
"""Gradient accumulation helpers — following design doc 03_training_loop.md §7.1."""

import gc
import inspect
import logging
import os
import tempfile
from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)

# FSDP2 wrapper — torch >= 2.4
try:
    from torch.distributed.fsdp import FSDPModule
except ImportError:  # pragma: no cover
    FSDPModule = None  # type: ignore


def get_sync_ctx(
    model_parts: list[nn.Module],
    *,
    is_optim_step: bool,
    defer_fsdp_grad_sync: bool = False,
):
    """Return context manager for forward — FSDP2 grad sync via set_requires_gradient_sync.

    All branches return nullcontext() — the side effect is toggling FSDPModule sync.
    """
    if not is_optim_step:
        return nullcontext()
    if defer_fsdp_grad_sync:
        for mp in model_parts:
            if FSDPModule is not None and isinstance(mp, FSDPModule):
                mp.set_requires_gradient_sync(False)
        return nullcontext()
    # Last microbatch: sync already enabled by prepare_for_final_backward
    return nullcontext()


def prepare_for_grad_accumulation(model_parts: list[nn.Module]) -> None:
    """Pre-grad-accumulation: disable FSDP grad sync, mark deferred state."""
    for mp in model_parts:
        if FSDPModule is not None and isinstance(mp, FSDPModule):
            mp.set_requires_gradient_sync(False)
            mp._grad_accum_state = "deferred"


def prepare_for_final_backward(model_parts: list[nn.Module]) -> None:
    """Pre-final-backward: enable FSDP grad sync, attach PP hooks if multi-stage."""
    for mp in model_parts:
        if FSDPModule is not None and isinstance(mp, FSDPModule):
            mp.set_requires_gradient_sync(True)
            mp._grad_accum_state = "final"
    if len(model_parts) > 1:
        _attach_pp_backward_hooks(model_parts)


def _attach_pp_backward_hooks(model_parts: list[nn.Module]) -> None:
    """Attach PP backward send/recv hooks across stages.

    Stub — PP runtime not yet implemented. Path reserved at
    hyper_models/components/parallel/pp_utils.py.
    """
    raise NotImplementedError(
        "Pipeline parallelism backward hooks not yet implemented. "
        "PP > 1 is unavailable until pp_utils.py lands."
    )


def prepare_after_first_microbatch(model_parts: list[nn.Module]) -> None:
    """Post-first-microbatch: reset lazy init, mark first microbatch done."""
    for mp in model_parts:
        if FSDPModule is not None and isinstance(mp, FSDPModule):
            if hasattr(mp, "reset_lazy_init"):
                mp.reset_lazy_init()
            mp._first_microbatch_done = True


def set_requires_gradient_sync(
    model_parts: list[nn.Module], is_last: bool
) -> None:
    """Bulk set FSDP2 grad sync (middle microbatch off, last on)."""
    for mp in model_parts:
        if FSDPModule is not None and isinstance(mp, FSDPModule):
            mp.set_requires_gradient_sync(is_last)


def scale_grads_and_clip_grad_norm(
    model_parts: list[nn.Module],
    max_norm: float,
    num_label_tokens: Optional[int] = None,
) -> float:
    """Gradient scaling + clipping, return grad_norm.

    Steps:
    1. If num_label_tokens not None (non-PP): div each grad by num_label_tokens
       (token-mean normalization — the only division point).
    2. clip_grad_norm_(params, max_norm) → pre-clip grad_norm.
    3. Return grad_norm.
    """
    params = [p for mp in model_parts for p in mp.parameters() if p.grad is not None]
    if num_label_tokens is not None and num_label_tokens > 0:
        for p in params:
            if p.grad is not None:
                p.grad.detach_().div_(num_label_tokens)
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    return float(grad_norm)


def _dp_cp_all_reduce_sum(tensor, dp_cp_mesh) -> torch.Tensor:
    """All-reduce sum across DP+CP joint mesh.

    Accepts Python scalar (wrapped to tensor). Returns reduced tensor.
    """
    if not torch.is_tensor(tensor):
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        tensor = torch.tensor(tensor, device=device)
    if tensor.device.type != "cuda" and torch.cuda.is_available():
        tensor = tensor.cuda()
    if dp_cp_mesh is not None:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=dp_cp_mesh.get_group())
    elif dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _dp_all_reduce_avg(tensor, dp_mesh=None) -> torch.Tensor:
    """Pure-DP all-reduce mean (divide by dp_world_size). CP not involved."""
    if not torch.is_tensor(tensor):
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        tensor = torch.tensor(tensor, device=device)
    if tensor.device.type != "cuda" and torch.cuda.is_available():
        tensor = tensor.cuda()
    if dp_mesh is not None:
        group = dp_mesh.get_group()
        world_size = dp_mesh.size()
    else:
        group = dist.group.WORLD if dist.is_initialized() else None
        world_size = dist.get_world_size(group) if dist.is_initialized() else 1
    if group is not None:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    if world_size > 0:
        tensor.div_(world_size)
    return tensor


def calculate_mfu(
    tps: float,
    flops_per_token: float,
    peak_tflops: float,
    world_size: int,
) -> float:
    """Model FLOPs Utilization = (tps * flops_per_token) / (peak_tflops * world_size).

    Returns scalar in [0, 1].
    """
    total_peak_tflops = peak_tflops * world_size
    if total_peak_tflops <= 0:
        return 0.0
    mfu = (tps * flops_per_token) / (total_peak_tflops * 1e12)
    return min(max(mfu, 0.0), 1.0)


def filter_forward_kwargs(model: nn.Module, batch: dict) -> dict:
    """Filter batch keys to only those accepted by model.forward."""
    try:
        sig = inspect.signature(model.forward)
    except (ValueError, TypeError):
        return dict(batch)
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in batch.items() if k in accepted}


def calculate_mtp_loss(
    mtp_per_depth_logits: list[torch.Tensor],
    mtp_per_depth_h: list[torch.Tensor],
    labels: torch.Tensor,
    loss: nn.Module,
) -> torch.Tensor:
    """Multi-Token-Prediction auxiliary loss (Qwen3.5 etc.)."""
    total_mtp_loss = torch.tensor(0.0, device=labels.device, dtype=torch.float32)
    for depth_idx, logits in enumerate(zip(mtp_per_depth_logits, mtp_per_depth_h)):
        mtp_logits = logits[0]
        logits_shifted = mtp_logits[..., :-1, :].contiguous()
        labels_shifted = labels[..., 1:].contiguous()
        depth_loss = loss(
            logits_shifted.view(-1, logits_shifted.size(-1)),
            labels_shifted.view(-1),
        )
        total_mtp_loss = total_mtp_loss + depth_loss
    return total_mtp_loss


def setup_magi(cfg, device_mesh):
    """Build MagiAttention context (optional); None if not configured."""
    magi_cfg = getattr(cfg, "magi", None)
    if magi_cfg is None:
        return None
    try:
        from magi_attention import MagiAttentionContext
    except ImportError:
        logger.warning("magi_attention not installed; skipping MagiAttention setup")
        return None
    return MagiAttentionContext(
        device_mesh=device_mesh,
        **({} if isinstance(magi_cfg, bool) else dict(magi_cfg)),
    )


class AutoMFU:
    """MFU calculator: caches flops_per_token / peak_tflops."""

    def __init__(self, flops_per_token: float, peak_tflops: float):
        self.flops_per_token = flops_per_token
        self.peak_tflops = peak_tflops

    @classmethod
    def from_config(cls, model: nn.Module) -> "AutoMFU":
        """Infer flops_per_token from model config; peak_tflops from device."""
        num_params = sum(p.numel() for p in model.parameters())
        flops_per_token = 6.0 * num_params
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        peak_tflops = _infer_peak_tflops(device_name)
        return cls(flops_per_token=flops_per_token, peak_tflops=peak_tflops)


def _infer_peak_tflops(device_name: str) -> float:
    """Infer bf16 peak TFLOPS from GPU name (conservative)."""
    name_lower = device_name.lower()
    if "h100" in name_lower or "h800" in name_lower:
        return 989.0
    elif "a100" in name_lower or "a800" in name_lower:
        return 312.0
    elif "h20" in name_lower:
        return 148.0
    elif "v100" in name_lower:
        return 125.0
    elif "4090" in name_lower:
        return 330.0
    else:
        return 200.0


def _is_rank_0() -> bool:
    """True if global rank 0 (or distributed not initialized)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _update_latest_symlink(checkpoint_dir: str, path: str) -> None:
    """Atomically update {checkpoint_dir}/LATEST symlink to point to path.

    Writes relative path. Uses os.symlink + rename for atomicity.
    """
    latest = os.path.join(checkpoint_dir, "LATEST")
    rel_path = os.path.relpath(path, checkpoint_dir)

    tmp = os.path.join(checkpoint_dir, ".LATEST.tmp")
    if os.path.lexists(tmp):
        os.unlink(tmp)
    os.symlink(rel_path, tmp)
    os.rename(tmp, latest)