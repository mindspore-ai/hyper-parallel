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
"""Shared HP optimizer construction and RL update execution."""
from typing import Any
import torch
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel import SkipDTensorDispatch, hsdp_sync_stream
from hyper_parallel.core.utils import clip_grad_norm_
from hyper_parallel.components.optim.mixed_precision_optimizer import MixedPrecisionOptimizer


def build_role_optimizer(runtime_config: object, model: torch.nn.Module) -> tuple[object, object]:
    """Build an independent optimizer and scheduler for one trainable role."""
    optimizer = runtime_config.optimizer.target.build(model=model).get_optimizer()
    if getattr(runtime_config.optimizer, "fp32_main_params", False):
        optimizer = Float16OptimizerWithFloat16Params(optimizer, model)
    lr_scheduler = None
    if runtime_config.lr_scheduler is not None:
        lr_scheduler = runtime_config.lr_scheduler.build(
            optimizer=optimizer,
            train_iters=runtime_config.training.train_iters,
        ).get_lr_scheduler()
    return optimizer, lr_scheduler


def optimizer_step(model: torch.nn.Module, optimizer: Any, max_grad_norm: float) -> float:
    """Synchronize streams, clip gradients, and update Actor parameters."""
    hsdp_sync_stream()
    main_optimizer = isinstance(optimizer, MixedPrecisionOptimizer)
    if main_optimizer:
        optimizer.prepare_grads()
    parameters = (
        (parameter for group in optimizer.param_groups for parameter in group["params"])
        if main_optimizer else model.parameters()
    )
    norm_limit = max_grad_norm if max_grad_norm > 0 else float("inf")
    grad_norm = clip_grad_norm_(
        parameters,
        norm_limit,
        error_if_nonfinite=True,
    )
    with SkipDTensorDispatch():
        if main_optimizer:
            optimizer.step_with_ready_grads()
        else:
            optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(grad_norm.item())
