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

"""Trainer-side FSDP gradient-sync and reshard policies.

Split out of the former ``auto_models/trainer/base.py`` in stage 7
(05 §15.11 step 3). ``BaseTrainer`` keeps the same-named methods as thin
delegating subclass hooks; the policy itself lives here. The FSDP config is
duck-typed (``reshard_after_backward`` / ``dp_shard_size`` /
``requires_grad_sync`` attributes) so this module does not import Trainer DTOs.
"""

from typing import Any, List


def model_reshard(
    hsdp_model_parts: List[Any],
    fsdp_config: Any,
    micro_step: int,
    num_micro_steps: int,
) -> None:
    """Reshard model after backward pass."""
    if (
            fsdp_config.reshard_after_backward is False
            and num_micro_steps > 1
    ):
        if micro_step == 0:
            for model_part in hsdp_model_parts:
                model_part.set_reshard_after_backward(False)
        elif micro_step == num_micro_steps - 1:
            for model_part in hsdp_model_parts:
                model_part.set_reshard_after_backward(True)


def configure_fsdp_gradient_sync(
    hsdp_model_parts: List[Any],
    fsdp_config: Any,
    dp_replicate_size: int,
    micro_step: int,
    num_micro_steps: int,
) -> None:
    """Configure FSDP gradient synchronization for one micro step."""
    if (
            fsdp_config.dp_shard_size > 1
            and num_micro_steps > 1
    ):
        is_last_micro_batch = micro_step == num_micro_steps - 1
        requires_gradient_sync = (
            fsdp_config.requires_grad_sync
            or is_last_micro_batch
        )
        is_hsdp = dp_replicate_size > 1
        for model_part in hsdp_model_parts:
            model_part.set_requires_gradient_sync(requires_gradient_sync)
            model_part.set_is_last_backward(is_last_micro_batch)
            if is_hsdp:
                model_part.set_requires_all_reduce(is_last_micro_batch)


__all__ = [
    "configure_fsdp_gradient_sync",
    "model_reshard",
]
