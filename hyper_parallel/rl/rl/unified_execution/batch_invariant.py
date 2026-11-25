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
"""Guard the shared TorchNPU policy against conflicting vLLM worker patches."""
from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any

from hyper_parallel.components.parallelism_invariant.utils import batch_invariant
from hyper_parallel.components.parallelism_invariant.utils.batch_invariant import normalize_policy

if TYPE_CHECKING:
    from torch import nn

__all__ = ["guard_training_numerics", "initialize_batch_invariant", "normalize_policy", "verify_batch_invariant"]


def _check_conflicts() -> None:
    if os.environ.get("VLLM_BATCH_INVARIANT") == "1":
        raise RuntimeError("TorchNPU Level 3 conflicts with VLLM_BATCH_INVARIANT=1")
    for name in ("vllm_ascend", "vllm_ascend.batch_invariant"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "_batch_invariant_LIB", None) is not None:
            raise RuntimeError("TorchNPU Level 3 conflicts with installed vLLM Ascend batch-invariant aten patches")


def initialize_batch_invariant(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Check the worker environment before applying the shared Level 3 policy."""
    _check_conflicts()
    return batch_invariant.initialize_batch_invariant(policy)


def verify_batch_invariant(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Check both worker patches and the current device policy at RL boundaries."""
    _check_conflicts()
    return batch_invariant.verify_batch_invariant(policy)


def _verify_training_numerics(unused_module: nn.Module, unused_inputs: tuple, *, policy: dict) -> None:
    """Fail before a training forward when another caller changes process numerics."""
    del unused_module, unused_inputs
    verify_batch_invariant(policy)


def guard_training_numerics(model: nn.Module, policy: dict) -> None:
    """Bind a read-only pre-forward check without changing parameters or HP wrappers."""
    normalized = normalize_policy(policy)
    verify_batch_invariant(normalized)
    previous = getattr(model, "unified_numeric_policy", None)
    if previous is not None:
        if previous != normalized:
            raise ValueError("Training model already owns a different numerical policy")
        return
    model.unified_numeric_policy = dict(normalized)
    model.register_forward_pre_hook(partial(_verify_training_numerics, policy=normalized))
