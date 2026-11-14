# Copyright 2025-2026 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Huawei Technologies Co., Ltd
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

"""Lightweight RNG snapshots for evaluation and checkpoint continuation."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.trainer.runtime.device import get_device_rng_state, set_device_rng_state


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU and local accelerator RNG for exact continuation.

    Returns:
        A weights-only-loadable mapping; NumPy state uses plain Python values.
    """
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_device": get_device_rng_state(),
        "python": random.getstate(),
        "numpy": capture_numpy_rng_state(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore available RNG domains, retaining compatibility with older checkpoints.

    Args:
        state: Mapping produced by capture_rng_state or a legacy checkpoint.
    """
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    set_device_rng_state(state.get("torch_device"))
    if state.get("python") is not None:
        random.setstate(state["python"])
    restore_numpy_rng_state(state.get("numpy"))


def capture_numpy_rng_state() -> dict[str, Any]:
    """Return the NumPy RNG state using weights-only-loadable Python values."""
    algorithm, keys, position, has_gauss, cached_gauss = np.random.get_state()
    return {
        "algorithm": algorithm,
        "keys": keys.tolist(),
        "position": position,
        "has_gauss": has_gauss,
        "cached_gauss": cached_gauss,
    }


def restore_numpy_rng_state(state: dict[str, Any] | None) -> None:
    """Restore NumPy when present; older checkpoints may omit this domain.

    Args:
        state: State from capture_numpy_rng_state, or None for a legacy save.
    """
    if state is not None:
        np.random.set_state((
            state["algorithm"],
            np.asarray(state["keys"], dtype=np.uint32),
            state["position"],
            state["has_gauss"],
            state["cached_gauss"],
        ))
