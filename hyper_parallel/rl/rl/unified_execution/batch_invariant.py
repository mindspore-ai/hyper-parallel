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
"""Process-wide TorchNPU Level 3 policy shared by training and inference."""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import logging
import os
import sys
from typing import Any

import torch
import torch.utils.deterministic


_POLICY = {
    "provider": "torch_npu_level3",
    "fill_uninitialized_memory": False,
    "level": 3,
    "torch_version": "2.10.0",
    "torch_npu_version": "2.10.0.post7",
}
_PROCESS_STATE: dict[str, Any] | None = None
_LOGGER = logging.getLogger(__name__)


def normalize_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a policy without importing or initializing the optional NPU backend.

    Args:
        policy: User fields or an already normalized, versioned worker policy.

    Returns:
        The requested fill policy and fixed supported provider/version identity.

    Raises:
        ValueError: The policy requests an unsupported field, provider, fill or version.
    """
    if not isinstance(policy, Mapping):
        raise ValueError("batch_invariant policy must be a mapping")
    unknown = set(policy) - set(_POLICY)
    if unknown:
        raise ValueError(f"Unknown batch_invariant policy fields: {sorted(unknown, key=str)}")
    normalized = dict(_POLICY)
    for name, value in policy.items():
        expected = _POLICY[name]
        if name == "fill_uninitialized_memory" and isinstance(value, bool):
            normalized[name] = value
            continue
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"batch_invariant {name} must be {expected!r}; got {value!r}")
    return normalized


def _check_conflicts() -> None:
    if os.environ.get("VLLM_BATCH_INVARIANT") == "1":
        raise RuntimeError("TorchNPU Level 3 conflicts with VLLM_BATCH_INVARIANT=1")
    for name in ("vllm_ascend", "vllm_ascend.batch_invariant"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "_batch_invariant_LIB", None) is not None:
            raise RuntimeError("TorchNPU Level 3 conflicts with installed vLLM Ascend batch-invariant aten patches")


def _read_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    backend = state["backend"]
    # The Python getter may synchronize by writing the level. This pinned C getter is read-only.
    level = backend._C._npu_get_deterministic_level()  # pylint: disable=protected-access
    return {
        "provider": state["policy"]["provider"],
        "level": int(level),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
        "torch_version": state["torch_version"],
        "torch_npu_version": state["torch_npu_version"],
        "pid": state["pid"],
        "device": f"npu:{backend.npu.current_device()}",
    }


def _verify_state(state: dict[str, Any]) -> dict[str, Any]:
    _check_conflicts()
    snapshot = _read_snapshot(state)
    expected = {"level": 3, "deterministic_algorithms": True,
                "fill_uninitialized_memory": state["policy"]["fill_uninitialized_memory"], "device": state["device"]}
    drift = {name: (value, snapshot[name]) for name, value in expected.items() if snapshot[name] != value}
    if drift:
        raise RuntimeError(f"batch_invariant process state drift (expected, actual): {drift}")
    return snapshot


def initialize_batch_invariant(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Install Level 3 once after selecting the worker's NPU device.

    Args:
        policy: The shared requested policy, normalized without optional dependencies.

    Returns:
        Actual process state after a native GEMM forces CANN option application.

    Raises:
        RuntimeError: Version, device, patch conflicts or previously installed state disagree.

    Note:
        The native setter checks CANN runtime/operator support. Its errors propagate.
        The startup GEMM checks option application, not numerical batch invariance.
    """
    global _PROCESS_STATE  # pylint: disable=global-statement
    normalized = normalize_policy(policy)
    _check_conflicts()
    pid = os.getpid()
    if _PROCESS_STATE is not None and _PROCESS_STATE["pid"] == pid:
        return verify_batch_invariant(normalized)

    # TorchNPU is optional in configuration-only and CPU processes.
    backend = importlib.import_module("torch_npu")
    versions = {"torch_version": str(torch.__version__), "torch_npu_version": str(backend.__version__)}
    for name, version in versions.items():
        if version.split("+", 1)[0] != normalized[name]:
            raise RuntimeError(f"batch_invariant requires {name}={normalized[name]}, got {version}")
    if not backend.npu.is_available() or not backend.npu.is_initialized():
        raise RuntimeError("batch_invariant requires an available NPU device selected before initialization")
    device = f"npu:{backend.npu.current_device()}"
    _PROCESS_STATE = {"pid": pid, "policy": normalized, "backend": backend,
                      "device": device, "ready": False, **versions}
    backend.npu.set_deterministic_level(3)
    torch.utils.deterministic.fill_uninitialized_memory = normalized["fill_uninitialized_memory"]
    with torch.no_grad():
        left = torch.ones((1, 16), dtype=torch.bfloat16, device=device)
        right = torch.ones((16, 16), dtype=torch.bfloat16, device=device)
        torch.mm(left, right)
    backend.npu.synchronize()
    snapshot = _verify_state(_PROCESS_STATE)
    _PROCESS_STATE["ready"] = True
    _LOGGER.info("Installed batch_invariant policy=%s actual_state=%s", normalized, snapshot)
    return snapshot


def verify_batch_invariant(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Read the current process policy and reject drift without modifying any state.

    Args:
        policy: The shared policy expected by this model or lifecycle boundary.

    Returns:
        Actual Level, Torch deterministic flag, fill, versions, PID and device.

    Raises:
        RuntimeError: Installation is missing, incomplete or no longer matches the policy.
    """
    normalized = normalize_policy(policy)
    state = _PROCESS_STATE
    if state is None or state["pid"] != os.getpid() or not state["ready"]:
        raise RuntimeError("batch_invariant policy has not completed initialization in this process")
    if normalized != state["policy"]:
        raise RuntimeError("batch_invariant policy conflicts with the policy already installed in this process")
    return _verify_state(state)
