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
# ============================================================================
"""Tensor staging for asynchronous distributed checkpoint save.

Staging materializes a host-side snapshot of ``state_dict`` so ``async_save`` can
run persistence (plan + I/O) on a copy while training continues on the original tensors.
"""
import copy
from typing import Any

from hyper_parallel.core.distributed_checkpoint.util import flatten_state_dict, set_element
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.platform import get_platform

# Used for ``isinstance(..., Tensor)`` on flattened leaves; matches the active backend tensor type.
platform = get_platform()
Tensor = platform.Tensor


def _tensor_to_staging_cpu(tensor: Any) -> Any:
    """Return a host-memory copy of a framework tensor, detached from autograd where applicable."""
    # ``to("cpu")`` is supported on both Torch and MindSpore tensor APIs used by HyperParallel.
    return tensor.detach().to("cpu")


def _stage_leaf(obj: Any) -> Any:
    """Stage a single flattened leaf for checkpoint serialization."""
    if isinstance(obj, DTensor):
        # Only the local shard is materialized; wrap again so planner metadata (mesh, placements) matches ``save``.
        local = obj.to_local()
        staged_local = _tensor_to_staging_cpu(local)
        return DTensor.from_local(
            staged_local,
            obj.device_mesh,
            obj.placements,
            shape=tuple(obj.shape),
        )
    if isinstance(obj, Tensor):
        return _tensor_to_staging_cpu(obj)
    if isinstance(obj, (bytes, bytearray)):
        # Avoid sharing mutable ``bytearray`` with the training process.
        return bytes(obj)
    # Optimizer state dict may hold plain Python containers; deep copy keeps staging independent.
    return copy.deepcopy(obj)


def build_staged_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Build a deep structural copy of ``state_dict`` with tensor / DTensor data
    copied to host memory so the original can be mutated while checkpoint I/O runs.

    Uses the same flattening rules as :class:`StandardSavePlanner`.

    Args:
        state_dict (dict[str, Any]): Nested or flat training state dict.

    Returns:
        dict[str, Any]: New dict with identical nesting and keys; tensor leaves are staging copies.
    """
    # Same dotted-key flattening as ``StandardSavePlanner.configure_planner`` so FQNs line up with save plans.
    flat, mappings = flatten_state_dict(state_dict)
    staged_flat = {fqn: _stage_leaf(obj) for fqn, obj in flat.items()}
    # Rebuild the original nested structure (dict / list) from FQN paths produced by ``flatten_state_dict``.
    root: dict[str, Any] = {}
    for fqn, path in mappings.items():
        set_element(root, path, staged_flat[fqn])
    return root
