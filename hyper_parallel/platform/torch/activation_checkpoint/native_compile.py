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
"""Adapters for using PyTorch-native selective checkpointing under compile."""

from functools import partial
from typing import Any, Callable, Tuple

from hyper_parallel.core.activation_checkpoint.activation_checkpoint import CheckpointPolicy


_SUPPORTED_POLICY_NAMES = (
    "MUST_SAVE",
    "PREFER_SAVE",
    "MUST_RECOMPUTE",
    "PREFER_RECOMPUTE",
)


def _to_torch_checkpoint_policy(policy: Any) -> Any:
    """Convert supported HyperParallel policies to the native Torch enum."""
    from torch.utils import checkpoint as torch_checkpoint  # pylint: disable=C0415

    torch_policy_cls = torch_checkpoint.CheckpointPolicy
    supported_native = {
        getattr(torch_policy_cls, name) for name in _SUPPORTED_POLICY_NAMES
    }
    if isinstance(policy, torch_policy_cls):
        if policy in supported_native:
            return policy
        raise ValueError(
            f"Torch checkpoint policy {policy.name} is not supported by "
            "HyperParallel compile mode. Only SAVE and RECOMPUTE policies are supported."
        )
    if isinstance(policy, CheckpointPolicy):
        if policy.name in _SUPPORTED_POLICY_NAMES:
            return getattr(torch_policy_cls, policy.name)
        raise ValueError(
            f"HyperParallel checkpoint policy {policy.name} is not supported in compile mode. "
            "Only SAVE and RECOMPUTE policies are supported."
        )
    raise TypeError(
        "Selective checkpoint policy_fn must return a HyperParallel or Torch "
        f"CheckpointPolicy, but got {type(policy).__name__}."
    )


def _torch_policy_adapter(
    policy_fn: Callable, torch_context: Any, op: Any, *args: Any, **kwargs: Any
) -> Any:
    """Pass native Torch inputs through and adapt only the policy result."""
    return _to_torch_checkpoint_policy(policy_fn(torch_context, op, *args, **kwargs))


def create_native_selective_checkpoint_contexts(policy_fn: Callable) -> Tuple[Any, Any]:
    """Create Torch-native selective-checkpoint contexts for compile capture."""
    if not callable(policy_fn):
        raise TypeError("policy_fn must be callable in HyperParallel compile mode.")
    from torch.utils import checkpoint as torch_checkpoint  # pylint: disable=C0415

    return torch_checkpoint.create_selective_checkpoint_contexts(
        partial(_torch_policy_adapter, policy_fn)
    )


__all__ = ["create_native_selective_checkpoint_contexts"]
