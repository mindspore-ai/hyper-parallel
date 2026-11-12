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
"""Torch-facing descriptors around native Hyper FSDP, not a second scheduler."""

from inspect import signature
from typing import Any

import torch
from torch.distributed.fsdp import CPUOffloadPolicy as TorchCPUOffloadPolicy
from torch.distributed.fsdp import OffloadPolicy as TorchOffloadPolicy
from torch.distributed.tensor import DTensor as TorchDTensor
from torch.utils._pytree import tree_map

from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.core.dtensor.dtensor import DTensor as HyperDTensor
from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.core.fully_shard.api import fully_shard as hyper_fully_shard
from hyper_parallel.core.fully_shard.state_dict_utils import (
    get_model_state_dict as hyper_model_state_dict,
)
from hyper_parallel.core.fully_shard.utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from hyper_parallel.integration.torch.dtensor import (
    from_torch_view,
    to_hyper_mesh,
    to_hyper_placement,
    to_torch_view,
)


def fully_shard(module: torch.nn.Module, **kwargs: Any) -> torch.nn.Module:
    """Translate Torch descriptors and execute Hyper's existing fully_shard.

    Ordinary parameters become Hyper DTensors. Existing Hyper parameters keep
    native semantics. This entry does not convert pre-sharded Torch parameters
    or choose wrapping, prefetch, reduction intervals or root lifetimes.
    """
    signature(hyper_fully_shard).bind(module, **kwargs)
    if not isinstance(kwargs.get("reshard_after_forward", True), bool):
        raise NotImplementedError(
            "The Hyper adapter supports boolean reshard_after_forward only."
        )
    ignored_params = kwargs.get("ignored_params") or set()
    if any(
        isinstance(parameter, TorchDTensor)
        for parameter in module.parameters()
        if parameter not in ignored_params
    ):
        raise NotImplementedError(
            "Pre-sharded Torch TP/EP inputs need a dedicated layout adapter."
        )
    if kwargs.get("mesh") is not None:
        kwargs["mesh"] = to_hyper_mesh(kwargs["mesh"])
    policy = kwargs.get("mp_policy")
    if policy is not None and not isinstance(policy, MixedPrecisionPolicy):
        kwargs["mp_policy"] = MixedPrecisionPolicy(
            **{
                key: getattr(policy, key)
                for key in (
                    "param_dtype",
                    "reduce_dtype",
                    "output_dtype",
                    "cast_forward_inputs",
                )
            }
        )
    offload = kwargs.get("offload_policy")
    if offload is not None and not isinstance(offload, OffloadPolicy):
        converters = {
            TorchOffloadPolicy: lambda policy: OffloadPolicy(),
            TorchCPUOffloadPolicy: lambda policy: CPUOffloadPolicy(
                pin_memory=policy.pin_memory
            ),
        }
        convert = converters.get(type(offload))
        if convert is None:
            raise NotImplementedError(
                f"Unsupported Torch offload policy: {type(offload).__name__}."
            )
        kwargs["offload_policy"] = convert(offload)
    placement = kwargs.get("shard_placement_fn")
    if placement is not None:
        kwargs["shard_placement_fn"] = lambda parameter: to_hyper_placement(
            placement(parameter)
        )
    return hyper_fully_shard(module, **kwargs)


def _reshard_for_checkpoint(model: torch.nn.Module) -> None:
    """Restore native parameter identities at an idle checkpoint boundary."""
    for module in model.modules():
        if isinstance(module, HSDPModule):
            module.reshard()


def _check_checkpoint_options(options: Any) -> None:
    """Reject unsupported transport/name semantics before resharding the model."""
    if getattr(options, "broadcast_from_rank0", False):
        raise NotImplementedError(
            "The adapter supports sharded DCP restore, not broadcast_from_rank0."
        )
    if getattr(options, "dsd_fqn_modifiers", "_fqn_modifiers") != "_fqn_modifiers":
        raise NotImplementedError(
            "Custom dsd_fqn_modifiers are not supported by the checkpoint adapter."
        )


def get_model_state_dict(model: torch.nn.Module, *, options: Any = None) -> dict:
    """Expose Hyper model state through detached, shared-storage Torch views.

    These are checkpoint transport views, not replacements for model parameters.
    The caller owns checkpoint synchronization and the views' storage lifetime.
    """
    _check_checkpoint_options(options)
    _reshard_for_checkpoint(model)
    return tree_map(to_torch_view, hyper_model_state_dict(model, options=options))


def set_model_state_dict(
    model: torch.nn.Module, model_state_dict: dict, *, options: Any = None
) -> Any:
    """Copy sharded checkpoint views into existing Hyper model parameters.

    Preserve parameter identity so previously constructed optimizers keep their
    references. Full-state gathering and rank-zero broadcast are not implemented
    by this checkpoint boundary.
    """
    _check_checkpoint_options(options)
    if getattr(options, "full_state_dict", False):
        raise NotImplementedError("The adapter restores sharded DCP state only.")
    _reshard_for_checkpoint(model)
    targets = model.state_dict()
    restored = {
        name: tree_map(
            lambda tensor, target=targets[name]: from_torch_view(
                tensor, reference=target
            ),
            value,
        )
        if isinstance(targets.get(name), HyperDTensor)
        else value
        for name, value in model_state_dict.items()
    }
    with torch.no_grad(), SkipDTensorDispatch():
        return model.load_state_dict(
            restored, strict=getattr(options, "strict", True), assign=False
        )
