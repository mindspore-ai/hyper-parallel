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

"""Generic expert-parallel preparation for the HyperParallel trainer."""

import logging
from dataclasses import dataclass
from typing import Any

# LlamaFactory is a PyTorch-only integration boundary.
# pylint: disable-next=forbidden-backend-import
from torch import nn

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, init_device_mesh
from hyper_parallel.core.expert_parallel.expert_parallel import ExpertParallel
from hyper_parallel.core.fully_shard.api import HSDPModule, fully_shard
from hyper_parallel.integration.llamafactory.expert_parallel.models import (
    get_expert_parallel_model_patches,
)
from hyper_parallel.integration.llamafactory.utils import (
    HyperParallelArguments,
    _collect_replicate_params,
    _resolve_device_type,
    _resolve_mp_policy,
    _resolve_offload_policy,
    _resolve_shard_size,
    get_parameters_from_modules,
)
from hyper_parallel.platform import get_platform

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ExpertParallelContext:
    """State shared between EP application and expert FSDP wrapping."""

    full_mesh: DeviceMesh
    expert_modules: list[nn.Module]
    ep_size: int


def _build_ep_mesh(hp_args: HyperParallelArguments) -> tuple[DeviceMesh, int, int]:
    """Build the EP x expert-FSDP mesh from validated arguments."""
    edp_size = get_platform().get_world_size() // hp_args.ep_size
    efsdp_size = hp_args.efsdp_size
    if efsdp_size is None:
        efsdp_size = edp_size
    ep_replicate_size = edp_size // efsdp_size

    full_mesh = init_device_mesh(
        _resolve_device_type(hp_args),
        (ep_replicate_size, efsdp_size, hp_args.ep_size),
        mesh_dim_names=("ep_replicate", "efsdp", "ep"),
    )
    return full_mesh, efsdp_size, ep_replicate_size


def _find_expert_modules(model: nn.Module) -> list[nn.Module]:
    """Find expert containers using framework-level naming conventions."""
    expert_modules: list[nn.Module] = []
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        if not module_name:
            continue

        leaf_name = module_name.rsplit(".", 1)[-1].lower()
        class_name = module.__class__.__name__.lower()
        if (
            leaf_name != "experts"
            and "experts" not in class_name
            and "grouped" not in class_name
        ):
            continue

        module_id = id(module)
        if module_id not in seen:
            seen.add(module_id)
            expert_modules.append(module)

    return expert_modules


def _apply_expert_parallel(
    model: nn.Module, hp_args: HyperParallelArguments
) -> _ExpertParallelContext:
    """Apply expert parallelism and return the resulting parallel context."""
    full_mesh, efsdp_size, ep_replicate_size = _build_ep_mesh(hp_args)
    ep_mesh = full_mesh["ep"]
    expert_modules = _find_expert_modules(model)
    if not expert_modules:
        raise ValueError(
            "No expert container was found. EP expects modules named `experts` "
            "or module class names containing `Experts`/`Grouped`."
        )

    for module in expert_modules:
        if getattr(module, "_hyper_parallel_ep_applied", False):
            raise RuntimeError(
                f"Expert parallelism has already been applied to {module.__class__.__name__}."
            )

        ExpertParallel(token_dispatcher=hp_args.token_dispatcher).apply(module, ep_mesh)
        module._hyper_parallel_ep_applied = True  # pylint: disable=protected-access

    logger.info(
        "Applied HyperParallel EP to %d expert container(s), ep_size=%d, efsdp_size=%d, "
        "ep_replicate_size=%d.",
        len(expert_modules),
        hp_args.ep_size,
        efsdp_size,
        ep_replicate_size,
    )
    return _ExpertParallelContext(
        full_mesh=full_mesh,
        expert_modules=expert_modules,
        ep_size=hp_args.ep_size,
    )


def _build_expert_fsdp2_kwargs(
    accelerator: Any,
    model: nn.Module,
    hp_args: HyperParallelArguments,
    fsdp2_plugin: Any,
    expert_mesh: DeviceMesh,
) -> dict[str, Any]:
    """Build fully_shard kwargs from accelerator and plugin settings."""
    reshard_after_forward = fsdp2_plugin.reshard_after_forward
    if hp_args.reshard_after_forward is not None:
        reshard_after_forward = hp_args.reshard_after_forward
    return {
        "reshard_after_forward": reshard_after_forward,
        "offload_policy": _resolve_offload_policy(fsdp2_plugin),
        "mp_policy": _resolve_mp_policy(fsdp2_plugin, hp_args),
        "mesh": expert_mesh,
        "ignored_params": get_parameters_from_modules(
            fsdp2_plugin.ignored_modules, model, accelerator.device
        ),
        "comm_fusion": False,
    }


def _wrap_expert_with_fsdp(
    module: nn.Module,
    expert_fsdp_kwargs: dict[str, Any],
    ep_size: int,
) -> None:
    """Apply one expert-scoped FSDP2 wrapper and its EP gradient scaling."""
    if not isinstance(module, HSDPModule):
        module_fsdp_kwargs = expert_fsdp_kwargs.copy()
        module_params = set(module.parameters())
        ignored_params = (
            set(module_fsdp_kwargs.get("ignored_params") or set()) & module_params
        )
        module_fsdp_kwargs["ignored_params"] = ignored_params
        replicate_params = _collect_replicate_params(
            module, _resolve_shard_size(module_fsdp_kwargs["mesh"])
        )
        replicate_params.difference_update(ignored_params)
        if replicate_params:
            module_fsdp_kwargs["replicate_params"] = replicate_params
        fully_shard(module, **module_fsdp_kwargs)
    module.set_gradient_scaling_factor(1.0 / ep_size)


def ep_prepare_model(
    model: nn.Module,
    accelerator: Any,
    hp_args: HyperParallelArguments,
) -> nn.Module:
    """Apply EP and FSDP2-wrap only the expert containers.

    The trainer calls the common ``fsdp2_prepare_model`` immediately after
    this function. Inner expert wrappers are excluded automatically from the
    parent/root FSDP2 units, so the common path manages only the remaining
    model parameters.
    """
    # Keep the pre-EP state dict for the common FSDP2 CPU-efficient loading
    # path. Expert wrapping replaces full parameters with rank-local DTensors,
    # while rank 0 still needs the original full tensors for distribution.
    model._hyper_parallel_pre_ep_state_dict = model.state_dict()  # pylint: disable=protected-access

    matched_patches = [
        patch for patch in get_expert_parallel_model_patches() if patch.supports(model)
    ]
    for patch in matched_patches:
        patch.prepare(model, hp_args)
    if matched_patches:
        logger.info(
            "Applied model-specific EP patch(es): %s.",
            ", ".join(patch.name for patch in matched_patches),
        )
    ep_context = _apply_expert_parallel(model, hp_args)
    expert_mesh = ep_context.full_mesh[("ep_replicate", "efsdp")]
    fsdp2_plugin = accelerator.state.fsdp_plugin
    expert_fsdp_kwargs = _build_expert_fsdp2_kwargs(
        accelerator, model, hp_args, fsdp2_plugin, expert_mesh
    )
    for module in ep_context.expert_modules:
        _wrap_expert_with_fsdp(module, expert_fsdp_kwargs, ep_context.ep_size)
    return model
