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
"""Core HyperParallel utilities for LlamaFactory integration."""

import copy
import functools
import json
import logging
import os
import re
import types
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)

from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.fully_shard.api import HSDPModule, fully_shard
from hyper_parallel.core.fully_shard.utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from hyper_parallel.integration.llamafactory.context_parallel.inputs import (
    _get_cp_dp_ranks,
    get_cp_group,
    get_cp_group_ranks,
    get_cp_rank,
    get_dp_rank,
    shard_inputs_for_cp,
)
from hyper_parallel.platform import get_platform

logger = logging.getLogger(__name__)

__all__ = [
    "HSDP_MODEL_NAME",
    "HSDP_OPTIMIZER_NAME",
    "HyperParallelArguments",
    "_build_device_mesh",
    "_build_fsdp2_kwargs",
    "_get_cp_dp_ranks",
    "_resolve_mp_policy",
    "fsdp2_prepare_model",
    "get_cp_group",
    "get_cp_group_ranks",
    "get_cp_rank",
    "get_dp_rank",
    "shard_inputs_for_cp",
    "export_to_hf_format",
    "load_hsdp_model",
    "load_hsdp_optimizer_and_scheduler",
    "patch_llamafactory_fp32_upcast",
    "save_hsdp_checkpoint",
    "wrap_optimizer_with_skip_dtensor_dispatch",
]


# LlamaFactory upcasts trainable params to fp32 on the host before FSDP wraps the
# model. FSDP2 re-upcasts them post-shard anyway, so the pre-shard upcast is
# redundant -- and with cpu_ram_efficient_loading it faults the whole model into
# host memory on every rank (host peak ~ N*M). We patch the two tuning-setup
# functions to skip it under FSDP.

_LLAMAFACTORY_UPCAST_PATCHED = False


def _should_defer_host_fp32_upcast(model) -> bool:
    """True when FSDP2 will re-upcast trainable params to fp32 after wrapping."""
    from transformers.integrations.fsdp import is_fsdp_enabled  # pylint: disable=C0415

    if not is_fsdp_enabled():
        return False
    return getattr(model, "dtype", None) in (torch.float16, torch.bfloat16)


def _skip_host_upcast_under_fsdp(orig, fn_name):
    """Wrap a LlamaFactory tuning-setup fn, forcing its fp32-upcast flag off under FSDP."""

    @functools.wraps(orig)
    def wrapper(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32):
        if cast_trainable_params_to_fp32 and _should_defer_host_fp32_upcast(model):
            cast_trainable_params_to_fp32 = False
            if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
                logger.info("[HP] FSDP detected; deferring %s fp32 upcast to post-shard.", fn_name)
        return orig(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)

    return wrapper


def patch_llamafactory_fp32_upcast() -> None:
    """Install the FSDP fp32-upcast deferral patch. Idempotent; no-op without LlamaFactory.

    Called from HyperParallelArguments.from_finetuning_args so the patch is in place
    before LlamaFactory's load_model runs.
    """
    global _LLAMAFACTORY_UPCAST_PATCHED
    if _LLAMAFACTORY_UPCAST_PATCHED:
        return
    try:
        from llamafactory.model import adapter as lf_adapter  # pylint: disable=C0415
    except ImportError:
        return

    for fn_name in ("_setup_full_tuning", "_setup_freeze_tuning"):
        orig = getattr(lf_adapter, fn_name, None)
        if orig is not None:
            setattr(lf_adapter, fn_name, _skip_host_upcast_under_fsdp(orig, fn_name))

    _LLAMAFACTORY_UPCAST_PATCHED = True

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

_VALID_DTYPES = {"float32", "float16", "bfloat16", "fp32", "fp16", "bf16"}
_VALID_TOKEN_DISPATCHERS = {"all_to_all", "deredundency"}

HSDP_MODEL_NAME = "hsdp_model"
HSDP_OPTIMIZER_NAME = "optimizer"


@dataclass
class HyperParallelArguments:
    """Minimal HyperParallel configuration needed by the trainer backend."""

    tp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1
    efsdp_size: Optional[int] = None
    token_dispatcher: str = "all_to_all"
    device_type: str = "auto"
    param_dtype: Optional[str] = None
    reduce_dtype: Optional[str] = None
    reshard_after_forward: Optional[bool] = None
    fsdp_size: Optional[int] = None

    activation_mode: str = "none"
    activation_swap_inputs: bool = True

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> None:
        """Validate a required parallel size."""
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")

    @staticmethod
    def _validate_optional_positive_int(name: str, value: Optional[int], type_name: str) -> None:
        """Validate an optional parallel size."""
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            raise ValueError(f"{name} must be a positive {type_name} when provided, got {value!r}.")

    def _validate_parallel_sizes(self) -> None:
        """Validate configured parallel dimensions."""
        if self.tp_size != 1:
            raise ValueError(
                "Current trainer backend only supports replacing FSDP/fully_shard. "
                f"Expected tp_size=1, got {self.tp_size}."
            )
        self._validate_positive_int("cp_size", self.cp_size)
        self._validate_positive_int("ep_size", self.ep_size)
        self._validate_optional_positive_int("efsdp_size", self.efsdp_size, "integer")
        self._validate_optional_positive_int("fsdp_size", self.fsdp_size, "int")

    def _validate_world_size(self) -> None:
        """Validate that parallel dimensions divide the runtime world size."""
        if self.cp_size == 1 and self.ep_size == 1 and self.efsdp_size is None:
            return
        world_size = get_platform().get_world_size()
        if self.cp_size > 1 and world_size % self.cp_size != 0:
            raise ValueError(f"world_size ({world_size}) must be divisible by cp_size ({self.cp_size}).")
        if self.ep_size == 1 and self.efsdp_size is None:
            return
        if world_size % self.ep_size != 0:
            raise ValueError(f"world_size ({world_size}) must be divisible by ep_size ({self.ep_size}).")
        edp_size = world_size // self.ep_size
        if self.efsdp_size is not None and edp_size % self.efsdp_size != 0:
            raise ValueError(
                "world_size / ep_size must be divisible by efsdp_size, got "
                f"({world_size} / {self.ep_size}) % {self.efsdp_size} != 0."
            )

    def _validate_runtime_options(self) -> None:
        """Validate dispatcher, dtype, device, and activation settings."""
        if self.token_dispatcher not in _VALID_TOKEN_DISPATCHERS:
            raise ValueError(
                "token_dispatcher must be one of "
                f"{sorted(_VALID_TOKEN_DISPATCHERS)}, got {self.token_dispatcher!r}."
            )
        if self.param_dtype is not None and self.param_dtype not in _VALID_DTYPES:
            raise ValueError(
                f"param_dtype must be one of {sorted(_VALID_DTYPES)}, got {self.param_dtype!r}."
            )
        if self.reduce_dtype is not None and self.reduce_dtype not in _VALID_DTYPES:
            raise ValueError(
                f"reduce_dtype must be one of {sorted(_VALID_DTYPES)}, got {self.reduce_dtype!r}."
            )
        if self.device_type not in {"auto", "npu", "cuda", "cpu"}:
            raise ValueError(
                f"device_type must be one of ['auto', 'cpu', 'cuda', 'npu'], got {self.device_type!r}."
            )
        if self.reshard_after_forward is not None and not isinstance(
            self.reshard_after_forward, bool
        ):
            raise ValueError(
                "reshard_after_forward must be a bool when provided, "
                f"got {type(self.reshard_after_forward).__name__}."
            )
        valid_activation_modes = {"none", "recompute", "swap"}
        if self.activation_mode not in valid_activation_modes:
            raise ValueError(
                f"activation_mode must be one of {sorted(valid_activation_modes)}, "
                f"got {self.activation_mode!r}."
            )

    def validate(self) -> None:
        """Validate supported argument values."""
        self._validate_parallel_sizes()
        self._validate_world_size()
        self._validate_runtime_options()

    @classmethod
    def from_dict(cls, config: dict) -> "HyperParallelArguments":
        """Build arguments from a plain dict."""
        known_fields = set(cls.__dataclass_fields__)  # pylint: disable=no-member
        hp_args = cls(
            **{key: value for key, value in config.items() if key in known_fields}
        )
        hp_args.validate()
        return hp_args

    @classmethod
    def from_finetuning_args(cls, finetuning_args) -> "HyperParallelArguments":
        """Extract HyperParallel arguments from LlamaFactory finetuning args."""
        # Install the host-memory patch before LlamaFactory's load_model runs.
        patch_llamafactory_fp32_upcast()
        raw = getattr(finetuning_args, "hyper_parallel_args", None)
        if raw is None:
            hp_args = cls()
            hp_args.validate()
            return hp_args
        if isinstance(raw, str):
            with open(raw, "r", encoding="utf-8") as file:
                raw = json.load(file)
        if not isinstance(raw, dict):
            raise ValueError(
                "finetuning_args.hyper_parallel_args must be a dict or JSON file path, "
                f"got {type(raw).__name__}."
            )
        return cls.from_dict(raw)


# ---------------------------------------------------------------------------
# Device / mesh / mixed precision resolution
# ---------------------------------------------------------------------------


def _resolve_device_type(hp_args) -> str:
    """Resolve the runtime device type for HyperParallel wrapping."""
    if hp_args.device_type != "auto":
        return hp_args.device_type
    if hasattr(torch, "npu") and torch.npu.is_available():  # pylint: disable=no-member
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_device_mesh(accelerator, hp_args):
    """Build an FSDP mesh compatible with Accelerate's FSDP2 expectations.

    When ``hp_args.fsdp_size`` is set, build a 2D HSDP mesh
    (``dp`` × ``fsdp``) directly so the run uses HSDP instead of plain 1D
    FSDP.  Otherwise inherit the mesh from Accelerate (1D FSDP).
    """
    if hp_args.fsdp_size is not None:
        device_type = _resolve_device_type(hp_args)
        world_size = get_platform().get_world_size()
        fsdp_size = hp_args.fsdp_size
        if fsdp_size >= world_size:
            return init_device_mesh(device_type, (world_size,), mesh_dim_names=("dp",))
        if world_size % fsdp_size != 0:
            raise ValueError(
                f"world_size={world_size} must be divisible by fsdp_size={fsdp_size}."
            )
        dp_size = world_size // fsdp_size
        return init_device_mesh(
            device_type,
            (dp_size, fsdp_size),
            mesh_dim_names=("dp", "fsdp"),
        )

    mesh = getattr(accelerator, "torch_device_mesh", None)
    if mesh is not None:
        cp_size = getattr(hp_args, "cp_size", 1)
        mesh_dim_names = getattr(mesh, "mesh_dim_names", None) or ()
        if cp_size > 1 and {"dp", "cp"}.issubset(set(mesh_dim_names)):
            return mesh[("dp", "cp")]
        fsdp_dim_names = getattr(
            getattr(accelerator, "parallelism_config", None), "fsdp_dim_names", None
        )
        if fsdp_dim_names:
            return mesh[tuple(fsdp_dim_names)]
        return mesh

    cached_mesh = getattr(accelerator, "_hp_device_mesh", None)
    if cached_mesh is not None:
        return cached_mesh

    device_type = _resolve_device_type(hp_args)
    world_size = get_platform().get_world_size()
    cp_size = getattr(hp_args, "cp_size", 1)
    if cp_size > 1:
        if world_size % cp_size != 0:
            raise ValueError(f"world_size ({world_size}) must be divisible by cp_size ({cp_size}).")
        dp_size = world_size // cp_size
        mesh = init_device_mesh(device_type, (dp_size, cp_size), mesh_dim_names=("dp", "cp"))
    else:
        mesh = init_device_mesh(device_type, (world_size,), mesh_dim_names=("dp",))

    setattr(accelerator, "_hp_device_mesh", mesh)
    return mesh


def _resolve_fsdp_mesh(mesh):
    """Return the mesh that should be used for FSDP parameter sharding."""
    if mesh is None:
        return None
    mesh_dim_names = getattr(mesh, "mesh_dim_names", None) or ()
    if {"dp", "cp"}.issubset(set(mesh_dim_names)):
        return mesh[("dp", "cp")].flatten(mesh_dim_name="fsdp")
    if "dp" in mesh_dim_names:
        return mesh["dp"]
    return mesh


def _build_mp_policy(hp_args) -> MixedPrecisionPolicy:
    """Build HyperParallel mixed precision policy."""
    return MixedPrecisionPolicy(
        param_dtype=_DTYPE_MAP[hp_args.param_dtype]
        if hp_args.param_dtype is not None
        else None,
        reduce_dtype=_DTYPE_MAP[hp_args.reduce_dtype]
        if hp_args.reduce_dtype is not None
        else None,
        output_dtype=_DTYPE_MAP[hp_args.param_dtype]
        if hp_args.param_dtype is not None
        else None,
        cast_forward_inputs=True,
    )


def _resolve_offload_policy(fsdp2_plugin) -> OffloadPolicy:
    """Translate Accelerate cpu_offload config to HyperParallel offload policy."""
    cpu_offload = getattr(fsdp2_plugin, "cpu_offload", None)
    if isinstance(cpu_offload, OffloadPolicy):
        return cpu_offload
    if cpu_offload is True:
        return CPUOffloadPolicy()
    if type(cpu_offload).__name__ == "CPUOffloadPolicy":
        return CPUOffloadPolicy()
    return OffloadPolicy()


def _resolve_mp_policy(fsdp2_plugin, hp_args) -> MixedPrecisionPolicy:
    """Resolve mixed precision with Accelerate defaults and optional HyperParallel overrides."""
    policy = getattr(fsdp2_plugin, "mixed_precision_policy", None)
    resolved_policy = MixedPrecisionPolicy()
    if policy is not None:
        resolved_policy = MixedPrecisionPolicy(
            param_dtype=getattr(policy, "param_dtype", None),
            reduce_dtype=getattr(policy, "reduce_dtype", None),
            output_dtype=getattr(policy, "output_dtype", None),
            cast_forward_inputs=getattr(policy, "cast_forward_inputs", True),
        )

    hp_policy = _build_mp_policy(hp_args)
    if hp_args.param_dtype is not None:
        resolved_policy.param_dtype = hp_policy.param_dtype
        resolved_policy.output_dtype = hp_policy.output_dtype
    if hp_args.reduce_dtype is not None:
        resolved_policy.reduce_dtype = hp_policy.reduce_dtype
    return resolved_policy


# ---------------------------------------------------------------------------
# Model traversal and wrapping helpers
# ---------------------------------------------------------------------------


def _is_compiled_module(model: nn.Module) -> bool:
    """Best-effort check for compiled modules."""
    return hasattr(model, "_orig_mod")


def _get_module_children_bottom_up(model: nn.Module, return_fqns: bool = False):
    """Return model children bottom-up, matching Accelerate helper semantics."""
    modules = []

    def _visit(module: nn.Module, prefix: str = ""):
        for child_name, child in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            _visit(child, child_prefix)
        modules.append((prefix, module) if return_fqns else module)

    _visit(model)
    return modules


def _get_non_persistent_buffers(
    model: nn.Module, recurse: bool = True, fqns: bool = True
):
    """Collect non-persistent buffers."""
    buffers = set()
    for module_name, module in model.named_modules():
        if not recurse and module is not model:
            continue
        for buffer_name in getattr(module, "_non_persistent_buffers_set", set()):
            if fqns and module_name:
                buffers.add(f"{module_name}.{buffer_name}")
            else:
                buffers.add(buffer_name)
    return buffers


def _get_module_class_from_name(module: nn.Module, class_name: str):
    """Find a module class by name from the model tree."""
    for child in module.modules():
        if child.__class__.__name__ == class_name:
            return child.__class__
    return None


def _move_model_to_meta(model: nn.Module) -> nn.Module:
    """Move the model to meta before fully_shard to match Accelerate FSDP2 loading order."""
    model = model.to(torch.device("meta"))
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model


def _move_unwrapped_model_state_to_meta(model: nn.Module) -> nn.Module:
    """Move all model state to meta without invalidating nested HSDP state.

    EP fully-shards its expert containers before this common FSDP2 path. A
    recursive ``model.to(meta)`` would replace the expert DTensor parameters
    only in the module tree, leaving the inner HSDP objects pointing at their
    old materialized parameters. Convert those managed parameters explicitly
    so their distributed metadata and both reference graphs stay consistent.
    """
    # Reuse one replacement for every module slot that shares the same Parameter.
    converted_parameters: dict[nn.Parameter, nn.Parameter] = {}
    # Reuse one replacement for shared buffers as well as shared parameters.
    converted_buffers: dict[torch.Tensor, torch.Tensor] = {}
    # Construct the device object once because every state tensor has the same target.
    meta_device = torch.device("meta")

    def _convert_parameter(parameter: nn.Parameter) -> nn.Parameter:
        """Create or return one metadata-preserving meta Parameter."""
        # Inner-HSDP parameters converted in the first pass must keep exact object identity.
        if parameter.is_meta:
            # Returning the installed object also avoids wrapping pre-existing meta parameters twice.
            return parameter
        # A tied/shared parameter must keep object identity across all owning modules.
        converted = converted_parameters.get(parameter)
        # Only the first occurrence allocates its corresponding meta object.
        if converted is None:
            # DTensor.to(meta) preserves its mesh, placements, global shape/stride, and dtype.
            converted_value = parameter.to(meta_device)
            # Re-wrap the converted tensor while retaining the trainability contract.
            converted = nn.Parameter(converted_value, requires_grad=parameter.requires_grad)
            # Inner-HSDP ownership is an out-of-band attribute not copied by Parameter().
            if getattr(parameter, "_hsdp_param_initialized", False):
                # Preserve it so the later parent/root fully_shard skips this expert parameter.
                converted._hsdp_param_initialized = True  # pylint: disable=protected-access
            # Cache the replacement before another shared module slot is visited.
            converted_parameters[parameter] = converted
        # Return the unique meta Parameter for this source object.
        return converted

    # One HSDP scheduler may be exposed by several module roots, so process it once.
    visited_hsdp_schedulers: set[int] = set()
    # Walk the current tree before replacing ordinary parameters.
    for module in model.modules():
        # Plain modules have no additional parameter references to synchronize.
        if not isinstance(module, HSDPModule):
            continue
        # Read the scheduler installed by fully_shard on the expert container.
        scheduler = getattr(module, "hsdp_scheduler", None)
        # A partially constructed mixin has no state to update.
        if scheduler is None or id(scheduler) in visited_hsdp_schedulers:
            continue
        # Mark the shared scheduler before visiting its managed parameter list.
        visited_hsdp_schedulers.add(id(scheduler))
        # The state owns the HSDPParam objects used by forward/backward communication.
        hsdp_state = getattr(scheduler, "hsdp_state", None)
        # Treat a scheduler without an initialized state as having no managed parameters.
        if hsdp_state is None:
            continue
        # Synchronize every HSDP-managed expert parameter with its meta replacement.
        for hsdp_param in hsdp_state.hsdp_params:
            # This is the materialized sharded DTensor currently installed on the module.
            sharded_parameter = hsdp_param.sharded_param
            # Preserve the DTensor metadata while replacing only its local storage by meta.
            meta_parameter = _convert_parameter(sharded_parameter)
            # Save hooks from the old object before switching HSDP's canonical reference.
            hsdp_param._parameter_hook_migrator._save_backward_hooks(  # pylint: disable=protected-access
                sharded_parameter
            )
            # Point HSDP at the same meta Parameter that will be visible in the module tree.
            hsdp_param.sharded_param = meta_parameter
            # Drop HSDP's second strong reference to the old materialized communication storage.
            hsdp_param._sharded_param_data = (  # pylint: disable=protected-access
                meta_parameter.to_local().reshape(-1)
                if isinstance(meta_parameter, DTensor)
                else meta_parameter.reshape(-1)
            )
            # Update the owner and every shared owner recorded by HSDP in one operation.
            hsdp_param._setattr_on_modules(meta_parameter)  # pylint: disable=protected-access

    # Convert parameters and buffers not managed by an already-created inner HSDP unit.
    for module in model.modules():
        # Only inspect direct parameter slots; recursive traversal would convert children twice.
        for name, parameter in module._parameters.items():  # pylint: disable=protected-access
            # None is a valid registered placeholder and needs no conversion.
            if parameter is None:
                continue
            # Expert parameters already resolve from the cache; ordinary parameters convert here.
            module._parameters[name] = _convert_parameter(parameter)  # pylint: disable=protected-access
        # Buffers are not represented by HSDPParam, so all modules follow this common path.
        for name, buffer in module._buffers.items():  # pylint: disable=protected-access
            # None is a valid registered placeholder and needs no conversion.
            if buffer is None:
                continue
            # Preserve shared-buffer identity instead of allocating one meta tensor per slot.
            converted_buffer = converted_buffers.get(buffer)
            # Convert the first occurrence and cache it for any aliases.
            if converted_buffer is None:
                converted_buffer = buffer.to(meta_device)
                converted_buffers[buffer] = converted_buffer
            # Install the meta buffer into this direct module slot.
            module._buffers[name] = converted_buffer  # pylint: disable=protected-access

    # Re-establish architecture-declared ties such as input embeddings and LM head.
    if hasattr(model, "tie_weights"):
        # Hugging Face models use this hook as the authoritative tying operation.
        model.tie_weights()
    # Keep the public preparation flow operating on the original model object.
    return model


def _get_parameters_from_modules(
    modules: Iterable[nn.Module] | str, model: nn.Module, device
) -> set[nn.Parameter]:
    """Convert ignored modules to ignored parameters, matching Accelerate behaviour."""
    if modules is None:
        return set()

    parameters = []
    if isinstance(modules, str):
        pattern = re.compile(modules)
        matched_modules = []
        for name, module in model.named_modules():
            if pattern.fullmatch(name):
                module.to(device)
                matched_modules.append(module)
        modules = matched_modules

    for module in modules:
        parameters.extend(list(module.parameters()))
    return set(parameters)


def _prepare_auto_wrap_policy(fsdp2_plugin, model: nn.Module):
    """Prepare auto-wrap policy, copied from Accelerate FSDP2 logic."""
    fn = fsdp2_plugin.auto_wrap_policy
    if isinstance(fn, functools.partial):
        fn = fn.func

    if fn is transformer_auto_wrap_policy:
        no_split_modules = getattr(model, "_no_split_modules", None) or []
        transformer_cls_names_to_wrap = list(no_split_modules)
        if fsdp2_plugin.transformer_cls_names_to_wrap is not None:
            transformer_cls_names_to_wrap = fsdp2_plugin.transformer_cls_names_to_wrap
        transformer_cls_to_wrap = set()

        for layer_class in transformer_cls_names_to_wrap:
            transformer_cls = _get_module_class_from_name(model, layer_class)
            if transformer_cls is None:
                raise ValueError(
                    f"Could not find the transformer layer class {layer_class} in the model."
                )
            transformer_cls_to_wrap.add(transformer_cls)

        def policy(module: nn.Module) -> bool:
            if fsdp2_plugin.transformer_cls_names_to_wrap is None:
                return False
            return isinstance(module, tuple(transformer_cls_to_wrap))

    elif fn is size_based_auto_wrap_policy:

        def policy(module: nn.Module) -> bool:
            return (
                sum(param.numel() for param in module.parameters())
                > fsdp2_plugin.min_num_params
            )

    else:
        return None

    return policy


# ---------------------------------------------------------------------------
# Checkpoint and export helpers
# ---------------------------------------------------------------------------


def _localize_optimizer_state(optim_sd: dict) -> dict:
    """Convert DTensors in optimizer state dict to local CPU tensors for serialization."""
    new_state = {}
    for param_idx, state in optim_sd.get("state", {}).items():
        local_state = {}
        for key, val in state.items():
            if isinstance(val, DTensor):
                local_state[key] = val.to_local().detach().cpu()
            elif isinstance(val, torch.Tensor):
                local_state[key] = val.detach().cpu()
            else:
                local_state[key] = val
        new_state[param_idx] = local_state
    return {"state": new_state, "param_groups": optim_sd.get("param_groups", [])}


def _get_optimizer_param_by_idx(optimizer) -> dict[int, torch.nn.Parameter]:
    """Map optimizer state indices to the current optimizer parameters."""
    param_by_idx: dict[int, torch.nn.Parameter] = {}
    param_idx = 0
    for group in optimizer.param_groups:
        for param in group["params"]:
            param_by_idx[param_idx] = param
            param_idx += 1
    return param_by_idx


def _get_optimizer_param_device(param):
    """Return the target device for a tensor restored into an optimizer state."""
    if isinstance(param, DTensor):
        return param.to_local().device
    return param.device


def _restore_optimizer_state_value(current_state, key, param, saved_val) -> None:
    """Restore one optimizer state entry while preserving existing tensor objects."""
    current_val = current_state.get(key)
    if current_val is None:
        if isinstance(saved_val, torch.Tensor):
            current_state[key] = saved_val.to(_get_optimizer_param_device(param))
        else:
            current_state[key] = saved_val
        return

    if isinstance(current_val, DTensor):
        local = current_val.to_local()
        local.copy_(saved_val.to(local.device))
    elif isinstance(current_val, torch.Tensor):
        current_val.copy_(saved_val.to(current_val.device))
    else:
        current_state[key] = saved_val


def _restore_optimizer_param_groups(optimizer, saved_sd: dict) -> None:
    """Restore non-parameter optimizer group options from a saved state dict."""
    for saved_group, current_group in zip(
        saved_sd.get("param_groups", []), optimizer.param_groups
    ):
        for key, val in saved_group.items():
            if key != "params":
                current_group[key] = val


def _load_local_optimizer_state(optimizer, saved_sd: dict) -> None:
    """Copy saved local optimizer state into the optimizer's current state."""
    param_by_idx = _get_optimizer_param_by_idx(optimizer)

    for param_idx, saved_state in saved_sd.get("state", {}).items():
        param_idx = int(param_idx) if isinstance(param_idx, str) else param_idx
        param = param_by_idx.get(param_idx)
        if param is None or param not in optimizer.state:
            continue
        current_state = optimizer.state[param]
        for key, saved_val in saved_state.items():
            _restore_optimizer_state_value(current_state, key, param, saved_val)

    _restore_optimizer_param_groups(optimizer, saved_sd)


# ---------------------------------------------------------------------------
# Accelerate compatibility shims
# ---------------------------------------------------------------------------


def fsdp2_load_full_state_dict(accelerator, model: nn.Module, full_sd: dict):
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        accelerator (`Accelerator`): The accelerator instance
        model (`nn.Module`):
            The model to load the state dict into, expected to be on meta device or a VRAM spike can occur
        full_sd (`dict`): The full state dict to load, can only be on rank 0
    """
    # Model was previously copied to meta device
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}

    # Rank 0 distributes the full state dict to other ranks
    def _infer_parameter_dtype(target_model, param_name, empty_param):
        try:
            old_param = target_model.get_parameter_or_buffer(param_name)
        except AttributeError:
            # Need this for LORA, as there some params are not *parameters* of sorts
            base_param_name, local_param_name = param_name.rsplit(".", 1)
            submodule = target_model.get_submodule(base_param_name)
            old_param = getattr(submodule, local_param_name)

        is_torch_e4m3fn_available = hasattr(torch, "float8_e4m3fn")
        casting_dtype = None
        is_param_float8_e4m3fn = (
            is_torch_e4m3fn_available and empty_param.dtype == torch.float8_e4m3fn
        )

        if empty_param.dtype.is_floating_point and not is_param_float8_e4m3fn:
            casting_dtype = old_param.dtype

        return old_param is not None and old_param.is_contiguous(), casting_dtype

    def _cast_and_contiguous(tensor, to_contiguous, dtype):
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if to_contiguous:
            tensor = tensor.contiguous()
        return tensor

    if accelerator.is_main_process:
        for (param_name, full_param), sharded_param in zip(
            full_sd.items(), meta_sharded_sd.values()
        ):
            device_mesh = sharded_param.device_mesh
            full_param = full_param.detach().to(device_mesh.device_type)
            dist.broadcast(full_param, src=0, group=dist.group.WORLD)
            sharded_tensor = distribute_tensor(
                full_param, device_mesh, sharded_param.placements
            )
            to_contiguous, casting_dtype = _infer_parameter_dtype(
                model,
                param_name,
                full_param,
            )
            sharded_tensor = _cast_and_contiguous(
                sharded_tensor, to_contiguous, casting_dtype
            )
            sharded_sd[param_name] = sharded_tensor
    # We need this else to have a matching `broadcast` for all of the ranks, else we deadlock
    else:
        for param_name, sharded_param in meta_sharded_sd.items():
            device_mesh = sharded_param.device_mesh
            full_tensor = torch.empty(
                sharded_param.size(),
                device=device_mesh.device_type,
                dtype=sharded_param.dtype,
            )
            dist.broadcast(full_tensor, src=0, group=dist.group.WORLD)
            sharded_tensor = distribute_tensor(
                full_tensor, device_mesh, sharded_param.placements
            )
            to_contiguous, casting_dtype = _infer_parameter_dtype(
                model,
                param_name,
                full_tensor,
            )
            sharded_tensor = _cast_and_contiguous(
                sharded_tensor, to_contiguous, casting_dtype
            )
            sharded_sd[param_name] = sharded_tensor

    # we set `assign=True` because our params are on meta device
    cast(nn.Module, model).load_state_dict(sharded_sd, assign=True)
    return model


def fsdp2_prepare_auto_wrap_policy(fsdp2_plugin, model: nn.Module):
    """Prepare auto-wrap policy, matching Accelerate helper naming and behavior."""
    return _prepare_auto_wrap_policy(fsdp2_plugin, model)


def get_parameters_from_modules(
    modules: Iterable[nn.Module] | str, model: nn.Module, device
) -> set[nn.Parameter]:
    """Convert ignored modules to ignored parameters."""
    return _get_parameters_from_modules(modules, model, device)


# ---------------------------------------------------------------------------
# Runtime preparation helpers
# ---------------------------------------------------------------------------


def _is_fsdp2_wrapped_model(model: nn.Module) -> bool:
    """Return whether the model is already wrapped by HyperParallel FSDP2."""
    return isinstance(model, HSDPModule) or (
        _is_compiled_module(model) and isinstance(model._orig_mod, HSDPModule)  # pylint: disable=protected-access
    )


def _resolve_shard_size(mesh) -> int:
    """Return the FSDP shard-dim size for a 1D FSDP or 2D HSDP mesh.

    HP ``fully_shard`` builds ``FSDPMeshInfo(shard_mesh_dim=0)`` for a 1D mesh
    and ``HSDPMeshInfo(shard_mesh_dim=1, replicate_mesh_dim=0)`` for a 2D mesh
    (see ``platform/*/fully_shard/scheduler.py``). In both cases the shard
    dim is the last mesh dim, so ``mesh.mesh_shape[-1]`` gives the actual
    per-param shard count regardless of HSDP layout.
    """
    if mesh is None:
        return get_platform().get_world_size()
    shape = getattr(mesh, "mesh_shape", None)
    if shape:
        return int(shape[-1])
    return mesh.size() if hasattr(mesh, "size") else get_platform().get_world_size()


def _collect_replicate_params(model: nn.Module, shard_size: int) -> set:
    """Collect params whose dim-0 isn't divisible by ``shard_size``.

    HP ``fully_shard`` raises ``Uneven sharding on dim 0`` for such params
    (e.g. ``shared_expert_gate.weight`` of shape ``(1, hidden)`` on
    ``shard_size > 1``). Routing them through ``replicate_params`` makes
    them DDP-replicated along the shard dim instead.
    """
    replicate = set()
    if shard_size <= 1:
        return replicate
    for _, param in model.named_parameters():
        if param.dim() == 0:
            continue
        if param.size(0) % shard_size != 0:
            replicate.add(param)
    return replicate


def _build_fsdp2_kwargs(accelerator, model: nn.Module, hp_args, fsdp2_plugin) -> dict:
    """Build fully_shard kwargs from accelerator and plugin settings."""
    mesh = _resolve_fsdp_mesh(_build_device_mesh(accelerator, hp_args))
    reshard_after_forward = fsdp2_plugin.reshard_after_forward
    if hp_args.reshard_after_forward is not None:
        reshard_after_forward = hp_args.reshard_after_forward
    kwargs = {
        "reshard_after_forward": reshard_after_forward,
        "offload_policy": _resolve_offload_policy(fsdp2_plugin),
        "mp_policy": _resolve_mp_policy(fsdp2_plugin, hp_args),
        "mesh": mesh if mesh is not None else None,
        "ignored_params": get_parameters_from_modules(
            fsdp2_plugin.ignored_modules, model, accelerator.device
        ),
        "comm_fusion": True,
    }
    replicate_params = _collect_replicate_params(model, _resolve_shard_size(mesh))
    if replicate_params:
        kwargs["replicate_params"] = replicate_params
    return kwargs


def _model_has_4bit_params(model: nn.Module) -> bool:
    """Return whether the model contains bitsandbytes 4-bit parameters."""
    return any(
        param.__class__.__name__ == "Params4bit"
        for _, param in model.named_parameters()
    )


def _prepare_cpu_ram_efficient_loading(
    model: nn.Module, enabled: bool
) -> dict[str, torch.Tensor]:
    """Capture non-persistent buffers before cpu_ram_efficient_loading rematerializes the model."""
    if not enabled:
        return {}

    non_persistent_buffer_fqns = _get_non_persistent_buffers(
        model, recurse=True, fqns=True
    )
    original_non_persistent_buffers = copy.deepcopy(
        {
            name: buffer
            for name, buffer in model.named_buffers()
            if name in non_persistent_buffer_fqns
        }
    )
    return original_non_persistent_buffers


def _apply_auto_wrap_policy(model: nn.Module, fsdp2_plugin, fsdp2_kwargs: dict) -> None:
    """Apply fully_shard to matching child modules before wrapping the root module."""
    auto_wrap_policy_func = fsdp2_prepare_auto_wrap_policy(fsdp2_plugin, model)
    if auto_wrap_policy_func is None:
        return

    for module in _get_module_children_bottom_up(model)[:-1]:
        if auto_wrap_policy_func(module) and not isinstance(module, HSDPModule):
            fully_shard(module, **fsdp2_kwargs)


def _setup_prefetch(model: nn.Module) -> None:
    """Set up forward and backward prefetch for HSDP-wrapped child modules.

    Each wrapped layer prefetches the next layer's allgather during forward,
    and the previous layer's allgather during backward, to overlap communication
    with computation.

    Backward prefetch uses reversed module order because backward execution
    proceeds from the last layer to the first.
    """
    wrapped_modules = [
        m for m in model.modules() if isinstance(m, HSDPModule) and m is not model
    ]
    num_to_forward_prefetch = 1
    num_to_backward_prefetch = 1

    # Forward prefetch: each layer prefetches the next layer(s)
    for i, layer in enumerate(wrapped_modules):
        j_end = min(len(wrapped_modules), i + 1 + num_to_forward_prefetch)
        forward_targets = wrapped_modules[i + 1 : j_end]
        if forward_targets:
            layer.set_modules_to_forward_prefetch(forward_targets)

    # Backward prefetch: reverse order since backward runs last-to-first
    wrapped_modules.reverse()
    for i, layer in enumerate(wrapped_modules):
        j_end = min(len(wrapped_modules), i + 1 + num_to_backward_prefetch)
        backward_targets = wrapped_modules[i + 1 : j_end]
        if backward_targets:
            layer.set_modules_to_backward_prefetch(backward_targets)


def _restore_non_persistent_buffers(
    model: nn.Module, buffers: dict[str, torch.Tensor], device
) -> None:
    """Restore non-persistent buffers after cpu_ram_efficient_loading finishes."""
    if not buffers:
        return

    for fqn, buffer_tensor in buffers.items():
        buffer_tensor = buffer_tensor.to(device)
        if "." in fqn:
            parent_fqn, local_buffer_name = fqn.rsplit(".", 1)
            parent_module = model.get_submodule(parent_fqn)
        else:
            local_buffer_name = fqn
            parent_module = model
        parent_module.register_buffer(
            local_buffer_name, buffer_tensor, persistent=False
        )

    if hasattr(model, "tie_weights"):
        model.tie_weights()


def _maybe_upcast_trainable_params(accelerator, model: nn.Module) -> None:
    """Upcast model parameters to fp32 when mixed precision requires Accelerate-compatible behavior.

    ``model.to(torch.float32)`` creates new fp32 parameters in the module tree.
    Refresh HSDP's cached sharded parameter references and mixed-precision dtypes
    so comm_fusion uses the new fp32 parameter dtype as well.
    """
    model_dtype = getattr(model, "dtype", None)
    should_upcast = accelerator.mixed_precision != "no" and (
        model_dtype is None or model_dtype != torch.float32
    )
    if not should_upcast:
        return

    model.to(torch.float32)

    for module in model.modules():
        if isinstance(module, HSDPModule):
            state = module.hsdp_scheduler.hsdp_state  # pylint: disable=protected-access
            for hsdp_param in state.hsdp_params:
                hsdp_param.reset_sharded_param()
            state._init_mp_dtypes()  # pylint: disable=protected-access

    if accelerator.is_main_process:
        warnings.warn(
            "FSDP upcast of low precision parameters to fp32 (since mixed_precision != 'no') "
            "may affect the precision of model checkpoints."
        )


# ---------------------------------------------------------------------------
# Optimizer wiring
# ---------------------------------------------------------------------------


def wrap_optimizer_with_skip_dtensor_dispatch(optimizer) -> None:
    """Wrap optimizer.step so DTensor dispatch is skipped during parameter updates."""
    if getattr(optimizer, "_hp_step_wrapped", False):
        return

    original_step = optimizer.step

    def _hp_step(bound_optimizer, *args, **kwargs):
        del bound_optimizer
        with SkipDTensorDispatch():
            return original_step(*args, **kwargs)

    optimizer.step = types.MethodType(_hp_step, optimizer)
    setattr(optimizer, "_hp_step_wrapped", True)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def export_to_hf_format(model: nn.Module, tokenizer, save_dir: str) -> None:
    """Gather full state dict via HyperParallel and save in HuggingFace-compatible format."""
    from hyper_parallel.core.fully_shard.api import (  # pylint: disable=C0415
        get_model_state_dict as hp_get_model_state_dict,
    )
    from torch.distributed.checkpoint.state_dict import StateDictOptions  # pylint: disable=C0415

    export_dir = Path(save_dir)
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state_dict = hp_get_model_state_dict(model, options=options)

    if get_platform().get_rank() == 0:
        export_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(export_dir), state_dict=state_dict)
        else:
            torch.save(state_dict, export_dir / "pytorch_model.bin")

        if tokenizer is not None:
            tokenizer.save_pretrained(str(export_dir))

    if get_platform().get_world_size() > 1:
        torch.distributed.barrier()


def save_hsdp_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    lr_scheduler,
    output_dir: str,
    should_save_scheduler: bool = True,
) -> None:
    """Save HSDP model/optimizer shards per-rank and scheduler."""
    from hyper_parallel.core.distributed_checkpoint.api import save as hp_save  # pylint: disable=C0415

    os.makedirs(output_dir, exist_ok=True)
    rank = get_platform().get_rank()

    model_dir = os.path.join(output_dir, f"{HSDP_MODEL_NAME}_0")
    os.makedirs(model_dir, exist_ok=True)
    logger.info("Saving HSDP model shards to %s (rank %d)", model_dir, rank)
    model_sd = model.state_dict()
    hp_save(model_sd, checkpoint_id=model_dir, use_collectives=False)

    if optimizer is not None:
        optim_file = os.path.join(output_dir, f"{HSDP_OPTIMIZER_NAME}_rank{rank}.pt")
        logger.info("Saving optimizer shard to %s", optim_file)
        local_optim_sd = _localize_optimizer_state(optimizer.state_dict())
        torch.save(local_optim_sd, optim_file)

    if should_save_scheduler and lr_scheduler is not None:
        torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))


def load_hsdp_model(model: nn.Module, checkpoint_dir: str) -> bool:
    """Load model from HSDP sharded checkpoint saved by ``hp_save``."""
    from hyper_parallel.core.distributed_checkpoint.api import load as hp_load  # pylint: disable=C0415

    model_dir = os.path.join(checkpoint_dir, f"{HSDP_MODEL_NAME}_0")

    if not os.path.isdir(model_dir):
        return False

    logger.info("Loading HSDP model shards from %s", model_dir)
    state_dict = model.state_dict()
    hp_load(state_dict, checkpoint_id=model_dir, use_collectives=False)
    model.load_state_dict(state_dict)
    return True


def load_hsdp_optimizer_and_scheduler(
    optimizer: Optional[torch.optim.Optimizer],
    lr_scheduler,
    checkpoint_dir: str,
) -> None:
    """Load optimizer/scheduler from per-rank checkpoint files."""
    if checkpoint_dir is None:
        return

    rank = get_platform().get_rank()
    optim_file = os.path.join(checkpoint_dir, f"{HSDP_OPTIMIZER_NAME}_rank{rank}.pt")

    if os.path.isfile(optim_file) and optimizer is not None:
        logger.info("Loading optimizer shard from %s", optim_file)
        saved_sd = torch.load(optim_file, map_location="cpu", weights_only=True)
        _load_local_optimizer_state(optimizer, saved_sd)

    scheduler_file = os.path.join(checkpoint_dir, "scheduler.pt")
    if os.path.isfile(scheduler_file) and lr_scheduler is not None:
        lr_scheduler.load_state_dict(
            torch.load(scheduler_file, map_location="cpu", weights_only=True)
        )


# ---------------------------------------------------------------------------
# Entry point: fsdp2_prepare_model
# ---------------------------------------------------------------------------


def fsdp2_prepare_model(accelerator, model: nn.Module, hp_args) -> nn.Module:
    """
    Prepare model following Accelerate FSDP2 flow, using HyperParallel fully_shard.

    This function is designed to be called with the runtime `accelerator`
    instance already created by `transformers.Trainer` / `accelerate`.

    Required accelerator attributes:
        state.fsdp_plugin: FSDP plugin configuration used to derive wrapping and
            state-dict behaviour.
        torch_device_mesh: Optional device mesh prepared by Accelerate.
        parallelism_config.fsdp_dim_names: Optional FSDP mesh dimension names
            used when `torch_device_mesh` is available.
        device: Current process device, used for ignored module parameter
            materialization and buffer restoration.
        is_main_process: Whether the current rank is the main process during
            full state-dict distribution.
        mixed_precision: Mixed precision mode string, used for the final
            parameter upcast behavior.
    """
    if _is_fsdp2_wrapped_model(model):
        return model

    fsdp2_plugin = accelerator.state.fsdp_plugin
    fsdp2_plugin.set_auto_wrap_policy(model)

    model_has_params4bit = _model_has_4bit_params(model)
    pre_ep_state_dict = getattr(model, "_hyper_parallel_pre_ep_state_dict", None)
    original_sd = pre_ep_state_dict if pre_ep_state_dict is not None else model.state_dict()
    should_restore_non_persistent_buffers = (
        fsdp2_plugin.cpu_ram_efficient_loading and not model_has_params4bit
    )
    original_non_persistent_buffers = _prepare_cpu_ram_efficient_loading(
        model, should_restore_non_persistent_buffers
    )
    if should_restore_non_persistent_buffers:
        if pre_ep_state_dict is None:
            model = _move_model_to_meta(model)
        else:
            model = _move_unwrapped_model_state_to_meta(model)

    fsdp2_kwargs = _build_fsdp2_kwargs(accelerator, model, hp_args, fsdp2_plugin)

    # Detect transformer blocks before fully_shard modifies the model tree.
    # The block references (parent, attr, ModuleList) survive FSDP wrapping
    # since fully_shard modifies modules in-place rather than replacing them.
    from hyper_parallel.integration.llamafactory.activation import (  # pylint: disable=C0415
        find_transformer_blocks,
        setup_activation_optimization,
    )

    block_info = (
        find_transformer_blocks(model) if hp_args.activation_mode != "none" else None
    )

    _apply_auto_wrap_policy(model, fsdp2_plugin, fsdp2_kwargs)
    if not isinstance(model, HSDPModule):
        fully_shard(model, **fsdp2_kwargs)

    _setup_prefetch(model)

    if fsdp2_plugin.cpu_ram_efficient_loading:
        model = fsdp2_load_full_state_dict(accelerator, model, original_sd)
    if pre_ep_state_dict is not None:
        delattr(model, "_hyper_parallel_pre_ep_state_dict")

    # Activation wrapping after loading: the loading path is identical to
    # the non-activation case, avoiding interaction between CheckpointWrapper's
    # setattr replacement and load_state_dict(assign=True).  We pass the
    # pre-detected block_info since FSDP may have changed the model structure.
    setup_activation_optimization(model, hp_args, block_info=block_info)

    _restore_non_persistent_buffers(
        model, original_non_persistent_buffers, accelerator.device
    )
    _maybe_upcast_trainable_params(accelerator, model)
    return model
