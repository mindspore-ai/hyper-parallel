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
"""Accelerate-style FSDP2 utilities backed by HyperParallel fully_shard."""
import copy
import functools
import re
import warnings
from collections.abc import Iterable
from typing import cast

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.fully_shard.api import HSDPModule, fully_shard
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy
from hyper_parallel.platform import get_platform

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


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
    """Build an FSDP mesh compatible with Accelerate's FSDP2 expectations."""
    mesh = getattr(accelerator, "torch_device_mesh", None)
    if mesh is not None:
        fsdp_dim_names = getattr(getattr(accelerator, "parallelism_config", None), "fsdp_dim_names", None)
        if fsdp_dim_names:
            return mesh[tuple(fsdp_dim_names)]
        return mesh

    device_type = _resolve_device_type(hp_args)
    world_size = get_platform().get_world_size()
    return init_device_mesh(device_type, (world_size,), mesh_dim_names=("dp",))


def _build_mp_policy(hp_args) -> MixedPrecisionPolicy:
    """Build HyperParallel mixed precision policy."""
    return MixedPrecisionPolicy(
        param_dtype=_DTYPE_MAP[hp_args.param_dtype] if hp_args.param_dtype is not None else None,
        reduce_dtype=_DTYPE_MAP[hp_args.reduce_dtype] if hp_args.reduce_dtype is not None else None,
        output_dtype=_DTYPE_MAP[hp_args.param_dtype] if hp_args.param_dtype is not None else None,
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


def _is_cpu_offload_enabled(cpu_offload) -> bool:
    """Return whether CPU offload is truly enabled."""
    if cpu_offload is True:
        return True
    if isinstance(cpu_offload, CPUOffloadPolicy):
        return True
    return type(cpu_offload).__name__ == "CPUOffloadPolicy"


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


def _get_non_persistent_buffers(model: nn.Module, recurse: bool = True, fqns: bool = True):
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



def _get_parameters_from_modules(modules: Iterable[nn.Module] | str, model: nn.Module, device) -> set[nn.Parameter]:
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
                raise ValueError(f"Could not find the transformer layer class {layer_class} in the model.")
            transformer_cls_to_wrap.add(transformer_cls)

        def policy(module: nn.Module) -> bool:
            if fsdp2_plugin.transformer_cls_names_to_wrap is None:
                return False
            return isinstance(module, tuple(transformer_cls_to_wrap))

    elif fn is size_based_auto_wrap_policy:

        def policy(module: nn.Module) -> bool:
            return sum(param.numel() for param in module.parameters()) > fsdp2_plugin.min_num_params

    else:
        return None

    return policy


def fsdp2_load_full_state_dict(accelerator, model: nn.Module, full_sd: dict, cpu_offload: bool = False):
    """Load full state dict into a HyperParallel-sharded model following Accelerate semantics."""
    meta_sharded_sd = model.state_dict()
    local_sd = {}

    def _infer_parameter_dtype(target_model: nn.Module, param_name: str, empty_param: torch.Tensor):
        try:
            old_param = target_model.get_parameter(param_name)
        except Exception:  # pylint: disable=broad-except
            old_param = None
        if old_param is None:
            try:
                old_param = target_model.get_buffer(param_name)
            except Exception:  # pylint: disable=broad-except
                old_param = None
        if old_param is None:
            base_name, local_name = param_name.rsplit(".", 1)
            old_param = getattr(target_model.get_submodule(base_name), local_name)

        is_torch_e4m3fn_available = hasattr(torch, "float8_e4m3fn")
        casting_dtype = None
        is_param_float8 = is_torch_e4m3fn_available and empty_param.dtype == torch.float8_e4m3fn
        if empty_param.dtype.is_floating_point and not is_param_float8:
            casting_dtype = old_param.dtype
        if isinstance(old_param, DTensor):
            local_param = old_param.to_local()
            return local_param is not None and local_param.is_contiguous(), casting_dtype
        return old_param is not None and old_param.is_contiguous(), casting_dtype

    def _cast_and_contiguous(tensor: torch.Tensor, to_contiguous: bool, dtype):
        if isinstance(tensor, DTensor):
            local_tensor = tensor.to_local()
            if dtype is not None:
                local_tensor = local_tensor.to(dtype=dtype)
            if to_contiguous:
                local_tensor = local_tensor.contiguous()
            return DTensor.from_local(local_tensor, tensor.device_mesh, tensor.placements)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if to_contiguous:
            tensor = tensor.contiguous()
        return tensor

    if accelerator.is_main_process:
        iterable = full_sd.items()
    else:
        iterable = meta_sharded_sd.items()

    for item in iterable:
        if accelerator.is_main_process:
            param_name, full_param = item
            sharded_param = meta_sharded_sd[param_name]
        else:
            param_name, sharded_param = item
            full_param = torch.empty(sharded_param.size(), device=accelerator.device, dtype=sharded_param.dtype)

        if isinstance(full_param, DTensor):
            full_param = full_param.to_local()

        full_param = full_param.detach().to(accelerator.device)
        dist.broadcast(full_param, src=0, group=dist.group.WORLD)

        if isinstance(sharded_param, DTensor):
            local_param = distribute_tensor(full_param, sharded_param.device_mesh, sharded_param.placements).to_local()
        else:
            local_param = full_param

        to_contiguous, casting_dtype = _infer_parameter_dtype(model, param_name, local_param)
        local_param = _cast_and_contiguous(local_param, to_contiguous, casting_dtype)
        if isinstance(local_param, DTensor):
            local_param = local_param.to_local()
        local_param = local_param.detach().clone()
        if not local_param.is_contiguous():
            local_param = local_param.contiguous()
        if cpu_offload:
            local_param = local_param.to("cpu")

        local_sd[param_name] = local_param

    cast(nn.Module, model).load_state_dict(local_sd, assign=True)
    return model


def fsdp2_prepare_auto_wrap_policy(fsdp2_plugin, model: nn.Module):
    """Prepare auto-wrap policy, matching Accelerate helper naming and behavior."""
    return _prepare_auto_wrap_policy(fsdp2_plugin, model)


def get_parameters_from_modules(modules: Iterable[nn.Module] | str, model: nn.Module, device) -> set[nn.Parameter]:
    """Convert ignored modules to ignored parameters."""
    return _get_parameters_from_modules(modules, model, device)


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
    mesh = _build_device_mesh(accelerator, hp_args)
    reshard_after_forward = fsdp2_plugin.reshard_after_forward
    if hp_args.reshard_after_forward is not None:
        reshard_after_forward = hp_args.reshard_after_forward
    kwargs = {
        "reshard_after_forward": reshard_after_forward,
        "offload_policy": _resolve_offload_policy(fsdp2_plugin),
        "mp_policy": _resolve_mp_policy(fsdp2_plugin, hp_args),
        "mesh": mesh if mesh is not None else None,
        "ignored_params": get_parameters_from_modules(fsdp2_plugin.ignored_modules, model, accelerator.device),
        "comm_fusion": True,
    }
    replicate_params = _collect_replicate_params(model, _resolve_shard_size(mesh))
    if replicate_params:
        kwargs["replicate_params"] = replicate_params
    return kwargs


def _model_has_4bit_params(model: nn.Module) -> bool:
    """Return whether the model contains bitsandbytes 4-bit parameters."""
    return any(param.__class__.__name__ == "Params4bit" for _, param in model.named_parameters())


def _prepare_cpu_ram_efficient_loading(model: nn.Module, enabled: bool) -> dict[str, torch.Tensor]:
    """Capture non-persistent buffers before cpu_ram_efficient_loading rematerializes the model."""
    if not enabled:
        return {}

    non_persistent_buffer_fqns = _get_non_persistent_buffers(model, recurse=True, fqns=True)
    original_non_persistent_buffers = copy.deepcopy(
        {name: buffer for name, buffer in model.named_buffers() if name in non_persistent_buffer_fqns}
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
    wrapped_modules = [m for m in model.modules() if isinstance(m, HSDPModule) and m is not model]
    num_to_forward_prefetch = 1
    num_to_backward_prefetch = 1

    # Forward prefetch: each layer prefetches the next layer(s)
    for i, layer in enumerate(wrapped_modules):
        j_end = min(len(wrapped_modules), i + 1 + num_to_forward_prefetch)
        forward_targets = wrapped_modules[i + 1:j_end]
        if forward_targets:
            layer.set_modules_to_forward_prefetch(forward_targets)

    # Backward prefetch: reverse order since backward runs last-to-first
    wrapped_modules.reverse()
    for i, layer in enumerate(wrapped_modules):
        j_end = min(len(wrapped_modules), i + 1 + num_to_backward_prefetch)
        backward_targets = wrapped_modules[i + 1:j_end]
        if backward_targets:
            layer.set_modules_to_backward_prefetch(backward_targets)


def _restore_non_persistent_buffers(model: nn.Module, buffers: dict[str, torch.Tensor], device) -> None:
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
        parent_module.register_buffer(local_buffer_name, buffer_tensor, persistent=False)

    if hasattr(model, "tie_weights"):
        model.tie_weights()


def _maybe_upcast_trainable_params(accelerator, model: nn.Module) -> None:
    """Upcast model parameters to fp32 when mixed precision requires Accelerate-compatible behavior.

    ``model.to(torch.float32)`` creates new fp32 parameters in the module tree.
    Refresh HSDP's cached sharded parameter references and mixed-precision dtypes
    so comm_fusion uses the new fp32 parameter dtype as well.
    """
    model_dtype = getattr(model, "dtype", None)
    should_upcast = accelerator.mixed_precision != "no" and (model_dtype is None or model_dtype != torch.float32)
    if not should_upcast:
        return

    model.to(torch.float32)

    for module in model.modules():
        if isinstance(module, HSDPModule):
            state = module.hsdp_scheduler.hsdp_state  # pylint: disable=protected-access
            for hsdp_param in state.hsdp_params:
                if hsdp_param.is_sharded:
                    hsdp_param.reset_sharded_param()
            param_group = getattr(state, "param_group", None)
            if param_group is not None:
                param_group._init_mp_dtypes()  # pylint: disable=protected-access

    if accelerator.is_main_process:
        warnings.warn(
            "FSDP upcast of low precision parameters to fp32 (since mixed_precision != 'no') "
            "may affect the precision of model checkpoints."
        )



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
    original_sd = model.state_dict()
    should_restore_non_persistent_buffers = fsdp2_plugin.cpu_ram_efficient_loading and not model_has_params4bit
    original_non_persistent_buffers = _prepare_cpu_ram_efficient_loading(model, should_restore_non_persistent_buffers)
    if should_restore_non_persistent_buffers:
        model = _move_model_to_meta(model)

    fsdp2_kwargs = _build_fsdp2_kwargs(accelerator, model, hp_args, fsdp2_plugin)

    _apply_auto_wrap_policy(model, fsdp2_plugin, fsdp2_kwargs)
    if not isinstance(model, HSDPModule):
        fully_shard(model, **fsdp2_kwargs)

    _setup_prefetch(model)

    if fsdp2_plugin.cpu_ram_efficient_loading:
        fsdp2_load_full_state_dict(
            accelerator,
            model,
            original_sd,
            cpu_offload=_is_cpu_offload_enabled(fsdp2_plugin.cpu_offload),
        )

    _restore_non_persistent_buffers(model, original_non_persistent_buffers, accelerator.device)
    _maybe_upcast_trainable_params(accelerator, model)
    return model
